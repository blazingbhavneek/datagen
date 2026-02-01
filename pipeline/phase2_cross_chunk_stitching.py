"""
phase2_cross_chunk_stitching.py

Takes the per-chunk relationship maps from Phase 1 and finds
connections ACROSS chunks. Merge-sort batching: within each pass,
all batches run concurrently (gated by phase2 semaphore). Passes
themselves are sequential — pass N+1 depends on pass N's output.

Usage:
    python phase2_cross_chunk_stitching.py
"""

import asyncio
import json
import math

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import (
    get_llm,
    get_semaphore,
    save_intermediate,
    load_intermediate,
    async_retry,
    StitchedGraph,
)


BATCH_SIZE = 5


# ─────────────────────────────────────────────
# LOAD PHASE 1 (sync — runs once before async work)
# ─────────────────────────────────────────────
def load_all_chunk_maps() -> list[dict]:
    manifest = load_intermediate("phase1", "manifest")
    if not manifest:
        raise RuntimeError("Phase 1 manifest not found. Run phase1 first.")

    maps = []
    for entry in manifest:
        data = load_intermediate("phase1", entry["chunk_id"])
        if data is None:
            raise RuntimeError(f"Missing Phase 1 output for {entry['chunk_id']}. Re-run phase1.")
        maps.append(data)
    return maps


# ─────────────────────────────────────────────
# CHAIN (built once, shared across all concurrent tasks)
# ─────────────────────────────────────────────
PHASE2_PROMPT = """You are given relationship maps extracted from SEPARATE sections of a large Japanese technical document. Each map was extracted independently — they don't know about each other.

Your job: find connections ACROSS these maps. Specifically:

1. EXPLICIT EDGES: Component A in one chunk directly references Component B in another chunk. The doc says so.
2. IMPLICIT EDGES: Component A in one chunk clearly depends on or affects Component B in another chunk, but neither chunk explicitly states this connection. These are the most important ones — flag them.
3. UNRESOLVED REFERENCES: Components mentioned in one chunk that never get fully defined in ANY chunk in this batch.

CRITICAL: Focus on IMPLICIT edges. These are the invisible links that RAG will never retrieve because there's no text connecting them.

--- CHUNK RELATIONSHIP MAPS ---
{chunk_maps_json}
--- END MAPS ---

{format_instructions}"""


def build_stitching_chain():
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=StitchedGraph)
    prompt = ChatPromptTemplate.from_template(PHASE2_PROMPT)
    return prompt | llm | parser, parser


# ─────────────────────────────────────────────
# ASYNC WORKER — one per batch within a pass
# ─────────────────────────────────────────────
async def stitch_batch(batch_maps: list[dict], batch_label: str, chain, parser) -> dict:
    """
    Acquires the phase2 semaphore, runs the stitch, saves intermediate.
    """
    sem = get_semaphore("phase2")

    async with sem:
        print(f"  [run]  {batch_label}")

        async def _call():
            result = await chain.ainvoke({
                "chunk_maps_json": json.dumps(batch_maps, ensure_ascii=False, indent=2),
                "format_instructions": parser.get_format_instructions(),
            })
            result.batch_label = batch_label
            return result

        result = await async_retry(_call)
        dumped = result.model_dump()
        await save_intermediate("phase2", batch_label, dumped)
        return dumped


# ─────────────────────────────────────────────
# ONE FULL PASS — all batches in this pass run concurrently
# ─────────────────────────────────────────────
async def run_stitching_pass(items: list[dict], pass_label: str, chain, parser) -> list[dict]:
    """
    Splits items into batches of BATCH_SIZE.
    Already-done batches are loaded from disk (no task created).
    Pending batches are all launched together — semaphore gates flow.
    Results are returned in original batch order.
    """
    # Pre-compute batch metadata
    batches = []
    for i in range(0, len(items), BATCH_SIZE):
        batch_idx = i // BATCH_SIZE
        batch_label = f"{pass_label}__batch_{batch_idx}"
        batches.append({
            "idx": batch_idx,
            "label": batch_label,
            "items": items[i : i + BATCH_SIZE],
        })

    # Separate into done vs pending
    results = [None] * len(batches)  # placeholder to preserve order
    pending_indices = []

    for i, b in enumerate(batches):
        existing = load_intermediate("phase2", b["label"])
        if existing:
            print(f"  [skip] {b['label']} — already stitched")
            results[i] = existing
        else:
            pending_indices.append(i)

    # Launch all pending concurrently
    if pending_indices:
        tasks = [
            stitch_batch(batches[i]["items"], batches[i]["label"], chain, parser)
            for i in pending_indices
        ]
        new_results = await asyncio.gather(*tasks)

        # Place results back in order
        for idx, result in zip(pending_indices, new_results):
            results[idx] = result

    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
async def run():
    print("[Phase 2] Cross-chunk stitching (merge-sort, async passes)")

    chunk_maps = load_all_chunk_maps()
    print(f"  Loaded {len(chunk_maps)} chunk maps from Phase 1")

    chain, parser = build_stitching_chain()

    # Pass 0: stitch raw chunk maps
    print(f"\n  Pass 0: stitching {len(chunk_maps)} chunk maps...")
    current_items = chunk_maps
    pass_num = 0
    current_items = await run_stitching_pass(current_items, f"pass_{pass_num}", chain, parser)

    # Subsequent passes: sequential (each depends on prior), but internal batches are concurrent
    while len(current_items) > 1:
        pass_num += 1
        print(f"\n  Pass {pass_num}: stitching {len(current_items)} results...")
        current_items = await run_stitching_pass(current_items, f"pass_{pass_num}", chain, parser)

    final = current_items[0] if current_items else {}
    await save_intermediate("phase2", "final_graph", final)

    print(f"\n[Phase 2] Done. Final graph saved.")
    print(f"  Total edges: {len(final.get('edges', []))}")
    print(f"  Implicit edges: {sum(1 for e in final.get('edges', []) if not e.get('is_explicit'))}")
    print(f"  Unresolved refs: {len(final.get('unresolved_references', []))}")

    return final


if __name__ == "__main__":
    asyncio.run(run())
