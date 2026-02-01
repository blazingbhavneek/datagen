"""
phase4_question_generation.py

For each gap from Phase 3, generates benchmark questions in Japanese.
All gaps are processed concurrently — gated by the phase4 semaphore.

Each gap saves its own intermediate. Already-done gaps are skipped
before they ever enter the semaphore.

Usage:
    python phase4_question_generation.py
"""

import asyncio
import json

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import (
    get_llm,
    get_semaphore,
    save_intermediate,
    load_intermediate,
    async_retry,
    QuestionBatch,
)


# ─────────────────────────────────────────────
# CHUNK TEXT RESOLVER
# Phase 1 manifest already has every chunk's text saved.
# No need to re-read the source file.
# ─────────────────────────────────────────────
_manifest_cache: list[dict] | None = None


def _get_manifest() -> list[dict]:
    global _manifest_cache
    if _manifest_cache is None:
        _manifest_cache = load_intermediate("phase1", "manifest")
        if _manifest_cache is None:
            raise RuntimeError("Phase 1 manifest not found. Run phase1 first.")
    return _manifest_cache


def resolve_chunk_text(chunk_id: str) -> str:
    """Looks up chunk text from the phase1 manifest. No file I/O after first load."""
    for entry in _get_manifest():
        if entry["chunk_id"] == chunk_id:
            return entry["text"]
    raise KeyError(f"chunk_id '{chunk_id}' not found in phase1 manifest")


# ─────────────────────────────────────────────
# CHAIN (built once, shared)
# ─────────────────────────────────────────────
PHASE4_PROMPT = """You are generating benchmark questions for a Japanese technical document.

These questions are specifically designed to BREAK RAG (Retrieval Augmented Generation) systems — even multi-pass RAG.

You have been given:
1. A GAP description — this is something that cannot be answered by reading any single section
2. The RELEVANT SOURCE SECTIONS that are involved in this gap (in their original Japanese)

Your job: generate questions where:
- The answer is a SINGLE WORD or maximum 3 words (in Japanese)
- The answer does NOT appear as a direct statement anywhere in the source
- Answering requires understanding how the involved sections CONNECT, not just reading one
- The question uses terminology from MULTIPLE sections so it can't be matched to one chunk

RULES:
- Questions MUST be in Japanese
- Answers MUST be in Japanese  
- Provide an English translation of both for review
- Do NOT create multiple choice. Short-answer only.
- Generate 3 questions per gap
- Each question must target a different aspect of the gap

--- GAP DESCRIPTION ---
{gap_description}

--- GAP METADATA ---
Involved chunks: {involved_chunks}
Involved components: {involved_components}
Why RAG fails: {why_rag_fails}

--- RELEVANT SOURCE SECTIONS ---
{source_sections}
--- END SOURCE ---

{format_instructions}"""


def build_question_chain():
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=QuestionBatch)
    prompt = ChatPromptTemplate.from_template(PHASE4_PROMPT)
    return prompt | llm | parser, parser


# ─────────────────────────────────────────────
# ASYNC WORKER — one per gap
# ─────────────────────────────────────────────
async def generate_for_gap(gap: dict, chain, parser) -> dict:
    """
    Acquires semaphore, loads source chunks from phase1 manifest, calls LLM async.
    """
    sem = get_semaphore("phase4")

    async with sem:
        gap_id = gap["gap_id"]
        print(f"  [run]  {gap_id}: {gap['description'][:60]}...")

        # Load source context from phase1 manifest — no file I/O needed
        source_sections = {}
        for chunk_id in gap["involved_chunks"]:
            try:
                source_sections[chunk_id] = resolve_chunk_text(chunk_id)
            except Exception as e:
                source_sections[chunk_id] = f"[ERROR resolving {chunk_id}: {e}]"

        source_text = "\n\n".join(
            f"=== {cid} ===\n{text}" for cid, text in source_sections.items()
        )

        async def _call():
            result = await chain.ainvoke({
                "gap_description": gap["description"],
                "involved_chunks": json.dumps(gap["involved_chunks"]),
                "involved_components": json.dumps(gap["involved_components"]),
                "why_rag_fails": gap["why_rag_fails"],
                "source_sections": source_text,
                "format_instructions": parser.get_format_instructions(),
            })
            # Tag questions with gap info
            for i, q in enumerate(result.questions):
                q.source_gap_id = gap_id
                if not q.question_id:
                    q.question_id = f"{gap_id}_q{i}"
            return result

        result = await async_retry(_call)
        dumped = result.model_dump()
        await save_intermediate("phase4", gap_id, dumped)
        return dumped


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
async def run():
    print("[Phase 4] Question generation")

    gap_analysis = load_intermediate("phase3", "gap_analysis")
    if not gap_analysis:
        raise RuntimeError("Phase 3 gap_analysis not found. Run phase3 first.")

    gaps = gap_analysis["gaps"]
    print(f"  Generating questions for {len(gaps)} gaps...")

    chain, parser = build_question_chain()

    # Split: already done vs pending
    all_questions = []
    pending = []

    for gap in gaps:
        existing = load_intermediate("phase4", gap["gap_id"])
        if existing:
            print(f"  [skip] {gap['gap_id']} — already generated")
            all_questions.extend(existing["questions"])
        else:
            pending.append(gap)

    # Fire all pending gaps concurrently
    if pending:
        print(f"  Launching {len(pending)} tasks...")
        results = await asyncio.gather(
            *(generate_for_gap(gap, chain, parser) for gap in pending)
        )
        for r in results:
            all_questions.extend(r["questions"])

    # Consolidated output
    await save_intermediate("phase4", "all_questions", {"questions": all_questions})

    print(f"\n[Phase 4] Done. {len(all_questions)} questions generated total.")
    return all_questions


if __name__ == "__main__":
    asyncio.run(run())
