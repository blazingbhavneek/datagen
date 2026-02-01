"""
phase1_chunk_extraction.py

Reads a markdown file, splits it into token-sized chunks on heading
boundaries, then extracts a relationship map from each chunk concurrently.

Chunking strategy:
  1. Split the markdown into sections at top-level headings (# or ##).
  2. Greedily accumulate sections into a chunk until adding the next
     section would exceed CHUNK_SIZE_TOKENS.
  3. If a single section is itself larger than CHUNK_SIZE_TOKENS, it
     gets its own chunk (we never split mid-section).

This preserves semantic coherence — a heading and its content always
stay together. RAG would chunk at arbitrary token boundaries and break
these apart; our chunking here mirrors what a well-tuned RAG splitter
would do at best, so any questions that still break it are genuinely hard.

Concurrency is capped by the phase1 semaphore (default 10).
Each chunk saves its own intermediate. Already-done chunks are skipped
before they ever hit the semaphore.

The manifest stores chunk text alongside metadata — phase4 and phase5
load chunk text from here instead of re-reading the source file.

Usage:
    python phase1_chunk_extraction.py --md path/to/doc.md
"""

import asyncio
import argparse
import json
import re
from pathlib import Path

import tiktoken
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import (
    get_llm,
    get_semaphore,
    save_intermediate,
    load_intermediate,
    chunk_id_for_index,
    async_retry,
    intermediate_path,
    ChunkRelationshipMap,
    CHUNK_SIZE_TOKENS,
)


# ─────────────────────────────────────────────
# TOKENIZER (shared, created once)
# ─────────────────────────────────────────────
# cl100k_base covers gpt-4 and gpt-4o. If you swap to a different
# model family, change this.
_ENCODER = tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    return len(_ENCODER.encode(text))


# ─────────────────────────────────────────────
# MARKDOWN → SECTIONS
# Split on heading boundaries. Each section = one heading + everything
# until the next heading of equal or higher level.
# ─────────────────────────────────────────────
def split_into_sections(md: str) -> list[str]:
    """
    Splits markdown into sections at any heading (# through ######).
    Any text before the first heading becomes its own section.
    """
    # Regex: match lines that start with one or more # followed by a space
    # We use a lookahead split so the heading itself stays with its section
    parts = re.split(r"(?=^#{1,6}\s)", md, flags=re.MULTILINE)
    # Filter out empty strings from the split
    return [p for p in parts if p.strip()]


# ─────────────────────────────────────────────
# SECTIONS → CHUNKS (greedy bin-packing on token count)
# ─────────────────────────────────────────────
def build_chunks(md: str, max_tokens: int = CHUNK_SIZE_TOKENS) -> list[dict]:
    """
    Greedy accumulation: keep adding sections to the current chunk
    until the next section would push it over max_tokens. Then seal
    the current chunk and start a new one.

    A single oversized section gets its own chunk — we never split
    mid-section.

    Returns list of:
        {
            "chunk_id": "chunk_0",
            "token_start": 0,
            "token_end": 8000,
            "token_count": 7842,
            "text": "..."
        }
    """
    sections = split_into_sections(md)
    chunks = []
    current_sections: list[str] = []
    current_tokens = 0
    running_token_offset = 0

    for section in sections:
        section_tokens = token_count(section)

        # Would adding this section exceed the limit?
        if current_sections and (current_tokens + section_tokens > max_tokens):
            # Seal the current chunk
            text = "".join(current_sections)
            chunks.append({
                "chunk_id": chunk_id_for_index(len(chunks)),
                "token_start": running_token_offset,
                "token_end": running_token_offset + current_tokens,
                "token_count": current_tokens,
                "text": text,
            })
            running_token_offset += current_tokens
            current_sections = []
            current_tokens = 0

        # Add section to current chunk (even if it alone exceeds the limit —
        # it'll become a solo chunk when the next section triggers the seal)
        current_sections.append(section)
        current_tokens += section_tokens

    # Seal the final chunk
    if current_sections:
        text = "".join(current_sections)
        chunks.append({
            "chunk_id": chunk_id_for_index(len(chunks)),
            "token_start": running_token_offset,
            "token_end": running_token_offset + current_tokens,
            "token_count": current_tokens,
            "text": text,
        })

    return chunks


# ─────────────────────────────────────────────
# LLM CHAIN (built once, shared across all concurrent tasks)
# ─────────────────────────────────────────────
PHASE1_PROMPT = """You are analyzing a section of a large Japanese technical document (likely programming/SDK documentation).

Your job is NOT to summarize. Your job is to extract the STRUCTURE: what components exist, how they connect, what flows where.

CRITICAL RULES:
- Extract component names exactly as they appear in the original (keep Japanese names as-is)
- Descriptions and explanations must be in English
- Pay close attention to SIDE EFFECTS — things a component changes that aren't its primary output
- Pay close attention to CROSS REFERENCES — components or concepts mentioned here but clearly defined elsewhere
- If something is referenced but not explained, flag it. These are the invisible edges.

--- DOCUMENT SECTION ({chunk_id}, tokens {token_start}-{token_end}) ---
{text}
--- END SECTION ---

{format_instructions}"""


def build_extraction_chain():
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=ChunkRelationshipMap)
    prompt = ChatPromptTemplate.from_template(PHASE1_PROMPT)
    return prompt | llm | parser, parser


# ─────────────────────────────────────────────
# ASYNC WORKER — one per chunk
# ─────────────────────────────────────────────
async def process_chunk(chunk: dict, chain, parser) -> dict:
    """
    Acquires the phase1 semaphore, runs LLM extraction, saves intermediate.
    """
    sem = get_semaphore("phase1")

    async with sem:
        print(f"  [run]  {chunk['chunk_id']} ({chunk['token_count']} tokens)")

        async def _call():
            result = await chain.ainvoke({
                "chunk_id": chunk["chunk_id"],
                "token_start": chunk["token_start"],
                "token_end": chunk["token_end"],
                "text": chunk["text"],
                "format_instructions": parser.get_format_instructions(),
            })
            # Overwrite in case LLM hallucinated these
            result.chunk_id = chunk["chunk_id"]
            result.chunk_range = f"{chunk['token_start']}-{chunk['token_end']}"
            return result

        result = await async_retry(_call)
        dumped = result.model_dump()
        await save_intermediate("phase1", chunk["chunk_id"], dumped)
        return dumped


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
async def run(md_path: str):
    print(f"[Phase 1] Extracting relationship maps from {md_path}")

    md_text = Path(md_path).read_text(encoding="utf-8")
    total_tokens = token_count(md_text)
    print(f"  Total tokens: {total_tokens}")

    chunks = build_chunks(md_text)
    print(f"  Chunks: {len(chunks)} (target={CHUNK_SIZE_TOKENS} tokens each)")
    for c in chunks:
        print(f"    {c['chunk_id']}: {c['token_count']} tokens")

    # Write manifest — includes chunk text so phase4/phase5 can resolve
    # chunk_ids back to text without needing the original file.
    manifest = [
        {
            "chunk_id": c["chunk_id"],
            "token_start": c["token_start"],
            "token_end": c["token_end"],
            "token_count": c["token_count"],
            "text": c["text"],
        }
        for c in chunks
    ]
    intermediate_path("phase1", "manifest").write_text(
        json.dumps({"_meta": {"phase": "phase1", "identifier": "manifest"}, "data": manifest}, indent=2, ensure_ascii=False)
    )
    print(f"  [saved] phase1__manifest.json")

    # Build chain once — shared across all tasks
    chain, parser = build_extraction_chain()

    # Split: already done vs pending
    results = []
    pending = []

    for chunk in chunks:
        existing = load_intermediate("phase1", chunk["chunk_id"])
        if existing:
            print(f"  [skip] {chunk['chunk_id']} — already processed")
            results.append(existing)
        else:
            pending.append(chunk)

    # Fire all pending concurrently — semaphore inside each task gates the flow
    if pending:
        print(f"  Launching {len(pending)} tasks (semaphore will gate concurrency)...")
        new_results = await asyncio.gather(
            *(process_chunk(chunk, chain, parser) for chunk in pending)
        )
        results.extend(new_results)

    print(f"[Phase 1] Done. {len(results)} chunk maps saved.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", required=True, help="Path to the markdown file")
    args = parser.parse_args()
    asyncio.run(run(args.md))
