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
PHASE4_PROMPT = """You are generating ADVERSARIAL benchmark questions for a Japanese technical document.

These questions are specifically designed to DEFEAT advanced RAG systems including:
- Multi-pass retrieval
- Reranking
- Query decomposition
- Hybrid search (dense + sparse)
- Context window stuffing

You have been given:
1. A GAP description — something requiring multi-hop inference across 3+ sections
2. The RELEVANT SOURCE SECTIONS that are involved (in original Japanese)
3. The INFERENCE PATH — what reasoning is needed to bridge the gap

Your job: generate questions where:
- The answer is a SINGLE WORD or maximum 3 words (in Japanese)
- The answer does NOT appear as a direct quote anywhere in the source
- Answering requires DERIVING the answer through inference, not retrieval
- The question uses terminology from MULTIPLE sections so retrieval gets confused
- Simple keyword matching or semantic similarity will retrieve WRONG chunks

ANTI-RAG TECHNIQUES TO APPLY:

1. TERMINOLOGICAL MISDIRECTION: Use terms from section A to ask about a concept only explained in sections B+C using different terminology. RAG will retrieve A (wrong) instead of B+C.

2. COUNTERFACTUAL FRAMING: Ask "what happens if X" where X contradicts a constraint. The answer requires understanding the constraint (scattered across chunks) + reasoning about the violation. Direct retrieval won't find "if X happens then Y" anywhere.

3. TEMPORAL SEQUENCING: Ask about "the first time" or "before" or "after" something, where the sequence is only derivable by combining initialization orders from multiple chunks. No single chunk states the sequence.

4. IMPLICIT QUANTIFICATION: Ask "how many" or "maximum" or "minimum" where the number must be calculated from constraints in different chunks. E.g., "max concurrent X" = (limit from chunk A) × (multiplier from chunk B) ÷ (overhead from chunk C).

5. CAUSAL CHAIN REVERSAL: Ask about the cause when the document only describes effects in scattered locations. Requires reverse-engineering the causal chain.

6. DOMAIN BRIDGING: Ask about the effect in domain Y of a property in domain X, where the document never explicitly bridges these domains. E.g., "What network condition causes data inconsistency?" when network is discussed in one section and consistency in another with no explicit link.

7. NEGATIVE CONSTRAINT PROBING: Ask what CAN'T happen, where the prohibition is structural (implied by what IS documented) not explicit. E.g., "Which component never processes input Y?" when the prohibition exists because another component always consumes Y first.

QUESTION REQUIREMENTS:
- Questions MUST be in Japanese (natural, technical Japanese)
- Answers MUST be in Japanese  
- Provide English translation of both for review
- NO multiple choice — short-answer only
- Generate 5 questions per gap (increased from 3)
- Each question must use a DIFFERENT anti-RAG technique
- Vary the inference depth: 2 questions requiring 2-hop reasoning, 3 questions requiring 3+ hops
- For each question, document:
  * Which anti-RAG technique is used
  * The inference path (step-by-step reasoning needed)
  * Which chunks RAG will incorrectly retrieve and why
  * The correct chunks needed (but RAG won't find them)

ANSWER VALIDATION:
- The answer must be unambiguous (only one correct answer)
- The answer must be verifiable from the source sections (via inference)
- The answer must NOT appear verbatim in any single chunk
- If the answer is a number, it must require calculation from multiple values

--- GAP DESCRIPTION ---
{gap_description}

--- GAP METADATA ---
Gap Type: {gap_type}
Involved chunks: {involved_chunks}
Involved components: {involved_components}
Inference path required: {inference_path}
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

        # Extract gap type from description or metadata if available
        gap_type = gap.get("gap_type", "UNKNOWN")
        inference_path = gap.get("inference_path", gap.get("why_rag_fails", "Not specified"))

        async def _call():
            result = await chain.ainvoke({
                "gap_description": gap["description"],
                "gap_type": gap_type,
                "involved_chunks": json.dumps(gap["involved_chunks"]),
                "involved_components": json.dumps(gap["involved_components"]),
                "inference_path": inference_path,
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
        
        print(f"  [done] {gap_id}: {len(dumped['questions'])} questions generated")
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
    print(f"  Generating adversarial questions for {len(gaps)} gaps...")

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
        print(f"  Launching {len(pending)} concurrent tasks...")
        results = await asyncio.gather(
            *(generate_for_gap(gap, chain, parser) for gap in pending)
        )
        for r in results:
            all_questions.extend(r["questions"])

    # Consolidated output
    await save_intermediate("phase4", "all_questions", {"questions": all_questions})

    print(f"\n[Phase 4] Done. {len(all_questions)} adversarial questions generated total.")
    print(f"  Average {len(all_questions) / len(gaps):.1f} questions per gap")
    return all_questions


if __name__ == "__main__":
    asyncio.run(run())
