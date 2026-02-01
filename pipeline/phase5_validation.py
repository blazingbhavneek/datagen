"""
phase5_validation.py

Self-validation loop. Each question gets three sequential LLM calls
(answer → semantic_check → reasoning) bundled into one async task. 
All question-tasks run concurrently, gated by the phase5 semaphore.

Questions where the model FAILS = good benchmark questions (keep).
Questions where the model SUCCEEDS = too easy (discard).

CRITICAL: Uses semantic matching to handle multilingual answers and 
near-misses. Only truly failed questions are kept.

Usage:
    python phase5_validation.py
"""

import asyncio
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from config import (
    get_llm,
    get_semaphore,
    save_intermediate,
    load_intermediate,
    async_retry,
    ValidationResult,
    SemanticMatch,
)


# ─────────────────────────────────────────────
# CHUNK TEXT RESOLVER (from phase1 manifest)
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
    for entry in _get_manifest():
        if entry["chunk_id"] == chunk_id:
            return entry["text"]
    raise KeyError(f"chunk_id '{chunk_id}' not found in phase1 manifest")


def simulate_rag_context(question: dict) -> str:
    """
    Gives the model the BEST possible context (the exact chunks involved).
    Generous to RAG on purpose — makes the filter stricter.
    
    This simulates a PERFECT retrieval system that gets exactly the right chunks.
    If the model still fails with perfect retrieval, the question is truly hard.
    """
    sections = []
    for chunk_id in question.get("target_chunks", []):
        try:
            text = resolve_chunk_text(chunk_id)
            sections.append(f"=== Retrieved: {chunk_id} ===\n{text}")
        except Exception as e:
            sections.append(f"=== {chunk_id} === [resolution failed: {e}]")
    return "\n\n".join(sections)


# ─────────────────────────────────────────────
# CHAINS (built once, shared across all tasks)
# ─────────────────────────────────────────────
VALIDATION_PROMPT = """You are a RAG-based QA system with perfect retrieval. You have been given the exact relevant sections and must answer a question.

STRICT RULES:
- Answer with ONLY the answer. No explanation, no hedging, no "I think", no preamble.
- The answer should be a single word or very short phrase (max 3 words).
- Answer in the SAME LANGUAGE as the question (if question is Japanese, answer in Japanese).
- If you cannot derive the answer from the provided sections through ANY reasoning, write: わからない
- DO NOT just copy text from the sections - you must understand and derive the answer.

--- RETRIEVED SECTIONS ---
{retrieved_context}
--- END SECTIONS ---

--- QUESTION ---
{question_ja}
--- END QUESTION ---

Answer (same language as question, 1-3 words only):"""


SEMANTIC_CHECK_PROMPT = """You are evaluating whether two answers are SEMANTICALLY EQUIVALENT, accounting for:
- Language differences (English vs Japanese for same concept)
- Minor spelling/transliteration variations
- Technical term variations (e.g., "queue" vs "キュー")

Question: {question_ja}
Correct answer: {correct_answer}
Model's answer: {model_answer}

Evaluate if these answers refer to the SAME CONCEPT/ENTITY.

MATCHING RULES:
1. "queue" = "キュー" = "待ち行列" = MATCH (same concept, different languages)
2. "parallel_for" = "パラレル・フォア" = "parallel for" = MATCH
3. "SYCL runtime" vs "データ転送" = NO MATCH (different concepts)
4. "わからない" = "unknown" = "不明" = MATCH (all mean "don't know")
5. Numerical answers must match exactly
6. Minor typos in technical terms = MATCH if clearly same term
7. Completely different concepts = NO MATCH

{format_instructions}"""


REASONING_PROMPT = """You attempted to answer this question using retrieved document sections.

Question: {question_ja}
Your answer: {model_answer}
Correct answer: {correct_answer}
Semantic match: {is_match}

Explain in 2-3 sentences:
1. If you got it RIGHT: What information in the sections allowed you to answer correctly? Was it straightforward retrieval or did you need to connect multiple pieces?
2. If you got it WRONG: What specific information was MISSING or SCATTERED that prevented you from answering? What connections could you not make?

Be specific about which sections helped/failed and why."""


def build_chains():
    llm = get_llm()
    answer_chain = ChatPromptTemplate.from_template(VALIDATION_PROMPT) | llm
    
    semantic_parser = PydanticOutputParser(pydantic_object=SemanticMatch)
    semantic_chain = (
        ChatPromptTemplate.from_template(SEMANTIC_CHECK_PROMPT) | 
        llm | 
        semantic_parser
    )
    
    reasoning_chain = ChatPromptTemplate.from_template(REASONING_PROMPT) | llm
    
    return answer_chain, semantic_chain, reasoning_chain, semantic_parser


# ─────────────────────────────────────────────
# ASYNC WORKER — one per question
# Three sequential LLM calls: answer → semantic_check → reasoning
# All questions run concurrently against each other.
# ─────────────────────────────────────────────
async def validate_question(
    question: dict,
    answer_chain,
    semantic_chain,
    reasoning_chain,
    semantic_parser,
) -> dict:
    sem = get_semaphore("phase5")

    async with sem:
        qid = question["question_id"]
        print(f"  [run]  {qid}: {question['question_ja'][:40]}...")

        # Build RAG context from phase1 manifest
        context = simulate_rag_context(question)

        # ── Call 1: get the model's answer ──
        async def _answer():
            return await answer_chain.ainvoke({
                "retrieved_context": context,
                "question_ja": question["question_ja"],
            })

        raw_answer = await async_retry(_answer)
        model_answer = (
            raw_answer.content.strip()
            if hasattr(raw_answer, "content")
            else str(raw_answer).strip()
        )

        correct = question["answer"].strip()

        # ── Call 2: semantic equivalence check ──
        async def _semantic():
            return await semantic_chain.ainvoke({
                "question_ja": question["question_ja"],
                "correct_answer": correct,
                "model_answer": model_answer,
                "format_instructions": semantic_parser.get_format_instructions(),
            })

        semantic_result = await async_retry(_semantic)
        is_semantically_correct = semantic_result.is_match
        match_confidence = semantic_result.confidence
        match_reasoning = semantic_result.reasoning

        # PASS if semantically correct, FAIL otherwise
        passed = is_semantically_correct

        # ── Call 3: get reasoning (needs both answer and semantic result) ──
        async def _reasoning():
            return await reasoning_chain.ainvoke({
                "question_ja": question["question_ja"],
                "model_answer": model_answer,
                "correct_answer": correct,
                "is_match": "YES - semantically equivalent" if is_semantically_correct else "NO - different concepts",
            })

        raw_reasoning = await async_retry(_reasoning)
        reasoning = (
            raw_reasoning.content.strip()
            if hasattr(raw_reasoning, "content")
            else str(raw_reasoning).strip()
        )

        result = ValidationResult(
            question_id=qid,
            question_ja=question["question_ja"],
            correct_answer=correct,
            model_answer=model_answer,
            passed=passed,
            semantic_match=is_semantically_correct,
            match_confidence=match_confidence,
            match_reasoning=match_reasoning,
            model_reasoning=reasoning,
        )

        dumped = result.model_dump()
        await save_intermediate("phase5", qid, dumped)
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [done] {qid}: {status} (confidence: {match_confidence})")
        
        return dumped


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
async def run():
    print("[Phase 5] Validation — filtering questions that actually break RAG")
    print("  Using semantic matching to handle multilingual answers")

    all_questions_data = load_intermediate("phase4", "all_questions")
    if not all_questions_data:
        raise RuntimeError("Phase 4 all_questions not found. Run phase4 first.")

    questions = all_questions_data["questions"]
    print(f"  Validating {len(questions)} questions...")

    answer_chain, semantic_chain, reasoning_chain, semantic_parser = build_chains()

    # Split: already done vs pending
    results = []
    pending = []

    for q in questions:
        existing = load_intermediate("phase5", q["question_id"])
        if existing:
            print(f"  [skip] {q['question_id']}")
            results.append(existing)
        else:
            pending.append(q)

    # Fire all pending concurrently
    if pending:
        print(f"  Launching {len(pending)} validation tasks...")
        new_results = await asyncio.gather(
            *(validate_question(q, answer_chain, semantic_chain, reasoning_chain, semantic_parser) for q in pending)
        )
        results.extend(new_results)

    # ─── FINAL SPLIT ───
    # passed=True  → model got it RIGHT (even with perfect retrieval) → too easy → DISCARD
    # passed=False → model FAILED (even with perfect chunks given) → breaks RAG → KEEP
    kept = [r for r in results if not r["passed"]]
    discarded = [r for r in results if r["passed"]]

    # Additional analysis: categorize kept questions by confidence
    high_confidence_fails = [r for r in kept if r.get("match_confidence", 0) >= 0.8]
    low_confidence_fails = [r for r in kept if r.get("match_confidence", 0) < 0.8]

    final_benchmark = {
        "kept_questions": kept,
        "discarded_questions": discarded,
        "summary": {
            "total": len(results),
            "kept": len(kept),
            "discarded": len(discarded),
            "rag_failure_rate": f"{len(kept)/len(results)*100:.1f}%" if results else "0%",
            "high_confidence_failures": len(high_confidence_fails),
            "low_confidence_failures": len(low_confidence_fails),
        },
        "metadata": {
            "validation_method": "semantic_matching",
            "retrieval_simulation": "perfect_chunk_retrieval",
            "note": "Questions kept are those where model failed even with perfect retrieval",
        }
    }

    await save_intermediate("phase5", "final_benchmark", final_benchmark)

    print(f"\n[Phase 5] Done.")
    print(f"  Total validated:              {len(results)}")
    print(f"  Kept (breaks RAG):            {len(kept)}")
    print(f"    - High confidence failures: {len(high_confidence_fails)}")
    print(f"    - Low confidence failures:  {len(low_confidence_fails)}")
    print(f"  Discarded (too easy):         {len(discarded)}")
    print(f"  RAG failure rate:             {final_benchmark['summary']['rag_failure_rate']}")

    if kept:
        print(f"\n  Sample kept question:")
        sample = kept[0]
        print(f"    Q: {sample['question_ja'][:60]}...")
        print(f"    Expected: {sample['correct_answer']}")
        print(f"    Got: {sample['model_answer']}")
        print(f"    Why: {sample.get('match_reasoning', 'N/A')[:80]}...")

    return final_benchmark


if __name__ == "__main__":
    asyncio.run(run())
