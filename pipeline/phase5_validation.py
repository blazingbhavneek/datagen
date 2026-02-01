"""
phase5_validation.py

Self-validation loop. Each question gets two sequential LLM calls
(answer → reasoning) bundled into one async task. All question-tasks
run concurrently, gated by the phase5 semaphore.

Questions where the model FAILS = good benchmark questions (keep).
Questions where the model SUCCEEDS = too easy (discard).

Usage:
    python phase5_validation.py
"""

import asyncio
import json

from langchain_core.prompts import ChatPromptTemplate

from config import (
    get_llm,
    get_semaphore,
    save_intermediate,
    load_intermediate,
    async_retry,
    ValidationResult,
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
VALIDATION_PROMPT = """You are a RAG-based QA system. You have been given retrieved document sections and must answer a question.

IMPORTANT:
- Answer with ONLY the answer. No explanation, no hedging, no "I think".
- The answer should be a single word or very short phrase (max 3 words), in Japanese.
- If you cannot find the answer in the provided sections, write: わからない

--- RETRIEVED SECTIONS ---
{retrieved_context}
--- END SECTIONS ---

--- QUESTION ---
{question_ja}
--- END QUESTION ---

Answer (Japanese, 1-3 words only):"""


REASONING_PROMPT = """You just attempted to answer this question using only the provided sections.

Question: {question_ja}
Your answer: {model_answer}
Correct answer: {correct_answer}

Explain in 1-2 sentences WHY you got it wrong (or right). What information was missing from the retrieved sections?"""


def build_chains():
    llm = get_llm()
    answer_chain = ChatPromptTemplate.from_template(VALIDATION_PROMPT) | llm
    reasoning_chain = ChatPromptTemplate.from_template(REASONING_PROMPT) | llm
    return answer_chain, reasoning_chain


# ─────────────────────────────────────────────
# ASYNC WORKER — one per question
# The two LLM calls inside are sequential (reasoning needs the answer).
# But all questions run concurrently against each other.
# ─────────────────────────────────────────────
async def validate_question(
    question: dict,
    answer_chain,
    reasoning_chain,
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

        # ── Compare ──
        correct = question["answer"].strip()
        passed = (model_answer == correct)

        # ── Call 2: get reasoning (needs model_answer from call 1) ──
        async def _reasoning():
            return await reasoning_chain.ainvoke({
                "question_ja": question["question_ja"],
                "model_answer": model_answer,
                "correct_answer": correct,
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
            model_reasoning=reasoning,
        )

        dumped = result.model_dump()
        await save_intermediate("phase5", qid, dumped)
        return dumped


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
async def run():
    print("[Phase 5] Validation — filtering questions that actually break RAG")

    all_questions_data = load_intermediate("phase4", "all_questions")
    if not all_questions_data:
        raise RuntimeError("Phase 4 all_questions not found. Run phase4 first.")

    questions = all_questions_data["questions"]
    print(f"  Validating {len(questions)} questions...")

    answer_chain, reasoning_chain = build_chains()

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
            *(validate_question(q, answer_chain, reasoning_chain) for q in pending)
        )
        results.extend(new_results)

    # ─── FINAL SPLIT ───
    # passed=True  → model got it RIGHT → too easy → DISCARD
    # passed=False → model FAILED      → breaks RAG → KEEP
    kept = [r for r in results if not r["passed"]]
    discarded = [r for r in results if r["passed"]]

    final_benchmark = {
        "kept_questions": kept,
        "discarded_questions": discarded,
        "summary": {
            "total": len(results),
            "kept": len(kept),
            "discarded": len(discarded),
            "rag_failure_rate": f"{len(kept)/len(results)*100:.1f}%" if results else "0%",
        },
    }

    await save_intermediate("phase5", "final_benchmark", final_benchmark)

    print(f"\n[Phase 5] Done.")
    print(f"  Total validated:   {len(results)}")
    print(f"  Kept (breaks RAG): {len(kept)}")
    print(f"  Discarded (easy):  {len(discarded)}")
    print(f"  RAG failure rate:  {final_benchmark['summary']['rag_failure_rate']}")

    return final_benchmark


if __name__ == "__main__":
    asyncio.run(run())
