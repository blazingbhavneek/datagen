"""
config.py — Shared config, Pydantic schemas, and utilities.
All stages import from here.
"""

import os
import json
import asyncio
import anyio
from pathlib import Path
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate


# ─────────────────────────────────────────────
# PATHS & ENV
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
INTERMEDIATES_DIR = BASE_DIR / "intermediates"
INTERMEDIATES_DIR.mkdir(exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

CHUNK_SIZE_TOKENS = 8000  # target token count per chunk. Tune based on your model's context.
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# ─────────────────────────────────────────────
# SEMAPHORE REGISTRY
# Per-phase concurrency caps. Tune independently via env vars:
#   CONCURRENCY_PHASE1=10
#   CONCURRENCY_PHASE2=5
#   ...
# Or just edit the defaults below.
# ─────────────────────────────────────────────
_SEMAPHORE_DEFAULTS = {
    "phase1": 10,   # chunk extraction — embarrassingly parallel, limited only by API rate
    "phase2": 5,    # stitching — heavier prompts, fewer concurrent
    "phase3": 1,    # single call, no parallelism
    "phase4": 8,    # question generation per gap
    "phase5": 10,   # validation — two LLM calls per question but lightweight
}

# Cache: one semaphore instance per phase, created lazily
_semaphores: dict[str, asyncio.Semaphore] = {}


def get_semaphore(phase: str) -> asyncio.Semaphore:
    """
    Returns (or creates) the semaphore for a given phase.
    Reads CONCURRENCY_{PHASE} env var, falls back to _SEMAPHORE_DEFAULTS.
    """
    if phase not in _semaphores:
        env_key = f"CONCURRENCY_{phase.upper()}"
        limit = int(os.getenv(env_key, _SEMAPHORE_DEFAULTS.get(phase, 5)))
        _semaphores[phase] = asyncio.Semaphore(limit)
        print(f"  [semaphore] {phase} → max {limit} concurrent")
    return _semaphores[phase]


# ─────────────────────────────────────────────
# PYDANTIC SCHEMAS — structured output contracts
# ─────────────────────────────────────────────

class Component(BaseModel):
    name: str = Field(..., description="Name of the component/module/class")
    description: str = Field(..., description="Brief description in English")
    inputs: list[str] = Field(default_factory=list, description="What it takes in")
    outputs: list[str] = Field(default_factory=list, description="What it produces")
    dependencies: list[str] = Field(
        default_factory=list,
        description="Other components this one depends on (by name)"
    )
    side_effects: list[str] = Field(
        default_factory=list,
        description="Things this component changes that aren't in its explicit outputs"
    )

class ChunkRelationshipMap(BaseModel):
    chunk_id: str
    chunk_range: str  # token range, e.g. "0-8000"
    components: list[Component]
    cross_references: list[str] = Field(
        default_factory=list,
        description="Components or concepts mentioned but NOT defined in this chunk"
    )


class Edge(BaseModel):
    from_component: str
    to_component: str
    from_chunk: str
    to_chunk: str
    relationship_type: str = Field(
        ...,
        description="One of: depends_on, triggers, modifies_state_of, implicit_input_to"
    )
    explanation: str = Field(..., description="Why these are connected, in English")
    is_explicit: bool = Field(
        ...,
        description="True if the doc explicitly states this connection. False if it must be inferred."
    )

class StitchedGraph(BaseModel):
    batch_label: str
    edges: list[Edge]
    unresolved_references: list[str] = Field(
        default_factory=list,
        description="Components referenced across chunks but never fully defined anywhere in this batch"
    )


class Gap(BaseModel):
    gap_id: str
    involved_chunks: list[str]
    involved_components: list[str]
    description: str = Field(
        ...,
        description="What information is missing or only inferrable. In English."
    )
    implicit_edges: list[str] = Field(
        ...,
        description="Which edges from Phase 2 are invisible/undocumented"
    )
    why_rag_fails: str = Field(
        ...,
        description="Exactly why retrieval cannot surface this. What textual signal is missing?"
    )

class GapAnalysis(BaseModel):
    gaps: list[Gap]


class BenchmarkQuestion(BaseModel):
    question_id: str
    source_gap_id: str
    question_ja: str = Field(..., description="The question in Japanese")
    question_en: str = Field(..., description="The question in English (for your review)")
    answer: str = Field(
        ...,
        description="Single word or max 3-word answer. Language matches question_ja."
    )
    target_chunks: list[str] = Field(
        ...,
        description="Which chunk_ids must be understood to answer this"
    )
    why_rag_fails: str

class QuestionBatch(BaseModel):
    questions: list[BenchmarkQuestion]


class ValidationResult(BaseModel):
    question_id: str
    question_ja: str
    correct_answer: str
    model_answer: str
    passed: bool = Field(
        ...,
        description="True if model got it RIGHT (bad — too easy). False if model failed (good — keep)."
    )
    model_reasoning: str


# ─────────────────────────────────────────────
# LLM CLIENT
# ─────────────────────────────────────────────

def get_llm() -> ChatOpenAI:
    """
    Returns a ChatOpenAI instance.
    LangChain's ChatOpenAI supports .ainvoke() natively — no wrapper needed.
    """
    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        temperature=0.2,
        max_retries=MAX_RETRIES,
        base_url="http://localhost:8000/v1" 
    )


# ─────────────────────────────────────────────
# ASYNC RETRY
# ─────────────────────────────────────────────

async def async_retry(coro_fn, max_retries=MAX_RETRIES, delay=RETRY_DELAY_SECONDS):
    """
    Async retry with exponential backoff.
    coro_fn: an async callable (no args) that returns the result or raises.

    Usage:
        result = await async_retry(lambda: my_chain.ainvoke(inputs))
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return await coro_fn()
        except Exception as e:
            last_error = e
            wait = delay * (2 ** attempt)
            print(f"  [retry {attempt+1}/{max_retries}] {e} — waiting {wait}s")
            await asyncio.sleep(wait)
    raise last_error


# ─────────────────────────────────────────────
# ASYNC FILE I/O
# Disk writes from many concurrent tasks would interleave with
# synchronous open(). Use anyio to offload to a thread pool.
# ─────────────────────────────────────────────

def intermediate_path(phase: str, identifier: str) -> Path:
    return INTERMEDIATES_DIR / f"{phase}__{identifier}.json"


async def save_intermediate(phase: str, identifier: str, data: Any):
    """Async-safe write. Builds the envelope, offloads the actual write."""
    path = intermediate_path(phase, identifier)
    envelope = {
        "_meta": {
            "phase": phase,
            "identifier": identifier,
            "saved_at": datetime.utcnow().isoformat(),
            "model": MODEL_NAME,
        },
        "data": (
            data if isinstance(data, (dict, list))
            else data.model_dump() if hasattr(data, "model_dump")
            else str(data)
        ),
    }
    payload = json.dumps(envelope, ensure_ascii=False, indent=2)

    # anyio.to_thread.run_sync offloads the blocking write without
    # blocking the event loop
    await anyio.to_thread.run_sync(path.write_text, payload)
    print(f"  [saved] {path.name}")


def load_intermediate(phase: str, identifier: str) -> Any:
    """
    Synchronous load — called at the START of each phase to check
    what's already done before we enter the async section.
    This is intentional: the skip-check happens before gather(),
    so it doesn't need to be async.
    """
    path = intermediate_path(phase, identifier)
    if not path.exists():
        return None
    envelope = json.loads(path.read_text())
    return envelope["data"]


def chunk_id_for_index(idx: int) -> str:
    return f"chunk_{idx}"
