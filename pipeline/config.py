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


"""
config.py - Additional models needed for enhanced Phase 5

Add these to your existing config.py file:
"""

from pydantic import BaseModel, Field
from typing import Optional


# ─────────────────────────────────────────────
# Existing models (for reference - you already have these)
# ─────────────────────────────────────────────
class ValidationResult(BaseModel):
    """Enhanced validation result with semantic matching"""
    question_id: str
    question_ja: str
    correct_answer: str
    model_answer: str
    passed: bool
    semantic_match: bool = Field(description="Whether answer is semantically equivalent to correct answer")
    match_confidence: float = Field(description="Confidence score 0-1 for semantic match")
    match_reasoning: str = Field(description="Explanation of why answers match or don't match")
    model_reasoning: str = Field(description="Model's explanation of its answer")


class SemanticMatch(BaseModel):
    """Result of semantic equivalence check between two answers"""
    is_match: bool = Field(description="True if answers are semantically equivalent")
    confidence: float = Field(
        ge=0.0, 
        le=1.0,
        description="Confidence score (0.0-1.0) in the match decision"
    )
    reasoning: str = Field(
        description="Explanation of why the answers do or don't match, considering language differences"
    )


# ─────────────────────────────────────────────
# Enhanced GapAnalysis (if you want stricter gaps)
# ─────────────────────────────────────────────
class Gap(BaseModel):
    """Individual gap in the dependency graph"""
    gap_id: str = ""
    gap_type: str = Field(description="Type: EMERGENT_BEHAVIOR, NEGATIVE_SPACE, TEMPORAL_DEPENDENCY, etc.")
    description: str
    involved_chunks: list[str] = Field(min_items=3)  # Increased minimum from 2
    involved_components: list[str]
    implicit_edges: list[str] = Field(
        min_items=2,
        description="List of implicit edge IDs from the graph"
    )
    inference_path: str = Field(
        description="Step-by-step reasoning required to bridge this gap"
    )
    why_rag_fails: str = Field(
        description="Specific explanation of why retrieval-based systems cannot solve this"
    )


class GapAnalysis(BaseModel):
    """Collection of identified gaps"""
    gaps: list[Gap]
    metadata: dict = Field(
        default_factory=lambda: {
            "total_gaps": 0,
            "gap_types_found": [],
        }
    )


# ─────────────────────────────────────────────
# Enhanced Question models
# ─────────────────────────────────────────────
class Question(BaseModel):
    """Individual benchmark question with anti-RAG metadata"""
    question_id: str = ""
    source_gap_id: str = ""
    question_ja: str = Field(description="Question in Japanese")
    question_en: str = Field(description="English translation for review")
    answer: str = Field(description="Correct answer in Japanese (1-3 words)")
    answer_en: str = Field(description="English translation of answer")
    
    # Anti-RAG metadata
    anti_rag_technique: str = Field(
        description="Which anti-RAG technique is used (e.g., TERMINOLOGICAL_MISDIRECTION)"
    )
    inference_path: str = Field(
        description="Step-by-step reasoning needed to derive the answer"
    )
    rag_will_retrieve: list[str] = Field(
        description="Chunk IDs that RAG will incorrectly retrieve based on keywords"
    )
    target_chunks: list[str] = Field(
        description="Chunk IDs that actually contain the necessary information"
    )
    requires_hops: int = Field(
        ge=2,
        description="Number of reasoning hops required (minimum 2)"
    )


class QuestionBatch(BaseModel):
    """Batch of questions generated for one gap"""
    questions: list[Question] = Field(min_items=5)  # Increased from 3
    gap_metadata: dict = Field(
        default_factory=dict,
        description="Metadata about the source gap"
    )