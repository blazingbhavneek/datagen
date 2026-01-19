# qa_generator/config.py
import os
from typing import Dict, List, Optional

from pydantic import BaseModel


class QAGeneratorConfig(BaseModel):
    # Data sources (from Parser Module)
    data_source_type: str  # "doc", "code", or "pair"
    manifest_path: Optional[str] = None  # Path to manifest.json (for pair mode)

    # For doc mode
    doc_markdown_path: Optional[str] = None
    chroma_db_path: Optional[str] = None
    chroma_collection: Optional[str] = None

    # For code mode
    code_graph_path: Optional[str] = None

    # LLM endpoints
    llm_endpoint: str  # Chat completion endpoint
    llm_model: str
    embedding_endpoint: str  # For semantic search
    embedding_model: str

    # Question generation parameters
    questions_per_chunk: int = 10  # Target per chunk
    easy_question_ratio: float = 0.3  # 30% easy, 70% medium

    # Agent parameters
    max_iterations_per_question: int = 10
    max_concurrent_agents: int = 10
    answer_quality_threshold: float = 0.7  # LLM judge score threshold

    # Output
    output_dir: str
    dataset_name: str

    # Caching and resumption
    cache_dir: str = "./cache"
    resume_from_checkpoint: bool = False


class DatasetEntry(BaseModel):
    id: str
    question: str
    answer: str
    reasoning: str  # Contains step-by-step reasoning trace
    question_type: str  # "easy" or "medium"
    source_chunks: List[str]  # IDs of source chunks used
    metadata: Dict
    quality_scores: Dict[str, float]  # Scores from judge
    generation_timestamp: str


class Question(BaseModel):
    id: str
    question: str
    question_type: str  # "easy" or "medium"
    source_chunk_id: str
    rationale: str  # Why this question was generated
    generation_timestamp: str


class AnswerResult(BaseModel):
    question_id: str
    answer: str
    reasoning_trace: List[Dict]  # List of thought-action-observation steps
    source_chunks_used: List[str]
    iterations: int
    completed: bool


class JudgeScore(BaseModel):
    completeness: float
    accuracy: float
    relevance: float
    clarity: float
    specificity: float
    reasoning: float
    overall: float
