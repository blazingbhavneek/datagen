"""
QA Dataset Generator using LangGraph
Generates question-answer pairs with reasoning traces from parsed code/docs
All-in-one implementation following original requirements

Usage:
    python lg.py --config config.yaml
"""

import asyncio
import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

import chromadb
import networkx as nx
import numpy as np
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# LangGraph dependencies
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from openai import AsyncOpenAI

# Core dependencies
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

# ============================================================================
# CONFIGURATION & DATA MODELS
# ============================================================================


class QAGeneratorConfig(BaseModel):
    data_source_type: str  # "doc", "code", or "pair"
    manifest_path: Optional[str] = None
    doc_markdown_path: Optional[str] = None
    chroma_db_path: Optional[str] = None
    chroma_collection: Optional[str] = None
    code_graph_path: Optional[str] = None
    llm_endpoint: str
    llm_model: str
    embedding_endpoint: str
    embedding_model: str
    questions_per_chunk: int = 10
    easy_question_ratio: float = 0.3
    max_iterations_per_question: int = 10
    max_concurrent_agents: int = 10
    answer_quality_threshold: float = 0.7
    output_dir: str
    dataset_name: str
    cache_dir: str = "./cache"


class Chunk(BaseModel):
    id: str
    content: str
    chunk_type: str
    source_file: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Question(BaseModel):
    id: str
    question: str
    question_type: str
    source_chunk_id: str
    rationale: str
    generation_timestamp: str


class ReasoningStep(BaseModel):
    step_number: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str


class AnswerResult(BaseModel):
    question_id: str
    answer: str
    reasoning_steps: List[ReasoningStep]
    source_chunks_used: List[str]
    iterations: int
    completed: bool
    quality_score: Optional[Dict[str, float]] = None


class DatasetEntry(BaseModel):
    id: str
    question: str
    answer: str
    reasoning: str
    question_type: str
    source_chunks: List[str]
    metadata: Dict[str, Any]
    quality_scores: Dict[str, float]
    generation_timestamp: str


# LangGraph States
class AnswerGenState(TypedDict):
    question: Question
    messages: List[Any]
    iteration: int
    max_iterations: int


# ============================================================================
# UTILITIES
# ============================================================================


def setup_logger(name: str, log_dir: str = "./logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def extract_json(text: str) -> Optional[Any]:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != 0:
            return json.loads(text[start:end])
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end != 0:
            return json.loads(text[start:end])
    except:
        pass
    return None


# ============================================================================
# CHUNK MANAGER
# ============================================================================


class ChunkManager:
    def __init__(self, config: QAGeneratorConfig):
        self.config = config
        self.logger = setup_logger("chunk_manager")

    def load_chunks(self) -> List[Chunk]:
        if self.config.data_source_type == "doc":
            return self._load_doc_chunks()
        elif self.config.data_source_type == "code":
            return self._load_code_chunks()
        else:  # pair
            return self._load_doc_chunks() + self._load_code_chunks()

    def _load_doc_chunks(self) -> List[Chunk]:
        if not self.config.chroma_db_path:
            return []
        try:
            client = chromadb.PersistentClient(path=self.config.chroma_db_path)
            collection = client.get_collection(name=self.config.chroma_collection)
            results = collection.get()
            chunks = []
            for doc_id, content, metadata in zip(
                results["ids"], results["documents"], results["metadatas"]
            ):
                chunks.append(
                    Chunk(
                        id=doc_id,
                        content=content,
                        chunk_type="doc",
                        source_file=metadata.get("source_file", "unknown"),
                        metadata=metadata,
                    )
                )
            self.logger.info(f"Loaded {len(chunks)} doc chunks")
            return chunks
        except Exception as e:
            self.logger.error(f"Error loading doc chunks: {e}")
            return []

    def _load_code_chunks(self) -> List[Chunk]:
        if not self.config.code_graph_path:
            return []
        try:
            with open(self.config.code_graph_path, "rb") as f:
                code_graph = pickle.load(f)
            graph = code_graph.graph if hasattr(code_graph, "graph") else code_graph
            chunks = []
            for node_id, node_data in graph.nodes(data=True):
                node_type = node_data.get("type", "").lower()
                if node_type in ["function", "class", "file"]:
                    chunks.append(
                        Chunk(
                            id=node_id,
                            content=node_data.get("code", "")
                            or node_data.get("summary", ""),
                            chunk_type=node_type,
                            source_file=node_data.get("file_path", "unknown"),
                            metadata=node_data,
                        )
                    )
            self.logger.info(f"Loaded {len(chunks)} code chunks")
            return chunks
        except Exception as e:
            self.logger.error(f"Error loading code chunks: {e}")
            return []


# ============================================================================
# TOOLS
# ============================================================================


class ToolRegistry:
    def __init__(self, config: QAGeneratorConfig):
        self.config = config
        self.logger = setup_logger("tools")
        self._init_clients()

    def _init_clients(self):
        if self.config.data_source_type in ["doc", "pair"]:
            try:
                self.chroma_client = chromadb.PersistentClient(
                    path=self.config.chroma_db_path
                )
                self.chroma_collection = self.chroma_client.get_collection(
                    self.config.chroma_collection
                )
            except:
                self.chroma_collection = None

        if self.config.data_source_type in ["code", "pair"]:
            try:
                with open(self.config.code_graph_path, "rb") as f:
                    obj = pickle.load(f)
                self.code_graph = obj.graph if hasattr(obj, "graph") else obj
            except:
                self.code_graph = None

        self.embedding_client = AsyncOpenAI(
            base_url=self.config.embedding_endpoint, api_key="dummy"
        )

    async def search_docs(self, query: str, n_results: int = 5) -> str:
        if not self.chroma_collection:
            return "Doc search unavailable"
        try:
            resp = await self.embedding_client.embeddings.create(
                input=query, model=self.config.embedding_model
            )
            emb = resp.data[0].embedding
            results = self.chroma_collection.query(
                query_embeddings=[emb], n_results=n_results
            )
            out = []
            for doc_id, doc, meta in zip(
                results["ids"][0], results["documents"][0], results["metadatas"][0]
            ):
                out.append(
                    {
                        "chunk_id": doc_id,
                        "content": doc[:500],
                        "source": meta.get("source_file"),
                    }
                )
            return json.dumps(out, indent=2)
        except Exception as e:
            return f"Error: {e}"

    async def search_code(self, query: str, top_k: int = 5) -> str:
        if not self.code_graph:
            return "Code search unavailable"
        try:
            resp = await self.embedding_client.embeddings.create(
                input=query, model=self.config.embedding_model
            )
            query_emb = np.array(resp.data[0].embedding)
            results = []
            for nid, ndata in self.code_graph.nodes(data=True):
                if "embedding" in ndata and ndata["embedding"]:
                    nemb = np.array(ndata["embedding"])
                    sim = np.dot(query_emb, nemb) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(nemb)
                    )
                    results.append((nid, ndata, float(sim)))
            results.sort(key=lambda x: x[2], reverse=True)
            out = []
            for nid, nd, sim in results[:top_k]:
                out.append(
                    {
                        "entity_id": nid,
                        "type": nd.get("type"),
                        "name": nd.get("name"),
                        "summary": nd.get("summary", "")[:200],
                        "file": nd.get("file_path"),
                        "score": sim,
                    }
                )
            return json.dumps(out, indent=2)
        except Exception as e:
            return f"Error: {e}"

    def get_tools(self):
        tools = []
        if self.config.data_source_type in ["doc", "pair"]:

            @tool
            async def search_documentation(query: str, n_results: int = 5) -> str:
                """Search documentation using semantic similarity"""
                return await self.search_docs(query, n_results)

            tools.append(search_documentation)

        if self.config.data_source_type in ["code", "pair"]:

            @tool
            async def search_codebase(query: str, top_k: int = 5) -> str:
                """Search code entities using semantic similarity"""
                return await self.search_code(query, top_k)

            tools.append(search_codebase)

        return tools


# ============================================================================
# QUESTION GENERATION
# ============================================================================


class QuestionGenerator:
    def __init__(self, config: QAGeneratorConfig):
        self.config = config
        self.logger = setup_logger("question_gen")
        self.llm = ChatOpenAI(
            base_url=config.llm_endpoint,
            api_key="dummy",
            model=config.llm_model,
            temperature=0.7,
        )

    async def generate_for_chunk(self, chunk: Chunk) -> List[Question]:
        easy_count = int(
            self.config.questions_per_chunk * self.config.easy_question_ratio
        )
        medium_count = self.config.questions_per_chunk - easy_count

        prompt = f"""Generate {self.config.questions_per_chunk} technical questions:

Content ({chunk.chunk_type}): {chunk.content[:2000]}

Requirements:
- {easy_count} easy (direct facts)
- {medium_count} medium (analysis required)

JSON array: [{{"question": "...", "question_type": "easy|medium", "rationale": "..."}}]"""

        resp = await self.llm.ainvoke([HumanMessage(content=prompt)])
        data = extract_json(resp.content) or []

        if isinstance(data, dict):
            data = [data]

        questions = []
        for i, q in enumerate(data):
            questions.append(
                Question(
                    id=f"{chunk.id}_q{i+1}",
                    question=q.get("question", ""),
                    question_type=q.get("question_type", "medium"),
                    source_chunk_id=chunk.id,
                    rationale=q.get("rationale", ""),
                    generation_timestamp=datetime.now().isoformat(),
                )
            )

        # Filter
        if questions:
            q_text = "\n".join(
                [f"{i+1}. {q.question}" for i, q in enumerate(questions)]
            )
            filter_prompt = f"""Filter questions. Keep good ones.

Content: {chunk.content[:1000]}
Questions:\n{q_text}

Return indices to keep: [1, 3, 5, ...]"""

            resp2 = await self.llm.ainvoke([HumanMessage(content=filter_prompt)])
            kept = extract_json(resp2.content)
            if isinstance(kept, list):
                questions = [questions[i - 1] for i in kept if 0 < i <= len(questions)]

        return questions


# ============================================================================
# ANSWER GENERATION WITH LANGGRAPH
# ============================================================================


class AnswerGenerator:
    def __init__(self, config: QAGeneratorConfig, tool_registry: ToolRegistry):
        self.config = config
        self.tools = tool_registry.get_tools()
        self.logger = setup_logger("answer_gen")
        self.llm = ChatOpenAI(
            base_url=config.llm_endpoint,
            api_key="dummy",
            model=config.llm_model,
            temperature=0.3,
        ).bind_tools(self.tools)

    def build_graph(self) -> StateGraph:
        workflow = StateGraph(AnswerGenState)
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", self.should_continue, {"continue": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    async def agent_node(self, state: AnswerGenState) -> AnswerGenState:
        response = await self.llm.ainvoke(state["messages"])
        state["messages"].append(response)
        state["iteration"] += 1
        return state

    def should_continue(self, state: AnswerGenState) -> Literal["continue", "end"]:
        if state["iteration"] >= state["max_iterations"]:
            return "end"
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "continue"
        return "end"

    async def generate_for_question(self, question: Question) -> AnswerResult:
        graph = self.build_graph()

        sys_msg = SystemMessage(
            content="You are a technical expert. Use tools to gather info and answer accurately."
        )
        user_msg = HumanMessage(
            content=f"Question: {question.question}\n\nAnswer using available tools."
        )

        initial_state: AnswerGenState = {
            "question": question,
            "messages": [sys_msg, user_msg],
            "iteration": 0,
            "max_iterations": self.config.max_iterations_per_question,
        }

        result = await graph.ainvoke(initial_state)

        # Extract reasoning and answer
        reasoning_steps = self._extract_reasoning(result["messages"])
        final_answer = self._extract_answer(result["messages"])
        source_chunks = self._extract_sources(result["messages"])

        return AnswerResult(
            question_id=question.id,
            answer=final_answer,
            reasoning_steps=reasoning_steps,
            source_chunks_used=source_chunks,
            iterations=result["iteration"],
            completed=True,
        )

    def _extract_reasoning(self, messages: List[Any]) -> List[ReasoningStep]:
        steps = []
        step_num = 1
        for i, msg in enumerate(messages):
            if isinstance(msg, AIMessage):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        obs = ""
                        if i + 1 < len(messages) and isinstance(
                            messages[i + 1], ToolMessage
                        ):
                            obs = messages[i + 1].content[:500]
                        steps.append(
                            ReasoningStep(
                                step_number=step_num,
                                thought=msg.content or "Using tool...",
                                action=tc.get("name", "tool"),
                                action_input=tc.get("args", {}),
                                observation=obs,
                            )
                        )
                        step_num += 1
        return steps

    def _extract_answer(self, messages: List[Any]) -> str:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content
        return "No answer generated"

    def _extract_sources(self, messages: List[Any]) -> List[str]:
        sources = set()
        for msg in messages:
            if hasattr(msg, "content") and isinstance(msg.content, str):
                data = extract_json(msg.content)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            if "chunk_id" in item:
                                sources.add(item["chunk_id"])
                            elif "entity_id" in item:
                                sources.add(item["entity_id"])
        return list(sources)


# ============================================================================
# ANSWER JUDGE
# ============================================================================


class AnswerJudge:
    def __init__(self, config: QAGeneratorConfig):
        self.config = config
        self.llm = ChatOpenAI(
            base_url=config.llm_endpoint,
            api_key="dummy",
            model=config.llm_model,
            temperature=0.1,
        )

    async def evaluate(
        self, question: Question, answer: AnswerResult
    ) -> Dict[str, float]:
        prompt = f"""Evaluate answer quality (0.0-1.0):

Q: {question.question}
A: {answer.answer}

JSON: {{"completeness": 0.x, "accuracy": 0.x, "relevance": 0.x, "clarity": 0.x, "specificity": 0.x, "reasoning": 0.x}}"""

        resp = await self.llm.ainvoke([HumanMessage(content=prompt)])
        scores = extract_json(resp.content) or {
            "completeness": 0.5,
            "accuracy": 0.5,
            "relevance": 0.5,
            "clarity": 0.5,
            "specificity": 0.5,
            "reasoning": 0.5,
        }
        scores["overall"] = sum(scores.values()) / len(scores)
        return scores


# ============================================================================
# DATASET ASSEMBLY
# ============================================================================


class DatasetAssembler:
    def __init__(self, config: QAGeneratorConfig):
        self.config = config

    def assemble(
        self, questions: List[Question], answers: List[AnswerResult]
    ) -> List[DatasetEntry]:
        q_map = {q.id: q for q in questions}
        entries = []
        for ans in answers:
            q = q_map.get(ans.question_id)
            if not q:
                continue
            reasoning = self._format_reasoning(ans.reasoning_steps)
            entries.append(
                DatasetEntry(
                    id=f"entry_{len(entries)}",
                    question=q.question,
                    answer=ans.answer,
                    reasoning=reasoning,
                    question_type=q.question_type,
                    source_chunks=ans.source_chunks_used,
                    metadata={"rationale": q.rationale, "iterations": ans.iterations},
                    quality_scores=ans.quality_score or {},
                    generation_timestamp=datetime.now().isoformat(),
                )
            )
        return entries

    def _format_reasoning(self, steps: List[ReasoningStep]) -> str:
        parts = []
        for s in steps:
            parts.append(
                f"<|step|>{s.step_number}<|end_step|>\n"
                f"<|thought|>{s.thought}<|end_thought|>\n"
                f"<|action|>{s.action}<|end_action|>\n"
                f"<|observation|>{s.observation}<|end_observation|>\n"
            )
        return "".join(parts)

    def save(self, entries: List[DatasetEntry], path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            for e in entries:
                f.write(json.dumps(e.dict()) + "\n")


# ============================================================================
# MAIN PIPELINE
# ============================================================================


class QAGeneratorPipeline:
    def __init__(self, config: QAGeneratorConfig):
        self.config = config
        self.logger = setup_logger("main")

    async def run(self):
        self.logger.info("=== Phase 1: Question Generation ===")
        chunk_mgr = ChunkManager(self.config)
        chunks = chunk_mgr.load_chunks()
        self.logger.info(f"Loaded {len(chunks)} chunks")

        q_gen = QuestionGenerator(self.config)
        all_questions = []

        sem = asyncio.Semaphore(self.config.max_concurrent_agents)

        async def gen_q(chunk):
            async with sem:
                return await q_gen.generate_for_chunk(chunk)

        tasks = [gen_q(c) for c in chunks]
        with tqdm(total=len(tasks), desc="Generating questions") as pbar:
            for coro in asyncio.as_completed(tasks):
                qs = await coro
                all_questions.extend(qs)
                pbar.update(1)

        self.logger.info(f"Generated {len(all_questions)} questions")

        self.logger.info("=== Phase 2: Answer Generation ===")
        tool_reg = ToolRegistry(self.config)
        a_gen = AnswerGenerator(self.config, tool_reg)
        judge = AnswerJudge(self.config)

        accepted_answers = []

        async def gen_a(q):
            async with sem:
                ans = await a_gen.generate_for_question(q)
                score = await judge.evaluate(q, ans)
                ans.quality_score = score
                if score["overall"] >= self.config.answer_quality_threshold:
                    return ans
                return None

        tasks = [gen_a(q) for q in all_questions]
        with tqdm(total=len(tasks), desc="Generating answers") as pbar:
            for coro in asyncio.as_completed(tasks):
                ans = await coro
                if ans:
                    accepted_answers.append(ans)
                pbar.update(1)

        self.logger.info(f"Generated {len(accepted_answers)} quality answers")

        self.logger.info("=== Phase 3: Dataset Assembly ===")
        assembler = DatasetAssembler(self.config)
        entries = assembler.assemble(all_questions, accepted_answers)

        os.makedirs(self.config.output_dir, exist_ok=True)
        output_path = os.path.join(
            self.config.output_dir, f"{self.config.dataset_name}.jsonl"
        )
        assembler.save(entries, output_path)

        self.logger.info(f"✅ Dataset saved: {output_path}")
        self.logger.info(f"Total entries: {len(entries)}")

        return output_path


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys

    import yaml

    if len(sys.argv) < 2:
        print("Usage: python lg.py config.yaml")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        config_data = yaml.safe_load(f)

    config = QAGeneratorConfig(**config_data)

    pipeline = QAGeneratorPipeline(config)
    asyncio.run(pipeline.run())
