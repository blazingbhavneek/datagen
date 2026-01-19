# qa_generator/main.py
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import List

from qa.a import AnswerGenerationPipeline
from qa.config import AnswerResult, DatasetEntry, QAGeneratorConfig, Question
from qa.data import DatasetAssembler
from qa.q import QuestionGenerationPipeline


def setup_logger(name: str, log_dir: str = "./logs"):
    """Set up logger for the agent"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# For backward compatibility
class QAGeneratorOutput:
    def __init__(
        self,
        dataset_path: str,
        statistics_path: str,
        agent_logs_dir: str,
        checkpoint_dir: str,
    ):
        self.dataset_path = dataset_path
        self.statistics_path = statistics_path
        self.agent_logs_dir = agent_logs_dir
        self.checkpoint_dir = checkpoint_dir


class QAGeneratorPipeline:
    """Main orchestrator for the entire QA generation process"""

    def __init__(self, config: QAGeneratorConfig):
        self.config = config
        self.logger = setup_logger("qa_generator_main")
        self._validate_config()

    def _validate_config(self):
        """Validate all required paths and endpoints"""
        # Check data source files exist
        if self.config.data_source_type == "doc":
            assert os.path.exists(
                self.config.doc_markdown_path
            ), f"Doc markdown path does not exist: {self.config.doc_markdown_path}"
            assert os.path.exists(
                self.config.chroma_db_path
            ), f"Chroma DB path does not exist: {self.config.chroma_db_path}"
        elif self.config.data_source_type == "code":
            assert os.path.exists(
                self.config.code_graph_path
            ), f"Code graph path does not exist: {self.config.code_graph_path}"
        elif self.config.data_source_type == "pair":
            assert os.path.exists(
                self.config.manifest_path
            ), f"Manifest path does not exist: {self.config.manifest_path}"

        # Validate endpoints (basic check)
        assert self.config.llm_endpoint, "LLM endpoint is required"
        assert self.config.embedding_endpoint, "Embedding endpoint is required"

    async def run_full_pipeline(self) -> QAGeneratorOutput:
        """Execute all phases of QA generation"""
        self.logger.info("Starting QA Generation Pipeline")

        # Phase 1: Question Generation
        self.logger.info("Phase 1: Question Generation")
        question_pipeline = QuestionGenerationPipeline(self.config)
        questions = await question_pipeline.generate_all_questions()
        self.logger.info(f"Generated {len(questions)} questions")

        # Phase 2: Answer Generation
        self.logger.info("Phase 2: Answer Generation")
        answer_pipeline = AnswerGenerationPipeline(self.config, questions)
        answers = await answer_pipeline.generate_all_answers()
        self.logger.info(f"Generated {len(answers)} answers")

        # Phase 3: Dataset Creation
        self.logger.info("Phase 3: Dataset Creation")
        assembler = DatasetAssembler(self.config)
        dataset_entries = assembler.assemble_dataset(questions, answers)

        # Save dataset
        os.makedirs(self.config.output_dir, exist_ok=True)
        dataset_filename = f"{self.config.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        dataset_path = os.path.join(self.config.output_dir, dataset_filename)
        assembler.save_dataset(dataset_entries, dataset_path)

        # Generate statistics
        stats_path = self._generate_statistics(dataset_entries)

        # Create final output object
        output = QAGeneratorOutput(
            dataset_path=dataset_path,
            statistics_path=stats_path,
            agent_logs_dir=os.path.join(self.config.cache_dir, "answer_checkpoints"),
            checkpoint_dir=os.path.join(self.config.cache_dir, "checkpoints"),
        )

        self.logger.info(f"Pipeline complete! Dataset: {dataset_path}")
        self.logger.info(f"Total entries: {len(dataset_entries)}")
        return output

    def _generate_statistics(self, entries: List[DatasetEntry]) -> str:
        """Generate comprehensive statistics about the dataset"""
        stats = {
            "total_entries": len(entries),
            "question_types": {},
            "average_reasoning_steps": 0,
            "quality_scores_stats": {},
            "generation_timestamp": datetime.now().isoformat(),
        }

        # Count question types
        for entry in entries:
            q_type = entry.question_type
            stats["question_types"][q_type] = stats["question_types"].get(q_type, 0) + 1

            # Count reasoning steps
            step_count = entry.reasoning.count("<|step|>")
            stats["average_reasoning_steps"] += step_count

        if entries:
            stats["average_reasoning_steps"] /= len(entries)

        # Quality score statistics
        if entries and entries[0].quality_scores:
            score_keys = list(entries[0].quality_scores.keys())
            for key in score_keys:
                scores = [entry.quality_scores.get(key, 0) for entry in entries]
                stats["quality_scores_stats"][key] = {
                    "avg": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                }

        # Save statistics
        stats_path = os.path.join(self.config.output_dir, "statistics.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        self.logger.info("Statistics generated")
        return stats_path
