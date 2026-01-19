# qa_generator/dataset_creation.py
import json
import logging
import os
from datetime import datetime
from typing import List

from qa.config import (
    AnswerResult,
    DatasetEntry,
    JudgeScore,
    QAGeneratorConfig,
    Question,
)


class ReasoningTraceConverter:
    """Converts reasoning steps into training-ready format"""

    def __init__(self, config: QAGeneratorConfig):
        self.config = config
        self.logger = logging.getLogger("reasoning_converter")

    def convert_to_reasoning_field(
        self, answer_result: AnswerResult, question: Question
    ) -> str:
        """Convert reasoning steps to single reasoning field string"""
        reasoning_steps = []

        for step_idx, step in enumerate(answer_result.reasoning_trace):
            thought = step.get("thought", "")
            action = step.get("action", {})
            observation = step.get("observation", "")

            step_text = (
                f"<|step|>{step_idx + 1}<|end_step|>\n"
                f"<|thought|>{thought}<|end_thought|>\n"
            )

            if action:
                step_text += f"<|action|>{json.dumps(action)}<|end_action|>\n"

            if observation:
                step_text += f"<|observation|>{observation}<|end_observation|>\n"

            reasoning_steps.append(step_text)

        return "".join(reasoning_steps)


class DatasetAssembler:
    """Assembles final dataset from question-answer pairs"""

    def __init__(self, config: QAGeneratorConfig):
        self.config = config
        self.converter = ReasoningTraceConverter(config)
        self.logger = logging.getLogger("dataset_assembler")

    def assemble_dataset(
        self, questions: List[Question], answers: List[AnswerResult]
    ) -> List[DatasetEntry]:
        """Assemble complete dataset entries"""
        self.logger.info(
            f"Assembling dataset from {len(questions)} questions and {len(answers)} answers"
        )

        # Create a mapping from question ID to question
        question_map = {q.id: q for q in questions}

        entries = []
        for answer in answers:
            question = question_map.get(answer.question_id)
            if not question:
                self.logger.warning(
                    f"No matching question found for answer {answer.question_id}"
                )
                continue

            # Convert reasoning trace
            reasoning = self.converter.convert_to_reasoning_field(answer, question)

            # Create quality scores dict
            quality_scores = {}
            if hasattr(answer, "quality_score") and answer.quality_score:
                quality_scores = {
                    "completeness": answer.quality_score.completeness,
                    "accuracy": answer.quality_score.accuracy,
                    "relevance": answer.quality_score.relevance,
                    "clarity": answer.quality_score.clarity,
                    "specificity": answer.quality_score.specificity,
                    "reasoning": answer.quality_score.reasoning,
                    "overall": answer.quality_score.overall,
                }

            entry = DatasetEntry(
                id=f"entry_{len(entries)}",
                question=question.question,
                answer=answer.answer,
                reasoning=reasoning,
                question_type=question.question_type,
                source_chunks=answer.source_chunks_used,
                metadata={
                    "question_rationale": question.rationale,
                    "generation_timestamp": question.generation_timestamp,
                    "answer_iterations": answer.iterations,
                },
                quality_scores=quality_scores,
                generation_timestamp=datetime.now().isoformat(),
            )

            entries.append(entry)

        self.logger.info(f"Assembled {len(entries)} dataset entries")
        return entries

    def save_dataset(self, entries: List[DatasetEntry], output_path: str) -> str:
        """Save dataset to JSONL file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(
                    json.dumps(
                        {
                            "id": entry.id,
                            "question": entry.question,
                            "answer": entry.answer,
                            "reasoning": entry.reasoning,
                            "question_type": entry.question_type,
                            "source_chunks": entry.source_chunks,
                            "metadata": entry.metadata,
                            "quality_scores": entry.quality_scores,
                            "generation_timestamp": entry.generation_timestamp,
                        }
                    )
                    + "\n"
                )

        self.logger.info(f"Dataset saved to {output_path}")
        return output_path
