# qa_generator/question_generation.py
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import List

from openai import OpenAI
from qa.chunk_manager import Chunk, ChunkManager
from qa.config import QAGeneratorConfig, Question
from tqdm.asyncio import tqdm


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


class QuestionGeneratorAgent:
    """LLM-based agent for generating questions"""

    def __init__(self, config: QAGeneratorConfig):
        self.config = config
        self.client = OpenAI(base_url=config.llm_endpoint, api_key="dummy")
        self.model = config.llm_model
        self.logger = setup_logger(f"question_gen")

    async def generate_questions(self, chunk: Chunk) -> List[Question]:
        """Generate questions for a given chunk"""
        prompt = self._create_question_prompt(chunk)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at generating high-quality technical questions.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1000,
        )

        # Parse the response to extract questions
        questions_text = response.choices[0].message.content
        questions = self._parse_questions(questions_text, chunk)

        return questions

    def _create_question_prompt(self, chunk: Chunk) -> str:
        """Create prompt for question generation"""
        return f"""
        Generate {self.config.questions_per_chunk} diverse questions based on the following content:
        
        Content Type: {chunk.chunk_type}
        Source File: {chunk.source_file}
        Content:
        {chunk.content}
        
        Requirements:
        - {int(self.config.easy_question_ratio * 100)}% should be easy questions (straightforward facts)
        - {int((1 - self.config.easy_question_ratio) * 100)}% should be medium questions (requiring analysis)
        - Questions should be specific and answerable from the provided content
        - Focus on key concepts, functions, relationships, or procedures
        - Output format: JSON array of objects with fields: question, question_type ("easy" or "medium"), rationale
        """

    def _parse_questions(self, questions_text: str, chunk: Chunk) -> List[Question]:
        """Parse the LLM response into Question objects"""
        try:
            # Try to extract JSON from response
            start_idx = questions_text.find("[")
            end_idx = questions_text.rfind("]") + 1

            if start_idx != -1 and end_idx != 0:
                json_str = questions_text[start_idx:end_idx]
                questions_data = json.loads(json_str)
            else:
                # If no JSON found, create a simple question from the response
                questions_data = [
                    {
                        "question": questions_text.strip(),
                        "question_type": "medium",
                        "rationale": "Generated from content",
                    }
                ]

            questions = []
            for i, q_data in enumerate(questions_data):
                question = Question(
                    id=f"{chunk.id}_q{i+1}",
                    question=q_data.get("question", ""),
                    question_type=q_data.get("question_type", "medium"),
                    source_chunk_id=chunk.id,
                    rationale=q_data.get("rationale", ""),
                    generation_timestamp=datetime.now().isoformat(),
                )
                questions.append(question)

            return questions
        except Exception as e:
            self.logger.error(f"Error parsing questions: {e}")
            # Return a default question if parsing fails
            return [
                Question(
                    id=f"{chunk.id}_default",
                    question=questions_text[:200] + "...",
                    question_type="medium",
                    source_chunk_id=chunk.id,
                    rationale="Default question due to parsing error",
                    generation_timestamp=datetime.now().isoformat(),
                )
            ]


class QuestionFilterAgent:
    """Agent that filters generated questions for quality"""

    def __init__(self, config: QAGeneratorConfig):
        self.config = config
        self.client = OpenAI(base_url=config.llm_endpoint, api_key="dummy")
        self.model = config.llm_model
        self.logger = setup_logger(f"question_filter")

    async def filter_questions(
        self, questions: List[Question], chunk: Chunk
    ) -> List[Question]:
        """Filter out low-quality questions"""
        if not questions:
            return []

        # Group questions by type to maintain balance
        easy_questions = [q for q in questions if q.question_type == "easy"]
        medium_questions = [q for q in questions if q.question_type == "medium"]

        # Filter each group separately
        filtered_easy = await self._filter_group(easy_questions, chunk)
        filtered_medium = await self._filter_group(medium_questions, chunk)

        return filtered_easy + filtered_medium

    async def _filter_group(
        self, questions: List[Question], chunk: Chunk
    ) -> List[Question]:
        """Filter a group of questions"""
        if not questions:
            return []

        # Create prompt for filtering
        questions_text = "\n".join([f"- {q.question}" for q in questions])
        prompt = f"""
        Evaluate these questions based on the provided content:
        
        Content:
        {chunk.content}
        
        Questions to evaluate:
        {questions_text}
        
        Criteria for good questions:
        - Directly answerable from the content
        - Clear and unambiguous
        - Focus on important concepts
        - Not too vague or general
        
        Return only the questions that meet these criteria as a JSON array.
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at evaluating question quality.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1000,
            )

            # Parse the response to get filtered questions
            response_text = response.choices[0].message.content
            start_idx = response_text.find("[")
            end_idx = response_text.rfind("]") + 1

            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                filtered_questions = json.loads(json_str)

                # Match with original questions
                filtered_ids = {q["question"] for q in filtered_questions}
                return [q for q in questions if q.question in filtered_ids]
            else:
                # If parsing fails, return original questions
                return questions
        except Exception as e:
            self.logger.error(f"Error filtering questions: {e}")
            return questions


class QuestionGenerationPipeline:
    """Orchestrates question generation process"""

    def __init__(self, config: QAGeneratorConfig):
        self.config = config
        self.chunk_manager = ChunkManager(config)
        self.generator = QuestionGeneratorAgent(config)
        self.filter = QuestionFilterAgent(config)
        self.logger = setup_logger("question_generation")

    async def generate_all_questions(self) -> List[Question]:
        """Generate questions for all chunks"""
        self.logger.info("Loading chunks...")
        chunks = self.chunk_manager.load_chunks()
        self.logger.info(f"Loaded {len(chunks)} chunks")

        all_questions = []

        # Process chunks in batches to manage concurrency
        batch_size = min(self.config.max_concurrent_agents, len(chunks))

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            # Generate questions for each chunk in the batch
            tasks = []
            for chunk in batch:
                task = self._process_chunk(chunk)
                tasks.append(task)

            # Execute batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results
            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Error processing chunk: {result}")
                elif result:
                    all_questions.extend(result)

            self.logger.info(
                f"Processed batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}"
            )

        self.logger.info(f"Generated {len(all_questions)} questions total")
        return all_questions

    async def _process_chunk(self, chunk: Chunk) -> List[Question]:
        """Process a single chunk to generate and filter questions"""
        try:
            # Generate questions
            generated_questions = await self.generator.generate_questions(chunk)

            # Filter questions
            filtered_questions = await self.filter.filter_questions(
                generated_questions, chunk
            )

            return filtered_questions
        except Exception as e:
            self.logger.error(f"Error processing chunk {chunk.id}: {e}")
            return []
