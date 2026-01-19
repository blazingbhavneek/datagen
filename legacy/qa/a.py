# qa_generator/answer_generation.py
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

from openai import OpenAI
from qa.config import AnswerResult, JudgeScore, QAGeneratorConfig, Question
from qa.search import CodeGraphQueryTool, VectorSearchTool


class ToolRegistry:
    """Registry of tools available to answer generation agents"""

    def __init__(self, config: QAGeneratorConfig):
        self.config = config
        self.tools = {}
        self._register_tools()

    def _register_tools(self):
        """Register all available tools based on data source type"""
        if self.config.data_source_type in ["doc", "pair"]:

            self.tools["vector_search_docs"] = VectorSearchTool(
                self.config.chroma_db_path,
                self.config.chroma_collection,
                self.config.embedding_endpoint,
                self.config.embedding_model,
            )

        if self.config.data_source_type in ["code", "pair"]:
            self.tools["query_code_graph"] = CodeGraphQueryTool(
                self.config.code_graph_path,
                self.config.embedding_endpoint,
                self.config.embedding_model,
            )

    async def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool and return result"""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not available"

        tool = self.tools[tool_name]
        try:
            if hasattr(tool, "search"):
                return await tool.search(**kwargs)
            elif hasattr(tool, "query"):
                return await tool.query(**kwargs)
            else:
                return f"Error: Tool '{tool_name}' has no callable method"
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"


class AnswerGenerationAgent:
    """Iterative agent that uses tools to answer questions
    Uses ReAct-style prompting: Thought -> Action -> Observation loop"""

    def __init__(
        self, agent_id: str, config: QAGeneratorConfig, tool_registry: ToolRegistry
    ):
        self.agent_id = agent_id
        self.config = config
        self.tools = tool_registry
        self.client = OpenAI(base_url=config.llm_endpoint, api_key="dummy")
        self.model = config.llm_model
        self.logger = logging.getLogger(f"answer_gen_{agent_id}")

    async def generate_answer(self, question: Question) -> AnswerResult:
        """Generate answer with reasoning trace"""
        reasoning_trace = []
        source_chunks_used = set()
        iteration = 0

        # Initial state
        current_state = {
            "question": question.question,
            "partial_answer": "",
            "context": "",
            "completed": False,
        }

        while (
            iteration < self.config.max_iterations_per_question
            and not current_state["completed"]
        ):
            iteration += 1

            # Generate next step
            step_result = await self._generate_step(current_state, question)

            # Add to reasoning trace
            reasoning_trace.append(step_result)

            # Update state
            current_state.update(
                {
                    "partial_answer": step_result.get("answer_update", ""),
                    "context": step_result.get("new_context", ""),
                    "completed": step_result.get("completed", False),
                }
            )

            # Track source chunks used
            if "sources" in step_result:
                source_chunks_used.update(step_result["sources"])

            self.logger.debug(
                f"Iteration {iteration}: {step_result.get('thought', 'No thought')}"
            )

        # Combine all partial answers for final answer
        final_answer = self._combine_answers(reasoning_trace)

        return AnswerResult(
            question_id=question.id,
            answer=final_answer,
            reasoning_trace=reasoning_trace,
            source_chunks_used=list(source_chunks_used),
            iterations=iteration,
            completed=current_state["completed"],
        )

    async def _generate_step(self, state: Dict, question: Question) -> Dict:
        """Generate a single reasoning step"""
        prompt = f"""
        You are answering the question: "{question.question}"
        
        Current state:
        - Partial answer so far: {state['partial_answer']}
        - Context: {state['context']}
        
        Think about what information you need to answer the question completely.
        Choose one of the following actions:
        1. Use a tool to gather information
        2. Formulate an answer if you have sufficient information
        3. Request more specific information
        
        Respond in JSON format:
        {{
          "thought": "Your reasoning",
          "action": {{
            "type": "use_tool|formulate_answer|request_info",
            "tool_name": "tool_name_if_applicable",
            "tool_args": {{"arg1": "value1"}},
            "answer_update": "partial answer update if applicable",
            "completed": true|false
          }},
          "new_context": "any new context gathered"
        }}
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at answering technical questions using tools and reasoning.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )

            response_text = response.choices[0].message.content
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                step_data = json.loads(json_str)

                action = step_data.get("action", {})

                # Execute tool if needed
                if action.get("type") == "use_tool":
                    tool_result = await self.tools.execute_tool(
                        action.get("tool_name", ""), **action.get("tool_args", {})
                    )
                    step_data["observation"] = tool_result

                return step_data
            else:
                # Fallback if JSON parsing fails
                return {
                    "thought": response_text[:200],
                    "action": {
                        "type": "formulate_answer",
                        "answer_update": response_text,
                    },
                    "completed": True,
                }
        except Exception as e:
            self.logger.error(f"Error in step generation: {e}")
            return {
                "thought": f"Error occurred: {str(e)}",
                "action": {
                    "type": "formulate_answer",
                    "answer_update": f"Could not answer due to error: {str(e)}",
                },
                "completed": True,
            }

    def _combine_answers(self, reasoning_trace: List[Dict]) -> str:
        """Combine partial answers from reasoning trace into final answer"""
        answers = []
        for step in reasoning_trace:
            action = step.get("action", {})
            if action.get("answer_update"):
                answers.append(action["answer_update"])

        return " ".join(answers)


class AnswerQualityJudge:
    """LLM judge to evaluate answer quality"""

    def __init__(self, config: QAGeneratorConfig):
        self.config = config
        self.client = OpenAI(base_url=config.llm_endpoint, api_key="dummy")
        self.model = config.llm_model
        self.logger = logging.getLogger("answer_judge")

    async def evaluate_answer(
        self, question: Question, answer_result: AnswerResult
    ) -> JudgeScore:
        """Evaluate answer quality"""
        prompt = f"""
        Evaluate the following answer to the question.
        
        Question: {question.question}
        
        Answer: {answer_result.answer}
        
        Reasoning Trace: {json.dumps(answer_result.reasoning_trace)}
        
        Evaluate on these criteria (score 0.0-1.0):
        1. Completeness: Does it address all parts of the question?
        2. Accuracy: Is the information correct based on the reasoning trace?
        3. Relevance: Does it stay on topic?
        4. Clarity: Is it well-explained?
        5. Specificity: Does it provide specific details/examples?
        6. Reasoning: Is the reasoning trace logical and well-supported?
        
        Return scores in JSON format:
        {{
          "completeness": 0.x,
          "accuracy": 0.x,
          "relevance": 0.x,
          "clarity": 0.x,
          "specificity": 0.x,
          "reasoning": 0.x
        }}
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert judge evaluating answers to technical questions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )

            response_text = response.choices[0].message.content
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                scores = json.loads(json_str)

                # Calculate overall score as average
                overall = sum(scores.values()) / len(scores)
                scores["overall"] = overall

                return JudgeScore(**scores)
            else:
                # Default scores if parsing fails
                return JudgeScore(
                    completeness=0.5,
                    accuracy=0.5,
                    relevance=0.5,
                    clarity=0.5,
                    specificity=0.5,
                    reasoning=0.5,
                    overall=0.5,
                )
        except Exception as e:
            self.logger.error(f"Error evaluating answer: {e}")
            return JudgeScore(
                completeness=0.0,
                accuracy=0.0,
                relevance=0.0,
                clarity=0.0,
                specificity=0.0,
                reasoning=0.0,
                overall=0.0,
            )


class AnswerGenerationPipeline:
    """Orchestrates answer generation for all questions"""

    def __init__(self, config: QAGeneratorConfig, questions: List[Question]):
        self.config = config
        self.questions = questions
        self.tool_registry = ToolRegistry(config)
        self.judge = AnswerQualityJudge(config)
        self.logger = logging.getLogger("answer_generation")
        self.checkpoint_dir = os.path.join(config.cache_dir, "answer_checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    async def generate_all_answers(self) -> List[AnswerResult]:
        """Generate answers for all questions in parallel"""
        self.logger.info(
            f"Starting answer generation for {len(self.questions)} questions"
        )

        # Process questions in batches to manage concurrency
        batch_size = min(self.config.max_concurrent_agents, len(self.questions))
        all_results = []

        for i in range(0, len(self.questions), batch_size):
            batch = self.questions[i : i + batch_size]

            # Create agents for batch
            tasks = []
            for j, question in enumerate(batch):
                agent = AnswerGenerationAgent(
                    agent_id=f"ans_{i+j}",
                    config=self.config,
                    tool_registry=self.tool_registry,
                )
                task = agent.generate_answer(question)
                tasks.append(task)

            # Execute batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Evaluate and filter results
            for idx, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    self.logger.error(
                        f"Error generating answer for question {batch[idx].id}: {result}"
                    )
                    continue

                # Evaluate answer quality
                question = batch[idx]
                score = await self.judge.evaluate_answer(question, result)

                # Only keep answers that meet quality threshold
                if score.overall >= self.config.answer_quality_threshold:
                    # Add quality scores to result
                    result.quality_score = score
                    all_results.append(result)
                else:
                    self.logger.info(
                        f"Low quality answer filtered for question {question.id} (score: {score.overall:.2f})"
                    )

            self.logger.info(
                f"Completed batch {i//batch_size + 1}/{(len(self.questions)-1)//batch_size + 1}"
            )

        self.logger.info(f"Generated {len(all_results)} high-quality answers")
        return all_results
