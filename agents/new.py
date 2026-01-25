# Enhanced Chunk Manager with Semantic Navigation
class SemanticChunkManager(ChunkManager):
    def __init__(self, config: QAGeneratorConfig):
        super().__init__(config)
        self.logger = setup_logger("semantic_chunk_manager")
        self.chunk_cache = {}  # Cache for loaded chunks with their relationships
    
    def load_chunks_with_links(self) -> List[Chunk]:
        """Load chunks with their semantic links and summary points"""
        chunks = self.load_chunks()
        # Add semantic navigation capabilities to chunks
        for i, chunk in enumerate(chunks):
            # Add navigation metadata
            chunk.has_previous = i > 0
            chunk.has_next = i < len(chunks) - 1
            chunk.previous_chunk_id = chunks[i-1].id if i > 0 else None
            chunk.next_chunk_id = chunks[i+1].id if i < len(chunks) - 1 else None
            
            # Store in cache for fast access
            self.chunk_cache[chunk.id] = chunk
            
        self.logger.info(f"Loaded {len(chunks)} chunks with semantic navigation capabilities")
        return chunks
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by ID with caching"""
        return self.chunk_cache.get(chunk_id)
    
    def get_adjacent_chunks(self, chunk_id: str, direction: str = "both") -> Dict[str, Optional[Chunk]]:
        """Get previous and/or next chunks based on semantic links"""
        chunk = self.get_chunk_by_id(chunk_id)
        if not chunk:
            return {"previous": None, "next": None}
        
        result = {"previous": None, "next": None}
        
        if direction in ["previous", "both"] and chunk.has_previous and chunk.previous_chunk_id:
            result["previous"] = self.get_chunk_by_id(chunk.previous_chunk_id)
        
        if direction in ["next", "both"] and chunk.has_next and chunk.next_chunk_id:
            result["next"] = self.get_chunk_by_id(chunk.next_chunk_id)
        
        return result
    
    def should_fetch_adjacent_chunks(self, question: Question, current_chunk: Chunk) -> Dict[str, bool]:
        """
        Determine if agent needs to see adjacent chunks based on:
        1. Summary point links
        2. Question context
        3. Content completeness
        """
        needs_context = {"previous": False, "next": False}
        
        # Check if summary points indicate relationships
        if hasattr(current_chunk, 'summary_points') and current_chunk.summary_points:
            for sp in current_chunk.summary_points:
                if sp.prev_link and sp.prev_link.get("relates", False):
                    needs_context["previous"] = True
                if sp.next_link and sp.next_link.get("relates", False):
                    needs_context["next"] = True
        
        # Check question context for continuity indicators
        question_lower = question.question.lower()
        continuity_keywords = ["continue", "following", "previous", "next", "before", "after", "context"]
        
        for keyword in continuity_keywords:
            if keyword in question_lower:
                if keyword in ["previous", "before", "context"]:
                    needs_context["previous"] = True
                elif keyword in ["next", "following", "after"]:
                    needs_context["next"] = True
        
        return needs_context

# Enhanced Tools with Chunk Navigation
class EnhancedToolRegistry(ToolRegistry):
    def __init__(self, config: QAGeneratorConfig, chunk_manager: SemanticChunkManager):
        super().__init__(config)
        self.chunk_manager = chunk_manager
        self.logger = setup_logger("enhanced_tools")
    
    def get_tools(self):
        tools = super().get_tools()
        
        @tool
        async def get_previous_chunk_context(current_chunk_id: str) -> str:
            """Get context from the previous chunk when current content needs background information"""
            try:
                adjacent = self.chunk_manager.get_adjacent_chunks(current_chunk_id, "previous")
                prev_chunk = adjacent["previous"]
                if prev_chunk:
                    return f"Previous chunk context (ID: {prev_chunk.id}):\n{prev_chunk.content[:1000]}"
                return "No previous chunk available"
            except Exception as e:
                return f"Error retrieving previous chunk: {e}"
        
        @tool
        async def get_next_chunk_context(current_chunk_id: str) -> str:
            """Get context from the next chunk when current content needs continuation or example"""
            try:
                adjacent = self.chunk_manager.get_adjacent_chunks(current_chunk_id, "next")
                next_chunk = adjacent["next"]
                if next_chunk:
                    return f"Next chunk context (ID: {next_chunk.id}):\n{next_chunk.content[:1000]}"
                return "No next chunk available"
            except Exception as e:
                return f"Error retrieving next chunk: {e}"
        
        @tool
        async def get_chunk_navigation_info(chunk_id: str) -> str:
            """Get information about chunk relationships and whether to fetch adjacent chunks"""
            try:
                chunk = self.chunk_manager.get_chunk_by_id(chunk_id)
                if not chunk:
                    return f"Chunk {chunk_id} not found"
                
                navigation_info = {
                    "current_chunk_id": chunk.id,
                    "has_previous": hasattr(chunk, 'has_previous') and chunk.has_previous,
                    "has_next": hasattr(chunk, 'has_next') and chunk.has_next,
                    "summary_points_links": []
                }
                
                # Add summary point link information if available
                if hasattr(chunk, 'summary_points') and chunk.summary_points:
                    for i, sp in enumerate(chunk.summary_points):
                        link_info = {
                            "point": sp.text[:100],
                            "connects_previous": bool(sp.prev_link and sp.prev_link.get("relates")),
                            "connects_next": bool(sp.next_link and sp.next_link.get("relates")),
                            "prev_relation": sp.prev_link.get("relation", "") if sp.prev_link else "",
                            "next_relation": sp.next_link.get("relation", "") if sp.next_link else ""
                        }
                        navigation_info["summary_points_links"].append(link_info)
                
                return json.dumps(navigation_info, indent=2)
            except Exception as e:
                return f"Error getting navigation info: {e}"
        
        tools.extend([get_previous_chunk_context, get_next_chunk_context, get_chunk_navigation_info])
        return tools

# Enhanced Answer Generation with Chunk Awareness
class SemanticAnswerGenerator(AnswerGenerator):
    def __init__(self, config: QAGeneratorConfig, tool_registry: EnhancedToolRegistry, chunk_manager: SemanticChunkManager):
        super().__init__(config, tool_registry)
        self.chunk_manager = chunk_manager
        self.logger = setup_logger("semantic_answer_gen")
    
    async def generate_for_question(self, question: Question, source_chunk: Chunk) -> AnswerResult:
        """Enhanced answer generation that intelligently uses chunk navigation"""
        
        # Determine if we need adjacent chunks based on question and chunk relationships
        needs_context = self.chunk_manager.should_fetch_adjacent_chunks(question, source_chunk)
        
        # Build initial context with potential adjacent chunks
        initial_context = f"Question: {question.question}\n\n"
        initial_context += f"Current Chunk (ID: {source_chunk.id}):\n{source_chunk.content[:1500]}\n\n"
        
        # Pre-fetch adjacent chunks if needed based on semantic analysis
        adjacent_chunks = self.chunk_manager.get_adjacent_chunks(source_chunk.id, "both")
        
        if needs_context["previous"] and adjacent_chunks["previous"]:
            initial_context += f"Previous Chunk Context (ID: {adjacent_chunks['previous'].id}):\n{adjacent_chunks['previous'].content[:1000]}\n\n"
        
        if needs_context["next"] and adjacent_chunks["next"]:
            initial_context += f"Next Chunk Context (ID: {adjacent_chunks['next'].id}):\n{adjacent_chunks['next'].content[:1000]}\n\n"
        
        # Add navigation instructions to the agent
        navigation_instructions = """
You have access to tools that can fetch previous and next chunk contexts when needed.
Use these tools when:
- The question refers to content that might be in adjacent chunks
- Your current chunk content seems incomplete for answering
- Summary points indicate relationships with neighboring chunks
- You need background context or continuation of a topic

Always check if you have sufficient context before answering. If not, use the chunk navigation tools first.
"""
        
        graph = self.build_graph()
        sys_msg = SystemMessage(content=f"""You are a technical expert with access to document chunks and their semantic relationships.
{navigations_instructions}

Current context includes content from the main chunk and potentially adjacent chunks based on semantic analysis.
Use chunk navigation tools if you need more context from neighboring sections.""")

        user_msg = HumanMessage(content=initial_context)
        
        initial_state: AnswerGenState = {
            "question": question,
            "messages": [sys_msg, user_msg],
            "iteration": 0,
            "max_iterations": self.config.max_iterations_per_question,
            "current_chunk_id": source_chunk.id,
            "needs_previous_context": needs_context["previous"],
            "needs_next_context": needs_context["next"],
            "chunk_navigation_used": False
        }
        
        result = await graph.ainvoke(initial_state)
        
        # Extract reasoning and sources as before
        actual_reasoning_steps = self._extract_reasoning(result["messages"])
        source_chunks = self._extract_sources(result["messages"])
        
        # Ensure we include the source chunk and any adjacent chunks used
        all_source_chunks = set(source_chunks)
        all_source_chunks.add(source_chunk.id)
        if needs_context["previous"] and adjacent_chunks["previous"]:
            all_source_chunks.add(adjacent_chunks["previous"].id)
        if needs_context["next"] and adjacent_chunks["next"]:
            all_source_chunks.add(adjacent_chunks["next"].id)
        
        # Get final answer with structured output
        formatted_context = self._format_context_for_synthesis(question.question, result["messages"])
        structured_llm_with_schema = self.synthesis_llm.with_structured_output(FinalAnswerResponse)
        final_response: FinalAnswerResponse = await structured_llm_with_schema.ainvoke(formatted_context)
        
        return AnswerResult(
            question_id=question.id,
            answer=final_response.answer,
            reasoning_steps=actual_reasoning_steps,
            source_chunks_used=list(all_source_chunks),
            iterations=result["iteration"],
            completed=True,
        )

# Enhanced Pipeline with Semantic Chunk Navigation
class SemanticQAGeneratorPipeline(QAGeneratorPipeline):
    def __init__(self, config: QAGeneratorConfig):
        super().__init__(config)
        self.chunk_manager = SemanticChunkManager(config)
        self.logger = setup_logger("semantic_pipeline")
    
    async def run(self):
        self.logger.info("=== Phase 1: Loading Chunks with Semantic Links ===")
        chunks = self.chunk_manager.load_chunks_with_links()
        
        # For testing: use a middle chunk that likely has relationships
        test_chunk = chunks[len(chunks) // 2]
        self.logger.info(f"Loaded {len(chunks)} chunks with semantic navigation capabilities")
        self.logger.info(f"Test chunk has previous: {test_chunk.has_previous}, next: {test_chunk.has_next}")
        
        self.logger.info("=== Phase 2: Question Generation ===")
        q_gen = QuestionGenerator(self.config)
        
        # Generate questions for our test chunk
        questions = await q_gen.generate_for_chunk(test_chunk)
        self.logger.info(f"Generated {len(questions)} questions for test chunk")
        
        self.logger.info("=== Phase 3: Enhanced Answer Generation with Chunk Navigation ===")
        tool_reg = EnhancedToolRegistry(self.config, self.chunk_manager)
        a_gen = SemanticAnswerGenerator(self.config, tool_reg, self.chunk_manager)
        judge = AnswerJudge(self.config)
        
        accepted_answers = []
        sem = asyncio.Semaphore(self.config.max_concurrent_agents)
        
        async def process_question(q):
            async with sem:
                # Generate answer with chunk awareness
                ans = await a_gen.generate_for_question(q, test_chunk)
                # Evaluate quality
                score = await judge.evaluate(q, ans)
                ans.quality_score = score
                self.logger.info(f"Question: {q.question}")
                self.logger.info(f"Answer quality score: {score['overall']:.2f}")
                return ans if score["overall"] >= self.config.answer_quality_threshold else None
        
        tasks = [process_question(q) for q in questions]
        with tqdm(total=len(tasks), desc="Generating answers with chunk navigation") as pbar:
            for coro in asyncio.as_completed(tasks):
                ans = await coro
                if ans:
                    accepted_answers.append(ans)
                pbar.update(1)
        
        self.logger.info(f"Generated {len(accepted_answers)} quality answers with semantic chunk navigation")
        
        self.logger.info("=== Phase 4: Dataset Assembly ===")
        assembler = DatasetAssembler(self.config)
        entries = assembler.assemble(questions, accepted_answers)
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        output_path = os.path.join(self.config.output_dir, f"{self.config.dataset_name}_semantic.jsonl")
        assembler.save(entries, output_path)
        
        self.logger.info(f"📁 Semantic dataset saved: {output_path}")
        self.logger.info(f"Total entries: {len(entries)}")
        return output_path
