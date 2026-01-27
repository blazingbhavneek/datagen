"""
LangGraph Agent for Function Documentation Extraction
Uses custom tool-calling LLM to navigate linked chunks and extract complete function documentation.

Flow:
1. Find function definition in chunks (mother chunk)
2. Use agent to navigate prev/next chunks via links
3. Extract additional fields (cautions, references, code examples)
4. Keep state minimal - only cache chunks, not in message history
"""

import asyncio
import json
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# Import your custom modules
from agents.chunks import Chunk, ChunkReader
from agents.nodes.agent_node import CustomLLMWithTools, OutputMode


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class FunctionParameter(BaseModel):
    """Model for a single function parameter"""
    name: str = Field(description="Parameter name")
    type: str = Field(description="Parameter type")
    description: str = Field(description="Parameter description")


class FunctionDefinition(BaseModel):
    """Core function definition fields - must be found in mother chunk"""
    name: str = Field(description="Function name")
    description: str = Field(description="Function description")
    parameters: List[FunctionParameter] = Field(
        default_factory=list,
        description="List of function parameters"
    )
    output_type: str = Field(description="Return type of the function")
    output_description: str = Field(description="Description of what the function returns")


class CompleteFunctionDoc(BaseModel):
    """Complete function documentation including extras from linked chunks"""
    # Core definition
    definition: FunctionDefinition
    
    # Additional fields from linked chunks
    cautions: List[str] = Field(default_factory=list, description="Warning/caution notes")
    references: List[str] = Field(default_factory=list, description="Related references")
    code_examples: List[str] = Field(default_factory=list, description="Code examples")
    
    # Metadata
    source_chunks: List[str] = Field(default_factory=list, description="Chunk IDs used")
    mother_chunk_id: str = Field(description="ID of the chunk with core definition")


class ExtractionDecision(BaseModel):
    """Decision on whether chunk contains function definition"""
    has_function_def: bool = Field(description="Whether chunk contains a complete function definition")
    function_def: Optional[FunctionDefinition] = Field(
        default=None,
        description="Extracted function definition if found"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the extraction (0-1)"
    )


# ============================================================================
# LANGGRAPH STATE
# ============================================================================

class FunctionExtractionState(TypedDict):
    """State for the function extraction graph"""
    # Mother chunk info
    mother_chunk: Optional[Chunk]
    mother_chunk_content: str
    
    # Current extraction
    current_function: CompleteFunctionDoc
    
    # Navigation state
    chunks_visited: List[str]  # Chunk IDs visited
    cached_chunks: Dict[str, Chunk]  # Cached chunks for reference
    
    # Navigation limits
    max_next: int
    max_prev: int
    next_count: int
    prev_count: int
    
    # Messages for agent (rebuilt each time, not accumulated)
    messages: List[BaseMessage]
    
    # Control
    iteration: int
    max_iterations: int
    completed: bool


# ============================================================================
# TOOLS FOR AGENT
# ============================================================================

class ExtractionTools:
    """Tools for the extraction agent"""
    
    def __init__(self, chunk_reader: ChunkReader):
        self.chunk_reader = chunk_reader
    
    def get_tools(self):
        """Return list of tools for the agent"""
        
        @tool
        async def navigate_to_linked_chunk(
            chunk_id: str,
            direction: str
        ) -> str:
            """
            Navigate to the next or previous linked chunk
            Use this when the current chunk's summary indicates relevant information in linked chunks
            Returns JSON with the linked chunk information including content and summary
            ---
            str chunk_id: ID of the current chunk
            str direction: Either 'next' or 'prev'
            """
            chunk = self.chunk_reader.get_chunk_by_id(chunk_id)
            if not chunk:
                return json.dumps({"error": f"Chunk {chunk_id} not found"})
            
            summary_points = chunk.get_summary_points()
            target_id = None
            relation = None
            topic = None
            
            if direction == "prev" and chunk.has_prev_links():
                for sp in summary_points:
                    if sp.prev_link:
                        target_id = sp.prev_link.get("chunk_id")
                        relation = sp.prev_link.get("relation")
                        topic = sp.prev_link.get("common_topic")
                        break
            elif direction == "next" and chunk.has_next_links():
                for sp in summary_points:
                    if sp.next_link:
                        target_id = sp.next_link.get("chunk_id")
                        relation = sp.next_link.get("relation")
                        topic = sp.next_link.get("common_topic")
                        break
            
            if not target_id:
                return json.dumps({
                    "error": f"No {direction} link found",
                    "chunk_id": chunk_id
                })
            
            linked = self.chunk_reader.get_chunk_by_id(target_id)
            if not linked:
                return json.dumps({"error": f"Linked chunk {target_id} not found"})
            
            return json.dumps({
                "success": True,
                "chunk_id": linked.id,
                "chunk_index": linked.chunk_index,
                "content": linked.content,
                "source": linked.source_file,
                "headers": linked.headers,
                "relation": relation,
                "common_topic": topic,
                "summary": [sp.text for sp in linked.get_summary_points()] if linked.has_summaries() else [],
                "has_next_link": linked.has_next_links(),
                "has_prev_link": linked.has_prev_links()
            }, indent=2)
        
        @tool
        async def update_function_field(
            field_name: str,
            value: Any,
            append: bool = False
        ) -> str:
            """
            Update a field in the function documentation being extracted
            Use this to add extracted information from chunks to the function documentation
            Returns confirmation message
            ---
            str field_name: Name of the field to update (cautions, references, code_examples)
            str value: Value to add (string or list)
            bool append: If True append to list fields if False replace
            """
            valid_fields = ["cautions", "references", "code_examples"]
            if field_name not in valid_fields:
                return json.dumps({
                    "error": f"Invalid field. Must be one of: {valid_fields}"
                })
            
            return json.dumps({
                "success": True,
                "field": field_name,
                "action": "append" if append else "replace",
                "value_added": value if isinstance(value, str) else f"{len(value)} items"
            })
        
        return [navigate_to_linked_chunk, update_function_field]


# ============================================================================
# AGENT NODES
# ============================================================================

class FunctionExtractionAgent:
    """Main agent for extracting function documentation"""
    
    def __init__(
        self,
        chunk_reader: ChunkReader,
        llm_config: Dict[str, Any],
        mode: OutputMode = OutputMode.MANUAL
    ):
        self.chunk_reader = chunk_reader
        self.mode = mode
        
        # Create tools
        self.tool_registry = ExtractionTools(chunk_reader)
        self.tools = self.tool_registry.get_tools()
        
        # Initialize custom LLM with tools
        self.llm = CustomLLMWithTools(
            mode=mode,
            base_url=llm_config.get("base_url", "http://localhost:8000/v1"),
            model_name=llm_config.get("model_name", "gpt-oss"),
            tools=self.tools,
            planner_system_prompt=self._get_planner_prompt(),
            executor_system_prompt=self._get_executor_prompt()
        )
    
    def _get_planner_prompt(self) -> str:
        return """You are a documentation extraction assistant. Your job is to:

1. Analyze function documentation chunks
2. Extract additional information (cautions, references, code examples) from linked chunks
3. Navigate to prev/next chunks when their summaries indicate relevant content
4. Update the function documentation fields with extracted information

IMPORTANT:
- Only navigate to linked chunks when summaries suggest relevant content
- Extract complete information before moving to next chunk
- Be specific about what information you're looking for
- Stop when you've explored enough chunks or found all information"""

    def _get_executor_prompt(self) -> str:
        return """You are a parameter generation assistant for documentation extraction.

Convert the planner's navigation/extraction requests into exact tool parameters.
Be precise with chunk IDs and field names."""

    def build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(FunctionExtractionState)
        
        # Add nodes
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("process_tool_calls", self.process_tool_calls_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "continue": "process_tool_calls",
                "end": END
            }
        )
        
        # Tool processing loops back to agent
        workflow.add_edge("process_tool_calls", "agent")
        
        return workflow.compile()
    
    async def agent_node(self, state: FunctionExtractionState) -> FunctionExtractionState:
        """Main agent node - decides what to do next"""
        
        # Build messages for this iteration (don't use accumulated history)
        messages = self._build_current_messages(state)
        
        # Invoke custom LLM
        response = await self.llm.ainvoke(messages)
        
        # Store response for tool processing
        state["messages"] = [response]
        state["iteration"] += 1
        
        return state
    
    def _build_current_messages(self, state: FunctionExtractionState) -> List[BaseMessage]:
        """Build messages for current iteration - only include relevant cached chunks"""
        messages = []
        
        # System message
        messages.append(SystemMessage(content=self._get_planner_prompt()))
        
        # Mother chunk context (always included)
        mother_context = f"""MOTHER CHUNK (Core Function Definition):
Chunk ID: {state['mother_chunk'].id if state['mother_chunk'] else 'unknown'}

{state['mother_chunk_content']}

CURRENT EXTRACTION STATE:
{json.dumps(state['current_function'].model_dump(), indent=2)}

NAVIGATION STATE:
- Chunks visited: {len(state['chunks_visited'])}
- Next chunks explored: {state['next_count']}/{state['max_next']}
- Prev chunks explored: {state['prev_count']}/{state['max_prev']}
"""
        messages.append(HumanMessage(content=mother_context))
        
        # Add recently cached chunks (last 2 for context)
        recent_chunks = list(state['cached_chunks'].values())[-2:]
        if recent_chunks:
            cached_context = "\n\nRECENTLY VIEWED CHUNKS:\n"
            for chunk in recent_chunks:
                summary = [sp.text for sp in chunk.get_summary_points()[:2]] if chunk.has_summaries() else []
                cached_context += f"\nChunk {chunk.id}:\n"
                cached_context += f"Summary: {summary}\n"
                cached_context += f"Content preview: {chunk.content[:300]}...\n"
            messages.append(HumanMessage(content=cached_context))
        
        # Add instruction for next action
        instruction = """Based on the current state:
1. Check if we need more information (cautions, references, code examples)
2. Look at chunk summaries to see if linked chunks might have relevant info
3. Navigate to linked chunks if needed, or update fields with extracted info
4. Stop when done or limits reached"""
        
        messages.append(HumanMessage(content=instruction))
        
        return messages
    
    async def process_tool_calls_node(self, state: FunctionExtractionState) -> FunctionExtractionState:
        """Process tool calls from agent response"""
        
        last_message = state["messages"][-1] if state["messages"] else None
        if not last_message or not hasattr(last_message, "tool_calls"):
            return state
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            
            if tool_name == "navigate_to_linked_chunk":
                # Execute navigation
                chunk_id = tool_args.get("chunk_id")
                direction = tool_args.get("direction")
                
                # Check limits
                if direction == "next" and state["next_count"] >= state["max_next"]:
                    continue
                if direction == "prev" and state["prev_count"] >= state["max_prev"]:
                    continue
                
                # Get the chunk
                result = await self._navigate_chunk(chunk_id, direction, state)
                
                # Update navigation counts
                if result:
                    if direction == "next":
                        state["next_count"] += 1
                    else:
                        state["prev_count"] += 1
            
            elif tool_name == "update_function_field":
                # Execute field update
                self._update_field(
                    state,
                    tool_args.get("field_name"),
                    tool_args.get("value"),
                    tool_args.get("append", True)
                )
        
        return state
    
    async def _navigate_chunk(
        self,
        chunk_id: str,
        direction: str,
        state: FunctionExtractionState
    ) -> Optional[Chunk]:
        """Navigate to linked chunk and cache it"""
        
        chunk = self.chunk_reader.get_chunk_by_id(chunk_id)
        if not chunk:
            return None
        
        summary_points = chunk.get_summary_points()
        target_id = None
        
        if direction == "prev" and chunk.has_prev_links():
            for sp in summary_points:
                if sp.prev_link:
                    target_id = sp.prev_link.get("chunk_id")
                    break
        elif direction == "next" and chunk.has_next_links():
            for sp in summary_points:
                if sp.next_link:
                    target_id = sp.next_link.get("chunk_id")
                    break
        
        if not target_id:
            return None
        
        linked = self.chunk_reader.get_chunk_by_id(target_id)
        if not linked:
            return None
        
        # Cache the chunk
        state["cached_chunks"][linked.id] = linked
        state["chunks_visited"].append(linked.id)
        state["current_function"].source_chunks.append(linked.id)
        
        return linked
    
    def _update_field(
        self,
        state: FunctionExtractionState,
        field_name: str,
        value: Any,
        append: bool
    ):
        """Update a field in the function documentation"""
        
        if field_name == "cautions":
            if append:
                if isinstance(value, list):
                    state["current_function"].cautions.extend(value)
                else:
                    state["current_function"].cautions.append(str(value))
            else:
                state["current_function"].cautions = [str(value)] if isinstance(value, str) else value
        
        elif field_name == "references":
            if append:
                if isinstance(value, list):
                    state["current_function"].references.extend(value)
                else:
                    state["current_function"].references.append(str(value))
            else:
                state["current_function"].references = [str(value)] if isinstance(value, str) else value
        
        elif field_name == "code_examples":
            if append:
                if isinstance(value, list):
                    state["current_function"].code_examples.extend(value)
                else:
                    state["current_function"].code_examples.append(str(value))
            else:
                state["current_function"].code_examples = [str(value)] if isinstance(value, str) else value
    
    def should_continue(self, state: FunctionExtractionState) -> Literal["continue", "end"]:
        """Decide whether to continue or end"""
        
        # Check iteration limit
        if state["iteration"] >= state["max_iterations"]:
            state["completed"] = True
            return "end"
        
        # Check navigation limits
        if state["next_count"] >= state["max_next"] and state["prev_count"] >= state["max_prev"]:
            state["completed"] = True
            return "end"
        
        # Check if agent has tool calls
        last_message = state["messages"][-1] if state["messages"] else None
        if not last_message or not hasattr(last_message, "tool_calls"):
            state["completed"] = True
            return "end"
        
        if not last_message.tool_calls:
            state["completed"] = True
            return "end"
        
        return "continue"


# ============================================================================
# FUNCTION FINDER
# ============================================================================

class FunctionFinder:
    """Find chunks containing function definitions"""
    
    def __init__(self, llm_config: Dict[str, Any]):
        from langchain_openai import ChatOpenAI
        
        self.llm = ChatOpenAI(
            base_url=llm_config.get("base_url", "http://localhost:8000/v1"),
            api_key="dummy",
            model=llm_config.get("model_name", "gpt-oss"),
            temperature=0.1,
            use_responses_api=True
        )
    
    async def find_function_def(self, chunk: Chunk) -> Optional[ExtractionDecision]:
        """Check if chunk contains a complete function definition"""
        
        prompt = f"""Analyze this chunk and determine if it contains a COMPLETE function definition.

A complete function definition must have:
- Function name
- Description of what it does
- Parameters (names, types, descriptions)
- Return type
- Return value description

Chunk content:
{chunk.content}

Extract the function definition if found, or indicate if not present/incomplete."""

        # Use structured output
        structured_llm = self.llm.with_structured_output(ExtractionDecision)
        result = await structured_llm.ainvoke(prompt)
        
        return result if result.has_function_def and result.confidence >= 0.7 else None


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class FunctionDocExtractor:
    """Main pipeline for extracting function documentation"""
    
    def __init__(
        self,
        chunk_reader: ChunkReader,
        llm_config: Dict[str, Any],
        max_next: int = 5,
        max_prev: int = 2,
        max_iterations: int = 15
    ):
        self.chunk_reader = chunk_reader
        self.llm_config = llm_config
        self.max_next = max_next
        self.max_prev = max_prev
        self.max_iterations = max_iterations
        
        # Initialize components
        self.finder = FunctionFinder(llm_config)
        self.agent = FunctionExtractionAgent(
            chunk_reader=chunk_reader,
            llm_config=llm_config,
            mode=OutputMode.MANUAL
        )
    
    async def extract_all_functions(
        self,
        chunks: Optional[List[Chunk]] = None
    ) -> List[CompleteFunctionDoc]:
        """Extract all function documentation from chunks"""
        
        if chunks is None:
            chunks = self.chunk_reader.load_all_chunks()
        
        print(f"\n{'='*80}")
        print(f"Searching {len(chunks)} chunks for function definitions...")
        print(f"{'='*80}\n")
        
        # Phase 1: Find mother chunks
        mother_chunks = []
        for chunk in chunks:
            decision = await self.finder.find_function_def(chunk)
            if decision and decision.function_def:
                mother_chunks.append((chunk, decision.function_def))
                print(f"✓ Found function: {decision.function_def.name} in chunk {chunk.id}")
        
        print(f"\nFound {len(mother_chunks)} function definitions\n")
        
        # Phase 2: Extract complete documentation
        results = []
        graph = self.agent.build_graph()
        
        for chunk, func_def in mother_chunks:
            print(f"\n{'='*80}")
            print(f"Extracting complete docs for: {func_def.name}")
            print(f"Mother chunk: {chunk.id}")
            print(f"{'='*80}\n")
            
            # Initialize state
            initial_state: FunctionExtractionState = {
                "mother_chunk": chunk,
                "mother_chunk_content": chunk.content,
                "current_function": CompleteFunctionDoc(
                    definition=func_def,
                    mother_chunk_id=chunk.id,
                    source_chunks=[chunk.id]
                ),
                "chunks_visited": [chunk.id],
                "cached_chunks": {chunk.id: chunk},
                "max_next": self.max_next,
                "max_prev": self.max_prev,
                "next_count": 0,
                "prev_count": 0,
                "messages": [],
                "iteration": 0,
                "max_iterations": self.max_iterations,
                "completed": False
            }
            
            # Run extraction
            final_state = await graph.ainvoke(initial_state)
            
            complete_doc = final_state["current_function"]
            results.append(complete_doc)
            
            print(f"\n✓ Extraction complete!")
            print(f"  - Chunks visited: {len(final_state['chunks_visited'])}")
            print(f"  - Cautions found: {len(complete_doc.cautions)}")
            print(f"  - References found: {len(complete_doc.references)}")
            print(f"  - Code examples found: {len(complete_doc.code_examples)}")
            print(f"  - Iterations: {final_state['iteration']}")
        
        return results
    
    def save_results(self, results: List[CompleteFunctionDoc], output_path: str):
        """Save extraction results to JSON file"""
        import os
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(
                [r.model_dump() for r in results],
                f,
                indent=2
            )
        
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_path}")
        print(f"Total functions extracted: {len(results)}")
        print(f"{'='*80}\n")


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Main function - loads config and runs extraction"""
    import sys
    import yaml
    import os
    
    if len(sys.argv) < 2:
        print("Usage: python function_extractor_agent.py config.yaml")
        sys.exit(1)
    
    # Load config
    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)
    
    print("="*80)
    print("Function Documentation Extractor")
    print("="*80)
    print(f"ChromaDB: {config['chroma_db_path']}")
    print(f"Collection: {config['chroma_collection']}")
    print(f"LLM: {config['llm_model']} @ {config['llm_endpoint']}")
    print("="*80 + "\n")
    
    # Initialize chunk reader
    chunk_reader = ChunkReader(
        chroma_db_path=config["chroma_db_path"],
        collection_name=config["chroma_collection"]
    )
    
    # Initialize extractor
    extractor = FunctionDocExtractor(
        chunk_reader=chunk_reader,
        llm_config={
            "base_url": config["llm_endpoint"],
            "model_name": config["llm_model"]
        },
        max_next=5,
        max_prev=2,
        max_iterations=15
    )
    
    # Extract all functions
    results = await extractor.extract_all_functions()
    
    # Save results
    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], f"{config['dataset_name']}_functions.json")
    extractor.save_results(results, output_path)
    
    # Print summary
    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    print(f"Total functions extracted: {len(results)}")
    
    for i, func_doc in enumerate(results, 1):
        print(f"\n{i}. {func_doc.definition.name}")
        print(f"   Description: {func_doc.definition.description[:80]}...")
        print(f"   Parameters: {len(func_doc.definition.parameters)}")
        print(f"   Extras: {len(func_doc.cautions)} cautions, "
              f"{len(func_doc.references)} refs, {len(func_doc.code_examples)} examples")
        print(f"   Sources: {len(func_doc.source_chunks)} chunks")
    
    print(f"\n{'='*80}")
    print(f"✅ Results saved to: {output_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())