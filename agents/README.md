## Document 1: `chunk_reader.py` - ChromaDB Chunk Reader

### Classes

#### `SummaryPoint` (dataclass)
**Purpose**: Represents a summary point with linking information between chunks

**Attributes**:
- `text: str` - Summary text
- `prev_link: Optional[Dict]` - Link to previous chunk (contains: chunk_id, chunk_index, relation, common_topic)
- `next_link: Optional[Dict]` - Link to next chunk (same structure)

---

#### `Chunk` (Pydantic BaseModel)
**Purpose**: Enhanced chunk model with summary points and linking capabilities

**Attributes**:
- `id: str` - Unique chunk identifier
- `content: str` - Chunk content
- `chunk_type: str` - Type: "doc", "function", "class", "file"
- `source_file: str` - Source file path
- `chunk_index: int` - Sequential index
- `start_char/end_char: int` - Character positions
- `headers: List[str]` - Document headers
- `metadata: Dict` - Additional metadata
- `summary_points: List[Dict]` - Stored as dicts in DB

**Methods**:

##### `get_summary_points() -> List[SummaryPoint]`
- **Purpose**: Convert stored summary point dicts to SummaryPoint objects
- **Input**: None (uses self.summary_points)
- **Output**: List of SummaryPoint objects
- **Flow**: 
  - Iterates through self.summary_points
  - Creates SummaryPoint object for each dict
  - Extracts text, prev_link, next_link using .get()

##### `has_prev_links() -> bool`
- **Purpose**: Check if any summary points have links to previous chunk
- **Input**: None
- **Output**: Boolean
- **Flow**: Uses `any()` with generator to check if any summary point has prev_link

##### `has_next_links() -> bool`
- **Purpose**: Check if any summary points have links to next chunk
- **Input**: None
- **Output**: Boolean
- **Flow**: Uses `any()` with generator to check if any summary point has next_link

---

#### `ChunkReader`
**Purpose**: Reader for navigating chunks stored in ChromaDB with sequential/random access

**Initialization**:
- **Input**: `chroma_db_path: str`, `collection_name: str`
- **Flow**:
  - Creates ChromaDB PersistentClient
  - Gets collection by name
  - Initializes cache and current index

**Methods**:

##### `_parse_metadata_field(field_value) -> Any`
- **Purpose**: Parse metadata field that might be JSON-encoded (ChromaDB stores complex types as JSON strings)
- **Input**: field_value (might be string or already parsed)
- **Output**: Parsed value (list/dict) or original
- **Flow**:
  - Check if value is string
  - Try JSON decode
  - Return parsed or original on error

##### `load_all_chunks(force_reload: bool = False) -> List[Chunk]`
- **Purpose**: Load all chunks from ChromaDB with caching
- **Input**: force_reload flag
- **Output**: List of Chunk objects sorted by chunk_index
- **Flow**:
  - Return cache if available and not force_reload
  - Call `collection.get()` to retrieve all documents
  - Iterate through ids, documents, metadatas together using `zip()`
  - Parse headers and summary_points using `_parse_metadata_field()`
  - Create Chunk objects
  - Sort by chunk_index
  - Cache and return

##### `get_chunk_by_index(chunk_index: int) -> Optional[Chunk]`
- **Purpose**: Get chunk by its chunk_index
- **Input**: chunk_index
- **Output**: Chunk or None
- **Flow**:
  - Load all chunks
  - Linear search for matching chunk_index

##### `get_chunk_by_id(chunk_id: str) -> Optional[Chunk]`
- **Purpose**: Get chunk by its ID
- **Input**: chunk_id
- **Output**: Chunk or None
- **Flow**:
  - Try/except block
  - Call `collection.get(ids=[chunk_id])`
  - Parse metadata fields
  - Create and return Chunk object

##### `get_current_chunk() -> Optional[Chunk]`
- **Purpose**: Get chunk at current navigation index
- **Output**: Chunk or None
- **Flow**: Check bounds, return chunk at _current_index

##### `next_chunk() -> Optional[Chunk]`
- **Purpose**: Move to and return next chunk
- **Output**: Chunk or None
- **Flow**: Increment _current_index if not at end, return chunk

##### `prev_chunk() -> Optional[Chunk]`
- **Purpose**: Move to and return previous chunk
- **Output**: Chunk or None
- **Flow**: Decrement _current_index if not at start, return chunk

##### `jump_to(index: int) -> Optional[Chunk]`
- **Purpose**: Jump to specific position in chunk sequence
- **Input**: index (0-based)
- **Output**: Chunk or None
- **Flow**: Validate bounds, set _current_index, return chunk

##### `get_chunks_by_source(source_file: str) -> List[Chunk]`
- **Purpose**: Filter chunks by source file
- **Input**: source_file path
- **Output**: List of matching chunks
- **Flow**: List comprehension filtering by source_file

##### `get_chunks_with_header(header: str) -> List[Chunk]`
- **Purpose**: Filter chunks containing specific header
- **Input**: header text
- **Output**: List of matching chunks
- **Flow**: List comprehension checking if header in c.headers

##### `get_chunks_with_summaries() -> List[Chunk]`
- **Purpose**: Get chunks that have summary points
- **Output**: List of chunks
- **Flow**: Filter chunks where summary_points is truthy

##### `get_linked_chunks() -> List[Chunk]`
- **Purpose**: Get chunks with prev/next links
- **Output**: List of chunks
- **Flow**: Filter using has_prev_links() or has_next_links()

##### `reset()`
- **Purpose**: Reset navigation to beginning
- **Flow**: Set _current_index = 0

---

### Demo Functions (not classes, but important)

All demo functions follow similar pattern:
- Print section headers
- Demonstrate specific functionality
- Print results

Key demonstrations:
- `demo_basic_operations()`: Loading, getting by index/ID
- `demo_navigation()`: Sequential navigation (next, prev, jump)
- `demo_filtering()`: Finding chunks with summaries, links, by source
- `demo_link_traversal()`: Following prev/next links between chunks, chain traversal
- `demo_statistics()`: Overall statistics about chunks

---

## Document 2: `lg.py` - QA Dataset Generator using LangGraph

### Data Models (Pydantic)

#### `QAGeneratorConfig`
Configuration for the entire pipeline

#### `Question`
Question with metadata (id, question, type, source_chunk_id, rationale, timestamp)

#### `ReasoningStep`
Single step in reasoning trace (step_number, thought, action, action_input, observation)

#### `AnswerResult`
Answer with reasoning steps, source chunks, iterations, completion status, quality score

#### `DatasetEntry`
Final dataset entry combining question, answer, reasoning, metadata

#### `AnswerGenState` (TypedDict)
LangGraph state containing question, messages, iteration count, max_iterations

---

### Classes

#### `ToolRegistry`
**Purpose**: Manages and provides tools for answer generation

**Initialization**:
- **Input**: config: QAGeneratorConfig
- **Flow**:
  - Initialize logger
  - Call `_init_clients()`
  - Based on config.data_source_type, initialize:
    - ChunkReader for doc/pair types
    - Code graph for code/pair types
  - Initialize embedding client (AsyncOpenAI)

**Methods**:

##### `_init_clients()`
- **Purpose**: Initialize ChromaDB reader, code graph, embedding client
- **Flow**:
  - Try to create ChunkReader if doc/pair type
  - Try to load code graph pickle if code/pair type
  - Create AsyncOpenAI embedding client

##### `async search_docs(query: str, n_results: int = 5) -> str`
- **Purpose**: Semantic search in documentation
- **Input**: query string, number of results
- **Output**: JSON string with search results
- **Flow**:
  - **Async call**: `await embedding_client.embeddings.create()`
  - Extract embedding from response
  - Call `collection.query()` with embedding
  - Iterate through results
  - For each result: get full Chunk object using `get_chunk_by_id()`
  - Build enhanced chunk_info dict with:
    - Basic info (id, index, content preview, source, headers)
    - Summary points if available
    - Navigation hints (prev_chunk, next_chunk) with relation/topic
  - Return JSON dumps

##### `async search_code(query: str, top_k: int = 5) -> str`
- **Purpose**: Semantic search in code graph
- **Input**: query, top_k
- **Output**: JSON string
- **Flow**:
  - **Async call**: `await embedding_client.embeddings.create()`
  - Calculate cosine similarity with all graph nodes
  - Sort by similarity
  - Return top_k results as JSON

##### `get_tools() -> List`
- **Purpose**: Create and return tool list based on config
- **Output**: List of tool functions
- **Flow**:
  - Initialize empty tools list
  - If doc/pair type:
    - **Define async tool** using `@tool` decorator:
      ```python
      @tool
      async def search_documentation(query: str, n_results: int = 5) -> str:
          """Search documentation using semantic similarity"""
          return await self.search_docs(query, n_results)
      ```
  - **Define get_linked_chunk tool** (async):
    ```python
    @tool
    async def get_linked_chunk(chunk_id: str, direction: str = "next") -> str:
        """Get previous or next linked chunk"""
        # Implementation fetches linked chunk and returns JSON
    ```
  - If code/pair type:
    - **Define async tool** `search_codebase`
  - Return tools list

**Important**: All tools are async and use `await` for operations

---

#### `QuestionGenerator`
**Purpose**: Generate questions from chunks using structured outputs

**Initialization**:
- **Input**: config
- **Flow**:
  - Setup logger
  - Create ChatOpenAI with `use_responses_api=True`

**Pydantic Models Used**:
- `GeneratedQuestion`: question, question_type, rationale
- `QuestionBatch`: questions list
- `FilterIndices`: indices_to_keep list

**Methods**:

##### `async generate_for_chunk(chunk: Chunk) -> List[Question]`
- **Purpose**: Generate and filter questions for a chunk
- **Input**: Chunk object
- **Output**: List of Question objects
- **Flow**:
  - Calculate easy/medium counts
  - Build generation prompt
  - **Use structured output**: `llm.with_structured_output(QuestionBatch)`
  - **Async call**: `await structured_llm.ainvoke(prompt)`
  - Response is typed as `QuestionBatch`
  - Convert to Question objects
  - Build filter prompt
  - **Use structured output**: `llm.with_structured_output(FilterIndices)`
  - **Async call**: `await filter_structured_llm.ainvoke(filter_prompt)`
  - Filter questions using 1-indexed positions from response
  - Return filtered questions

---

#### `AnswerGenerator`
**Purpose**: Generate answers using LangGraph with tool calling

**Initialization**:
- **Input**: config, tool_registry
- **Flow**:
  - Get tools from registry
  - Setup logger
  - Create main_llm with `use_responses_api=True` and `.bind_tools(self.tools)`
  - Create synthesis_llm (without tools bound)

**Pydantic Model**:
- `FinalAnswerResponse`: answer, source_chunks_used

**Methods**:

##### `build_graph() -> StateGraph`
- **Purpose**: Build LangGraph workflow
- **Output**: Compiled StateGraph
- **Flow**:
  - Create StateGraph with AnswerGenState
  - Add "agent" node → self.agent_node
  - Add "tools" node → ToolNode(self.tools)
  - Set entry point to "agent"
  - Add conditional edges: agent → should_continue → {continue: tools, end: END}
  - Add edge: tools → agent
  - **Compile and return**

##### `async agent_node(state: AnswerGenState) -> AnswerGenState`
- **Purpose**: LangGraph agent node - invokes LLM
- **Input**: State
- **Output**: Updated state
- **Flow**:
  - **Async call**: `await self.main_llm.ainvoke(state["messages"])`
  - Append response to messages
  - Increment iteration
  - Return state

##### `should_continue(state: AnswerGenState) -> Literal["continue", "end"]`
- **Purpose**: Decide if workflow should continue or end
- **Input**: State
- **Output**: "continue" or "end"
- **Flow**:
  - Check if max_iterations reached → "end"
  - Check if last message has tool_calls → "continue"
  - Otherwise → "end"

##### `async generate_for_question(question: Question) -> AnswerResult`
- **Purpose**: Generate answer for a question using LangGraph
- **Input**: Question
- **Output**: AnswerResult
- **Flow**:
  - Build graph using `build_graph()`
  - Create initial state with system message and user message
  - **Async call**: `await graph.ainvoke(initial_state)`
  - Extract reasoning steps from messages using `_extract_reasoning()`
  - Extract source chunks using `_extract_sources()`
  - Format context for synthesis
  - **Use structured output**: `synthesis_llm.with_structured_output(FinalAnswerResponse)`
  - **Async call**: `await structured_llm_with_schema.ainvoke(formatted_context)`
  - Build and return AnswerResult

##### `_format_context_for_synthesis(original_question: str, messages: List[BaseMessage]) -> str`
- **Purpose**: Format conversation history for final answer synthesis
- **Flow**:
  - Build list of context parts
  - Iterate through messages
  - Format each type (HumanMessage, AIMessage with/without tool_calls, ToolMessage)
  - Join and return

##### `_extract_reasoning(messages: List[BaseMessage]) -> List[ReasoningStep]`
- **Purpose**: Extract actual reasoning steps from workflow execution
- **Flow**:
  - Iterate through messages
  - For AIMessage with tool_calls:
    - Extract tool call info
    - Get observation from next ToolMessage if exists
    - Create ReasoningStep
  - Return steps list

##### `_extract_sources(messages: List[BaseMessage]) -> List[str]`
- **Purpose**: Extract source chunk/entity IDs from messages
- **Flow**:
  - Iterate messages
  - Extract JSON from content
  - Look for chunk_id or entity_id
  - Return unique sources

---

#### `AnswerJudge`
**Purpose**: Evaluate answer quality using structured outputs

**Pydantic Model**:
- `EvaluationScores`: completeness, accuracy, relevance, clarity, specificity, reasoning (all float 0-1)

**Methods**:

##### `async evaluate(question: Question, answer: AnswerResult) -> Dict[str, float]`
- **Purpose**: Evaluate answer quality
- **Input**: Question, AnswerResult
- **Output**: Dict of scores
- **Flow**:
  - Build evaluation prompt
  - **Use structured output**: `llm.with_structured_output(EvaluationScores)`
  - **Async call**: `await structured_llm.ainvoke(prompt)`
  - Convert to dict using `scores.model_dump()` (Pydantic v2)
  - Calculate overall score
  - Return scores dict

---

#### `DatasetAssembler`
**Purpose**: Assemble final dataset from questions and answers

**Methods**:

##### `assemble(questions: List[Question], answers: List[AnswerResult]) -> List[DatasetEntry]`
- **Purpose**: Combine questions and answers into dataset entries
- **Flow**:
  - Create question map by ID
  - Iterate answers
  - Match with question
  - Format reasoning using `_format_reasoning()`
  - Create DatasetEntry
  - Return entries list

##### `_format_reasoning(steps: List[ReasoningStep]) -> str`
- **Purpose**: Format reasoning steps with special tokens
- **Flow**:
  - Iterate steps
  - Format with tags: `<|step|>`, `<|thought|>`, `<|action|>`, `<|observation|>`
  - Join and return

##### `save(entries: List[DatasetEntry], path: str)`
- **Purpose**: Save dataset to JSONL file
- **Flow**:
  - Create output directory
  - Write each entry as JSON line using `model_dump()`

---

#### `QAGeneratorPipeline`
**Purpose**: Main pipeline orchestrating all phases

**Methods**:

##### `async run() -> str`
- **Purpose**: Run complete pipeline
- **Output**: Output file path
- **Flow**:
  
  **Phase 1 - Question Generation**:
  - Load chunks from ChromaDB
  - Create semaphore for concurrency control
  - Define async gen_q function wrapper
  - Create tasks for all chunks
  - Use `asyncio.as_completed()` with tqdm
  - **Async iteration**: `for coro in asyncio.as_completed(tasks): await coro`
  
  **Phase 2 - Answer Generation**:
  - Create ToolRegistry
  - Create AnswerGenerator and AnswerJudge
  - Define async gen_a function wrapper that:
    - Generates answer
    - Evaluates quality
    - Returns if quality threshold met
  - Create tasks for all questions
  - Use `asyncio.as_completed()` pattern
  
  **Phase 3 - Assembly**:
  - Create DatasetAssembler
  - Assemble entries
  - Save to file
  - Return path

---

## Document 3: `custom_llm_tool.py` - Custom LLM with Tool Calling

### Key Concepts

**Two-Stage Architecture**:
1. **Planner LLM**: Decides which tool to call and describes what information is needed (natural language)
2. **Executor LLM**: Converts planner's description into structured parameters matching tool schema

**Three Output Modes**:
- `LANGCHAIN`: Uses LangChain's `.with_structured_output()`
- `OPENAI_JSON`: Uses OpenAI's JSON schema API
- `MANUAL`: Manual JSON parsing with retry logic

---

### Enums

#### `OutputMode`
Values: LANGCHAIN, OPENAI_JSON, MANUAL

---

### Pydantic Models

#### `ToolCallDecision`
- content: str (final answer OR description of tool info needed)
- tool_name: Optional[str] (null for final answer)

---

### Classes

#### `CustomLLMWithTools`
**Purpose**: Two-stage LLM tool calling with multiple structured output modes, supports sync and async

**Initialization**:
- **Input**: mode, base_url, model_name, tools, planner_system_prompt, executor_system_prompt
- **Flow**:
  - Based on mode:
    - **LANGCHAIN**: Create two ChatOpenAI instances (planner, executor)
    - **OPENAI_JSON**: Create OpenAI client
    - **MANUAL**: Create two ChatOpenAI instances
  - Initialize empty tools and tool_schemas dicts
  - Register tools if provided

**Methods**:

##### `_get_tool_function(tool_func: Callable) -> Callable`
- **Purpose**: Extract actual function from tool wrapper (handles LangChain decorators)
- **Input**: Tool function (possibly wrapped)
- **Output**: Actual callable or None
- **Flow**:
  - **Try multiple attributes**: 'func', 'coroutine', '_target', '__wrapped__'
  - Check if callable
  - Check for invoke/ainvoke (LangChain pattern)
  - If LangChain tool, check for args_schema → return None (signal schema-based approach)
  - Raise error if cannot extract

##### `_parse_docstring(docstring: str) -> tuple[str, dict]`
- **Purpose**: Parse custom docstring format with '---' separator
- **Input**: Docstring
- **Output**: (description, parameters_dict)
- **Flow**:
  - Split on '---'
  - First part = description
  - Second part = parameters
  - Parse each parameter line: `<type> <param_name>: Description`
  - Convert type string to Python type
  - Build params_dict with type and description
  - Return tuple

##### `_register_tool(tool_func: Callable)`
- **Purpose**: Register tool and extract schema from docstring
- **Flow**:
  - Extract tool name
  - Get docstring
  - Parse docstring → description, params_from_doc
  - **Get actual function** using `_get_tool_function()`
  - **If None (LangChain tool with schema)**:
    - Extract from `tool_func.args_schema.model_fields`
    - Iterate schema fields
    - Match with documented params
    - Build params_info dict
    - Store in tool_schemas
    - **Early return**
  - **Otherwise (regular function)**:
    - Get function signature using `inspect.signature()`
    - Iterate parameters
    - Validate against documented params
    - Build params_info with type, description, default, required
    - Verify all documented params in signature
    - Store in tool_schemas
  - Print registration confirmation

##### `_create_tool_param_model(tool_name: str) -> type[BaseModel]`
- **Purpose**: Dynamically create Pydantic model for tool parameters
- **Flow**:
  - Get schema from tool_schemas
  - Build fields dict for Pydantic
  - For each parameter:
    - If optional: use Optional[type] with default
    - If required: use type with Field(...)
  - Use `create_model()` to generate Pydantic model dynamically
  - Return model class

##### `_format_tools_for_planning() -> str`
- **Purpose**: Format tool descriptions for planner LLM
- **Flow**:
  - Build string with all tools
  - For each tool: name, description, parameters with types and requirements
  - Return formatted string

---

**Planner Invocation Methods** (3 modes × 2 async variants = 6 methods):

##### `_invoke_planner_langchain(messages, tools_desc) -> ToolCallDecision`
- **Purpose**: Invoke planner using LangChain structured output (sync)
- **Flow**:
  - Create dynamic PlannerDecision model with tool_name validator
  - Create structured_llm using `.with_structured_output(PlannerDecision)`
  - Build messages: system prompt + conversation + tools description
  - **Call**: `structured_llm.invoke(messages)`
  - Return ToolCallDecision

##### `async _invoke_planner_langchain_async(messages, tools_desc) -> ToolCallDecision`
- **Same as above but uses `await structured_llm.ainvoke()`**

##### `_invoke_planner_openai(messages, tools_desc) -> ToolCallDecision`
- **Purpose**: Invoke planner using OpenAI JSON schema (sync)
- **Flow**:
  - Build JSON schema with properties: content, tool_name (enum of tool names)
  - Convert LangChain messages to OpenAI format
  - **Call**: `client.chat.completions.create()` with response_format specifying json_schema
  - Parse JSON response
  - Return ToolCallDecision

##### `async _invoke_planner_openai_async(messages, tools_desc) -> ToolCallDecision`
- **Same but creates AsyncOpenAI client if needed and uses `await`**

##### `_invoke_planner_manual(messages, tools_desc) -> ToolCallDecision`
- **Purpose**: Invoke planner with manual JSON parsing (sync)
- **Flow**:
  - Build prompt with JSON format specification
  - **Retry loop (max 3)**:
    - **Call**: `llm.invoke(messages)`
    - Extract JSON (handle ```json``` code blocks with regex)
    - Parse JSON
    - Validate content field not empty
    - Return ToolCallDecision
    - On error: append error message, retry

##### `async _invoke_planner_manual_async(messages, tools_desc) -> ToolCallDecision`
- **Same but uses `await llm.ainvoke()`**

---

**Executor Invocation Methods** (3 modes × 2 async variants = 6 methods):

##### `_invoke_executor_langchain(tool_name, planner_content, ParamModel) -> BaseModel`
- **Purpose**: Invoke executor using LangChain structured output (sync)
- **Flow**:
  - Get tool schema
  - Create structured_llm using `.with_structured_output(ParamModel)`
  - Build prompt with tool description and planner's request
  - **Call**: `structured_llm.invoke(messages)`
  - Return parameters (instance of ParamModel)

##### `async _invoke_executor_langchain_async(...) -> BaseModel`
- **Same but uses `await structured_llm.ainvoke()`**

##### `_invoke_executor_openai(tool_name, planner_content, ParamModel) -> BaseModel`
- **Purpose**: Invoke executor using OpenAI JSON schema (sync)
- **Flow**:
  - Get Pydantic schema using `ParamModel.model_json_schema()`
  - Convert to OpenAI schema format
  - Build prompt
  - **Call**: `client.chat.completions.create()` with json_schema
  - Parse response
  - Create and return ParamModel instance

##### `async _invoke_executor_openai_async(...) -> BaseModel`
- **Same but creates AsyncOpenAI client and uses `await`**

##### `_invoke_executor_manual(tool_name, planner_content, ParamModel) -> BaseModel`
- **Purpose**: Invoke executor with manual JSON parsing (sync)
- **Flow**:
  - Build expected JSON structure string
  - Build prompt
  - **Retry loop (max 3)**:
    - **Call**: `llm.invoke(messages)`
    - Extract JSON (handle code blocks)
    - Parse JSON
    - Create ParamModel instance (validates)
    - Return
    - On error: append error, retry

##### `async _invoke_executor_manual_async(...) -> BaseModel`
- **Same but uses `await llm.ainvoke()`**

---

**Main Invocation Methods**:

##### `invoke(messages: List[BaseMessage]) -> AIMessage`
- **Purpose**: Main synchronous invocation - coordinates planner and executor
- **Input**: List of messages
- **Output**: AIMessage (with content OR tool_calls)
- **Flow**:
  
  **Stage 1: Planning**
  - Format tools description
  - Based on mode, call appropriate planner method:
    - LANGCHAIN: `_invoke_planner_langchain()`
    - OPENAI_JSON: `_invoke_planner_openai()`
    - MANUAL: `_invoke_planner_manual()`
  - Get ToolCallDecision
  - If no tool_name → return AIMessage with final answer
  - Validate tool exists
  
  **Stage 2: Parameter Generation**
  - Create ParamModel using `_create_tool_param_model()`
  - **Retry loop (max 3)**:
    - Try executor invocation based on mode
    - On success: break
    - On failure:
      - Build error message
      - Re-invoke planner with error feedback
      - Get revised planner description
      - Continue retry
  - Create ToolCall object from parameters
  - Return AIMessage with tool_calls

##### `async ainvoke(messages: List[BaseMessage]) -> AIMessage`
- **Same flow as invoke() but uses async variants**:
  - `_invoke_planner_*_async()`
  - `_invoke_executor_*_async()`
  - All calls use `await`

##### `bind_tools(tools: List[Callable])`
- **Purpose**: Register tools (for compatibility with LangChain API)
- **Flow**: Iterate and call `_register_tool()` for each

---

### Important Implementation Notes

1. **Async Tool Registration**:
   - Tools can be sync or async
   - `@tool` decorator from LangChain wraps functions
   - Extraction uses `_get_tool_function()` to unwrap
   - For LangChain tools, uses `args_schema` instead of inspecting function

2. **Docstring Format**:
   ```
   Description
   ---
   <type> <param_name>: Description
   ```

3. **Dynamic Pydantic Models**:
   - Uses `create_model()` to generate models at runtime
   - Field validation using Pydantic Field() and validators

4. **Error Handling**:
   - Retry loops for both planner and executor
   - Planner gets feedback from executor failures
   - JSON extraction handles code blocks with regex

5. **Async Client Management**:
   - For OPENAI_JSON mode, creates AsyncOpenAI client lazily
   - Stored as `_async_client` attribute
