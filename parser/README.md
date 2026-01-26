## Document 1: `chunks.py` - Chunking and LLM Linking

### Classes

#### 1. **SummaryPointsResponse** (Pydantic Model)
- **Purpose**: Structured output for LLM summary generation
- **Fields**: `points: List[str]` - 3-5 key summary points

#### 2. **LinkInfo** (Pydantic Model)
- **Purpose**: Structured output for chunk relationship analysis
- **Fields**:
  - `relates: bool` - whether chunks are related
  - `relation: Optional[str]` - type of relation (e.g., "continues discussion")
  - `common_topic: Optional[str]` - shared topic

#### 3. **ChunkLinksResponse** (Pydantic Model)
- **Purpose**: Container for multiple LinkInfo objects
- **Fields**: `links: List[LinkInfo]`

#### 4. **SummaryPoint** (Dataclass)
- **Purpose**: Represents a summary point with bidirectional links
- **Fields**:
  - `text: str` - summary text
  - `prev_link: Optional[Dict]` - link to previous chunk
  - `next_link: Optional[Dict]` - link to next chunk

#### 5. **Chunk** (Main Class)
- **Purpose**: Represents a text chunk with metadata and linking
- **Fields**:
  - `id: str` (UUID)
  - `content: str`
  - `source_file: str`
  - `chunk_index: int`
  - `start_char/end_char: int`
  - `headers: List[str]`
  - `metadata: Dict`
  - `summary_points: List[SummaryPoint]`

##### Methods:

**`to_dict() -> Dict`**
- **Purpose**: Serialize chunk to dictionary
- **I/O Flow**:
  - Input: None (uses instance data)
  - Output: Dict with all chunk fields
  - Converts SummaryPoint objects to dicts

---

#### 6. **SemanticChunker** (Main Chunking Class)

**`__init__(...)`**
- **Purpose**: Initialize chunker with optional LLM linking
- **I/O Flow**:
  - Input: chunk_size, overlap, LLM config params
  - Creates `asyncio.Semaphore` for rate limiting
  - Initializes `ChatOpenAI` from langchain_openai
  - Creates structured output chains:
    ```python
    self.llm = ChatOpenAI(model=..., api_key=..., temperature=...)
    self.summary_chain = self.llm.with_structured_output(SummaryPointsResponse)
    self.linking_chain = self.llm.with_structured_output(ChunkLinksResponse)
    ```

**`chunk_markdown(md_content: str, source_file: str) -> List[Chunk]`**
- **Purpose**: Sync wrapper for chunking with optional LLM linking
- **I/O Flow**:
  1. Calls `_do_basic_chunking()` → returns chunks
  2. If LLM enabled:
     - Creates new event loop
     - Calls `_apply_llm_linking_async(chunks)`
     - Closes loop
  3. Returns chunks

**`chunk_markdown_async(md_content: str, source_file: str) -> List[Chunk]`**
- **Purpose**: Async version for async contexts
- **I/O Flow**:
  1. Calls `_do_basic_chunking()` → chunks
  2. If LLM enabled: `await _apply_llm_linking_async(chunks)`
  3. Returns chunks

**`_do_basic_chunking(md_content: str, source_file: str) -> List[Chunk]`**
- **Purpose**: Core markdown chunking logic
- **I/O Flow**:
  1. Split by headers (regex: `^(#{1,6})\s+(.*)$`)
  2. Track header hierarchy
  3. Create sections: `List[(start_idx, end_idx, headers)]`
  4. For each section → `_chunk_section()`
  5. Returns all chunks

**`_chunk_section(...) -> List[Chunk]`**
- **Purpose**: Chunk a single section
- **I/O Flow**:
  1. If section ≤ chunk_size → single Chunk
  2. Else: split by paragraphs (`\n\n`)
  3. Accumulate paragraphs until chunk_size
  4. If paragraph > chunk_size → `_split_large_paragraph()`
  5. Create Chunk objects with headers/metadata
  6. Returns chunks

**`_split_large_paragraph(...) -> List[Chunk]`**
- **Purpose**: Handle oversized paragraphs
- **I/O Flow**:
  1. Split by sentences (regex: `[.!?]+`)
  2. Accumulate sentences until chunk_size
  3. If sentence > chunk_size → hard cut by character count
  4. Mark metadata: `part_of_large_para: True`, `truncated: True`
  5. Returns chunks

**`_apply_llm_linking_async(chunks: List[Chunk]) -> List[Chunk]`**
- **Purpose**: Apply LLM-based linking in 3 concurrent phases
- **I/O Flow**:
  1. **Phase 1**: Generate summaries (concurrent)
     - Tasks: `[_generate_summary_async(chunk.content) for chunk in chunks]`
     - `await asyncio.gather(*tasks)`
     - Assign summaries to chunks
  2. **Phase 2**: Link with previous (concurrent)
     - Tasks: `[_link_with_previous_async(chunks[i-1], chunks[i])]`
     - `await asyncio.gather(*tasks)`
  3. **Phase 3**: Link with next (concurrent)
     - Tasks: `[_link_with_next_async(chunks[i], chunks[i+1])]`
     - `await asyncio.gather(*tasks)`
  4. Returns chunks with populated summary_points

**`_generate_summary_async(content: str) -> List[SummaryPoint]`**
- **Purpose**: Generate 3-5 summary points via LLM
- **I/O Flow**:
  1. Acquire semaphore (`async with self.llm_semaphore`)
  2. Create prompt with content
  3. Call `await self.summary_chain.ainvoke(prompt)`
  4. Returns: `[SummaryPoint(text=point) for point in response.points]`
  5. On error: returns `[SummaryPoint(text="Summary generation failed")]`

**`_link_with_previous_async(prev_chunk: Chunk, curr_chunk: Chunk)`**
- **Purpose**: Link current chunk's summaries to previous chunk
- **I/O Flow**:
  1. Acquire semaphore
  2. Build prompt with both chunks + current summary points
  3. Call `await self.linking_chain.ainvoke(prompt)`
  4. For each link_info:
     - If `relates == True`:
       - Set `curr_chunk.summary_points[i].prev_link = {chunk_id, chunk_index, relation, common_topic}`

**`_link_with_next_async(curr_chunk: Chunk, next_chunk: Chunk)`**
- **Purpose**: Link current chunk's summaries to next chunk
- **I/O Flow**:
  1. Acquire semaphore
  2. Build prompt with both chunks + current summary points
  3. Call `await self.linking_chain.ainvoke(prompt)`
  4. For each link_info:
     - If `relates == True`:
       - Set `curr_chunk.summary_points[i].next_link = {chunk_id, chunk_index, relation, common_topic}`

---

## Document 2: `converter.py` - Document Conversion

### Class: **DocumentConverter**

**`__init__(config: DocParserConfig)`**
- **Purpose**: Initialize converter with config
- **I/O**: Stores config reference

**`convert_to_markdown(file_path: str) -> str`**
- **Purpose**: Route to appropriate converter based on extension
- **I/O Flow**:
  - Input: file path
  - Checks extension → calls specific converter
  - Output: markdown string
  - Supported: `.pdf, .docx, .html, .htm, .md, .txt, .xlsx, .pptx`

**`_convert_pdf(path: str) -> str`**
- **Purpose**: Convert PDF using MinerU CLI
- **I/O Flow**:
  1. Create temp directory
  2. Run subprocess: `["mineru", "-p", path, "-o", temp_dir]`
  3. Find output `.md` files in temp_dir
  4. Read first `.md` file
  5. Returns markdown content
  6. Cleans up temp dir automatically

**`_convert_docx(path: str) -> str`**
- **Purpose**: Extract text from Word documents
- **I/O Flow**:
  1. Import: `import docx`
  2. Load: `doc = docx.Document(path)`
  3. Extract: `[p.text for p in doc.paragraphs if p.text.strip()]`
  4. Join with `\n\n`
  5. Returns markdown string

**`_convert_html(path: str) -> str`**
- **Purpose**: Convert HTML to markdown
- **I/O Flow**:
  1. Import: `from bs4 import BeautifulSoup; import html2text`
  2. Read file content
  3. Parse: `soup = BeautifulSoup(content, "html.parser")`
  4. Configure html2text: `h.ignore_links = True; h.body_width = 0`
  5. Convert: `h.handle(str(soup))`
  6. Returns markdown

**`_convert_md(path: str) -> str`**
- **Purpose**: Read markdown file directly
- **I/O Flow**: Simple file read with UTF-8 encoding

**`_convert_txt(path: str) -> str`**
- **Purpose**: Read text file directly
- **I/O Flow**: Simple file read with UTF-8 encoding

**`_convert_xlsx(path: str) -> str`**
- **Purpose**: Convert Excel to markdown tables
- **I/O Flow**:
  1. Import: `import pandas as pd`
  2. Load: `xl_file = pd.ExcelFile(path)`
  3. For each sheet:
     - `df = pd.read_excel(path, sheet_name=sheet_name)`
     - `table_md = df.to_markdown(index=False)`
     - Create section: `## Sheet: {name}\n\n{table_md}`
  4. Join all sheets
  5. Returns combined markdown

**`_convert_pptx(path: str) -> str`**
- **Purpose**: Extract text and notes from PowerPoint
- **I/O Flow**:
  1. Import: `from pptx import Presentation`
  2. Load: `prs = Presentation(path)`
  3. For each slide:
     - Extract text from shapes
     - Extract notes from `slide.notes_slide.notes_text_frame.text`
     - Create section: `### Slide {i+1}\n{text}\nNotes:\n{notes}`
  4. Join all slides
  5. Returns markdown

---

## Document 3: `embeddings.py` - Embeddings and ChromaDB

### Class 1: **EmbeddingGenerator**

**`__init__(endpoint: str, model: str)`**
- **Purpose**: Initialize OpenAI-compatible embedding client
- **I/O Flow**:
  ```python
  from openai import OpenAI
  self.client = OpenAI(base_url=endpoint, api_key="dummy")
  self.model = model
  ```

**`generate_embeddings(chunks: List, batch_size: int = 32, include_summary: bool = False) -> List[np.ndarray]`**
- **Purpose**: Generate embeddings with batching and optional summary augmentation
- **I/O Flow**:
  1. Process in batches of size `batch_size`
  2. For each batch:
     - If `include_summary`: prepare text with `_prepare_text_with_summary()`
     - Else: use `chunk.content`
  3. Call OpenAI API:
     ```python
     response = self.client.embeddings.create(
         input=batch_texts, 
         model=self.model
     )
     ```
  4. Extract embeddings: `[np.array(data.embedding) for data in response.data]`
  5. Returns list of numpy arrays

**`_prepare_text_with_summary(chunk) -> str`**
- **Purpose**: Augment chunk text with summary and linking context
- **I/O Flow**:
  1. Start with `chunk.content`
  2. If `summary_points` exist:
     - Add "Key Points:" section
     - For each summary point:
       - Add numbered point text
       - If `prev_link`: add context line
       - If `next_link`: add leads-to line
  3. Join all parts with `\n`
  4. Returns enriched text

**`_embed_batch(texts: List[str]) -> List[np.ndarray]`**
- **Purpose**: Single batch embedding with error handling
- **I/O Flow**:
  1. Call `self.client.embeddings.create(input=texts, model=self.model)`
  2. Extract: `[np.array(data.embedding) for data in response.data]`
  3. Returns embeddings

---

### Class 2: **ChromaDBManager**

**`__init__(db_path: str, collection_name: str)`**
- **Purpose**: Initialize persistent ChromaDB client
- **I/O Flow**:
  ```python
  import chromadb
  os.makedirs(db_path, exist_ok=True)
  self.client = chromadb.PersistentClient(path=db_path)
  self.collection = self.client.get_or_create_collection(
      name=collection_name,
      metadata={"hnsw:space": "cosine"}  # cosine similarity
  )
  ```

**`add_chunks(chunks: List, embeddings: List[np.ndarray])`**
- **Purpose**: Store chunks with embeddings in ChromaDB
- **I/O Flow**:
  1. Extract IDs: `[chunk.id for chunk in chunks]`
  2. Extract documents: `[chunk.content for chunk in chunks]`
  3. Build metadata for each chunk:
     - Basic fields: `source_file, chunk_index, start_char, end_char, headers, timestamp`
     - Convert headers to JSON: `json.dumps(chunk.headers)`
     - If summary_points exist:
       ```python
       summary_data = [{
           "text": sp.text,
           "prev_link": sp.prev_link,
           "next_link": sp.next_link
       } for sp in chunk.summary_points]
       metadata["summary_points"] = json.dumps(summary_data)
       ```
     - Convert additional metadata (lists/dicts → JSON strings)
  4. Convert embeddings: `[emb.tolist() for emb in embeddings]`
  5. **ChromaDB Insert**:
     ```python
     self.collection.add(
         ids=ids,
         documents=documents,
         metadatas=metadatas,
         embeddings=embeddings_list
     )
     ```

**`query(query_embedding: np.ndarray, n_results: int = 5, include_context: bool = True) -> List[Dict]`**
- **Purpose**: Vector similarity search with optional context parsing
- **I/O Flow**:
  1. **ChromaDB Query**:
     ```python
     results = self.collection.query(
         query_embeddings=[query_embedding.tolist()],
         n_results=n_results
     )
     ```
  2. For each result:
     - Parse `summary_points` from JSON if `include_context=True`
     - Build result dict: `{id, content, metadata, summary_points, distance}`
  3. Returns formatted results list

**`get_chunk_with_context(chunk_id: str) -> Optional[Dict]`**
- **Purpose**: Retrieve chunk and identify linked chunks
- **I/O Flow**:
  1. **ChromaDB Get by ID**:
     ```python
     result = self.collection.get(
         ids=[chunk_id],
         include=["documents", "metadatas"]
     )
     ```
  2. If no results: return None
  3. Build context dict:
     - `current`: chunk data
     - Parse `summary_points` from JSON
     - Extract `prev_topics` and `next_topics` from links
  4. Returns context dict

---

## Document 4: `parser.py` - Main Pipeline

### Class 1: **MarkdownConsolidator**

**`__init__(output_path: str)`**
- **Purpose**: Initialize consolidation file manager
- **I/O Flow**:
  - Creates output directory
  - Sets file separator: `"\n\n" + "=" * 80 + "\n\n"`

**`append_document(md_content: str, source_file: str)`**
- **Purpose**: Append markdown with metadata header
- **I/O Flow**:
  1. Create header:
     ```
     # SOURCE: {source_file}
     # TIMESTAMP: {iso_timestamp}
     # FORMAT: {extension}
     ---
     ```
  2. Open file in append mode
  3. If file has content: write separator
  4. Write header + content

**`finalize() -> str`**
- **Purpose**: Return path to consolidated file
- **I/O**: Returns `self.output_path`

---

### Class 2: **DocumentParser** (Main Pipeline)

**`__init__(config: DocParserConfig)`**
- **Purpose**: Initialize all pipeline components
- **I/O Flow**:
  1. Create `DocumentConverter(config)`
  2. Create `MarkdownConsolidator(config.output_md_path)`
  3. Create `SemanticChunker` with:
     - Basic params: chunk_size, overlap
     - If LLM linking enabled: add LLM params (api_key, base_url, model, concurrency, temperature)
  4. Create `EmbeddingGenerator(endpoint, model)`
  5. Create `ChromaDBManager(db_path, collection_name)`
  6. Create cache directories:
     - `.doc_parser_cache/` in output dir
     - `.doc_parser_cache/markdown/` subdirectory

**`_get_cache_path(step: str, file_path: str) -> str`**
- **Purpose**: Generate cache file path
- **I/O Flow**:
  1. Hash file_path with MD5
  2. Returns: `{cache_dir}/{step}_{hash}.pkl`

**`_get_md_cache_path(file_path: str) -> str`**
- **Purpose**: Generate markdown cache path
- **I/O Flow**:
  1. Hash file_path with MD5
  2. Extract source filename stem
  3. Returns: `{md_cache_dir}/{stem}_{hash}.md`

**`_discover_files() -> List[str]`**
- **Purpose**: Recursively find all supported files
- **I/O Flow**:
  1. Walk directory tree: `os.walk(config.input_dir)`
  2. Filter by `config.supported_formats`
  3. Returns absolute paths

**`process() -> DocParserOutput`** (Main Pipeline - ASYNC)
- **Purpose**: Execute full document processing pipeline
- **I/O Flow**:

  **Step 1: File Discovery**
  - Call `_discover_files()`
  - Returns list of file paths

  **Step 2: Convert to Markdown (with caching)**
  - For each file:
    - Check markdown cache: `_get_md_cache_path(file_path)`
    - If exists: load from cache
    - Else: 
      - `converter.convert_to_markdown(file_path)`
      - Save to cache
    - Store: `(file_path, md_content)`

  **Step 3: Chunk Markdown (with caching)**
  - Cache key includes LLM status: `chunks{_llm if LLM enabled}`
  - For each file:
    - Check chunk cache: `_get_cache_path(f"chunks{llm_suffix}", file_path)`
    - If exists: `pickle.load(chunks)`
    - Else:
      - If LLM enabled: `await chunker.chunk_markdown_async(...)`
      - Else: `chunker.chunk_markdown(...)`
      - Adjust chunk indices globally
      - `pickle.dump(chunks)`
    - Accumulate all chunks

  **Step 4: Consolidate Markdown**
  - For each file:
    - `consolidator.append_document(md_content, file_path)`
  - `consolidated_path = consolidator.finalize()`

  **Step 5: Generate Embeddings (with caching)**
  - Determine: `include_summary = LLM enabled && embed_with_summary`
  - Cache key: `embedding{_with_summary if include_summary}`
  - Try to load all cached embeddings:
    - For each chunk: `_get_cache_path(f"embedding{suffix}_{i}", chunk.id)`
    - If all cached: load from pickle
  - If not all cached:
    - `await embedder.generate_embeddings(chunks, include_summary=...)`
    - Cache each embedding individually
    - `pickle.dump(embedding)`

  **Step 6: Store in ChromaDB**
  - `chroma.add_chunks(all_chunks, embeddings)`

  **Step 7: Cleanup Cache**
  - If `config.cleanup_cache=True`:
    - `shutil.rmtree(cache_dir)`
  - Else: preserve cache

  **Step 8: Return Output**
  - Create `DocParserOutput`:
    - `consolidated_md_path`
    - `chroma_collection_name`
    - `chunk_metadata` (list of chunk dicts)
    - `total_chunks`
    - `processing_log`
  - If LLM enabled: add metadata with linking stats
  - Returns output

---

## Key ChromaDB Patterns

### Storage Pattern
```python
# Initialize
client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection(
    name=name,
    metadata={"hnsw:space": "cosine"}  # similarity metric
)

# Add documents
collection.add(
    ids=[...],              # unique IDs
    documents=[...],        # text content
    metadatas=[...],        # JSON-serializable metadata dicts
    embeddings=[...]        # lists of floats
)
```

### Retrieval Pattern
```python
# Query by vector
results = collection.query(
    query_embeddings=[vector.tolist()],
    n_results=5
)
# Returns: {ids, documents, metadatas, distances}

# Get by ID
result = collection.get(
    ids=[chunk_id],
    include=["documents", "metadatas"]
)
# Returns: {ids, documents, metadatas}
```

### Metadata Serialization
- Lists/Dicts → `json.dumps()` before storing
- Retrieve → `json.loads()` to parse back
- All metadata values must be JSON-serializable
