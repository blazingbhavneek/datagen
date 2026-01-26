import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# Pydantic models for structured outputs
class SummaryPointsResponse(BaseModel):
    """Response model for summary points generation."""
    points: List[str] = Field(description="List of 3-5 key summary points from the text")


class LinkInfo(BaseModel):
    """Information about how a summary point links to another chunk."""
    relates: bool = Field(description="Whether this summary point relates to the other chunk")
    relation: Optional[str] = Field(default=None, description="Type of relation (e.g., 'continues discussion', 'provides example')")
    common_topic: Optional[str] = Field(default=None, description="Common topic connecting the chunks")


class ChunkLinksResponse(BaseModel):
    """Response model for chunk linking analysis."""
    links: List[LinkInfo] = Field(description="Link information for each summary point")


@dataclass
class SummaryPoint:
    """Represents a summary point with linking information."""
    text: str
    prev_link: Optional[Dict[str, str]] = None  # {"relation": "...", "common_topic": "..."}
    next_link: Optional[Dict[str, str]] = None  # {"relation": "...", "common_topic": "..."}


class Chunk:
    """Represents a text chunk with metadata and optional LLM-based linking."""

    def __init__(
        self,
        content: str,
        source_file: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
        headers: List[str] = None,
        metadata: Dict = None,
        summary_points: List[SummaryPoint] = None,
    ):
        self.id = str(uuid.uuid4())
        self.content = content
        self.source_file = source_file
        self.chunk_index = chunk_index
        self.start_char = start_char
        self.end_char = end_char
        self.headers = headers or []
        self.metadata = metadata or {}
        self.summary_points = summary_points or []

    def to_dict(self) -> Dict:
        """Convert chunk to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "headers": self.headers,
            "metadata": self.metadata,
            "summary_points": [
                {
                    "text": sp.text,
                    "prev_link": sp.prev_link,
                    "next_link": sp.next_link,
                }
                for sp in self.summary_points
            ],
        }


class SemanticChunker:
    """Chunks markdown content preserving structure with optional LLM-based linking."""

    def __init__(
        self,
        chunk_size: int,
        overlap: int,
        enable_llm_linking: bool = False,
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_model: str = "gpt-4",
        llm_concurrency: int = 20,
        llm_temperature: float = 0.3,
    ):
        """
        Initialize with size constraints and optional LLM linking.
        
        Args:
            chunk_size: Maximum size of each chunk
            overlap: Overlap between chunks
            enable_llm_linking: Whether to enable LLM-based chunk linking
            llm_api_key: API key for LLM service
            llm_base_url: Base URL for LLM service (optional, for custom endpoints)
            llm_model: Model to use for LLM operations
            llm_concurrency: Maximum concurrent LLM API calls
            llm_temperature: Temperature for LLM operations
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.enable_llm_linking = enable_llm_linking
        self.llm_concurrency = llm_concurrency
        
        # Create semaphore for LLM API rate limiting
        self.llm_semaphore = asyncio.Semaphore(llm_concurrency) if enable_llm_linking else None
        
        if enable_llm_linking:
            if not llm_api_key:
                raise ValueError("llm_api_key must be provided when enable_llm_linking is True")
            
            # Initialize LangChain ChatOpenAI
            llm_kwargs = {
                "model": llm_model,
                "api_key": llm_api_key,
                "temperature": llm_temperature,
            }
            if llm_base_url:
                llm_kwargs["base_url"] = llm_base_url
            
            self.llm = ChatOpenAI(**llm_kwargs)
            
            # Create structured output chains
            self.summary_chain = self.llm.with_structured_output(SummaryPointsResponse)
            self.linking_chain = self.llm.with_structured_output(ChunkLinksResponse)

    def chunk_markdown(self, md_content: str, source_file: str) -> List[Chunk]:
        """
        Chunk markdown preserving structure with optional LLM linking.
        Sync version - creates event loop if LLM linking is needed.
        """
        chunks = self._do_basic_chunking(md_content, source_file)

        # Apply LLM-based linking if enabled
        if self.enable_llm_linking and chunks:
            # Run async linking in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                chunks = loop.run_until_complete(self._apply_llm_linking_async(chunks))
            finally:
                loop.close()
                asyncio.set_event_loop(None)

        return chunks
    
    async def chunk_markdown_async(self, md_content: str, source_file: str) -> List[Chunk]:
        """Async version of chunk_markdown for use in async contexts."""
        chunks = self._do_basic_chunking(md_content, source_file)

        # Apply LLM-based linking if enabled
        if self.enable_llm_linking and chunks:
            chunks = await self._apply_llm_linking_async(chunks)

        return chunks

    def _do_basic_chunking(self, md_content: str, source_file: str) -> List[Chunk]:
        """
        Perform basic markdown chunking without LLM linking.
        
        Strategy:
        1. Split on markdown headers first (##, ###, etc.)
        2. If section > chunk_size, split on paragraphs
        3. If paragraph > chunk_size, split on sentences
        4. Maintain overlap with previous chunk
        """
        import re
        
        chunks = []
        chunk_index = 0

        # First, split by main headers
        header_pattern = r"^(#{1,6})\s+(.*)$"
        lines = md_content.split("\n")

        sections = []  # List of (start_idx, end_idx, headers) tuples
        current_headers = []
        section_start = 0

        for i, line in enumerate(lines):
            match = re.match(header_pattern, line.strip())
            if match:
                # Found a header
                header_level = len(match.group(1))  # Number of # symbols
                header_text = match.group(2)

                # Update headers list based on hierarchy
                current_headers = current_headers[: header_level - 1]  # Keep parent headers
                current_headers.append((header_level, header_text))

                # If we have content since last section, save it
                if i > section_start:
                    sections.append((section_start, i, current_headers.copy()))

                # Start new section after header line
                section_start = i + 1

        # Add the final section
        if section_start < len(lines):
            sections.append((section_start, len(lines), current_headers.copy()))

        # Process each section
        for start_idx, end_idx, headers in sections:
            section_text = "\n".join(lines[start_idx:end_idx])
            section_chunks = self._chunk_section(
                section_text, source_file, headers, chunk_index
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        return chunks

    def _chunk_section(
        self,
        section_text: str,
        source_file: str,
        headers: List[Tuple[int, str]],
        base_chunk_index: int,
    ) -> List[Chunk]:
        """Chunk a single section of markdown."""
        import re
        
        chunks = []
        chunk_index = base_chunk_index

        # If the section is small enough, keep as one chunk
        if len(section_text) <= self.chunk_size:
            chunk = Chunk(
                content=section_text,
                source_file=source_file,
                chunk_index=chunk_index,
                start_char=0,
                end_char=len(section_text),
                headers=[h[1] for h in headers],
                metadata={"header_levels": [h[0] for h in headers]},
            )
            chunks.append(chunk)
            return chunks

        # Otherwise, break down further
        paragraphs = section_text.split("\n\n")
        current_chunk = ""
        current_headers = [h[1] for h in headers]

        for para in paragraphs:
            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # Current chunk is full, save it
                if current_chunk.strip():
                    chunk = Chunk(
                        content=current_chunk,
                        source_file=source_file,
                        chunk_index=chunk_index,
                        start_char=0,
                        end_char=0,
                        headers=current_headers,
                        metadata={"header_levels": [str(h[0]) for h in headers]},
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Start new chunk
                if len(para) > self.chunk_size:
                    subchunks = self._split_large_paragraph(
                        para, source_file, current_headers, chunk_index
                    )
                    chunks.extend(subchunks)
                    chunk_index += len(subchunks)
                    current_chunk = ""
                else:
                    current_chunk = para

        # Add the final chunk if there's content left
        if current_chunk.strip():
            chunk = Chunk(
                content=current_chunk,
                source_file=source_file,
                chunk_index=chunk_index,
                start_char=0,
                end_char=0,
                headers=current_headers,
                metadata={"header_levels": [str(h[0]) for h in headers]},
            )
            chunks.append(chunk)

        return chunks

    def _split_large_paragraph(
        self,
        paragraph: str,
        source_file: str,
        headers: List[str],
        base_chunk_index: int,
    ) -> List[Chunk]:
        """Split a large paragraph into smaller chunks."""
        import re
        
        chunks = []
        chunk_index = base_chunk_index

        # Split by sentences
        sentences = re.split(r"[.!?]+", paragraph)
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) <= self.chunk_size:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
            else:
                # Current chunk is full
                if current_chunk.strip():
                    chunk = Chunk(
                        content=current_chunk + ".",
                        source_file=source_file,
                        chunk_index=chunk_index,
                        start_char=0,
                        end_char=0,
                        headers=headers,
                        metadata={"header_levels": [], "part_of_large_para": True},
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Start new chunk
                if len(sentence) > self.chunk_size:
                    # Sentence is too long, just cut it
                    parts = [
                        sentence[i : i + self.chunk_size]
                        for i in range(0, len(sentence), self.chunk_size)
                    ]
                    for part in parts[:-1]:
                        chunk = Chunk(
                            content=part,
                            source_file=source_file,
                            chunk_index=chunk_index,
                            start_char=0,
                            end_char=0,
                            headers=headers,
                            metadata={
                                "header_levels": [],
                                "part_of_large_para": True,
                                "truncated": True,
                            },
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                    current_chunk = parts[-1]
                else:
                    current_chunk = sentence

        # Add final chunk
        if current_chunk.strip():
            chunk = Chunk(
                content=current_chunk + ("" if current_chunk.endswith((".", "!", "?")) else "."),
                source_file=source_file,
                chunk_index=chunk_index,
                start_char=0,
                end_char=0,
                headers=headers,
                metadata={"header_levels": [], "part_of_large_para": True},
            )
            chunks.append(chunk)

        return chunks

    async def _apply_llm_linking_async(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Apply LLM-based linking to chunks with concurrent processing.
        
        Process:
        1. Generate summaries for all chunks (concurrent)
        2. Link each chunk with previous chunk (concurrent)
        3. Link each chunk with next chunk (concurrent)
        """
        # Step 1: Generate summaries for all chunks (concurrent)
        logging.info(f"Generating summaries for {len(chunks)} chunks with concurrency={self.llm_concurrency}...")
        summary_tasks = [self._generate_summary_async(chunk.content) for chunk in chunks]
        summaries = await asyncio.gather(*summary_tasks)
        
        # Assign summaries to chunks
        for chunk, summary_points in zip(chunks, summaries):
            chunk.summary_points = summary_points
        
        logging.info(f"Generated summaries for {len(chunks)} chunks")

        # Step 2: Link chunks with previous (concurrent)
        logging.info("Linking chunks with previous...")
        prev_link_tasks = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_link_tasks.append(self._link_with_previous_async(chunks[i - 1], chunk))
        
        if prev_link_tasks:
            await asyncio.gather(*prev_link_tasks)
        
        logging.info(f"Completed {len(prev_link_tasks)} previous links")

        # Step 3: Link chunks with next (concurrent)
        logging.info("Linking chunks with next...")
        next_link_tasks = []
        for i, chunk in enumerate(chunks):
            if i < len(chunks) - 1:
                next_link_tasks.append(self._link_with_next_async(chunk, chunks[i + 1]))
        
        if next_link_tasks:
            await asyncio.gather(*next_link_tasks)
        
        logging.info(f"Completed {len(next_link_tasks)} next links")

        return chunks

    async def _generate_summary_async(self, content: str) -> List[SummaryPoint]:
        """Generate summary points for a chunk using LLM with concurrency control."""
        async with self.llm_semaphore:
            prompt = f"""Analyze the following text chunk and extract 3-5 key summary points.
Each point should be a concise sentence describing a main topic or idea.

Text:
{content}"""

            try:
                # Use ainvoke for async operation
                response = await self.summary_chain.ainvoke(prompt)
                return [SummaryPoint(text=point) for point in response.points]
            except Exception as e:
                logging.error(f"Error generating summary: {e}")
                return [SummaryPoint(text="Summary generation failed")]

    async def _link_with_previous_async(self, prev_chunk: Chunk, curr_chunk: Chunk):
        """Link current chunk's summary points with previous chunk (async with concurrency control)."""
        if not curr_chunk.summary_points:
            return

        async with self.llm_semaphore:
            summary_texts = [sp.text for sp in curr_chunk.summary_points]
            prompt = f"""Analyze how the current chunk relates to the previous chunk.

Previous Chunk:
{prev_chunk.content}

Current Chunk:
{curr_chunk.content}

Current Summary Points:
{chr(10).join(f"{i+1}. {text}" for i, text in enumerate(summary_texts))}

For each summary point in the current chunk, determine:
1. Does it relate to the previous chunk? (yes/no)
2. If yes, what is the relation? (e.g., "continues discussion", "provides example", "contrasts with")
3. If yes, what common topic connects them?"""

            try:
                response = await self.linking_chain.ainvoke(prompt)

                for i, link_info in enumerate(response.links):
                    if i < len(curr_chunk.summary_points) and link_info.relates:
                        for i, link_info in enumerate(response.links):
                            if i < len(curr_chunk.summary_points) and link_info.relates:
                                curr_chunk.summary_points[i].prev_link = {
                                    "chunk_id": prev_chunk.id,
                                    "chunk_index": prev_chunk.chunk_index,
                                    "relation": link_info.relation or "",
                                    "common_topic": link_info.common_topic or "",
                                }
            except Exception as e:
                logging.error(f"Error linking with previous chunk: {e}")

    async def _link_with_next_async(self, curr_chunk: Chunk, next_chunk: Chunk):
        """Link current chunk's summary points with next chunk (async with concurrency control)."""
        if not curr_chunk.summary_points:
            return

        async with self.llm_semaphore:
            summary_texts = [sp.text for sp in curr_chunk.summary_points]
            prompt = f"""Analyze how the current chunk relates to the next chunk.

Current Chunk:
{curr_chunk.content}

Next Chunk:
{next_chunk.content}

Current Summary Points:
{chr(10).join(f"{i+1}. {text}" for i, text in enumerate(summary_texts))}

For each summary point in the current chunk, determine:
1. Does it relate to the next chunk? (yes/no)
2. If yes, what is the relation? (e.g., "leads into", "is elaborated in", "sets up")
3. If yes, what common topic connects them?"""

            try:
                response = await self.linking_chain.ainvoke(prompt)

                for i, link_info in enumerate(response.links):
                    if i < len(curr_chunk.summary_points) and link_info.relates:
                        curr_chunk.summary_points[i].next_link = {
                            "chunk_id": next_chunk.id,
                            "chunk_index": next_chunk.chunk_index,
                            "relation": link_info.relation or "",
                            "common_topic": link_info.common_topic or "",
                        }
            except Exception as e:
                logging.error(f"Error linking with next chunk: {e}")
