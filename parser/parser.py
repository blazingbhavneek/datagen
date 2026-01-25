import asyncio
import json
import logging
import os
import pickle
import re
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from parser.chunks import Chunk, SemanticChunker
from parser.configs import DocParserConfig, DocParserOutput
from parser.converter import DocumentConverter
from parser.embeddings import ChromaDBManager, EmbeddingGenerator
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
import docx
import html2text
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from chromadb.config import Settings
from openai import OpenAI
from pptx import Presentation
from tqdm.asyncio import tqdm

from utils.logger import setup_logger


class MarkdownConsolidator:
    """Manages the consolidation of multiple markdown documents."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.file_separator = "\n\n" + "=" * 80 + "\n\n"
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def append_document(self, md_content: str, source_file: str):
        """
        Append markdown with metadata header

        Header format:
        # SOURCE: {source_file}
        # TIMESTAMP: {iso_timestamp}
        # FORMAT: {original_format}
        ---
        {content}
        """
        source_path = Path(source_file)
        header = (
            f"# SOURCE: {source_file}\n"
            f"# TIMESTAMP: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
            f"# FORMAT: {source_path.suffix}\n---\n"
        )

        with open(self.output_path, "a", encoding="utf-8") as f:
            # Add separator if file already has content
            if f.tell() > 0:
                f.write(self.file_separator)

            f.write(header)
            f.write(md_content)

    def finalize(self) -> str:
        """Return path to consolidated file."""
        return self.output_path


class DocumentParser:
    """Main pipeline for document parsing with optional LLM-based linking."""

    def __init__(self, config: DocParserConfig):
        self.config = config
        self.converter = DocumentConverter(config)
        self.consolidator = MarkdownConsolidator(config.output_md_path)
        
        # Initialize chunker with optional LLM linking
        chunker_kwargs = {
            'chunk_size': config.chunk_size,
            'overlap': config.chunk_overlap,
            'enable_llm_linking': getattr(config, 'enable_llm_linking', False),
        }
        
        if chunker_kwargs['enable_llm_linking']:
            chunker_kwargs.update({
                'llm_api_key': getattr(config, 'llm_api_key', os.getenv('OPENAI_API_KEY')),
                'llm_base_url': getattr(config, 'llm_base_url', None),
                'llm_model': getattr(config, 'llm_model', 'gpt-4'),
                'llm_concurrency': getattr(config, 'llm_concurrency', 20),
                'llm_temperature': getattr(config, 'llm_temperature', 0.3),
            })
        
        self.chunker = SemanticChunker(**chunker_kwargs)
        
        self.embedder = EmbeddingGenerator(
            config.embedding_endpoint, config.embedding_model
        )
        self.chroma = ChromaDBManager(config.chroma_db_path, f"docs_{int(time.time())}")
        self.logger = setup_logger("doc_parser")
        
        # Create cache directory in output directory
        output_dir = Path(config.output_md_path).parent
        self.cache_dir = output_dir / ".doc_parser_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create markdown cache subdirectory
        self.md_cache_dir = self.cache_dir / "markdown"
        self.md_cache_dir.mkdir(exist_ok=True)

    def _get_cache_path(self, step: str, file_path: str) -> str:
        """Generate a cache file path for a specific step and file."""
        import hashlib

        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{step}_{file_hash}.pkl")
    
    def _get_md_cache_path(self, file_path: str) -> str:
        """Generate a markdown cache file path."""
        import hashlib
        
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        source_name = Path(file_path).stem
        return os.path.join(self.md_cache_dir, f"{source_name}_{file_hash}.md")

    async def process(self) -> DocParserOutput:
        """
        Main processing pipeline with caching and optional LLM linking

        Steps:
        1. Discover all supported files in input_dir (recursive)
        2. Convert each file to markdown (parallel processing) - CACHED
        3. Chunk the markdown content - CACHED (including LLM linking if enabled)
        4. Consolidate all markdown files
        5. Generate embeddings for all chunks (batched, optionally with summary context) - CACHED
        6. Store in ChromaDB
        7. Clean up temp directory
        8. Save metadata and return output
        """

        # Step 1: File discovery
        files = self._discover_files()
        self.logger.info(f"Found {len(files)} files to process")

        if not files:
            raise ValueError(f"No supported files found in {self.config.input_dir}")

        # Step 2: Convert files to markdown with caching
        converted_files = []
        for file_path in files:
            md_cache_path = self._get_md_cache_path(file_path)
            
            # Check if markdown cache exists
            if os.path.exists(md_cache_path):
                self.logger.debug(f"Loading cached markdown for {file_path}")
                with open(md_cache_path, "r", encoding="utf-8") as f:
                    md_content = f.read()
            else:
                self.logger.info(f"Converting {file_path} to markdown")
                md_content = self.converter.convert_to_markdown(file_path)
                # Save to markdown cache
                with open(md_cache_path, "w", encoding="utf-8") as f:
                    f.write(md_content)
            
            converted_files.append((file_path, md_content))

        # Step 3: Chunk the markdown content with caching
        # Note: Cache key includes LLM linking status to avoid stale cache
        all_chunks = []
        chunk_index_offset = 0
        
        llm_suffix = "_llm" if getattr(self.config, 'enable_llm_linking', False) else ""
        
        for file_path, md_content in converted_files:
            cache_path = self._get_cache_path(f"chunks{llm_suffix}", file_path)
            if os.path.exists(cache_path):
                self.logger.debug(f"Loading cached chunks for {file_path}")
                with open(cache_path, "rb") as f:
                    chunks = pickle.load(f)
            else:
                self.logger.info(f"Chunking {file_path}" + 
                               (" with LLM linking" if llm_suffix else ""))
                
                # Use async chunking if LLM linking is enabled
                if getattr(self.config, 'enable_llm_linking', False):
                    chunks = await self.chunker.chunk_markdown_async(md_content, file_path)
                else:
                    chunks = self.chunker.chunk_markdown(md_content, file_path)
                
                # Adjust chunk indices to be globally unique
                for chunk in chunks:
                    chunk.chunk_index += chunk_index_offset
                chunk_index_offset += len(chunks)
                with open(cache_path, "wb") as f:
                    pickle.dump(chunks, f)
            all_chunks.extend(chunks)

        # Step 4: Consolidate all markdown files
        for file_path, md_content in converted_files:
            self.consolidator.append_document(md_content, file_path)
        consolidated_path = self.consolidator.finalize()
        self.logger.info(f"Created consolidated markdown: {consolidated_path}")

        # Step 5: Generate embeddings with caching
        if all_chunks:
            self.logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
            
            # Determine if we should include summary context in embeddings
            include_summary = (
                getattr(self.config, 'enable_llm_linking', False) and
                getattr(self.config, 'embed_with_summary', True)
            )

            # Cache key includes summary status
            embedding_suffix = "_with_summary" if include_summary else ""

            # Load cached embeddings if available
            embeddings = []
            all_cached = True
            for i, chunk in enumerate(all_chunks):
                cache_path = self._get_cache_path(
                    f"embedding{embedding_suffix}_{i}", chunk.id
                )
                if os.path.exists(cache_path):
                    with open(cache_path, "rb") as f:
                        embedding = pickle.load(f)
                    embeddings.append(embedding)
                else:
                    # Mark that not all embeddings are cached
                    all_cached = False
                    break

            # If not all embeddings were cached, generate them
            if not all_cached:
                self.logger.info(
                    f"Generating embeddings" + 
                    (" with summary context" if include_summary else "")
                )
                embeddings = await self.embedder.generate_embeddings(
                    all_chunks,
                    include_summary=include_summary
                )
                # Cache each embedding individually
                for i, embedding in enumerate(embeddings):
                    cache_path = self._get_cache_path(
                        f"embedding{embedding_suffix}_{i}", all_chunks[i].id
                    )
                    with open(cache_path, "wb") as f:
                        pickle.dump(embedding, f)
            else:
                self.logger.info(f"Loaded {len(embeddings)} cached embeddings")

            # Step 6: Store in ChromaDB
            self.logger.info("Storing in ChromaDB")
            self.chroma.add_chunks(all_chunks, embeddings)
        else:
            self.logger.warning(
                "No chunks were created, skipping embedding and ChromaDB steps"
            )

        # Step 7: Clean up cache directory
        if getattr(self.config, 'cleanup_cache', True):
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.logger.info("Cleaned up cache directory")
        else:
            self.logger.info(f"Cache directory preserved at: {self.cache_dir}")

        # Step 8: Return output
        output = DocParserOutput(
            consolidated_md_path=consolidated_path,
            chroma_collection_name=self.chroma.collection.name,
            chunk_metadata=[chunk.to_dict() for chunk in all_chunks],
            total_chunks=len(all_chunks),
            processing_log=self.logger.handlers[0].baseFilename if self.logger.handlers else None,
        )
        
        # Add LLM linking stats if enabled
        if getattr(self.config, 'enable_llm_linking', False):
            linked_chunks = sum(
                1 for chunk in all_chunks 
                if hasattr(chunk, 'summary_points') and chunk.summary_points
            )
            output.metadata = output.metadata or {}
            output.metadata['llm_linking_enabled'] = True
            output.metadata['chunks_with_summaries'] = linked_chunks
            self.logger.info(f"LLM linking: {linked_chunks}/{len(all_chunks)} chunks have summaries")

        return output

    def _discover_files(self) -> List[str]:
        """Recursively find all supported files."""
        files = []
        for root, _, filenames in os.walk(self.config.input_dir):
            for filename in filenames:
                if Path(filename).suffix.lower() in self.config.supported_formats:
                    files.append(os.path.join(root, filename))
        return files


async def retry_with_backoff(func, max_retries=3, base_delay=1.0):
    """Utility for retrying failed operations."""
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            delay = base_delay * (2**attempt)  # Exponential backoff
            await asyncio.sleep(delay)
