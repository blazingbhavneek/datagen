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


class Chunk:
    """Represents a text chunk with metadata."""

    def __init__(
        self,
        content: str,
        source_file: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
        headers: List[str] = None,
        metadata: Dict = None,
    ):
        self.id = str(uuid.uuid4())
        self.content = content
        self.source_file = source_file
        self.chunk_index = chunk_index
        self.start_char = start_char
        self.end_char = end_char
        self.headers = headers or []
        self.metadata = metadata or {}

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
        }


class SemanticChunker:
    """Chunks markdown content preserving structure."""

    def __init__(self, chunk_size: int, overlap: int):
        """Initialize with size constraints."""
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_markdown(self, md_content: str, source_file: str) -> List[Chunk]:
        """
        Chunk markdown preserving structure

        Strategy:
        1. Split on markdown headers first (##, ###, etc.)
        2. If section > chunk_size, split on paragraphs
        3. If paragraph > chunk_size, split on sentences
        4. Maintain overlap with previous chunk

        Returns:
            List of Chunk objects with metadata
        """
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
                current_headers = current_headers[
                    : header_level - 1
                ]  # Keep parent headers
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
        # First try splitting by paragraphs
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
                        start_char=0,  # Would need to track actual positions
                        end_char=0,
                        headers=current_headers,
                        metadata={
                            "header_levels": [str(h[0]) for h in headers]
                        },  # Fixed: Convert to string
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Start new chunk with overlap
                if len(para) > self.chunk_size:
                    # Paragraph itself is too large, need to split by sentences
                    subchunks = self._split_large_paragraph(
                        para, source_file, current_headers, chunk_index
                    )
                    chunks.extend(subchunks)
                    chunk_index += len(subchunks)
                    current_chunk = ""  # After splitting large paragraph, reset
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
                metadata={
                    "header_levels": [str(h[0]) for h in headers]
                },  # Fixed: Convert to string
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

                    current_chunk = parts[-1]  # Last part becomes new chunk
                else:
                    current_chunk = sentence

        # Add final chunk
        if current_chunk.strip():
            chunk = Chunk(
                content=current_chunk
                + ("" if current_chunk.endswith((".", "!", "?")) else "."),
                source_file=source_file,
                chunk_index=chunk_index,
                start_char=0,
                end_char=0,
                headers=headers,
                metadata={"header_levels": [], "part_of_large_para": True},
            )
            chunks.append(chunk)

        return chunks
