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


class EmbeddingGenerator:
    """Generates embeddings using an OpenAI-compatible endpoint with LLM linking support."""

    def __init__(self, endpoint: str, model: str):
        """Initialize OpenAI-compatible client."""
        self.client = OpenAI(base_url=endpoint, api_key="dummy")
        self.model = model

    async def generate_embeddings(
        self, chunks: List, batch_size: int = 32, include_summary: bool = False
    ) -> List[np.ndarray]:
        """
        Generate embeddings with batching and optional summary inclusion.

        Process:
        1. Batch chunks
        2. Optionally augment text with summary points and linking info
        3. Send async requests to endpoint
        4. Handle retries with exponential backoff
        5. Return embeddings in same order

        Args:
            chunks: List of Chunk objects
            batch_size: Number of chunks to process per batch
            include_summary: Whether to include summary points in embedding text
        """
        embeddings = []

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            if include_summary:
                batch_texts = [
                    self._prepare_text_with_summary(chunk) for chunk in batch
                ]
            else:
                batch_texts = [chunk.content for chunk in batch]

            try:
                response = self.client.embeddings.create(
                    input=batch_texts, model=self.model
                )

                # Extract embeddings from response
                batch_embeddings = [np.array(data.embedding) for data in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                raise RuntimeError(f"Failed to generate embeddings for batch: {e}")

        return embeddings

    def _prepare_text_with_summary(self, chunk) -> str:
        """
        Prepare chunk text augmented with summary points and linking information.

        This creates a richer representation that includes:
        - Original content
        - Summary points
        - Context from previous/next chunk relationships
        """
        parts = [chunk.content]

        if hasattr(chunk, "summary_points") and chunk.summary_points:
            # Add summary points
            summary_text = "\n\nKey Points:\n"
            for i, sp in enumerate(chunk.summary_points, 1):
                summary_text += f"{i}. {sp.text}\n"

                # Add context from linking
                if sp.prev_link:
                    summary_text += f"   (Context from previous: {sp.prev_link['relation']} regarding {sp.prev_link['common_topic']})\n"
                if sp.next_link:
                    summary_text += f"   (Leads to: {sp.next_link['relation']} regarding {sp.next_link['common_topic']})\n"

            parts.append(summary_text)

        return "\n".join(parts)

    async def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Single batch embedding with error handling."""
        try:
            response = self.client.embeddings.create(input=texts, model=self.model)
            return [np.array(data.embedding) for data in response.data]
        except Exception as e:
            raise RuntimeError(f"Failed to embed batch: {e}")


class ChromaDBManager:
    """Manages ChromaDB storage and retrieval with LLM linking support."""

    def __init__(self, db_path: str, collection_name: str):
        """Initialize persistent ChromaDB client."""
        # Ensure the directory exists
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List, embeddings: List[np.ndarray]):
        """
        Add chunks with embeddings to ChromaDB, including LLM linking metadata.

        Metadata stored per chunk:
        - source_file
        - chunk_index
        - start_char, end_char
        - headers (as JSON string)
        - summary_points (as JSON string with linking info)
        - timestamp
        """
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = []

        for chunk in chunks:
            # Convert metadata values to acceptable types for ChromaDB
            metadata = {
                "source_file": chunk.source_file,
                "chunk_index": str(chunk.chunk_index),
                "start_char": str(chunk.start_char),
                "end_char": str(chunk.end_char),
                "headers": json.dumps(chunk.headers),
                "timestamp": str(time.time()),
            }

            # Add summary points if available
            if hasattr(chunk, "summary_points") and chunk.summary_points:
                summary_data = [
                    {
                        "text": sp.text,
                        "prev_link": sp.prev_link,
                        "next_link": sp.next_link,
                    }
                    for sp in chunk.summary_points
                ]
                metadata["summary_points"] = json.dumps(summary_data)

            # Add additional metadata, converting values as needed
            for key, value in chunk.metadata.items():
                if isinstance(value, list):
                    metadata[key] = json.dumps(value)
                elif isinstance(value, dict):
                    metadata[key] = json.dumps(value)
                elif isinstance(value, (int, float, str, bool)) or value is None:
                    metadata[key] = value
                else:
                    metadata[key] = str(value)

            metadatas.append(metadata)

        embeddings_list = [emb.tolist() for emb in embeddings]

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings_list,
        )

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        include_context: bool = True,
    ) -> List[Dict]:
        """
        Query vector DB, return chunks with metadata and optional context.

        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            include_context: Whether to parse and include summary/linking info
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=n_results
        )

        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]

            # Parse summary points if available and context is requested
            summary_points = None
            if include_context and "summary_points" in metadata:
                try:
                    summary_points = json.loads(metadata["summary_points"])
                except:
                    pass

            result = {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": metadata,
                "summary_points": summary_points,
                "distance": (
                    results["distances"][0][i] if results["distances"] else None
                ),
            }
            formatted_results.append(result)

        return formatted_results

    def get_chunk_with_context(self, chunk_id: str) -> Optional[Dict]:
        """
        Retrieve a chunk and its linked previous/next chunks based on linking metadata.

        Returns a dict with:
        - current: The requested chunk
        - previous: Previous chunk info (if linked)
        - next: Next chunk info (if linked)
        """
        # Get the current chunk
        result = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])

        if not result["ids"]:
            return None

        current_metadata = result["metadatas"][0]
        current_doc = result["documents"][0]

        # Parse summary points to find links
        context = {
            "current": {
                "id": chunk_id,
                "content": current_doc,
                "metadata": current_metadata,
            },
            "previous": None,
            "next": None,
        }

        if "summary_points" in current_metadata:
            try:
                summary_points = json.loads(current_metadata["summary_points"])

                # Extract common topics from prev/next links
                prev_topics = set()
                next_topics = set()

                for sp in summary_points:
                    if sp.get("prev_link"):
                        prev_topics.add(sp["prev_link"].get("common_topic", ""))
                    if sp.get("next_link"):
                        next_topics.add(sp["next_link"].get("common_topic", ""))

                context["prev_topics"] = list(prev_topics)
                context["next_topics"] = list(next_topics)
            except:
                pass

        return context
