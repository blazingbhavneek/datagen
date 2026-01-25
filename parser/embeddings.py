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
from parser.chunks import Chunk
from parser.configs import DocParserConfig
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
    """Generates embeddings using an OpenAI-compatible endpoint."""

    def __init__(self, endpoint: str, model: str):
        """Initialize OpenAI-compatible client [[31]]."""
        self.client = OpenAI(base_url=endpoint, api_key="dummy")
        self.model = model

    async def generate_embeddings(
        self, chunks: List[Chunk], batch_size: int = 32
    ) -> List[np.ndarray]:
        """
        Generate embeddings with batching

        Process:
        1. Batch chunks
        2. Send async requests to endpoint
        3. Handle retries with exponential backoff
        4. Return embeddings in same order
        """
        embeddings = []

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
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

    async def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Single batch embedding with error handling."""
        try:
            response = self.client.embeddings.create(input=texts, model=self.model)
            return [np.array(data.embedding) for data in response.data]
        except Exception as e:
            raise RuntimeError(f"Failed to embed batch: {e}")


class ChromaDBManager:
    """Manages ChromaDB storage and retrieval."""

    def __init__(self, db_path: str, collection_name: str):
        """Initialize persistent ChromaDB client [[22]]."""
        # Ensure the directory exists
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Chunk], embeddings: List[np.ndarray]):
        """
        Add chunks with embeddings to ChromaDB

        Metadata stored per chunk:
        - source_file
        - chunk_index
        - start_char, end_char
        - headers (as JSON string)
        - timestamp
        """
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = []

        for chunk in chunks:
            # Convert metadata values to acceptable types for ChromaDB
            metadata = {
                "source_file": chunk.source_file,
                "chunk_index": str(chunk.chunk_index),  # Convert to string
                "start_char": str(chunk.start_char),  # Convert to string
                "end_char": str(chunk.end_char),  # Convert to string
                "headers": json.dumps(chunk.headers),
                "timestamp": str(time.time()),  # Convert to string
            }
            # Add additional metadata, converting values as needed
            for key, value in chunk.metadata.items():
                if isinstance(value, list):
                    # Convert list to JSON string
                    metadata[key] = json.dumps(value)
                elif isinstance(value, dict):
                    # Convert dict to JSON string
                    metadata[key] = json.dumps(value)
                elif isinstance(value, (int, float, str, bool)) or value is None:
                    # Acceptable types for ChromaDB
                    metadata[key] = value
                else:
                    # Convert other types to string
                    metadata[key] = str(value)

            metadatas.append(metadata)

        embeddings_list = [
            emb.tolist() for emb in embeddings
        ]  # Convert numpy arrays to lists

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings_list,
        )

    def query(self, query_embedding: np.ndarray, n_results: int = 5) -> List[Dict]:
        """Query vector DB, return chunks with metadata."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=n_results
        )

        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            result = {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": (
                    results["distances"][0][i] if results["distances"] else None
                ),
            }
            formatted_results.append(result)

        return formatted_results
