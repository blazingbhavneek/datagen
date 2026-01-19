# qa_generator/utils/chunk_manager.py
import pickle
from typing import Any, Dict, List

import chromadb
import networkx as nx
import numpy as np
from pydantic import BaseModel
from qa.config import QAGeneratorConfig


class Chunk(BaseModel):
    id: str
    content: str
    chunk_type: str  # "doc", "function", "class", "file"
    source_file: str
    metadata: Dict[str, Any]
    embedding: (
        Any  # Using Any since numpy arrays aren't directly supported in Pydantic v2
    )


class ChunkManager:
    """Manages data chunks from parsed sources"""

    def __init__(self, config: QAGeneratorConfig):
        self.config = config
        self.data_source_type = config.data_source_type
        self.chunks: List[Chunk] = []

    def load_chunks(self) -> List[Chunk]:
        """Load chunks based on data source type"""
        if self.data_source_type == "doc":
            return self._load_doc_chunks()
        elif self.data_source_type == "code":
            return self._load_code_chunks()
        elif self.data_source_type == "pair":
            return self._load_pair_chunks()
        else:
            raise ValueError(f"Unsupported data source type: {self.data_source_type}")

    def _load_doc_chunks(self) -> List[Chunk]:
        """Load chunks from ChromaDB for document mode"""
        client = chromadb.PersistentClient(path=self.config.chroma_db_path)
        collection = client.get_collection(name=self.config.chroma_collection)

        # Retrieve all chunks from the collection - specify 'documents' in include
        results = collection.get(include=["documents", "metadatas", "embeddings"])

        chunks = []
        for i, (doc_id, metadata, embedding) in enumerate(
            zip(results["ids"], results["metadatas"], results["embeddings"])
        ):
            chunk = Chunk(
                id=doc_id,
                content=results["documents"][i],  # This was None before
                chunk_type="doc",
                source_file=metadata.get("source_file", ""),
                metadata=metadata,
                embedding=embedding,
            )
            chunks.append(chunk)

        return chunks

    def _load_code_chunks(self) -> List[Chunk]:
        """Load chunks from NetworkX graph for code mode"""
        with open(self.config.code_graph_path, "rb") as f:
            code_graph = pickle.load(f)

        chunks = []
        for node_id, node_attrs in code_graph.graph.nodes(data=True):
            # Create chunks for different types of code entities
            chunk_type = node_attrs.get("type", "unknown")
            content = f"{node_attrs.get('name', '')}\n{node_attrs.get('code', '')}"

            chunk = Chunk(
                id=node_id,
                content=content,
                chunk_type=chunk_type,
                source_file=node_attrs.get("file_path", ""),
                metadata=node_attrs,
                embedding=node_attrs.get("embedding", np.array([])),
            )
            chunks.append(chunk)

        return chunks

    def _load_pair_chunks(self) -> List[Chunk]:
        """Load chunks from both document and code sources"""
        # First load doc chunks
        doc_chunks = self._load_doc_chunks()

        # Then load code chunks
        code_chunks = self._load_code_chunks()

        # Combine both
        return doc_chunks + code_chunks
