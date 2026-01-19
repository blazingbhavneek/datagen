# qa_generator/tools/vector_search.py
import pickle
from parser.docs import EmbeddingGenerator
from typing import Dict, List

import chromadb
import networkx as nx


class VectorSearchTool:
    """Tool for semantic search in document chunks"""

    def __init__(
        self,
        chroma_db_path: str,
        collection: str,
        embedding_endpoint: str,
        embedding_model: str,
    ):
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.client.get_collection(name=collection)
        self.embedder = EmbeddingGenerator(embedding_endpoint, embedding_model)

    async def search(self, query: str, n_results: int = 5) -> str:
        """Semantic search in document chunks"""
        try:
            # Generate embedding for query
            query_embedding = await self.embedder.generate_embeddings([query])

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                include=["documents", "metadatas"],
            )

            # Format results
            formatted_results = []
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                formatted_results.append(
                    {
                        "content": doc,
                        "source": meta.get("source_file", "Unknown"),
                        "page": meta.get("page", "N/A"),
                    }
                )

            return str(formatted_results)
        except Exception as e:
            return f"Error during vector search: {str(e)}"


class CodeGraphQueryTool:
    """Tool for querying the code graph"""

    def __init__(self, graph_path: str, embedding_endpoint: str, embedding_model: str):
        with open(graph_path, "rb") as f:
            self.graph_data = pickle.load(f)
        self.graph = self.graph_data.graph
        self.embedder = EmbeddingGenerator(embedding_endpoint, embedding_model)

    async def query(self, query: str, n_results: int = 5) -> str:
        """Query the code graph semantically"""
        try:
            # Find nodes by name first
            matching_nodes = []
            query_lower = query.lower()

            for node_id, attrs in self.graph.nodes(data=True):
                if query_lower in attrs.get("name", "").lower():
                    matching_nodes.append((node_id, attrs))

            # If not enough exact matches, try semantic search
            if len(matching_nodes) < n_results:
                query_embedding = await self.embedder.generate_embeddings([query])

                # Calculate similarity with node embeddings
                node_similarities = []
                for node_id, attrs in self.graph.nodes(data=True):
                    if "embedding" in attrs and attrs["embedding"] is not None:
                        # Simple cosine similarity calculation
                        # In practice, you'd want a more robust similarity function
                        similarity = self._calculate_similarity(
                            query_embedding[0], attrs["embedding"]
                        )
                        node_similarities.append((node_id, attrs, similarity))

                # Sort by similarity and add to results
                node_similarities.sort(key=lambda x: x[2], reverse=True)
                for node_id, attrs, sim in node_similarities:
                    if len(matching_nodes) >= n_results:
                        break
                    if not any(node_id == existing[0] for existing in matching_nodes):
                        matching_nodes.append((node_id, attrs))

            # Format results
            results = []
            for node_id, attrs in matching_nodes[:n_results]:
                results.append(
                    {
                        "type": attrs.get("type", "unknown"),
                        "name": attrs.get("name", "unnamed"),
                        "code": attrs.get("code", "")[:500],  # Limit code length
                        "file": attrs.get("file_path", "unknown"),
                        "summary": attrs.get("summary", ""),
                    }
                )

            return str(results)
        except Exception as e:
            return f"Error during code graph query: {str(e)}"

    def _calculate_similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings"""
        # Simplified similarity calculation
        # In practice, use numpy or scipy for efficient computation
        try:
            dot_product = sum(a * b for a, b in zip(emb1, emb2))
            magnitude1 = sum(a * a for a in emb1) ** 0.5
            magnitude2 = sum(b * b for b in emb2) ** 0.5
            if magnitude1 == 0 or magnitude2 == 0:
                return 0
            return dot_product / (magnitude1 * magnitude2)
        except:
            return 0
