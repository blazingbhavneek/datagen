"""
Demonstration script for reading and navigating chunks from the new ChromaDB format.
Shows how to work with chunks that have summary points and linking information.
"""

import chromadb
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class SummaryPoint:
    """Represents a summary point with linking information."""
    text: str
    prev_link: Optional[Dict[str, str]] = None  # {"chunk_id": "...", "chunk_index": N, "relation": "...", "common_topic": "..."}
    next_link: Optional[Dict[str, str]] = None  # {"chunk_id": "...", "chunk_index": N, "relation": "...", "common_topic": "..."}


class Chunk(BaseModel):
    """Enhanced chunk model with summary points and linking."""
    id: str
    content: str
    chunk_type: str = "doc"  # "doc", "function", "class", "file"
    source_file: str
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    headers: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    summary_points: List[Dict[str, Any]] = Field(default_factory=list)  # Stored as dicts in DB
    
    def get_summary_points(self) -> List[SummaryPoint]:
        """Convert stored summary point dicts to SummaryPoint objects."""
        return [
            SummaryPoint(
                text=sp.get("text", ""),
                prev_link=sp.get("prev_link"),
                next_link=sp.get("next_link")
            )
            for sp in self.summary_points
        ]
    
    def has_prev_links(self) -> bool:
        """Check if any summary points have links to previous chunk."""
        return any(sp.get("prev_link") for sp in self.summary_points)
    
    def has_next_links(self) -> bool:
        """Check if any summary points have links to next chunk."""
        return any(sp.get("next_link") for sp in self.summary_points)


# ============================================================================
# CHUNK READER
# ============================================================================

class ChunkReader:
    """
    Reader for navigating chunks stored in ChromaDB with the new format.
    Supports sequential navigation, random access, and filtering.
    """
    
    def __init__(self, chroma_db_path: str, collection_name: str):
        """
        Initialize the chunk reader.
        
        Args:
            chroma_db_path: Path to the ChromaDB directory
            collection_name: Name of the collection to read from
        """
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.client.get_collection(name=collection_name)
        self._chunks_cache: Optional[List[Chunk]] = None
        self._current_index: int = 0
    
    def _parse_metadata_field(self, field_value):
        """
        Parse metadata field that might be JSON-encoded.
        ChromaDB stores complex types as JSON strings.
        
        Args:
            field_value: The field value (might be string or already parsed)
            
        Returns:
            Parsed value (list/dict) or original value
        """
        import json
        
        if isinstance(field_value, str):
            try:
                return json.loads(field_value)
            except json.JSONDecodeError:
                return field_value
        return field_value
    
    def load_all_chunks(self, force_reload: bool = False) -> List[Chunk]:
        """
        Load all chunks from ChromaDB.
        Results are cached for efficient navigation.
        
        Args:
            force_reload: Force reload from DB even if cached
            
        Returns:
            List of all chunks sorted by chunk_index
        """
        if self._chunks_cache is not None and not force_reload:
            return self._chunks_cache
        
        # Get all documents from ChromaDB
        results = self.collection.get()
        
        chunks = []
        for doc_id, content, metadata in zip(
            results["ids"], 
            results["documents"], 
            results["metadatas"]
        ):
            # Parse JSON-encoded fields from metadata
            headers = self._parse_metadata_field(metadata.get("headers", []))
            summary_points = self._parse_metadata_field(metadata.get("summary_points", []))
            
            # Extract fields from metadata
            chunk = Chunk(
                id=doc_id,
                content=content,
                chunk_type="doc",
                source_file=metadata.get("source_file", "unknown"),
                chunk_index=metadata.get("chunk_index", 0),
                start_char=metadata.get("start_char", 0),
                end_char=metadata.get("end_char", 0),
                headers=headers,
                metadata=metadata,
                summary_points=summary_points
            )
            chunks.append(chunk)
        
        # Sort by chunk_index for sequential navigation
        chunks.sort(key=lambda x: x.chunk_index)
        
        self._chunks_cache = chunks
        print(f"Loaded {len(chunks)} chunks from ChromaDB")
        return chunks
    
    def get_chunk_by_index(self, chunk_index: int) -> Optional[Chunk]:
        """
        Get a chunk by its chunk_index.
        
        Args:
            chunk_index: The chunk index to retrieve
            
        Returns:
            Chunk if found, None otherwise
        """
        chunks = self.load_all_chunks()
        for chunk in chunks:
            if chunk.chunk_index == chunk_index:
                return chunk
        return None
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """
        Get a chunk by its ID.
        
        Args:
            chunk_id: The chunk ID to retrieve
            
        Returns:
            Chunk if found, None otherwise
        """
        try:
            results = self.collection.get(ids=[chunk_id])
            if not results["ids"]:
                return None
            
            metadata = results["metadatas"][0]
            
            # Parse JSON-encoded fields
            headers = self._parse_metadata_field(metadata.get("headers", []))
            summary_points = self._parse_metadata_field(metadata.get("summary_points", []))
            
            return Chunk(
                id=results["ids"][0],
                content=results["documents"][0],
                chunk_type="doc",
                source_file=metadata.get("source_file", "unknown"),
                chunk_index=metadata.get("chunk_index", 0),
                start_char=metadata.get("start_char", 0),
                end_char=metadata.get("end_char", 0),
                headers=headers,
                metadata=metadata,
                summary_points=summary_points
            )
        except Exception as e:
            print(f"Error retrieving chunk {chunk_id}: {e}")
            return None
    
    def get_current_chunk(self) -> Optional[Chunk]:
        """Get the chunk at the current navigation index."""
        chunks = self.load_all_chunks()
        if 0 <= self._current_index < len(chunks):
            return chunks[self._current_index]
        return None
    
    def next_chunk(self) -> Optional[Chunk]:
        """Move to and return the next chunk."""
        chunks = self.load_all_chunks()
        if self._current_index < len(chunks) - 1:
            self._current_index += 1
            return chunks[self._current_index]
        return None
    
    def prev_chunk(self) -> Optional[Chunk]:
        """Move to and return the previous chunk."""
        if self._current_index > 0:
            self._current_index -= 1
            return self.load_all_chunks()[self._current_index]
        return None
    
    def jump_to(self, index: int) -> Optional[Chunk]:
        """
        Jump to a specific position in the chunk sequence.
        
        Args:
            index: Position to jump to (0-based)
            
        Returns:
            Chunk at that position if valid, None otherwise
        """
        chunks = self.load_all_chunks()
        if 0 <= index < len(chunks):
            self._current_index = index
            return chunks[index]
        return None
    
    def get_chunks_by_source(self, source_file: str) -> List[Chunk]:
        """
        Get all chunks from a specific source file.
        
        Args:
            source_file: Source file to filter by
            
        Returns:
            List of chunks from that source file
        """
        chunks = self.load_all_chunks()
        return [c for c in chunks if c.source_file == source_file]
    
    def get_chunks_with_header(self, header: str) -> List[Chunk]:
        """
        Get all chunks containing a specific header.
        
        Args:
            header: Header text to search for
            
        Returns:
            List of chunks containing that header
        """
        chunks = self.load_all_chunks()
        return [c for c in chunks if header in c.headers]
    
    def get_chunks_with_summaries(self) -> List[Chunk]:
        """Get all chunks that have summary points."""
        chunks = self.load_all_chunks()
        return [c for c in chunks if c.summary_points]
    
    def get_linked_chunks(self) -> List[Chunk]:
        """Get all chunks that have links to previous or next chunks."""
        chunks = self.load_all_chunks()
        return [c for c in chunks if c.has_prev_links() or c.has_next_links()]
    
    def reset(self):
        """Reset navigation to the beginning."""
        self._current_index = 0


# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def print_chunk_info(chunk: Chunk, show_content: bool = True):
    """Pretty print chunk information."""
    print("\n" + "="*80)
    print(f"Chunk Index: {chunk.chunk_index}")
    print(f"ID: {chunk.id}")
    print(f"Source File: {chunk.source_file}")
    print(f"Headers: {' > '.join(chunk.headers) if chunk.headers else 'None'}")
    print(f"Position: {chunk.start_char} - {chunk.end_char}")
    
    if chunk.summary_points:
        print(f"\nSummary Points ({len(chunk.summary_points)}):")
        for i, sp_dict in enumerate(chunk.summary_points, 1):
            sp = SummaryPoint(**sp_dict)
            print(f"  {i}. {sp.text}")
            if sp.prev_link:
                print(f"     ← Prev: {sp.prev_link.get('relation', 'N/A')} "
                      f"(Topic: {sp.prev_link.get('common_topic', 'N/A')})")
            if sp.next_link:
                print(f"     → Next: {sp.next_link.get('relation', 'N/A')} "
                      f"(Topic: {sp.next_link.get('common_topic', 'N/A')})")
    else:
        print("\nNo summary points available")
    
    if show_content:
        print(f"\nContent Preview ({len(chunk.content)} chars):")
        preview = chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content
        print(preview)
    print("="*80)


def demo_basic_operations(reader: ChunkReader):
    """Demonstrate basic chunk reading operations."""
    print("\n" + "#"*80)
    print("# DEMO: Basic Operations")
    print("#"*80)
    
    # Load all chunks
    print("\n1. Loading all chunks...")
    all_chunks = reader.load_all_chunks()
    print(f"   Total chunks: {len(all_chunks)}")
    
    # Get first chunk
    print("\n2. Getting first chunk...")
    first_chunk = reader.get_chunk_by_index(0)
    if first_chunk:
        print_chunk_info(first_chunk, show_content=False)
    
    # Get a specific chunk by ID
    print("\n3. Getting chunk by ID...")
    if all_chunks:
        chunk_id = all_chunks[0].id
        chunk = reader.get_chunk_by_id(chunk_id)
        print(f"   Retrieved chunk: {chunk.chunk_index if chunk else 'Not found'}")


def demo_navigation(reader: ChunkReader):
    """Demonstrate sequential navigation."""
    print("\n" + "#"*80)
    print("# DEMO: Sequential Navigation")
    print("#"*80)
    
    reader.reset()
    
    # Get current chunk
    print("\n1. Current chunk (start)...")
    current = reader.get_current_chunk()
    if current:
        print(f"   Chunk index: {current.chunk_index}")
    
    # Move forward
    print("\n2. Moving forward 3 chunks...")
    for i in range(3):
        chunk = reader.next_chunk()
        if chunk:
            print(f"   Step {i+1}: Chunk {chunk.chunk_index}")
    
    # Move backward
    print("\n3. Moving backward 2 chunks...")
    for i in range(2):
        chunk = reader.prev_chunk()
        if chunk:
            print(f"   Step {i+1}: Chunk {chunk.chunk_index}")
    
    # Jump to specific position
    print("\n4. Jumping to position 10...")
    chunk = reader.jump_to(10)
    if chunk:
        print(f"   At chunk index: {chunk.chunk_index}")


def demo_filtering(reader: ChunkReader):
    """Demonstrate filtering operations."""
    print("\n" + "#"*80)
    print("# DEMO: Filtering Chunks")
    print("#"*80)
    
    # Get chunks with summaries
    print("\n1. Finding chunks with summaries...")
    with_summaries = reader.get_chunks_with_summaries()
    print(f"   Found {len(with_summaries)} chunks with summary points")
    if with_summaries:
        print(f"   Example: Chunk {with_summaries[0].chunk_index} has "
              f"{len(with_summaries[0].summary_points)} summary points")
    
    # Get linked chunks
    print("\n2. Finding chunks with links...")
    linked = reader.get_linked_chunks()
    print(f"   Found {len(linked)} chunks with prev/next links")
    if linked:
        chunk = linked[0]
        print(f"   Example: Chunk {chunk.chunk_index}")
        print(f"   - Has prev links: {chunk.has_prev_links()}")
        print(f"   - Has next links: {chunk.has_next_links()}")
    
    # Get chunks by source file
    print("\n3. Finding chunks by source file...")
    all_chunks = reader.load_all_chunks()
    if all_chunks:
        source_file = all_chunks[0].source_file
        by_source = reader.get_chunks_by_source(source_file)
        print(f"   Source: {source_file}")
        print(f"   Found {len(by_source)} chunks from this source")


def demo_link_traversal(reader: ChunkReader):
    """Demonstrate traversing chunks using their links."""
    print("\n" + "#"*80)
    print("# DEMO: Link Traversal")
    print("#"*80)
    
    # Find a chunk with both prev and next links
    linked_chunks = reader.get_linked_chunks()
    
    if not linked_chunks:
        print("\n   No linked chunks found (LLM linking not enabled)")
        return
    
    # Find chunk with both directions
    target_chunk = None
    for chunk in linked_chunks:
        if chunk.has_prev_links() and chunk.has_next_links():
            target_chunk = chunk
            break
    
    if not target_chunk:
        target_chunk = linked_chunks[0]
    
    print(f"\n1. Starting with chunk {target_chunk.chunk_index}...")
    print_chunk_info(target_chunk, show_content=False)
    
    # Show links and traverse
    print("\n2. Analyzing links...")
    summary_points = target_chunk.get_summary_points()
    
    prev_chunk_id = None
    next_chunk_id = None
    
    for i, sp in enumerate(summary_points, 1):
        print(f"\n   Summary Point {i}: {sp.text[:80]}...")
        
        if sp.prev_link:
            prev_chunk_id = sp.prev_link.get('chunk_id')
            print(f"   ← PREVIOUS: {sp.prev_link.get('relation')}")
            print(f"     Topic: {sp.prev_link.get('common_topic')}")
            print(f"     Target ID: {prev_chunk_id}")
            print(f"     Target Index: {sp.prev_link.get('chunk_index')}")
        
        if sp.next_link:
            next_chunk_id = sp.next_link.get('chunk_id')
            print(f"   → NEXT: {sp.next_link.get('relation')}")
            print(f"     Topic: {sp.next_link.get('common_topic')}")
            print(f"     Target ID: {next_chunk_id}")
            print(f"     Target Index: {sp.next_link.get('chunk_index')}")
    
    # Navigate to previous chunk using link
    if prev_chunk_id:
        print(f"\n3. Following link to PREVIOUS chunk...")
        prev_chunk = reader.get_chunk_by_id(prev_chunk_id)
        if prev_chunk:
            print(f"   Successfully navigated to chunk {prev_chunk.chunk_index}")
            print(f"   Source: {prev_chunk.source_file}")
            print(f"   Headers: {' > '.join(prev_chunk.headers[:2]) if prev_chunk.headers else 'None'}")
            print(f"   Content preview: {prev_chunk.content[:100]}...")
    
    # Navigate to next chunk using link
    if next_chunk_id:
        print(f"\n4. Following link to NEXT chunk...")
        next_chunk = reader.get_chunk_by_id(next_chunk_id)
        if next_chunk:
            print(f"   Successfully navigated to chunk {next_chunk.chunk_index}")
            print(f"   Source: {next_chunk.source_file}")
            print(f"   Headers: {' > '.join(next_chunk.headers[:2]) if next_chunk.headers else 'None'}")
            print(f"   Content preview: {next_chunk.content[:100]}...")
    
    # Demonstrate chain traversal
    print(f"\n5. Traversing forward through link chain...")
    current = target_chunk
    chain_length = 0
    max_chain = 3
    
    while chain_length < max_chain:
        # Find next link
        next_id = None
        for sp in current.get_summary_points():
            if sp.next_link:
                next_id = sp.next_link.get('chunk_id')
                relation = sp.next_link.get('relation')
                topic = sp.next_link.get('common_topic')
                break
        
        if not next_id:
            print(f"   Chain ends at chunk {current.chunk_index} (no more links)")
            break
        
        next_chunk = reader.get_chunk_by_id(next_id)
        if not next_chunk:
            print(f"   Could not find linked chunk {next_id}")
            break
        
        print(f"   Step {chain_length + 1}: {current.chunk_index} → {next_chunk.chunk_index}")
        print(f"      Relation: {relation}, Topic: {topic}")
        
        current = next_chunk
        chain_length += 1
    
    print(f"   Traversed {chain_length} links in the chain")


def demo_statistics(reader: ChunkReader):
    """Show overall statistics about the chunks."""
    print("\n" + "#"*80)
    print("# DEMO: Chunk Statistics")
    print("#"*80)
    
    all_chunks = reader.load_all_chunks()
    
    print(f"\nTotal chunks: {len(all_chunks)}")
    
    # Count chunks with summaries
    with_summaries = sum(1 for c in all_chunks if c.summary_points)
    print(f"Chunks with summaries: {with_summaries} ({with_summaries/len(all_chunks)*100:.1f}%)")
    
    # Count total summary points
    total_points = sum(len(c.summary_points) for c in all_chunks)
    print(f"Total summary points: {total_points}")
    if with_summaries > 0:
        print(f"Average points per chunk: {total_points/with_summaries:.2f}")
    
    # Count links
    chunks_with_prev = sum(1 for c in all_chunks if c.has_prev_links())
    chunks_with_next = sum(1 for c in all_chunks if c.has_next_links())
    print(f"Chunks with prev links: {chunks_with_prev}")
    print(f"Chunks with next links: {chunks_with_next}")
    
    # Source file breakdown
    sources = {}
    for chunk in all_chunks:
        sources[chunk.source_file] = sources.get(chunk.source_file, 0) + 1
    print(f"\nSource files: {len(sources)}")
    for source, count in list(sources.items())[:5]:  # Show first 5
        print(f"  - {source}: {count} chunks")


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Run all demonstrations."""
    
    # Configuration - UPDATE THESE PATHS
    config = {
        "chroma_db_path": "/run/media/blazingbhavneek/Common/Code/datagen/parser/tests/output/chroma_db",  # TODO: Update this
        "collection_name": "docs_1769455393",     # TODO: Update this
    }
    
    print("="*80)
    print("ChromaDB Chunk Reader - Demonstration")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  DB Path: {config['chroma_db_path']}")
    print(f"  Collection: {config['collection_name']}")
    
    try:
        # Initialize reader
        reader = ChunkReader(
            chroma_db_path=config["chroma_db_path"],
            collection_name=config["collection_name"]
        )
        
        # Run demonstrations
        demo_basic_operations(reader)
        demo_navigation(reader)
        demo_filtering(reader)
        demo_link_traversal(reader)
        demo_statistics(reader)
        
        print("\n" + "="*80)
        print("Demo completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError running demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
