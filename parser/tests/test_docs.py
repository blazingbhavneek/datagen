import asyncio
import os
import random
from parser.parser import DocParserConfig, DocumentParser
from parser.embeddings import EmbeddingGenerator
from pathlib import Path
from pprint import pprint
import chromadb
import json


def print_separator(title="", char="=", width=80):
    """Print a separator line with optional title."""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(f"\n{char * width}")


def print_chunk_details(chunk_data, index=None, show_content_length=500):
    """Pretty print chunk details."""
    if index is not None:
        print(f"\n{'='*80}")
        print(f"CHUNK #{index}")
        print(f"{'='*80}")
    
    print(f"ID: {chunk_data.get('id', 'N/A')}")
    print(f"Source File: {chunk_data.get('source_file', 'N/A')}")
    print(f"Chunk Index: {chunk_data.get('chunk_index', 'N/A')}")
    print(f"Headers: {chunk_data.get('headers', [])}")
    
    # Print content (truncated)
    content = chunk_data.get('content', '')
    print(f"\nContent ({len(content)} chars):")
    print("-" * 80)
    if len(content) > show_content_length:
        print(content[:show_content_length] + "...")
    else:
        print(content)
    print("-" * 80)
    
    # Print summary points if available
    if 'summary_points' in chunk_data and chunk_data['summary_points']:
        print(f"\n📝 SUMMARY POINTS ({len(chunk_data['summary_points'])} points):")
        for i, sp in enumerate(chunk_data['summary_points'], 1):
            print(f"\n  {i}. {sp.get('text', 'N/A')}")
            
            if sp.get('prev_link'):
                prev = sp['prev_link']
                print(f"     ⬅️  PREV: {prev.get('relation', 'N/A')}")
                print(f"        Topic: {prev.get('common_topic', 'N/A')}")
            
            if sp.get('next_link'):
                next_link = sp['next_link']
                print(f"     ➡️  NEXT: {next_link.get('relation', 'N/A')}")
                print(f"        Topic: {next_link.get('common_topic', 'N/A')}")
    
    # Print metadata
    metadata = chunk_data.get('metadata', {})
    if metadata and metadata != {}:
        print(f"\n🏷️  METADATA:")
        for key, value in metadata.items():
            if key not in ['summary_points']:  # Skip already printed items
                print(f"  {key}: {value}")


async def main():
    # Get absolute path to input directory
    input_dir = Path(
        "/run/media/blazingbhavneek/Common/Code/datagen/parser/tests/input/test_docs_mini"
    )
    
    # Create output directory in the same location as this script
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Define paths for outputs
    markdown_path = output_dir / "consolidated.md"
    chroma_db_path = output_dir / "chroma_db"
    
    print_separator("CONFIGURATION", "=")
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Markdown Output: {markdown_path}")
    print(f"ChromaDB Path: {chroma_db_path}")
    
    # Configure the parser
    config = DocParserConfig(
        input_dir=str(input_dir),
        output_md_path=str(markdown_path),
        chroma_db_path=str(chroma_db_path),
        embedding_endpoint="http://localhost:8001/v1/",
        embedding_model="ruri-embed",
        chunk_size=2000,
        chunk_overlap=200,
        # LLM Linking Configuration (OPTIONAL)
        enable_llm_linking=True,  # Set to True to enable LLM linking
        llm_api_key=os.getenv('OPENAI_API_KEY', "sk-dummy"),  # Or provide directly
        llm_model="gpt-oss",
        llm_base_url="http://localhost:8000/v1",
        embed_with_summary=True,
        cleanup_temp=False,  # Keep temp files for inspection
        cleanup_cache=False,  # Keep cache for faster re-runs
    )
    
    print(f"\nLLM Linking: {'ENABLED ✓' if config.enable_llm_linking else 'DISABLED ✗'}")
    
    # Initialize and run the parser
    print_separator("PROCESSING DOCUMENTS", "=")
    pipeline = DocumentParser(config)
    result = await pipeline.process()
    
    print_separator("PROCESSING COMPLETE", "=")
    print(f"✓ Markdown output: {result.consolidated_md_path}")
    print(f"✓ ChromaDB collection: {result.chroma_collection_name}")
    print(f"✓ Total chunks created: {result.total_chunks}")
    
    if hasattr(result, 'metadata') and result.metadata:
        print(f"\n📊 Additional Stats:")
        for key, value in result.metadata.items():
            print(f"  {key}: {value}")
    
    # Display random chunks
    print_separator("RANDOM CHUNK SAMPLES", "=")
    num_samples = min(5, result.total_chunks)
    print(f"Displaying {num_samples} random chunks from {result.total_chunks} total:\n")
    
    # Get random chunk indices
    random_indices = random.sample(range(result.total_chunks), num_samples)
    random_indices.sort()
    
    for idx in random_indices:
        chunk_data = result.chunk_metadata[idx]
        print_chunk_details(chunk_data, index=idx)
    
    # Display sequential chunks (to see linking)
    if result.total_chunks >= 3:
        print_separator("SEQUENTIAL CHUNK SAMPLES (to see linking)", "=")
        print("Displaying 3 consecutive chunks to show prev/next relationships:\n")
        
        start_idx = random.randint(0, max(0, result.total_chunks - 3))
        for i in range(3):
            chunk_data = result.chunk_metadata[start_idx + i]
            print_chunk_details(chunk_data, index=start_idx + i, show_content_length=300)
    
    # Query ChromaDB
    print_separator("QUERYING CHROMADB", "=")
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=str(chroma_db_path))
    collection = client.get_collection(name=result.chroma_collection_name)
    
    # Get collection stats
    print(f"Collection: {result.chroma_collection_name}")
    print(f"Total documents in collection: {collection.count()}")
    
    # Query 1: Using text query
    query_text = "-fiopenmp -fopenmp-targets=spir64 オプションは何を有効にしますか?"
    print(f"\n🔍 Query: '{query_text}'")
    print(f"Retrieving top 5 results...\n")
    
    results = collection.query(
        query_texts=[query_text],
        n_results=5
    )
    
    # Print query results
    print("=" * 80)
    print("QUERY RESULTS")
    print("=" * 80)
    
    for i in range(len(results['ids'][0])):
        print(f"\n{'─'*80}")
        print(f"RESULT #{i+1}")
        print(f"{'─'*80}")
        
        chunk_id = results['ids'][0][i]
        distance = results['distances'][0][i] if 'distances' in results else None
        document = results['documents'][0][i]
        metadata = results['metadatas'][0][i]
        
        print(f"Similarity Score: {1 - distance:.4f}" if distance else "N/A")
        print(f"Distance: {distance:.4f}" if distance else "N/A")
        print(f"Chunk ID: {chunk_id}")
        print(f"Source: {metadata.get('source_file', 'N/A')}")
        print(f"Chunk Index: {metadata.get('chunk_index', 'N/A')}")
        
        if 'headers' in metadata:
            try:
                headers = json.loads(metadata['headers'])
                print(f"Headers: {headers}")
            except:
                print(f"Headers: {metadata['headers']}")
        
        print(f"\nContent Preview ({len(document)} chars):")
        print("-" * 80)
        preview_length = 400
        if len(document) > preview_length:
            print(document[:preview_length] + "...")
        else:
            print(document)
        print("-" * 80)
        
        # Show summary if available
        if 'summary_points' in metadata:
            try:
                summary_points = json.loads(metadata['summary_points'])
                print(f"\n📝 Summary Points:")
                for j, sp in enumerate(summary_points, 1):
                    print(f"  {j}. {sp.get('text', 'N/A')}")
            except:
                pass
    
    # Additional query examples
    print_separator("ADDITIONAL QUERY EXAMPLES", "=")
    
    # Query by embedding (if you want to test embedding-based search)
    print("\n💡 To query by embedding:")
    print("1. Generate embedding for your query text")
    print("2. Use collection.query(query_embeddings=[embedding], n_results=5)")
    
    # Show how to get specific chunk with context (if LLM linking is enabled)
    if config.enable_llm_linking and results['ids'][0]:
        print("\n💡 To get chunk with full context (prev/next):")
        sample_id = results['ids'][0][0]
        print(f"from parser.embeddings import ChromaDBManager")
        print(f"db = ChromaDBManager('{chroma_db_path}', '{result.chroma_collection_name}')")
        print(f"context = db.get_chunk_with_context('{sample_id}')")
    
    # Show raw ChromaDB output
    print_separator("RAW CHROMADB OUTPUT (for debugging)", "=")
    print("\nRaw ChromaDB query results structure:")
    print(f"Keys: {results.keys()}")
    print(f"Number of results: {len(results['ids'][0])}")
    
    print("\n📋 Full raw output:")
    pprint(results, depth=3, width=120)
    
    print_separator("TEST COMPLETE", "=")
    print("✓ All operations completed successfully!")
    print(f"\n📁 Output files:")
    print(f"  - Markdown: {markdown_path}")
    print(f"  - ChromaDB: {chroma_db_path}")
    if hasattr(result, 'processing_log') and result.processing_log:
        print(f"  - Log: {result.processing_log}")


if __name__ == "__main__":
    asyncio.run(main())
