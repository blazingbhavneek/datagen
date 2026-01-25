import asyncio
import os
from parser.parser import DocParserConfig, DocumentParser
from pathlib import Path
from pprint import pprint

import chromadb


async def main():
    # Get absolute path to input directory (replace with your actual path)
    input_dir = Path(
        "/run/media/blazingbhavneek/Common/Code/datagen/parser/tests/input/test_docs_mini"
    )  # Replace with actual path

    # Create output directory in the same location as this script
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # Define paths for outputs
    markdown_path = output_dir / "consolidated.md"
    chroma_db_path = output_dir / "chroma_db"

    # Configure the parser
    config = DocParserConfig(
        input_dir=str(input_dir),
        output_md_path=str(markdown_path),
        chroma_db_path=str(chroma_db_path),
        embedding_endpoint="http://localhost:8000/v1/",  # Replace with your endpoint
        embedding_model="ruri-embed",  # Replace with your model
        chunk_size=2000,
        chunk_overlap=200,
    )

    # Initialize and run the parser
    pipeline = DocumentParser(config)
    result = await pipeline.process()

    print(f"Processing complete!")
    print(f"Markdown output: {result.consolidated_md_path}")
    print(f"ChromaDB collection: {result.chroma_collection_name}")
    print(f"Total chunks created: {result.total_chunks}")

    # Load and query the ChromaDB
    print("\n--- Querying ChromaDB ---")

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=str(chroma_db_path))
    collection = client.get_collection(name=result.chroma_collection_name)

    query_text = "-fiopenmp -fopenmp-targets=spir64 オプションは何を有効にしますか?"  # Replace with your actual query
    results = collection.query(
        query_texts=[query_text], n_results=10  # Number of results to retrieve
    )

    # Print raw output from ChromaDB
    print("Raw ChromaDB query results:")
    pprint(results)


if __name__ == "__main__":
    asyncio.run(main())
