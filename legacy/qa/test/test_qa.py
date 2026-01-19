# test_qa_generator.py
import asyncio
import os
import tempfile

from qa.config import QAGeneratorConfig
from qa.main import QAGeneratorPipeline


async def main():
    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Adjust these paths based on your actual parser outputs
        doc_markdown_path = "/run/media/blazingbhavneek/Common/Code/datagen/parser/tests/output/consolidated.md"  # Replace with actual path
        chroma_db_path = "parser/tests/output/chroma_db"  # Replace with actual path
        chroma_collection = "docs_1768829901"  # Replace with actual collection name

        # Configuration for document-based QA generation
        config = QAGeneratorConfig(
            data_source_type="doc",  # Change to "code" or "pair" as needed
            doc_markdown_path=doc_markdown_path,
            chroma_db_path=chroma_db_path,
            chroma_collection=chroma_collection,
            code_graph_path=None,  # Only needed for code mode
            manifest_path=None,  # Only needed for pair mode
            llm_endpoint="http://localhost:8001/v1",  # Replace with your endpoint
            llm_model="lfm2-chat",  # Replace with your model
            embedding_endpoint="http://localhost:8000/v1",  # Replace with your endpoint
            embedding_model="ruri-embed",  # Replace with your model
            output_dir=os.path.join(temp_dir, "output"),
            dataset_name="test_dataset",
            cache_dir=os.path.join(temp_dir, "cache"),
            questions_per_chunk=5,  # Reduce for testing
            max_concurrent_agents=2,  # Reduce for testing
        )

        # Initialize and run the pipeline
        pipeline = QAGeneratorPipeline(config)

        try:
            result = await pipeline.run_full_pipeline()
            print(f"Pipeline completed successfully!")
            print(f"Dataset saved to: {result.dataset_path}")
            print(f"Statistics saved to: {result.statistics_path}")
        except Exception as e:
            print(f"Pipeline failed: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
