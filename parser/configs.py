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


@dataclass
class DocParserConfig:
    input_dir: str  # Directory containing documents
    output_md_path: str  # Path for consolidated markdown output
    chroma_db_path: str  # Path for ChromaDB persistence
    embedding_endpoint: str  # e.g., "http://192.168.1.100:8000/v1/embeddings"
    embedding_model: str  # Model name for embeddings
    chunk_size: int = 1000  # Characters per chunk
    chunk_overlap: int = 200  # Character overlap between chunks
    supported_formats: List[str] = None  # Default formats if not provided
    enable_llm_linking: bool = True  # Whether to link chunks using LLM
    llm_api_key: Optional[str] = None  # API key for LLM
    llm_model: str = "gpt-oss"  # LLM model name
    llm_base_url: str = "http://localhost:8000/v1"  # Base URL for LLM API
    embed_with_summary: bool = True  # Whether to embed chunk summaries
    cleanup_temp: bool = False # Whether to delete temp files after processing
    cleanup_cache: bool = False # Whether to clear cache after processing

    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [
                ".pdf",
                ".docx",
                ".html",
                ".md",
                ".txt",
                ".xlsx",
                ".pptx",
            ]


@dataclass
class DocParserOutput:
    consolidated_md_path: str  # Path to final markdown file
    chroma_collection_name: str  # ChromaDB collection identifier
    chunk_metadata: List[Dict]  # Metadata for each chunk
    total_chunks: int
    processing_log: str  # Path to processing log
    metadata: Optional[Dict] = None  # Additional stats or info