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


def setup_logger(name: str, log_dir: str = "./logs") -> logging.Logger:
    """
    Create logger with file and console handlers

    - Separate log file per agent instance
    - Timestamped entries
    - Configurable log level
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if logger already exists
    if logger.handlers:
        return logger

    # Create file handler
    log_file = os.path.join(log_dir, f"{name}_{int(time.time())}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
