"""
Configuration settings for Deep Researcher Agent
"""

import os
from pathlib import Path

# Model configurations
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"

# Alternative models for different use cases
ALTERNATIVE_EMBEDDING_MODELS = {
    "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "large": "sentence-transformers/all-mpnet-base-v2",
    "fast": "sentence-transformers/all-MiniLM-L6-v2"
}

# Storage configurations
VECTOR_STORE_PATH = "./vector_store"
RESEARCH_HISTORY_PATH = "./research_history.json"
EXPORT_PATH = "./exports"

# Processing configurations
MAX_DOCUMENT_LENGTH = 10000  # Maximum characters per document
CHUNK_SIZE = 1000  # Size of text chunks for processing
CHUNK_OVERLAP = 200  # Overlap between chunks

# Search configurations
DEFAULT_MAX_RESULTS = 10
MAX_SEARCH_RESULTS = 50
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score for results

# Export configurations
PDF_PAGE_SIZE = "A4"
PDF_MARGIN = 1.0  # inches
MARKDOWN_EXTENSIONS = ['codehilite', 'fenced_code', 'tables']

# UI configurations
STREAMLIT_THEME = {
    "primaryColor": "#1f77b4",
    "backgroundColor": "#ffffff",
    "secondaryBackgroundColor": "#f0f2f6",
    "textColor": "#262730"
}

# Logging configurations
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Performance configurations
BATCH_SIZE = 32  # For processing multiple documents
MAX_WORKERS = 4  # For parallel processing
CACHE_EMBEDDINGS = True  # Cache embeddings to disk

# Supported file formats
SUPPORTED_FORMATS = {
    '.pdf': 'PDF documents',
    '.docx': 'Microsoft Word documents',
    '.txt': 'Plain text files',
    '.html': 'HTML files',
    '.md': 'Markdown files'
}

# Create necessary directories
def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [VECTOR_STORE_PATH, EXPORT_PATH]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

# Environment variables
def get_env_config():
    """Get configuration from environment variables."""
    return {
        'embedding_model': os.getenv('EMBEDDING_MODEL', EMBEDDING_MODEL),
        'summarization_model': os.getenv('SUMMARIZATION_MODEL', SUMMARIZATION_MODEL),
        'vector_store_path': os.getenv('VECTOR_STORE_PATH', VECTOR_STORE_PATH),
        'log_level': os.getenv('LOG_LEVEL', LOG_LEVEL),
        'max_workers': int(os.getenv('MAX_WORKERS', MAX_WORKERS)),
        'batch_size': int(os.getenv('BATCH_SIZE', BATCH_SIZE))
    }

