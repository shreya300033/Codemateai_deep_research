"""
Enhanced Embedding Engine with Hugging Face API support
Supports both local and online models for better accuracy
"""

import os
import logging
import numpy as np
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


class EnhancedEmbeddingEngine:
    """Enhanced embedding engine with local and online Hugging Face support."""
    
    def __init__(self, 
                 local_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 online_model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 use_online: bool = False,
                 api_token: Optional[str] = None):
        """
        Initialize the enhanced embedding engine.
        
        Args:
            local_model_name: Local model to use as fallback
            online_model_name: Online model for better accuracy
            use_online: Whether to use online API
            api_token: Hugging Face API token
        """
        self.local_model_name = local_model_name
        self.online_model_name = online_model_name
        self.use_online = use_online
        self.api_token = api_token or os.getenv("HUGGINGFACE_API_TOKEN")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize local model
        self.local_model = SentenceTransformer(local_model_name, device=self.device)
        self.local_embedding_dim = self.local_model.get_sentence_embedding_dimension()
        logger.info(f"Loaded local model: {local_model_name}, embedding dimension: {self.local_embedding_dim}")
        
        # Initialize online client if enabled
        self.online_client = None
        self.online_embedding_dim = None
        
        if self.use_online and self.api_token:
            try:
                self.online_client = InferenceClient(token=self.api_token)
                # Get embedding dimension for online model
                test_embedding = self.online_client.feature_extraction("test", model=online_model_name)
                self.online_embedding_dim = len(test_embedding[0])
                logger.info(f"Initialized online model: {online_model_name}, embedding dimension: {self.online_embedding_dim}")
            except Exception as e:
                logger.warning(f"Failed to initialize online model: {e}")
                logger.info("Falling back to local model only")
                self.use_online = False
                self.online_client = None
        else:
            if not self.api_token:
                logger.info("No API token provided - using local model only")
            else:
                logger.info("Online model disabled - using local model only")
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension of the active model."""
        if self.use_online and self.online_client:
            return self.online_embedding_dim
        return self.local_embedding_dim
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.array([])
        
        # Try online first if enabled
        if self.use_online and self.online_client:
            try:
                logger.info(f"Using online model for {len(texts)} texts")
                embeddings = self.online_client.feature_extraction(texts, model=self.online_model_name)
                return np.array(embeddings)
            except Exception as e:
                logger.warning(f"Online embedding failed: {e}")
                logger.info("Falling back to local model")
        
        # Use local model
        logger.info(f"Using local model for {len(texts)} texts")
        embeddings = self.local_model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.generate_embeddings([text])[0]
    
    def get_model_info(self) -> dict:
        """Get information about the current model configuration."""
        return {
            "local_model": self.local_model_name,
            "online_model": self.online_model_name if self.use_online else None,
            "using_online": self.use_online,
            "embedding_dimension": self.get_embedding_dimension(),
            "device": self.device,
            "api_available": self.api_token is not None
        }


def create_embedding_engine(use_online: bool = False, api_token: str = None) -> EnhancedEmbeddingEngine:
    """
    Factory function to create an embedding engine.
    
    Args:
        use_online: Whether to use online Hugging Face API
        api_token: Hugging Face API token
        
    Returns:
        EnhancedEmbeddingEngine instance
    """
    return EnhancedEmbeddingEngine(
        local_model_name="sentence-transformers/all-MiniLM-L6-v2",
        online_model_name="sentence-transformers/all-mpnet-base-v2",
        use_online=use_online,
        api_token=api_token
    )
