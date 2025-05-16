"""
Embeddings Module

This module provides text embedding functionality for the RAG application.
"""
from typing import List, Dict, Any, Optional, Union
import numpy as np
import logging
import os
import hashlib

logger = logging.getLogger(__name__)

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Sentence Transformers not available. Using mock embeddings.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Global model instance for reuse
_model = None

def get_embedding_model():
    """
    Get or initialize the embedding model.
    
    Returns:
        The embedding model instance
    """
    global _model
    
    if _model is not None:
        return _model
        
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            # Use a suitable model for embeddings
            model_name = os.environ.get("EMBEDDING_MODEL", "all-mpnet-base-v2")
            _model = SentenceTransformer(model_name)
            logger.info(f"Initialized embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            _model = None
    
    return _model

async def get_embeddings(text: Union[str, List[str]], normalize: bool = True) -> Union[List[float], List[List[float]]]:
    """
    Generate embeddings for text.
    
    Args:
        text: Text string or list of strings to embed
        normalize: Whether to normalize the vectors
        
    Returns:
        Vector or list of vectors
    """
    if not text:
        return [] if isinstance(text, list) else []
        
    # Get model
    model = get_embedding_model()
    
    if model:
        try:
            # Generate embeddings using sentence-transformers
            if isinstance(text, list):
                embeddings = model.encode(text, normalize_embeddings=normalize)
                return embeddings.tolist()
            else:
                embedding = model.encode(text, normalize_embeddings=normalize)
                return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return _mock_embeddings(text, normalize)
    else:
        # Use mock embeddings if model not available
        return _mock_embeddings(text, normalize)

def _mock_embeddings(text: Union[str, List[str]], normalize: bool = True) -> Union[List[float], List[List[float]]]:
    """
    Generate deterministic mock embeddings for text.
    
    Args:
        text: Text string or list of strings
        normalize: Whether to normalize vectors
        
    Returns:
        Mock embedding vector(s)
    """
    dim = 384  # Common embedding dimension
    
    if isinstance(text, list):
        vectors = []
        for item in text:
            vectors.append(_deterministic_vector(item, dim, normalize))
        return vectors
    else:
        return _deterministic_vector(text, dim, normalize)
    
def _deterministic_vector(text: str, dim: int = 384, normalize: bool = True) -> List[float]:
    """
    Generate a deterministic vector based on the text.
    
    Args:
        text: Input text
        dim: Vector dimension
        normalize: Whether to normalize the vector
        
    Returns:
        Deterministic vector
    """
    # Create a deterministic seed from the text
    seed = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % 10000
    np.random.seed(seed)
    
    # Generate a random vector
    vector = np.random.random(dim).astype(np.float32)
    
    # Normalize if requested
    if normalize and np.linalg.norm(vector) > 0:
        vector = vector / np.linalg.norm(vector)
        
    return vector.tolist()