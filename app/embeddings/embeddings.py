"""
Embeddings Module

This module provides text embedding functionality for the RAG application,
integrated with LangChain's embedding interfaces.
"""
from typing import List, Dict, Any, Optional, Union
import numpy as np
import logging
import os
import hashlib

logger = logging.getLogger(__name__)

# Try to import langchain and sentence-transformers
try:
    from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain not available. Falling back to direct embedding methods.")
    LANGCHAIN_AVAILABLE = False
    
    try:
        from sentence_transformers import SentenceTransformer
        SENTENCE_TRANSFORMERS_AVAILABLE = True
    except ImportError:
        logger.warning("Sentence Transformers not available. Using mock embeddings.")
        SENTENCE_TRANSFORMERS_AVAILABLE = False

# Global model instance for reuse
_model = None

def get_embedding_model(model_name=None):
    """
    Get or initialize the embedding model using LangChain's interface when available.
    
    Args:
        model_name: Optional model name to override the default or environment setting
        
    Returns:
        The embedding model instance
    """
    global _model
    
    if _model is not None:
        return _model
    
    model_name = model_name or os.environ.get("EMBEDDING_MODEL", "all-mpnet-base-v2")
    
    if LANGCHAIN_AVAILABLE:
        try:
            # Use OpenAI embeddings if configured, otherwise use HuggingFace
            if os.environ.get("USE_OPENAI_EMBEDDINGS", "false").lower() == "true":
                _model = OpenAIEmbeddings()
                logger.info("Initialized OpenAI embedding model")
            else:
                _model = HuggingFaceEmbeddings(model_name=model_name)
                logger.info(f"Initialized HuggingFace embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing LangChain embedding model: {str(e)}")
            _model = None
    elif SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            # Fallback to direct sentence-transformers usage
            _model = SentenceTransformer(model_name)
            logger.info(f"Initialized sentence-transformers model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            _model = None
    
    return _model

async def get_embeddings(text: Union[str, List[str]], normalize: bool = True) -> Union[List[float], List[List[float]]]:
    """
    Generate embeddings for text using LangChain or fallback methods.
    
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
    
    if model and LANGCHAIN_AVAILABLE:
        try:
            # Generate embeddings using LangChain's interface
            if isinstance(text, list):
                embeddings = model.embed_documents(text)
                return embeddings
            else:
                embedding = model.embed_query(text)
                return embedding
        except Exception as e:
            logger.error(f"Error generating embeddings with LangChain: {str(e)}")
            # Try fallback if LangChain fails
            if hasattr(model, 'encode'):
                try:
                    return await _generate_with_sentence_transformers(model, text, normalize)
                except Exception as inner_e:
                    logger.error(f"Fallback embedding also failed: {str(inner_e)}")
                    return _mock_embeddings(text, normalize)
            else:
                return _mock_embeddings(text, normalize)
    elif model and SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            # Use direct sentence-transformers as fallback
            return await _generate_with_sentence_transformers(model, text, normalize)
        except Exception as e:
            logger.error(f"Error generating embeddings with sentence-transformers: {str(e)}")
            return _mock_embeddings(text, normalize)
    else:
        # Use mock embeddings if no model available
        return _mock_embeddings(text, normalize)

async def _generate_with_sentence_transformers(model, text, normalize):
    """Helper method to generate embeddings with sentence-transformers"""
    if isinstance(text, list):
        embeddings = model.encode(text, normalize_embeddings=normalize)
        return embeddings.tolist()
    else:
        embedding = model.encode(text, normalize_embeddings=normalize)
        return embedding.tolist()

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