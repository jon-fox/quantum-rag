"""
Embedding utilities for generating vector embeddings for semantic search queries
and finding similar items.

Supports both OpenAI embeddings and SentenceTransformers.
"""
import os # Ensure os is imported
import logging # Ensure logging is imported
import numpy as np # Ensure numpy is imported
from typing import List, Dict, Union, Optional, Tuple, Any # Ensure these are imported
from dotenv import load_dotenv

# Try to load environment variables
load_dotenv()

# Disable progress bars to prevent long log lines
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import models based on what's installed
try:
    from openai import OpenAI  # Import the OpenAI client class for v1.0+
    OPENAI_AVAILABLE = True
    # Initialize OpenAI client using API key from environment variable
    if os.getenv("OPENAI_API_KEY"):
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        openai_client = None
except ImportError:
    logger.warning("OpenAI package not found. OpenAI embeddings will not be available.")
    OPENAI_AVAILABLE = False
    openai_client = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("SentenceTransformers package not found. Local embeddings will not be available.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Default models
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

# Embedding cache to avoid repeated calls for the same content
_embedding_cache = {}


class EmbeddingProvider:
    """Base class for embedding providers"""
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        raise NotImplementedError("Subclasses must implement get_embeddings")
    
    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension"""
        raise NotImplementedError("Subclasses must implement embedding_dim")


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI API-based embeddings"""
    
    def __init__(self, model_name: str = DEFAULT_OPENAI_EMBEDDING_MODEL):
        """Initialize OpenAI embedding provider
        
        Args:
            model_name: Name of the OpenAI embedding model to use
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package is not installed. Install with 'pip install openai'")
        
        if not openai_client:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.model_name = model_name
        self._embedding_dim = 1536 if model_name == "text-embedding-3-small" else 3072  # Default dimensions
        self.client = openai_client

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        # Check if any texts are in the cache
        cache_keys = [f"openai:{self.model_name}:{text}" for text in texts]
        uncached_indices = [i for i, key in enumerate(cache_keys) if key not in _embedding_cache]
        
        if uncached_indices:
            # Only process texts that aren't cached
            uncached_texts = [texts[i] for i in uncached_indices]
            
            try:
                # Using the new OpenAI client API format (v1.0+)
                response = self.client.embeddings.create(
                    input=uncached_texts,
                    model=self.model_name
                )
                
                # Process the embeddings from the response
                for i, embedding_data in enumerate(response.data):
                    # Store in cache
                    original_idx = uncached_indices[i]
                    _embedding_cache[cache_keys[original_idx]] = embedding_data.embedding
                    
            except Exception as e:
                logger.error(f"Error generating OpenAI embeddings: {e}")
                raise
        
        # Retrieve all embeddings (cached and newly generated)
        embeddings = np.array([_embedding_cache[key] for key in cache_keys])
        return embeddings
    
    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension for the model"""
        return self._embedding_dim


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Local sentence-transformers embeddings"""
    
    def __init__(self, model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL):
        """Initialize SentenceTransformer embedding provider
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "SentenceTransformers package is not installed. "
                "Install with 'pip install sentence-transformers'"
            )
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._embedding_dim_st: Optional[int] = self.model.get_sentence_embedding_dimension()
        if self._embedding_dim_st is None:
            # Fallback or error if dimension cannot be determined
            logger.warning(f"Could not determine embedding dimension for {model_name}. Defaulting or erroring might be needed.")
            # Depending on desired behavior, you could raise an error or set a default
            # For now, let's raise an error if it's None, as it's crucial for some operations.
            raise ValueError(f"Embedding dimension for {model_name} could not be determined.")

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using SentenceTransformer
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        # Check if any texts are in the cache
        cache_keys = [f"st:{self.model_name}:{text}" for text in texts]
        uncached_indices = [i for i, key in enumerate(cache_keys) if key not in _embedding_cache]
        
        if uncached_indices:
            # Only process texts that aren't cached
            uncached_texts = [texts[i] for i in uncached_indices]
            
            try:
                # Disable progress bar to prevent long log lines
                embeddings = self.model.encode(uncached_texts, show_progress_bar=False)
                
                for i, embedding in enumerate(embeddings):
                    # Store in cache
                    original_idx = uncached_indices[i]
                    _embedding_cache[cache_keys[original_idx]] = embedding.tolist()
                    
            except Exception as e:
                logger.error(f"Error generating SentenceTransformer embeddings: {e}")
                raise
        
        # Retrieve all embeddings (cached and newly generated)
        embeddings = np.array([_embedding_cache[key] for key in cache_keys])
        return embeddings
    
    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension for the model"""
        if self._embedding_dim_st is None: # Should have been caught in __init__
            raise ValueError("Embedding dimension was not properly initialized.")
        return self._embedding_dim_st


def get_embedding_provider(provider_type: Optional[str] = None) -> EmbeddingProvider: # Corrected type hint
    """Get an embedding provider based on configuration or availability
    
    Args:
        provider_type: Type of provider ('openai' or 'sentence_transformers')
            If None, will try to use environment variable EMBEDDING_PROVIDER
            
    Returns:
        An instance of EmbeddingProvider
    """
    # If provider_type not specified, check environment variable
    if provider_type is None:
        provider_type = os.getenv("EMBEDDING_PROVIDER", "").lower()
    
    # If still not specified, use what's available
    if not provider_type:
        if OPENAI_AVAILABLE:
            provider_type = "openai"
        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            provider_type = "sentence_transformers"
        else:
            raise ImportError(
                "Neither OpenAI nor SentenceTransformers are available. "
                "Install at least one with 'pip install openai' or 'pip install sentence-transformers'"
            )
    
    # Create provider based on type
    if provider_type == "openai":
        model_name = os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_EMBEDDING_MODEL)
        return OpenAIEmbedding(model_name=model_name)
    elif provider_type in ["sentence_transformers", "st", "transformer"]:
        model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", DEFAULT_SENTENCE_TRANSFORMER_MODEL)
        return SentenceTransformerEmbedding(model_name=model_name)
    else:
        raise ValueError(f"Unknown embedding provider type: {provider_type}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity (between -1 and 1)
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_similar_energy_data(
    query_embedding: np.ndarray,
    all_embeddings: np.ndarray,
    all_items: List[Dict[str, Any]],
    top_k: int = 5,
    min_similarity: float = 0.7
) -> List[Dict[str, Any]]:
    """Find energy data similar to the query embedding
    
    Args:
        query_embedding: Embedding of the query
        all_embeddings: Array of embeddings for all energy data points
        all_items: List of energy data items corresponding to all_embeddings
        top_k: Number of results to return
        min_similarity: Minimum similarity score to include in results
        
    Returns:
        List of dictionaries with items and their similarity scores
    """
    # Calculate similarity between query and all embeddings
    similarities = np.array([
        cosine_similarity(query_embedding, emb) for emb in all_embeddings
    ])
    
    # Get indices of top K similar items
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # Create result list with items and scores
    results = []
    for idx in top_indices:
        similarity = similarities[idx]
        
        # Only include if above minimum similarity threshold
        if similarity >= min_similarity:
            results.append({
                "item": all_items[idx],
                "similarity": float(similarity)
            })
    
    return results


def embed_query(
    query: str, 
    provider: Optional[EmbeddingProvider] = None
) -> np.ndarray:
    """Generate embedding for a search query
    
    Args:
        query: Search query string
        provider: EmbeddingProvider to use (if None, one will be created)
        
    Returns:
        Embedding vector for the query
    """
    # Create provider if not provided
    if provider is None:
        provider = get_embedding_provider()
    
    # Generate embedding
    embeddings = provider.get_embeddings([query])
    return embeddings[0]
