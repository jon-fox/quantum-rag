"""
Embedding utilities for generating, storing, and searching vector embeddings
of energy consumption data and efficiency metrics for semantic search.

Supports both OpenAI embeddings and SentenceTransformers.
"""
import os
import json
import logging
from typing import List, Dict, Union, Optional, Tuple, Any
import numpy as np
from dotenv import load_dotenv

# Try to load environment variables
load_dotenv()

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
                embeddings = self.model.encode(uncached_texts)
                
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
        return self.model.get_sentence_embedding_dimension()


def get_embedding_provider(provider_type: str = None) -> EmbeddingProvider:
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


def create_energy_embedding_text(item: Dict[str, Any]) -> str:
    """Create a combined text representation of energy data details for embedding
    
    Args:
        item: Dictionary containing energy data details
        
    Returns:
        String combining relevant energy data details
    """
    # Extract fields with fallbacks to empty strings
    source_type = item.get("source_type", "").strip()
    location = item.get("location", "").strip()
    meter_id = item.get("meter_id", "").strip()
    description = item.get("description", "").strip()
    period = item.get("period", "").strip()
    year = item.get("year", "").strip()
    
    # Combine fields with importance weighting (repeat important fields)
    text_parts = []
    
    # Source type and location are most important - repeat them
    if source_type:
        text_parts.extend([source_type] * 3)
    if location:
        text_parts.extend([location] * 3)
    if meter_id:
        text_parts.extend([meter_id] * 2)
        
    # Add year and period once
    if year:
        text_parts.append(f"Year {year}")
    if period:
        text_parts.append(f"Period: {period}")
        
    # Add full description at the end
    if description:
        text_parts.append(description)
        
    # Join all parts with spaces
    return " ".join(text_parts)


def embed_energy_data_items(
    items: List[Dict[str, Any]], 
    provider: Optional[EmbeddingProvider] = None
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Generate embeddings for a list of energy data items
    
    Args:
        items: List of dictionaries containing energy data details
        provider: EmbeddingProvider to use (if None, one will be created)
        
    Returns:
        Tuple of (processed_items, embeddings_array)
    """
    # Create provider if not provided
    if provider is None:
        provider = get_embedding_provider()
    
    # Create text representations for each energy data entry
    texts = []
    for item in items:
        # Add the embedding text to each item
        item["embedding_text"] = create_energy_embedding_text(item)
        texts.append(item["embedding_text"])
    
    # Generate embeddings
    embeddings = provider.get_embeddings(texts)
    
    return items, embeddings


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


def efficiency_aware_embedding(
    description: str,
    efficiency: float,
    provider: Optional[EmbeddingProvider] = None,
    efficiency_weight: float = 0.2
) -> np.ndarray:
    """Generate a specialized embedding that includes energy efficiency awareness
    
    This combines a normal text embedding with efficiency information
    to allow semantic search that's aware of efficiency ranges.
    
    Args:
        description: Text description of the energy data
        efficiency: Efficiency metric of the energy source
        provider: EmbeddingProvider to use (if None, one will be created)
        efficiency_weight: Weight to give the efficiency factor (0-1)
        
    Returns:
        Modified embedding vector
    """
    # Create provider if not provided
    if provider is None:
        provider = get_embedding_provider()
    
    # Generate normal embedding
    text_embedding = provider.get_embeddings([description])[0]
    
    # Create an efficiency-aware prefix
    efficiency_context = f"This energy source has efficiency rating of {efficiency:.2f}. "
    efficiency_embedding = provider.get_embeddings([efficiency_context + description])[0]
    
    # Combine embeddings with weighting
    combined_embedding = (1 - efficiency_weight) * text_embedding + efficiency_weight * efficiency_embedding
    
    # Normalize the result
    combined_embedding /= np.linalg.norm(combined_embedding)
    
    return combined_embedding


# def save_embeddings(
#     items: List[Dict[str, Any]], 
#     embeddings: np.ndarray, 
#     filename: str
# ) -> None:
#     """Save embeddings and items to a file
    
#     Args:
#         items: List of energy data dictionaries
#         embeddings: Array of embeddings
#         filename: Path to save the file
#     """
#     data = {
#         "items": items,
#         "embeddings": embeddings.tolist()
#     }
    
#     with open(filename, 'w') as f:
#         json.dump(data, f)
    
#     logger.info(f"Saved {len(items)} embeddings to {filename}")

# def load_embeddings(filename: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
#     """Load embeddings and items from a file
    
#     Args:
#         filename: Path to load the file from
        
#     Returns:
#         Tuple of (items, embeddings)
#     """
#     with open(filename, 'r') as f:
#         data = json.load(f)
    
#     items = data["items"]
#     embeddings = np.array(data["embeddings"])
    
#     logger.info(f"Loaded {len(items)} embeddings from {filename}")
#     return items, embeddings