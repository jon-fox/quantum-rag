"""
Storage utility functions for working with watch data in multiple data stores.
"""
import numpy as np
from typing import Dict, Any, Optional, Union
import logging

# Import storage implementations
from src.storage.dual_storage import default_dual_storage, DualStorage

# Set up logging
logger = logging.getLogger(__name__)

def store_ebay_listing_with_embedding(
    listing: Dict[str, Any],
    embedding: np.ndarray,
    storage_instance: Optional[DualStorage] = None
) -> Dict[str, Any]:
    """
    Store an eBay listing with its embedding in both DynamoDB and PostgreSQL with pgvector.
    
    Args:
        listing: Dictionary containing the eBay listing data. Must include 'itemId'.
        embedding: NumPy array of the embedding vector (1536 dimensions for OpenAI embeddings).
        storage_instance: Optional custom DualStorage instance.
        
    Returns:
        Dictionary with success/failure results for each storage system.
        
    Example:
        ```python
        import numpy as np
        from src.storage.utils import store_ebay_listing_with_embedding
        
        # Example listing data
        listing = {
            "itemId": "12345678",
            "title": "Rolex Submariner 116610LN",
            "price": {"value": 12500.00, "currency": "USD"},
            "condition": "Used",
            "itemWebUrl": "https://www.ebay.com/itm/12345678"
        }
        
        # Example embedding (1536 dimensions)
        embedding = np.random.rand(1536)
        
        # Store the listing with its embedding
        result = store_ebay_listing_with_embedding(listing, embedding)
        print(f"Storage successful: {result['overall_success']}")
        ```
    """
    # Validate the embedding dimensions
    if embedding is None or not isinstance(embedding, np.ndarray):
        raise ValueError("Embedding must be a NumPy array")
    
    if embedding.shape != (1536,):
        logger.warning(f"Expected embedding of dimension 1536, got {embedding.shape}")
    
    # Use default storage instance if none provided
    storage = storage_instance or default_dual_storage
    
    # Store the listing with its embedding
    return storage.store_listing_with_embedding(listing, embedding)

# Additional convenience function for batch operations
def batch_store_ebay_listings_with_embeddings(
    listings: list,
    embeddings: np.ndarray,
    storage_instance: Optional[DualStorage] = None
) -> Dict[str, Any]:
    """
    Store multiple eBay listings with their embeddings in both DynamoDB and PostgreSQL.
    
    Args:
        listings: List of dictionaries containing eBay listing data.
        embeddings: NumPy array of embedding vectors, shape (n_listings, embedding_dim).
        storage_instance: Optional custom DualStorage instance.
        
    Returns:
        Dictionary with summary of success/failure counts for each storage system.
    """
    # Validate inputs
    if not listings:
        return {"error": "No listings provided", "overall_success": False}
    
    if embeddings is None or not isinstance(embeddings, np.ndarray):
        return {"error": "Embeddings must be a NumPy array", "overall_success": False}
    
    if len(listings) != embeddings.shape[0]:
        return {"error": "Number of listings must match number of embeddings", "overall_success": False}
    
    # Use default storage instance if none provided
    storage = storage_instance or default_dual_storage
    
    # Store the listings with their embeddings
    return storage.batch_store_listings_with_embeddings(listings, embeddings)

def normalize_listing_for_embedding(listing: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a watch listing from any source (eBay, WatchRecon, etc.) to ensure
    consistent data structure for embedding generation and storage.
    
    Args:
        listing: Dictionary containing watch listing data
        
    Returns:
        Normalized listing dictionary
    """
    # Create a copy to avoid modifying the original
    normalized = listing.copy()
    
    # Ensure required fields exist
    if "itemId" not in normalized:
        # Generate a placeholder ID if missing
        source = normalized.get("source", "unknown")
        normalized["itemId"] = f"{source}|unknown|{id(normalized)}"
        logger.warning(f"Missing itemId, generated placeholder: {normalized['itemId']}")
    
    # Ensure metadata exists
    if "metadata" not in normalized:
        normalized["metadata"] = {}
    
    # Normalize price structure to match eBay format
    if "price" in normalized:
        if not isinstance(normalized["price"], dict):
            try:
                price_value = float(normalized["price"])
                normalized["price"] = {
                    "value": price_value,
                    "currency": "USD"  # Default currency
                }
            except (ValueError, TypeError):
                normalized["price"] = {"value": 0.0, "currency": "USD"}
    else:
        normalized["price"] = {"value": 0.0, "currency": "USD"}
    
    # Add source if not present
    if "source" not in normalized:
        # Try to infer source from itemId
        if "itemId" in normalized and "|" in normalized["itemId"]:
            normalized["source"] = normalized["itemId"].split("|")[0]
        else:
            normalized["source"] = "unknown"
    
    # Ensure condition exists
    if "condition" not in normalized:
        normalized["condition"] = "Pre-owned"  # Default condition
    
    # Ensure URL field exists (itemWebUrl is used in storage)
    if "itemWebUrl" not in normalized and "url" in normalized:
        normalized["itemWebUrl"] = normalized["url"]
    
    return normalized

def get_embedding_text(listing: Dict[str, Any]) -> str:
    """
    Extract text from a listing that will be used to generate embeddings.
    Ensures consistency between different data sources.
    
    Args:
        listing: Dictionary containing watch listing data
        
    Returns:
        String containing text to be used for embedding generation
    """
    # Start with the title as the base text, ensure it's a string
    text = listing.get('title', '') or ''
    
    # Add description if available (most important for semantic search)
    if 'description' in listing and listing['description']:
        text += " " + str(listing['description'])
    
    # Add short description if available (sometimes used in eBay data)
    if 'shortDescription' in listing and listing['shortDescription']:
        text += " " + str(listing['shortDescription'])
        
    # Add brand and model information if available
    if 'brand' in listing and listing['brand']:
        text += " " + str(listing['brand'])
    if 'model' in listing and listing['model']:
        text += " " + str(listing['model'])
    
    # Add condition information
    condition = listing.get('condition', '')
    if condition:
        text += " " + str(condition)
    
    # If listing has metadata with additional useful attributes
    metadata = listing.get('metadata', {})
    if isinstance(metadata, dict):
        # Add reference number if available
        if 'reference' in metadata and metadata['reference']:
            text += " " + str(metadata['reference'])
        
        # Add material info if available
        if 'material' in metadata and metadata['material']:
            text += " " + str(metadata['material'])
            
        # Add tags if available (from WatchRecon)
        if 'tags' in metadata and isinstance(metadata['tags'], list):
            for tag in metadata['tags']:
                if tag:
                    text += " " + str(tag)
    
    # Ensure we return a valid string even if everything was None
    return text if text else "watch listing"