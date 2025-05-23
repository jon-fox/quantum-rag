"""
Embedding API endpoints for semantic search of energy consumption data and efficiency metrics.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import json
import os
from pathlib import Path
import logging # Add logging import

from src.embeddings.embed_utils import (
    get_embedding_provider,
    OpenAIEmbedding,  # Import OpenAI embedding provider explicitly
    embed_query,
    efficiency_aware_embedding,
    find_similar_energy_data,
    load_embeddings # Add load_embeddings to the import list
)

# Configure logger for this module
logger = logging.getLogger(__name__)
# BasicConfig should ideally be called once at application startup, 
# but for a module-specific logger, this ensures it has a handler if not configured globally.
# If global config is present (e.g. in app.py), this might be redundant or could be adjusted.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Default path for embeddings storage
EMBEDDINGS_PATH = Path(os.environ.get("EMBEDDINGS_PATH", "data/embeddings"))


class SearchQuery(BaseModel):
    """Search query model"""
    query: str
    top_k: int = 5
    min_similarity: float = 0.6
    efficiency_aware: bool = False
    min_efficiency: Optional[float] = None
    max_efficiency: Optional[float] = None


class EmbeddingResponse(BaseModel):
    """Response model for embedding requests"""
    vector: List[float]
    dimension: int
    model: str


class SearchResult(BaseModel):
    """Single search result"""
    item: Dict[str, Any]
    similarity: float


class SearchResponse(BaseModel):
    """Response model for search requests"""
    query: str
    results: List[SearchResult]
    count: int
    embedding_type: str
    model: str


def create_embedding_router() -> APIRouter:
    """Create FastAPI router for embedding endpoints"""
    router = APIRouter(
        prefix="",
        responses={404: {"description": "Not found"}},
    )
    
    # Ensure embeddings directory exists
    os.makedirs(EMBEDDINGS_PATH, exist_ok=True)
    
    # Track loaded embeddings
    _loaded_embeddings = {}
    
    # Initialize OpenAI embedding provider explicitly
    try:
        openai_provider = OpenAIEmbedding()
        default_provider = openai_provider
        default_model = openai_provider.model_name
    except (ImportError, ValueError) as e:
        # Fall back to default provider if OpenAI is not available
        default_provider = get_embedding_provider()
        default_model = getattr(default_provider, 'model_name', 'unknown')
    
    @router.get("/info", response_model=Dict[str, Any])
    async def get_embedding_info():
        """Get information about the embedding service"""
        return {
            "provider_type": default_provider.__class__.__name__,
            "embedding_dimension": default_provider.embedding_dim,
            "model": default_model,
            "available_indices": list(_loaded_embeddings.keys()) if _loaded_embeddings else [],
        }
    
    @router.post("/create", response_model=EmbeddingResponse)
    async def create_embedding(query: str = Query(..., description="Text to embed")):
        """Generate an embedding for the provided text"""
        embedding = embed_query(query, default_provider)
        
        return {
            "vector": embedding.tolist(),
            "dimension": len(embedding),
            "model": default_model
        }
    
    @router.post("/efficiency-aware", response_model=EmbeddingResponse)
    async def create_efficiency_aware_embedding(
        description: str = Query(..., description="Energy data description"),
        efficiency: float = Query(..., description="Energy efficiency metric"),
        weight: float = Query(0.2, description="Efficiency weight factor (0-1)")
    ):
        """Generate an efficiency-aware embedding"""
        embedding = efficiency_aware_embedding(description, efficiency, default_provider, efficiency_weight=weight)
        
        return {
            "vector": embedding.tolist(),
            "dimension": len(embedding),
            "model": default_model
        }
    
    @router.post("/search", response_model=SearchResponse)
    async def search_embeddings(search_request: SearchQuery):
        """
        Search for similar energy data using embeddings
        
        This endpoint first loads embeddings from the specified source file,
        then performs semantic search based on the query.
        """
        # Determine which embedding file to use - prefer OpenAI embeddings if available
        embedding_file = "openai_energy_embeddings.json"
        embedding_path = EMBEDDINGS_PATH / embedding_file
        
        if not embedding_path.exists():
            embedding_file = "energy_embeddings.json"
            embedding_path = EMBEDDINGS_PATH / embedding_file
        
        # Load embeddings if not already in memory
        # Check if the specific file's content is loaded, not just the file name as key
        if embedding_file not in _loaded_embeddings or not _loaded_embeddings[embedding_file][0]: # Check if items are loaded
            # Use a logger if available, otherwise print
            # Assuming logger is configured at the module level in embedding_api.py
            # If not, replace with `print` or ensure logger is passed/available
            logger.info(f"Cache miss or empty data for {embedding_file}. Attempting to load from {embedding_path}...")
            try:
                # Use the imported load_embeddings function
                items, embeddings_array = load_embeddings(str(embedding_path))
                if not items and embeddings_array.size == 0: # Check if loading actually failed or returned empty
                    logger.warning(f"load_embeddings returned empty for {embedding_path}. Using fallback empty data.")
                    _loaded_embeddings[embedding_file] = ([], np.array([]))
                else:
                    _loaded_embeddings[embedding_file] = (items, embeddings_array)
                    logger.info(f"Successfully loaded and cached embeddings from {embedding_path}")
            except Exception as e: # Catch any exception from load_embeddings itself
                logger.error(f"Critical error during load_embeddings for {embedding_path}: {e}. Using fallback empty data.")
                _loaded_embeddings[embedding_file] = ([], np.array([]))
        
        items, embeddings = _loaded_embeddings[embedding_file]

        if not items or embeddings.size == 0:
            logger.warning(f"No items or embeddings available for {embedding_file} after attempting to load. Returning empty search results.")
            return SearchResponse(
                query=search_request.query,
                results=[],
                count=0,
                embedding_type="text" if not search_request.efficiency_aware else "efficiency_aware",
                model=default_model
            )
        
        # Generate embedding for the query
        query_embedding = embed_query(search_request.query, default_provider)
        
        # Apply efficiency filtering if needed
        filtered_items = []
        filtered_embeddings_list = [] # Use a list to append before converting to np.array
        
        for idx, item_data in enumerate(items):
            # Apply efficiency filters if specified
            efficiency = item_data.get('efficiency', 0)
            if search_request.min_efficiency is not None and efficiency < search_request.min_efficiency:
                continue
            if search_request.max_efficiency is not None and efficiency > search_request.max_efficiency:
                continue
            
            filtered_items.append(item_data)
            if idx < len(embeddings): # Ensure index is within bounds for embeddings
                filtered_embeddings_list.append(embeddings[idx])
            else:
                logger.warning(f"Index {idx} out of bounds for embeddings array of length {len(embeddings)}. Skipping embedding for item.")


        if not filtered_items or not filtered_embeddings_list:
            logger.info("No items matched efficiency criteria or embeddings were missing.")
            return SearchResponse(
                query=search_request.query,
                results=[],
                count=0,
                embedding_type="text" if not search_request.efficiency_aware else "efficiency_aware",
                model=default_model
            )

        filtered_embeddings = np.array(filtered_embeddings_list)
        
        if filtered_embeddings.size == 0:
             logger.info("Filtered embeddings array is empty.")
             return SearchResponse(
                 query=search_request.query,
                 results=[],
                 count=0,
                 embedding_type="text" if not search_request.efficiency_aware else "efficiency_aware",
                 model=default_model
             )
        
        # Find similar energy data
        search_results = find_similar_energy_data(
            query_embedding=query_embedding,
            all_embeddings=filtered_embeddings,
            all_items=filtered_items,
            top_k=search_request.top_k,
            min_similarity=search_request.min_similarity
        )
        
        return {
            "query": search_request.query,
            "results": search_results,
            "count": len(search_results),
            "embedding_type": "text" if not search_request.efficiency_aware else "efficiency_aware",
            "model": default_model
        }
    
    return router