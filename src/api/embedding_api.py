"""
Embedding API endpoints for semantic search of watch descriptions and prices.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import json
import os
from pathlib import Path

from src.embeddings.embed_utils import (
    get_embedding_provider,
    OpenAIEmbedding,  # Import OpenAI embedding provider explicitly
    embed_query,
    price_aware_embedding,
    find_similar_watches,
    load_embeddings
)

# Default path for embeddings storage
EMBEDDINGS_PATH = Path(os.environ.get("EMBEDDINGS_PATH", "data/embeddings"))


class SearchQuery(BaseModel):
    """Search query model"""
    query: str
    top_k: int = 5
    min_similarity: float = 0.6
    price_aware: bool = False
    min_price: Optional[float] = None
    max_price: Optional[float] = None


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
        prefix="/embeddings",
        tags=["embeddings"],
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
    
    @router.post("/price-aware", response_model=EmbeddingResponse)
    async def create_price_aware_embedding(
        description: str = Query(..., description="Watch description"),
        price: float = Query(..., description="Watch price"),
        weight: float = Query(0.2, description="Price weight factor (0-1)")
    ):
        """Generate a price-aware embedding"""
        embedding = price_aware_embedding(description, price, default_provider, price_weight=weight)
        
        return {
            "vector": embedding.tolist(),
            "dimension": len(embedding),
            "model": default_model
        }
    
    @router.post("/search", response_model=SearchResponse)
    async def search_embeddings(search_request: SearchQuery):
        """
        Search for similar watches using embeddings
        
        This endpoint first loads embeddings from the specified source file,
        then performs semantic search based on the query.
        """
        # Determine which embedding file to use - prefer OpenAI embeddings if available
        embedding_file = "openai_watch_embeddings.json"  # Try to use OpenAI embeddings first
        
        # Construct full path
        embedding_path = EMBEDDINGS_PATH / embedding_file
        
        # If OpenAI embeddings don't exist, fall back to default embeddings
        if not embedding_path.exists():
            embedding_file = "watch_embeddings.json"
            embedding_path = EMBEDDINGS_PATH / embedding_file
        
        # Load embeddings if not already in memory
        if embedding_file not in _loaded_embeddings:
            try:
                items, embeddings = load_embeddings(str(embedding_path))
                _loaded_embeddings[embedding_file] = (items, embeddings)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                # If we can't load embeddings, use the demo data from example script
                from src.scripts.example_embedding import SAMPLE_WATCHES
                from src.embeddings.embed_utils import embed_watch_items
                
                # Use OpenAI for embedding generation
                items, embeddings = embed_watch_items(SAMPLE_WATCHES, default_provider)
                _loaded_embeddings[embedding_file] = (items, embeddings)
        else:
            items, embeddings = _loaded_embeddings[embedding_file]
        
        # Generate embedding for the query
        query_embedding = embed_query(search_request.query, default_provider)
        
        # Apply price filtering if needed
        filtered_items = []
        filtered_embeddings = []
        
        for idx, item in enumerate(items):
            # Apply price filters if specified
            price = item.get('price', 0)
            if search_request.min_price is not None and price < search_request.min_price:
                continue
            if search_request.max_price is not None and price > search_request.max_price:
                continue
            
            filtered_items.append(item)
            filtered_embeddings.append(embeddings[idx])
        
        filtered_embeddings = np.array(filtered_embeddings)
        
        # If no items match the criteria
        if not filtered_items:
            return {
                "query": search_request.query,
                "results": [],
                "count": 0,
                "embedding_type": "text" if not search_request.price_aware else "price_aware",
                "model": default_model
            }
        
        # Find similar watches
        search_results = find_similar_watches(
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
            "embedding_type": "text" if not search_request.price_aware else "price_aware",
            "model": default_model
        }
    
    return router