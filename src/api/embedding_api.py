"""
Embedding API endpoints for semantic search of energy consumption data and efficiency metrics.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging

from src.embeddings.embed_utils import (
    get_embedding_provider,
    OpenAIEmbedding,
    embed_query,
)
# Assuming these imports are resolvable in the project environment
from src.config.env_manager import get_env_var 
from src.storage.pgvector_storage import PgVectorStorage

# Configure logger for this module
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class SearchQuery(BaseModel):
    """Search query model"""
    query: str
    top_k: int = 5
    min_similarity: float = 0.6


class EmbeddingResponse(BaseModel):
    """Response model for embedding requests"""
    vector: List[float]
    dimension: int
    model: str


class SearchResult(BaseModel):
    """Single search result"""
    item: Dict[str, Any] # Will now be e.g. {"vector_id": "some-uuid"}
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
    
    try:
        openai_provider = OpenAIEmbedding()
        default_provider = openai_provider
        default_model = openai_provider.model_name
    except (ImportError, ValueError): 
        default_provider = get_embedding_provider()
        default_model = getattr(default_provider, 'model_name', 'unknown')

    pg_vector_storage_instance = None # Initialize to None
    try:
        # Attempt to gather specific PGVECTOR_* variables
        pg_host = get_env_var("PGVECTOR_HOST")
        pg_port = get_env_var("PGVECTOR_PORT")
        pg_dbname = get_env_var("PGVECTOR_DB_NAME")
        pg_user = get_env_var("PGVECTOR_USER")
        pg_password = get_env_var("PGVECTOR_PASSWORD") # Can be None or empty string

        required_vars_present = all([pg_host, pg_port, pg_dbname, pg_user])

        db_params_for_pg: Optional[Dict[str, str]] = None
        if required_vars_present:
            db_params_for_pg = {
                "host": str(pg_host),
                "port": str(pg_port),
                "dbname": str(pg_dbname),
                "user": str(pg_user),
            }
            if pg_password is not None: 
                db_params_for_pg["password"] = str(pg_password)
            logger.info(f"Using specific PGVECTOR_* env vars for PgVectorStorage: host={db_params_for_pg['host']}, dbname={db_params_for_pg['dbname']}")
        else:
            logger.info("Not all specific PGVECTOR_* env vars found. PgVectorStorage will use its default loading (SSM/general DB_* env vars).")

        pg_vector_storage_instance = PgVectorStorage(db_params=db_params_for_pg) # Renamed variable
        
        logger.info("PgVectorStorage initialized successfully for embedding_api.")
    except Exception as e:
        logger.error(f"Failed to initialize PgVectorStorage in embedding_api: {e}")
        # pg_vector_storage_instance remains None. Endpoints relying on it will fail gracefully.

    @router.get("/info", response_model=Dict[str, Any])
    async def get_embedding_info():
        """Get information about the embedding service"""
        return {
            "provider_type": default_provider.__class__.__name__,
            "embedding_dimension": default_provider.embedding_dim,
            "model": default_model,
        }
    
    @router.post("/create", response_model=EmbeddingResponse)
    async def create_embedding(query: str = Query(..., description="Text to embed")):
        """Generate an embedding for the provided text"""
        embedding_np = embed_query(query, default_provider)
        
        return {
            "vector": embedding_np.tolist(),
            "dimension": len(embedding_np),
            "model": default_model
        }
    
    @router.post("/search", response_model=SearchResponse)
    async def search_embeddings(search_request: SearchQuery):
        """
        Search for similar embeddings using PgVectorStorage.
        """
        if pg_vector_storage_instance is None:
            logger.error("PgVectorStorage not initialized. Cannot perform search in embedding_api.")
            raise HTTPException(status_code=503, detail="Search service is unavailable due to storage initialization failure.")

        query_embedding_np = embed_query(search_request.query, default_provider)

        try:
            # pg_vector_storage.find_similar_embeddings returns List[Tuple[vector_id, distance]]
            # Assuming default metric 'cosine', so distance = 1 - similarity
            retrieved_vectors: List[Tuple[str, float]] = pg_vector_storage_instance.find_similar_embeddings(
                query_embedding=query_embedding_np,
                top_k=search_request.top_k * 3, # Fetch more for min_similarity filtering
            )
        except Exception as e:
            logger.error(f"Error during pg_vector_storage.find_similar_embeddings in embedding_api: {e}")
            raise HTTPException(status_code=500, detail="Error searching documents")

        results: List[SearchResult] = []
        for vector_id, distance in retrieved_vectors:
            similarity = 1.0 - distance # Convert cosine distance to similarity

            if similarity < search_request.min_similarity:
                continue

            results.append(SearchResult(item={"vector_id": vector_id}, similarity=similarity))
            if len(results) >= search_request.top_k:
                break
        
        return SearchResponse(
            query=search_request.query,
            results=results,
            count=len(results),
            embedding_type="text",
            model=default_model
        )
    
    return router