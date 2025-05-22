import os
import logging
import sys
from typing import List, Optional # Added Optional, List

import numpy as np # Kept for query_embedding.shape
from fastapi import FastAPI, HTTPException, Depends # Changed from Flask
from pydantic import BaseModel # For request/response models
import uvicorn # For running the app

# Adjust sys.path to allow imports from the 'src' directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..')) # Adds quantum_work to sys.path

try:
    from src.storage.pgvector_storage import PgVectorStorage
    from src.embeddings.embed_utils import get_embedding_provider, EmbeddingProvider
except ImportError as e:
    logging.error(f"Error importing necessary modules: {e}")
    logging.error("Please ensure the script is run from a context where 'src' is discoverable, or adjust PYTHONPATH.")
    sys.exit(1)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class SimilarityQuery(BaseModel):
    query_text: str
    top_k: int = 5

class SimilarDocument(BaseModel):
    vector_id: str
    distance_score: float

class SimilarityResponse(BaseModel):
    query_text: str
    top_k: int
    similar_documents: List[SimilarDocument]

class HealthResponse(BaseModel):
    status: str

# --- FastAPI app and Global instances ---
app = FastAPI(title="PgVector Query API", version="1.0.0")

# Global variables to hold instances, initialized at startup
pg_vector_storage_instance: Optional[PgVectorStorage] = None
embedding_provider_instance: Optional[EmbeddingProvider] = None

# --- Lifespan Events (Startup/Shutdown) ---
@app.on_event("startup")
async def startup_event():
    global pg_vector_storage_instance, embedding_provider_instance
    logger.info("Application startup...")
    try:
        logger.info("Initializing PgVectorStorage...")
        # PgVectorStorage will attempt to load DB params from SSM or .env
        # APP_ENVIRONMENT env var can be set to 'dev', 'prod', etc.
        pg_vector_storage_instance = PgVectorStorage(
            app_environment=os.environ.get("APP_ENVIRONMENT", "dev")
        )
        logger.info("PgVectorStorage initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize PgVectorStorage during startup: {e}", exc_info=True)
        raise RuntimeError(f"Could not initialize PgVectorStorage: {e}")

    try:
        logger.info("Initializing EmbeddingProvider...")
        # get_embedding_provider will use EMBEDDING_PROVIDER env var or defaults
        embedding_provider_instance = get_embedding_provider()
        logger.info(f"EmbeddingProvider initialized with type: {type(embedding_provider_instance).__name__}")
    except Exception as e:
        logger.error(f"Failed to initialize EmbeddingProvider during startup: {e}", exc_info=True)
        raise RuntimeError(f"Could not initialize EmbeddingProvider: {e}")
    logger.info("Application startup completed.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown...")
    global pg_vector_storage_instance, embedding_provider_instance
    if pg_vector_storage_instance:
        try:
            logger.info("Closing PgVectorStorage connection...")
            pg_vector_storage_instance.close_db_connection() # Use the new method
            logger.info("PgVectorStorage connection closed.")
        except Exception as e:
            logger.error(f"Error closing PgVectorStorage connection: {e}", exc_info=True)
    
    pg_vector_storage_instance = None
    embedding_provider_instance = None # Clear instance
    logger.info("Application shutdown completed.")

# --- Dependencies ---
# These dependencies will provide the initialized instances to the path operations.
# They rely on the startup event to have initialized the global instances.

def get_pg_storage_dependency() -> PgVectorStorage:
    if pg_vector_storage_instance is None:
        logger.error("PgVectorStorage instance is not available. Startup might have failed.")
        raise HTTPException(status_code=503, detail="Service not available: PgVectorStorage not initialized.")
    return pg_vector_storage_instance

def get_embedding_provider_dependency() -> EmbeddingProvider:
    if embedding_provider_instance is None:
        logger.error("EmbeddingProvider instance is not available. Startup might have failed.")
        raise HTTPException(status_code=503, detail="Service not available: EmbeddingProvider not initialized.")
    return embedding_provider_instance

# --- API Endpoints ---
@app.post("/find_similar_documents", response_model=SimilarityResponse)
async def find_similar_documents_api(
    query: SimilarityQuery,
    storage: PgVectorStorage = Depends(get_pg_storage_dependency),
    embed_provider: EmbeddingProvider = Depends(get_embedding_provider_dependency)
):
    # Pydantic handles query_text presence and top_k type.
    # Additional validation for top_k's value:
    if query.top_k <= 0:
        raise HTTPException(status_code=400, detail="'top_k' must be a positive integer")

    try:
        logger.info(f"Received query: '{query.query_text}', top_k: {query.top_k}")

        # 1. Generate embedding for the query text
        query_embedding_list = embed_provider.get_embeddings([query.query_text])
        if not query_embedding_list: # Should not happen if get_embeddings is robust
            logger.error("Failed to generate embedding for query text: empty list returned.")
            raise HTTPException(status_code=500, detail="Failed to generate query embedding.")
        query_embedding = query_embedding_list[0]
        
        logger.info(f"Generated query embedding. Shape: {getattr(query_embedding, 'shape', 'N/A')}")

        # 2. Find similar embeddings using PgVectorStorage
        similar_results = storage.find_similar_embeddings(
            query_embedding=query_embedding,
            top_k=query.top_k,
            metric="cosine" # Or "l2", "inner_product" depending on your setup
        )
        logger.info(f"Found {len(similar_results)} similar results from database.")

        return SimilarityResponse(
            query_text=query.query_text,
            top_k=query.top_k,
            similar_documents=[
                SimilarDocument(vector_id=str(vec_id), distance_score=float(score)) # Ensure types
                for vec_id, score in similar_results
            ]
        )
    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        logger.error(f"Error processing /find_similar_documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred processing your request.")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    db_healthy = pg_vector_storage_instance is not None
    embed_healthy = embedding_provider_instance is not None
    
    # More sophisticated checks can be added here (e.g., pinging the DB)
    # For example:
    # if db_healthy:
    #     try:
    #         # pg_vector_storage_instance.ping_db() # Hypothetical method
    #     except Exception:
    #         db_healthy = False
    #         logger.warning("Health check: DB ping failed.")

    if db_healthy and embed_healthy:
        return HealthResponse(status="healthy")
    else:
        details = []
        if not db_healthy: details.append("PgVectorStorage: uninitialized or unhealthy")
        if not embed_healthy: details.append("EmbeddingProvider: uninitialized or unhealthy")
        status_detail = "; ".join(details)
        logger.warning(f"Health check status: unhealthy ({status_detail})")
        # To return a 503 status for unhealthy, you would raise HTTPException here:
        # raise HTTPException(status_code=503, detail=f"Service unhealthy: {status_detail}")
        # For now, returning 200 with unhealthy status in body:
        return HealthResponse(status=f"unhealthy ({status_detail})")

if __name__ == '__main__':
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "5001"))
    
    # For development, reload=True is useful. It watches for file changes.
    # For production, set reload=False and consider using Gunicorn with Uvicorn workers.
    # Example: uvicorn.run("your_module:app", host="0.0.0.0", port=80, workers=4)
    uvicorn.run("pgvector_query_api:app", host=host, port=port, reload=True, log_level="info")

