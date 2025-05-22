import os
import logging
import sys
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Adjust sys.path to allow imports from the 'src' directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.storage.pgvector_storage import PgVectorStorage
    from src.embeddings.embed_utils import get_embedding_provider, EmbeddingProvider
except ImportError as e:
    logging.error(f"Error importing necessary modules: {e}")
    logging.error("Please ensure the script is run from a context where 'src' is discoverable, or adjust PYTHONPATH.")
    sys.exit(1) # Exit if critical imports fail

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Direct Module-Level Initialization ---
# Instances are created when this module is imported.
# If initialization fails, an error will be raised during import.
try:
    logger.info("Attempting direct initialization of PgVectorStorage at module level...")
    pg_storage = PgVectorStorage(
        app_environment=os.environ.get("APP_ENVIRONMENT", "prod") # Changed from dev to prod to match PgVectorStorage default
    )
    logger.info("PgVectorStorage initialized directly at module level.")
except Exception as e:
    logger.error(f"CRITICAL: Failed to initialize PgVectorStorage at module level: {e}", exc_info=True)
    raise RuntimeError(f"CRITICAL_ERROR_PGVECTOR_INIT: {e}") from e

try:
    logger.info("Attempting direct initialization of EmbeddingProvider at module level...")
    embed_provider = get_embedding_provider()
    logger.info(f"EmbeddingProvider initialized directly at module level. Type: {type(embed_provider).__name__}")
except Exception as e:
    logger.error(f"CRITICAL: Failed to initialize EmbeddingProvider at module level: {e}", exc_info=True)
    raise RuntimeError(f"CRITICAL_ERROR_EMBEDDING_PROVIDER_INIT: {e}") from e


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

class DatabaseListResponse(BaseModel):
    databases: List[str]

class TableListResponse(BaseModel):
    tables: List[str]

class ExecuteQueryRequest(BaseModel):
    query: str

class ExecuteQueryResponse(BaseModel):
    columns: List[str]
    rows: List[List[str]]
    row_count: int
    error: Optional[str] = None

# --- APIRouter ---
# Changed from FastAPI app to APIRouter
router = APIRouter()

# --- API Endpoints ---
@router.post("/find_similar_documents", response_model=SimilarityResponse)
async def find_similar_documents_api(query: SimilarityQuery): # Removed Depends
    if not pg_storage or not embed_provider:
        raise HTTPException(status_code=503, detail="A critical service (DB or Embeddings) is not available.")
    if query.top_k <= 0:
        raise HTTPException(status_code=400, detail="'top_k' must be a positive integer")
    try:
        logger.info(f"Received query: '{query.query_text}', top_k: {query.top_k}")
        query_embedding_list = embed_provider.get_embeddings([query.query_text])
        if not query_embedding_list:
            logger.error("Failed to generate embedding for query text: empty list returned.")
            raise HTTPException(status_code=500, detail="Failed to generate query embedding.")
        query_embedding = query_embedding_list[0]
        logger.info(f"Generated query embedding. Shape: {getattr(query_embedding, 'shape', 'N/A')}")
        similar_results = pg_storage.find_similar_embeddings(
            query_embedding=query_embedding,
            top_k=query.top_k,
            metric="cosine"
        )
        logger.info(f"Found {len(similar_results)} similar results from database.")
        return SimilarityResponse(
            query_text=query.query_text,
            top_k=query.top_k,
            similar_documents=[
                SimilarDocument(vector_id=str(vec_id), distance_score=float(score))
                for vec_id, score in similar_results
            ]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing /find_similar_documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred processing your request.")

@router.get("/health", response_model=HealthResponse)
async def health_check():
    db_healthy = pg_storage is not None 
    embed_healthy = embed_provider is not None
    
    if db_healthy and embed_healthy:
        # To be more thorough, attempt a lightweight DB operation for db_healthy
        db_truly_healthy = False
        if pg_storage:
            try:
                pg_storage.list_databases() 
                db_truly_healthy = True
            except Exception as e:
                logger.warning(f"Health check: PgVectorStorage seems initialized but a test operation failed: {e}")
                db_truly_healthy = False
        
        if db_truly_healthy and embed_healthy:
            return HealthResponse(status="healthy")
        else:
            details = []
            if not db_truly_healthy: details.append("PgVectorStorage: initialized but unhealthy or test operation failed")
            if not embed_healthy: details.append("EmbeddingProvider: uninitialized (should not happen if app started)") # Should be caught at startup
            status_detail = "; ".join(details)
            logger.warning(f"Health check status: unhealthy ({status_detail})")
            return HealthResponse(status=f"unhealthy ({status_detail})")

    else: # This case should ideally not be reached if init failures are fatal
        details = []
        if not db_healthy: details.append("PgVectorStorage: FAILED_TO_INITIALIZE_AT_MODULE_LEVEL")
        if not embed_healthy: details.append("EmbeddingProvider: FAILED_TO_INITIALIZE_AT_MODULE_LEVEL")
        status_detail = "; ".join(details)
        logger.error(f"CRITICAL HEALTH CHECK FAILURE: {status_detail}")
        return HealthResponse(status=f"CRITICALLY_UNHEALTHY ({status_detail})")


@router.get("/databases", response_model=DatabaseListResponse)
async def list_databases_api(): # Removed Depends
    if not pg_storage:
        raise HTTPException(status_code=503, detail="PgVectorStorage is not available.")
    try:
        databases = pg_storage.list_databases()
        return DatabaseListResponse(databases=databases)
    except Exception as e:
        logger.error(f"Error listing databases: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list databases: {str(e)}")

@router.get("/tables", response_model=TableListResponse)
async def list_tables_api(): # Removed Depends
    if not pg_storage:
        raise HTTPException(status_code=503, detail="PgVectorStorage is not available.")
    try:
        tables = pg_storage.list_tables()
        return TableListResponse(tables=tables)
    except Exception as e:
        logger.error(f"Error listing tables: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list tables: {str(e)}")

@router.post("/execute_query", response_model=ExecuteQueryResponse)
async def execute_query_api(request: ExecuteQueryRequest): # Removed Depends
    if not pg_storage:
        raise HTTPException(status_code=503, detail="PgVectorStorage is not available.")
    if not request.query.strip().upper().startswith("SELECT"):
        raise HTTPException(status_code=400, detail="Only SELECT queries are allowed.")
    try:
        column_names, result_rows, error_msg = pg_storage.execute_select_query(request.query)
        if error_msg:
            logger.warning(f"Query execution failed with error from storage: {error_msg}. Query: {request.query[:200]}")
            return ExecuteQueryResponse(columns=[], rows=[], row_count=0, error=error_msg)
        
        processed_rows = []
        if result_rows:
            processed_rows = [list(map(str, r)) for r in result_rows]
        
        final_columns = column_names if column_names is not None else []
        
        return ExecuteQueryResponse(
            columns=final_columns,
            rows=processed_rows,
            row_count=len(processed_rows),
            error=None
        )
    except ValueError as ve:
        logger.warning(f"Query execution denied or failed API-level validation: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error in /execute_query endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

