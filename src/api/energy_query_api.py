"""
FastAPI endpoints for querying ERCOT energy data using quantum-enhanced reranking.
This API provides endpoints for:
1. Semantic search on energy data using vector embeddings
2. Comparing classical vs quantum reranking of energy forecast analysis
"""
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple # Ensured Tuple is imported
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import logging

from src.reranker.classical import Document as RerankerDocument
from src.reranker.quantum import QuantumReranker
from src.reranker.controller import RerankerController
from src.storage.dual_storage import DualStorage # Assuming src.storage.dual_storage is resolvable at runtime
from src.embeddings.embed_utils import get_embedding_provider, EmbeddingProvider
import os

# Initialize logger early
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) 

# Define schemas
class Document(BaseModel):
    """Schema for energy document"""
    id: str
    content: str
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EnergyQueryRequest(BaseModel):
    """Request schema for energy data queries"""
    query: str = Field(..., description="Search query for energy forecasts")
    limit: int = Field(10, description="Number of results to return", ge=1, le=100)
    reranker_type: str = Field("auto", description="Reranker type: 'classical', 'quantum', or 'auto'")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "forecast vs actual load during summer peak hours",
                "limit": 5,
                "reranker_type": "auto"
            }
        }

class QueryResult(BaseModel):
    """Single query result"""
    document: Document
    score: float
    reranker_used: str

class EnergyQueryResponse(BaseModel):
    """Response schema for energy data queries"""
    query: str
    results: List[QueryResult]
    reranker_used: str
    # Add a field for detailed_error for debugging if needed
    # detailed_error: Optional[str] = None 

# Create router
router = APIRouter(
    prefix="" # Consider adding a prefix like "/api/v1"
)

# Initialize DualStorage and EmbeddingProvider
dual_storage_manager: Optional[DualStorage] = None
embed_provider: Optional[EmbeddingProvider] = None

try:
    dual_storage_manager = DualStorage()
    embed_provider = get_embedding_provider()
    logger.info("DualStorage and EmbeddingProvider initialized successfully.")
except Exception as e:
    logger.critical(f"CRITICAL: Error initializing DualStorage or EmbeddingProvider: {e}", exc_info=True)
    # dual_storage_manager and embed_provider remain None, caught by endpoint check


@router.post("/query", response_model=EnergyQueryResponse)
async def query_energy_data(request: EnergyQueryRequest):
    logger.info(f"Received query request: {request.model_dump_json()}")

    if not dual_storage_manager or not embed_provider:
        logger.error("DualStorage or EmbeddingProvider not initialized. Check startup logs.")
        raise HTTPException(status_code=503, detail="Core services (database/embedding) not available.")

    controller = RerankerController({
        "classical_config": {"method": "cosine"},
        "quantum_config": {"method": "state_fidelity", "n_qubits": 4}
    })
    
    try:
        query_embedding_array = embed_provider.get_embeddings([request.query])
        if not isinstance(query_embedding_array, np.ndarray) or query_embedding_array.ndim == 0 or query_embedding_array.size == 0:
            logger.error("Failed to generate query embedding: Result is not a valid numpy array or is empty.")
            raise HTTPException(status_code=500, detail="Failed to generate query embedding.")
        query_embedding = query_embedding_array[0]
        logger.info(f"Successfully generated query embedding for query: {request.query}")
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating query embedding: {e}")

    try:
        similar_docs_with_metadata = dual_storage_manager.find_similar_documents(
            query_embedding=query_embedding,
            top_k=request.limit * 2, 
            metric="cosine" 
        )
        logger.info(f"Retrieved {len(similar_docs_with_metadata)} documents with metadata via DualStorage for query: {request.query}")
        
        retrieved_documents: List[RerankerDocument] = []
        for doc_info in similar_docs_with_metadata:
            metadata = doc_info.get("metadata", {})
            content = metadata.get("content") or metadata.get("summary", f"Content for {doc_info.get('document_id', 'N/A')}")
            if not content:
                 logger.warning(f"Document ID {doc_info.get('document_id')} missing 'content' and 'summary' field in metadata, using placeholder.")
                 content = f"Placeholder content for document {doc_info.get('document_id', 'N/A')}"

            retrieved_documents.append(RerankerDocument(
                id=str(doc_info.get("document_id", doc_info.get("vector_id"))),
                content=content,
                source=metadata.get("source", "DualStorage"), 
                metadata=metadata
            ))
        
        logger.info(f"Processed {len(retrieved_documents)} documents for reranking.")
        
        if not retrieved_documents:
            logger.warning(f"No documents found for query: {request.query} after initial retrieval and processing.")
            return EnergyQueryResponse(
                query=request.query,
                results=[],
                reranker_used="N/A - No documents found"
            )

    except Exception as e:
        logger.error(f"Error fetching/processing documents via DualStorage: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching or processing documents: {e}")
    
    reranked_results_tuples: List[Tuple[RerankerDocument, float]] = [] 
    reranker_name = "unknown"

    logger.info(f"Using reranker type: {request.reranker_type}")
    if request.reranker_type == "classical":
        reranked_results_tuples = controller.classical_reranker.rerank(request.query, retrieved_documents, top_k=request.limit)
        reranker_name = "classical"
    elif request.reranker_type == "quantum":
        reranked_results_tuples = controller.quantum_reranker.rerank(request.query, retrieved_documents, top_k=request.limit)
        reranker_name = "quantum"
    else:  # Auto mode
        result = controller.rerank(request.query, retrieved_documents, top_k=request.limit)
        if isinstance(result.get("documents"), list) and all(isinstance(item, tuple) and len(item) == 2 for item in result["documents"]):
            reranked_results_tuples = result["documents"]
        else:
            logger.error(f"Auto reranker returned unexpected document structure: {result.get('documents')}")
            reranked_results_tuples = [] 
        reranker_name = result.get("reranker_used", "unknown") + " (auto-selected)"
    
    logger.info(f"Reranking complete. Reranker used: {reranker_name}. Number of results from reranker: {len(reranked_results_tuples)}")
    
    results = []
    for doc_tuple in reranked_results_tuples: 
        if isinstance(doc_tuple, tuple) and len(doc_tuple) == 2:
            doc, score = doc_tuple
            if isinstance(doc, RerankerDocument) and isinstance(score, (float, int)):
                api_doc = Document(id=doc.id, content=doc.content, source=doc.source, metadata=doc.metadata)
                results.append(QueryResult(document=api_doc, score=float(score), reranker_used=reranker_name))
            else:
                logger.warning(f"Skipping malformed reranked item: Type mismatch - Document: {type(doc)}, Score: {type(score)}")
        else:
            logger.warning(f"Skipping malformed reranked tuple: Expected (Document, float), got {type(doc_tuple)}")
            
    response = EnergyQueryResponse(
        query=request.query,
        results=results,
        reranker_used=reranker_name
    )
    logger.info(f"Returning response for query: {request.query}. Number of results: {len(results)}")
    return response

@router.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "quantum-energy-rag"}

