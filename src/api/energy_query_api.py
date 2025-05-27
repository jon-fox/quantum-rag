"""
FastAPI endpoints for querying ERCOT energy data using quantum-enhanced reranking.
This API provides endpoints for:
1. Semantic search on energy data using vector embeddings
2. Comparing classical vs quantum reranking of energy forecast analysis
"""
import json
import numpy as np
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from src.reranker.classical import Document as RerankerDocument
from src.reranker.quantum import QuantumReranker
from src.reranker.controller import RerankerController
from src.storage.pgvector_storage import PgVectorStorage # Added import
from src.embeddings.embed_utils import get_embedding_provider, EmbeddingProvider # Added import
import os # Added import for environment variables

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

# Create router
router = APIRouter(
    prefix=""
)

# Initialize PgVectorStorage and EmbeddingProvider
# These should ideally be managed with FastAPI's lifespan events or dependency injection for production
try:
    pg_storage = PgVectorStorage(app_environment=os.environ.get("APP_ENVIRONMENT", "prod"))
    embed_provider = get_embedding_provider()
except Exception as e:
    # Log this error appropriately in a real application
    print(f"Error initializing PgVectorStorage or EmbeddingProvider: {e}")
    pg_storage = None
    embed_provider = None


@router.post("/query", response_model=EnergyQueryResponse)
async def query_energy_data(request: EnergyQueryRequest):
    """
    Query ERCOT energy data using semantic search and quantum-enhanced reranking.
    
    The endpoint supports three reranker modes:
    - classical: Uses traditional similarity metrics only
    - quantum: Uses quantum circuit-based similarity
    - auto: Intelligently selects the best reranker for the query
    """
    if not pg_storage or not embed_provider:
        raise HTTPException(status_code=503, detail="Database or embedding service not available.")

    # Initialize controller that manages both rerankers
    controller = RerankerController({
        "classical_config": {"method": "cosine"},
        "quantum_config": {"method": "state_fidelity", "n_qubits": 4}
    })
    
    # 1. Generate embedding for the input query
    try:
        query_embedding_list = embed_provider.get_embeddings([request.query])
        if not query_embedding_list:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding.")
        query_embedding = query_embedding_list[0]
    except Exception as e:
        # Log error
        raise HTTPException(status_code=500, detail=f"Error generating query embedding: {e}")

    # 2. Find similar documents from PgVector
    try:
        # Assuming find_similar_embeddings returns (id, score)
        # We need to fetch full document content based on these IDs.
        # This part needs a way to get document content by ID.
        # For now, let's assume find_similar_embeddings could be extended or
        # we have another method to fetch documents.
        # Let's simulate fetching documents for now, focusing on the pg_storage call.
        
        similar_docs_from_db = pg_storage.find_similar_embeddings(
            query_embedding=query_embedding,
            top_k=request.limit * 2, # Fetch more to allow reranking
            metric="cosine" 
        )
        
        # Mocking document retrieval based on IDs from pg_storage
        # In a real scenario, you'd query your main database for content using these IDs.
        retrieved_documents: List[RerankerDocument] = []
        for doc_id, _ in similar_docs_from_db:
            # This is a placeholder. You need to implement fetching document content by ID.
            # For example, query another table or service.
            # Here, we'll create dummy RerankerDocument objects.
            # If your pg_storage also stores content or can join to get it, that would be ideal.
            retrieved_documents.append(RerankerDocument(
                id=str(doc_id), 
                content=f"Content for {doc_id}", # Placeholder content
                source="Database", # Placeholder source
                metadata={"retrieved_from_pgvector": True} # Placeholder metadata
            ))
        
        if not retrieved_documents:
            # Handle case where no documents are found
             return EnergyQueryResponse(
                query=request.query,
                results=[],
                reranker_used="N/A - No documents found"
            )

    except Exception as e:
        # Log error
        raise HTTPException(status_code=500, detail=f"Error fetching documents from database: {e}")
    
    # Use explicit reranker type if specified, otherwise let controller decide
    if request.reranker_type == "classical":
        reranked_docs_tuples = controller.classical_reranker.rerank(request.query, retrieved_documents, top_k=request.limit)
        reranker_name = "classical"
    elif request.reranker_type == "quantum":
        reranked_docs_tuples = controller.quantum_reranker.rerank(request.query, retrieved_documents, top_k=request.limit)
        reranker_name = "quantum"
    else:  # Auto mode
        # The controller.rerank method expects a list of RerankerDocument objects
        result = controller.rerank(request.query, retrieved_documents, top_k=request.limit)
        reranked_docs_tuples = result["documents"] # This should be a list of (RerankerDocument, score)
        reranker_name = result["reranker_used"] + " (auto-selected)"
    
    # Convert to response format
    results = []
    # The rerank methods in classical_reranker and quantum_reranker return List[Tuple[RerankerDocument, float]]
    # The controller.rerank also returns a similar structure under result["documents"]
    for doc_tuple in reranked_docs_tuples:
        doc, score = doc_tuple # Unpack the tuple
        api_doc = Document(id=doc.id, content=doc.content, source=doc.source, metadata=doc.metadata)
        results.append(QueryResult(document=api_doc, score=score, reranker_used=reranker_name))
    
    return EnergyQueryResponse(
        query=request.query,
        results=results,
        reranker_used=reranker_name
    )

@router.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "quantum-energy-rag"}

