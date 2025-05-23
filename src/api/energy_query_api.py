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

@router.post("/query", response_model=EnergyQueryResponse)
async def query_energy_data(request: EnergyQueryRequest):
    """
    Query ERCOT energy data using semantic search and quantum-enhanced reranking.
    
    The endpoint supports three reranker modes:
    - classical: Uses traditional similarity metrics only
    - quantum: Uses quantum circuit-based similarity
    - auto: Intelligently selects the best reranker for the query
    """
    # Initialize controller that manages both rerankers
    controller = RerankerController({
        "classical_config": {"method": "cosine"},
        "quantum_config": {"method": "state_fidelity", "n_qubits": 4}
    })
    
    # Mock documents for demo (would come from database in production)
    mock_documents: List[RerankerDocument] = [
        RerankerDocument(
            id="ercot-2024-05-01",
            content="ERCOT forecasted peak load for May 1, 2024: 58.2 GW. Actual load: 57.8 GW.",
            source="ERCOT Daily Report",
            metadata={"date": "2024-05-01", "type": "load_forecast"}
        ),
        RerankerDocument(
            id="ercot-2024-05-02",
            content="ERCOT forecasted peak load for May 2, 2024: 59.1 GW. Actual load: 60.3 GW.",
            source="ERCOT Daily Report",
            metadata={"date": "2024-05-02", "type": "load_forecast"}
        ),
        # Additional documents would be retrieved from database in production
    ]
    
    # Use explicit reranker type if specified, otherwise let controller decide
    if request.reranker_type == "classical":
        reranked_docs = controller.classical_reranker.rerank(request.query, mock_documents, top_k=request.limit)
        reranker_name = "classical"
    elif request.reranker_type == "quantum":
        reranked_docs = controller.quantum_reranker.rerank(request.query, mock_documents, top_k=request.limit)
        reranker_name = "quantum"
    else:  # Auto mode
        result = controller.rerank(request.query, mock_documents, top_k=request.limit)
        reranked_docs = result["documents"]
        reranker_name = result["reranker_used"] + " (auto-selected)"
    
    # Convert to response format
    results = []
    for i, doc in enumerate(reranked_docs):
        # In a real implementation, scores would come from the reranker
        score = 1.0 - (i * 0.1)  # Placeholder scoring logic
        # Create a new Document instance for the QueryResult
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

