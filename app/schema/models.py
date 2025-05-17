"""
Application Schema Definitions
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class Document(BaseModel):
    """Document schema for retrieval and storage"""
    id: str = Field(..., description="Unique identifier for the document")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    source: Optional[str] = Field(None, description="Document source identifier")
    timestamp: Optional[datetime] = Field(None, description="Document timestamp")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of document content")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "doc123",
                "content": "ERCOT forecast shows a 15% increase in energy demand for summer months.",
                "metadata": {"category": "forecast", "region": "texas"},
                "source": "ercot_report_2025",
                "timestamp": "2025-05-01T12:00:00Z"
            }
        }

class Query(BaseModel):
    """User query schema"""
    text: str = Field(..., description="Query text")
    use_quantum: bool = Field(False, description="Whether to use quantum reranking")
    top_k: Optional[int] = Field(5, description="Number of results to return")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "summer energy demand forecast for 2025",
                "use_quantum": True,
                "top_k": 5
            }
        }

class SearchResponse(BaseModel):
    """Search response schema"""
    query: str = Field(..., description="Original query")
    documents: List[Document] = Field(default_factory=list, description="Retrieved documents")
    reranker_used: str = Field("classical", description="Type of reranker used")
    execution_time_ms: Optional[float] = Field(None, description="Query execution time in milliseconds")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "summer energy demand forecast for 2025",
                "documents": [
                    {
                        "id": "doc123",
                        "content": "ERCOT forecast shows a 15% increase in energy demand for summer months.",
                        "metadata": {"category": "forecast", "region": "texas"},
                        "source": "ercot_report_2025",
                        "timestamp": "2025-05-01T12:00:00Z"
                    }
                ],
                "reranker_used": "quantum",
                "execution_time_ms": 152.4
            }
        }
