"""
FastAPI application for quantum-enhanced reranking in RAG pipelines
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging

from src.reranker.controller import RerankerController
from src.reranker.classical import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Pydantic models for API
class DocumentRequest(BaseModel):
    id: str
    content: str
    source: Optional[str] = None


class RerankRequest(BaseModel):
    query: str
    documents: List[DocumentRequest]
    reranker_type: Optional[str] = "auto"  # "quantum", "classical", or "auto"
    top_k: Optional[int] = 5


# Create FastAPI app
app = FastAPI(
    title="Quantum RAG Reranker",
    description="API for quantum-enhanced reranking of documents for podcast ad detection",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize reranker controller
reranker_controller = RerankerController()


@app.post("/rerank")
async def rerank_documents(request: RerankRequest):
    """Rerank documents based on query relevance using quantum or classical methods."""
    try:
        # Convert request documents to Document objects
        documents = [
            Document(doc.id, doc.content, doc.source) for doc in request.documents
        ]

        # Perform reranking
        result = reranker_controller.rerank(
            query=request.query,
            documents=documents,
            top_k=request.top_k,
            reranker_type=request.reranker_type,
        )

        return result

    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        return {"error": str(e)}


# Add a simple root endpoint
@app.get("/")
async def root():
    """Root endpoint that provides basic information about the API."""
    return {
        "message": "Quantum RAG Reranker API",
        "docs_url": "/docs",
        "version": "0.1.0",
        "use_case": "Podcast advertisement detection",
        "endpoints": {
            "rerank": "POST /rerank - Rerank documents using quantum or classical methods"
        },
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
