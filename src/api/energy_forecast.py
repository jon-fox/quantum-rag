"""
FastAPI endpoint for energy forecasting using retrieval and classical reranking.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
import os
import logging

from src.storage.pgvector_storage import PgVectorStorage
from src.reranker.classical import ClassicalReranker, Document
from src.embeddings.embed_utils import get_embedding_provider
from src.prompts.energy_forecast import build_energy_forecast_prompt

logger = logging.getLogger(__name__)

# Request/Response schemas
class ForecastRequest(BaseModel):
    query: str = Field(..., description="Natural language query for energy forecasting analysis")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "difference between DAM forecast and telemetry generation in May and June 2025 during afternoon peak"
            }
        }

class ForecastResponse(BaseModel):
    forecast: str

# Create router
router = APIRouter()

# Initialize services
pg_storage = PgVectorStorage(app_environment=os.environ.get("APP_ENVIRONMENT", "prod"))
embed_provider = get_embedding_provider()
classical_reranker = ClassicalReranker()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@router.post("/", response_model=ForecastResponse)
async def generate_energy_forecast(request: ForecastRequest):
    """Generate energy forecast using retrieval, reranking, and LLM analysis."""
    
    if not pg_storage or not embed_provider:
        raise HTTPException(status_code=503, detail="Core services not available")
    
    try:
        # Generate query embedding
        query_embedding = embed_provider.get_embeddings([request.query])[0]
        
        # Retrieve ~100 documents
        similar_docs = pg_storage.find_similar_documents(
            query_embedding=query_embedding,
            top_k=100,
            metric="cosine"
        )
        
        # Convert to Document format for reranking
        documents = []
        for doc_info in similar_docs:
            metadata = doc_info.get("metadata", {})
            content = metadata.get("content") or metadata.get("semantic_sentence", "")
            if content:
                documents.append(Document(
                    id=str(doc_info.get("vector_id", "")),
                    content=content,
                    source=metadata.get("source", "PostgreSQL"),
                    metadata=metadata
                ))
        
        # Rerank and get top 5
        reranked_results = classical_reranker.rerank(request.query, documents, top_k=5)
        
        # Build prompt using the template
        prompt = build_energy_forecast_prompt(request.query, reranked_results)
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        
        forecast_text = response.choices[0].message.content
        if forecast_text:
            forecast_text = forecast_text.strip()
        else:
            forecast_text = "Unable to generate forecast"
        
        return ForecastResponse(forecast=forecast_text)
        
    except Exception as e:
        logger.error(f"Error generating forecast: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {e}")
