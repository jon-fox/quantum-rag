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
from src.prompts.builders import build_prompt
from src.query_intent import QueryIntentClassifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # enable debug logging for this module

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
intent_classifier = QueryIntentClassifier()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@router.post("/", response_model=ForecastResponse)
async def generate_energy_forecast(request: ForecastRequest):
    """Generate energy forecast using retrieval, reranking, and LLM analysis."""
    
    if not pg_storage or not embed_provider:
        raise HTTPException(status_code=503, detail="Core services not available")
    
    try:
        # Classify query intent and get retrieval strategy
        strategy = intent_classifier.classify_and_get_strategy(request.query)
        logger.info(f"Query intent classified: {strategy['description']} - retrieving {strategy['num_documents']} documents")
        
        # Generate query embedding
        query_embedding = embed_provider.get_embeddings([request.query])[0]
        
        # Retrieve documents using intent-based strategy
        similar_docs = pg_storage.find_similar_documents(
            query_embedding=query_embedding,
            top_k=strategy['num_documents'],  # Dynamic document count based on intent
            metric="cosine",
            intent_filters=strategy.get('query_filters'),
            sort_strategy=strategy.get('sort_strategy')
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

        logger.info("Printing retrieved documents for debugging (full documents):")
        for doc in documents:
            logger.debug(
                "Document ID: %s\nContent: %s\nSource: %s\nMetadata: %s",
                doc.id,
                doc.content,
                doc.source,
                doc.metadata
            )
        
        # Rerank and get top 5
        reranked_results = classical_reranker.rerank(request.query, documents, top_k=5)
        
        debug_results = []
        for item in reranked_results:
            if isinstance(item, tuple) and len(item) == 2:
                doc, score = item
            else:
                doc = item
            debug_results.append(doc)

        logger.info("Reranked results documents for debugging (full docs):")
        for doc in debug_results:
            logger.debug("Document ID: %s\nContent: %s\nSource: %s\nMetadata: %s",
                        doc.id, doc.content, doc.source, doc.metadata)

        # Build prompt using the template
        # Generate intent-specific prompt - use the strategy's focus to determine prompt type
        prompt = build_prompt(strategy['focus'], request.query, reranked_results)
        
        # For direct queries, use simpler LLM parameters
        if strategy.get('focus') == 'exact_data_lookup':
            temperature = 0.0  # More deterministic for direct queries
            max_tokens = 100   # Shorter responses for direct answers
        else:
            temperature = 0.1
            max_tokens = 500
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
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
