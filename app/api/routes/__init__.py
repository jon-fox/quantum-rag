"""
API Routes Definition

This module defines the API routes for the RAG Energy application.
"""
from fastapi import APIRouter, Depends, HTTPException, Body
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
import logging

from app.schema import Query, Document, SearchResponse
from app.agent import RagAgent
from app.embeddings import get_embeddings
from app.vector_store import VectorStore
from app.generator import Generator

# Create router
router = APIRouter(prefix="/api", tags=["API"])

# Configure logging
logger = logging.getLogger(__name__)

# Dependency injection
async def get_agent():
    """Dependency to get the RAG Agent"""
    return RagAgent()

async def get_vector_store():
    """Dependency to get the vector store"""
    vector_store = VectorStore()
    await vector_store.load()
    return vector_store

async def get_generator():
    """Dependency to get the generator"""
    return Generator()

@router.post("/query", response_model=SearchResponse)
async def query_documents(
    query: Query,
    agent: RagAgent = Depends(get_agent),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Query the RAG system and get reranked results
    """
    start_time = time.time()
    
    try:
        # Generate embedding for query
        query_embedding = await get_embeddings(query.text)
        
        # Retrieve documents from vector store
        retrieved_docs = await vector_store.search(query_embedding, top_k=query.top_k * 2)  # Get more docs for reranking
        
        # Process with agent (select reranker and apply it)
        response = await agent.process_query(query, retrieved_docs)
        
        # Log successful query
        logger.info(f"Query processed successfully in {response.execution_time_ms:.2f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.post("/generate", response_model=Dict[str, Any])
async def generate_response(
    query: Query,
    agent: RagAgent = Depends(get_agent),
    vector_store: VectorStore = Depends(get_vector_store),
    generator: Generator = Depends(get_generator)
):
    """
    Complete RAG pipeline: query, rerank, and generate response
    """
    try:
        # Generate embedding for query
        query_embedding = await get_embeddings(query.text)
        
        # Retrieve documents from vector store
        retrieved_docs = await vector_store.search(query_embedding, top_k=query.top_k * 2)
        
        # Process with agent (select reranker and apply it)
        search_response = await agent.process_query(query, retrieved_docs)
        
        # Generate response based on reranked documents
        generation_response = await generator.generate(query.text, search_response.documents)
        
        # Combine responses
        full_response = {
            "query": query.text,
            "reranker_used": search_response.reranker_used,
            "retrieval_time_ms": search_response.execution_time_ms,
            "generation_time_ms": generation_response.get("execution_time_ms", 0),
            "total_time_ms": search_response.execution_time_ms + generation_response.get("execution_time_ms", 0),
            "generated_text": generation_response.get("text", ""),
            "model": generation_response.get("model", "unknown"),
            "documents": [doc.model_dump() for doc in search_response.documents]
        }
        
        return full_response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@router.get("/documents/{doc_id}", response_model=Document)
async def get_document(
    doc_id: str,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Get a specific document by ID
    """
    doc = await vector_store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return doc

@router.get("/status")
async def get_status():
    """
    Get system status
    """
    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0"
    }