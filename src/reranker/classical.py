"""
Classical Reranking Implementation

This module implements traditional scoring and reranking methods for 
retrieved documents based on their relevance to the input query.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
import logging
import torch
from sentence_transformers import CrossEncoder

# Initialize Cross-Encoder model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

# Define Document class locally to avoid import issues
class Document:
    """Document class for reranking."""
    def __init__(self, id: str, content: str, source: str = None, metadata: Dict[str, Any] = None):
        self.id = id
        self.content = content
        self.source = source
        self.metadata = metadata or {}

logger = logging.getLogger(__name__)

class ClassicalReranker:
    """Classical implementation of document reranking using traditional similarity metrics."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the classical reranker.
        
        Args:
            config: Configuration dictionary for the reranker
        """
        self.config = config or {}
        self.method = self.config.get("method", "cosine")
    
    def rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Tuple[Document, float]]:
        """
        Rerank documents based on their relevance to the query using Cross-Encoder.
        
        Args:
            query: The search query
            documents: List of retrieved documents to rerank
            top_k: Number of documents to return after reranking
        
        Returns:
            Reranked list of (document, score) tuples
        """
        if not documents:
            return []
        
        # Build inputs for Cross-Encoder as (query, document_content) pairs
        inputs = [(query, doc.content) for doc in documents]
        
        # Get scores from Cross-Encoder model
        scores = model.predict(inputs)
        
        # Zip scores back onto documents
        scored_docs = list(zip(documents, scores))
        
        # Sort by score in descending order
        reranked_docs_with_scores = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        
        # Return top_k (document, score) tuples if specified
        if top_k is not None:
            reranked_docs_with_scores = reranked_docs_with_scores[:top_k]
        
        return reranked_docs_with_scores
