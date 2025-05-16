"""
Agent Module

This module implements the agent-based routing logic that selects between
classical and quantum reranking methods based on query characteristics.
"""
from typing import Dict, List, Any, Optional
import logging
from app.schema import Document, Query, SearchResponse
from app.reranker.classical import ClassicalReranker
from app.reranker.quantum import QuantumReranker

logger = logging.getLogger(__name__)

class RagAgent:
    """
    RAG Agent that intelligently routes between classical and quantum reranking
    based on query characteristics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RAG Agent.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize rerankers
        self.classical_reranker = ClassicalReranker(self.config.get("classical_config"))
        self.quantum_reranker = QuantumReranker(self.config.get("quantum_config"))
        
        # Decision threshold for when to use quantum reranking
        # Higher values = less likely to use quantum
        self.quantum_threshold = self.config.get("quantum_threshold", 0.7)

    async def process_query(self, query: Query, retrieved_docs: List[Document]) -> SearchResponse:
        """
        Process a query by determining the best reranking method and applying it.
        
        Args:
            query: The user's query
            retrieved_docs: Documents retrieved from the vector store
            
        Returns:
            Search response with reranked documents
        """
        # Start timing
        import time
        start_time = time.time()
        
        # First, decide whether to use quantum or classical reranking
        use_quantum = self._should_use_quantum(query, retrieved_docs)
        
        # Override with explicit user preference if specified
        if query.use_quantum is not None:
            use_quantum = query.use_quantum
            
        # Apply the selected reranking method
        if use_quantum:
            logger.info(f"Using quantum reranking for query: {query.text}")
            reranked_docs = self.quantum_reranker.rerank(
                query.text, 
                retrieved_docs,
                top_k=query.top_k
            )
            reranker_used = "quantum"
        else:
            logger.info(f"Using classical reranking for query: {query.text}")
            reranked_docs = self.classical_reranker.rerank(
                query.text, 
                retrieved_docs,
                top_k=query.top_k
            )
            reranker_used = "classical"
        
        # Calculate execution time in milliseconds
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Create response
        response = SearchResponse(
            query=query.text,
            documents=reranked_docs,
            reranker_used=reranker_used,
            execution_time_ms=execution_time_ms
        )
        
        return response
    
    def _should_use_quantum(self, query: Query, docs: List[Document]) -> bool:
        """
        Determine whether to use quantum reranking based on query characteristics.
        
        Args:
            query: The user's query
            docs: Retrieved documents
            
        Returns:
            Boolean indicating whether to use quantum reranking
        """
        # Simple decision logic - can be expanded with more sophisticated criteria
        # This is just a placeholder implementation
        
        # Criteria that might favor quantum reranking:
        # 1. Complex queries with multiple concepts
        # 2. Queries about forecasting or predictive analysis
        # 3. Queries with high ambiguity
        # 4. Large number of documents that might benefit from quantum advantage
        
        score = 0.0
        
        # Check for energy forecasting keywords
        forecast_keywords = ["forecast", "prediction", "future", "trend", "projected"]
        if any(keyword in query.text.lower() for keyword in forecast_keywords):
            score += 0.3
            
        # Check for complex query (based on length as a simple heuristic)
        if len(query.text.split()) > 10:
            score += 0.2
            
        # Check for document volume
        if len(docs) > 50:
            score += 0.2
            
        # Example: If query explicitly mentions quantum
        if "quantum" in query.text.lower():
            score += 0.3
            
        logger.debug(f"Query '{query.text}' quantum score: {score}")
        
        # Return true if score exceeds threshold
        return score >= self.quantum_threshold
