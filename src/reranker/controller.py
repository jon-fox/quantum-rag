"""
Agent controller for deciding when to use quantum vs. classical reranking for podcast ad detection.
"""

from typing import Dict, Any, List
import numpy as np
from src.reranker.classical import Document, ClassicalReranker
from src.reranker.quantum import QuantumReranker


class RerankerController:
    """
    Agent controller that decides when to use quantum vs. classical reranking.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the controller with configuration."""
        self.config = config or {}
        self.classical_reranker = ClassicalReranker(
            self.config.get("classical_config", {})
        )
        self.quantum_reranker = QuantumReranker(self.config.get("quantum_config", {}))

        # Keywords that might benefit from quantum reranking
        self.quantum_keywords = [
            "advertisement",
            "ad",
            "sponsor",
            "commercial",
            "promotion",
            "product",
            "brand",
            "discount",
            "offer",
            "deal",
        ]

        # Complexity threshold - queries with word count above this might
        # benefit from quantum processing
        self.complexity_threshold = self.config.get("complexity_threshold", 8)

    def select_reranker(self, query: str) -> str:
        """
        Decide whether to use quantum or classical reranking based on query content.

        Args:
            query: The user query

        Returns:
            String indicating reranker type: "quantum" or "classical"
        """
        # Check for quantum keywords
        words = query.lower().split()
        query_complexity = len(words)

        # Check for keyword matches
        quantum_keyword_matches = sum(
            1
            for word in words
            if any(keyword in word for keyword in self.quantum_keywords)
        )

        # Decision logic
        if query_complexity > self.complexity_threshold or quantum_keyword_matches > 0:
            return "quantum"
        else:
            return "classical"

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None,
        reranker_type: str = "auto",
    ) -> Dict[str, Any]:
        """
        Rerank documents using the selected reranker.

        Args:
            query: The query string
            documents: List of documents to rerank
            top_k: Number of top documents to return
            reranker_type: "quantum", "classical", or "auto" for automatic selection

        Returns:
            Dictionary with reranked documents and metadata
        """
        if reranker_type == "auto":
            selected_reranker = self.select_reranker(query)
        else:
            selected_reranker = reranker_type

        if selected_reranker == "quantum":
            reranked_docs = self.quantum_reranker.rerank(query, documents, top_k)
            used_reranker = "quantum"
        else:
            reranked_docs = self.classical_reranker.rerank(query, documents, top_k)
            used_reranker = "classical"

        return {
            "documents": reranked_docs,
            "reranker_used": used_reranker,
            "query": query,
        }
