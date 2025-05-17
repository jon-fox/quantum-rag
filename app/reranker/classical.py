"""
Classical Reranking Implementation

This module implements traditional scoring and reranking methods for 
retrieved documents based on their relevance to the input query,
integrated with LangChain's retriever and document compressor interfaces.
"""
from typing import List, Dict, Any, Tuple, Protocol, runtime_checkable
import numpy as np
import logging
from app.schema.models import Document
import abc

logger = logging.getLogger(__name__)

# Try to import LangChain
try:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter
    from langchain.schema.document import Document as LCDocument
    from langchain.schema.retriever import BaseRetriever
    from langchain_community.llms import OpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain not available. Using traditional reranking methods.")
    LANGCHAIN_AVAILABLE = False
    
    # Define a fallback BaseRetriever when LangChain is not available
    class BaseRetriever(abc.ABC):
        """Abstract base class for a document retriever when LangChain is not available."""
        
        @abc.abstractmethod
        def _get_relevant_documents(self, query, **kwargs):
            """Get documents relevant to a query."""
            pass
            
        @abc.abstractmethod
        async def _aget_relevant_documents(self, query, **kwargs):
            """Get documents relevant to a query asynchronously."""
            pass

# Simple retriever wrapper for our documents
class ListRetriever(BaseRetriever):
    """A simple retriever that returns documents from a list"""
    
    def __init__(self, documents):
        """Initialize with documents"""
        self.documents = documents
    
    def _get_relevant_documents(self, query, **kwargs):
        """Return all documents"""
        return self.documents
    
    async def _aget_relevant_documents(self, query, **kwargs):
        """Return all documents (async)"""
        return self.documents

class ClassicalReranker:
    """Classical implementation of document reranking using traditional similarity metrics 
    and LangChain's document compression when available."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the classical reranker.
        
        Args:
            config: Configuration dictionary for the reranker
        """
        self.config = config or {}
        self.method = self.config.get("method", "cosine")
        self.use_langchain = self.config.get("use_langchain", LANGCHAIN_AVAILABLE)
        self.llm_name = self.config.get("llm_name", "gpt-3.5-turbo")
        self.llm_temperature = self.config.get("llm_temperature", 0)
        
        # Initialize LLM for LangChain reranking if enabled
        if self.use_langchain and LANGCHAIN_AVAILABLE:
            try:
                self.llm = OpenAI(temperature=self.llm_temperature, model_name=self.llm_name)
                
                # Initialize document compressors based on method
                if self.method == "llm_filter":
                    self.compressor = LLMChainFilter.from_llm(self.llm)
                elif self.method == "llm_extractor":
                    self.compressor = LLMChainExtractor.from_llm(self.llm)
                else:
                    self.compressor = None
            except Exception as e:
                logger.error(f"Error initializing LangChain components: {str(e)}")
                self.use_langchain = False
    
    def rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Document]:
        """
        Rerank documents based on their relevance to the query using classical methods
        or LangChain document compression.
        
        Args:
            query: The search query
            documents: List of retrieved documents to rerank
            top_k: Number of documents to return after reranking
        
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        # Use LangChain reranking if available and enabled
        if self.use_langchain and LANGCHAIN_AVAILABLE and self.compressor:
            try:
                return self._langchain_rerank(query, documents, top_k)
            except Exception as e:
                logger.error(f"LangChain reranking failed: {str(e)}. Falling back to traditional methods.")
            
        # Traditional reranking as fallback    
        # Score documents using chosen method
        scored_docs = self._score_documents(query, documents)
        
        # Sort by score in descending order
        reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        
        # Return top_k documents if specified
        if top_k is not None:
            reranked_docs = reranked_docs[:top_k]
        
        # Return just the documents without scores
        return [doc for doc, _ in reranked_docs]
    
    def _langchain_rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Document]:
        """
        Rerank documents using LangChain's document compression.
        
        Args:
            query: The search query
            documents: List of retrieved documents to rerank
            top_k: Number of documents to return after reranking
            
        Returns:
            Reranked list of documents
        """
        # Convert our documents to LangChain format
        lc_docs = []
        for doc in documents:
            lc_docs.append(LCDocument(
                page_content=doc.content,
                metadata={"id": doc.id, "source": doc.source, **doc.metadata}
            ))
        
        # Create base retriever that returns all documents
        base_retriever = ListRetriever(lc_docs)
        
        # Create compression retriever
        retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=base_retriever
        )
        
        # Get compressed (reranked) documents
        reranked_lc_docs = retriever.get_relevant_documents(query)
        
        # Limit to top_k if specified
        if top_k is not None and len(reranked_lc_docs) > top_k:
            reranked_lc_docs = reranked_lc_docs[:top_k]
        
        # Convert back to our Document format
        reranked_docs = []
        for lc_doc in reranked_lc_docs:
            meta = lc_doc.metadata.copy()
            doc_id = meta.pop("id", None)
            source = meta.pop("source", None)
            
            reranked_docs.append(Document(
                id=doc_id or str(hash(lc_doc.page_content))[:16],
                content=lc_doc.page_content,
                source=source,
                metadata=meta
            ))
            
        return reranked_docs
    
    def _score_documents(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """
        Score documents based on their relevance to the query.
        
        Args:
            query: The search query
            documents: List of retrieved documents to score
            
        Returns:
            List of (document, score) tuples
        """
        # Implement scoring logic based on the method
        if self.method == "cosine":
            # Actual cosine similarity scoring with embedded vectors
            # For now, just a placeholder
            return [(doc, 0.5) for doc in documents]  # Replace with actual scoring
        elif self.method == "bm25":
            # BM25 scoring could be implemented here
            return [(doc, 0.5) for doc in documents]  # Replace with actual scoring
        else:
            # Default fallback scoring
            return [(doc, 0.5) for doc in documents]  # Replace with actual scoring