"""
Vector Store Module

This module provides functionality for storing and retrieving document embeddings
using vector databases like FAISS.
"""
from typing import List, Dict, Any, Optional, Union
import os
import logging
import numpy as np
import json
from app.schema import Document

logger = logging.getLogger(__name__)

# Import vector store implementations
try:
    from app.vector_store.implementations.faiss_store import FAISSVectorStore
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not installed. FAISS vector store functionality will be unavailable.")
    FAISS_AVAILABLE = False

class VectorStore:
    """
    Vector store manager for document storage and retrieval
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the vector store.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.store_type = self.config.get("store_type", "faiss")
        self.embedding_dim = self.config.get("embedding_dim", 384)
        self.index_path = self.config.get("index_path", "./data/vectors")
        
        # Create vector store directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Initialize the appropriate vector store implementation
        if self.store_type == "faiss" and FAISS_AVAILABLE:
            self.store = FAISSVectorStore(self.config)
        else:
            # Fall back to simple in-memory vector store if FAISS not available
            from app.vector_store.implementations.simple_store import SimpleVectorStore
            logger.info("Using simple in-memory vector store")
            self.store = SimpleVectorStore(self.config)
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        return await self.store.add_documents(documents)
    
    async def search(self, query_vector: List[float], top_k: int = 5) -> List[Document]:
        """
        Search for documents similar to the query vector.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        return await self.store.search(query_vector, top_k)
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        return await self.store.get_document(doc_id)
    
    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the vector store.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if document was deleted, False otherwise
        """
        return await self.store.delete_document(doc_id)
    
    async def save(self) -> bool:
        """
        Save the vector store to disk.
        
        Returns:
            True if saved successfully, False otherwise
        """
        return await self.store.save()
    
    async def load(self) -> bool:
        """
        Load the vector store from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        return await self.store.load()