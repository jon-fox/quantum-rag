"""
Vector Store Module

This module provides the main VectorStore class for storing and retrieving document embeddings
using vector databases like FAISS, integrated with LangChain's vector stores.
"""
from typing import List, Dict, Any, Optional, Union
import os
import logging
import numpy as np
import json
from app.schema.models import Document
from app.embeddings.embeddings import get_embedding_model

logger = logging.getLogger(__name__)

# Try to import LangChain vector stores
try:
    from langchain.vectorstores import FAISS as LangChainFAISS
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain not available. Using traditional vector store implementations.")
    LANGCHAIN_AVAILABLE = False

# Import traditional vector store implementations as fallback
try:
    from app.vector_store.implementations.faiss_store import FAISSVectorStore
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not installed. FAISS vector store functionality will be unavailable.")
    FAISS_AVAILABLE = False

class VectorStore:
    """
    Vector store manager for document storage and retrieval,
    with LangChain integration when available.
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
        self.use_langchain = self.config.get("use_langchain", LANGCHAIN_AVAILABLE)
        
        # Create vector store directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Get embedding model
        self.embedding_model = None
        if self.use_langchain:
            self.embedding_model = get_embedding_model()
        
        # Initialize store
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize the vector store implementation"""
        # Try to use LangChain implementation if requested
        if self.use_langchain and LANGCHAIN_AVAILABLE and self.embedding_model:
            logger.info("Using LangChain FAISS vector store")
            # The store will be initialized in load() as we need to check
            # for existing index file first
            self.store = None
            self.using_langchain = True
        elif self.store_type == "faiss" and FAISS_AVAILABLE:
            logger.info("Using traditional FAISS vector store")
            self.store = FAISSVectorStore(self.config)
            self.using_langchain = False
        else:
            # Fall back to simple in-memory vector store
            from app.vector_store.implementations.simple_store import SimpleVectorStore
            logger.info("Using simple in-memory vector store")
            self.store = SimpleVectorStore(self.config)
            self.using_langchain = False

    async def load(self):
        """Load the vector store if needed"""
        if self.using_langchain:
            # Check if FAISS index exists already
            index_file = os.path.join(self.index_path, "index.faiss")
            if os.path.exists(index_file):
                logger.info(f"Loading existing LangChain FAISS index from {index_file}")
                try:
                    self.store = LangChainFAISS.load_local(
                        folder_path=self.index_path,
                        embeddings=self.embedding_model,
                        allow_dangerous_deserialization=True
                    )
                except Exception as e:
                    logger.error(f"Error loading LangChain FAISS index: {str(e)}")
                    self._fallback_to_traditional_store()
            else:
                # Create empty LangChain FAISS store
                logger.info("Initializing empty LangChain FAISS index")
                try:
                    self.store = LangChainFAISS.from_texts(
                        texts=["placeholder"],  # Need at least one document
                        embedding=self.embedding_model
                    )
                    # Save the empty index
                    self.store.save_local(self.index_path)
                    # Delete the placeholder document
                    # This is a bit of a hack, but LangChain doesn't support creating empty stores
                    # Will be fixed when we add real documents
                except Exception as e:
                    logger.error(f"Error initializing LangChain FAISS index: {str(e)}")
                    self._fallback_to_traditional_store()
        elif hasattr(self.store, 'load'):
            # Use traditional store's load method if it exists
            await self.store.load()
    
    def _fallback_to_traditional_store(self):
        """Fall back to traditional vector store implementation"""
        if self.store_type == "faiss" and FAISS_AVAILABLE:
            logger.info("Falling back to traditional FAISS vector store")
            self.store = FAISSVectorStore(self.config)
        else:
            from app.vector_store.implementations.simple_store import SimpleVectorStore
            logger.info("Falling back to simple in-memory vector store")
            self.store = SimpleVectorStore(self.config)
        self.using_langchain = False
    
    def _to_langchain_format(self, docs):
        """Convert our Document objects to LangChain document format"""
        from langchain.schema.document import Document as LCDocument
        
        result = []
        for doc in docs:
            result.append(LCDocument(
                page_content=doc.content,
                metadata={
                    "id": doc.id,
                    "source": doc.source,
                    **doc.metadata
                }
            ))
        return result
    
    def _from_langchain_format(self, lc_docs):
        """Convert LangChain documents to our Document format"""
        result = []
        for lc_doc in lc_docs:
            meta = lc_doc.metadata.copy()
            doc_id = meta.pop("id", None)
            source = meta.pop("source", None)
            
            result.append(Document(
                id=doc_id or str(hash(lc_doc.page_content))[:16],
                content=lc_doc.page_content,
                source=source,
                metadata=meta
            ))
        return result
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
            
        if self.using_langchain:
            try:
                # Convert to LangChain format and add
                texts = [doc.content for doc in documents]
                metadatas = [{
                    "id": doc.id,
                    "source": doc.source,
                    **doc.metadata
                } for doc in documents]
                
                # Add documents to store
                self.store.add_texts(texts=texts, metadatas=metadatas)
                
                # Save updated index
                self.store.save_local(self.index_path)
                
                return [doc.id for doc in documents]
            except Exception as e:
                logger.error(f"Error adding documents to LangChain store: {str(e)}")
                # Try to recover by falling back to traditional implementation
                self._fallback_to_traditional_store()
                return await self.store.add_documents(documents)
        else:
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
        if self.using_langchain:
            try:
                # LangChain FAISS store doesn't support direct vector search
                # So we use a workaround to find similar documents
                docs_with_scores = self.store.similarity_search_with_score_by_vector(
                    embedding=query_vector,
                    k=top_k
                )
                
                # Convert to our Document format
                results = []
                for doc, score in docs_with_scores:
                    meta = doc.metadata.copy()
                    doc_id = meta.pop("id", None)
                    source = meta.pop("source", None)
                    
                    results.append(Document(
                        id=doc_id or str(hash(doc.page_content))[:16],
                        content=doc.page_content,
                        source=source,
                        metadata={**meta, "score": score}
                    ))
                
                return results
            except Exception as e:
                logger.error(f"Error searching LangChain store: {str(e)}")
                if hasattr(self.store, 'search'):
                    return await self.store.search(query_vector, top_k)
                return []
        else:
            return await self.store.search(query_vector, top_k)
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        if self.using_langchain:
            # LangChain doesn't provide direct lookup by ID, so we have to search
            # This is inefficient, but works as a fallback
            try:
                # We'd need to implement a search by metadata
                # For now, we fall back to traditional implementation
                if hasattr(self.store, 'get_document'):
                    return await self.store.get_document(doc_id)
                return None
            except Exception:
                return None
        else:
            return await self.store.get_document(doc_id)
    
    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the vector store.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if document was deleted, False otherwise
        """
        if self.using_langchain:
            # LangChain FAISS doesn't support direct deletion
            # Would need to implement our own method
            logger.warning("Document deletion not implemented for LangChain store")
            return False
        else:
            return await self.store.delete_document(doc_id)
            
    def as_retriever(self, search_kwargs=None):
        """
        Get a LangChain retriever for this vector store.
        Useful for integration with LangChain chains.
        
        Args:
            search_kwargs: Arguments to pass to the similarity search
            
        Returns:
            LangChain retriever
        """
        if not self.using_langchain:
            raise ValueError("LangChain not available or not enabled")
            
        search_kwargs = search_kwargs or {"k": 5}
        return self.store.as_retriever(search_kwargs=search_kwargs)
    
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
