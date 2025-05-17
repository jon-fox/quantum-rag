"""
Vector Store Module

This module provides functionality for storing and retrieving document embeddings
using vector databases like FAISS, integrated with LangChain's vector stores.
"""

# Import and expose the VectorStore class from store.py
from app.vector_store.store import VectorStore

# Re-export important constants for backward compatibility
from app.vector_store.store import LANGCHAIN_AVAILABLE, FAISS_AVAILABLE
