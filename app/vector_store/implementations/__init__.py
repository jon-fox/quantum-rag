"""
Vector Store Implementations

This package contains different vector store implementations for document retrieval.
"""
# Allow direct imports from implementations
from app.vector_store.implementations.simple_store import SimpleVectorStore

# Try to import FAISS implementation if available
try:
    from app.vector_store.implementations.faiss_store import FAISSVectorStore
    FAISS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAISS_AVAILABLE = False