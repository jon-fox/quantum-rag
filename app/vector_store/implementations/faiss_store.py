"""
FAISS Vector Store Implementation
"""
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import logging
import numpy as np
from app.schema import Document

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed. FAISSVectorStore is not functional.")

class FAISSVectorStore:
    """
    Vector store implementation using FAISS for efficient similarity search
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the FAISS vector store.
        
        Args:
            config: Configuration dictionary
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not installed. Please install it using 'pip install faiss-cpu' or 'pip install faiss-gpu'")
            
        self.config = config or {}
        self.embedding_dim = self.config.get("embedding_dim", 384)
        self.index_path = self.config.get("index_path", "./data/vectors/faiss_index")
        self.metadata_path = self.index_path + "_metadata.json"
        
        # Create directory for index if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Initialize the FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Document metadata storage (id -> document mapping)
        self.documents = {}
        
        # ID to index mapping
        self.id_to_index = {}
        self.next_index = 0
    
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
            
        doc_ids = []
        vectors = []
        
        for doc in documents:
            # Skip documents without embeddings
            if not doc.embedding:
                logger.warning(f"Document {doc.id} has no embedding, skipping")
                continue
                
            # Convert embedding to numpy array
            embedding = np.array(doc.embedding, dtype=np.float32)
            
            # Ensure embedding has the right dimension
            if embedding.shape[0] != self.embedding_dim:
                logger.warning(f"Document {doc.id} embedding dimension {embedding.shape[0]} != expected {self.embedding_dim}, skipping")
                continue
                
            # Store the document
            self.documents[doc.id] = doc
            
            # Map document ID to index
            self.id_to_index[doc.id] = self.next_index
            self.next_index += 1
            
            # Add to vectors for batch insertion
            vectors.append(embedding)
            doc_ids.append(doc.id)
        
        if vectors:
            # Convert to numpy array and add to index
            vectors_array = np.vstack(vectors).astype(np.float32)
            self.index.add(vectors_array)
            
            logger.info(f"Added {len(vectors)} documents to FAISS index")
        
        return doc_ids
    
    async def search(self, query_vector: List[float], top_k: int = 5) -> List[Document]:
        """
        Search for documents similar to the query vector.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        if not self.index.ntotal:
            logger.warning("Search attempted on empty index")
            return []
            
        # Convert to numpy array
        query_np = np.array([query_vector], dtype=np.float32)
        
        # Search the index
        distances, indices = self.index.search(query_np, min(top_k, self.index.ntotal))
        
        # Get document IDs from indices
        results = []
        index_to_id = {idx: doc_id for doc_id, idx in self.id_to_index.items()}
        
        for i, idx in enumerate(indices[0]):
            # Skip invalid indices
            if idx == -1:
                continue
                
            # Get document ID from index
            doc_id = index_to_id.get(idx)
            if not doc_id:
                continue
                
            # Get document
            doc = self.documents.get(doc_id)
            if doc:
                # Attach distance score to metadata
                doc_copy = Document(**doc.model_dump())
                if not doc_copy.metadata:
                    doc_copy.metadata = {}
                doc_copy.metadata["score"] = float(distances[0][i])
                results.append(doc_copy)
        
        return results
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        return self.documents.get(doc_id)
    
    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the vector store.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if document was deleted, False otherwise
        """
        # FAISS doesn't support direct deletion
        # We would need to rebuild the index, which is inefficient
        # For a production system, consider using ScaNN or other libraries
        # that support efficient updates/deletions
        logger.warning("Direct deletion not supported in FAISS. Document will be removed from metadata only.")
        
        if doc_id in self.documents:
            del self.documents[doc_id]
            # Note: the vector is still in the FAISS index, just unreachable
            return True
        return False
    
    async def save(self) -> bool:
        """
        Save the vector store to disk.
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save metadata (documents and id->index mapping)
            metadata = {
                "next_index": self.next_index,
                "id_to_index": self.id_to_index,
                "documents": {
                    doc_id: doc.model_dump() for doc_id, doc in self.documents.items()
                }
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f)
                
            logger.info(f"Saved FAISS index with {self.index.ntotal} vectors to {self.index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
            return False
    
    async def load(self) -> bool:
        """
        Load the vector store from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Check if files exist
            if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
                logger.warning(f"Index or metadata file not found: {self.index_path}")
                return False
                
            # Load FAISS index
            self.index = faiss.read_index(self.index_path)
            
            # Load metadata
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
                
            self.next_index = metadata.get("next_index", 0)
            self.id_to_index = metadata.get("id_to_index", {})
            
            # Load documents
            self.documents = {}
            for doc_id, doc_data in metadata.get("documents", {}).items():
                self.documents[doc_id] = Document(**doc_data)
                
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors from {self.index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            return False
