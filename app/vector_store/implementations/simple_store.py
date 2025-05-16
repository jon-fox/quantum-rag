"""
Simple In-Memory Vector Store Implementation

A lightweight vector store implementation using in-memory structures
for development and testing purposes.
"""
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import logging
import numpy as np
from datetime import datetime
from app.schema import Document

logger = logging.getLogger(__name__)

class SimpleVectorStore:
    """
    Simple in-memory vector store implementation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the simple vector store.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.embedding_dim = self.config.get("embedding_dim", 384)
        self.storage_path = self.config.get("storage_path", "./data/vectors/simple_store.json")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        # Initialize document storage
        self.documents = {}
        self.vectors = {}
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        doc_ids = []
        
        for doc in documents:
            # Skip documents without embeddings
            if not doc.embedding:
                logger.warning(f"Document {doc.id} has no embedding, skipping")
                continue
                
            # Store the document and its vector
            self.documents[doc.id] = doc
            self.vectors[doc.id] = np.array(doc.embedding, dtype=np.float32)
            doc_ids.append(doc.id)
        
        logger.info(f"Added {len(doc_ids)} documents to simple vector store")
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
        if not self.documents:
            logger.warning("Search attempted on empty store")
            return []
            
        # Convert to numpy array
        query_np = np.array(query_vector, dtype=np.float32)
        
        # Calculate cosine similarity for each document
        similarities = []
        for doc_id, vec in self.vectors.items():
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_np, vec)
            similarities.append((doc_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k results
        results = []
        for doc_id, similarity in similarities[:top_k]:
            doc = self.documents.get(doc_id)
            if doc:
                # Attach similarity score to metadata
                doc_copy = Document(**doc.model_dump())
                if not doc_copy.metadata:
                    doc_copy.metadata = {}
                doc_copy.metadata["score"] = float(similarity)
                results.append(doc_copy)
        
        return results
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (between -1 and 1)
        """
        if vec1.shape != vec2.shape:
            raise ValueError(f"Vector shapes do not match: {vec1.shape} vs {vec2.shape}")
            
        # Calculate dot product
        dot_product = np.dot(vec1, vec2)
        
        # Calculate magnitudes
        mag1 = np.linalg.norm(vec1)
        mag2 = np.linalg.norm(vec2)
        
        # Calculate cosine similarity
        if mag1 > 0 and mag2 > 0:
            return dot_product / (mag1 * mag2)
        else:
            return 0.0
    
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
        if doc_id in self.documents:
            del self.documents[doc_id]
            
            if doc_id in self.vectors:
                del self.vectors[doc_id]
                
            return True
        return False
    
    async def save(self) -> bool:
        """
        Save the vector store to disk.
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Convert document objects to dictionaries
            docs_dict = {
                doc_id: doc.model_dump() for doc_id, doc in self.documents.items()
            }
            
            # Convert numpy vectors to lists
            vectors_dict = {
                doc_id: vec.tolist() for doc_id, vec in self.vectors.items()
            }
            
            # Create storage object
            storage = {
                "documents": docs_dict,
                "vectors": vectors_dict,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to disk
            with open(self.storage_path, 'w') as f:
                json.dump(storage, f)
                
            logger.info(f"Saved {len(docs_dict)} documents to {self.storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            return False
    
    async def load(self) -> bool:
        """
        Load the vector store from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(self.storage_path):
                logger.warning(f"Storage file not found: {self.storage_path}")
                return False
            
            # Load from disk
            with open(self.storage_path, 'r') as f:
                storage = json.load(f)
            
            # Load documents
            self.documents = {}
            for doc_id, doc_data in storage.get("documents", {}).items():
                self.documents[doc_id] = Document(**doc_data)
            
            # Load vectors
            self.vectors = {}
            for doc_id, vec_data in storage.get("vectors", {}).items():
                self.vectors[doc_id] = np.array(vec_data, dtype=np.float32)
                
            logger.info(f"Loaded {len(self.documents)} documents from {self.storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
