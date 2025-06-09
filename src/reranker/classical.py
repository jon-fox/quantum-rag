"""
Classical Reranking Implementation

This module implements traditional scoring and reranking methods for 
retrieved documents based on their relevance to the input query.
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import logging
import torch
import time
import re
import os

# Disable tqdm progress bars to prevent long log lines
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Define Document class locally to avoid import issues
class Document:
    """Document class for reranking."""
    def __init__(self, id: str, content: str, source: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        self.id = id
        self.content = content
        self.source = source
        self.metadata = metadata or {}

class ClassicalReranker:
    """Classical implementation of document reranking using Cross-Encoder."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the classical reranker.
        
        Args:
            config: Configuration dictionary for the reranker
        """
        self.config = config or {}
        self.method = self.config.get("method", "cross-encoder")
        self.model_name = self.config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.batch_size = self.config.get("batch_size", 32)
        self.max_sequence_length = self.config.get("max_sequence_length", 512)
        self.max_retries = self.config.get("max_retries", 3)
        self.timeout = self.config.get("timeout", 30)
        
        # Device selection with fallback
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model with error handling
        self.model: Optional[CrossEncoder] = None
        self.model_loaded = False
        self._initialize_model()
        
        # Cache for repeated queries/documents
        self.score_cache = {}
        self.enable_cache = self.config.get("enable_cache", True)
    
    def _initialize_model(self):
        """Initialize the Cross-Encoder model with fallback handling."""
        try:
            # Check for common HuggingFace token environment variable names
            hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HUGGINGFACE_API_TOKEN') or os.getenv('HF_TOKEN')
            
            if hf_token:
                from huggingface_hub import login
                login(token=hf_token)
                logger.info("Successfully authenticated with Hugging Face")
            else:
                logger.warning("No Hugging Face token found in environment variables")
                
            self.model = CrossEncoder(self.model_name, device=self.device)
            self.model_loaded = True
            logger.info(f"Successfully loaded Cross-Encoder model '{self.model_name}' on device '{self.device}'")
        except Exception as e:
            logger.error(f"Failed to load Cross-Encoder model: {e}")
            
            # Try alternative models that don't require authentication
            fallback_models = [
                "cross-encoder/ms-marco-TinyBERT-L-2-v2",  # Smaller, more likely to be public
                "cross-encoder/ms-marco-MiniLM-L-2-v2",    # Even smaller version
            ]
            
            for fallback_model in fallback_models:
                try:
                    logger.info(f"Trying fallback model: {fallback_model}")
                    self.model = CrossEncoder(fallback_model, device=self.device)
                    self.model_name = fallback_model  # Update model name
                    self.model_loaded = True
                    logger.info(f"Successfully loaded fallback Cross-Encoder model '{fallback_model}' on device '{self.device}'")
                    return
                except Exception as fallback_e:
                    logger.warning(f"Fallback model {fallback_model} also failed: {fallback_e}")
    
    def _sanitize_text(self, text: str) -> str:
        """Clean and truncate text to avoid input issues."""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove excessive whitespace and special characters
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate to max sequence length (rough approximation)
        if len(text) > self.max_sequence_length * 4:  # ~4 chars per token estimate
            text = text[:self.max_sequence_length * 4]
        
        return text
    
    def _validate_inputs(self, query: str, documents: List[Document]) -> bool:
        """Validate input types and non-emptiness."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Query must be a non-empty string")
            return False
        
        if not isinstance(documents, list) or not documents:
            logger.error("Documents must be a non-empty list")
            return False
        
        for i, doc in enumerate(documents):
            if not isinstance(doc, Document):
                logger.error(f"Document at index {i} is not a Document instance")
                return False
            if not hasattr(doc, 'content') or not doc.content:
                logger.error(f"Document at index {i} has empty content")
                return False
        
        return True
    
    def _get_cache_key(self, query: str, doc_content: str) -> str:
        """Generate cache key for query-document pair."""
        return f"{hash(query)}_{hash(doc_content)}"
    
    def _predict_with_retries(self, inputs: List[Tuple[str, str]]) -> np.ndarray:
        """Predict scores with retry logic and timeout handling."""
        if self.model is None:
            raise RuntimeError("Model is not initialized")
            
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                # Disable tqdm progress bars by setting show_progress_bar=False
                scores = self.model.predict(inputs, show_progress_bar=False)
                elapsed_time = time.time() - start_time
                
                logger.debug(f"Cross-Encoder prediction completed in {elapsed_time:.2f}s for {len(inputs)} pairs")
                return scores
                
            except Exception as e:
                logger.warning(f"Prediction attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
        
        raise RuntimeError("All retry attempts failed")
    
    def _handle_reranker_failure(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """Handle reranker failure by logging and returning original documents with neutral scores."""
        logger.error("Cross-Encoder reranking failed - returning original document order with neutral scores")
        logger.error(f"Failed to rerank {len(documents)} documents for query: {query[:100]}...")
        # Return original documents with neutral scores (0.5) to indicate no reranking occurred
        return [(doc, 0.5) for doc in documents]
    
    def rerank(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """
        Rerank documents based on their relevance to the query using Cross-Encoder.
        
        Args:
            query: The search query
            documents: List of retrieved documents to rerank
            top_k: Number of documents to return after reranking
        
        Returns:
            Reranked list of (document, score) tuples
        """
        start_time = time.time()
        
        # Input validation
        if not self._validate_inputs(query, documents):
            logger.error("Input validation failed - returning original documents with neutral scores")
            return [(doc, 0.5) for doc in documents]
        
        # Sanitize inputs
        query = self._sanitize_text(query)
        
        # Use fallback if model not loaded
        if not self.model_loaded or self.model is None:
            logger.error("Cross-Encoder model not available - reranking failed")
            return self._handle_reranker_failure(query, documents)
        else:
            try:
                # Check cache first
                cached_scores = []
                uncached_pairs = []
                uncached_docs = []
                
                for doc in documents:
                    sanitized_content = self._sanitize_text(doc.content)
                    cache_key = self._get_cache_key(query, sanitized_content)
                    
                    if self.enable_cache and cache_key in self.score_cache:
                        cached_scores.append((doc, self.score_cache[cache_key]))
                    else:
                        uncached_pairs.append((query, sanitized_content))
                        uncached_docs.append(doc)
                
                # Process uncached pairs in batches
                if uncached_pairs:
                    all_scores = []
                    for i in range(0, len(uncached_pairs), self.batch_size):
                        batch = uncached_pairs[i:i + self.batch_size]
                        batch_scores = self._predict_with_retries(batch)
                        all_scores.extend(batch_scores)
                    
                    # Cache new scores
                    for i, (doc, score) in enumerate(zip(uncached_docs, all_scores)):
                        if self.enable_cache:
                            cache_key = self._get_cache_key(query, self._sanitize_text(doc.content))
                            self.score_cache[cache_key] = float(score)
                        cached_scores.append((doc, float(score)))
                
                scored_docs = cached_scores
                
            except Exception as e:
                logger.error(f"Cross-Encoder prediction failed: {e}")
                return self._handle_reranker_failure(query, documents)
        
        # Sort by score in descending order
        reranked_docs_with_scores = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        
        # Apply top_k limit
        if top_k is not None and top_k > 0:
            reranked_docs_with_scores = reranked_docs_with_scores[:top_k]
        
        # Log performance metrics
        elapsed_time = time.time() - start_time
        logger.info(f"Reranking completed in {elapsed_time:.2f}s for {len(documents)} documents")
        
        if reranked_docs_with_scores:
            top_score = reranked_docs_with_scores[0][1]
            logger.debug(f"Top score: {top_score:.4f}")
        
        return reranked_docs_with_scores
