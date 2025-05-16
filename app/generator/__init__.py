"""
Generator Module for LLM Integration

This module provides the interface to large language models for generating
responses based on the retrieved and reranked content.
"""
from typing import Dict, List, Any, Optional
import logging
import json
import os
import time
from app.schema import Document

logger = logging.getLogger(__name__)

class Generator:
    """
    Text generation component that interfaces with language models to provide
    responses based on retrieved content.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.model_name = self.config.get("model_name", "gpt-4")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 1000)
        self.api_key = self.config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        
        # Try to import the OpenAI library
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            self.openai_available = True
        except ImportError:
            logger.warning("OpenAI package not installed. Using mock generator.")
            self.openai_available = False
    
    async def generate(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """
        Generate a response based on the query and retrieved documents.
        
        Args:
            query: User query
            documents: List of retrieved and reranked documents
            
        Returns:
            Dictionary containing the generated response and metadata
        """
        # Create system prompt with context from documents
        system_prompt = self._create_system_prompt()
        
        # Create user message with query and context from documents
        user_message = self._create_user_message(query, documents)
        
        # Generate response using language model
        if self.openai_available:
            try:
                response = await self._call_openai(system_prompt, user_message)
            except Exception as e:
                logger.error(f"Error calling OpenAI API: {str(e)}")
                response = self._mock_generate(query, documents)
        else:
            response = self._mock_generate(query, documents)
            
        return response
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the language model"""
        return """You are an energy analysis assistant with expertise in ERCOT (Electric Reliability Council of Texas) data.
Your role is to provide accurate information and analysis based on the reference information provided.
Only use information contained in the references to answer the query.
If the references don't contain relevant information, acknowledge that you don't have enough information.
When discussing forecasts or predictions, be clear about the timeframes and assumptions involved.
Always cite your sources by referring to the document ID when providing information."""
    
    def _create_user_message(self, query: str, documents: List[Document]) -> str:
        """
        Create the user message combining query and document context
        
        Args:
            query: User query
            documents: Retrieved documents
            
        Returns:
            Formatted user message string
        """
        # Format documents as context
        doc_texts = []
        for i, doc in enumerate(documents):
            doc_text = f"[{i+1}] Document ID: {doc.id}\n"
            doc_text += f"Source: {doc.source or 'unknown'}\n"
            if hasattr(doc, 'metadata') and doc.metadata:
                doc_text += f"Metadata: {json.dumps(doc.metadata)}\n"
            doc_text += f"Content: {doc.content}\n"
            doc_texts.append(doc_text)
            
        context = "\n\n".join(doc_texts)
        
        # Construct the final user message
        user_message = f"""Query: {query}

Reference Information:
{context}

Please answer the query based on the reference information above. If the information provided isn't sufficient, please indicate that."""
        
        return user_message
    
    async def _call_openai(self, system_prompt: str, user_message: str) -> Dict[str, Any]:
        """
        Call OpenAI API to generate response
        
        Args:
            system_prompt: System prompt
            user_message: User message with context
            
        Returns:
            Dictionary with generated text and metadata
        """
        start_time = time.time()
        
        # Create the messages for the API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Call the API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Extract the generated text
        generated_text = response.choices[0].message.content
        
        # Calculate timing
        execution_time_ms = (time.time() - start_time) * 1000
        
        return {
            "text": generated_text,
            "model": self.model_name,
            "execution_time_ms": execution_time_ms,
            "tokens": {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens
            }
        }
    
    def _mock_generate(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """
        Generate a mock response when OpenAI is not available
        
        Args:
            query: User query
            documents: Retrieved documents
            
        Returns:
            Dictionary with generated text and metadata
        """
        # Create a simple mock response based on query and documents
        doc_texts = [doc.content[:100] + "..." for doc in documents[:2]]
        context_summary = "\n".join(doc_texts)
        
        response_text = f"""Based on the ERCOT data you provided, I can offer the following analysis:

The documents indicate that {context_summary}

This information helps address your query about "{query}". 

Note: This is a mock response as the LLM integration is not currently available."""
        
        return {
            "text": response_text,
            "model": "mock-model",
            "execution_time_ms": 150.0,
            "tokens": {
                "prompt": 0,
                "completion": 0, 
                "total": 0
            }
        }