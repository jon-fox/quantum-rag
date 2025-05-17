"""
Agent Module

This module implements the agent-based routing logic that selects between
classical and quantum reranking methods based on query characteristics,
leveraging LangChain's agent capabilities.
"""
from typing import Dict, List, Any, Optional
import logging
import os
from app.schema.models import Document, Query, SearchResponse
from app.reranker.classical import ClassicalReranker
from app.reranker.quantum import QuantumReranker

logger = logging.getLogger(__name__)

# Try to import LangChain agent components
try:
    from langchain.agents import initialize_agent, Tool
    from langchain.agents import AgentType
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.schema.runnable import RunnablePassthrough
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain not available. Using traditional agent logic.")
    LANGCHAIN_AVAILABLE = False

class RagAgent:
    """
    RAG Agent that intelligently routes between classical and quantum reranking
    based on query characteristics, integrating LangChain's agent framework when available.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RAG Agent.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.use_langchain = self.config.get("use_langchain", LANGCHAIN_AVAILABLE)
        
        # Initialize rerankers
        classical_config = self.config.get("classical_config", {})
        quantum_config = self.config.get("quantum_config", {})
        
        # Ensure reranker configs have langchain setting
        if "use_langchain" not in classical_config:
            classical_config["use_langchain"] = self.use_langchain
        if "use_langchain" not in quantum_config:
            quantum_config["use_langchain"] = self.use_langchain
            
        self.classical_reranker = ClassicalReranker(classical_config)
        self.quantum_reranker = QuantumReranker(quantum_config)
        
        # Decision threshold for when to use quantum reranking
        # Higher values = less likely to use quantum
        self.quantum_threshold = self.config.get("quantum_threshold", 0.7)
        
        # Initialize LangChain agent if available
        if self.use_langchain and LANGCHAIN_AVAILABLE:
            try:
                # Initialize LLM for agent
                model_name = self.config.get("model_name", "gpt-3.5-turbo")
                temperature = self.config.get("temperature", 0)
                
                if model_name.startswith("gpt"):
                    self.llm = ChatOpenAI(temperature=temperature, model_name=model_name)
                else:
                    self.llm = OpenAI(temperature=temperature, model_name=model_name)
                
                # Create tools for the agent
                self.tools = [
                    Tool(
                        name="ClassicalReranker",
                        func=self._use_classical_reranker_tool,
                        description="Use classical methods to rerank documents. Best for standard information retrieval tasks."
                    ),
                    Tool(
                        name="QuantumReranker",
                        func=self._use_quantum_reranker_tool,
                        description="Use quantum computing methods to rerank documents. Best for capturing complex semantic relationships."
                    )
                ]
                
                # Initialize agent
                self.agent = initialize_agent(
                    self.tools, 
                    self.llm, 
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=self.config.get("verbose", False)
                )
                
                self.agent_initialized = True
                logger.info("LangChain agent initialized successfully")
                
            except Exception as e:
                logger.error(f"Error initializing LangChain agent: {str(e)}")
                self.agent_initialized = False
        else:
            self.agent_initialized = False
    
    def _use_classical_reranker_tool(self, query_text, docs_dict_str, top_k=5):
        """Tool function for the LangChain agent to use classical reranker"""
        import json
        
        # Parse documents from string representation
        try:
            docs_dict = json.loads(docs_dict_str)
            documents = [Document(**doc) for doc in docs_dict]
        except Exception as e:
            return f"Error parsing documents: {str(e)}"
        
        # Apply reranking
        reranked = self.classical_reranker.rerank(query_text, documents, top_k)
        return [doc.model_dump() for doc in reranked]
    
    def _use_quantum_reranker_tool(self, query_text, docs_dict_str, top_k=5):
        """Tool function for the LangChain agent to use quantum reranker"""
        import json
        
        # Parse documents from string representation
        try:
            docs_dict = json.loads(docs_dict_str)
            documents = [Document(**doc) for doc in docs_dict]
        except Exception as e:
            return f"Error parsing documents: {str(e)}"
        
        # Apply reranking
        reranked = self.quantum_reranker.rerank(query_text, documents, top_k)
        return [doc.model_dump() for doc in reranked]

    async def process_query(self, query: Query, retrieved_docs: List[Document]) -> SearchResponse:
        """
        Process a query by determining the best reranking method and applying it,
        using LangChain agent when available.
        
        Args:
            query: The user's query
            retrieved_docs: Documents retrieved from the vector store
            
        Returns:
            Search response with reranked documents
        """
        # Start timing
        import time
        start_time = time.time()
        
        # Try using LangChain agent if available
        if self.agent_initialized and len(retrieved_docs) > 0:
            try:
                reranked_docs, reranker_used = await self._agent_process_query(query, retrieved_docs)
            except Exception as e:
                logger.error(f"LangChain agent error: {str(e)}. Falling back to traditional method.")
                reranked_docs, reranker_used = await self._traditional_process_query(query, retrieved_docs)
        else:
            # Use traditional processing
            reranked_docs, reranker_used = await self._traditional_process_query(query, retrieved_docs)
        
        # Calculate execution time in milliseconds
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Create response
        response = SearchResponse(
            query=query.text,
            documents=reranked_docs,
            reranker_used=reranker_used,
            execution_time_ms=execution_time_ms
        )
        
        return response
        
    async def _agent_process_query(self, query: Query, retrieved_docs: List[Document]):
        """Process query using LangChain agent"""
        import json
        
        # Prepare input for agent
        query_text = query.text
        docs_dict = [doc.model_dump() for doc in retrieved_docs]
        docs_str = json.dumps(docs_dict)
        
        # Prompt for the agent
        agent_prompt = f"""
        Determine the best reranking method for this query: "{query_text}"
        
        You have two options:
        1. ClassicalReranker: Best for standard information retrieval tasks
        2. QuantumReranker: Best for capturing complex semantic relationships and subtle connections
        
        Query complexity: {self._assess_query_complexity(query_text)}
        Query type: {self._assess_query_type(query_text)}
        
        User preference for quantum: {"Yes" if query.use_quantum else "Not specified"}
        Number of documents: {len(retrieved_docs)}
        Top k to return: {query.top_k}
        
        Analyze the query and decide which reranker to use.
        """
        
        # Run the agent
        agent_response = await self.agent.arun(agent_prompt)
        
        # Process agent response
        if "QuantumReranker" in agent_response:
            # Agent chose quantum reranker
            reranked_docs = self.quantum_reranker.rerank(
                query_text, 
                retrieved_docs,
                top_k=query.top_k
            )
            reranker_used = "quantum"
            logger.info(f"Agent selected quantum reranking: {agent_response[:100]}")
        else:
            # Agent chose classical reranker or we couldn't determine
            reranked_docs = self.classical_reranker.rerank(
                query_text, 
                retrieved_docs,
                top_k=query.top_k
            )
            reranker_used = "classical"
            logger.info(f"Agent selected classical reranking: {agent_response[:100]}")
            
        return reranked_docs, reranker_used
    
    async def _traditional_process_query(self, query: Query, retrieved_docs: List[Document]):
        """Process query using traditional method"""
        # First, decide whether to use quantum or classical reranking
        use_quantum = self._should_use_quantum(query, retrieved_docs)
        
        # Override with explicit user preference if specified
        if query.use_quantum is not None:
            use_quantum = query.use_quantum
            
        # Apply the selected reranking method
        if use_quantum:
            logger.info(f"Using quantum reranking for query: {query.text}")
            reranked_docs = self.quantum_reranker.rerank(
                query.text, 
                retrieved_docs,
                top_k=query.top_k
            )
            reranker_used = "quantum"
        else:
            logger.info(f"Using classical reranking for query: {query.text}")
            reranked_docs = self.classical_reranker.rerank(
                query.text, 
                retrieved_docs,
                top_k=query.top_k
            )
            reranker_used = "classical"
            
        return reranked_docs, reranker_used
    
    def _should_use_quantum(self, query: Query, docs: List[Document]) -> bool:
        """
        Determine whether to use quantum reranking based on query characteristics.
        
        Args:
            query: The user's query
            docs: Retrieved documents
            
        Returns:
            Boolean indicating whether to use quantum reranking
        """
        # Assess query complexity
        query_complexity = self._assess_query_complexity(query.text)
        
        # Assess query type
        query_type_score = self._assess_query_type_score(query.text)
        
        # Calculate final score
        final_score = 0.7 * query_complexity + 0.3 * query_type_score
        
        # Use quantum if score exceeds threshold
        return final_score > self.quantum_threshold
        
    def _assess_query_complexity(self, query_text: str) -> float:
        """
        Assess the complexity of a query on a scale from 0.0 to 1.0.
        Higher values indicate more complex queries that might benefit from quantum methods.
        
        Args:
            query_text: The query text
            
        Returns:
            Complexity score between 0.0 and 1.0
        """
        # Simple heuristics for complexity
        # In a real implementation, this could use more sophisticated NLP
        
        # Word count (more words = more complex)
        words = query_text.split()
        word_count = len(words)
        
        # Sentence count
        import re
        sentences = re.split(r'[.!?]+', query_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)
        
        # Term complexity (longer words = more complex)
        avg_word_length = sum(len(word) for word in words) / max(1, len(words))
        
        # Calculate complexity score
        complexity = min(1.0, (
            (0.1 * word_count / 10) +  # Normalize for ~10 words
            (0.2 * sentence_count / 2) +  # Normalize for ~2 sentences
            (0.3 * (avg_word_length - 3) / 5)  # Normalize for word length
        ))
        
        return max(0.0, complexity)  # Ensure non-negative
    
    def _assess_query_type(self, query_text: str) -> str:
        """
        Assess the type of query.
        
        Args:
            query_text: The query text
            
        Returns:
            String indicating query type
        """
        query_lower = query_text.lower()
        
        # Check for different query types
        if any(term in query_lower for term in ['compare', 'difference', 'versus', 'vs']):
            return "comparative"
        elif any(term in query_lower for term in ['why', 'how', 'explain']):
            return "explanatory"
        elif any(term in query_lower for term in ['when', 'where', 'who', 'what']):
            return "factual"
        elif any(term in query_lower for term in ['forecast', 'predict', 'future', 'trend']):
            return "predictive"
        else:
            return "general"
            
    def _assess_query_type_score(self, query_text: str) -> float:
        """
        Calculate a score for query type indicating suitability for quantum methods.
        
        Args:
            query_text: The query text
            
        Returns:
            Score between 0.0 and 1.0
        """
        query_type = self._assess_query_type(query_text)
        
        # Score different query types based on suitability for quantum methods
        type_scores = {
            "comparative": 0.9,  # Comparing items benefits from quantum
            "explanatory": 0.7,  # Explanations may benefit from quantum
            "predictive": 0.8,   # Predictions may benefit from quantum
            "factual": 0.3,      # Simple facts don't need quantum
            "general": 0.5       # General queries are neutral
        }
        
        return type_scores.get(query_type, 0.5)
        # Simple decision logic - can be expanded with more sophisticated criteria
        # This is just a placeholder implementation
        
        # Criteria that might favor quantum reranking:
        # 1. Complex queries with multiple concepts
        # 2. Queries about forecasting or predictive analysis
        # 3. Queries with high ambiguity
        # 4. Large number of documents that might benefit from quantum advantage
        
        score = 0.0
        
        # Check for energy forecasting keywords
        forecast_keywords = ["forecast", "prediction", "future", "trend", "projected"]
        if any(keyword in query.text.lower() for keyword in forecast_keywords):
            score += 0.3
            
        # Check for complex query (based on length as a simple heuristic)
        if len(query.text.split()) > 10:
            score += 0.2
            
        # Check for document volume
        if len(docs) > 50:
            score += 0.2
            
        # Example: If query explicitly mentions quantum
        if "quantum" in query.text.lower():
            score += 0.3
            
        logger.debug(f"Query '{query.text}' quantum score: {score}")
        
        # Return true if score exceeds threshold
        return score >= self.quantum_threshold
