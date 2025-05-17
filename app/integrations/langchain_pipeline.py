"""
LangChain Full RAG Pipeline Integration

This module sets up a complete RAG pipeline using LangChain components
for use with both classical and quantum reranking.
"""
import os
import logging
from typing import Dict, List, Any, Optional, Union

try:
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser
    from langchain.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from app.embeddings.embeddings import get_embedding_model
    from app.reranker.classical import ClassicalReranker
    from app.reranker.quantum import QuantumReranker
    from app.schema.models import Document, Query
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available. Unable to set up LangChain pipeline.")

class LangChainPipeline:
    """
    A full LangChain RAG pipeline that integrates with quantum and classical rerankers.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LangChain pipeline.
        
        Args:
            config: Configuration dictionary
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for this pipeline")
            
        self.config = config or {}
        self.index_path = self.config.get("index_path", "./data/vectors")
        self.embedding_model = get_embedding_model()
        
        # Initialize LLM
        model_name = self.config.get("model_name", "gpt-3.5-turbo")
        temperature = self.config.get("temperature", 0.7)
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        
        # Initialize rerankers
        self.classical_reranker = ClassicalReranker(self.config.get("classical_config"))
        self.quantum_reranker = QuantumReranker(self.config.get("quantum_config"))
        
        # Set up vector store
        self._setup_vector_store()
        
        # Create the prompt
        self.prompt = ChatPromptTemplate.from_template("""
        You are an energy analysis assistant with expertise in ERCOT (Electric Reliability Council of Texas) data.
        
        Answer the following question based only on the provided context:
        
        Question: {question}
        
        Context:
        {context}
        
        Answer:
        """)
        
    def _setup_vector_store(self):
        """Set up the vector store"""
        # Check if FAISS index exists
        if os.path.exists(os.path.join(self.index_path, "index.faiss")):
            self.vector_store = FAISS.load_local(
                self.index_path,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            # Create a placeholder store - will need to be populated
            self.vector_store = FAISS.from_texts(
                texts=["This is a placeholder document"],
                embedding=self.embedding_model
            )
    
    async def _format_docs(self, docs):
        """Format documents for the prompt"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _convert_to_langchain_docs(self, documents):
        """Convert app Document objects to LangChain document format"""
        from langchain.schema.document import Document as LCDocument
        
        result = []
        for doc in documents:
            result.append(LCDocument(
                page_content=doc.content,
                metadata={"id": doc.id, "source": doc.source, **doc.metadata}
            ))
        return result
    
    def _convert_from_langchain_docs(self, lc_docs):
        """Convert LangChain documents to app Document format"""
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
    
    def _rerank_documents(self, query, docs, use_quantum=False, top_k=5):
        """Rerank documents using the appropriate reranker"""
        # Convert LangChain docs to app format for reranking
        app_docs = self._convert_from_langchain_docs(docs)
        
        # Apply reranking
        if use_quantum:
            reranked_docs = self.quantum_reranker.rerank(query, app_docs, top_k)
        else:
            reranked_docs = self.classical_reranker.rerank(query, app_docs, top_k)
        
        # Convert back to LangChain format
        return self._convert_to_langchain_docs(reranked_docs)

    def create_chain(self, use_quantum=False):
        """
        Create a LangChain RAG chain.
        
        Args:
            use_quantum: Whether to use quantum reranking
            
        Returns:
            A runnable chain
        """
        # Create retriever that does base retrieval
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        
        # Define reranking function
        def rerank(inputs):
            query = inputs["question"]
            docs = inputs["docs"]
            return self._rerank_documents(query, docs, use_quantum=use_quantum, top_k=5)
            
        # Create the chain
        rag_chain = (
            {"question": RunnablePassthrough(), "docs": retriever}
            | {"question": RunnablePassthrough(), "context": rerank | self._format_docs}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
        
    async def query(self, query_text: str, use_quantum: bool = False) -> Dict[str, Any]:
        """
        Run a query through the RAG pipeline.
        
        Args:
            query_text: The query text
            use_quantum: Whether to use quantum reranking
            
        Returns:
            Dictionary with response and metadata
        """
        import time
        start_time = time.time()
        
        # Create the chain
        chain = self.create_chain(use_quantum=use_quantum)
        
        # Run the chain
        response = await chain.ainvoke(query_text)
        
        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000
        
        return {
            "query": query_text,
            "response": response,
            "use_quantum": use_quantum,
            "execution_time_ms": execution_time_ms
        }
        
    async def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        # Convert to LangChain format
        texts = [doc.content for doc in documents]
        metadatas = [{"id": doc.id, "source": doc.source, **doc.metadata} for doc in documents]
        
        # Add to vector store
        self.vector_store.add_texts(texts=texts, metadatas=metadatas)
        
        # Save the updated store
        self.vector_store.save_local(self.index_path)
