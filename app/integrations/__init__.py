"""
Integration Module

This module provides integrations with external libraries like LangChain.
"""
import logging

logger = logging.getLogger(__name__)

# Try to import LangChain
try:
    from langchain.schema.runnable import RunnablePassthrough
    LANGCHAIN_AVAILABLE = True
    logger.info("LangChain is available for integrations")
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain is not available. Some integrations will be disabled.")

# Import pipeline if LangChain is available
if LANGCHAIN_AVAILABLE:
    try:
        from app.integrations.langchain_pipeline import LangChainPipeline
        logger.info("LangChain pipeline module loaded")
    except ImportError as e:
        logger.warning(f"Error loading LangChain pipeline: {str(e)}")
