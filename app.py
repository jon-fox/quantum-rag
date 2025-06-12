"""
FastAPI application for quantum-enhanced reranking in RAG pipelines
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import os  # for LOG_LEVEL

from src.api.energy_query_api import router as energy_router
from src.api.embedding_api import create_embedding_router # Import the function
from src.api.pgvector_query_api import router as pgvector_api_router 
from src.api.energy_forecast import router as forecast_router 

from src.config.env_manager import load_environment

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
numeric_level = getattr(logging, log_level, logging.INFO)
logging.basicConfig(level=numeric_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Log level set to {log_level}")

# Load environment variables at startup
load_environment()

# Create FastAPI app
app = FastAPI(
    title="Quantum-Enhanced Energy RAG Pipeline",
    description="API for quantum-enhanced reranking of energy data in RAG pipelines",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(energy_router, prefix="/energy", tags=["Energy Queries"])
app.include_router(create_embedding_router(), prefix="/embeddings", tags=["Embedding Operations"]) # Call the function here
app.include_router(pgvector_api_router, prefix="/pgvector", tags=["PgVector Operations"]) 
app.include_router(forecast_router, prefix="/energy/forecast", tags=["Energy Forecasting"]) 


# Add a simple root endpoint
@app.get("/")
async def root():
    """Root endpoint that provides basic information about the API."""
    return {
        "message": "Quantum-Enhanced Energy RAG Pipeline API",
        "docs_url": "/docs",
        "version": "0.1.0",
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
