# Quantum-Enhanced Energy RAG Pipeline

A research project exploring quantum computation in document reranking for energy data analysis using RAG (Retrieval-Augmented Generation) systems.

## Overview

This project investigates quantum-enhanced reranking within RAG pipelines, specifically for energy forecast analysis. It maintains a classical retrieval pipeline while comparing quantum and classical reranking approaches for document similarity.

### Key Features

- **Quantum Reranking**: Quantum circuit-based document similarity comparison using Qiskit
- **Classical Baseline**: Traditional reranking methods for performance comparison  
- **Intelligent Controller**: Automatic selection between quantum and classical reranking
- **Energy Focus**: Specialized for energy data analysis and forecasting
- **Vector Storage**: PostgreSQL with pgvector for document embeddings

## API Endpoints

The FastAPI application provides the following endpoints:

- `POST /energy/query`: Query energy data with classical, quantum, or automatic reranking
- `POST /energy/forecast/`: Generate energy forecasts
- `POST /embeddings/create`: Create document embeddings
- `POST /embeddings/search`: Search documents by similarity
- `POST /pgvector/find_similar_documents`: Vector similarity search
- `POST /pgvector/execute_query`: Execute custom vector queries

Access interactive API documentation at `http://localhost:8000/docs` when running.

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   Configure your environment variables for database and API access:
   ```bash
   # Add to .env file or export
   POSTGRES_CONNECTION_STRING=your_postgres_url
   OPENAI_API_KEY=your_openai_key
   # Add other required credentials
   ```

3. **Run the Application**
   ```bash
   python app.py
   # or
   uvicorn app:app --reload
   ```

4. **Access API Documentation**
   Open `http://localhost:8000/docs` for interactive API documentation.

## Architecture

- **Quantum Reranking** (`src/reranker/quantum.py`): Quantum circuit-based similarity computation
- **Classical Reranking** (`src/reranker/classical.py`): Traditional reranking methods
- **Controller** (`src/reranker/controller.py`): Intelligent selection between approaches
- **Vector Storage** (`src/storage/pgvector_storage.py`): PostgreSQL with pgvector extension
- **Embeddings** (`src/embeddings/`): Document and query embedding utilities

## Dependencies

Key dependencies include:
- **Qiskit**: Quantum computing framework
- **FastAPI**: Web framework for APIs
- **PostgreSQL + pgvector**: Vector database for embeddings
- **Transformers**: Neural language models
- **Atomic Agents**: Agent framework components
