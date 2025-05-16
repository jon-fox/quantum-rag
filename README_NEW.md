# RAG Energy App with Quantum Reranking

This project investigates whether a quantum-enhanced reranking component can improve Retrieval-Augmented Generation (RAG) systems. The research maintains a classical RAG pipeline while replacing only the reranking mechanism with a quantum alternative to evaluate its effectiveness for energy forecast analysis using ERCOT data.

## Project Structure

```
rag-energy-app/
├── app/                      # Main application code
│   ├── main.py               # FastAPI entry point
│   ├── api/                  # Route definitions
│   ├── config/               # Configs, env loaders
│   ├── embeddings/           # Embedding logic
│   ├── vector_store/         # FAISS index and retriever
│   ├── reranker/
│   │   ├── classical.py      # Classical reranking implementation
│   │   └── quantum.py        # Quantum reranking implementation
│   ├── generator/            # LLM interface
│   ├── agent/                # Agent routing logic
│   ├── utils/                # Text preprocessors, helpers
│   └── schema/               # Request/response models
│
├── data/                     # Source ERCOT documents (raw, processed)
├── ingestion/                # Data ingestion pipeline
│   ├── ercot/
│   │   ├── fetch_api.py      # Calls ERCOT APIs
│   │   ├── parse_reports.py  # Extract/clean data from PDFs or CSVs
│   │   └── schedule.py       # Job runner if needed
│   └── pipeline.py           # Ingest + clean + write to vector store
```

## Key Components

1. **Reranker Module**: Contains both classical and quantum implementations of document reranking algorithms
2. **Vector Store**: Handles document storage and retrieval using FAISS or an in-memory alternative
3. **Agent**: Intelligently selects between classical and quantum reranking based on query characteristics
4. **Ingestion Pipeline**: Fetches, processes, and stores ERCOT energy data

## Getting Started

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
uvicorn app.main:app --reload
```

3. Access the API documentation at http://localhost:8000/docs

## Running the Ingestion Pipeline

```bash
python -m ingestion.pipeline
```

## Research Objectives

This project focuses on:
- Developing a quantum circuit-based reranker within a primarily classical RAG pipeline
- Encoding documents and queries into quantum states for similarity comparison
- Comparing classical vs. quantum reranking approaches with identical retrieval and generation components
- Building an agent-based controller to intelligently select between classical or quantum reranking based on query characteristics
