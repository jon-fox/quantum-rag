# Quantum Semantic Reranking in RAG Pipelines

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Resources

| Resource | Link |
|----------|------|
| Qiskit (Quantum SDK) | https://github.com/Qiskit/qiskit |
| Project Discussion | https://chatgpt.com/c/68254a64-f578-8001-b942-33e437225165 |
| Original Proposal | https://docs.google.com/document/d/19WuIULxvqFG6xaQ2Sa7sYMlx8o4hZBwX4khceGAqRag/edit?tab=t.0 |
| Atomic Agents | https://github.com/BrainBlend-AI/atomic-agents |

## Project Objective

This graduate independent study project investigates whether a quantum-enhanced reranking component can improve Retrieval-Augmented Generation (RAG) systems. The research maintains a classical RAG pipeline while replacing only the reranking mechanism with a quantum alternative to evaluate its effectiveness for energy forecast analysis using ERCOT data.

The research focuses on:
- Developing a quantum circuit-based reranker within a primarily classical RAG pipeline
- Encoding documents and queries into quantum states for similarity comparison
- Comparing classical vs. quantum reranking approaches with identical retrieval and generation components
- Building an agent-based controller to intelligently select between classical or quantum reranking based on query characteristics
- Evaluating end-to-end performance across multiple LLM backends
- Analyzing real-time ERCOT energy data for market trend predictions

## API Endpoints

The project provides a focused API for energy data analysis:

- `POST /api/energy/query`: Query ERCOT energy data using either classical, quantum, or automatic reranking selection
- `GET /api/energy/health`: Simple health check endpoint

## ERCOT API Integration

The project includes integration with the ERCOT (Electric Reliability Council of Texas) API for accessing real-time and historical energy data:

- Authentication with automatic token refresh every 55 minutes (tokens expire after 60 minutes)
- Access to real-time pricing data, historical load data, and forecasts
- Environment variable management for secure credential storage

### Setup ERCOT API Credentials

1. Copy the `.env.example` file to `.env` in the project root
2. Add your ERCOT API credentials to the `.env` file:
   ```
   ERCOT_API_USERNAME=your-username
   ERCOT_API_PASSWORD=your-password
   ```
3. The application will automatically load these credentials on startup

### Example Usage

The `examples/ercot_api_example.py` file demonstrates how to use the ERCOT API client and queries module:

```python
# Initialize ERCOT queries helper
queries = ERCOTQueries(client)

# Get 2-Day Aggregated Generation Summary
gen_summary = queries.get_aggregated_generation_summary(
    delivery_date_from="2025-05-17",
    delivery_date_to="2025-05-18"
)

# Get 2-Day Aggregated Load Summary for Houston region
load_houston = queries.get_aggregated_load_summary(
    delivery_date_from="2025-05-17",
    delivery_date_to="2025-05-18",
    region="Houston"
)

# Get 2-Day Aggregated Ancillary Service Offers
ancillary = queries.get_ancillary_service_offers(
    service_type="REGUP",
    delivery_date_from="2025-05-17",
    delivery_date_to="2025-05-18"
)
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/quantum-rag.git
cd quantum-rag

# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn app:app --reload

# Access API documentation
# Open http://localhost:8000/docs in your browser
```

This work explores the targeted application of quantum computation to a specific NLP pipeline component, with potential applications in energy system forecasting and operational analysis.
