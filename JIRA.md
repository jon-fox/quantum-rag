# Project Tasks for Quantum Reranking RAG Pipeline

This document outlines the tasks for developing the quantum-enhanced RAG pipeline for ERCOT data analysis.

| Phase | Task ID | Task Description | Affected Files/Modules | Status | Notes/Actions |
|---|---|---|---|---|---|
| **Phase 1: Solidify Data Ingestion & Initial Processing (ERCOT API)** | | | | | |
| 1 | 1.1 | **Verify and Finalize ERCOT API Endpoints** | `src/data/ercot_api/queries.py` | To Do |  |
| 1 | 1.1.1 | Sub-Task: Correct and Enhance `get_aggregated_load_summary` | `src/data/ercot_api/queries.py` | To Do | Confirm report ID (e.g., `np3-910-er`). Implement dynamic `endpoint_suffix` for region and overall. Verify default overall suffix from ERCOT docs. |
| 1 | 1.1.2 | Sub-Task: Correct and Enhance `get_agg_gen_summary` | `src/data/ercot_api/queries.py` | To Do | Confirm report ID (e.g., `np3-911-er` vs `np3-910-er`). Resolve `endpoint_suffix` naming (`_summary` vs `_sum`) and regional variations. Verify from ERCOT docs. Uncomment region logic. Add logging for region. |
| 1 | 1.2 | **Define Data Schemas for ERCOT Responses** | `src/schema/models.py` | To Do | Create Pydantic models for JSON responses from all used ERCOT endpoints. |
| 1 | 1.3 | **Implement Initial Data Transformation** | `src/data_processing/ercot_transformer.py` (new) | To Do | Develop functions to transform raw ERCOT JSON (from `ERCOTQueries`) into Pydantic models (from Task 1.2). Include basic cleaning. |
| 1 | 1.4 | **Comprehensive ERCOT Client Unit Testing** | `src/tests/data/ercot_api/`, `examples/ercot_api_example.py` | To Do | Write unit tests for `queries.py` methods, mocking API responses. Update `ercot_api_example.py`. |
| **Phase 2: Build Core RAG Pipeline Components (Pre-Generation)** | | | | | |
| 2 | 2.1 | **Document and Implement Embedding Strategy** | `src/embeddings/README.md` (new/update), `src/embeddings/embed_utils.py` | To Do | Document *what* from processed ERCOT data gets embedded and why. Implement functions to generate embeddings. |
| 2 | 2.2 | **Set Up and Integrate Vector Storage** | `src/storage/pgvector_storage.py`, `src/config/env_manager.py` | To Do | Configure vector DB. Implement functions for storing data+embeddings and performing similarity searches. |
| 2 | 2.3 | **Develop Classical Reranker** | `src/reranker/classical.py` | To Do | Implement at least one classical reranking algorithm (e.g., cross-encoder). |
| 2 | 2.4 | **Develop Quantum Reranker (Core Logic)** | `src/reranker/quantum.py` | To Do | Define quantum state encoding, implement quantum circuits for similarity/reranking, and logic for translating measurements to a reranked list. |
| 2 | 2.5 | **Create a Dedicated Reranking Test API Endpoint** | `src/api/rerank_test_api.py` (new) | To Do | FastAPI endpoint `POST /api/v1/rerank/test` accepting query, documents, reranker_type. Returns reranked documents. |
| 2 | 2.6 | **Develop Reranker Controller/Agent** | `src/reranker/controller.py` | To Do | Implement logic to choose between classical or quantum reranking. |
| **Phase 3: Integrate Full Pipeline and Expose Main Query Endpoint** | | | | | |
| 3 | 3.1 | **Orchestrate the RAG Pipeline (excluding LLM generation)** | `src/agents/pipeline_orchestrator.py` (new) | To Do | Manage end-to-end flow: query -> fetch -> transform -> embed -> retrieve -> rerank -> output reranked documents. |
| 3 | 3.2 | **Implement Main Energy Query API Endpoint** | `src/api/energy_query_api.py` | To Do | Implement/refine `POST /api/energy/query`. Use `PipelineOrchestrator`. Initially return structured, reranked data. |
| **Phase 4: LLM Integration and Evaluation (Future)** | | | | | |
| 4 | 4.1 | **Integrate LLM for Response Generation** | `src/agents/pipeline_orchestrator.py`, LLM integration modules | To Do | Add LLM call to generate natural language response from reranked documents. |
| 4 | 4.2 | **Define and Implement Evaluation Metrics & Strategy** | Documentation, `src/evaluation/` (new) | To Do | Define metrics for reranking effectiveness and overall RAG quality. Implement evaluation scripts. |

