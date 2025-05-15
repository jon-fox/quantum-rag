# Quantum Semantic Reranking in RAG Pipelines for Energy Forecast Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project explores whether a hybrid quantum-classical retrieval architecture can improve document reranking in Retrieval-Augmented Generation (RAG) pipelines specifically for forecasting and operational mismatch analysis in ERCOT energy data compared to classical similarity methods.

### Research Question
Can a quantum circuit-based similarity function improve document reranking in RAG pipelines for energy system analysis compared to classical similarity methods?

## System Architecture

![Architecture Diagram](https://via.placeholder.com/800x400?text=Quantum+RAG+Architecture+Diagram)

### 1. Retriever (Classical)
- Dense retriever (e.g., FAISS, SBERT, or Cohere) fetches top-k relevant chunks from a vector DB

### 2. Quantum Reranker
- Each document and query are encoded into a quantum state (angle/amplitude encoding)
- Uses quantum circuit to:
  - Estimate inner product (similarity)
  - Sort or re-rank based on quantum scores
- Tools: Qiskit (or PennyLane), simulated locally

### 3. LLM Generator
- Final context is fed into an LLM (e.g., GPT-4 API, Claude, Gemini, or open-source like Mistral)
- Compares outputs across different reranking methods (classical vs quantum)
- Optionally compares performance across LLMs (OpenAI, Claude, Gemini)

### 4. Agent Controller
- Agent evaluates the input query and decides:
  - Whether to invoke classical or quantum reranking
  - Which retriever to use
  - How many documents to pass into the generator
- Uses LangChain or custom routing logic
- Logs decisions and adapts based on prior outcomes

## Experimental Plan

| Step | Description |
|------|-------------|
| 🔍 **Baseline** | Classical reranking using cosine similarity or SBERT dot product |
| ⚛️ **Quantum Reranking** | Replace reranker with quantum similarity module using Qiskit or PennyLane |
| 🤖 **Agent-Controlled Flow** | Add decision-making logic to choose between rerankers based on query features |
| 💬 **LLM Performance Testing** | Compare outputs using OpenAI, Claude, and Gemini on same reranked context |
| 📊 **Evaluation** | Use QA tasks built from ERCOT forecast vs. actual load reports |
| 💻 **Hardware** | Start with local simulators, test on real IBM hardware if needed |

## Expected Deliverables

| Deliverable | Description |
|-------------|-------------|
| 🧩 **Prototype** | Working hybrid RAG system with quantum reranker |
| 🎛️ **Agent Layer** | Intelligent controller for reranking logic using LangChain or FSM |
| 📈 **Evaluation Results** | Metric comparisons: classical vs quantum vs agent-controlled |
| 📝 **LLM Comparison Report** | Analysis of LLM output performance across providers |
| 📄 **Final Report** | 6–8 page research paper draft |
| 🌐 **GitHub Repo** | Public codebase with README, notebooks/scripts |

## Timeline

| Timeframe | Goal |
|-----------|------|
| 🌱 **Summer (Pre-Semester)** | Set up development environment, complete Qiskit tutorials, build initial RAG pipeline |
| 📊 **Weeks 1–2** | Begin benchmarking baseline RAG system |
| ⚛️ **Weeks 3–8** | Develop and test quantum reranker module and agent routing logic |
| 🔄 **Weeks 9–10** | Finalize and test integrated agent routing logic |
| 🤖 **Weeks 11–12** | Integrate LLM variants and run evaluations |
| 📝 **Weeks 13–14** | Finalize report and GitHub repository |

## Technologies & Tools

| Tool | Purpose |
|------|---------|
| ![Qiskit](https://img.shields.io/badge/Qiskit-6929C4?style=for-the-badge&logo=qiskit&logoColor=white) | Quantum circuit simulation and execution |
| ![FAISS](https://img.shields.io/badge/FAISS-00BFFF?style=for-the-badge) | Dense retrieval backend |
| ![LangChain](https://img.shields.io/badge/LangChain-00C48C?style=for-the-badge) | Tool orchestration and agent behavior |
| ![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white) | Text generation via LLMs |
| ![Dataset](https://img.shields.io/badge/ERCOT_Data-FF5722?style=for-the-badge) | Primary dataset for document retrieval |
| ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white) | Interactive demos |

## Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/quantum-rag.git
cd quantum-rag

# Install dependencies
pip install -r requirements.txt

# Run example notebook
jupyter notebook examples/quantum_reranker_demo.ipynb
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- **Project Lead**: Jonathan Fox
- **Email**: foxj7@nku.edu
- **GitHub**: [@jon-fox](https://github.com/jon-fox)
