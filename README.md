# Quantum RAG Reranker

A simple research project exploring quantum computation in document reranking for podcast ad detection using RAG (Retrieval-Augmented Generation).

## Overview

This project compares quantum and classical reranking approaches for document similarity in the context of detecting advertisements in podcast transcripts.

### Key Features

- **Quantum Reranking**: Quantum circuit-based document similarity using Qiskit
- **Classical Baseline**: Traditional reranking methods for comparison  
- **Podcast Ad Detection**: Focused on identifying ads in podcast content

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   python app.py
   ```

3. **Access API Documentation**
   Open `http://localhost:8000/docs` for interactive API documentation.

## Core Components

- **Quantum Reranking** (`src/reranker/quantum.py`): Quantum circuit-based similarity computation
- **Classical Reranking** (`src/reranker/classical.py`): Traditional reranking methods
- **Controller** (`src/reranker/controller.py`): Automatic selection between approaches
