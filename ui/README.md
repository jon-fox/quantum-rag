# Quantum Energy RAG Pipeline - Streamlit UI

A simple web interface for the Quantum Energy RAG Pipeline FastAPI backend.

## Features

- 🔍 Search energy data using natural language queries
- ⚙️ Choose between classical, quantum, or auto reranking
- 📊 View detailed search results with scores and metadata
- 🔗 Real-time API health monitoring

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Make sure your FastAPI backend is running:**
   ```bash
   # From the root directory
   python app.py
   ```
   The API should be accessible at `http://localhost:8000`

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser to:**
   ```
   http://localhost:8501
   ```

## Usage

1. Enter a natural language query about energy data (e.g., "forecast vs actual load during summer peak hours")
2. Choose your preferred reranker type (auto, classical, or quantum)
3. Set the number of results you want
4. Click "Search" to query the backend
5. Review the results with similarity scores and metadata

## Configuration

The app is configured to connect to FastAPI at `http://localhost:8000` by default. You can modify the `FASTAPI_BASE_URL` variable in `app.py` to point to a different endpoint.

## Example Queries

* forecast vs actual telemetry generation in May–June 2025 during peak hours (2–7 PM)
* renewable contribution trends in ERCOT: wind vs solar breakdown May–June 2025
* ERCOT demand spikes: comparison of base-point and dispatchable generation May–June 2025
* daily solar generation variation and its correlation with average Texas temperature in May–June 2025

