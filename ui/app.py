import streamlit as st
import requests
import json
from typing import Dict, Any, List

# Configure Streamlit page
st.set_page_config(
    page_title="Quantum Energy RAG Pipeline",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
FASTAPI_BASE_URL = "http://localhost:8000"

def check_api_health() -> bool:
    """Check if the FastAPI backend is running"""
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/energy/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def query_energy_data(query: str, limit: int = 10, reranker_type: str = "auto") -> Dict[str, Any]:
    """Query energy data from the FastAPI backend"""
    try:
        payload = {
            "query": query,
            "limit": limit,
            "reranker_type": reranker_type
        }
        response = requests.post(
            f"{FASTAPI_BASE_URL}/energy/query",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying API: {str(e)}")
        return {}

def main():
    st.title("⚡ Quantum Energy RAG Pipeline")
    st.markdown("Search and analyze energy data using quantum-enhanced reranking")
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    
    # API Health Check
    if check_api_health():
        st.sidebar.success("✅ API is running")
    else:
        st.sidebar.error("❌ API is not accessible")
        st.sidebar.markdown(f"Make sure FastAPI is running at: `{FASTAPI_BASE_URL}`")
    
    # Query parameters
    st.sidebar.subheader("Query Parameters")
    reranker_type = st.sidebar.selectbox(
        "Reranker Type",
        ["auto", "classical", "quantum"],
        index=0,
        help="Choose the reranking algorithm"
    )
    
    limit = st.sidebar.slider(
        "Number of Results",
        min_value=1,
        max_value=20,
        value=5,
        help="Maximum number of results to return"
    )
    
    # Main content area
    st.header("Energy Data Query")
    
    # Query input
    query = st.text_input(
        "Enter your energy-related query:",
        placeholder="e.g., difference between DAM forecast and telemetry generation in May and June 2025 during afternoon peak",
        help="Enter a natural language query about energy data, forecasts, or efficiency"
    )
    
    # Example queries
    st.subheader("Example Queries")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Peak Load Analysis"):
            query = "forecast vs actual load during summer peak hours"
            
    with col2:
        if st.button("Energy Efficiency"):
            query = "energy efficiency trends renewable sources"
    
    # Search button and results
    if st.button("🔍 Search", type="primary") and query:
        with st.spinner("Searching energy data..."):
            results = query_energy_data(query, limit, reranker_type)
            
            if results:
                st.success(f"Found {len(results.get('results', []))} results")
                
                # Display query info
                st.subheader("Query Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Query:** {results.get('query', 'N/A')}")
                with col2:
                    st.info(f"**Reranker Used:** {results.get('reranker_used', 'N/A')}")
                
                # Display results
                st.subheader("Search Results")
                
                for i, result in enumerate(results.get('results', []), 1):
                    document = result.get('document', {})
                    score = result.get('score', 0)
                    
                    with st.expander(f"Result {i} (Score: {score:.4f})", expanded=i==1):
                        st.write("**Content:**")
                        st.write(document.get('content', 'No content available'))
                        
                        if document.get('source'):
                            st.write(f"**Source:** {document.get('source')}")
                        
                        if document.get('metadata'):
                            st.write("**Metadata:**")
                            st.json(document.get('metadata'))
            else:
                st.warning("No results found or API error occurred")
    
    elif not query and st.button("🔍 Search", type="primary"):
        st.warning("Please enter a query")
    
    # API Information
    st.sidebar.markdown("---")
    st.sidebar.subheader("API Information")
    st.sidebar.markdown(f"**Base URL:** `{FASTAPI_BASE_URL}`")
    st.sidebar.markdown("**Available Endpoints:**")
    st.sidebar.markdown("- `/energy/query` - Search energy data")
    st.sidebar.markdown("- `/energy/health` - Health check")
    st.sidebar.markdown("- `/embeddings/*` - Embedding operations")
    st.sidebar.markdown("- `/pgvector/*` - Vector operations")
    
    # Footer
    st.markdown("---")
    st.markdown("**About:** This app demonstrates quantum-enhanced reranking for energy data search and analysis.")

if __name__ == "__main__":
    main()
