"""
Example script demonstrating LangChain integration with the quantum reranking project.
"""
import asyncio
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.integrations.langchain_pipeline import LangChainPipeline
from app.schema.models import Document
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """Main function demonstrating LangChain integration"""
    
    logger.info("Initializing LangChain pipeline...")
    pipeline = LangChainPipeline()
    
    # Example documents about ERCOT
    documents = [
        Document(
            id=str(uuid.uuid4()),
            content="ERCOT (Electric Reliability Council of Texas) manages the flow of electric power to more than 26 million Texas customers, representing about 90 percent of the state's electric load.",
            source="ercot_info",
            metadata={"category": "organization", "year": 2023}
        ),
        Document(
            id=str(uuid.uuid4()),
            content="In February 2021, Texas experienced a major power crisis due to three severe winter storms. ERCOT ordered rolling blackouts to prevent grid failure as electricity demand exceeded supply.",
            source="ercot_crisis",
            metadata={"category": "event", "year": 2021}
        ),
        Document(
            id=str(uuid.uuid4()),
            content="The Texas Interconnection is one of the three main grids in the U.S. power transmission system. Unlike other states, Texas maintains its own power grid which is managed by ERCOT.",
            source="texas_grid",
            metadata={"category": "infrastructure", "year": 2023}
        ),
        Document(
            id=str(uuid.uuid4()),
            content="ERCOT forecasts show a 15% increase in summer energy demand for 2025 compared to 2023 levels. This is attributed to population growth and increased adoption of data centers across the state.",
            source="demand_forecast",
            metadata={"category": "forecast", "year": 2024}
        ),
        Document(
            id=str(uuid.uuid4()),
            content="Texas added 7.3GW of new solar power capacity in 2024, making it the leading state for renewable energy growth. ERCOT expects solar to comprise 25% of generation capacity by 2026.",
            source="renewables_growth",
            metadata={"category": "energy_mix", "year": 2024}
        )
    ]
    
    # Add documents to the pipeline
    logger.info("Adding example documents to the vector store...")
    await pipeline.add_documents(documents)
    
    # Example queries
    queries = [
        ("What is ERCOT and what does it do?", False),
        ("What happened to the Texas power grid in 2021?", False),
        ("Compare Texas power grid to other states", True),
        ("What are the energy demand forecasts for 2025?", True),
        ("How much solar capacity was added in 2024?", False)
    ]
    
    # Process each query with both classical and quantum reranking
    for query_text, use_quantum in queries:
        reranker_type = "quantum" if use_quantum else "classical"
        logger.info(f"Processing query with {reranker_type} reranking: {query_text}")
        
        response = await pipeline.query(query_text, use_quantum)
        
        print(f"\nQuery: {query_text}")
        print(f"Using {reranker_type} reranking")
        print(f"Response: {response['response']}")
        print(f"Execution time: {response['execution_time_ms']:.2f}ms")
        print("-" * 80)

if __name__ == "__main__":
    asyncio.run(main())
