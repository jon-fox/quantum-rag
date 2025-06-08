"""
Prompt templates for energy forecasting using retrieval and reranking.
"""

ENERGY_FORECAST_PROMPT = """You're an energy analyst. Based on the following query and top relevant data points, provide a 150-word human-readable forecast highlighting load vs telemetry trends.

Query: {query}

Top 5 Relevant Data Points:
{relevant_data}

Provide a concise 150-word forecast focusing on load vs telemetry trends:"""

def build_energy_forecast_prompt(query: str, reranked_results: list) -> str:
    """
    Build the energy forecast prompt with the given query and reranked results.
    
    Args:
        query: The user's query
        reranked_results: List of (Document, score) tuples from reranking
        
    Returns:
        Formatted prompt string
    """
    relevant_data = ""
    for i, (doc, score) in enumerate(reranked_results[:5], 1):
        # Extract date and key metrics from document content
        content_preview = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
        relevant_data += f"{i}. {content_preview}\n"
    
    return ENERGY_FORECAST_PROMPT.format(
        query=query,
        relevant_data=relevant_data
    )
