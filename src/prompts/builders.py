"""
Prompt builders for various energy analysis tasks.
"""

from src.prompts.energy_forecast import ENERGY_FORECAST_PROMPT


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
        preview = doc.content.strip().replace('\n', ' ')
        preview = preview[:200] + "..." if len(preview) > 200 else preview
        relevant_data += f"{i}. {preview}\n"

    return ENERGY_FORECAST_PROMPT.format(
        query=query,
        relevant_data=relevant_data
    )
