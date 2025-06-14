"""
Prompt builders for various energy analysis tasks.
"""

from src.prompts.energy_forecast import ENERGY_FORECAST_PROMPT
from src.prompts.trend_analysis import TREND_ANALYSIS_PROMPT
from src.prompts.outlier_detection import OUTLIER_DETECTION_PROMPT
from src.prompts.comparative_analysis import COMPARATIVE_ANALYSIS_PROMPT
from src.prompts.time_comparative import TIME_COMPARATIVE_PROMPT


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


def build_trend_analysis_prompt(query: str, reranked_results: list) -> str:
    relevant_data = ""
    for i, (doc, score) in enumerate(reranked_results[:5], 1):
        preview = doc.content.strip().replace('\n', ' ')
        preview = preview[:200] + '...' if len(preview) > 200 else preview
        relevant_data += f"{i}. {preview}\n"
    return TREND_ANALYSIS_PROMPT.format(
        query=query,
        relevant_data=relevant_data
    )


def build_outlier_detection_prompt(query: str, reranked_results: list) -> str:
    relevant_data = ""
    for i, (doc, score) in enumerate(reranked_results[:5], 1):
        preview = doc.content.strip().replace('\n', ' ')
        preview = preview[:200] + '...' if len(preview) > 200 else preview
        relevant_data += f"{i}. {preview}\n"
    return OUTLIER_DETECTION_PROMPT.format(
        query=query,
        relevant_data=relevant_data
    )


def build_comparative_analysis_prompt(query: str, reranked_results: list) -> str:
    relevant_data = ""
    for i, (doc, score) in enumerate(reranked_results[:5], 1):
        preview = doc.content.strip().replace('\n', ' ')
        preview = preview[:200] + '...' if len(preview) > 200 else preview
        relevant_data += f"{i}. {preview}\n"
    return COMPARATIVE_ANALYSIS_PROMPT.format(
        query=query,
        relevant_data=relevant_data
    )


def build_time_comparative_prompt(query: str, reranked_results: list) -> str:
    relevant_data = ""
    for i, (doc, score) in enumerate(reranked_results[:5], 1):
        preview = doc.content.strip().replace('\n', ' ')
        preview = preview[:200] + '...' if len(preview) > 200 else preview
        relevant_data += f"{i}. {preview}\n"
    return TIME_COMPARATIVE_PROMPT.format(
        query=query,
        relevant_data=relevant_data
    )


def build_prompt(intent: str, query: str, reranked_results: list) -> str:
    """
    Dispatch to the appropriate prompt builder based on intent.
    """
    if intent == 'forecasting':
        return build_energy_forecast_prompt(query, reranked_results)
    if intent == 'trend_analysis':
        return build_trend_analysis_prompt(query, reranked_results)
    if intent == 'outlier_detection':
        return build_outlier_detection_prompt(query, reranked_results)
    if intent == 'comparative_analysis':
        return build_comparative_analysis_prompt(query, reranked_results)
    if intent == 'time_comparative':
        return build_time_comparative_prompt(query, reranked_results)
    # fallback
    return build_energy_forecast_prompt(query, reranked_results)
