COMPARATIVE_ANALYSIS_PROMPT = """
You are an energy data analyst performing comparative analysis. Given the query and top 5 relevant data points, compare categories or scenarios as requested.

Query: {query}

Top 5 Relevant Data Points:
{relevant_data}

Instructions:
- Compare specified categories (e.g., generation types).
- Highlight differences and similarities.
- Reference specific dates and metrics.

Output:
"""
