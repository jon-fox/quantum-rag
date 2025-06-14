"""
Prompt template for time comparative analysis.
"""
TIME_COMPARATIVE_PROMPT = """
You are an energy data analyst comparing multiple time periods. Given the query and top 5 relevant data points per period, analyze differences across periods.

Query: {query}

Top 5 Relevant Data Points:
{relevant_data}

Instructions:
- Ensure each period’s data is clearly separated.
- Highlight key differences across periods.
- Reference specific dates, values, and context.

Output:
"""
