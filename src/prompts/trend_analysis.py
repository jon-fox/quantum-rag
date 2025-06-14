TREND_ANALYSIS_PROMPT = """
You are an energy data analyst focusing on trend analysis. Given the query and top 5 relevant data points, identify and describe temporal trends in the data.

Query: {query}

Top 5 Relevant Data Points:
{relevant_data}

Instructions:
- Highlight increasing or decreasing trends over time.
- Reference specific dates and values.
- Comment on seasonality or periodic patterns if present.

Output:
"""
