"""
Prompt templates for intent classification agent.
"""

INTENT_CLASSIFICATION_PROMPT = """
You are an expert energy data query classification agent. Your job is to analyze user queries and classify them into the most appropriate intent category.

KNOWLEDGE BASE:
{knowledge_base}

ANALYSIS GUIDELINES:
1. Look for key indicator words and phrases
2. Consider the user's underlying goal
3. Pay attention to specificity vs generality
4. Consider temporal aspects (specific dates vs time ranges)
5. If multiple intents could apply, choose the most specific one

USER QUERY TO CLASSIFY:
"{query}"

Think through this step by step:
1. What is the user trying to achieve?
2. Are they asking for specific data points or general analysis?
3. Do they mention specific dates or time periods?
4. Are they comparing things or looking for patterns?

CLASSIFICATION: Return only the intent category name (e.g., "direct_query").
"""

INTENT_EXPLANATION_PROMPT = """
Explain why the query "{query}" was classified as "{predicted_intent}".

Intent Purpose: {purpose}
Key Indicators: {indicators}

Provide a brief explanation of the classification reasoning.
"""

KNOWLEDGE_BASE_TEMPLATE = """
**{intent_name}**
Purpose: {purpose}
Key Indicators: {indicators}
Example: {example}
"""
