"""
Prompt templates used for various LLM-based evaluations in the watch-arb system.
These templates can be easily modified to improve evaluation quality.
"""

WATCH_DEAL_EVALUATION_PROMPT = """
You are a luxury watch pricing expert evaluating whether a given listing is underpriced and worth flagging as a potential deal.

Your goal is to make an objective decision based on the watch's title, condition, metadata (e.g. papers, box, year), and comparable market prices.

Current Watch:
- Title: {title}
- Price: ${price:.2f}
- Condition: {condition}

Additional details:
{metadata_str}

Comparable watches:
{comparable_str}

Average price of comparables: ${market_estimate:.2f}
Discount: {discount_percent:.1f}%

Instructions:
Evaluate whether this listing represents a good deal. Be precise. Use the discount % and comparable info as key factors.

When assigning a confidence rating (1-100):
- 90-100 = Strongly underpriced, highly confident it's a good deal
- 70-89 = Likely a good deal based on price and condition
- 50-69 = Possibly a good deal, but has risks (e.g. missing papers, condition concerns)
- Below 50 = Unlikely to be a good deal or lacks enough discount

Respond in the following format:

Good Deal: [Yes/No]  
Confidence: [1-100]  
Reason: [One or two sentences explaining your decision clearly and factually]
"""

# Additional prompt templates can be added here as the system grows