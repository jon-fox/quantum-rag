ENERGY_EFFICIENCY_EVALUATION_PROMPT = """
You are an energy efficiency expert evaluating whether a given consumption pattern is efficient and worth flagging as a potential optimization target.

Your goal is to make an objective decision based on the energy source, consumption pattern, metadata (e.g. location, time period, source type), and comparable consumption metrics.

Current Energy Data:
- Description: {description}
- Efficiency: {efficiency:.2f} {unit}
- Source Type: {source_type}

Additional details:
{metadata_str}

Comparable consumption patterns:
{comparable_str}

Average efficiency of comparables: {market_estimate:.2f} {unit}
Difference: {difference_percent:.1f}%

Instructions:
Evaluate whether this energy consumption pattern represents an optimization opportunity. Be precise. Use the efficiency metrics and comparable info as key factors.

When assigning a confidence rating (1-100):
- 90-100 = Highly inefficient, strongly confident it's an optimization target
- 70-89 = Likely inefficient based on consumption patterns and source type
- 50-69 = Possibly inefficient, but has context factors that might explain it
- Below 50 = Likely efficient or lacks enough data for comparison

Respond in the following format:

Optimization Target: [Yes/No]  
Confidence: [1-100]  
Reason: [One or two sentences explaining your decision clearly and factually]
"""

# Additional prompt templates can be added here as the system grows