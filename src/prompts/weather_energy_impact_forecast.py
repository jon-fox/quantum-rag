"""
Prompt templates for LLM-based weather impact on energy forecasting.
"""

WEATHER_ENERGY_IMPACT_FORECAST_PROMPT = """
You are an energy market analyst specializing in the impact of weather on energy consumption and pricing, with a focus on the ERCOT market.

Your goal is to provide a detailed forecast of how upcoming weather conditions will affect energy usage and potential costs.

Weather Forecast Data:
- Location: {location}
- Time Period: {time_period}
- Temperature Forecast: {temperature_forecast}
- Precipitation Forecast: {precipitation_forecast}
- Wind Speed Forecast: {wind_speed_forecast}
- Other Relevant Weather (e.g., cloud cover, humidity): {other_weather_factors}

Historical Energy Data (for context):
- Average consumption during similar past weather conditions: {historical_consumption}
- Average price during similar past weather conditions: {historical_price}

ERCOT Market Conditions (if available):
- Current Grid Status: {grid_status}
- Fuel Mix: {fuel_mix}
- Demand Forecasts: {ercot_demand_forecast}

Instructions:
Analyze the provided weather forecast and historical data to predict the impact on energy consumption and cost in the specified ERCOT region. Consider factors like heating/cooling demand, renewable energy generation (wind/solar), and potential grid strain.

When assigning a confidence rating (1-100) for your forecast:
- 90-100 = High confidence based on strong correlations in historical data and clear weather forecast.
- 70-89 = Moderate confidence, some uncertainty in weather patterns or historical data.
- 50-69 = Low confidence, significant uncertainty or conflicting data points.
- Below 50 = Very low confidence, speculative forecast due to lack of data or highly unpredictable conditions.

Respond in the following format:

Forecast Summary: [Brief overview of the expected impact]
Predicted Consumption Change: [e.g., Increase/Decrease by X%, or specific MWh range]
Predicted Cost Impact: [e.g., Potential price spike, moderate increase, stable prices]
Confidence: [1-100]
Reasoning: [Detailed explanation of your forecast, citing specific weather factors, historical trends, and ERCOT conditions. Explain how these factors are likely to influence demand and price.]
Potential Risks/Uncertainties: [Identify any factors that could significantly alter the forecast]
"""

# Additional prompt templates can be added here as the system grows
