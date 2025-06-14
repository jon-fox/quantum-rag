ENERGY_FORECAST_PROMPT = """You are an ERCOT energy analyst. Given the query and top 5 relevant ERCOT daily system reports, write a concise summary analyzing trends in DAM forecasted load vs actual telemetry generation during afternoon peak hours.

Context:
- Each document includes values such as forecasted system load, telemetry generation, renewable generation (wind/solar), dispatchable capacity, and average temperature.
- "Load" refers to forecasted system demand.
- "Telemetry generation" refers to actual measured generation.
- Afternoon peak hours are typically between 2 PM and 7 PM.
- Use only the provided data. Do not speculate or generalize beyond what's shown.

Query: {query}

Top 5 Relevant Data Points:
{relevant_data}

Instructions:
- Compare forecasted load vs telemetry generation during afternoon peak hours.
- Highlight differences between months if applicable (e.g., April vs June).
- Note any significant over- or under-generation, especially if tied to renewable output levels.
- Reference specific dates to support observations.
- If helpful, include a table showing load vs telemetry for relevant dates.
- Write the output in clear, professional language.
- Target 150-200 words, but use more if necessary to capture meaningful trends.

Output:

In April and June 2025, telemetry generation during afternoon peak hours generally tracked DAM forecasted load with minor discrepancies. The table below summarizes selected data points across both months:

| Date       | Forecasted Load (MW) | Telemetry Generation (MW) | Renewable Share (%) | Trend               |
|------------|----------------------|----------------------------|----------------------|---------------------|
| 2025-04-26 | 52,422               | 52,345                     | 50%                  | Slight under-gen    |
| 2025-05-08 | 51,405               | 51,438                     | 33%                  | Slight over-gen     |
| 2025-05-29 | 58,427               | 58,500                     | 39%                  | Slight over-gen     |
| 2025-05-30 | 55,708               | 55,644                     | 28%                  | Slight under-gen    |
| 2025-06-02 | 63,520               | 63,489                     | 48%                  | Near exact match    |

April's data shows slight under-generation possibly tied to higher renewable shares and variable spring conditions. In contrast, June data suggests tighter alignment, likely due to more stable solar contributions. Overall, the forecasting model performs well, with only small day-to-day variances mostly driven by renewable intermittency.

"""
