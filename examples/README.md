# Examples

This directory contains example scripts demonstrating the usage of different components of the quantum energy project.

## ERCOT API Example

The `ercot_api_example.py` script demonstrates how to use the ERCOT API client to:

1. Authenticate with the ERCOT API
2. Fetch real-time pricing data
3. Get historical energy data
4. Retrieve energy forecasts

### Running the Example

```bash
# From the project root directory
python -m examples.ercot_api_example
```

Note: Make sure you have set up your `.env` file with the appropriate ERCOT API credentials:

```
ERCOT_API_USERNAME=your-username
ERCOT_API_PASSWORD=your-password
```
