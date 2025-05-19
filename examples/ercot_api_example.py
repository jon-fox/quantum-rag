"""
Example script demonstrating how to use the ERCOT API client
"""
import logging
import json
from src.config.env_manager import load_environment
from src.data.ercot_api.client import ERCOTClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to demonstrate ERCOT API usage"""
    # Load environment variables
    load_environment()
    
    # Initialize ERCOT API client
    client = ERCOTClient()
    
    try:
        # Example 1: Get real-time prices
        logger.info("Fetching real-time prices from ERCOT API...")
        rt_prices = client.get_data("energy/prices/realtime", {"location": "HB_HOUSTON"})
        print("Real-time prices:")
        print(json.dumps(rt_prices, indent=2))
        
        # Example 2: Get historical data
        logger.info("Fetching historical data from ERCOT API...")
        historical = client.get_data("energy/historical", {
            "start_date": "2025-05-01",
            "end_date": "2025-05-15",
            "dataset": "load"
        })
        print("\nHistorical data:")
        print(json.dumps(historical, indent=2))
        
        # Example 3: Get forecasts
        logger.info("Fetching forecast data from ERCOT API...")
        forecast = client.get_data("energy/forecast", {
            "type": "demand", 
            "days": 7
        })
        print("\nForecast data:")
        print(json.dumps(forecast, indent=2))
        
    except Exception as e:
        logger.error(f"Error accessing ERCOT API: {str(e)}")

if __name__ == "__main__":
    main()
