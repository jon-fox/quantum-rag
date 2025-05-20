"""
Example script demonstrating how to use the ERCOT API client and queries
"""
import logging
import json
import os
import sys
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.env_manager import load_environment
from src.data.ercot_api.client import ERCOTClient
from src.data.ercot_api.queries import ERCOTQueries

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to demonstrate ERCOT API usage"""
    # Load environment variables
    load_environment()
    
    # Initialize ERCOT API client
    client = ERCOTClient()
    
    today = datetime.today()
    yesterday = today - timedelta(days=1)
    three_days_ago = today - timedelta(days=3)

    delivery_date_from_str = three_days_ago.strftime('%Y-%m-%d')
    delivery_date_to_str = yesterday.strftime('%Y-%m-%d')
    
    # Initialize ERCOT queries helper with delivery dates
    queries = ERCOTQueries(
        client=client,
        delivery_date_from=delivery_date_from_str,
        delivery_date_to=delivery_date_to_str
    )
    
    try:
        # Example 2: Get 2-Day Aggregated Load Summary for Houston region
        # The delivery dates are now set at the instance level
        logger.info(f"Fetching 2-Day Aggregated Load Summary for Houston from {delivery_date_from_str} to {delivery_date_to_str}...")
        load_houston = queries.get_aggregated_load_summary(
            region="Houston"
        )
        
        print("\n2-Day Aggregated Load Summary (Houston):")
        print(json.dumps(load_houston, indent=2))
        
    except Exception as e:
        logger.error(f"Error accessing ERCOT API: {str(e)}")

if __name__ == "__main__":
    main()
