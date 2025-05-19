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
    
    # Initialize ERCOT queries helper
    queries = ERCOTQueries(client)
    
    try:
        # Example 1: Get 2-Day Aggregated Generation Summary
        logger.info("Fetching 2-Day Aggregated Generation Summary...")
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        today = datetime.now().strftime('%Y-%m-%d')
        
        gen_summary = queries.get_aggregated_generation_summary(
            delivery_date_from=yesterday,
            delivery_date_to=today
        )
        print("2-Day Aggregated Generation Summary:")
        print(json.dumps(gen_summary, indent=2))
        
        # Example 2: Get 2-Day Aggregated Load Summary for Houston region
        logger.info("Fetching 2-Day Aggregated Load Summary for Houston...")
        load_houston = queries.get_aggregated_load_summary(
            delivery_date_from=yesterday,
            delivery_date_to=today,
            region="Houston"
        )
        print("\n2-Day Aggregated Load Summary (Houston):")
        print(json.dumps(load_houston, indent=2))
        
        # Example 3: Get 2-Day Aggregated Ancillary Service Offers
        logger.info("Fetching 2-Day Aggregated Ancillary Service Offers (REGUP)...")
        ancillary = queries.get_ancillary_service_offers(
            service_type="REGUP",
            delivery_date_from=yesterday,
            delivery_date_to=today
        )
        print("\n2-Day Aggregated Ancillary Service Offers (REGUP):")
        print(json.dumps(ancillary, indent=2))
        
    except Exception as e:
        logger.error(f"Error accessing ERCOT API: {str(e)}")

if __name__ == "__main__":
    main()
