#!/usr/bin/env python3
"""
Script for automated data collection from eBay.
Designed to be run as a cron job to regularly fetch and store watch pricing data.

Example usage:
    # Run with default profile
    python -m src.scripts.fetch_watch_data
    
    # Run with specific profile
    python -m src.scripts.fetch_watch_data --profile luxury_watches_bargains
    
    # Run all profiles
    python -m src.scripts.fetch_watch_data --all-profiles
"""

import os
import sys
import time
import argparse
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add parent directory to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import project modules
from src.api.ebay_inventory import EbayInventoryAPI
from src.storage.dynamodb import DynamoDBStorage
from src.config.ebay_config import get_search_profile, DEFAULT_PROFILES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("watch_fetch.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("watch_fetch")

class DataCollector:
    """Handles fetching and storing watch data from eBay"""
    
    def __init__(self, use_aws: bool = None):
        """Initialize collectors with AWS settings"""
        # Determine AWS usage setting
        if use_aws is None:
            use_aws = os.environ.get('USE_AWS', 'false').lower() in ('true', '1', 'yes')
            
        # Initialize API client
        self.ebay_api = EbayInventoryAPI(use_aws=use_aws)
        
        # Initialize storage - can override table name with environment var
        table_name = os.environ.get("DYNAMODB_TABLE", "ebay_watch_listings")
        self.storage = DynamoDBStorage(table_name=table_name)
        
        logger.info(f"Initialized DataCollector (use_aws={use_aws}, table={table_name})")
    
    def collect_profile_data(self, profile_name: str, skip_storage: bool = False) -> Dict[str, Any]:
        """
        Fetch and store data using a specific search profile
        
        Args:
            profile_name: Name of the search profile to use
            skip_storage: If True, will fetch data but not store in DynamoDB
            
        Returns:
            Dictionary with execution statistics
        """
        start_time = time.time()
        stats = {
            "profile": profile_name,
            "timestamp": datetime.utcnow().isoformat(),
            "items_found": 0,
            "items_stored": 0,
            "items_failed": 0,
            "execution_time_sec": 0
        }
        
        try:
            logger.info(f"Starting data collection for profile: {profile_name}")
            
            # Get search results using profile
            results = self.ebay_api.search_with_profile(profile_name)
            
            # Check for errors
            if "error" in results:
                logger.error(f"Error fetching data: {results['error']}")
                stats["error"] = results["error"]
                return stats
            
            # Get items from results
            items = results.get("itemSummaries", [])
            stats["items_found"] = len(items)
            
            if not items:
                logger.warning(f"No items found for profile: {profile_name}")
                return stats
                
            logger.info(f"Found {len(items)} items for profile {profile_name}")
            
            # Store items if not in skip mode
            if not skip_storage:
                batch_results = self.storage.batch_store_listings(items)
                stats["items_stored"] = batch_results["succeeded"]
                stats["items_failed"] = batch_results["failed"]
                
                if batch_results["failed"] > 0:
                    logger.warning(f"Failed to store {batch_results['failed']} items")
                    
                    # Log first few failures for debugging
                    for i, failure in enumerate(batch_results["failures"][:5]):
                        logger.warning(f"Failure {i+1}: Item {failure['item_id']} - {failure['error']}")
                        
            else:
                logger.info("Skipping storage as requested")
                
            # Calculate execution time
            stats["execution_time_sec"] = round(time.time() - start_time, 2)
            logger.info(f"Completed {profile_name} in {stats['execution_time_sec']}s")
            
            return stats
            
        except Exception as e:
            logger.exception(f"Unhandled error in collect_profile_data: {str(e)}")
            stats["error"] = str(e)
            stats["execution_time_sec"] = round(time.time() - start_time, 2)
            return stats
    
    def collect_all_profiles(self, skip_storage: bool = False) -> List[Dict[str, Any]]:
        """
        Collect data from all available search profiles
        
        Args:
            skip_storage: If True, will fetch data but not store in DynamoDB
            
        Returns:
            List of execution statistics for each profile
        """
        results = []
        
        for profile_name in DEFAULT_PROFILES.keys():
            profile_result = self.collect_profile_data(profile_name, skip_storage)
            results.append(profile_result)
            
            # Add a small delay between profiles to avoid rate limiting
            time.sleep(2)
            
        return results

def main():
    """Main entry point for script execution"""
    parser = argparse.ArgumentParser(description='Fetch and store eBay watch listings')
    parser.add_argument('--profile', type=str, help='Search profile name to use')
    parser.add_argument('--all-profiles', action='store_true', help='Run all search profiles')
    parser.add_argument('--aws', action='store_true', help='Force AWS usage')
    parser.add_argument('--no-aws', action='store_true', help='Force local mode (no AWS)')
    parser.add_argument('--dry-run', action='store_true', help='Fetch data but do not store it')
    
    args = parser.parse_args()
    
    # Determine AWS usage
    use_aws = None
    if args.aws:
        use_aws = True
    elif args.no_aws:
        use_aws = False
    
    # Create the data collector
    collector = DataCollector(use_aws=use_aws)
    
    # Execute data collection based on arguments
    if args.all_profiles:
        logger.info("Starting collection for all search profiles")
        results = collector.collect_all_profiles(skip_storage=args.dry_run)
        
        # Log summary
        total_found = sum(r["items_found"] for r in results)
        total_stored = sum(r["items_stored"] for r in results)
        logger.info(f"Completed all profiles: found {total_found} items, stored {total_stored} items")
        
    elif args.profile:
        logger.info(f"Starting collection for profile: {args.profile}")
        result = collector.collect_profile_data(args.profile, skip_storage=args.dry_run)
        
        if "error" in result:
            logger.error(f"Collection failed: {result['error']}")
            sys.exit(1)
        else:
            logger.info(f"Collection complete: found {result['items_found']} items, stored {result['items_stored']} items")
            
    else:
        # Default to using default profile
        default_profile = os.environ.get("EBAY_DEFAULT_PROFILE", "luxury_watches")
        logger.info(f"No profile specified, using default profile: {default_profile}")
        result = collector.collect_profile_data(default_profile, skip_storage=args.dry_run)
        
        if "error" in result:
            logger.error(f"Collection failed: {result['error']}")
            sys.exit(1)
        else:
            logger.info(f"Collection complete: found {result['items_found']} items, stored {result['items_stored']} items")
    
if __name__ == "__main__":
    main()