#!/usr/bin/env python3
"""
Script to fetch watch listings data from WatchRecon and store in dual storage system.
Generates embeddings for semantic search functionality.

Usage:
  python fetch_watchrecon_data.py --brand rolex --pages 3 --delay 2 --test-mode
  python fetch_watchrecon_data.py --brand rolex --limit 5 --test-mode
  python fetch_watchrecon_data.py --brand rolex --check-duplicates --pages 3
"""
import argparse
import logging
import time
import os
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any, List, Set

# Import our modules
from src.api.watchrecon_api import WatchReconAPI
from src.storage.dual_storage import DualStorage
from src.storage.utils import (
    batch_store_ebay_listings_with_embeddings, 
    normalize_listing_for_embedding,
    get_embedding_text
)
from src.storage.dynamodb import DynamoDBStorage
from src.embeddings.embed_utils import get_embedding_provider, embed_query

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WatchReconDataCollector:
    """Handles fetching and storing watch data from WatchRecon"""
    
    def __init__(self, test_mode=False):
        """Initialize collector with API client and storage"""
        self.watchrecon_api = WatchReconAPI()
        
        # Initialize dual storage for postgres + dynamodb
        self.storage = DualStorage()
        self.dynamo_storage = DynamoDBStorage()
        
        # Set test mode flag
        self.test_mode = test_mode
        
        # Try to get the embedding provider
        try:
            self.embedding_provider = get_embedding_provider()
            logger.info("Embedding provider initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding provider: {e}")
            self.embedding_provider = None
            
    def get_existing_cids(self, check_postgres=True, check_dynamo=True) -> Set[str]:
        """
        Get set of existing CIDs in the database to avoid storing duplicates
        
        Args:
            check_postgres: Whether to check PostgreSQL for existing CIDs
            check_dynamo: Whether to check DynamoDB for existing CIDs
            
        Returns:
            Set of CIDs already in the database
        """
        existing_cids = set()
        
        if check_postgres and self.storage:
            try:
                conn = self.storage._get_pg_connection()
                if conn:
                    with conn.cursor() as cur:
                        # Query metadata for WatchRecon listings with CID
                        cur.execute(f"""
                        SELECT metadata->>'cid' 
                        FROM {self.storage.postgres_table_name}
                        WHERE 
                            metadata->>'cid' IS NOT NULL 
                            AND source = 'watchrecon'
                        """)
                        results = cur.fetchall()
                        for row in results:
                            if row[0]:
                                existing_cids.add(row[0])
                        
                        logger.info(f"Found {len(existing_cids)} existing CIDs in PostgreSQL")
            except Exception as e:
                logger.error(f"Error fetching existing CIDs from PostgreSQL: {e}")
                
        if check_dynamo and self.dynamo_storage:
            try:
                # This is a simplified approach - in a real implementation you'd want to use
                # DynamoDB query/scan operations with filters for source=watchrecon
                # and projecting only the metadata.cid attribute
                logger.warning("DynamoDB CID checking not implemented yet - would require scan operation")
                pass
            except Exception as e:
                logger.error(f"Error fetching existing CIDs from DynamoDB: {e}")
                
        return existing_cids
    
    def collect_brand_data(self, brand: str, max_pages: int = 3, delay: int = 2, limit: int = None, check_duplicates=False) -> Dict[str, Any]:
        """
        Fetch and store data for a specific watch brand from WatchRecon
        
        Args:
            brand: Watch brand to search for (e.g., 'rolex', 'omega')
            max_pages: Maximum number of pages to fetch
            delay: Delay between requests in seconds
            limit: Maximum number of listings to process (for testing)
            check_duplicates: Whether to check for and skip existing listings by CID
            
        Returns:
            Dictionary with execution statistics
        """
        start_time = time.time()
        stats = {
            "brand": brand,
            "timestamp": datetime.utcnow().isoformat(),
            "items_found": 0,
            "items_stored": 0,
            "items_failed": 0,
            "items_skipped": 0,
            "execution_time_sec": 0,
            "test_mode": self.test_mode
        }
        
        try:
            logger.info(f"Starting WatchRecon data collection for brand: {brand}")
            if self.test_mode:
                logger.info(f"Running in TEST MODE with limit: {limit or 'No limit'}")
            
            # Get existing CIDs if checking for duplicates
            existing_cids = set()
            if check_duplicates:
                logger.info("Checking for existing listings to avoid duplicates")
                existing_cids = self.get_existing_cids()
                stats["duplicates_check"] = True
                stats["existing_cids_count"] = len(existing_cids)
            
            # Get search results
            results = self.watchrecon_api.search_listings(brand, max_pages, delay)
            
            # Check for errors
            if "errors" in results and results["errors"]:
                logger.warning(f"Some errors occurred during fetch: {len(results['errors'])} errors")
                stats["warnings"] = results["errors"]
            
            # Get items from results
            items = results.get("itemSummaries", [])
            stats["items_found"] = len(items)
            
            if not items:
                logger.warning(f"No items found for brand: {brand}")
                return stats
                
            logger.info(f"Found {len(items)} items for brand {brand}")
            
            # Filter out duplicates if checking for them
            if check_duplicates and existing_cids:
                filtered_items = []
                for item in items:
                    cid = None
                    if "metadata" in item and "cid" in item["metadata"]:
                        cid = item["metadata"]["cid"]
                    
                    if cid and cid in existing_cids:
                        logger.debug(f"Skipping duplicate item with CID {cid}")
                        stats["items_skipped"] += 1
                    else:
                        filtered_items.append(item)
                
                if len(filtered_items) != len(items):
                    logger.info(f"Filtered out {len(items) - len(filtered_items)} duplicate items")
                    items = filtered_items
            
            # Apply limit for testing if specified
            if limit is not None and limit > 0:
                items = items[:limit]
                logger.info(f"Limiting to {limit} items for processing")
                stats["limited_to"] = limit
            
            # Normalize items for consistent storage and embedding
            normalized_items = []
            for item in items:
                normalized_item = normalize_listing_for_embedding(item)
                normalized_items.append(normalized_item)
            
            logger.info(f"Normalized {len(normalized_items)} items for storage compatibility")
            items = normalized_items
            
            # Generate embeddings if provider is available
            if self.embedding_provider:
                logger.info("Generating embeddings for items...")
                embeddings = []
                
                for item in items:
                    # Get text for embedding using the standardized function
                    text = get_embedding_text(item)
                    
                    # Generate embedding
                    try:
                        embedding = embed_query(text, provider=self.embedding_provider)
                        embeddings.append(embedding)
                    except Exception as e:
                        logger.error(f"Failed to generate embedding for item {item.get('itemId')}: {e}")
                        embeddings.append(None)
                
                # Remove items with failed embeddings
                valid_items = []
                valid_embeddings = []
                
                for item, embedding in zip(items, embeddings):
                    if embedding is not None:
                        valid_items.append(item)
                        valid_embeddings.append(embedding)
                    else:
                        stats["items_failed"] += 1
                
                if len(valid_items) != len(items):
                    logger.warning(f"Removed {len(items) - len(valid_items)} items due to embedding failures")
                    items = valid_items
                
                # Store items with embeddings
                if items:
                    logger.info(f"Storing {len(items)} items with embeddings...")
                    embeddings_array = np.array(valid_embeddings)
                    
                    if self.test_mode:
                        # In test mode, print sample data and don't actually store
                        logger.info("TEST MODE: Would store the following items:")
                        for idx, item in enumerate(items[:2]):  # Show first 2 items max
                            cid = item.get("metadata", {}).get("cid", "unknown")
                            logger.info(f"Item {idx+1}: {item.get('itemId')} - CID: {cid} - {item.get('title')} - {item.get('price')}")
                        
                        if len(items) > 2:
                            logger.info(f"... and {len(items) - 2} more items")
                            
                        logger.info(f"Embeddings shape: {embeddings_array.shape}")
                        stats["items_stored"] = len(items)
                        stats["postgres_stored"] = len(items)
                    else:
                        # Normal mode - actually store the items
                        batch_results = batch_store_ebay_listings_with_embeddings(
                            items, embeddings_array, self.storage
                        )
                        
                        stats["items_stored"] = batch_results.get("dynamo_succeeded", 0)
                        stats["postgres_stored"] = batch_results.get("postgres_succeeded", 0)
                        stats["items_failed"] += batch_results.get("dynamo_failed", 0)
                    
                    logger.info(f"Stored {stats['items_stored']} items in DynamoDB and {stats['postgres_stored']} items in PostgreSQL")
            else:
                # Fallback: Store items in DynamoDB only (without embeddings)
                logger.warning("No embedding provider available, storing in DynamoDB only (without embeddings)")
                
                if self.test_mode:
                    # In test mode, print sample data and don't actually store
                    logger.info("TEST MODE: Would store the following items in DynamoDB only:")
                    for idx, item in enumerate(items[:2]):  # Show first 2 items max
                        cid = item.get("metadata", {}).get("cid", "unknown")
                        logger.info(f"Item {idx+1}: {item.get('itemId')} - CID: {cid} - {item.get('title')}")
                    stats["items_stored"] = len(items)
                else:
                    # Normal mode - store in DynamoDB
                    dynamo_results = self.dynamo_storage.batch_store_listings(items)
                    stats["items_stored"] = dynamo_results.get("succeeded", 0)
                    stats["items_failed"] = dynamo_results.get("failed", 0)
                    stats["note"] = "Items stored in DynamoDB only (no embeddings)"
                
                logger.info(f"Stored {stats['items_stored']} items in DynamoDB without embeddings")
            
            # Calculate execution time
            stats["execution_time_sec"] = round(time.time() - start_time, 2)
            logger.info(f"Completed {brand} in {stats['execution_time_sec']}s")
            
            return stats
            
        except Exception as e:
            logger.exception(f"Unhandled error in collect_brand_data: {str(e)}")
            stats["error"] = str(e)
            stats["execution_time_sec"] = round(time.time() - start_time, 2)
            return stats

def main():
    """Main entry point for script"""
    parser = argparse.ArgumentParser(description="Fetch watch data from WatchRecon")
    parser.add_argument("--brand", type=str, default="rolex", help="Watch brand to search for")
    parser.add_argument("--pages", type=int, default=3, help="Maximum number of pages to fetch")
    parser.add_argument("--delay", type=int, default=2, help="Delay between requests in seconds")
    parser.add_argument("--limit", type=int, help="Limit the number of processed items (for testing)")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode (no actual storage)")
    parser.add_argument("--check-duplicates", action="store_true", help="Check for and skip existing listings by CID")
    
    args = parser.parse_args()
    
    logger.info(f"Starting WatchRecon data collection for {args.brand}")
    
    collector = WatchReconDataCollector(test_mode=args.test_mode)
    results = collector.collect_brand_data(
        brand=args.brand,
        max_pages=args.pages,
        delay=args.delay,
        limit=args.limit,
        check_duplicates=args.check_duplicates
    )
    
    logger.info(f"Collection complete: found {results['items_found']}, stored {results.get('items_stored', 0)}")
    if results.get("items_skipped", 0) > 0:
        logger.info(f"Skipped {results['items_skipped']} duplicate items")
    
    if "error" in results:
        logger.error(f"Error during collection: {results['error']}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())