"""
API module for WatchRecon data integration.
Provides functionality to fetch watch listings from WatchRecon.com
and prepare them for storage in the dual storage system.
"""
import time
import requests
import logging
import os
import json
import random
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from typing import Dict, Any, List, Optional, Tuple
from fastapi import APIRouter, HTTPException, Query

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Constants
BASE_URL = "https://www.watchrecon.com/"
DEFAULT_DELAY = 2  # Base delay between requests in seconds
MAX_RETRIES = 3    # Maximum number of retries for each request
RETRY_BACKOFF = 2  # Exponential backoff factor
JITTER_RANGE = 0.5 # Random jitter range as a fraction of delay

class WatchReconAPI:
    """Client for fetching data from WatchRecon"""
    
    def __init__(self):
        """Initialize the WatchRecon API client"""
        self.user_agent = UserAgent()
        # Create a persistent session for cookies
        self.session = requests.Session()
        # Initialize cookies and session state
        self.initialize_session()
    
    def initialize_session(self):
        """Initialize session with a home page request to get cookies"""
        try:
            headers = self.get_headers()
            self.session.headers.update(headers)
            
            # Visit home page to get initial cookies
            resp = self.session.get(BASE_URL)
            if resp.status_code == 200:
                logger.info("Session initialized successfully")
            else:
                logger.warning(f"Session initialization returned status code {resp.status_code}")
        except Exception as e:
            logger.error(f"Failed to initialize session: {e}")
    
    def get_headers(self) -> Dict[str, str]:
        """Get request headers with rotating user agent and realistic browser settings"""
        headers = {
            "User-Agent": self.user_agent.random,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }
        return headers
    
    def _apply_delay(self, base_delay: float) -> None:
        """Apply a randomized delay to make requests appear more human-like"""
        jitter = random.uniform(-JITTER_RANGE * base_delay, JITTER_RANGE * base_delay)
        actual_delay = max(0.5, base_delay + jitter)  # Ensure minimum delay of 0.5s
        time.sleep(actual_delay)
        
    def fetch_with_retry(self, url: str, max_retries: int = MAX_RETRIES) -> Tuple[Optional[str], int]:
        """
        Fetch a URL with retry logic and exponential backoff
        
        Args:
            url: The URL to fetch
            max_retries: Maximum number of retries
            
        Returns:
            Tuple of (HTML content or None, status code)
        """
        retry_count = 0
        status_code = 0
        
        while retry_count <= max_retries:
            try:
                if retry_count > 0:
                    # Apply exponential backoff for retries
                    retry_delay = DEFAULT_DELAY * (RETRY_BACKOFF ** retry_count)
                    logger.info(f"Retry {retry_count}/{max_retries} after {retry_delay:.2f}s delay")
                    self._apply_delay(retry_delay)
                    
                    # Rotate user agent on retry
                    self.session.headers.update({"User-Agent": self.user_agent.random})
                
                # Make request
                resp = self.session.get(url)
                status_code = resp.status_code
                
                # Handle different status codes
                if status_code == 200:
                    return resp.text, status_code
                elif status_code == 429:  # Too Many Requests
                    logger.warning(f"Rate limit hit (429): {url}")
                    # Apply longer delay for rate limits
                    self._apply_delay(DEFAULT_DELAY * 5)
                elif status_code >= 500:  # Server errors
                    logger.warning(f"Server error {status_code}: {url}")
                else:
                    logger.warning(f"HTTP error {status_code}: {url}")
                
                retry_count += 1
                
            except requests.RequestException as e:
                logger.error(f"Request error: {e}")
                retry_count += 1
                
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                retry_count += 1
        
        # Return None if all retries failed
        logger.error(f"All retries failed for {url}")
        return None, status_code
    
    def fetch_page(self, page: int, brand: str = "rolex") -> Optional[str]:
        """
        Fetch a specific page of listings from WatchRecon
        
        Args:
            page: Page number to fetch
            brand: Brand name to filter by
            
        Returns:
            HTML content of the page or None if request failed
        """
        url = f"{BASE_URL}?current_page={page}&brand={brand}"
        
        try:
            logger.info(f"Fetching WatchRecon page {page} for brand '{brand}'")
            
            # Add a small delay before request to appear more human-like
            if page > 1:
                self._apply_delay(DEFAULT_DELAY)
                
            html, status = self.fetch_with_retry(url)
            
            if html:
                return html
            else:
                logger.error(f"Failed to fetch page {page} (status {status})")
                return None
                
        except Exception as e:
            logger.error(f"Exception while fetching page {page}: {str(e)}")
            return None
    
    def parse_listing_links(self, html: str) -> List[Dict[str, str]]:
        """
        Parse listing links and cid values from WatchRecon HTML
        
        Args:
            html: HTML content of a WatchRecon page
            
        Returns:
            List of dictionaries with listing URLs and cid values
        """
        listing_data = []
        
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Look for listing links and extract cid values
            # Format of href might be: detail.php?current_page=5&brand=rolex&cid=7472146
            for a in soup.select("a[href^='/listing/'], a[href*='detail.php']"):
                href = a.get("href")
                if not href:
                    continue
                    
                item_data = {"url": None, "cid": None}
                
                # Handle direct listing links
                if href.startswith("/listing/"):
                    full_link = BASE_URL.rstrip("/") + href
                    item_data["url"] = full_link
                    # Try to extract ID from URL
                    parts = href.split("/")
                    if len(parts) > 2:
                        item_data["cid"] = parts[-1]
                
                # Handle detail.php links with cid parameter
                elif "detail.php" in href and "cid=" in href:
                    # Fix: Ensure proper URL construction with slash between domain and path
                    if href.startswith("/"):
                        full_link = BASE_URL.rstrip("/") + href
                    elif href.startswith("http"):
                        full_link = href
                    else:
                        # Add slash between domain and detail.php
                        full_link = BASE_URL.rstrip("/") + "/" + href
                    
                    item_data["url"] = full_link
                    # Extract cid from query parameters
                    try:
                        cid_part = href.split("cid=")[1].split("&")[0]
                        item_data["cid"] = cid_part
                    except (IndexError, ValueError):
                        pass
                
                # Only add if we have a URL
                if item_data["url"] and item_data["cid"]:
                    listing_data.append(item_data)
            
            # Remove duplicates based on cid
            unique_data = []
            seen_cids = set()
            
            for item in listing_data:
                if item["cid"] not in seen_cids:
                    seen_cids.add(item["cid"])
                    unique_data.append(item)
            
            logger.info(f"Found {len(unique_data)} unique listings by cid")
            return unique_data
            
        except Exception as e:
            logger.error(f"Exception parsing listing links: {str(e)}")
            return []
    
    def fetch_listing_details(self, url: str) -> Dict[str, Any]:
        """
        Fetch details for a specific listing
        
        Args:
            url: URL of the listing to fetch details for
            
        Returns:
            Dictionary with listing details
        """
        try:
            logger.info(f"Fetching listing details from {url}")
            
            # Apply a small randomized delay to appear more human-like
            self._apply_delay(DEFAULT_DELAY * 1.5)
            
            html, status = self.fetch_with_retry(url)
            
            if html:
                return self.parse_listing_details(html, url)
            else:
                return {"error": f"HTTP error {status}"}
            
        except Exception as e:
            logger.error(f"Exception fetching listing details from {url}: {str(e)}")
            return {"error": str(e)}
    
    def parse_listing_details(self, html: str, url: str) -> Dict[str, Any]:
        """
        Parse details from a listing page
        
        Args:
            html: HTML content of the listing page
            url: Original URL of the listing
            
        Returns:
            Dictionary with extracted listing details
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Extract unique ID from URL
            listing_id = url.split("/")[-1] if url else "unknown"
            
            # Try to extract cid from URL if present
            cid = None
            if "cid=" in url:
                try:
                    cid = url.split("cid=")[1].split("&")[0]
                except (IndexError, ValueError):
                    pass
                    
            # Use cid as listing_id if available (preferred)
            if cid:
                listing_id = cid
            
            # Basic result structure
            result = {
                "itemId": f"wr|{listing_id}|0",  # Format: source|id|version
                "source": "watchrecon",
                "url": url,
                "createdAt": None,
                "price": None,
                "title": None,
                "description": None,
                "images": [],
                "metadata": {
                    "cid": listing_id  # Store the original cid for future reference
                }
            }
            
            # Extract title
            title_elem = soup.select_one("h1.listing-title")
            if title_elem:
                result["title"] = title_elem.text.strip()
            
            # Extract price
            price_elem = soup.select_one("span.price")
            if price_elem:
                price_text = price_elem.text.strip()
                try:
                    # Remove currency symbol and commas
                    price_clean = ''.join(c for c in price_text if c.isdigit() or c == '.')
                    price_value = float(price_clean)
                    
                    # Create price object similar to eBay structure
                    result["price"] = {
                        "value": price_value,
                        "currency": "USD"  # Default to USD, could improve by detecting currency
                    }
                except ValueError:
                    logger.warning(f"Could not parse price from '{price_text}'")
            
            # Extract description
            desc_elem = soup.select_one("div.listing-description")
            if desc_elem:
                result["description"] = desc_elem.text.strip()
                
            # Look for tag elements that might contain additional info (like size, material, etc.)
            tag_elems = soup.select("div.tagContainer div.tag")
            if tag_elems:
                for tag in tag_elems:
                    tag_text = tag.text.strip()
                    if tag_text:
                        # Store tags in metadata for filtering later
                        if "tags" not in result["metadata"]:
                            result["metadata"]["tags"] = []
                        result["metadata"]["tags"].append(tag_text)
            
            # Extract seller info
            seller_elem = soup.select_one("div.seller-info")
            if seller_elem:
                seller_name = seller_elem.select_one("a.seller-name")
                if seller_name:
                    result["metadata"]["seller"] = seller_name.text.strip()
                
                feedback_elem = seller_elem.select_one("span.feedback")
                if feedback_elem:
                    result["metadata"]["seller_feedback"] = feedback_elem.text.strip()
            
            # Extract images
            for img in soup.select("div.listing-images img"):
                src = img.get("src")
                if src:
                    if not src.startswith(("http:", "https:")):
                        src = "https:" + src if src.startswith("//") else BASE_URL + src
                    result["images"].append(src)
            
            if result["images"]:
                result["image"] = {"imageUrl": result["images"][0]}
            
            # Format for compatibility with dual storage system
            result["itemWebUrl"] = url
            result["condition"] = "Pre-owned"  # Default value, could extract from description
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing listing details: {str(e)}")
            return {
                "itemId": f"wr|error|0",
                "error": str(e),
                "url": url
            }
    
    def search_listings(self, brand: str = "rolex", max_pages: int = 3, delay: int = DEFAULT_DELAY) -> Dict[str, Any]:
        """
        Search for listings by brand
        
        Args:
            brand: Brand name to search for
            max_pages: Maximum number of pages to fetch
            delay: Base delay between requests in seconds
            
        Returns:
            Dictionary with search results
        """
        all_links = []
        all_listings = []
        errors = []
        seen_cids = set()  # Track seen cids to avoid duplicates
        
        start_time = time.time()
        
        logger.info(f"Starting WatchRecon search for brand '{brand}', max_pages={max_pages}")
        
        # Randomize the number of pages slightly to appear less bot-like
        if max_pages > 1:
            actual_max_pages = max(1, round(max_pages * random.uniform(0.9, 1.1)))
            if actual_max_pages != max_pages:
                logger.info(f"Randomized max_pages from {max_pages} to {actual_max_pages}")
                max_pages = actual_max_pages
        
        # Step 1: Collect all listing links
        for page in range(1, max_pages + 1):
            logger.info(f"Fetching page {page}/{max_pages}")
            html = self.fetch_page(page, brand)
            
            if html:
                links = self.parse_listing_links(html)
                logger.info(f"Found {len(links)} listings on page {page}")
                all_links.extend(links)
            else:
                logger.warning(f"Failed to fetch page {page}")
                errors.append(f"Failed to fetch page {page}")
            
            if page < max_pages:
                # Apply a randomized delay between page fetches
                self._apply_delay(delay)
        
        # Remove duplicates based on cid
        unique_links = []
        for link in all_links:
            cid = link.get("cid")
            if cid and cid not in seen_cids:
                seen_cids.add(cid)
                unique_links.append(link)
        
        # Randomize the order to appear more human-like
        random.shuffle(unique_links)
        
        logger.info(f"Found {len(unique_links)} unique listings by cid across {max_pages} pages")
        
        # Step 2: Fetch details for each listing (limited to first 20 for API calls)
        limit = min(20, len(unique_links))
        for idx, link in enumerate(unique_links[:limit]):
            logger.info(f"Fetching details for listing {idx+1}/{limit} (cid: {link.get('cid')})")
            
            listing = self.fetch_listing_details(link["url"])
            
            if "error" not in listing:
                # Store the cid in the listing metadata if not already there
                if "metadata" not in listing:
                    listing["metadata"] = {}
                
                if "cid" not in listing["metadata"] and link.get("cid"):
                    listing["metadata"]["cid"] = link["cid"]
                    
                all_listings.append(listing)
            else:
                errors.append(f"Error fetching details for {link['url']}: {listing['error']}")
            
            if idx < limit - 1:
                # Apply a randomized delay between listing fetches
                self._apply_delay(delay * random.uniform(0.8, 1.2))
        
        execution_time = time.time() - start_time
        
        # Format response similar to eBay API response
        return {
            "itemSummaries": all_listings,
            "total": len(all_listings),
            "errors": errors,
            "execution_time_sec": round(execution_time, 2)
        }

def create_watchrecon_router():
    """Create and return a FastAPI router for WatchRecon API endpoints"""
    
    router = APIRouter(
        prefix="/api/watchrecon",
        tags=["watchrecon"],
        responses={404: {"description": "Not found"}}
    )
    
    # Initialize client
    watchrecon_api = WatchReconAPI()
    
    @router.get("/search")
    async def search_watchrecon(
        brand: str = Query("rolex", description="Watch brand to search for"),
        max_pages: int = Query(3, description="Maximum number of pages to fetch"),
        delay: int = Query(DEFAULT_DELAY, description="Delay between requests in seconds"),
    ) -> Dict[str, Any]:
        """Search for watch listings on WatchRecon by brand"""
        try:
            results = watchrecon_api.search_listings(brand, max_pages, delay)
            return results
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error searching WatchRecon: {str(e)}")
    
    @router.get("/listing/{listing_id}")
    async def get_listing(
        listing_id: str,
    ) -> Dict[str, Any]:
        """Get details for a specific listing by ID"""
        try:
            url = f"{BASE_URL}listing/{listing_id}"
            listing = watchrecon_api.fetch_listing_details(url)
            
            if "error" in listing:
                raise HTTPException(status_code=404, detail=f"Listing not found: {listing['error']}")
                
            return listing
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching listing details: {str(e)}")
    
    return router

def main():
    """Run this module directly for testing"""
    api = WatchReconAPI()
    results = api.search_listings(brand="rolex", max_pages=1)
    
    print(f"Found {len(results['itemSummaries'])} listings")
    for i, item in enumerate(results['itemSummaries'][:3]):
        print(f"\nItem {i+1}:")
        print(f"Title: {item.get('title', 'N/A')}")
        print(f"Price: {item.get('price', {}).get('value', 'N/A')} {item.get('price', {}).get('currency', '')}")
        print(f"URL: {item.get('itemWebUrl', 'N/A')}")

if __name__ == "__main__":
    main()