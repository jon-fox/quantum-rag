"""
eBay API configuration and search parameters.
Define search profiles here to be used in various parts of the application.
"""
import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(verbose=False)  # Silently load
except ImportError:
    pass  # dotenv not installed

# Sorting constants for eBay API
class SortOptions:
    """Sort options for eBay search results"""
    PRICE_ASCENDING = "price"             # Low to high
    PRICE_DESCENDING = "-price"           # High to low
    NEWLY_LISTED = "newlyListed"          # Newly listed first
    ENDING_SOONEST = "endingSoonest"      # Ending soonest first
    DISTANCE_ASCENDING = "distance"       # Nearest first
    BEST_MATCH = "bestMatch"              # Default eBay sort


class EbaySearchProfile(BaseModel):
    """Model for eBay search profile with parameters"""
    name: str = Field(..., description="Name of the search profile")
    query: str = Field(..., description="Search query")
    category_ids: str = Field(..., description="eBay category ID")
    limit: int = Field(5, description="Maximum results to return")
    price_min: int = Field(0, description="Minimum price")
    price_max: int = Field(1000000, description="Maximum price")
    condition_ids: List[int] = Field(default_factory=lambda: [1000, 3000], description="Condition IDs")
    sort_order: str = Field(SortOptions.BEST_MATCH, description="Sort order for results")
    date_from: Optional[str] = Field(None, description="Start date for items (format: YYYY-MM-DD)")
    date_to: Optional[str] = Field(None, description="End date for items (format: YYYY-MM-DD)")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API use"""
        result = self.dict(exclude_none=True)
        return result


# Define default search profiles
DEFAULT_PROFILES = {
    "luxury_watches": EbaySearchProfile(
        name="luxury_watches",
        query=os.environ.get("EBAY_SEARCH_QUERY", "rolex watch"),
        category_ids=os.environ.get("EBAY_CATEGORY_IDS", "31387"),
        limit=int(os.environ.get("EBAY_RESULT_LIMIT", "5")),
        price_min=int(os.environ.get("EBAY_PRICE_MIN", "500")),
        price_max=int(os.environ.get("EBAY_PRICE_MAX", "20000")),
        condition_ids=[1000, 3000],  # New, Used
        sort_order=os.environ.get("EBAY_SORT_ORDER", SortOptions.BEST_MATCH)
    ),
    "luxury_watches_expensive": EbaySearchProfile(
        name="luxury_watches_expensive",
        query="rolex watch",
        category_ids="31387",
        limit=5,
        price_min=5000,
        price_max=50000,
        condition_ids=[1000, 3000],  # New, Used
        sort_order=SortOptions.PRICE_DESCENDING  # High to low
    ),
    "luxury_watches_bargains": EbaySearchProfile(
        name="luxury_watches_bargains",
        query="rolex watch",
        category_ids="31387",
        limit=10,
        price_min=500,
        price_max=10000,
        condition_ids=[1000, 3000],  # New, Used
        sort_order=SortOptions.PRICE_ASCENDING  # Low to high
    ),
    "vintage_watches": EbaySearchProfile(
        name="vintage_watches",
        query="vintage watch",
        category_ids="31387",
        limit=10,
        price_min=100,
        price_max=5000,
        condition_ids=[3000],  # Used only
        sort_order=SortOptions.NEWLY_LISTED
    ),
    "watch_parts": EbaySearchProfile(
        name="watch_parts",
        query="watch parts",
        category_ids="57720",  # Watch parts category
        limit=20,
        price_min=10,
        price_max=1000,
        sort_order=SortOptions.BEST_MATCH
    )
}


def get_search_profile(profile_name: str = None) -> EbaySearchProfile:
    """Get a search profile by name or the default profile if not found"""
    if not profile_name:
        profile_name = os.environ.get("EBAY_DEFAULT_PROFILE", "luxury_watches")
    
    return DEFAULT_PROFILES.get(profile_name, DEFAULT_PROFILES["luxury_watches"])


def create_custom_profile(
    name: str,
    query: str,
    category_ids: str,
    limit: int = 5,
    price_min: int = 0,
    price_max: int = 1000000,
    condition_ids: List[int] = None,
    sort_order: str = SortOptions.BEST_MATCH,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
) -> EbaySearchProfile:
    """Create a custom search profile"""
    if condition_ids is None:
        condition_ids = [1000, 3000]  # New, Used by default
        
    return EbaySearchProfile(
        name=name,
        query=query,
        category_ids=category_ids,
        limit=limit,
        price_min=price_min,
        price_max=price_max,
        condition_ids=condition_ids,
        sort_order=sort_order,
        date_from=date_from,
        date_to=date_to
    )