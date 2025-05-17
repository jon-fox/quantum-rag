import boto3
import requests
import base64
import time
import json
import os
import pathlib
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add dotenv support for local development
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file if present
    print("Loaded .env file")
except ImportError:
    print("python-dotenv not installed, skipping .env file loading")

# Import search profile configuration
from src.config.ebay_config import get_search_profile, EbaySearchProfile

# === Config ===
# Allow environment variable configuration
USE_AWS = os.environ.get('USE_AWS', 'true').lower() in ('true', '1', 'yes')
PARAM_NAME = os.environ.get('EBAY_PARAM_NAME', "/application/ebay/prod/oauth")
EBAY_OAUTH_URL = os.environ.get('EBAY_OAUTH_URL', "https://api.ebay.com/identity/v1/oauth2/token")
CLIENT_ID = os.environ.get('EBAY_CLIENT_ID', "Jonathan-watcharb-PRD-480007c95-c1c26485")
CLIENT_SECRET = os.environ.get('EBAY_CLIENT_SECRET', "PRD-80007c956ebf-8e95-47d5-a692-bfd8")
OAUTH_SCOPE = os.environ.get('EBAY_OAUTH_SCOPE', "https://api.ebay.com/oauth/api_scope")
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')

# Local token file path
LOCAL_TOKEN_FILE = os.environ.get('LOCAL_TOKEN_FILE', 
                                os.path.join(pathlib.Path.home(), '.ebay_token.json'))

class EbayTokenManager:
    """Manager class for handling eBay OAuth tokens with AWS and local support"""
    
    def __init__(self, use_aws: bool = USE_AWS):
        self.use_aws = use_aws
        
    def get_token(self) -> Dict[str, Any]:
        """Get the existing token from either Parameter Store or local file"""
        token_data = {}
        
        if self.use_aws:
            # AWS environment - use Parameter Store
            try:
                ssm = boto3.client("ssm", region_name=AWS_REGION)
                response = ssm.get_parameter(Name=PARAM_NAME, WithDecryption=True)
                raw_value = response["Parameter"]["Value"]

                try:
                    # Try to parse as JSON
                    token_data = json.loads(raw_value)
                    expires_at = token_data.get("expires_at", 0)
                    
                    if time.time() < expires_at:
                        print("Token is still valid.")
                        return token_data
                    else:
                        print("Token expired. Generating new one.")
                except json.JSONDecodeError:
                    # Handle legacy plain-token format
                    token_data = {
                        "access_token": raw_value.strip(),
                        "expires_at": 0  # Force refresh
                    }
                    print("Raw token detected, assuming expired.")
            except Exception as e:
                print(f"Error getting token from Parameter Store: {e}")
        else:
            # Local development mode - use local file
            print(f"Using local token file: {LOCAL_TOKEN_FILE}")
            try:
                if os.path.exists(LOCAL_TOKEN_FILE):
                    with open(LOCAL_TOKEN_FILE, 'r') as f:
                        token_data = json.load(f)
                        expires_at = token_data.get("expires_at", 0)
                        
                        if time.time() < expires_at:
                            print(f"Token is still valid (expires {datetime.fromtimestamp(expires_at)})")
                            return token_data
                        else:
                            print("Token expired. Generating new one.")
                else:
                    print("No local token file found. Will generate a new token.")
            except Exception as e:
                print(f"Error reading local token: {e}")
                
        # If we get here, we need a new token
        return self.generate_new_token(token_data)

    def generate_new_token(self, old_token_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a new eBay OAuth token"""
        # === Generate new token ===
        auth = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
        headers = {
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {
            "grant_type": "client_credentials",
            "scope": OAUTH_SCOPE
        }

        try:
            res = requests.post(EBAY_OAUTH_URL, headers=headers, data=data)
            res.raise_for_status()
            res_data = res.json()

            # === Prepare token with expiry time ===
            access_token = res_data["access_token"]
            expires_in = res_data["expires_in"]  # in seconds
            expires_at = int(time.time()) + expires_in - 60  # refresh 1 min early

            token_data = {
                "access_token": access_token,
                "expires_at": expires_at
            }

            # Save the token
            self.save_token(token_data)
            
            print(f"New token generated (expires {datetime.fromtimestamp(expires_at)})")
            return token_data
        except Exception as e:
            print(f"Error generating token: {e}")
            # Return the old token data if available
            return old_token_data or {"access_token": "", "expires_at": 0}

    def save_token(self, token_data: Dict[str, Any]) -> None:
        """Save token either to Parameter Store or local file"""
        if self.use_aws:
            # AWS environment - save to Parameter Store
            try:
                ssm = boto3.client("ssm", region_name=AWS_REGION)
                ssm.put_parameter(
                    Name=PARAM_NAME,
                    Value=json.dumps(token_data),
                    Type="SecureString",
                    Overwrite=True
                )
                print("Token saved to Parameter Store")
            except Exception as e:
                print(f"Error saving token to Parameter Store: {e}")
                # Fallback to local file if AWS fails
                self._save_to_local_file(token_data)
        else:
            # Local development mode - save to file
            self._save_to_local_file(token_data)
            
    def _save_to_local_file(self, token_data: Dict[str, Any]) -> None:
        """Helper method to save token to local file"""
        try:
            token_dir = os.path.dirname(LOCAL_TOKEN_FILE)
            if token_dir and not os.path.exists(token_dir):
                os.makedirs(token_dir, exist_ok=True)
                
            with open(LOCAL_TOKEN_FILE, 'w') as f:
                json.dump(token_data, f)
            print(f"Token saved to {LOCAL_TOKEN_FILE}")
        except Exception as e:
            print(f"Error saving token to local file: {e}")

class EbayInventoryAPI:
    """eBay Inventory API client"""
    
    def __init__(self, use_aws: bool = USE_AWS):
        self.token_manager = EbayTokenManager(use_aws)
        
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for eBay API requests"""
        token_data = self.token_manager.get_token()
        access_token = token_data.get("access_token")
        
        if not access_token:
            raise ValueError("No valid access token available")
            
        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"
        }
    
    def search_with_profile(self, profile_name: str = None) -> Dict[str, Any]:
        """Search for items on eBay using a predefined search profile"""
        profile = get_search_profile(profile_name)
        return self.search_items(
            query=profile.query,
            category_ids=profile.category_ids,
            limit=profile.limit,
            price_min=profile.price_min,
            price_max=profile.price_max,
            condition_ids=profile.condition_ids,
            sort_order=profile.sort_order,
            date_from=profile.date_from,
            date_to=profile.date_to
        )
        
    def search_items(self, 
                     query: str = None, 
                     category_ids: str = None, 
                     limit: int = None, 
                     price_min: int = None, 
                     price_max: int = None,
                     condition_ids: List[int] = None,
                     sort_order: str = None,
                     date_from: str = None,
                     date_to: str = None) -> Dict[str, Any]:
        """Search for items on eBay"""
        # If any parameter is None, get from default profile
        profile = get_search_profile()
        
        # Use provided parameters or defaults from profile
        query = query or profile.query
        category_ids = category_ids or profile.category_ids
        limit = limit if limit is not None else profile.limit
        price_min = price_min if price_min is not None else profile.price_min
        price_max = price_max if price_max is not None else profile.price_max
        condition_ids = condition_ids or profile.condition_ids
        sort_order = sort_order or profile.sort_order
        date_from = date_from or profile.date_from
        date_to = date_to or profile.date_to
            
        headers = self.get_auth_headers()
        url = os.environ.get('EBAY_API_URL', "https://api.ebay.com/buy/browse/v1/item_summary/search")
        
        # Convert price parameters to integers and validate
        try:
            price_min = int(price_min)
            price_max = int(price_max)
            # Swap if min > max
            if price_min > price_max:
                print(f"Warning: price_min ({price_min}) > price_max ({price_max}), swapping values")
                price_min, price_max = price_max, price_min
        except (ValueError, TypeError):
            print(f"Warning: Invalid price values provided. Using defaults: min={profile.price_min}, max={profile.price_max}")
            price_min = profile.price_min
            price_max = profile.price_max
            
        # Format condition IDs
        condition_str = ",".join(map(str, condition_ids)) if condition_ids else ""
        
        # Build base parameters - always exclude sort to avoid API limitations
        # We'll apply sorting client-side later
        params = {
            "q": query,
            "category_ids": category_ids,
            "limit": min(limit, 200)  # eBay max is 200
        }
        
        # Build filter string
        filter_parts = []
        if price_min > 0 or price_max < 1000000:
            filter_parts.append(f"price:[{price_min}..{price_max}]")
            
        if condition_ids:
            filter_parts.append(f"conditionIds:{{{condition_str}}}")
        
        # Add date filter if provided
        # Validate and format dates for eBay API
        if date_from or date_to:
            try:
                date_filter = "itemCreationDate:"
                
                # Format date range with proper validation
                if date_from and date_to:
                    # Validate date format (YYYY-MM-DD)
                    from datetime import datetime
                    
                    # Convert to proper format and validate
                    date_from_obj = datetime.strptime(date_from, "%Y-%m-%d")
                    date_to_obj = datetime.strptime(date_to, "%Y-%m-%d")
                    
                    # Ensure date_from is before date_to
                    if date_from_obj > date_to_obj:
                        print(f"Warning: date_from ({date_from}) > date_to ({date_to}), swapping values")
                        date_from_obj, date_to_obj = date_to_obj, date_from_obj
                    
                    # Format for eBay API: YYYY-MM-DDT00:00:00Z
                    date_from_formatted = date_from_obj.strftime("%Y-%m-%dT00:00:00Z")
                    date_to_formatted = date_to_obj.strftime("%Y-%m-%dT23:59:59Z")
                    date_filter += f"[{date_from_formatted}..{date_to_formatted}]"
                
                elif date_from:
                    # Only start date provided
                    date_from_obj = datetime.strptime(date_from, "%Y-%m-%d")
                    date_from_formatted = date_from_obj.strftime("%Y-%m-%dT00:00:00Z")
                    date_filter += f"[{date_from_formatted}..]"
                
                elif date_to:
                    # Only end date provided
                    date_to_obj = datetime.strptime(date_to, "%Y-%m-%d")
                    date_to_formatted = date_to_obj.strftime("%Y-%m-%dT23:59:59Z")
                    date_filter += f"[..{date_to_formatted}]"
                
                filter_parts.append(date_filter)
                print(f"Added date filter: {date_filter}")
                
            except ValueError as e:
                print(f"Warning: Invalid date format. Dates must be in YYYY-MM-DD format: {e}")
                # Skip date filter if invalid
                    
        # Add filters to params if any exist
        if filter_parts:
            params["filter"] = ",".join(filter_parts)
            
        # Log request details
        print(f"eBay API Request URL: {url}")
        print(f"eBay API Request Parameters: {params}")
        
        try:
            # Make API request - always without sort for consistent results
            response = requests.get(url, headers=headers, params=params)
            print(f"Request URL: {response.request.url}")
            response.raise_for_status()
            
            # Process results
            result = response.json()
            items = result.get("itemSummaries", [])
            
            if not items:
                print("No results found.")
                return {"itemSummaries": [], "total": 0}
                
            # Apply price filtering
            filtered_items = []
            for item in items:
                price_value = float(item.get('price', {}).get('value', 0))
                if price_min <= price_value <= price_max:
                    filtered_items.append(item)
                else:
                    print(f"Excluding item with price: {price_value} (outside range {price_min}-{price_max})")
            
            # Apply client-side sorting based on requested sort order
            if sort_order:
                if sort_order == "price":  # ascending
                    filtered_items.sort(key=lambda item: float(item.get('price', {}).get('value', 0)))
                    print("Applied ascending price sort client-side")
                elif sort_order == "-price":  # descending
                    filtered_items.sort(key=lambda item: float(item.get('price', {}).get('value', 0)), reverse=True)
                    print("Applied descending price sort client-side")
                # If other sort types are available, they would be implemented here
                # For now, we're focused on price sorting
                
                result["note"] = "Results client-side sorted"
                
            # Update result with filtered and sorted items
            result["itemSummaries"] = filtered_items
            result["total"] = len(filtered_items)
            print(f"Retrieved and processed {len(filtered_items)} items")
            return result
                
        except requests.RequestException as e:
            print(f"Error making eBay API request: {e}")
            if hasattr(e, 'response'):
                print(f"Response status: {e.response.status_code}")
                print(f"Response text: {e.response.text}")
            return {"error": str(e), "itemSummaries": [], "total": 0}

# Create FastAPI integration
def create_ebay_router():
    """Create a FastAPI router for eBay APIs"""
    try:
        from fastapi import APIRouter, Query, HTTPException, Depends
        from pydantic import BaseModel, Field
        from src.storage.dynamodb import store_listing
        
        router = APIRouter(prefix="/ebay", tags=["ebay"])
        ebay_api = EbayInventoryAPI(use_aws=USE_AWS)
        
        class SearchResponse(BaseModel):
            total: int = Field(0, description="Total number of results")
            items: List[Dict[str, Any]] = Field(default_factory=list, description="Items found")
        
        class StoreItemRequest(BaseModel):
            item_id: str = Field(None, description="eBay item ID to fetch and store")
            item_data: Dict[str, Any] = Field(None, description="Complete item data to store directly")
            
        class StoreResponse(BaseModel):
            success: bool = Field(..., description="Whether the operation was successful")
            item_id: str = Field(None, description="The ID of the stored item")
            message: str = Field(None, description="Additional information about the operation")
        
        @router.get("/search", response_model=SearchResponse)
        async def search_items(
            profile: str = Query(None, description="Search profile name (e.g., luxury_watches, vintage_watches, watch_parts)"),
            query: str = Query(None, description="Search query (overrides profile)"),
            category_ids: str = Query(None, description="eBay category IDs (overrides profile)"),
            limit: int = Query(None, description="Number of results to return", ge=1, le=200),
            price_min: int = Query(None, description="Minimum price (overrides profile)"),
            price_max: int = Query(None, description="Maximum price (overrides profile)"),
            sort_order: str = Query(None, description="Sort order (price, -price, newlyListed, endingSoonest, etc.)"),
            date_from: str = Query(None, description="Start date for item creation (YYYY-MM-DD format)"),
            date_to: str = Query(None, description="End date for item creation (YYYY-MM-DD format)")
        ):
            """Search for items on eBay using a profile or custom parameters"""
            try:
                # Debug logging for parameters
                print(f"Search parameters: profile={profile}, price_min={price_min}, price_max={price_max}, date_from={date_from}, date_to={date_to}")
                
                if profile:
                    if any([query, category_ids, limit, price_min is not None, price_max is not None, sort_order, date_from, date_to]):
                        # Get profile but override specific parameters
                        profile_obj = get_search_profile(profile)
                        results = ebay_api.search_items(
                            query=query or profile_obj.query,
                            category_ids=category_ids or profile_obj.category_ids,
                            limit=limit or profile_obj.limit,
                            price_min=price_min if price_min is not None else profile_obj.price_min,
                            price_max=price_max if price_max is not None else profile_obj.price_max,
                            condition_ids=profile_obj.condition_ids,
                            sort_order=sort_order or profile_obj.sort_order,
                            date_from=date_from,
                            date_to=date_to
                        )
                    else:
                        # Use predefined profile exactly as is
                        results = ebay_api.search_with_profile(profile)
                else:
                    # Use parameters (falling back to defaults when needed)
                    # Explicitly pass price_min and price_max, including if they're set to 0
                    results = ebay_api.search_items(
                        query=query,
                        category_ids=category_ids,
                        limit=limit,
                        price_min=price_min,
                        price_max=price_max,
                        sort_order=sort_order,
                        date_from=date_from,
                        date_to=date_to
                    )
                
                if "error" in results:
                    raise HTTPException(status_code=500, detail=results["error"])
                
                return {
                    "total": results.get("total", 0),
                    "items": results.get("itemSummaries", [])
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @router.post("/store", response_model=StoreResponse)
        async def store_ebay_item(request: StoreItemRequest):
            """Store an eBay item in DynamoDB"""
            try:
                # Check if we have item data directly or need to fetch by ID
                if request.item_data:
                    # Use provided item data directly
                    item = request.item_data
                    store_listing(item)
                    return {
                        "success": True,
                        "item_id": item["itemId"],
                        "message": "Item stored successfully from provided data"
                    }
                elif request.item_id:
                    # Need to get single item details by ID from eBay API
                    # In a real implementation, you would call eBay's getItem API here
                    # For now, search for it using existing functionality
                    
                    # Get headers for API call
                    headers = ebay_api.get_auth_headers()
                    
                    # Call eBay Get Item API
                    url = f"https://api.ebay.com/buy/browse/v1/item/{request.item_id}"
                    
                    response = requests.get(url, headers=headers)
                    response.raise_for_status()
                    
                    item = response.json()
                    store_listing(item)
                    
                    return {
                        "success": True,
                        "item_id": request.item_id,
                        "message": "Item fetched and stored successfully"
                    }
                else:
                    raise HTTPException(
                        status_code=400, 
                        detail="Either item_id or item_data must be provided"
                    )
            except requests.RequestException as e:
                error_message = str(e)
                if hasattr(e, 'response') and e.response:
                    error_message = f"{e} - {e.response.text}"
                return {
                    "success": False,
                    "item_id": request.item_id,
                    "message": f"Failed to fetch item: {error_message}"
                }
            except Exception as e:
                return {
                    "success": False,
                    "item_id": request.item_id or "unknown",
                    "message": f"Failed to store item: {str(e)}"
                }
        
        @router.post("/store-batch", response_model=Dict[str, Any])
        async def store_ebay_items_batch(items: List[Dict[str, Any]]):
            """Store multiple eBay items in DynamoDB in a batch"""
            results = {
                "total": len(items),
                "succeeded": 0,
                "failed": 0,
                "failures": []
            }
            
            for item in items:
                try:
                    store_listing(item)
                    results["succeeded"] += 1
                except Exception as e:
                    results["failed"] += 1
                    item_id = item.get("itemId", "unknown")
                    results["failures"].append({
                        "item_id": item_id,
                        "error": str(e)
                    })
            
            return results
        
        @router.get("/profiles", response_model=Dict[str, Any])
        async def list_search_profiles():
            """List available search profiles"""
            from src.config.ebay_config import DEFAULT_PROFILES
            
            profiles = {}
            for name, profile in DEFAULT_PROFILES.items():
                profiles[name] = profile.dict(exclude={"name"})
                
            return {
                "profiles": profiles,
                "default_profile": os.environ.get("EBAY_DEFAULT_PROFILE", "luxury_watches")
            }
        
        return router
    except ImportError:
        print("FastAPI not installed, skipping router creation")
        return None

def main():
    """Run this module directly for testing"""
    import argparse
    from src.config.ebay_config import SortOptions
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='eBay Inventory Search')
    parser.add_argument('--profile', type=str, help='Search profile name (luxury_watches, vintage_watches, etc.)')
    parser.add_argument('--query', type=str, help='Search query to override profile')
    parser.add_argument('--aws', action='store_true', help='Use AWS Parameter Store for token')
    parser.add_argument('--limit', type=int, help='Maximum results to return')
    parser.add_argument('--price-min', type=int, help='Minimum price')
    parser.add_argument('--price-max', type=int, help='Maximum price')
    parser.add_argument('--sort', type=str, help=f'Sort order (options: {SortOptions.PRICE_ASCENDING}, {SortOptions.PRICE_DESCENDING}, {SortOptions.NEWLY_LISTED}, {SortOptions.ENDING_SOONEST})')
    parser.add_argument('--date-from', type=str, help='Start date for item creation (YYYY-MM-DD format)')
    parser.add_argument('--date-to', type=str, help='End date for item creation (YYYY-MM-DD format)')
    
    args = parser.parse_args()
    
    # Use local file for token storage by default when running script directly
    use_aws = args.aws or os.environ.get('USE_AWS', '').lower() in ('true', '1', 'yes')
    ebay_api = EbayInventoryAPI(use_aws=use_aws)
    
    # Debug output for command line arguments
    print(f"CLI parameters: profile={args.profile}, price_min={args.price_min}, price_max={args.price_max}, date_from={args.date_from}, date_to={args.date_to}")
    
    # Use search profile
    if args.profile:
        print(f"Using search profile: {args.profile}")
        
        # Check if any overrides are provided
        if any([args.query, args.limit, args.sort, args.price_min is not None, args.price_max is not None, args.date_from, args.date_to]):
            profile = get_search_profile(args.profile)
            results = ebay_api.search_items(
                query=args.query or profile.query,
                category_ids=profile.category_ids,
                limit=args.limit or profile.limit,
                price_min=args.price_min if args.price_min is not None else profile.price_min,
                price_max=args.price_max if args.price_max is not None else profile.price_max,
                condition_ids=profile.condition_ids,
                sort_order=args.sort or profile.sort_order,
                date_from=args.date_from,
                date_to=args.date_to
            )
        else:
            # Use profile as is
            results = ebay_api.search_with_profile(args.profile)
    else:
        # Use default profile with optional overrides
        profile = get_search_profile()
        query = args.query or profile.query
        limit = args.limit or profile.limit
        sort_order = args.sort or profile.sort_order
        price_min = args.price_min if args.price_min is not None else profile.price_min
        price_max = args.price_max if args.price_max is not None else profile.price_max
        
        print(f"Using default profile with query: {query}")
        if args.sort:
            print(f"Using sort order: {args.sort}")
        if args.price_min is not None or args.price_max is not None:
            print(f"Using price range: ${price_min} - ${price_max}")
        if args.date_from or args.date_to:
            print(f"Using date range: {args.date_from or 'any'} to {args.date_to or 'any'}")
            
        results = ebay_api.search_items(
            query=query,
            category_ids=profile.category_ids,
            limit=limit,
            price_min=price_min,
            price_max=price_max,
            condition_ids=profile.condition_ids,
            sort_order=sort_order,
            date_from=args.date_from,
            date_to=args.date_to
        )
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    # Display results
    total = results.get("total", 0)
    items = results.get("itemSummaries", [])
    
    print(f"\nFound {total} items. Displaying {len(items)} results:\n")
    
    for item in items:
        price = item.get('price', {}).get('value', 'N/A')
        currency = item.get('price', {}).get('currency', 'USD')
        print(f"- {item['title']}")
        print(f"  Price: {price} {currency}")
        print(f"  URL: {item['itemWebUrl']}")
        print()

if __name__ == "__main__":
    main()
