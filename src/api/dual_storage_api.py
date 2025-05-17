"""
FastAPI endpoints for demonstrating dual storage functionality
for eBay listings with vector embeddings in DynamoDB and PostgreSQL.
"""
import json
import numpy as np
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Response, status, Query

from src.storage.utils import store_ebay_listing_with_embedding, batch_store_ebay_listings_with_embeddings
from src.storage.dual_storage import DualStorage, default_dual_storage
from src.embeddings.embed_utils import get_embedding_provider, embed_query
from src.api.ebay_inventory import EbayInventoryAPI
from src.config.ebay_config import get_search_profile, DEFAULT_PROFILES

# Pydantic models for request/response validation
from pydantic import BaseModel, Field

class EbayListingSchema(BaseModel):
    """Schema for eBay listing input"""
    itemId: str
    title: str
    price: Dict[str, Any]
    condition: Optional[str] = None
    itemWebUrl: Optional[str] = None
    description: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "itemId": "123456789",
                "title": "Rolex Submariner 116610LN Stainless Steel Black Ceramic Bezel",
                "price": {"value": 12500.00, "currency": "USD"},
                "condition": "Pre-owned",
                "itemWebUrl": "https://www.ebay.com/itm/123456789",
                "description": "Excellent condition Rolex Submariner with box and papers"
            }
        }

class StorageResponseSchema(BaseModel):
    """Schema for storage operation response"""
    overall_success: bool
    item_id: str
    dynamo_success: bool
    postgres_success: bool
    error_message: Optional[str] = None

class BatchFetchResponseSchema(BaseModel):
    """Schema for batch fetch and store operation response"""
    total_items_found: int
    items_stored_dynamo: int
    items_stored_postgres: int
    failed_items: int
    profile_used: str
    embedding_status: str
    execution_time_sec: float

def create_dual_storage_router() -> APIRouter:
    """Create and return the dual storage API router"""
    
    router = APIRouter(
        prefix="/api/storage",
        tags=["storage"],
        responses={404: {"description": "Not found"}},
    )
    
    # Try to get the embedding provider - will use OpenAI if available
    try:
        embedding_provider = get_embedding_provider()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to initialize embedding provider: {e}")
        embedding_provider = None
    
    # Initialize eBay API client
    ebay_api = EbayInventoryAPI()
    
    @router.post("/store-with-embedding", response_model=StorageResponseSchema)
    async def store_listing_with_embedding(listing: EbayListingSchema) -> Dict[str, Any]:
        """
        Store an eBay listing in both DynamoDB and PostgreSQL with pgvector embedding.
        
        Uses the title and description (if available) to generate the embedding.
        
        This endpoint demonstrates:
        1. Generating an embedding from listing text
        2. Storing the listing in DynamoDB
        3. Storing the listing with its vector embedding in PostgreSQL
        """
        try:
            # Check if embedding provider is available
            if embedding_provider is None:
                return Response(
                    content=json.dumps({
                        "overall_success": False,
                        "item_id": listing.itemId,
                        "dynamo_success": False,
                        "postgres_success": False,
                        "error_message": "Embedding provider not available - check OPENAI_API_KEY environment variable"
                    }),
                    media_type="application/json",
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            # Create text for embedding generation
            embedding_text = listing.title
            if listing.description:
                embedding_text += " " + listing.description
            
            # Generate embedding
            embedding = embed_query(embedding_text, provider=embedding_provider)
            
            # Convert Pydantic model to dict
            listing_dict = json.loads(listing.model_dump_json())
            
            # Store the listing with its embedding
            result = store_ebay_listing_with_embedding(listing_dict, embedding)
            
            # Format the response
            response = {
                "overall_success": result.get("overall_success", False),
                "item_id": listing.itemId,
                "dynamo_success": result.get("dynamo", {}).get("success", False),
                "postgres_success": result.get("postgres", {}).get("success", False),
                "error_message": None
            }
            
            # Add error message if anything failed
            if not response["overall_success"]:
                dynamo_error = result.get("dynamo", {}).get("error")
                postgres_error = result.get("postgres", {}).get("error")
                errors = []
                if dynamo_error:
                    errors.append(f"DynamoDB: {dynamo_error}")
                if postgres_error:
                    errors.append(f"PostgreSQL: {postgres_error}")
                response["error_message"] = "; ".join(errors) if errors else "Unknown error"
            
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error storing listing: {str(e)}")
    
    @router.get("/search-similar/{item_id}", response_model=List[Dict[str, Any]])
    async def search_similar_listings(item_id: str, limit: int = 5):
        """
        Find similar listings based on vector embedding similarity.
        
        This endpoint demonstrates querying the PostgreSQL database using 
        pgvector's vector similarity search capabilities.
        """
        try:
            # Get a connection to the database
            storage = default_dual_storage
            conn = storage._get_pg_connection(retry=False)
            
            # Check if we have a valid database connection
            if conn is None:
                return Response(
                    content=json.dumps({
                        "error": "PostgreSQL database not available",
                        "message": "This endpoint requires PostgreSQL with pgvector to be configured"
                    }),
                    media_type="application/json",
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE
                )
            
            # Initialize schema if needed
            if not storage.schema_initialized:
                initialized = storage._init_postgres_schema()
                if not initialized:
                    return Response(
                        content=json.dumps({
                            "error": "Could not initialize PostgreSQL schema",
                            "message": "Failed to set up the required database tables and extensions"
                        }),
                        media_type="application/json",
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
            
            with conn.cursor() as cur:
                # First get the embedding for the reference item
                cur.execute(
                    f"SELECT embedding FROM {storage.postgres_table_name} WHERE item_id = %s",
                    (item_id,)
                )
                result = cur.fetchone()
                if not result:
                    raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
                
                reference_embedding = result[0]
                
                # Find similar items using cosine similarity
                cur.execute(
                    f"""
                    SELECT 
                        item_id, 
                        title, 
                        price, 
                        metadata,
                        1 - (embedding <=> %s) AS similarity
                    FROM 
                        {storage.postgres_table_name}
                    WHERE 
                        item_id != %s
                    ORDER BY 
                        embedding <=> %s
                    LIMIT %s
                    """,
                    (reference_embedding, item_id, reference_embedding, limit)
                )
                
                similar_items = []
                for row in cur.fetchall():
                    similar_items.append({
                        "item_id": row[0],
                        "title": row[1],
                        "price": row[2],
                        "metadata": row[3],
                        "similarity_score": row[4]
                    })
                
                return similar_items
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error searching similar listings: {str(e)}")
    
    @router.get("/health", response_model=Dict[str, Any])
    async def health_check():
        """
        Health check endpoint to verify the status of storage services
        """
        storage = default_dual_storage
        
        # Check DynamoDB availability
        dynamo_status = "available"
        try:
            # Try to access the table
            storage.dynamo_storage.table.table_status
        except Exception as e:
            dynamo_status = f"unavailable: {str(e)}"
            
        # Check PostgreSQL availability
        pg_status = "available"
        pg_version = None
        try:
            conn = storage._get_pg_connection(retry=False)
            if conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version();")
                    pg_version = cur.fetchone()[0]
            else:
                pg_status = "unavailable: connection failed"
        except Exception as e:
            pg_status = f"unavailable: {str(e)}"
            
        # Check pgvector extension
        pgvector_status = "not checked"
        if pg_status == "available":
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
                    if cur.fetchone():
                        pgvector_status = "installed"
                    else:
                        pgvector_status = "not installed"
            except Exception as e:
                pgvector_status = f"error: {str(e)}"
                
        # Check if the tables exist
        table_status = "not checked"
        if pg_status == "available":
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT to_regclass('{storage.postgres_table_name}');")
                    if cur.fetchone()[0]:
                        table_status = "exists"
                    else:
                        table_status = "missing"
            except Exception as e:
                table_status = f"error: {str(e)}"
                
        return {
            "status": "degraded" if pg_status != "available" else "healthy",
            "dynamodb": {
                "status": dynamo_status,
                "table": storage.dynamo_storage.table_name
            },
            "postgres": {
                "status": pg_status,
                "version": pg_version,
                "pgvector": pgvector_status,
                "table": {
                    "name": storage.postgres_table_name,
                    "status": table_status
                }
            }
        }
    
    @router.post("/fetch-and-store", response_model=BatchFetchResponseSchema)
    async def fetch_and_store_listings(
        profile: str = Query(None, description="Search profile name (e.g., luxury_watches, vintage_watches)"),
        query: str = Query(None, description="Custom search query, overrides profile if provided"),
        limit: int = Query(20, description="Number of items to fetch", ge=1, le=100),
        price_min: Optional[int] = Query(None, description="Minimum price, overrides profile if provided"),
        price_max: Optional[int] = Query(None, description="Maximum price, overrides profile if provided"),
        store_embeddings: bool = Query(True, description="Whether to generate and store embeddings"),
    ):
        """
        Fetch eBay listings and store them in both DynamoDB and PostgreSQL with vector embeddings.
        
        This endpoint:
        1. Fetches listings from eBay using the specified profile or custom query
        2. Generates embeddings for the listings (unless disabled)
        3. Stores the listings in DynamoDB
        4. Stores the listings with their embeddings in PostgreSQL with pgvector
        
        You can use predefined search profiles or specify custom search parameters.
        """
        import time
        start_time = time.time()
        
        # Check if embedding provider is available when needed
        if store_embeddings and embedding_provider is None:
            return Response(
                content=json.dumps({
                    "error": "Embedding provider not available",
                    "message": "Cannot generate embeddings. Check OPENAI_API_KEY environment variable."
                }),
                media_type="application/json",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        try:
            # Determine search parameters
            if profile:
                # Use existing profile (with potential overrides)
                profile_obj = get_search_profile(profile)
                results = ebay_api.search_items(
                    query=query or profile_obj.query,
                    category_ids=profile_obj.category_ids,
                    limit=limit or profile_obj.limit,
                    price_min=price_min if price_min is not None else profile_obj.price_min,
                    price_max=price_max if price_max is not None else profile_obj.price_max,
                    condition_ids=profile_obj.condition_ids,
                    sort_order=profile_obj.sort_order
                )
                profile_used = profile
            else:
                # Use default profile with potential overrides
                default_profile = get_search_profile()
                results = ebay_api.search_items(
                    query=query or default_profile.query,
                    category_ids=default_profile.category_ids,
                    limit=limit or default_profile.limit,
                    price_min=price_min if price_min is not None else default_profile.price_min,
                    price_max=price_max if price_max is not None else default_profile.price_max,
                    condition_ids=default_profile.condition_ids,
                    sort_order=default_profile.sort_order
                )
                profile_used = "default"
            
            # Check for errors
            if "error" in results:
                raise HTTPException(status_code=500, detail=f"Error fetching eBay data: {results['error']}")
            
            # Get items from results
            items = results.get("itemSummaries", [])
            
            if not items:
                return Response(
                    content=json.dumps({
                        "total_items_found": 0,
                        "items_stored_dynamo": 0,
                        "items_stored_postgres": 0,
                        "failed_items": 0,
                        "profile_used": profile_used,
                        "embedding_status": "not_needed",
                        "execution_time_sec": round(time.time() - start_time, 2)
                    }),
                    media_type="application/json",
                    status_code=status.HTTP_200_OK
                )
            
            # Generate embeddings if requested
            embeddings = None
            embedding_status = "skipped"
            
            if store_embeddings:
                try:
                    # Prepare text for embeddings
                    texts = []
                    for item in items:
                        text = item.get('title', '')
                        if 'shortDescription' in item:
                            text += " " + item['shortDescription']
                        texts.append(text)
                    
                    # Generate embeddings in batch
                    embeddings = np.array([embed_query(text, provider=embedding_provider) for text in texts])
                    embedding_status = "generated"
                except Exception as e:
                    # Continue with dynamo-only storage if embeddings fail
                    import logging
                    logging.getLogger(__name__).error(f"Failed to generate embeddings: {e}")
                    embedding_status = f"failed: {str(e)}"
            
            # Store in both databases
            if embeddings is not None:
                # Store with embeddings
                result = batch_store_ebay_listings_with_embeddings(items, embeddings)
                return {
                    "total_items_found": len(items),
                    "items_stored_dynamo": result.get("dynamo_succeeded", 0),
                    "items_stored_postgres": result.get("postgres_succeeded", 0),
                    "failed_items": result.get("dynamo_failed", 0) + result.get("postgres_failed", 0),
                    "profile_used": profile_used,
                    "embedding_status": embedding_status,
                    "execution_time_sec": round(time.time() - start_time, 2)
                }
            else:
                # Store in DynamoDB only
                from src.storage.dynamodb import batch_store_listings
                result = batch_store_listings(items)
                return {
                    "total_items_found": len(items),
                    "items_stored_dynamo": result.get("succeeded", 0),
                    "items_stored_postgres": 0,
                    "failed_items": result.get("failed", 0),
                    "profile_used": profile_used,
                    "embedding_status": embedding_status,
                    "execution_time_sec": round(time.time() - start_time, 2)
                }
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    
    return router