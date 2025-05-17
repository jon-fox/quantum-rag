"""
FastAPI endpoints for querying watch listings stored in PostgreSQL with pgvector.
This API provides endpoints for:
1. Semantic search using vector embeddings
2. Fetching similar items by ID
3. Searching listings by text and price
4. Finding good deals using price comparison and RAG
"""
import json
import numpy as np
import statistics
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Response, status, Query
from pydantic import BaseModel, Field

from src.storage.dual_storage import DualStorage, default_dual_storage
from src.embeddings.embed_utils import get_embedding_provider, embed_query
from src.valuation.prompt_templates import WATCH_DEAL_EVALUATION_PROMPT

# Schema definitions
class PriceRange(BaseModel):
    min: float = Field(0.0, description="Minimum price")
    max: float = Field(1000000.0, description="Maximum price")

class SemanticSearchRequest(BaseModel):
    query: str = Field(..., description="Search query text to find similar items")
    limit: int = Field(10, description="Number of results to return", ge=1, le=100)
    price_range: Optional[PriceRange] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "vintage stainless steel submariner in excellent condition",
                "limit": 5,
                "price_range": {"min": 5000, "max": 15000}
            }
        }

class SimilarItemsRequest(BaseModel):
    item_id: str = Field(..., description="Item ID to find similar items for")
    limit: int = Field(5, description="Number of results to return", ge=1, le=50)
    min_similarity: float = Field(0.5, description="Minimum similarity score (0-1)", ge=0, le=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "item_id": "v1|123456789|0",
                "limit": 5,
                "min_similarity": 0.7
            }
        }

class ListingSearchRequest(BaseModel):
    text: Optional[str] = Field(None, description="Text to search for in title")
    min_price: Optional[float] = Field(None, description="Minimum price")
    max_price: Optional[float] = Field(None, description="Maximum price")
    condition: Optional[str] = Field(None, description="Condition to filter by")
    limit: int = Field(20, description="Number of results to return", ge=1, le=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Rolex Submariner",
                "min_price": 5000,
                "max_price": 15000,
                "condition": "Pre-owned",
                "limit": 10
            }
        }

class DealFinderRequest(BaseModel):
    brand: Optional[str] = Field(None, description="Brand to search for (e.g., 'Rolex', 'Omega')")
    model: Optional[str] = Field(None, description="Model to search for (e.g., 'Submariner', 'Speedmaster')")
    min_discount_percent: float = Field(10.0, description="Minimum discount threshold as percentage", ge=1.0, le=90.0)
    price_range: Optional[PriceRange] = None
    min_similarity: float = Field(0.75, description="Minimum similarity score for comparable listings", ge=0.0, le=1.0)
    comparable_count: int = Field(5, description="Number of comparable listings to analyze", ge=3, le=20)
    max_deals: int = Field(10, description="Maximum number of deals to return", ge=1, le=50)
    use_llm: bool = Field(False, description="Whether to use LLM for deal evaluation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "brand": "Rolex",
                "model": "Submariner",
                "min_discount_percent": 15.0,
                "price_range": {"min": 5000, "max": 20000},
                "min_similarity": 0.8,
                "comparable_count": 5,
                "max_deals": 10,
                "use_llm": False
            }
        }

class ListingResponse(BaseModel):
    item_id: str
    title: str
    price: float
    condition: Optional[str] = None
    item_url: Optional[str] = None
    similarity_score: Optional[float] = None
    created_at: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "item_id": "v1|123456789|0",
                "title": "Rolex Submariner 116610LN Stainless Steel Black Ceramic",
                "price": 12500.0,
                "condition": "Pre-owned",
                "item_url": "https://www.ebay.com/itm/...",
                "similarity_score": 0.92,
                "created_at": "2025-05-11T12:30:45"
            }
        }

class DealResponse(BaseModel):
    item_id: str
    title: str
    price: float
    condition: Optional[str] = None
    item_url: Optional[str] = None
    market_estimate: float
    discount_percent: float
    confidence: str
    confidence_score: int
    reason: str
    comparables: List[Dict[str, Any]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "item_id": "v1|123456789|0",
                "title": "Rolex Submariner 116610LN Stainless Steel Black Ceramic",
                "price": 11500.0,
                "condition": "Pre-owned",
                "item_url": "https://www.ebay.com/itm/...",
                "market_estimate": 13500.0,
                "discount_percent": 14.8,
                "confidence": "High",
                "confidence_score": 85,
                "reason": "Full set in excellent condition, 15% below recent comparable sales",
                "comparables": [
                    {"item_id": "v1|987654321|0", "title": "Similar Submariner", "price": 13700.0, "similarity": 0.92},
                    {"item_id": "v1|876543210|0", "title": "Another Submariner", "price": 13200.0, "similarity": 0.88}
                ]
            }
        }

def create_deals_router() -> APIRouter:
    """Create and return the deals API router"""
    
    router = APIRouter(
        prefix="/api",
        tags=["deals"],
        responses={404: {"description": "Not found"}},
    )
    
    # Try to get the embedding provider for semantic search
    try:
        embedding_provider = get_embedding_provider()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to initialize embedding provider: {e}")
        embedding_provider = None
    
    @router.post("/find_deals", response_model=List[DealResponse])
    async def find_deals(request: DealFinderRequest):
        """
        Find good deals by comparing prices with similar items.
        
        This endpoint:
        1. Searches for listings matching the brand, model, and price range
        2. For each listing, finds comparable watches using vector similarity search
        3. Calculates market price estimates based on comparable watches
        4. Identifies listings that are significantly underpriced
        5. Optionally uses LLM-based reasoning for deal evaluation and explanation
        6. Returns the top 10 results by confidence score, even if they don't meet the discount threshold
        """
        if embedding_provider is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Embedding provider not available - check OPENAI_API_KEY environment variable"
            )
        
        # Get a connection to the database
        storage = default_dual_storage
        conn = storage._get_pg_connection(retry=False)
        
        if conn is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PostgreSQL database not available"
            )
        
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Construct the search query based on brand and model
            search_query = ""
            if request.brand and request.model:
                search_query = f"{request.brand} {request.model}"
            elif request.brand:
                search_query = request.brand
            elif request.model:
                search_query = request.model
            else:
                # If no brand or model provided, we'll search all watches
                search_query = "luxury watch"
            
            logger.info(f"Finding deals with search query: {search_query}")
            
            # Generate embedding for search query
            query_embedding = embed_query(search_query, provider=embedding_provider)
            query_embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            
            with conn.cursor() as cur:
                # Step 1: Find watches matching our criteria
                candidates_query = f"""
                SELECT 
                    item_id, 
                    title, 
                    price,
                    metadata->>'condition' as condition,
                    metadata->>'item_url' as item_url,
                    metadata,
                    created_at,
                    embedding
                FROM 
                    {storage.postgres_table_name}
                WHERE 
                    embedding IS NOT NULL
                """
                
                params = []
                
                # Add price filter if provided
                if request.price_range:
                    candidates_query += " AND price BETWEEN %s AND %s"
                    params.extend([request.price_range.min, request.price_range.max])
                
                # Update brand filter to search only in title
                if request.brand:
                    candidates_query += " AND title ILIKE %s"
                    params.append(f"%{request.brand}%")
                    
                # Update model filter to search only in title
                if request.model:
                    candidates_query += " AND title ILIKE %s"
                    params.append(f"%{request.model}%")
                
                # Order by similarity to our query
                candidates_query += """
                ORDER BY 
                    embedding <=> %s::vector
                LIMIT %s
                """
                params.extend([query_embedding_list, 100])  # Get top 100 candidates to analyze
                
                # Execute query to get candidate listings
                cur.execute(candidates_query, params)
                candidates = cur.fetchall()
                
                logger.info(f"Found {len(candidates)} candidate listings to analyze for deals")
                
                # Step 2: For each candidate, analyze and score regardless of discount threshold
                analyzed_listings = []
                
                for candidate in candidates:
                    item_id = candidate[0]
                    title = candidate[1]
                    price = float(candidate[2]) if candidate[2] is not None else 0.0
                    condition = candidate[3]
                    item_url = candidate[4]
                    metadata = candidate[5] if candidate[5] else {}
                    created_at = candidate[6]
                    embedding = candidate[7]
                    
                    # Find comparable items for this watch
                    comparables_query = f"""
                    SELECT 
                        item_id, 
                        title, 
                        price,
                        metadata->>'condition' as condition,
                        1 - (embedding <=> %s) AS similarity
                    FROM 
                        {storage.postgres_table_name}
                    WHERE 
                        item_id != %s
                        AND (1 - (embedding <=> %s)) >= %s
                        AND embedding IS NOT NULL
                    ORDER BY 
                        embedding <=> %s
                    LIMIT %s
                    """
                    
                    cur.execute(
                        comparables_query,
                        [
                            embedding, 
                            item_id, 
                            embedding,
                            request.min_similarity,
                            embedding,
                            request.comparable_count
                        ]
                    )
                    
                    comparable_items = cur.fetchall()
                    
                    # Instead of skipping when we don't have enough comparable items,
                    # include them with an appropriate confidence level and reason
                    if len(comparable_items) < 3:
                        logger.info(f"Item {item_id} has only {len(comparable_items)} comparable items, marking as low confidence")
                        
                        # Format comparable items (even if fewer than expected)
                        formatted_comparables = []
                        for comp in comparable_items:
                            formatted_comparables.append({
                                "item_id": comp[0],
                                "title": comp[1],
                                "price": float(comp[2]) if comp[2] is not None else 0.0,
                                "condition": comp[3],
                                "similarity": float(comp[4]) if comp[4] is not None else 0.0
                            })
                        
                        # Use the average price as market estimate if we have some comparables
                        # Otherwise, use the item's own price as a fallback
                        if comparable_items:
                            comparable_prices = [float(item[2]) for item in comparable_items]
                            market_estimate = statistics.mean(comparable_prices)  # Use mean since we have too few for median
                            discount_percent = ((market_estimate - price) / market_estimate) * 100 if market_estimate > 0 else 0
                        else:
                            # No comparables at all - use the item's own price
                            market_estimate = price
                            discount_percent = 0
                        
                        confidence_text = "Low"
                        confidence_score = 10  # Very low confidence score
                        reason = f"Insufficient comparable items ({len(comparable_items)}) found for accurate pricing"
                        
                        # Even with insufficient comparables, use LLM if requested
                        # This will allow LLM to evaluate the watch based on its characteristics alone
                        if request.use_llm:
                            logger.info(f"LLM evaluation requested for item {item_id} with use_llm={request.use_llm} - proceeding despite insufficient comparables")
                            try:
                                from openai import OpenAI
                                import os
                                
                                # Get OpenAI API key from environment variables
                                openai_api_key = os.getenv("OPENAI_API_KEY")
                                
                                if openai_api_key:
                                    logger.info(f"OpenAI API key found, preparing LLM request for item {item_id}")
                                    client = OpenAI(api_key=openai_api_key)
                                    
                                    # Prepare context for LLM
                                    comparable_str = "\n".join([
                                        f"- {c['title']}, Price: ${c['price']:.2f}, Condition: {c['condition'] or 'Unknown'}, Similarity: {c['similarity']:.2f}"
                                        for c in formatted_comparables[:5]
                                    ])
                                    if not comparable_str:
                                        comparable_str = "No comparable items found. Please evaluate based on watch characteristics alone."
                                    
                                    metadata_str = "\n".join([f"- {k}: {v}" for k, v in metadata.items() if v])
                                    
                                    # Modify the prompt template for cases with insufficient comparables
                                    if len(comparable_items) == 0:
                                        # Special prompt when no comparables exist
                                        prompt = f"""
                                        You are a luxury watch expert evaluating if a listing is a good deal.
                                        
                                        Current Watch:
                                        - Title: {title}
                                        - Price: ${price:.2f}
                                        - Condition: {condition or 'Unknown'}
                                        
                                        Additional details:
                                        {metadata_str}
                                        
                                        No comparable watches were found in our database. Based solely on your knowledge of the watch market:
                                        1. Is this price likely to be a good deal for this watch model?
                                        2. Assign a confidence level (High, Medium, Low)
                                        3. Provide 1-2 sentences explaining your reasoning.
                                        
                                        Format your answer as:
                                        Good Deal: [Yes/No]
                                        Confidence: [High/Medium/Low]
                                        Reason: [Your 1-2 sentence explanation]
                                        """
                                    else:
                                        # Regular prompt with limited comparables
                                        prompt = WATCH_DEAL_EVALUATION_PROMPT.format(
                                            title=title,
                                            price=price,
                                            condition=condition or 'Unknown',
                                            metadata_str=metadata_str,
                                            comparable_str=comparable_str,
                                            market_estimate=market_estimate,
                                            discount_percent=discount_percent
                                        )
                                    
                                    logger.info(f"Calling OpenAI for item {item_id} with model: gpt-4-1106-preview")
                                    
                                    # Call the LLM for deal evaluation
                                    response = client.chat.completions.create(
                                        model="gpt-4-1106-preview",  # Using full model name for clarity
                                        messages=[{"role": "user", "content": prompt}],
                                        max_tokens=300,
                                        temperature=0.2
                                    )
                                    
                                    llm_response = response.choices[0].message.content
                                    logger.info(f"Received OpenAI response for item {item_id}: {llm_response[:100]}...")
                                    
                                    # Extract information from LLM response
                                    import re
                                    
                                    good_deal_match = re.search(r"Good Deal:\s*(Yes|No)", llm_response, re.IGNORECASE)
                                    confidence_match = re.search(r"Confidence:\s*(High|Medium|Low)", llm_response, re.IGNORECASE)
                                    reason_match = re.search(r"Reason:\s*(.+)(?:\n|$)", llm_response, re.IGNORECASE)
                                    
                                    # Parse and convert the LLM's confidence level to a numeric score (1-100)
                                    if confidence_match:
                                        llm_confidence = confidence_match.group(1).lower()
                                        logger.info(f"LLM confidence for item {item_id}: {llm_confidence}")
                                        
                                        # Convert High/Medium/Low to numeric scores
                                        if llm_confidence == "high":
                                            llm_numeric_score = 55  # Lower than normal due to lack of comparables
                                            confidence_text = "Medium"  # Cap at Medium due to insufficient data
                                        elif llm_confidence == "medium":
                                            llm_numeric_score = 40
                                            confidence_text = "Medium" 
                                        else:  # low
                                            llm_numeric_score = 25
                                            confidence_text = "Low"
                                        
                                        # If LLM says it's a good deal, boost its score
                                        if good_deal_match and good_deal_match.group(1).lower() == "yes":
                                            is_good_deal = True
                                            logger.info(f"LLM says item {item_id} is a GOOD DEAL")
                                            llm_numeric_score += 10
                                        else:
                                            is_good_deal = False
                                            logger.info(f"LLM says item {item_id} is NOT a good deal")
                                        
                                        # Update confidence score with LLM's assessment
                                        confidence_score = llm_numeric_score
                                        
                                        # Update reason based on LLM explanation
                                        if reason_match:
                                            reason = reason_match.group(1).strip() + " (Note: Limited comparable data available)"
                                            logger.info(f"LLM reason for {item_id}: {reason}")
                                    
                                    else:
                                        logger.warning(f"Could not parse confidence from LLM response for item {item_id}")
                                else:
                                    logger.error("OpenAI API key not found in environment variables")
                                            
                            except Exception as e:
                                logger.error(f"Error using LLM for evaluation of item {item_id}: {str(e)}")
                                import traceback
                                logger.error(traceback.format_exc())
                        else:
                            logger.info(f"Skipping LLM evaluation for item {item_id} (use_llm={request.use_llm})")
                        
                        # Add to analyzed listings with low confidence and appropriate explanation
                        analyzed_listings.append({
                            "item_id": item_id,
                            "title": title,
                            "price": price,
                            "condition": condition,
                            "item_url": item_url,
                            "market_estimate": market_estimate,
                            "discount_percent": discount_percent,
                            "confidence": confidence_text,
                            "confidence_score": confidence_score,
                            "reason": reason,
                            "comparables": formatted_comparables,
                            "meets_threshold": False
                        })
                        
                        # Continue to the next item
                        continue
                    
                    # Calculate market price from comparables
                    comparable_prices = [float(item[2]) for item in comparable_items]
                    
                    market_estimate = statistics.median(comparable_prices)
                    
                    # Calculate discount percentage
                    if market_estimate > 0:
                        discount_percent = ((market_estimate - price) / market_estimate) * 100
                    else:
                        discount_percent = 0
                    
                    # Format comparable items for response
                    formatted_comparables = []
                    for comp in comparable_items:
                        formatted_comparables.append({
                            "item_id": comp[0],
                            "title": comp[1],
                            "price": float(comp[2]) if comp[2] is not None else 0.0,
                            "condition": comp[3],
                            "similarity": float(comp[4]) if comp[4] is not None else 0.0
                        })
                    
                    # Determine initial confidence score (1-100) based on number and similarity of comparables
                    avg_similarity = sum(c["similarity"] for c in formatted_comparables) / len(formatted_comparables)
                    confidence_score = 0
                    
                    # Base confidence score on comparable quality
                    if avg_similarity > 0.9 and len(formatted_comparables) >= 5:
                        confidence_score = 75  # High base confidence
                    elif avg_similarity > 0.8 and len(formatted_comparables) >= 3:
                        confidence_score = 50  # Medium base confidence
                    else:
                        confidence_score = 30  # Low base confidence
                    
                    # Adjust confidence score based on discount percentage
                    if discount_percent >= request.min_discount_percent:
                        # Boost score for items that meet discount threshold
                        confidence_score = min(100, confidence_score + int(discount_percent))
                        reason = f"{discount_percent:.1f}% below market based on {len(formatted_comparables)} similar watches"
                        confidence_text = "High" if confidence_score >= 70 else "Medium" if confidence_score >= 50 else "Low"
                    else:
                        # For items that don't meet threshold, provide transparent explanation
                        reason = f"Only {discount_percent:.1f}% below market (threshold: {request.min_discount_percent:.1f}%)"
                        confidence_score = max(1, confidence_score - 20)  # Reduce confidence if discount threshold not met
                        confidence_text = "Low"
                        
                    # Use LLM for evaluation if requested
                    if request.use_llm:
                        logger.info(f"LLM evaluation requested for item {item_id} with use_llm={request.use_llm}")
                        try:
                            from openai import OpenAI
                            import os
                            
                            # Get OpenAI API key from environment variables
                            openai_api_key = os.getenv("OPENAI_API_KEY")
                            
                            if openai_api_key:
                                logger.info(f"OpenAI API key found, preparing LLM request for item {item_id}")
                                client = OpenAI(api_key=openai_api_key)
                                
                                # Prepare context for LLM
                                comparable_str = "\n".join([
                                    f"- {c['title']}, Price: ${c['price']:.2f}, Condition: {c['condition'] or 'Unknown'}, Similarity: {c['similarity']:.2f}"
                                    for c in formatted_comparables[:5]
                                ])
                                
                                metadata_str = "\n".join([f"- {k}: {v}" for k, v in metadata.items() if v])
                                
                                prompt = WATCH_DEAL_EVALUATION_PROMPT.format(
                                    title=title,
                                    price=price,
                                    condition=condition or 'Unknown',
                                    metadata_str=metadata_str,
                                    comparable_str=comparable_str,
                                    market_estimate=market_estimate,
                                    discount_percent=discount_percent
                                )
                                
                                logger.info(f"Calling OpenAI for item {item_id} with model: gpt-4-1106-preview")
                                
                                # Call the LLM for deal evaluation
                                response = client.chat.completions.create(
                                    model="gpt-4-1106-preview",  # Using full model name for clarity
                                    messages=[{"role": "user", "content": prompt}],
                                    max_tokens=300,
                                    temperature=0.2
                                )
                                
                                llm_response = response.choices[0].message.content
                                logger.info(f"Received OpenAI response for item {item_id}: {llm_response[:100]}...")
                                
                                # Extract information from LLM response
                                import re
                                
                                good_deal_match = re.search(r"Good Deal:\s*(Yes|No)", llm_response, re.IGNORECASE)
                                confidence_match = re.search(r"Confidence:\s*(High|Medium|Low)", llm_response, re.IGNORECASE)
                                reason_match = re.search(r"Reason:\s*(.+)(?:\n|$)", llm_response, re.IGNORECASE)
                                
                                # Parse and convert the LLM's confidence level to a numeric score (1-100)
                                if confidence_match:
                                    llm_confidence = confidence_match.group(1).lower()
                                    logger.info(f"LLM confidence for item {item_id}: {llm_confidence}")
                                    
                                    # Convert High/Medium/Low to numeric scores
                                    if llm_confidence == "high":
                                        llm_numeric_score = 85
                                        confidence_text = "High"
                                    elif llm_confidence == "medium":
                                        llm_numeric_score = 60
                                        confidence_text = "Medium"
                                    else:  # low
                                        llm_numeric_score = 35
                                        confidence_text = "Low"
                                    
                                    # If LLM says it's a good deal, boost its score
                                    if good_deal_match and good_deal_match.group(1).lower() == "yes":
                                        is_good_deal = True
                                        logger.info(f"LLM says item {item_id} is a GOOD DEAL")
                                        llm_numeric_score += 15
                                        # But don't exceed 100
                                        llm_numeric_score = min(100, llm_numeric_score)
                                    else:
                                        is_good_deal = False
                                        logger.info(f"LLM says item {item_id} is NOT a good deal")
                                    
                                    # Update confidence score with LLM's assessment
                                    # Weighted combination of rule-based and LLM score
                                    old_score = confidence_score
                                    confidence_score = int((confidence_score * 0.3) + (llm_numeric_score * 0.7))
                                    logger.info(f"Updated confidence score for {item_id}: {old_score} → {confidence_score}")
                                    
                                    # Update reason based on LLM explanation
                                    if reason_match:
                                        reason = reason_match.group(1).strip()
                                        logger.info(f"LLM reason for {item_id}: {reason}")
                                    
                                else:
                                    logger.warning(f"Could not parse confidence from LLM response for item {item_id}")
                            else:
                                logger.error("OpenAI API key not found in environment variables")
                                        
                        except Exception as e:
                            logger.error(f"Error using LLM for evaluation of item {item_id}: {str(e)}")
                            import traceback
                            logger.error(traceback.format_exc())
                    else:
                        logger.info(f"Skipping LLM evaluation for item {item_id} (use_llm={request.use_llm})")
                    
                    # Add to analyzed listings list with confidence score for sorting
                    analyzed_listings.append({
                        "item_id": item_id,
                        "title": title,
                        "price": price,
                        "condition": condition,
                        "item_url": item_url,
                        "market_estimate": market_estimate,
                        "discount_percent": discount_percent,
                        "confidence": confidence_text,
                        "confidence_score": confidence_score,  # Numeric score (1-100)
                        "reason": reason,
                        "comparables": formatted_comparables,
                        "meets_threshold": discount_percent >= request.min_discount_percent
                    })
                
                # Sort by confidence score (highest first)
                analyzed_listings.sort(key=lambda x: x["confidence_score"], reverse=True)
                
                # Take top N results (regardless of whether they meet threshold)
                top_listings = analyzed_listings[:request.max_deals]
                
                # Log statistics for transparency
                meets_threshold_count = sum(1 for item in top_listings if item["meets_threshold"])
                logger.info(f"Returning {len(top_listings)} listings (threshold met: {meets_threshold_count}, threshold not met: {len(top_listings) - meets_threshold_count})")
                
                # Format for response (remove internal fields like confidence_score and meets_threshold)
                formatted_results = []
                for item in top_listings:
                    formatted_results.append({
                        "item_id": item["item_id"],
                        "title": item["title"],
                        "price": item["price"],
                        "condition": item["condition"],
                        "item_url": item["item_url"],
                        "market_estimate": item["market_estimate"],
                        "discount_percent": item["discount_percent"],
                        "confidence": item["confidence"],
                        "confidence_score": item["confidence_score"],  # Including the numeric score in the response
                        "reason": item["reason"],
                        "comparables": item["comparables"]
                    })
                
                # If we didn't find any results that meet our criteria, provide an explanation
                if not formatted_results:
                    logger.info("No results meet the criteria, returning empty list with explanation")
                    return [{
                        "item_id": "",
                        "title": "No suitable watches found",
                        "price": 0,
                        "condition": None,
                        "item_url": None,
                        "market_estimate": 0,
                        "discount_percent": 0,
                        "confidence": "Low",
                        "confidence_score": 0,
                        "reason": f"Could not find watches matching your criteria. Try adjusting filters like brand, model, or price range.",
                        "comparables": []
                    }]
                
                return formatted_results
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error finding deals: {str(e)}"
            )
    
    return router

def create_pgvector_query_router() -> APIRouter:
    """Create and return the pgvector query API router"""
    
    router = APIRouter(
        prefix="/api/pgvector",
        tags=["pgvector"],
        responses={404: {"description": "Not found"}},
    )
    
    # Try to get the embedding provider for semantic search
    try:
        embedding_provider = get_embedding_provider()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to initialize embedding provider: {e}")
        embedding_provider = None
    
    @router.post("/search/semantic", response_model=List[ListingResponse])
    async def semantic_search(request: SemanticSearchRequest):
        """
        Search for listings semantically using vector embedding similarity.
        
        This endpoint:
        1. Converts your text query into a vector embedding
        2. Finds listings with similar vector embeddings using pgvector
        3. Returns the most semantically similar listings
        """
        if embedding_provider is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Embedding provider not available - check OPENAI_API_KEY environment variable"
            )
        
        # Get a connection to the database
        storage = default_dual_storage
        conn = storage._get_pg_connection(retry=False)
        
        if conn is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PostgreSQL database not available"
            )
        
        import logging
        logger = logging.getLogger(__name__)
            
        try:
            # Generate embedding for query
            query_embedding = embed_query(request.query, provider=embedding_provider)
            
            # Convert numpy array to Python list
            query_embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            
            with conn.cursor() as cur:
                # Use a much simpler approach for semantic search
                query = ""
                params = []
                
                # Build the query
                if request.price_range:
                    # With price filter
                    query = f"""
                    SELECT 
                        item_id, 
                        title, 
                        price,
                        metadata->>'condition' as condition,
                        metadata->>'item_url' as item_url,
                        created_at,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM 
                        {storage.postgres_table_name}
                    WHERE 
                        price BETWEEN %s AND %s 
                    ORDER BY 
                        embedding <=> %s::vector
                    LIMIT %s
                    """
                    params = [
                        query_embedding_list,
                        request.price_range.min, 
                        request.price_range.max,
                        query_embedding_list,
                        request.limit
                    ]
                else:
                    # Without price filter - even simpler query
                    query = f"""
                    SELECT 
                        item_id, 
                        title, 
                        price,
                        metadata->>'condition' as condition,
                        metadata->>'item_url' as item_url,
                        created_at,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM 
                        {storage.postgres_table_name}
                    ORDER BY 
                        embedding <=> %s::vector
                    LIMIT %s
                    """
                    params = [
                        query_embedding_list,
                        query_embedding_list,
                        request.limit
                    ]
                
                # Log the query and parameters for debugging
                logger.info(f"Executing query: {query}")
                logger.info(f"With parameters: {[str(p)[:50] + '...' if isinstance(p, list) else p for p in params]}")
                
                # Execute the query
                cur.execute(query, params)
                
                # Format the results
                results = []
                for row in cur.fetchall():
                    results.append({
                        "item_id": row[0],
                        "title": row[1],
                        "price": float(row[2]) if row[2] is not None else 0.0,
                        "condition": row[3],
                        "item_url": row[4],
                        "created_at": row[5].isoformat() if row[5] else None,
                        "similarity_score": float(row[6]) if row[6] is not None else 0.0
                    })
                
                logger.info(f"Found {len(results)} results for semantic search query: '{request.query}'")
                
                return results
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error performing semantic search: {str(e)}"
            )
    
    @router.post("/search/similar", response_model=List[ListingResponse])
    async def similar_items(request: SimilarItemsRequest):
        """
        Find similar items to a specific listing.
        
        Given an item ID, this endpoint:
        1. Retrieves the vector embedding for that item
        2. Finds other listings with similar vectors using pgvector
        3. Returns the most similar items
        """
        # Get a connection to the database
        storage = default_dual_storage
        conn = storage._get_pg_connection(retry=False)
        
        if conn is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PostgreSQL database not available"
            )
            
        try:
            with conn.cursor() as cur:
                # First get the embedding for the reference item
                cur.execute(
                    f"SELECT embedding FROM {storage.postgres_table_name} WHERE item_id = %s",
                    (request.item_id,)
                )
                result = cur.fetchone()
                if not result:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Item {request.item_id} not found"
                    )
                
                reference_embedding = result[0]
                
                # Find similar items using cosine similarity
                cur.execute(
                    f"""
                    SELECT 
                        item_id, 
                        title, 
                        price,
                        metadata->>'condition' as condition,
                        metadata->>'item_url' as item_url,
                        created_at,
                        1 - (embedding <=> %s) AS similarity
                    FROM 
                        {storage.postgres_table_name}
                    WHERE 
                        item_id != %s
                        AND (1 - (embedding <=> %s)) >= %s
                    ORDER BY 
                        embedding <=> %s
                    LIMIT %s
                    """,
                    (
                        reference_embedding, 
                        request.item_id, 
                        reference_embedding,
                        request.min_similarity,
                        reference_embedding,
                        request.limit
                    )
                )
                
                # Format the results
                results = []
                for row in cur.fetchall():
                    results.append({
                        "item_id": row[0],
                        "title": row[1],
                        "price": float(row[2]) if row[2] is not None else 0.0,
                        "condition": row[3],
                        "item_url": row[4],
                        "created_at": row[5].isoformat() if row[5] else None,
                        "similarity_score": float(row[6]) if row[6] is not None else 0.0
                    })
                
                return results
                
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error finding similar items: {str(e)}"
            )
    
    @router.post("/search/listings", response_model=List[ListingResponse])
    async def search_listings(request: ListingSearchRequest):
        """
        Search for listings with traditional filtering.
        
        This endpoint allows searching by:
        1. Text in the title
        2. Price range
        3. Condition
        """
        # Get a connection to the database
        storage = default_dual_storage
        conn = storage._get_pg_connection(retry=False)
        
        if conn is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PostgreSQL database not available"
            )
            
        try:
            with conn.cursor() as cur:
                # Build the query with filters
                params = []
                conditions = []
                
                if request.text:
                    conditions.append("title ILIKE %s")
                    params.append(f"%{request.text}%")
                
                if request.min_price is not None:
                    conditions.append("price >= %s")
                    params.append(request.min_price)
                    
                if request.max_price is not None:
                    conditions.append("price <= %s")
                    params.append(request.max_price)
                    
                if request.condition:
                    conditions.append("metadata->>'condition' ILIKE %s")
                    params.append(f"%{request.condition}%")
                
                # Construct the WHERE clause
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                
                # Add the limit
                params.append(request.limit)
                
                # Execute the query
                query = f"""
                SELECT 
                    item_id, 
                    title, 
                    price,
                    metadata->>'condition' as condition,
                    metadata->>'item_url' as item_url,
                    created_at
                FROM 
                    {storage.postgres_table_name}
                WHERE 
                    {where_clause}
                ORDER BY 
                    updated_at DESC
                LIMIT %s
                """
                
                cur.execute(query, params)
                
                # Format the results
                results = []
                for row in cur.fetchall():
                    results.append({
                        "item_id": row[0],
                        "title": row[1],
                        "price": float(row[2]),
                        "condition": row[3],
                        "item_url": row[4],
                        "created_at": row[5].isoformat() if row[5] else None
                    })
                
                return results
                
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error searching listings: {str(e)}"
            )
    
    @router.get("/stats", response_model=Dict[str, Any])
    async def get_statistics():
        """
        Get statistics about the pgvector database.
        
        Returns information about:
        1. Total number of listings
        2. Price range statistics
        3. Most common conditions
        4. Recent items
        """
        # Get a connection to the database
        storage = default_dual_storage
        conn = storage._get_pg_connection(retry=False)
        
        if conn is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PostgreSQL database not available"
            )
            
        try:
            stats = {}
            
            with conn.cursor() as cur:
                # Get total count
                cur.execute(f"SELECT COUNT(*) FROM {storage.postgres_table_name}")
                stats["total_listings"] = cur.fetchone()[0]
                
                # Get price statistics
                cur.execute(f"""
                SELECT 
                    MIN(price),
                    MAX(price),
                    AVG(price)::numeric(10,2),
                    PERCENTILE_CONT(0.5) WITHIN GROUP(ORDER BY price)::numeric(10,2) as median
                FROM {storage.postgres_table_name}
                """)
                
                price_stats = cur.fetchone()
                stats["price_stats"] = {
                    "min": float(price_stats[0]) if price_stats[0] else 0,
                    "max": float(price_stats[1]) if price_stats[1] else 0,
                    "average": float(price_stats[2]) if price_stats[2] else 0,
                    "median": float(price_stats[3]) if price_stats[3] else 0
                }
                
                # Get top conditions
                cur.execute(f"""
                SELECT 
                    metadata->>'condition' as condition,
                    COUNT(*) as count
                FROM {storage.postgres_table_name}
                WHERE metadata->>'condition' IS NOT NULL
                GROUP BY metadata->>'condition'
                ORDER BY count DESC
                LIMIT 5
                """)
                
                stats["top_conditions"] = [
                    {"condition": row[0], "count": row[1]}
                    for row in cur.fetchall()
                ]
                
                # Get embedding statistics
                cur.execute(f"""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) as with_embedding,
                    SUM(CASE WHEN embedding IS NULL THEN 1 ELSE 0 END) as without_embedding
                FROM {storage.postgres_table_name}
                """)
                
                embedding_stats = cur.fetchone()
                stats["embedding_stats"] = {
                    "total": embedding_stats[0],
                    "with_embedding": embedding_stats[1],
                    "without_embedding": embedding_stats[2],
                    "coverage_percent": round((embedding_stats[1] / embedding_stats[0]) * 100, 2) if embedding_stats[0] > 0 else 0
                }
                
                # Get recent items
                cur.execute(f"""
                SELECT 
                    created_at::date as date,
                    COUNT(*) as count
                FROM {storage.postgres_table_name}
                GROUP BY date
                ORDER BY date DESC
                LIMIT 7
                """)
                
                stats["recent_items"] = [
                    {"date": row[0].isoformat(), "count": row[1]}
                    for row in cur.fetchall()
                ]
                
                return stats
                
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting statistics: {str(e)}"
            )
    
    return router