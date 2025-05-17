#!/usr/bin/env python3
"""
Script to review embeddings stored in PostgreSQL with pgvector.
This script connects to the database and retrieves stored embeddings
for analysis and verification.

Usage:
    python src/scripts/review_pgvector_embeddings.py [--limit 10] [--display-vectors]
"""

import os
import sys
import argparse
import boto3
import psycopg2
import numpy as np
import json
from tabulate import tabulate
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pgvector_review")

# Ensure the parent directory is in the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from project
from src.storage.dual_storage import DualStorage

def get_embeddings_from_postgres(limit=10, display_vectors=False, storage=None):
    """
    Fetch embeddings from PostgreSQL
    
    Args:
        limit: Maximum number of embeddings to fetch
        display_vectors: Whether to include the full vector data in output
        storage: Optional DualStorage instance
        
    Returns:
        List of dictionaries with embedding data
    """
    storage = storage or DualStorage(lazy_init=False)
    conn = storage._get_pg_connection()
    
    if not conn:
        logger.error("Failed to connect to PostgreSQL")
        return []
    
    try:
        with conn.cursor() as cur:
            # Check if the table exists
            cur.execute(f"SELECT to_regclass('{storage.postgres_table_name}');")
            if not cur.fetchone()[0]:
                logger.error(f"Table '{storage.postgres_table_name}' does not exist")
                return []
            
            # Count stored embeddings
            cur.execute(f"SELECT COUNT(*) FROM {storage.postgres_table_name};")
            count = cur.fetchone()[0]
            logger.info(f"Found {count} embeddings in the database")
            
            # Fetch embeddings with basic information
            if display_vectors:
                cur.execute(f"""
                SELECT 
                    item_id, 
                    title, 
                    price, 
                    created_at, 
                    updated_at, 
                    embedding,
                    metadata
                FROM {storage.postgres_table_name}
                ORDER BY updated_at DESC
                LIMIT %s;
                """, (limit,))
            else:
                # Skip the large embedding vector to save output space
                cur.execute(f"""
                SELECT 
                    item_id, 
                    title, 
                    price, 
                    created_at, 
                    updated_at,
                    array_length(embedding, 1) as embedding_dimension,
                    metadata
                FROM {storage.postgres_table_name}
                ORDER BY updated_at DESC
                LIMIT %s;
                """, (limit,))
                
            # Fetch results
            columns = [desc[0] for desc in cur.description]
            results = []
            
            for row in cur.fetchall():
                result = dict(zip(columns, row))
                
                # Process embedding if included
                if display_vectors and 'embedding' in result:
                    # Convert the embedding to a NumPy array for analysis
                    embedding = np.array(result['embedding'])
                    result['embedding'] = {
                        'shape': embedding.shape,
                        'min': float(embedding.min()),
                        'max': float(embedding.max()),
                        'mean': float(embedding.mean()),
                        'std': float(embedding.std()),
                        # Sample first 5 values
                        'sample': embedding[:5].tolist()
                    }
                
                # Format timestamps for display
                if 'created_at' in result:
                    result['created_at'] = result['created_at'].isoformat()
                if 'updated_at' in result:
                    result['updated_at'] = result['updated_at'].isoformat()
                
                results.append(result)
            
            return results
    
    except Exception as e:
        logger.error(f"Error fetching embeddings: {e}")
        return []

def perform_cosine_similarity_test(storage=None, item_id=None):
    """
    Test cosine similarity search for a specific item or random item
    
    Args:
        storage: Optional DualStorage instance
        item_id: Optional item ID to use as reference
        
    Returns:
        Dictionary with search results
    """
    storage = storage or DualStorage(lazy_init=False)
    conn = storage._get_pg_connection()
    
    if not conn:
        logger.error("Failed to connect to PostgreSQL")
        return {}
    
    try:
        with conn.cursor() as cur:
            # Find a reference item if none specified
            if not item_id:
                cur.execute(f"SELECT item_id FROM {storage.postgres_table_name} LIMIT 1;")
                result = cur.fetchone()
                if not result:
                    logger.error("No items found in database")
                    return {}
                item_id = result[0]
            
            # Get the embedding for the reference item
            cur.execute(
                f"SELECT embedding FROM {storage.postgres_table_name} WHERE item_id = %s",
                (item_id,)
            )
            result = cur.fetchone()
            if not result:
                logger.error(f"Item {item_id} not found")
                return {}
            
            reference_embedding = result[0]
            
            # Find similar items using cosine similarity
            cur.execute(
                f"""
                SELECT 
                    item_id, 
                    title, 
                    price, 
                    metadata->>'condition' as condition,
                    1 - (embedding <=> %s) AS similarity
                FROM 
                    {storage.postgres_table_name}
                ORDER BY 
                    embedding <=> %s
                LIMIT 5
                """,
                (reference_embedding, reference_embedding)
            )
            
            # Fetch results
            columns = [desc[0] for desc in cur.description]
            similar_items = []
            
            for row in cur.fetchall():
                similar_items.append(dict(zip(columns, row)))
                
            return {
                "reference_item_id": item_id,
                "similar_items": similar_items
            }
    
    except Exception as e:
        logger.error(f"Error performing similarity test: {e}")
        return {}

def main():
    """Main function to fetch and display embeddings"""
    parser = argparse.ArgumentParser(description='Review pgvector embeddings from PostgreSQL')
    parser.add_argument('--limit', type=int, default=10, 
                        help='Maximum number of embeddings to display')
    parser.add_argument('--display-vectors', action='store_true',
                        help='Include vector data in the output')
    parser.add_argument('--similarity-test', action='store_true',
                        help='Run a cosine similarity test')
    parser.add_argument('--item-id', type=str,
                        help='Item ID to use for similarity test')
    
    args = parser.parse_args()
    
    # Initialize storage
    storage = DualStorage(lazy_init=False)
    
    # Fetch and display embeddings
    embeddings = get_embeddings_from_postgres(
        limit=args.limit, 
        display_vectors=args.display_vectors,
        storage=storage
    )
    
    if not embeddings:
        logger.error("No embeddings found or could not connect to database")
        sys.exit(1)
    
    # Display results
    print("\n===== Embeddings Review =====\n")
    
    # Create a simplified version for tabular display
    table_data = []
    for e in embeddings:
        row = {
            'item_id': e.get('item_id'),
            'title': (e.get('title', '')[:30] + '...') if len(e.get('title', '')) > 30 else e.get('title', ''),
            'price': e.get('price'),
            'condition': e.get('metadata', {}).get('condition', 'Unknown')
        }
        
        # Add embedding info if available
        if 'embedding_dimension' in e:
            row['embedding_dim'] = e.get('embedding_dimension')
        elif 'embedding' in e:
            row['embedding_info'] = f"dim={e['embedding']['shape'][0]}, mean={e['embedding']['mean']:.4f}"
        
        table_data.append(row)
    
    # Print table
    print(tabulate(table_data, headers="keys", tablefmt="pretty"))
    
    # Run similarity test if requested
    if args.similarity_test:
        print("\n===== Similarity Test =====\n")
        sim_results = perform_cosine_similarity_test(storage=storage, item_id=args.item_id)
        
        if sim_results and 'similar_items' in sim_results:
            print(f"Reference item: {sim_results['reference_item_id']}")
            print("\nSimilar items:")
            
            sim_table = []
            for item in sim_results['similar_items']:
                sim_table.append({
                    'item_id': item.get('item_id'),
                    'title': (item.get('title', '')[:40] + '...') if len(item.get('title', '')) > 40 else item.get('title', ''),
                    'price': item.get('price'),
                    'condition': item.get('condition', 'Unknown'),
                    'similarity': f"{item.get('similarity', 0):.4f}"
                })
            
            print(tabulate(sim_table, headers="keys", tablefmt="pretty"))
        else:
            print("Could not perform similarity test")
    
    print("\nDone!")

if __name__ == "__main__":
    main()