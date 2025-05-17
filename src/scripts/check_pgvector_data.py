#!/usr/bin/env python3
"""
Simple script to check if there are embeddings in the PostgreSQL database.
This script will help diagnose issues with pgvector queries.

Usage:
    python src/scripts/check_pgvector_data.py
"""

import os
import sys
import psycopg2
import json
import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pgvector_check")

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the DualStorage class
from src.storage.dual_storage import DualStorage

def check_postgres_connection():
    """Check if we can connect to PostgreSQL"""
    try:
        storage = DualStorage()
        conn = storage._get_pg_connection()
        if conn:
            logger.info("Successfully connected to PostgreSQL")
            return conn, storage
        else:
            logger.error("Failed to connect to PostgreSQL")
            return None, None
    except Exception as e:
        logger.error(f"Error connecting to PostgreSQL: {e}")
        return None, None

def check_table_exists(conn, table_name):
    """Check if the table exists in the database"""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (table_name,))
            result = cur.fetchone()[0]
            if result:
                logger.info(f"Table '{table_name}' exists")
                return True
            else:
                logger.error(f"Table '{table_name}' does not exist")
                return False
    except Exception as e:
        logger.error(f"Error checking if table exists: {e}")
        return False

def check_table_structure(conn, table_name):
    """Check the structure of the table"""
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
            SELECT 
                column_name, 
                data_type 
            FROM 
                information_schema.columns 
            WHERE 
                table_name = %s
            ORDER BY 
                ordinal_position
            """, (table_name,))
            
            columns = cur.fetchall()
            if columns:
                logger.info(f"Table structure for '{table_name}':")
                for col in columns:
                    logger.info(f"  {col[0]}: {col[1]}")
                
                # Check specifically for the embedding column
                embedding_cols = [col for col in columns if col[0] == 'embedding']
                if embedding_cols:
                    logger.info(f"Embedding column found with type: {embedding_cols[0][1]}")
                else:
                    logger.warning("No 'embedding' column found in the table")
                
                return columns
            else:
                logger.error(f"No columns found for table '{table_name}'")
                return []
    except Exception as e:
        logger.error(f"Error checking table structure: {e}")
        return []

def check_row_counts(conn, table_name):
    """Check the number of rows in the table"""
    try:
        with conn.cursor() as cur:
            # Total rows
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            total = cur.fetchone()[0]
            
            # Rows with embeddings
            cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE embedding IS NOT NULL")
            with_embedding = cur.fetchone()[0]
            
            # Rows without embeddings
            without_embedding = total - with_embedding
            
            logger.info(f"Row counts for '{table_name}':")
            logger.info(f"  Total rows: {total}")
            logger.info(f"  Rows with embeddings: {with_embedding}")
            logger.info(f"  Rows without embeddings: {without_embedding}")
            
            return {
                'total': total,
                'with_embedding': with_embedding,
                'without_embedding': without_embedding
            }
    except Exception as e:
        logger.error(f"Error checking row counts: {e}")
        return {
            'total': 0,
            'with_embedding': 0,
            'without_embedding': 0
        }

def check_sample_data(conn, table_name, limit=3):
    """Check sample data from the table"""
    try:
        with conn.cursor() as cur:
            # Get sample rows with embeddings
            cur.execute(f"""
            SELECT 
                item_id, 
                title, 
                price,
                created_at,
                updated_at
            FROM {table_name}
            WHERE embedding IS NOT NULL
            LIMIT %s
            """, (limit,))
            
            rows = cur.fetchall()
            
            if rows:
                logger.info(f"Sample data from '{table_name}' (with embeddings):")
                for row in rows:
                    logger.info(f"  ID: {row[0]}, Title: {row[1]}, Price: {row[2]}")
                    
                # Try to check the embedding dimensions for the first row
                try:
                    cur.execute(f"""
                    SELECT embedding
                    FROM {table_name}
                    WHERE item_id = %s
                    """, (rows[0][0],))
                    
                    embedding = cur.fetchone()[0]
                    
                    # Check if embedding is a list (pgvector typically returns as list)
                    if isinstance(embedding, list):
                        logger.info(f"  Embedding dimensions: {len(embedding)}")
                        logger.info(f"  Embedding sample: {embedding[:5]}...")
                    else:
                        logger.info(f"  Embedding type: {type(embedding)}")
                except Exception as e:
                    logger.error(f"Error checking embedding dimensions: {e}")
            else:
                logger.warning("No rows found with embeddings")
    except Exception as e:
        logger.error(f"Error checking sample data: {e}")

def test_simple_vector_query(conn, table_name):
    """Test a simple vector query to see if pgvector is working"""
    try:
        with conn.cursor() as cur:
            # First check if there's data
            cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE embedding IS NOT NULL")
            count = cur.fetchone()[0]
            
            if count == 0:
                logger.warning("No embeddings to test vector query with")
                return False
            
            # Get a sample embedding
            cur.execute(f"SELECT item_id, embedding FROM {table_name} WHERE embedding IS NOT NULL LIMIT 1")
            sample = cur.fetchone()
            
            if not sample:
                logger.warning("Could not retrieve a sample embedding")
                return False
            
            sample_id, sample_embedding = sample
            
            logger.info(f"Testing vector query with sample ID: {sample_id}")
            
            # Try a simple self-similarity query (should return the item itself first)
            try:
                cur.execute(f"""
                SELECT 
                    item_id, 
                    1 - (embedding <=> %s) AS similarity
                FROM {table_name}
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s
                LIMIT 3
                """, (sample_embedding, sample_embedding))
                
                results = cur.fetchall()
                
                if results:
                    logger.info("Vector query successful! Results:")
                    for r in results:
                        logger.info(f"  ID: {r[0]}, Similarity: {r[1]}")
                    return True
                else:
                    logger.warning("Vector query returned no results")
                    return False
                    
            except Exception as e:
                logger.error(f"Error executing vector query: {e}")
                return False
                
    except Exception as e:
        logger.error(f"Error testing vector query: {e}")
        return False

def main():
    """Main function"""
    logger.info("Checking PostgreSQL and pgvector setup...")
    
    conn, storage = check_postgres_connection()
    if not conn:
        logger.error("Exiting due to connection failure")
        sys.exit(1)
    
    table_name = storage.postgres_table_name
    
    if not check_table_exists(conn, table_name):
        logger.error(f"Table '{table_name}' does not exist. Exiting.")
        sys.exit(1)
    
    # Check table structure
    check_table_structure(conn, table_name)
    
    # Check row counts
    counts = check_row_counts(conn, table_name)
    
    if counts['with_embedding'] == 0:
        logger.warning("No rows have embeddings. That's why semantic search returns no results.")
        logger.info("You need to store items with embeddings first.")
        sys.exit(0)
    
    # Check sample data
    check_sample_data(conn, table_name)
    
    # Test vector query
    if test_simple_vector_query(conn, table_name):
        logger.info("pgvector seems to be working properly!")
    else:
        logger.warning("pgvector queries may not be working correctly.")
    
    logger.info("Check complete!")

if __name__ == "__main__":
    main()