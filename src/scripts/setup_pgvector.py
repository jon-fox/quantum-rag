#!/usr/bin/env python3
"""
Script to connect to the RDS PostgreSQL instance and create the pgvector extension.

This script will:
1. Fetch database credentials from AWS SSM Parameter Store
2. Connect to the PostgreSQL database
3. Create the pgvector extension
4. Create the watch_embeddings table with vector support

Usage:
    python src/scripts/setup_pgvector.py [--environment prod]
"""

import os
import sys
import argparse
import boto3
import psycopg2
import logging
from psycopg2 import sql

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pgvector_setup")

# Ensure the parent directory is in the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def get_db_params_from_ssm(environment="prod", region="us-east-1"):
    """
    Get database connection parameters from SSM Parameter Store
    
    Args:
        environment: The deployment environment (prod, dev, etc.)
        region: AWS region
        
    Returns:
        Dictionary with connection parameters
    """
    try:
        # Initialize SSM client
        ssm = boto3.client('ssm', region_name=region)
        
        # Parameter path prefix
        param_prefix = f"/{environment}/watch-arb/db"
        
        # Get parameters
        params = {}
        param_names = ["username", "password", "address", "port", "name"]
        
        for param_name in param_names:
            try:
                response = ssm.get_parameter(
                    Name=f"{param_prefix}/{param_name}",
                    WithDecryption=True if param_name == "password" else False
                )
                params[param_name] = response["Parameter"]["Value"]
                logger.info(f"Retrieved {param_name} from SSM Parameter Store")
            except Exception as e:
                logger.warning(f"Failed to get parameter {param_name}: {e}")
                return None
        
        # Return connection parameters
        return {
            "user": params["username"],
            "password": params["password"],
            "host": params["address"],
            "port": params["port"],
            "dbname": params["name"]
        }
        
    except Exception as e:
        logger.error(f"Failed to get PostgreSQL connection parameters from SSM: {e}")
        return None

def get_db_params_from_env():
    """
    Get database connection parameters from environment variables as fallback
    
    Returns:
        Dictionary with connection parameters
    """
    params = {
        "user": os.environ.get("DB_USER", "watchadmin"),
        "password": os.environ.get("DB_PASSWORD", ""),
        "host": os.environ.get("DB_HOST", ""),
        "port": os.environ.get("DB_PORT", "5432"),
        "dbname": os.environ.get("DB_NAME", "watch_arb")
    }
    
    # Check if we have the minimum required parameters
    if not params["password"] or not params["host"]:
        logger.error("Missing required database parameters in environment variables")
        return None
        
    return params

def setup_pgvector(db_params, vector_dim=1536, table_name="watch_embeddings"):
    """
    Connect to the database and set up pgvector
    
    Args:
        db_params: Dictionary with connection parameters
        vector_dim: Dimension of vectors to use
        table_name: Name of the table to create
        
    Returns:
        True if successful, False otherwise
    """
    if not db_params:
        logger.error("No database parameters provided")
        return False
    
    conn = None
    try:
        # Connect to PostgreSQL
        logger.info(f"Connecting to PostgreSQL at {db_params['host']}:{db_params['port']}")
        conn = psycopg2.connect(**db_params)
        conn.autocommit = False
        
        with conn.cursor() as cur:
            # Create the pgvector extension
            logger.info("Creating pgvector extension...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Check if extension was created
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            if cur.fetchone():
                logger.info("✅ pgvector extension created successfully")
            else:
                logger.error("❌ Failed to create pgvector extension")
                return False
            
            # Create the watch_embeddings table
            logger.info(f"Creating {table_name} table with vector({vector_dim}) support...")
            cur.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS {} (
                item_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                price NUMERIC(10, 2) NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                embedding vector({}),
                metadata JSONB
            );
            """).format(sql.Identifier(table_name), sql.Literal(vector_dim)))
            
            # Create index for vector search - FIX: Correctly format the index name
            logger.info("Creating vector similarity search index...")
            index_name = f"idx_{table_name}_embedding"
            cur.execute(sql.SQL("""
            CREATE INDEX IF NOT EXISTS {} 
            ON {} USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """).format(sql.Identifier(index_name), sql.Identifier(table_name)))
            
            # Commit the changes
            conn.commit()
            logger.info(f"✅ Table {table_name} created successfully with vector support")
            
            return True
            
    except Exception as e:
        logger.error(f"Error setting up pgvector: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def main():
    """Main function to set up pgvector"""
    parser = argparse.ArgumentParser(description='Set up pgvector extension in RDS PostgreSQL')
    parser.add_argument('--environment', type=str, default="prod", 
                        help='Environment (prod, dev, etc.) for SSM parameters')
    parser.add_argument('--region', type=str, default="us-east-1",
                        help='AWS region for SSM parameters')
    parser.add_argument('--use-env', action='store_true',
                        help='Use environment variables instead of SSM')
    parser.add_argument('--table', type=str, default="watch_embeddings",
                        help='Name of the table to create')
    parser.add_argument('--vector-dim', type=int, default=1536,
                        help='Dimension of vector embeddings (default: 1536 for OpenAI)')
    
    args = parser.parse_args()
    
    # Get database parameters
    db_params = None
    if args.use_env:
        logger.info("Using database parameters from environment variables")
        db_params = get_db_params_from_env()
    else:
        logger.info(f"Getting database parameters from SSM Parameter Store in {args.environment} environment")
        db_params = get_db_params_from_ssm(args.environment, args.region)
        
    if not db_params:
        logger.error("Failed to get database parameters")
        sys.exit(1)
    
    # Set up pgvector
    success = setup_pgvector(db_params, args.vector_dim, args.table)
    if success:
        logger.info("✅ pgvector setup completed successfully")
    else:
        logger.error("❌ pgvector setup failed")
        sys.exit(1)

if __name__ == "__main__":
    main()