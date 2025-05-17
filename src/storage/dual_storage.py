"""
Dual storage module for storing energy consumption data in both DynamoDB and PostgreSQL.
The PostgreSQL database uses pgvector extension to store and query vector embeddings.
"""
import json
import boto3
import logging
import os
import psycopg2
import numpy as np
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple, Union
from botocore.exceptions import ClientError
from psycopg2.extras import Json, execute_values
from psycopg2 import sql

# Import existing DynamoDB storage
from src.storage.dynamodb import DynamoDBStorage

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DualStorage:
    """
    Class for storing energy consumption data in both DynamoDB and PostgreSQL with pgvector.
    
    Utilizes AWS SSM Parameter Store to retrieve database credentials and connection info.
    """
    
    def __init__(
        self,
        dynamodb_table_name: str = None,
        postgres_table_name: str = "energy_embeddings",
        region: str = None,
        environment: str = None,
        ssm_param_prefix: str = None,
        lazy_init: bool = True
    ):
        """
        Initialize dual storage handler
        
        Args:
            dynamodb_table_name: Override default DynamoDB table name
            postgres_table_name: PostgreSQL table name for embeddings
            region: AWS region, defaults to env var or boto default
            environment: Environment (prod, dev, etc.) for SSM parameters
            ssm_param_prefix: Prefix for SSM parameters
            lazy_init: Whether to delay PostgreSQL initialization until first use
        """
        # Initialize attributes
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.environment = environment or os.environ.get("ENVIRONMENT", "prod")
        self.ssm_param_prefix = ssm_param_prefix or f"/{self.environment}/energy-data/db"
        self.postgres_table_name = postgres_table_name
        self.schema_initialized = False
        
        # Initialize DynamoDB storage
        self.dynamo_storage = DynamoDBStorage(table_name=dynamodb_table_name)
        
        # Initialize PostgreSQL connection (lazy - will connect when needed)
        self.pg_conn = None
        self.pg_params = None
        
        # Set up vector dimension
        self.vector_dim = 1536  # Default for OpenAI embeddings
        
        # Initialize PostgreSQL schema if not using lazy init
        if not lazy_init:
            self._init_postgres_schema()

    def _get_pg_connection_params(self) -> Dict[str, str]:
        """
        Get PostgreSQL connection parameters from SSM Parameter Store
        
        Returns:
            Dictionary with connection parameters
        """
        if self.pg_params is not None:
            return self.pg_params
            
        try:
            # Initialize SSM client
            ssm = boto3.client('ssm', region_name=self.region)
            
            # Get parameters from SSM
            params = {}
            param_names = ["username", "password", "address", "port", "name"]
            
            for param_name in param_names:
                try:
                    response = ssm.get_parameter(
                        Name=f"{self.ssm_param_prefix}/{param_name}",
                        WithDecryption=True if param_name == "password" else False
                    )
                    params[param_name] = response["Parameter"]["Value"]
                except Exception as e:
                    logger.warning(f"Failed to get parameter {param_name}: {e}")
                    # Use defaults if parameter not found
                    if param_name == "username":
                        params[param_name] = "energyadmin"
                    elif param_name == "password":
                        params[param_name] = os.environ.get("DB_PASSWORD", "")
                    elif param_name == "address":
                        params[param_name] = os.environ.get("DB_HOST", "localhost")
                    elif param_name == "port":
                        params[param_name] = os.environ.get("DB_PORT", "5432")
                    elif param_name == "name":
                        params[param_name] = os.environ.get("DB_NAME", "energy_data")
            
            # Cache the parameters for future use
            self.pg_params = {
                "user": params.get("username", "energyadmin"),
                "password": params.get("password", os.environ.get("DB_PASSWORD", "")),
                "host": params.get("address", os.environ.get("DB_HOST", "localhost")),
                "port": params.get("port", os.environ.get("DB_PORT", "5432")),
                "dbname": params.get("name", os.environ.get("DB_NAME", "energy_data"))
            }
            
            return self.pg_params
            
        except Exception as e:
            logger.error(f"Failed to get PostgreSQL connection parameters: {e}")
            # For development fallback - NOT for production use!
            self.pg_params = {
                "user": os.environ.get("DB_USER", "energyadmin"),
                "password": os.environ.get("DB_PASSWORD", ""),
                "host": os.environ.get("DB_HOST", "localhost"),
                "port": os.environ.get("DB_PORT", "5432"),
                "dbname": os.environ.get("DB_NAME", "energy_data")
            }
            return self.pg_params

    def _get_pg_connection(self, retry=True):
        """
        Get PostgreSQL connection, creating it if needed
        
        Args:
            retry: Whether to retry connection on failure
            
        Returns:
            PostgreSQL connection object or None if connection fails
        """
        if self.pg_conn is None or self.pg_conn.closed:
            try:
                params = self._get_pg_connection_params()
                self.pg_conn = psycopg2.connect(**params)
                self.pg_conn.autocommit = False
                logger.info(f"Connected to PostgreSQL at {params['host']}")
            except Exception as e:
                logger.error(f"Failed to connect to PostgreSQL: {e}")
                self.pg_conn = None
                if not retry:
                    raise
        return self.pg_conn

    def _init_postgres_schema(self) -> bool:
        """
        Initialize PostgreSQL schema with pgvector extension and table
        
        Returns:
            Boolean indicating success
        """
        if self.schema_initialized:
            return True
            
        conn = None
        try:
            # Connect to PostgreSQL
            conn = self._get_pg_connection(retry=False)
            if conn is None:
                logger.warning("Skipping schema initialization, no PostgreSQL connection available")
                return False
                
            with conn.cursor() as cur:
                # Enable pgvector extension if not already enabled
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Create table if not exists
                cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.postgres_table_name} (
                    item_id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    efficiency NUMERIC(10, 2) NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    embedding vector({self.vector_dim}),
                    metadata JSONB
                );
                """)
                
                # Create index for vector search
                cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.postgres_table_name}_embedding 
                ON {self.postgres_table_name} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
                """)
                
                conn.commit()
                logger.info(f"PostgreSQL schema initialized with table {self.postgres_table_name}")
                self.schema_initialized = True
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL schema: {e}")
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass  # Ignore rollback errors, connection might be broken
            return False

    def store_energy_data_with_embedding(
        self, 
        energy_data: Dict[str, Any], 
        embedding: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Store energy consumption data in both DynamoDB and PostgreSQL with its embedding
        
        Args:
            energy_data: Energy data dictionary
            embedding: Vector embedding as numpy array (1536 dimensions)
            
        Returns:
            Dict with results for both storage operations
        """
        results = {
            "dynamo": {"success": False},
            "postgres": {"success": False},
            "overall_success": False
        }
        
        # Extract required fields with validation
        item_id = energy_data.get("dataId")
        if not item_id:
            error_msg = "Missing required field: dataId"
            logger.error(error_msg)
            results["error"] = error_msg
            return results
        
        # Store in DynamoDB
        try:
            dynamo_result = self.dynamo_storage.store_energy_data(energy_data)
            results["dynamo"] = dynamo_result
        except Exception as e:
            logger.error(f"DynamoDB storage error for item {item_id}: {e}")
            results["dynamo"] = {"success": False, "error": str(e)}
        
        # Store in PostgreSQL if embedding is provided and connection works
        if embedding is not None:
            # Try to initialize schema if needed
            if not self.schema_initialized:
                self._init_postgres_schema()
                
            # Only attempt to store if we can connect
            if self._get_pg_connection(retry=False) is not None:
                try:
                    pg_result = self._store_in_postgres(energy_data, embedding)
                    results["postgres"] = pg_result
                except Exception as e:
                    logger.error(f"PostgreSQL storage error for item {item_id}: {e}")
                    results["postgres"] = {"success": False, "error": str(e)}
            else:
                results["postgres"] = {"success": False, "error": "PostgreSQL connection unavailable"}
        else:
            results["postgres"] = {"success": False, "error": "No embedding provided"}
        
        # Determine overall success - consider it successful if DynamoDB worked
        # This makes the app resilient to PostgreSQL availability issues
        results["overall_success"] = results["dynamo"].get("success", False)
        results["item_id"] = item_id
        
        return results
    
    def _store_in_postgres(self, energy_data: Dict[str, Any], embedding: np.ndarray) -> Dict[str, Any]:
        """
        Store energy data with its embedding in PostgreSQL
        
        Args:
            energy_data: Energy consumption data dictionary
            embedding: Vector embedding as numpy array
            
        Returns:
            Dict with result information
        """
        # Extract required fields
        item_id = energy_data.get("dataId")
        title = energy_data.get("description", "Unknown")
        
        # Extract efficiency with validation
        efficiency = 0.0
        efficiency_data = energy_data.get("efficiency", {})
        if efficiency_data and isinstance(efficiency_data, dict):
            efficiency_value = efficiency_data.get("value")
            if efficiency_value:
                try:
                    efficiency = float(efficiency_value)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid efficiency value for item {item_id}, using 0.0")
        elif isinstance(energy_data.get("efficiency"), (int, float)):
            efficiency = float(energy_data.get("efficiency"))
        
        # Prepare metadata (store additional useful fields)
        metadata = {
            "source_type": energy_data.get("source_type"),
            "data_url": energy_data.get("dataUrl"),
            "unit": efficiency_data.get("unit", "kWh") if isinstance(efficiency_data, dict) else "kWh",
            "raw_data": json.dumps({k: v for k, v in energy_data.items() if k not in ["dataId", "description", "efficiency"]})
        }
        
        try:
            # Connect to PostgreSQL
            conn = self._get_pg_connection()
            with conn.cursor() as cur:
                # Insert or update using UPSERT (INSERT ... ON CONFLICT ... DO UPDATE)
                cur.execute(
                    sql.SQL("""
                    INSERT INTO {} (item_id, title, price, embedding, metadata, updated_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (item_id) 
                    DO UPDATE SET 
                        title = EXCLUDED.title,
                        price = EXCLUDED.price,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    """).format(sql.Identifier(self.postgres_table_name)),
                    (
                        item_id, 
                        title, 
                        efficiency, 
                        embedding.tolist(),  # Convert numpy array to Python list
                        Json(metadata)
                    )
                )
                
                conn.commit()
                logger.info(f"Successfully stored energy data {item_id} in PostgreSQL with embedding")
                return {"success": True, "item_id": item_id}
                
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to store energy data {item_id} in PostgreSQL: {e}")
            return {"success": False, "item_id": item_id, "error": str(e)}

    def batch_store_energy_data_with_embeddings(
        self, 
        items: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Store multiple energy data entries in both DynamoDB and PostgreSQL
        
        Args:
            items: List of energy data dictionaries
            embeddings: Optional array of embeddings (one per item)
            
        Returns:
            Dict with results summary
        """
        results = {
            "total": len(items),
            "dynamo_succeeded": 0,
            "dynamo_failed": 0,
            "postgres_succeeded": 0,
            "postgres_failed": 0,
            "failures": []
        }
        
        # Store in DynamoDB first (batch)
        dynamo_results = self.dynamo_storage.batch_store_energy_data(items)
        results["dynamo_succeeded"] = dynamo_results.get("succeeded", 0)
        results["dynamo_failed"] = dynamo_results.get("failed", 0)
        
        # Store in PostgreSQL if embeddings are provided
        if embeddings is not None and len(embeddings) == len(items):
            try:
                conn = self._get_pg_connection()
                with conn.cursor() as cur:
                    # Prepare data for batch insertion
                    rows = []
                    for idx, item in enumerate(items):
                        item_id = item.get("dataId")
                        if not item_id:
                            results["postgres_failed"] += 1
                            results["failures"].append({
                                "error": "Missing dataId",
                                "index": idx
                            })
                            continue
                        
                        # Extract data
                        description = item.get("description", "Unknown")
                        efficiency = 0.0
                        efficiency_data = item.get("efficiency", {})
                        if efficiency_data and isinstance(efficiency_data, dict):
                            efficiency_value = efficiency_data.get("value")
                            if efficiency_value:
                                try:
                                    efficiency = float(efficiency_value)
                                except (ValueError, TypeError):
                                    pass
                        elif isinstance(item.get("efficiency"), (int, float)):
                            try:
                                efficiency = float(item["efficiency"])
                            except (ValueError, TypeError):
                                pass
                        
                        # Prepare metadata
                        metadata = {
                            "source_type": item.get("source_type"),
                            "data_url": item.get("dataUrl"),
                            "unit": efficiency_data.get("unit", "kWh") if isinstance(efficiency_data, dict) else "kWh"
                        }
                        
                        # Add to batch
                        rows.append((
                            item_id,
                            description,
                            efficiency,
                            embeddings[idx].tolist(),
                            Json(metadata),
                            "NOW()"  # Updated timestamp
                        ))
                    
                    # Execute batch upsert using execute_values with correct column mapping
                    if rows:
                        # Issue: execute_values can't handle NOW() function inside the rows
                        # Solution: Use template to replace the timestamp placeholder with NOW()
                        query = sql.SQL("""
                        INSERT INTO {} (item_id, description, efficiency, embedding, metadata, updated_at)
                        VALUES %s
                        ON CONFLICT (item_id) 
                        DO UPDATE SET 
                            description = EXCLUDED.description,
                            efficiency = EXCLUDED.efficiency,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            updated_at = NOW()
                        """).format(sql.Identifier(self.postgres_table_name))
                        
                        # Fix: Use a template with NOW() as the timestamp
                        template = "(%(item_id)s, %(description)s, %(efficiency)s, %(embedding)s, %(metadata)s, NOW())"
                        
                        # Convert list of tuples to list of dicts for template-based insertion
                        dict_rows = []
                        for row in rows:
                            dict_rows.append({
                                "item_id": row[0],
                                "description": row[1],
                                "efficiency": row[2],
                                "embedding": row[3],
                                "metadata": row[4],
                            })
                        
                        execute_values(cur, query, dict_rows, template=template)
                        conn.commit()
                        
                        results["postgres_succeeded"] = len(rows)
                        results["postgres_failed"] = len(items) - len(rows)
                        
                        logger.info(f"Successfully batch stored {len(rows)} energy data entries in PostgreSQL with embeddings")
                    
            except Exception as e:
                if conn:
                    conn.rollback()
                results["postgres_succeeded"] = 0
                results["postgres_failed"] = len(items)
                results["failures"].append({"error": str(e)})
                logger.error(f"Failed to batch store energy data in PostgreSQL: {e}")
        else:
            results["postgres_failed"] = len(items)
            results["failures"].append({
                "error": "Missing or mismatched embeddings array"
            })
        
        # Determine overall success
        results["overall_success"] = (
            results["dynamo_succeeded"] > 0 and
            (results["postgres_succeeded"] > 0 if embeddings is not None else True)
        )
        
        return results

# Create a default instance for convenience WITHOUT auto-initialization
default_dual_storage = DualStorage(lazy_init=True)