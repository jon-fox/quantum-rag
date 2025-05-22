"""
Storage module for managing vector embeddings in PostgreSQL with pgvector.
"""
import os
import logging
import psycopg2
import numpy as np
from psycopg2 import sql
from psycopg2.extras import Json, execute_values # type: ignore
from typing import List, Tuple, Optional, Dict, Any

try:
    import boto3
except ImportError:
    boto3 = None # type: ignore
    logging.getLogger(__name__).warning("boto3 not found, AWS SSM Parameter Store functionality will be disabled.")

from dotenv import load_dotenv # Import dotenv
load_dotenv() # Load .env file from root or parent directories

logger = logging.getLogger(__name__)

class PgVectorStorage:
    """
    Handles storage and retrieval of vector embeddings in a PostgreSQL database
    using the pgvector extension.
    """
    def __init__(
        self,
        db_params: Optional[Dict[str, str]] = None,
        table_name: str = "document_embeddings",
        vector_dim: int = 1536, # Default for OpenAI text-embedding-3-small
        lazy_init: bool = True,
        app_environment: Optional[str] = None # Added for SSM path
    ):
        """
        Initialize PgVectorStorage.

        Args:
            db_params: PostgreSQL connection parameters (host, port, dbname, user, password).
                       If None, attempts to load from AWS SSM Parameter Store, then environment variables.
            table_name: Name of the table to store embeddings.
            vector_dim: Dimension of the embeddings.
            lazy_init: If True, schema initialization is deferred until the first operation.
            app_environment: The application environment (e.g., 'dev', 'prod') for SSM path construction.
                             Defaults to APP_ENVIRONMENT env var or 'dev'.
        """
        self.table_name = table_name
        self.vector_dim = vector_dim
        self.pg_conn: Optional[psycopg2.extensions.connection] = None
        self.pg_params: Optional[Dict[str, str]] = db_params
        self.schema_initialized = False
        self.app_environment = app_environment or os.environ.get("APP_ENVIRONMENT", "prod")

        if not self.pg_params:
            logger.info("db_params not provided, attempting to load from AWS SSM Parameter Store...")
            if boto3 and self._load_db_params_from_ssm():
                logger.info("Successfully loaded DB params from AWS SSM Parameter Store.")
            else:
                logger.info("Failed to load DB params from SSM or boto3 is not available. Falling back to environment variables.")
                self._load_db_params_from_env()
        
        if not self.pg_params:
            logger.warning("DB connection parameters not loaded. Connection attempts will likely fail.")

        if not lazy_init:
            self._init_schema()

    def _load_db_params_from_ssm(self) -> bool:
        """
        Loads PostgreSQL connection parameters from AWS SSM Parameter Store.
        Relies on self.app_environment to construct parameter paths.
        Returns True if parameters are successfully loaded, False otherwise.
        """
        if not boto3:
            logger.warning("boto3 is not installed or importable. Cannot load DB params from SSM.")
            return False
        if not self.app_environment:
            logger.warning("APP_ENVIRONMENT is not set. Cannot load DB params from SSM.")
            return False

        try:
            ssm_client = boto3.client('ssm', region_name=os.environ.get("AWS_REGION", "us-east-1"))
            base_path = f"/{self.app_environment}/quantum-rag/db"

            param_names_map = {
                "host": f"{base_path}/address",
                "port": f"{base_path}/port",
                "dbname": f"{base_path}/name",
                "user": f"{base_path}/username",
                "password": f"{base_path}/password" # SecureString
            }

            loaded_params: Dict[str, str] = {}
            for key, param_name in param_names_map.items():
                try:
                    response = ssm_client.get_parameter(
                        Name=param_name,
                        WithDecryption=(key == "password") # Only decrypt password
                    )
                    value = response.get('Parameter', {}).get('Value')
                    if value is None:
                        logger.error(f"SSM parameter \'{param_name}\' is missing Value field in response.")
                        return False
                    loaded_params[key] = value
                except ssm_client.exceptions.ParameterNotFound:
                    logger.error(f"SSM parameter \'{param_name}\' not found.")
                    return False
                except Exception as e:
                    logger.error(f"Failed to fetch SSM parameter \'{param_name}\': {e}")
                    return False
            
            # Ensure all required parameters were loaded
            required_keys = ["host", "port", "dbname", "user", "password"]
            if not all(key in loaded_params for key in required_keys):
                logger.error(f"One or more required DB parameters missing after SSM load attempt. Loaded: {list(loaded_params.keys())}")
                return False

            self.pg_params = loaded_params
            logger.info(f"Loaded PostgreSQL params from SSM: host={self.pg_params.get('host')}, dbname={self.pg_params.get('dbname')}")
            return True

        except Exception as e:
            logger.error(f"Error initializing SSM client or loading parameters: {e}")
            return False

    def _load_db_params_from_env(self):
        """Loads PostgreSQL connection parameters from environment variables."""
        self.pg_params = {
            "host": os.environ.get("DB_HOST", "localhost"),
            "port": os.environ.get("DB_PORT", "5432"),
            "dbname": os.environ.get("DB_NAME", "energy_data"),
            "user": os.environ.get("DB_USER", "energyadmin"),
            "password": os.environ.get("DB_PASSWORD", "")
        }
        logger.info(f"Loaded PostgreSQL params from environment: host={self.pg_params.get('host')}, dbname={self.pg_params.get('dbname')}")

    def close_db_connection(self) -> None:
        """
        Closes the PostgreSQL connection if it is open.
        """
        if self.pg_conn and not self.pg_conn.closed:
            try:
                self.pg_conn.close()
                logger.info("PostgreSQL connection closed successfully.")
            except psycopg2.Error as e:
                logger.error(f"Error closing PostgreSQL connection: {e}")
            finally:
                self.pg_conn = None
        else:
            logger.info("PostgreSQL connection was not open or already closed.")

    def _get_connection(self, retry: bool = True) -> Optional[psycopg2.extensions.connection]:
        """
        Establishes or returns an existing PostgreSQL connection.

        Args:
            retry: Whether to retry connection on failure.

        Returns:
            A psycopg2 connection object or None if connection fails.
        """
        if self.pg_conn is None or self.pg_conn.closed:
            if not self.pg_params:
                logger.error("PostgreSQL connection parameters are not set.")
                return None
            
            # Ensure all parameters are strings, especially port if it was loaded as int elsewhere
            connect_params = {k: str(v) for k, v in self.pg_params.items()}

            try:
                self.pg_conn = psycopg2.connect(**connect_params) # type: ignore
                # self.pg_conn.autocommit = False # Use autocommit False for explicit transaction control
                logger.info(f"Successfully connected to PostgreSQL database '{connect_params.get('dbname')}' on {connect_params.get('host')}.")
            except psycopg2.Error as e:
                logger.error(f"Failed to connect to PostgreSQL: {e}")
                self.pg_conn = None
                if not retry:
                    raise
        return self.pg_conn

    def _init_schema(self):
        """
        Initializes the database schema, creating the pgvector extension and the embeddings table.
        """
        if self.schema_initialized:
            return True

        conn = self._get_connection(retry=False)
        if not conn:
            logger.warning("Skipping schema initialization as PostgreSQL connection is unavailable.")
            return False

        try:
            with conn.cursor() as cur:
                logger.info("Initializing pgvector schema...")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                table_creation_query = sql.SQL("""
                CREATE TABLE IF NOT EXISTS {table} (
                    vector_id UUID PRIMARY KEY,
                    embedding VECTOR({vector_dim}),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """).format(
                    table=sql.Identifier(self.table_name),
                    vector_dim=sql.Literal(self.vector_dim)
                )
                cur.execute(table_creation_query)

                index_name = f"idx_{self.table_name}_embedding_hnsw"
                cur.execute(f"SELECT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relname = '{index_name}' AND n.nspname = 'public');") # Assumes public schema
                row = cur.fetchone()
                index_exists = row[0] if row else False

                if not index_exists:
                    # Using HNSW index as an example. For cosine distance, use vector_cosine_ops.
                    # For L2 distance, use vector_l2_ops.
                    # m and ef_construction are HNSW parameters.
                    index_creation_query = sql.SQL("""
                    CREATE INDEX {index_name} ON {table} USING HNSW (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                    """).format(
                        index_name=sql.Identifier(index_name),
                        table=sql.Identifier(self.table_name)
                    )
                    # cur.execute(index_creation_query) # Uncomment to create HNSW index
                    # logger.info(f"Created HNSW index '{index_name}' on {self.table_name}.embedding")

                    ivfflat_index_name = f"idx_{self.table_name}_embedding_ivfflat"
                    cur.execute(f"SELECT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relname = '{ivfflat_index_name}' AND n.nspname = 'public');")
                    row = cur.fetchone()
                    ivfflat_index_exists = row[0] if row else False
                    if not ivfflat_index_exists:
                        ivfflat_index_creation_query = sql.SQL("""
                        CREATE INDEX {index_name} ON {table} USING IVFFLAT (embedding vector_cosine_ops)
                        WITH (lists = 100);
                        """).format(
                            index_name=sql.Identifier(ivfflat_index_name),
                            table=sql.Identifier(self.table_name)
                        )
                        cur.execute(ivfflat_index_creation_query)
                        logger.info(f"Created IVFFlat index '{ivfflat_index_name}' on {self.table_name}.embedding")


            conn.commit()
            self.schema_initialized = True
            logger.info(f"pgvector schema initialized successfully for table '{self.table_name}'.")
            return True
        except psycopg2.Error as e:
            logger.error(f"Error initializing pgvector schema: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn and not conn.closed:
                conn.close()
    
    def store_embedding(self, vector_id: str, embedding: np.ndarray) -> bool:
        """
        Stores a single embedding vector.

        Args:
            vector_id: The UUID string for the vector.
            embedding: The numpy array representing the embedding.

        Returns:
            True if storage was successful, False otherwise.
        """
        if not self.schema_initialized:
            self._init_schema()
            if not self.schema_initialized: # Check again after attempting init
                logger.error("Cannot store embedding, schema not initialized and initialization failed.")
                return False

        conn = self._get_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cur:
                # pgvector expects a list or string representation of the vector
                embedding_list = embedding.tolist()
                
                upsert_query = sql.SQL("""
                INSERT INTO {table} (vector_id, embedding, updated_at)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (vector_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    updated_at = CURRENT_TIMESTAMP;
                """).format(table=sql.Identifier(self.table_name))
                
                cur.execute(upsert_query, (vector_id, embedding_list))
            conn.commit()
            logger.info(f"Successfully stored/updated embedding for vector_id: {vector_id}")
            return True
        except psycopg2.Error as e:
            logger.error(f"Error storing embedding for vector_id {vector_id}: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn and not conn.closed:
                conn.close()

    def batch_store_embeddings(self, embeddings_data: List[Tuple[str, np.ndarray]]) -> bool:
        """
        Stores multiple embeddings in a batch.

        Args:
            embeddings_data: A list of tuples, where each tuple is (vector_id, embedding_array).

        Returns:
            True if batch storage was successful, False otherwise.
        """
        if not embeddings_data:
            return True 
        
        if not self.schema_initialized:
            self._init_schema()
            if not self.schema_initialized:
                logger.error("Cannot batch store embeddings, schema not initialized and initialization failed.")
                return False

        conn = self._get_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cur:
                # Prepare data for execute_values
                # (vector_id, embedding_list_as_string, created_at, updated_at)
                # pgvector can take string representation '[1,2,3]'
                data_to_insert = [
                    (vector_id, np.array(embedding).tolist()) for vector_id, embedding in embeddings_data
                ]

                upsert_query = sql.SQL("""
                INSERT INTO {table} (vector_id, embedding, updated_at)
                VALUES %s
                ON CONFLICT (vector_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    updated_at = CURRENT_TIMESTAMP;
                """).format(table=sql.Identifier(self.table_name))

                # page_size is important for performance with large batches
                execute_values(cur, upsert_query, data_to_insert, template=None, page_size=100)
            conn.commit()
            logger.info(f"Successfully batch stored/updated {len(embeddings_data)} embeddings.")
            return True
        except psycopg2.Error as e:
            logger.error(f"Error batch storing embeddings: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn and not conn.closed:
                conn.close()

    def get_embedding(self, vector_id: str) -> Optional[np.ndarray]:
        """
        Retrieves a single embedding by its vector_id.

        Args:
            vector_id: The UUID of the vector.

        Returns:
            A numpy array of the embedding, or None if not found or error.
        """
        conn = self._get_connection()
        if not conn:
            return None
        
        try:
            with conn.cursor() as cur:
                select_query = sql.SQL("SELECT embedding FROM {table} WHERE vector_id = %s;").format(
                    table=sql.Identifier(self.table_name)
                )
                cur.execute(select_query, (vector_id,))
                result = cur.fetchone()
                if result:
                    # Embedding is returned as a list from pgvector
                    return np.array(result[0])
                else:
                    logger.info(f"Embedding not found for vector_id: {vector_id}")
                    return None
        except psycopg2.Error as e:
            logger.error(f"Error retrieving embedding for vector_id {vector_id}: {e}")
            return None
        finally:
            if conn and not conn.closed:
                conn.close()

    def delete_embedding(self, vector_id: str) -> bool:
        """
        Deletes an embedding by its vector_id.

        Args:
            vector_id: The UUID of the vector to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """
        conn = self._get_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cur:
                delete_query = sql.SQL("DELETE FROM {table} WHERE vector_id = %s;").format(
                    table=sql.Identifier(self.table_name)
                )
                cur.execute(delete_query, (vector_id,))
                deleted_count = cur.rowcount
            conn.commit()
            if deleted_count > 0:
                logger.info(f"Successfully deleted embedding for vector_id: {vector_id}")
                return True
            else:
                logger.info(f"No embedding found to delete for vector_id: {vector_id}")
                return False
        except psycopg2.Error as e:
            logger.error(f"Error deleting embedding for vector_id {vector_id}: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn and not conn.closed:
                conn.close()

    def find_similar_embeddings(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        metric: str = "cosine" 
    ) -> List[Tuple[str, float]]:
        """
        Finds embeddings similar to the query_embedding.

        Args:
            query_embedding: The numpy array of the query embedding.
            top_k: The number of similar embeddings to return.
            metric: The distance metric to use ('cosine', 'l2', 'inner_product').
                    'cosine' uses cosine distance (1 - cosine_similarity).
                    'l2' uses Euclidean distance.
                    'inner_product' uses inner product (negative for similarity with normalized vectors).

        Returns:
            A list of tuples: (vector_id, distance_score).
            Lower scores mean more similar for 'cosine' and 'l2'.
            Higher scores mean more similar for 'inner_product' if vectors are normalized.
        """
        if not self.schema_initialized:
            logger.warning("Schema not initialized. Attempting to initialize now.")
            self._init_schema()
            if not self.schema_initialized:
                 logger.error("Cannot find similar embeddings, schema not initialized and initialization failed.")
                 return []

        conn = self._get_connection()
        if not conn:
            return []

        # pgvector operators:
        # '<->' for L2 distance
        # '<#>' for inner product (negative of inner product, so for similarity with normalized vectors, you'd want to order by this ASC)
        # '<=>' for cosine distance (1 - cosine_similarity)
        operator_map = {
            "l2": "<->",
            "inner_product": "<#>", # For normalized vectors, lower <#> is higher inner product (more similar)
            "cosine": "<=>"
        }
        
        if metric not in operator_map:
            logger.error(f"Unsupported similarity metric: {metric}. Supported: {list(operator_map.keys())}")
            return []

        distance_operator = operator_map[metric]
        query_embedding_list = query_embedding.tolist()

        try:
            with conn.cursor() as cur:
                # Note: The distance operator returns the distance.
                # For cosine similarity, distance = 1 - similarity. So smaller distance is better.
                # For inner product, it's negative inner product. Smaller (more negative) is better for similarity.
                # For L2, smaller distance is better.
                # So, always ORDER BY distance ASC.
                
                # Example: For IVFFlat, you might want to set probes:
                # cur.execute("SET LOCAL ivfflat.probes = 10;") # Adjust as needed

                similarity_query = sql.SQL("""
                SELECT vector_id, embedding {operator} %s AS distance
                FROM {table}
                ORDER BY distance ASC
                LIMIT %s;
                """).format(
                    operator=sql.SQL(distance_operator), # Safely inject operator
                    table=sql.Identifier(self.table_name)
                )
                
                cur.execute(similarity_query, (query_embedding_list, top_k))
                results = cur.fetchall()
                
                # Convert to (vector_id, score)
                # If using cosine distance, score = 1 - distance to get similarity
                # If using L2, score is the distance
                # If using inner product, score = -distance to get actual inner product
                processed_results = []
                for row in results:
                    vector_id, distance = row
                    score = distance
                    if metric == "cosine":
                        # score = 1 - distance # This would be cosine similarity
                        pass # Keep as cosine distance for now, user can convert if needed
                    elif metric == "inner_product":
                        # score = -distance # This would be the actual inner product
                        pass # Keep as negative inner product
                    processed_results.append((vector_id, float(score)))

                return processed_results
        except psycopg2.Error as e:
            logger.error(f"Error finding similar embeddings: {e}")
            return []
        finally:
            if conn and not conn.closed:
                conn.close()

    def close_connection(self):
        """Closes the PostgreSQL connection if it's open."""
        if self.pg_conn and not self.pg_conn.closed:
            self.pg_conn.close()
            logger.info("PostgreSQL connection closed.")
