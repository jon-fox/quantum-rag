import os
import logging
import psycopg2
import numpy as np
from psycopg2 import sql
from typing import List, Tuple, Optional, Dict, Any

try:
    import boto3
except ImportError:
    boto3 = None

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class PgVectorStorage:
    def __init__(
        self,
        db_params: Optional[Dict[str, str]] = None,
        table_name: str = "document_embeddings",
        app_environment: Optional[str] = None
    ):
        self.table_name = table_name
        self.pg_conn: Optional[psycopg2.extensions.connection] = None
        self.app_environment = app_environment or os.environ.get("APP_ENVIRONMENT", "prod")
        
        self.pg_params = db_params or self._load_db_params()
        
        if not self.pg_params:
            logger.error("Failed to load database connection parameters")

    def _load_db_params(self) -> Optional[Dict[str, str]]:
        if boto3 and self._try_load_from_ssm():
            return self.pg_params
        return self._load_from_env()

    def _try_load_from_ssm(self) -> bool:
        if not self.app_environment or not boto3:
            return False

        try:
            ssm_client = boto3.client('ssm', region_name=os.environ.get("AWS_REGION", "us-east-1"))
            base_path = f"/{self.app_environment}/quantum-rag/db"

            param_map = {
                "host": f"{base_path}/address",
                "port": f"{base_path}/port", 
                "dbname": f"{base_path}/name",
                "user": f"{base_path}/username",
                "password": f"{base_path}/password"
            }

            params = {}
            for key, param_name in param_map.items():
                response = ssm_client.get_parameter(
                    Name=param_name,
                    WithDecryption=(key == "password")
                )
                params[key] = response['Parameter']['Value']

            self.pg_params = params
            return True
        except Exception:
            return False

    def _load_from_env(self) -> Dict[str, str]:
        return {
            "host": os.environ.get("DB_HOST", "localhost"),
            "port": os.environ.get("DB_PORT", "5432"),
            "dbname": os.environ.get("DB_NAME", "energy_data"),
            "user": os.environ.get("DB_USER", "energyadmin"),
            "password": os.environ.get("DB_PASSWORD", "")
        }


    def close_db_connection(self) -> None:
        if self.pg_conn and not self.pg_conn.closed:
            try:
                self.pg_conn.close()
                logger.info("PostgreSQL connection closed.")
            except psycopg2.Error as e:
                logger.error(f"Error closing PostgreSQL connection: {e}")
            finally:
                self.pg_conn = None

    def _get_connection(self) -> Optional[psycopg2.extensions.connection]:
        if self.pg_conn is None or self.pg_conn.closed:
            if not self.pg_params:
                logger.error("PostgreSQL connection parameters are not set.")
                return None
            
            connect_params = {k: str(v) for k, v in self.pg_params.items()}

            try:
                self.pg_conn = psycopg2.connect(**connect_params)  # type: ignore
                logger.info(f"Connected to PostgreSQL database '{connect_params.get('dbname')}'.")
            except psycopg2.Error as e:
                logger.error(f"Failed to connect to PostgreSQL: {e}")
                self.pg_conn = None
        return self.pg_conn

    def execute_select_query(self, query: str) -> Tuple[Optional[List[str]], Optional[List[Tuple[Any, ...]]], Optional[str]]:
        if not query.strip().upper().startswith("SELECT"):
            return None, None, "Only SELECT queries are allowed."

        conn = self._get_connection()
        if not conn:
            return None, None, "Database connection not available."

        try:
            if conn.status == psycopg2.extensions.STATUS_IN_TRANSACTION and conn.get_transaction_status() == psycopg2.extensions.TRANSACTION_STATUS_INERROR:
                conn.rollback()

            with conn.cursor() as cur:
                cur.execute(query)
                
                if cur.description:
                    column_names = [desc[0] for desc in cur.description]
                    results = cur.fetchall()
                    return column_names, results, None
                else:
                    return [], [], None

        except psycopg2.Error as e:
            logger.error(f"Error executing query: {e}")
            if conn and not conn.closed:
                try:
                    conn.rollback()
                except psycopg2.Error:
                    pass
            return None, None, str(e)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None, None, str(e)

    def find_similar_embeddings(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        metric: str = "cosine"
    ) -> List[Tuple[str, float]]:
        conn = self._get_connection()
        if not conn:
            logger.error("Cannot find similar embeddings, no database connection.")
            return []

        try:
            with conn.cursor() as cur:
                query_embedding_list = query_embedding.tolist()

                if metric == "cosine":
                    query = sql.SQL("""
                        SELECT vector_id, embedding <=> CAST(%s AS vector) AS distance
                        FROM {table}
                        ORDER BY distance ASC
                        LIMIT %s;
                    """).format(table=sql.Identifier(self.table_name))
                elif metric == "l2":
                    query = sql.SQL("""
                        SELECT vector_id, embedding <-> CAST(%s AS vector) AS distance
                        FROM {table}
                        ORDER BY distance ASC
                        LIMIT %s;
                    """).format(table=sql.Identifier(self.table_name))
                elif metric == "inner_product":
                    query = sql.SQL("""
                        SELECT vector_id, (embedding <#> CAST(%s AS vector)) * -1 AS negative_inner_product
                        FROM {table}
                        ORDER BY negative_inner_product ASC
                        LIMIT %s;
                    """).format(table=sql.Identifier(self.table_name))
                else:
                    logger.error(f"Unsupported metric: {metric}")
                    return []

                cur.execute(query, (query_embedding_list, top_k))
                rows = cur.fetchall()
                
                if metric == "inner_product":
                    return [(str(row[0]), float(-row[1])) for row in rows]
                else:
                    return [(str(row[0]), float(row[1])) for row in rows]

        except Exception as e:
            logger.error(f"Error finding similar embeddings: {e}")
            if conn and not conn.closed:
                try:
                    conn.rollback()
                except psycopg2.Error:
                    pass
            return []

    def find_similar_documents(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        metric: str = "cosine"
    ) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        if not conn:
            logger.error("Cannot find similar documents, no database connection.")
            return []

        results: List[Dict[str, Any]] = []
        try:
            with conn.cursor() as cur:
                query_embedding_list = query_embedding.tolist()

                if metric == "cosine":
                    query = sql.SQL("""
                        SELECT vector_id, semantic_sentence, updated_at, embedding <=> CAST(%s AS vector) AS distance
                        FROM {table}
                        ORDER BY distance ASC
                        LIMIT %s;
                    """).format(table=sql.Identifier(self.table_name))
                elif metric == "l2":
                    query = sql.SQL("""
                        SELECT vector_id, semantic_sentence, updated_at, embedding <-> CAST(%s AS vector) AS distance
                        FROM {table}
                        ORDER BY distance ASC
                        LIMIT %s;
                    """).format(table=sql.Identifier(self.table_name))
                elif metric == "inner_product":
                    query = sql.SQL("""
                        SELECT vector_id, semantic_sentence, updated_at, (embedding <#> CAST(%s AS vector)) * -1 AS negative_inner_product
                        FROM {table}
                        ORDER BY negative_inner_product ASC
                        LIMIT %s;
                    """).format(table=sql.Identifier(self.table_name))
                else:
                    logger.error(f"Unsupported metric: {metric}")
                    return []

                cur.execute(query, (query_embedding_list, top_k))
                rows = cur.fetchall()
                
                for row in rows:
                    if metric == "cosine":
                        similarity_score = 1.0 - float(row[3])
                    elif metric == "inner_product":
                        similarity_score = float(-row[3])
                    else:  # L2 distance
                        similarity_score = float(row[3])
                    
                    results.append({
                        "vector_id": str(row[0]),
                        "document_id": str(row[0]),
                        "metadata": {
                            "content": str(row[1]) if row[1] else "",
                            "semantic_sentence": str(row[1]) if row[1] else "",
                            "updated_at": row[2] if row[2] else None,
                            "source": "PostgreSQL"
                        },
                        "similarity_score": similarity_score
                    })

        except psycopg2.Error as e:
            logger.error(f"Error finding similar documents: {e}")
            if conn and not conn.closed:
                try:
                    conn.rollback()
                except psycopg2.Error:
                    pass
            return []
        except Exception as e:
            logger.error(f"Unexpected error while finding similar documents: {e}")
            return []

        return results

    def get_embedding_by_id(self, vector_id: str) -> Optional[np.ndarray]:
        conn = self._get_connection()
        if not conn:
            logger.error("Cannot retrieve embedding, no database connection.")
            return None

        try:
            with conn.cursor() as cur:
                query = sql.SQL("""
                    SELECT embedding
                    FROM {table}
                    WHERE vector_id = %s;
                """).format(table=sql.Identifier(self.table_name))

                cur.execute(query, (vector_id,))
                row = cur.fetchone()

                if row is not None and len(row) > 0:
                    return np.array(row[0])
                else:
                    logger.warning(f"No embedding found for vector_id: {vector_id}")
                    return None
        except psycopg2.Error as e:
            logger.error(f"Error retrieving embedding for vector_id {vector_id}: {e}")
            if conn and not conn.closed:
                try:
                    conn.rollback()
                except psycopg2.Error:
                    pass
            return None

    def close_connection(self):
        if self.pg_conn and not self.pg_conn.closed:
            self.pg_conn.close()
            logger.info("PostgreSQL connection closed.")
