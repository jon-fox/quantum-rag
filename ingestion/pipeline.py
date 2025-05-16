"""
Data Ingestion Pipeline

This module coordinates the data ingestion process from ERCOT sources
to the vector store for use in the RAG application.
"""
import os
import sys
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import ERCOT components
from ingestion.ercot.fetch_api import ERCOTFetcher
from ingestion.ercot.parse_reports import ERCOTReportParser
from ingestion.ercot.schedule import ERCOTScheduler

class IngestionPipeline:
    """Main ingestion pipeline for RAG data sources"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ingestion pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Set up directories
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.vector_dir = os.path.join(self.data_dir, "vectors")
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.vector_dir, exist_ok=True)
        
        # Initialize components
        self.fetcher = ERCOTFetcher(self.config.get("fetch_config"))
        self.parser = ERCOTReportParser(self.config.get("parser_config"))
        
        # Setup embedding function (simulated for now)
        self.embedding_dim = self.config.get("embedding_dim", 384)
        
        # Try to import sentence transformers for embeddings
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(
                'all-mpnet-base-v2'  # Default model, can be overridden in config
            )
        except ImportError:
            logger.warning("SentenceTransformers not available. Using mock embeddings.")
            self.embedding_model = None
    
    async def run_pipeline(self):
        """
        Execute the full ingestion pipeline.
        
        1. Fetch data from ERCOT sources
        2. Parse and clean data
        3. Create embeddings
        4. Write to vector store
        """
        logger.info("Starting ingestion pipeline")
        
        # Step 1: Fetch data
        await self.fetch_data()
        
        # Step 2: Process raw data
        self.process_raw_data()
        
        # Step 3: Generate embeddings and save to vector store
        self.generate_embeddings()
        
        logger.info("Completed ingestion pipeline")
    
    async def fetch_data(self):
        """
        Fetch data from ERCOT sources.
        """
        logger.info("Fetching data from ERCOT")
        
        # Get market data for last 30 days
        today = datetime.now()
        start_date = today.replace(day=1)  # First day of current month
        
        try:
            # Fetch market data
            market_data = await self.fetcher.fetch_market_data(start_date)
            market_path = os.path.join(self.raw_dir, f"market_data_{today.strftime('%Y%m%d')}.json")
            with open(market_path, 'w') as f:
                json.dump(market_data, f)
            logger.info(f"Saved market data to {market_path}")
            
            # Fetch forecasts
            for forecast_type in ["short_term", "mid_term", "long_term"]:
                forecast_data = await self.fetcher.fetch_forecasts(forecast_type)
                forecast_path = os.path.join(
                    self.raw_dir, 
                    f"forecast_{forecast_type}_{today.strftime('%Y%m%d')}.json"
                )
                with open(forecast_path, 'w') as f:
                    json.dump(forecast_data, f)
                logger.info(f"Saved {forecast_type} forecast to {forecast_path}")
                
            # Fetch report list
            report_list = await self.fetcher.fetch_report_list()
            report_list_path = os.path.join(self.raw_dir, f"report_list_{today.strftime('%Y%m%d')}.json")
            with open(report_list_path, 'w') as f:
                json.dump(report_list, f)
            logger.info(f"Saved report list to {report_list_path}")
            
        except Exception as e:
            logger.error(f"Error fetching ERCOT data: {str(e)}")
            raise
    
    def process_raw_data(self):
        """
        Process raw data files into a clean format for embedding.
        """
        logger.info("Processing raw data files")
        
        processed_docs = []
        
        # Process all JSON files in raw directory
        for filename in os.listdir(self.raw_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.raw_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Process based on file type
                    if "market_data" in filename:
                        docs = self._process_market_data(data, filename)
                    elif "forecast" in filename:
                        docs = self._process_forecast_data(data, filename)
                    elif "report_list" in filename:
                        docs = self._process_report_list(data, filename)
                    else:
                        docs = self._process_generic_data(data, filename)
                        
                    processed_docs.extend(docs)
                    
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")
        
        # Save processed documents
        processed_path = os.path.join(self.processed_dir, f"processed_docs_{datetime.now().strftime('%Y%m%d')}.json")
        with open(processed_path, 'w') as f:
            json.dump(processed_docs, f)
            
        logger.info(f"Processed {len(processed_docs)} documents, saved to {processed_path}")
        return processed_docs
    
    def generate_embeddings(self):
        """
        Generate embeddings for processed documents and save to vector store.
        """
        logger.info("Generating document embeddings")
        
        # Find latest processed documents file
        processed_files = [f for f in os.listdir(self.processed_dir) if f.startswith('processed_docs_')]
        if not processed_files:
            logger.warning("No processed documents found")
            return
            
        latest_file = sorted(processed_files)[-1]
        processed_path = os.path.join(self.processed_dir, latest_file)
        
        try:
            with open(processed_path, 'r') as f:
                documents = json.load(f)
                
            logger.info(f"Generating embeddings for {len(documents)} documents")
            
            # Generate embeddings
            embeddings = []
            for doc in documents:
                text = doc.get('content', '')
                
                if self.embedding_model:
                    # Use sentence-transformers model if available
                    vector = self.embedding_model.encode(text).tolist()
                else:
                    # Mock embedding if model not available
                    vector = self._mock_embedding(text)
                    
                # Add embedding to document
                doc['embedding'] = vector
                embeddings.append(doc)
            
            # Save embeddings
            embeddings_path = os.path.join(
                self.vector_dir, 
                f"embeddings_{datetime.now().strftime('%Y%m%d')}.json"
            )
            with open(embeddings_path, 'w') as f:
                json.dump(embeddings, f)
                
            logger.info(f"Generated {len(embeddings)} embeddings, saved to {embeddings_path}")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
    
    def _process_market_data(self, data: List[Dict[str, Any]], source_file: str) -> List[Dict[str, Any]]:
        """
        Process market data into documents suitable for embedding.
        """
        documents = []
        
        for entry in data:
            # Create a document for each market data entry
            doc = {
                "id": f"market_{entry.get('date', 'unknown')}",
                "content": self._format_market_data_text(entry),
                "source": source_file,
                "type": "market_data",
                "metadata": {
                    "date": entry.get("date"),
                    "load": entry.get("load", {}),
                    "prices": entry.get("prices", {})
                },
                "timestamp": datetime.now().isoformat()
            }
            documents.append(doc)
            
        return documents
    
    def _process_forecast_data(self, data: Dict[str, Any], source_file: str) -> List[Dict[str, Any]]:
        """
        Process forecast data into documents suitable for embedding.
        """
        documents = []
        
        # Extract forecast type and intervals
        forecast_type = data.get("type", "unknown")
        intervals = data.get("intervals", [])
        
        for i, interval in enumerate(intervals):
            # Create a document for each forecast interval
            interval_id = interval.get("timestamp", interval.get("date", f"interval_{i}"))
            
            doc = {
                "id": f"forecast_{forecast_type}_{interval_id}",
                "content": self._format_forecast_text(forecast_type, interval),
                "source": source_file,
                "type": f"{forecast_type}_forecast",
                "metadata": interval,
                "timestamp": datetime.now().isoformat()
            }
            documents.append(doc)
            
        return documents
    
    def _process_report_list(self, data: List[Dict[str, Any]], source_file: str) -> List[Dict[str, Any]]:
        """
        Process report list into documents suitable for embedding.
        """
        documents = []
        
        for report in data:
            # Create a document for each report
            doc = {
                "id": f"report_info_{report.get('id', 'unknown')}",
                "content": self._format_report_info_text(report),
                "source": source_file,
                "type": "report_info",
                "metadata": report,
                "timestamp": datetime.now().isoformat()
            }
            documents.append(doc)
            
        return documents
    
    def _process_generic_data(self, data: Any, source_file: str) -> List[Dict[str, Any]]:
        """
        Process generic data into documents suitable for embedding.
        """
        # For generic data, create a single document
        doc = {
            "id": f"generic_{os.path.basename(source_file)}",
            "content": json.dumps(data, indent=2),
            "source": source_file,
            "type": "generic",
            "metadata": {
                "file": source_file
            },
            "timestamp": datetime.now().isoformat()
        }
        return [doc]
    
    def _format_market_data_text(self, entry: Dict[str, Any]) -> str:
        """Format market data as text for embedding"""
        date = entry.get("date", "unknown date")
        
        load = entry.get("load", {})
        load_forecast = load.get("forecast", "N/A")
        load_actual = load.get("actual", "N/A")
        
        prices = entry.get("prices", {})
        day_ahead = prices.get("day_ahead", "N/A")
        real_time = prices.get("real_time", "N/A")
        
        text = f"""ERCOT Market Data for {date}:
        
        Load:
        - Forecasted Load: {load_forecast} MW
        - Actual Load: {load_actual} MW
        
        Prices:
        - Day-Ahead Market: ${day_ahead}/MWh
        - Real-Time Market: ${real_time}/MWh
        """
        
        return text
    
    def _format_forecast_text(self, forecast_type: str, interval: Dict[str, Any]) -> str:
        """Format forecast data as text for embedding"""
        if forecast_type == "short_term":
            timestamp = interval.get("timestamp", "unknown time")
            load = interval.get("load_forecast", "N/A")
            renewable = interval.get("renewable_forecast", "N/A")
            price = interval.get("price_forecast", "N/A")
            
            text = f"""ERCOT Short-Term Forecast for {timestamp}:
            
            - Forecasted Load: {load} MW
            - Forecasted Renewable Generation: {renewable} MW
            - Forecasted Price: ${price}/MWh
            """
            
        elif forecast_type == "mid_term":
            date = interval.get("date", "unknown date")
            load = interval.get("peak_load_forecast", "N/A")
            peak_hour = interval.get("peak_hour", "N/A")
            price = interval.get("avg_price_forecast", "N/A")
            
            text = f"""ERCOT Mid-Term Forecast for {date}:
            
            - Peak Load Forecast: {load} MW
            - Expected Peak Hour: {peak_hour}:00
            - Average Price Forecast: ${price}/MWh
            """
            
        else:  # long_term
            month = interval.get("month", "unknown month")
            load = interval.get("peak_load_forecast", "N/A")
            reserve = interval.get("reserve_margin", "N/A")
            scenario = interval.get("scenario", "base")
            
            text = f"""ERCOT Long-Term Forecast for {month}:
            
            - Peak Load Forecast: {load} MW
            - Reserve Margin: {reserve}%
            - Scenario: {scenario}
            """
            
        return text
    
    def _format_report_info_text(self, report: Dict[str, Any]) -> str:
        """Format report info as text for embedding"""
        title = report.get("title", "Untitled Report")
        report_id = report.get("id", "unknown")
        published = report.get("published_date", "unknown")
        report_type = report.get("type", "unknown")
        url = report.get("url", "N/A")
        
        text = f"""ERCOT Report: {title}
        
        - Report ID: {report_id}
        - Published Date: {published}
        - Type: {report_type}
        - URL: {url}
        """
        
        return text
    
    def _mock_embedding(self, text: str) -> List[float]:
        """
        Create a mock embedding vector for a text string.
        In a real implementation, this would use an actual embedding model.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector
        """
        # Create a deterministic but simple vector based on the text
        hash_val = sum(ord(c) for c in text)
        np.random.seed(hash_val)
        vector = np.random.random(self.embedding_dim).tolist()
        return vector

# Command line interface
async def main():
    """Command line entry point"""
    pipeline = IngestionPipeline()
    await pipeline.run_pipeline()

if __name__ == "__main__":
    asyncio.run(main())