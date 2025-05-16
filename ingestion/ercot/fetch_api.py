"""
ERCOT API Data Fetcher

This module provides functionality to fetch data from the ERCOT APIs
and store it for processing in the RAG pipeline.
"""
import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ERCOTFetcher:
    """Client for fetching data from ERCOT APIs and reports"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ERCOT fetcher with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.base_url = self.config.get("base_url", "https://www.ercot.com/api")
        self.api_key = self.config.get("api_key") or os.environ.get("ERCOT_API_KEY")
        self.timeout = self.config.get("timeout", 30)
        
        self.client = httpx.Client(timeout=self.timeout)
        
    async def fetch_market_data(self, start_date: datetime, end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Fetch market data from the ERCOT APIs.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval (defaults to current date)
            
        Returns:
            List of market data records
        """
        end_date = end_date or datetime.now()
        
        logger.info(f"Fetching market data from {start_date} to {end_date}")
        
        # In a real implementation, this would make actual API calls
        # For demonstration purposes, we're generating mock data
        
        # Simulate API call latency
        time.sleep(0.5)
        
        # Generate mock data
        data = []
        current_date = start_date
        while current_date <= end_date:
            # Create mock data entry for each day
            entry = {
                "date": current_date.strftime("%Y-%m-%d"),
                "load": {
                    "forecast": 45000 + (hash(current_date.strftime("%Y%m%d")) % 5000),
                    "actual": 44800 + (hash(current_date.strftime("%Y%m%d")) % 5500),
                },
                "prices": {
                    "day_ahead": 35.20 + (hash(current_date.strftime("%Y%m%d")) % 25),
                    "real_time": 37.50 + (hash(current_date.strftime("%Y%m%d")) % 30),
                }
            }
            data.append(entry)
            current_date += timedelta(days=1)
            
        logger.info(f"Retrieved {len(data)} market data records")
        return data
    
    async def fetch_forecasts(self, forecast_type: str = "short_term") -> Dict[str, Any]:
        """
        Fetch forecast data from ERCOT.
        
        Args:
            forecast_type: Type of forecast to retrieve (short_term, mid_term, long_term)
            
        Returns:
            Dictionary of forecast data
        """
        valid_types = ["short_term", "mid_term", "long_term"]
        if forecast_type not in valid_types:
            raise ValueError(f"Invalid forecast type. Must be one of: {', '.join(valid_types)}")
            
        logger.info(f"Fetching {forecast_type} forecast data")
        
        # Simulate API call latency
        time.sleep(0.7)
        
        # Generate mock forecast data
        now = datetime.now()
        
        if forecast_type == "short_term":
            # Hourly forecasts for next 24 hours
            hours = 24
            data = {
                "type": "short_term",
                "generated_at": now.isoformat(),
                "intervals": [
                    {
                        "timestamp": (now + timedelta(hours=h)).isoformat(),
                        "load_forecast": 42000 + (hash(f"{h}") % 8000),
                        "renewable_forecast": 12000 + (hash(f"{h}") % 5000),
                        "price_forecast": 38.50 + (hash(f"{h}") % 25),
                    }
                    for h in range(hours)
                ]
            }
        elif forecast_type == "mid_term":
            # Daily forecasts for next 7 days
            days = 7
            data = {
                "type": "mid_term",
                "generated_at": now.isoformat(),
                "intervals": [
                    {
                        "date": (now + timedelta(days=d)).strftime("%Y-%m-%d"),
                        "peak_load_forecast": 45000 + (hash(f"{d}") % 10000),
                        "peak_hour": 16 + (hash(f"{d}") % 4),
                        "avg_price_forecast": 40.25 + (hash(f"{d}") % 15),
                    }
                    for d in range(days)
                ]
            }
        else:  # long_term
            # Monthly forecasts
            months = 6
            data = {
                "type": "long_term",
                "generated_at": now.isoformat(),
                "intervals": [
                    {
                        "month": (now + timedelta(days=30*m)).strftime("%Y-%m"),
                        "peak_load_forecast": 50000 + (hash(f"{m}") % 15000),
                        "reserve_margin": 12.5 + (hash(f"{m}") % 5),
                        "scenario": "base" if m % 3 != 0 else "high_demand",
                    }
                    for m in range(months)
                ]
            }
            
        logger.info(f"Retrieved {forecast_type} forecast with {len(data['intervals'])} intervals")
        return data
        
    async def fetch_report_list(self, category: str = "system_planning") -> List[Dict[str, str]]:
        """
        Fetch a list of available reports from ERCOT.
        
        Args:
            category: Report category to filter by
            
        Returns:
            List of report metadata
        """
        logger.info(f"Fetching report list for category: {category}")
        
        # Simulate API call latency
        time.sleep(0.3)
        
        # Generate mock report list
        current_year = datetime.now().year
        
        reports = [
            {
                "id": f"report_{category}_{current_year-i:04d}_{j:02d}",
                "title": f"{'Monthly' if j%3==0 else 'Quarterly' if j%3==1 else 'Annual'} {category.replace('_', ' ').title()} Report",
                "published_date": f"{current_year-i:04d}-{(j%12)+1:02d}-01",
                "url": f"https://www.ercot.com/reports/{category}/{current_year-i:04d}/{j:02d}/report.pdf",
                "type": "pdf"
            }
            for i in range(3)  # 3 years
            for j in range(1, 5)  # 4 reports per year
        ]
        
        logger.info(f"Retrieved {len(reports)} reports")
        return reports
    
    def close(self):
        """Close HTTP client connection"""
        self.client.close()