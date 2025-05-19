"""
ERCOT API queries for common data endpoints
"""
import logging
import datetime
from typing import Dict, Any, Optional, List
from .client import ERCOTClient

# Configure logging
logger = logging.getLogger(__name__)

class ERCOTQueries:
    """
    Provides structured access to common ERCOT API endpoints.
    """
    
    # Base path for the public reports API
    PUBLIC_REPORTS_BASE = "api/public-reports"
    
    def __init__(self, client: Optional[ERCOTClient] = None):
        """
        Initialize the ERCOT Queries helper.
        
        Args:
            client (ERCOTClient, optional): An existing ERCOT client instance.
                                           If not provided, a new one will be created.
        """
        self.client = client or ERCOTClient()
    
    def get_aggregated_generation_summary(
        self, 
        delivery_date_from: Optional[str] = None, 
        delivery_date_to: Optional[str] = None,
        region: Optional[str] = None,
        page: int = 1,
        size: int = 100
    ) -> Dict[str, Any]:
        """
        Get 2-Day Aggregated Generation Summary data.
        
        Args:
            delivery_date_from (str, optional): Start date in format YYYY-MM-DD.
                                              If not provided, defaults to yesterday.
            delivery_date_to (str, optional): End date in format YYYY-MM-DD.
                                            If not provided, defaults to today.
            region (str, optional): Region filter (Houston, North, South, West).
                                   If not provided, gets the overall summary.
            page (int): Page number for pagination.
            size (int): Number of records per page.
            
        Returns:
            dict: The API response containing generation summary data.
        """
        # Set default dates if not provided
        if not delivery_date_from:
            yesterday = datetime.date.today() - datetime.timedelta(days=1)
            delivery_date_from = yesterday.strftime('%Y-%m-%d')
            
        if not delivery_date_to:
            today = datetime.date.today()
            delivery_date_to = today.strftime('%Y-%m-%d')
            
        # Determine the endpoint based on region
        endpoint_suffix = "2d_agg_gen_sum"
        if region:
            region_lower = region.lower()
            if region_lower in ["houston", "north", "south", "west"]:
                endpoint_suffix = f"2d_agg_gen_sum_{region_lower}"
        
        endpoint = f"{self.PUBLIC_REPORTS_BASE}/np3-911-er/{endpoint_suffix}"
        
        # Build query parameters
        params = {
            "deliveryDateFrom": delivery_date_from,
            "deliveryDateTo": delivery_date_to,
            "page": page,
            "size": size
        }
        
        logger.info(f"Fetching generation summary from {delivery_date_from} to {delivery_date_to}")
        return self.client.get_data(endpoint, params)
    
    def get_aggregated_load_summary(
        self, 
        delivery_date_from: Optional[str] = None, 
        delivery_date_to: Optional[str] = None,
        region: Optional[str] = None,
        page: int = 1,
        size: int = 100
    ) -> Dict[str, Any]:
        """
        Get 2-Day Aggregated Load Summary data.
        
        Args:
            delivery_date_from (str, optional): Start date in format YYYY-MM-DD.
                                              If not provided, defaults to yesterday.
            delivery_date_to (str, optional): End date in format YYYY-MM-DD.
                                            If not provided, defaults to today.
            region (str, optional): Region filter (Houston, North, South, West).
                                   If not provided, gets the overall summary.
            page (int): Page number for pagination.
            size (int): Number of records per page.
            
        Returns:
            dict: The API response containing load summary data.
        """
        # Set default dates if not provided
        if not delivery_date_from:
            yesterday = datetime.date.today() - datetime.timedelta(days=1)
            delivery_date_from = yesterday.strftime('%Y-%m-%d')
            
        if not delivery_date_to:
            today = datetime.date.today()
            delivery_date_to = today.strftime('%Y-%m-%d')
            
        # Determine the endpoint based on region
        endpoint_suffix = "2d_agg_load_sum"
        if region:
            region_lower = region.lower()
            if region_lower in ["houston", "north", "south", "west"]:
                endpoint_suffix = f"2d_agg_load_sum_{region_lower}"
        
        endpoint = f"{self.PUBLIC_REPORTS_BASE}/np3-911-er/{endpoint_suffix}"
        
        # Build query parameters
        params = {
            "deliveryDateFrom": delivery_date_from,
            "deliveryDateTo": delivery_date_to,
            "page": page,
            "size": size
        }
        
        logger.info(f"Fetching load summary from {delivery_date_from} to {delivery_date_to}")
        return self.client.get_data(endpoint, params)
    
    def get_ancillary_service_offers(
        self,
        service_type: str,
        delivery_date_from: Optional[str] = None,
        delivery_date_to: Optional[str] = None,
        hour_ending_from: Optional[int] = None,
        hour_ending_to: Optional[int] = None,
        page: int = 1,
        size: int = 100
    ) -> Dict[str, Any]:
        """
        Get 2-Day Aggregated Ancillary Service Offers.
        
        Args:
            service_type (str): Type of ancillary service. 
                               Options: ECRSM, ECRSS, OFFNS, ONNS, REGDN, REGUP, RRSFFR, RRSPFR, RRSUFR
            delivery_date_from (str, optional): Start date in format YYYY-MM-DD.
            delivery_date_to (str, optional): End date in format YYYY-MM-DD.
            hour_ending_from (int, optional): Starting hour ending.
            hour_ending_to (int, optional): Ending hour ending.
            page (int): Page number for pagination.
            size (int): Number of records per page.
            
        Returns:
            dict: The API response containing ancillary service offers data.
        """
        # Validate service type
        valid_types = ["ecrsm", "ecrss", "offns", "onns", "regdn", "regup", "rrsffr", "rrspfr", "rrsufr"]
        service_type_lower = service_type.lower()
        
        if service_type_lower not in valid_types:
            raise ValueError(f"Invalid service type. Must be one of: {', '.join(valid_types)}")
            
        # Set default dates if not provided
        if not delivery_date_from:
            yesterday = datetime.date.today() - datetime.timedelta(days=1)
            delivery_date_from = yesterday.strftime('%Y-%m-%d')
            
        if not delivery_date_to:
            today = datetime.date.today()
            delivery_date_to = today.strftime('%Y-%m-%d')
        
        # Build the endpoint
        endpoint = f"{self.PUBLIC_REPORTS_BASE}/np3-911-er/2d_agg_as_offers_{service_type_lower}"
        
        # Build query parameters
        params = {
            "deliveryDateFrom": delivery_date_from,
            "deliveryDateTo": delivery_date_to,
            "page": page,
            "size": size
        }
        
        # Add optional hour ending parameters if provided
        if hour_ending_from is not None:
            params["hourEndingFrom"] = hour_ending_from
            
        if hour_ending_to is not None:
            params["hourEndingTo"] = hour_ending_to
        
        logger.info(f"Fetching {service_type} offers from {delivery_date_from} to {delivery_date_to}")
        return self.client.get_data(endpoint, params)
