"""
ERCOT Data Fetching Scheduler

This module provides scheduling for regular data fetching from ERCOT APIs
and document processing.
"""
import os
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ERCOTScheduler:
    """Scheduler for ERCOT data fetching and processing jobs"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the scheduler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.data_dir = self.config.get("data_dir", "../data")
        self.jobs = []
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
    
    def add_job(self, job_func: Callable, interval_hours: float, job_id: str, job_args: Dict[str, Any] = None):
        """
        Add a scheduled job.
        
        Args:
            job_func: Function to call when job runs
            interval_hours: Hours between job runs
            job_id: Unique ID for the job
            job_args: Arguments to pass to job function
        """
        job_args = job_args or {}
        
        job = {
            "id": job_id,
            "func": job_func,
            "interval_seconds": interval_hours * 3600,
            "last_run": datetime.now() - timedelta(hours=interval_hours),  # Run immediately on start
            "args": job_args,
            "enabled": True
        }
        
        self.jobs.append(job)
        logger.info(f"Added job {job_id} with {interval_hours} hour interval")
    
    async def run(self, run_once: bool = False):
        """
        Run the scheduler.
        
        Args:
            run_once: If True, run each job once then exit
        """
        logger.info("Starting ERCOT data scheduler")
        
        while True:
            now = datetime.now()
            
            for job in self.jobs:
                if not job["enabled"]:
                    continue
                    
                elapsed = now - job["last_run"]
                if elapsed.total_seconds() >= job["interval_seconds"]:
                    # Time to run this job
                    logger.info(f"Running job {job['id']}")
                    try:
                        if asyncio.iscoroutinefunction(job["func"]):
                            await job["func"](**job["args"])
                        else:
                            job["func"](**job["args"])
                        job["last_run"] = now
                        job["last_status"] = "success"
                    except Exception as e:
                        logger.error(f"Error in job {job['id']}: {str(e)}")
                        job["last_status"] = f"error: {str(e)}"
            
            if run_once:
                logger.info("Completed one-time run of all jobs")
                break
            
            # Wait a bit before checking jobs again
            await asyncio.sleep(60)  # Check every minute
    
    def get_job_status(self) -> List[Dict[str, Any]]:
        """
        Get status of all scheduled jobs.
        
        Returns:
            List of job status dictionaries
        """
        return [
            {
                "id": job["id"],
                "last_run": job["last_run"].isoformat() if "last_run" in job else None,
                "last_status": job.get("last_status", "never_run"),
                "interval_hours": job["interval_seconds"] / 3600,
                "enabled": job["enabled"]
            }
            for job in self.jobs
        ]
    
    def disable_job(self, job_id: str) -> bool:
        """
        Disable a scheduled job.
        
        Args:
            job_id: ID of job to disable
            
        Returns:
            True if job was found and disabled, False otherwise
        """
        for job in self.jobs:
            if job["id"] == job_id:
                job["enabled"] = False
                logger.info(f"Disabled job {job_id}")
                return True
        return False
    
    def enable_job(self, job_id: str) -> bool:
        """
        Enable a scheduled job.
        
        Args:
            job_id: ID of job to enable
            
        Returns:
            True if job was found and enabled, False otherwise
        """
        for job in self.jobs:
            if job["id"] == job_id:
                job["enabled"] = True
                logger.info(f"Enabled job {job_id}")
                return True
        return False