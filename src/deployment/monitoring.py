#monitoring.py
import psutil
import requests
from datetime import datetime
from fastapi import HTTPException
import logging

class Monitor:
    def __init__(self, service_url: str = "http://localhost:8000"):
        self.service_url = service_url
        self.start_time = datetime.now()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Monitoring initialized")

    def get_system_metrics(self) -> dict:
        """Get current system resource usage"""
        try:
            return {
                "cpu": psutil.cpu_percent(),
                "memory": psutil.virtual_memory().percent,
                "uptime": str(datetime.now() - self.start_time),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Metrics error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def check_service_health(self) -> dict:
        """Check if API service is responsive"""
        try:
            response = requests.get(f"{self.service_url}/health", timeout=5)
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time": response.elapsed.total_seconds()
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {"status": "down", "error": str(e)}