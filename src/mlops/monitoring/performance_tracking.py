import mlflow
from datetime import datetime

class PerformanceTracker:
    def __init__(self):
        mlflow.set_tracking_uri("http://localhost:5000")
        
    def log_inference_metrics(self, latency: float, throughput: float):
        mlflow.log_metrics({
            'inference_latency': latency,
            'throughput': throughput,
            'timestamp': datetime.now().timestamp()
        })