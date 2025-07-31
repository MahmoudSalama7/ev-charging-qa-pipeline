from mlflow import MlflowClient
from dataclasses import asdict
import time
import torch

class ExperimentTracker:
    """Basic tracker for CPU-only training"""
    def __init__(self):
        self.metrics = {}
    
    def start_run(self):
        print("🔬 Starting experiment tracking")
        
    def log_metric(self, name, value):
        self.metrics[name] = value
        
    def end_run(self, success=True):
        print(f"📊 Training metrics: {self.metrics}")