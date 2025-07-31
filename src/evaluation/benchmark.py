#benchmark.py
import json
from pathlib import Path
from typing import List, Dict
import random

class EVChargingBenchmarkGenerator:
    def __init__(self, data_dir: str = "data/evaluation"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.station_types = ["Level 2", "DC Fast", "Tesla Supercharger"]
        self.locations = ["Los Angeles", "New York", "Berlin", "Tokyo"]
        self.payment_methods = ["Credit Card", "Mobile App", "RFID Card"]

    
        
    def generate_question(self, station_type: str, location: str) -> Dict:
        """Generate diverse EV charging questions"""
        question_types = [
            f"What type of charging is available at {location} station?",
            f"Does the {location} station have {station_type} charging?",
            f"What payment methods are accepted at {location} station?",
            f"What are the hours of operation for {location} station?",
            f"Which connectors are available at {location} {station_type} station?"
        ]
        return {
            "question": random.choice(question_types),
            "context": {
                "station_type": station_type,
                "location": location,
                "payment": random.choice(self.payment_methods)
            }
        }

    def generate_benchmark(self, num_samples: int = 100) -> Path:
        """Generate benchmark dataset"""
        benchmark_data = []
        for _ in range(num_samples):
            station = random.choice(self.station_types)
            location = random.choice(self.locations)
            benchmark_data.append(self.generate_question(station, location))
        
        output_path = self.data_dir / "ev_charging_benchmark.json"
        with open(output_path, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        print(f"✅ Generated benchmark with {num_samples} samples at {output_path}")
        return output_path

if __name__ == "__main__":
    generator = EVChargingBenchmarkGenerator()
    generator.generate_benchmark()