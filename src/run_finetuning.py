#!/usr/bin/env python3
import os
import sys
import torch
from fine_tuning.config import FineTuningConfig
from fine_tuning.trainer import EVQATrainer

def print_system_info():
    print("\n🖥️  System Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CPU: {os.cpu_count()} cores available")
    if torch.cuda.is_available():
        print("⚠️  GPU is available but will not be used in this configuration")

def main():
    try:
        print_system_info()
        
        # Initialize with CPU-optimized config
        config = FineTuningConfig()
        
        print("\n⚙️  Configuration:")
        print(f"Model: {config.model_name}")
        print(f"Batch size: {config.batch_size}")
        print(f"Seq length: {config.max_seq_length}")
        print(f"Training samples: {len(open(config.train_data_path).readlines())}")
        
        trainer = EVQATrainer(config)
        trainer.train()
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()