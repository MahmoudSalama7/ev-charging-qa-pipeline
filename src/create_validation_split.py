import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def detect_file_format(filepath):
    """Detect if file is JSONL, CSV, or other format"""
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        
        try:
            json.loads(first_line)
            return 'jsonl'
        except json.JSONDecodeError:
            if ',' in first_line and '"' in first_line:
                return 'csv'
            return 'txt'

def load_data(filepath):
    """Load data based on detected format"""
    fmt = detect_file_format(filepath)
    
    if fmt == 'jsonl':
        with open(filepath, 'r') as f:
            return [json.loads(line) for line in f]
    elif fmt == 'csv':
        return pd.read_csv(filepath).to_dict('records')
    else:  # Assume plain text with Q&A pairs
        data = []
        with open(filepath, 'r') as f:
            current_q = None
            for line in f:
                line = line.strip()
                if line.startswith('Q:') or line.startswith('Question:'):
                    current_q = line.split(':', 1)[1].strip()
                elif line.startswith('A:') or line.startswith('Answer:') and current_q:
                    data.append({
                        'question': current_q,
                        'answer': line.split(':', 1)[1].strip()
                    })
                    current_q = None
        return data

def create_validation_split():
    # Create directories
    Path("data/validation").mkdir(parents=True, exist_ok=True)
    input_path = "data/training/ev_qa_dataset.txt"
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load data
    data = load_data(input_path)
    if not data:
        raise ValueError("No valid data found in the input file")
    
    # Split data (80% train, 20% validation)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Save splits
    with open("data/training/ev_qa_dataset.txt", 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    
    with open("data/validation/ev_qa_eval.txt", 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"✅ Created validation split")
    print(f"Total samples: {len(data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

if __name__ == "__main__":
    create_validation_split()