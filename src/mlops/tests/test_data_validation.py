import pandas as pd
from pathlib import Path

def test_training_data_quality():
    train_path = Path("data/training/ev_qa_dataset.txt")
    assert train_path.exists()
    
    with open(train_path) as f:
        lines = f.readlines()
        assert len(lines) > 100  # Minimum expected samples
        for line in lines:
            assert "Instruction:" in line
            assert "Response:" in line