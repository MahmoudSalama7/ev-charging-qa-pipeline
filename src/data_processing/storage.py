# src/data_processing/storage.py
import pandas as pd
import json
import logging
from datetime import datetime
from pathlib import Path
from src.config.paths import PROCESSED_DIR

logger = logging.getLogger(__name__)

class DataStorage:
    def __init__(self, output_dir: Path = PROCESSED_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, data, name: str, format: str = "parquet") -> Path:
        """Generic save method"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.{format}"
        path = self.output_dir / filename
        
        try:
            if format == "parquet":
                if isinstance(data, pd.DataFrame):
                    data.to_parquet(path)
                else:
                    pd.DataFrame(data).to_parquet(path)
            elif format == "json":
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            logger.info(f"Saved {path.name} ({len(data) if hasattr(data, '__len__') else '?'} records)")
            return path
            
        except Exception as e:
            logger.error(f"Failed to save {path}: {str(e)}")
            raise

    def load_latest(self, prefix: str) -> pd.DataFrame:
        """Load most recent processed file"""
        files = sorted(self.output_dir.glob(f"{prefix}*.parquet"))
        if not files:
            raise FileNotFoundError(f"No files found with prefix {prefix}")
        return pd.read_parquet(files[-1])