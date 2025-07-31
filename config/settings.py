import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Project Structure
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # Data Collection
    PDF_DIR = DATA_DIR / "pdfs"
    STATIONS_API_URL = "https://developer.nrel.gov/api/alt-fuel-stations/v1"
    
    # Model Training
    BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    LORA_RANK = 8
    MAX_SEQ_LENGTH = 256
    
    # API Settings
    API_KEYS = os.getenv("API_KEYS", "").split(",")
    API_RATE_LIMIT = 100  # requests/minute
    
    # Evaluation
    METRICS = ["rouge", "bleu", "exact_match"]
    
    @classmethod
    def setup_dirs(cls):
        """Ensure all directories exist"""
        dirs = [cls.PDF_DIR, cls.DATA_DIR, cls.MODELS_DIR]
        for d in dirs:
            d.mkdir(exist_ok=True, parents=True)

Config.setup_dirs()