# src/config/paths.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
PROCESSED_DIR = DATA_DIR / "processed"

def get_pdf_path(filename: str) -> Path:
    """Get absolute path to PDF file"""
    return PDF_DIR / filename