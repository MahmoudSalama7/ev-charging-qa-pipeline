# src/data_collection/pdf_extractor.py
import PyPDF2
import re
import logging
from pathlib import Path
from typing import Dict, Optional

class PDFExtractor:
    def __init__(self, pdf_path: str):
        """Initialize with either absolute or relative path"""
        self.pdf_path = Path(pdf_path).absolute()
        self.logger = logging.getLogger(__name__)
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at {self.pdf_path}")

    def extract_text(self) -> Dict[str, Optional[str]]:
        """Extract text with improved formatting preservation"""
        result = {
            "source": self.pdf_path.name,
            "text": None,
            "pages": 0,
            "status": "error"
        }

        try:
            with open(self.pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                result['pages'] = len(reader.pages)
                
                text_parts = []
                for i, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            cleaned = self._clean_text(page_text)
                            text_parts.append(f"--- PAGE {i+1} ---\n{cleaned}")
                    except Exception as e:
                        self.logger.warning(f"Page {i+1} error: {str(e)}")
                
                if text_parts:
                    result['text'] = "\n\n".join(text_parts)
                    result['status'] = "success"

            return result

        except Exception as e:
            self.logger.error(f"Extraction failed: {str(e)}")
            return result

    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\w\s.,:;\-\n]', '', text)  # Keep basic punctuation
        return text.strip()