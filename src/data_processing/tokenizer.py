# src/data_processing/tokenizer.py
import spacy
import logging
from typing import Dict, List
from src.config.paths import PROCESSED_DIR

logger = logging.getLogger(__name__)

class TextTokenizer:
    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.error(f"Spacy model {model_name} not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)

    def process_text(self, text: str) -> Dict:
        """Tokenize single document"""
        doc = self.nlp(text)
        return {
            "tokens": [token.text for token in doc],
            "lemmas": [token.lemma_ for token in doc],
            "entities": [
                {"text": ent.text, "label": ent.label_}
                for ent in doc.ents
            ],
            "sentences": [sent.text for sent in doc.sents]
        }

    def batch_process(self, documents: List[Dict]) -> List[Dict]:
        """Process multiple PDF extracts"""
        results = []
        for doc in documents:
            if not doc.get('text'):
                logger.warning(f"Skipping empty document: {doc.get('source')}")
                continue
                
            try:
                analysis = self.process_text(doc['text'])
                results.append({
                    **doc,
                    **analysis,
                    "token_count": len(analysis['tokens']),
                    "entity_count": len(analysis['entities'])
                })
            except Exception as e:
                logger.error(f"Failed to process {doc.get('source')}: {str(e)}")
        return results