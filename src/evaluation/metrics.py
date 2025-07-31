#metric.py
import evaluate
import numpy as np
from rapidfuzz import fuzz  # For fuzzy matching
from typing import List

class QAEvaluator:
    def __init__(self):
        self.rouge = None
        try:
            import rouge_score  # First try direct import
            self.rouge = evaluate.load("rouge")
        except:
            pass
            
    def clean_answer(self, text: str) -> str:
        """Extract just the answer part"""
        if "Response:" in text:
            return text.split("Response:")[-1].strip()
        return text.strip()

    def calculate_metrics(self, preds: List[str], refs: List[str]) -> dict:
        """Calculate all metrics with fuzzy matching"""
        preds_clean = [self.clean_answer(p) for p in preds]
        refs_clean = [self.clean_answer(r) for r in refs]
        
        metrics = {
            "avg_length": np.mean([len(p.split()) for p in preds_clean]),
        }
        
        # Add fuzzy match ratio (0-100)
        metrics["fuzzy_match"] = np.mean([
            fuzz.ratio(p.lower(), r.lower())
            for p, r in zip(preds_clean, refs_clean)
        ])
        
        # Add ROUGE if available
        if self.rouge:
            try:
                rouge_scores = self.rouge.compute(
                    predictions=preds_clean,
                    references=refs_clean,
                    use_stemmer=True
                )
                metrics.update({
                    "rouge1": rouge_scores["rouge1"],
                    "rouge2": rouge_scores["rouge2"],
                    "rougeL": rouge_scores["rougeL"]
                })
            except:
                pass
                
        return metrics