# src/dataset_preparation/formatter.py
from typing import List, Dict
import pandas as pd

class EVQAFormatter:
    def __init__(self, format_template: str = "alpaca"):
        self.format_template = format_template
        self.templates = {
            "alpaca": self._format_alpaca,
            "chatml": self._format_chatml,
            "plain": self._format_plain
        }
        
        if format_template not in self.templates:
            raise ValueError(f"Unknown format template: {format_template}")

    def format_dataset(self, qa_df: pd.DataFrame) -> List[str]:
        """Convert QA pairs to specified format"""
        format_fn = self.templates[self.format_template]
        return [format_fn(row) for _, row in qa_df.iterrows()]

    def _format_alpaca(self, qa: Dict) -> str:
        return (
            "Below is an instruction about EV charging. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{qa['question']}\n\n"
            f"### Context:\n{qa['context']}\n\n"
            f"### Response:\n{qa['answer']}"
        )

    def _format_chatml(self, qa: Dict) -> str:
        return (
            f"<|im_start|>user\n{qa['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n{qa['answer']}<|im_end|>"
        )

    def _format_plain(self, qa: Dict) -> str:
        return f"Q: {qa['question']}\nA: {qa['answer']}"