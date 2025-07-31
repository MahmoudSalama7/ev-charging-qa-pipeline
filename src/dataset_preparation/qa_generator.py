import logging
import re
from pathlib import Path
from typing import List, Dict, Union
import pandas as pd
import json  
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class EVQAGenerator:
    def __init__(self, model_name: str = "gpt2-medium"):
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.logger.info(f"Loaded {model_name} on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def generate_from_source(self, source: Union[Path, pd.DataFrame], source_type: str) -> List[Dict]:
        if source_type == 'pdf':
            return self._generate_from_pdf(source)
        elif source_type == 'stations':
            return self._generate_from_stations(source)
        else:
            raise ValueError(f"Unknown source type: {source_type}")

    def _generate_from_pdf(self, pdf_path: Path) -> List[Dict]:
        try:
            with open(pdf_path, 'r') as f:
                pdf_data = json.load(f)[0]['text']
            return self._gpt_generate_qa(pdf_data)
        except Exception as e:
            self.logger.error(f"PDF processing failed: {str(e)}")
            return []

    def _gpt_generate_qa(self, text: str, num_questions: int = 5) -> List[Dict]:
        prompt = f"""Generate exactly {num_questions} technical question-answer pairs about EV charging from this text.
Follow these rules:
1. Questions must be specific to EV charging technology
2. Answers must be factual and derived from the text
3. Use this exact format for each pair:
Question: [Your question here?]
Answer: [The answer here.]

Example of good questions:
Question: What types of EV connectors support fast charging?
Answer: CCS and CHAdeMO connectors support fast charging.

Text to analyze:
{text[:1500]}

Generate {num_questions} Q&A pairs:
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=1
        )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_gpt_output(generated)

    def _parse_gpt_output(self, text: str) -> List[Dict]:
        pairs = []
        pattern = r'Question:\s*(.*?)\s*Answer:\s*(.*?)(?=\nQuestion:|$)'
        
        for match in re.finditer(pattern, text, re.DOTALL):
            question = match.group(1).strip()
            answer = match.group(2).strip()
            
            if question and answer and question.endswith('?'):
                pairs.append({
                    "question": question,
                    "answer": answer,
                    "context": "GPT-generated",
                    "source": "pdf"
                })
        
        return pairs

    def _generate_from_stations(self, stations_df: pd.DataFrame) -> List[Dict]:
        qa_pairs = []
        
        for _, row in stations_df.iterrows():
            # Handle connector information
            connectors = [c for c in row.get('connectors', []) 
                         if c not in ['Unknown', '[Unknown]']]
            connector_text = ', '.join(connectors) if connectors else "No connector information available"
            
            qa_pairs.extend([
                {
                    "question": f"What connectors are available at {row['station_name']}?",
                    "answer": connector_text,
                    "context": f"Station {row['station_name']}",
                    "source": "stations"
                },
                {
                    "question": f"What type of charging is available at {row['station_name']}?",
                    "answer": "DC fast charging" if row.get('has_fast_charging') else "AC Level 2 charging",
                    "context": f"Station ID: {row.name}",
                    "source": "stations"
                }
            ])
        
        return qa_pairs