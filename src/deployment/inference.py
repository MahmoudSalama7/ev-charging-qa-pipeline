#inference.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Dict, Any
import logging
from pathlib import Path

class EVQAInference:
    def __init__(self, model_dir: str = "models/finetuned_tinyllama_evqa_cpu"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = Path(model_dir)
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the fine-tuned model"""
        try:
            base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            self.model = PeftModel.from_pretrained(
                self.model,
                self.model_dir
            ).to(self.device)
            
            self.logger.info("Model loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            return False

    def generate_response(self, input_text: str, max_length: int = 100) -> Dict[str, Any]:
        """Generate response for EV charging questions"""
        if not self.model or not self.tokenizer:
            return {"error": "Model not loaded"}
        
        try:
            prompt = f"Instruction: {input_text}\nResponse:"
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=0.7
            )
            
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = full_text.replace(prompt, "").strip()
            
            return {
                "question": input_text,
                "answer": answer,
                "status": "success"
            }
        except Exception as e:
            self.logger.error(f"Inference error: {str(e)}")
            return {"error": str(e), "status": "failed"}