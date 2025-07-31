#comprator.py
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.evaluation.metrics import QAEvaluator
import numpy as np
import warnings

class ModelComparator:
    def __init__(self, benchmark_path: str = "data/evaluation/ev_charging_benchmark.json"):
        self.benchmark_path = Path(benchmark_path)
        self.evaluator = QAEvaluator()
        
    def load_models(self):
        """Load both base and fine-tuned models with error handling"""
        base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            
            print("Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            print("Loading fine-tuned model...")
            fine_tuned_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            fine_tuned_model = PeftModel.from_pretrained(
                fine_tuned_model, 
                "models/finetuned_tinyllama_evqa_cpu"
            )
            
            return base_model, fine_tuned_model, tokenizer
            
        except Exception as e:
            warnings.warn(f"Model loading failed: {str(e)}")
            raise

    def generate_answers(self, model, tokenizer, questions: list) -> list:
        """Generate clean answers without prompt repetition"""
        answers = []
        for question in questions:
            inputs = tokenizer(
                f"Instruction: {question}\nResponse:",
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id
            )
            # Extract just the generated part
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = full_text.split("Response:")[-1].strip()
            answers.append(answer)
        return answers
        
    def evaluate_models(self):
        """Run full evaluation pipeline with error handling"""
        try:
            with open(self.benchmark_path) as f:
                benchmark = json.load(f)
            
            # Use first 3 samples for testing
            questions = [q["question"] for q in benchmark][:3]
            references = [self._get_reference_answer(q) for q in benchmark][:3]
            
            base_model, fine_tuned_model, tokenizer = self.load_models()
            
            print("\nEvaluating base model...")
            base_answers = self.generate_answers(base_model, tokenizer, questions)
            print("Evaluating fine-tuned model...")
            ft_answers = self.generate_answers(fine_tuned_model, tokenizer, questions)
            
            print("\nCalculating metrics...")
            base_metrics = self.evaluator.calculate_metrics(base_answers, references)
            ft_metrics = self.evaluator.calculate_metrics(ft_answers, references)
            
            # Sample comparison
            sample_idx = 0
            sample_comparison = {
                "question": questions[sample_idx],
                "base_answer": base_answers[sample_idx],
                "fine_tuned_answer": ft_answers[sample_idx],
                "reference": references[sample_idx]
            }
            
            return {
                "base_model_metrics": base_metrics,
                "fine_tuned_metrics": ft_metrics,
                "sample_comparison": sample_comparison
            }
            
        except Exception as e:
            print(f"Evaluation failed: {str(e)}")
            return None

    def _get_reference_answer(self, item: dict) -> str:
        """Generate reference answer from context"""
        context = item["context"]
        question = item["question"].lower()
        
        if "type" in question or "kind" in question:
            return f"This station offers {context['station_type']} charging."
        elif "payment" in question:
            return f"Payment methods accepted: {context['payment']}."
        elif "hour" in question or "time" in question:
            return "Open 24/7"
        return f"Station information: {context['station_type']} in {context['location']}."

if __name__ == "__main__":
    print("🚀 Starting EV Charging QA Model Evaluation")
    
    try:
        comparator = ModelComparator()
        results = comparator.evaluate_models()
        
        if results:
            print("\n📊 Evaluation Results:")
            print("Base Model Metrics:", json.dumps(results["base_model_metrics"], indent=2))
            print("\nFine-Tuned Model Metrics:", json.dumps(results["fine_tuned_metrics"], indent=2))
            
            print("\n🔍 Sample Comparison:")
            print("Question:", results["sample_comparison"]["question"])
            print("Base Model Answer:", results["sample_comparison"]["base_answer"])
            print("Fine-Tuned Answer:", results["sample_comparison"]["fine_tuned_answer"])
            print("Reference Answer:", results["sample_comparison"]["reference"])
        else:
            print("❌ Evaluation failed to produce results")
            
    except Exception as e:
        print(f"❌ Critical error in evaluation: {str(e)}")