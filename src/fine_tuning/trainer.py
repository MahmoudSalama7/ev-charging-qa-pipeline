import os
import time
import torch
import warnings
from typing import Optional
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset
from .config import FineTuningConfig
from .tracker import ExperimentTracker

class EVQATrainer:
    def __init__(self, config: FineTuningConfig):
        set_seed(42)
        self._setup_environment()
        self.config = config
        self.tracker = ExperimentTracker()
        self.device = "cpu"  # Force CPU mode
        self.model, self.tokenizer = self._load_model_and_tokenizer()

    def _setup_environment(self):
        load_dotenv()
        if token := os.getenv("HUGGINGFACE_TOKEN"):
            login(token=token)

    def _load_model_and_tokenizer(self):
        """Load model without quantization for CPU"""
        model = AutoModelForCausalLM.from_pretrained(
            self.config.local_model_dir or self.config.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.local_model_dir or self.config.model_name,
            padding_side="right",
            add_eos_token=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer

    def _format_instruction(self, example):
        """Simplified formatting for CPU"""
        text = example.get("text", "")
        if "### Instruction:" in text and "### Response:" in text:
            instruction = text.split("### Instruction:")[1].split("### Response:")[0].strip()
            response = text.split("### Response:")[1].strip()
            return f"Instruction: {instruction}\nResponse: {response}"
        return text

    def _load_dataset(self):
        """Load dataset with basic formatting"""
        def parse_file(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return [{"text": line.strip()} for line in f if line.strip()]
                
        train_data = parse_file(self.config.train_data_path)
        train_dataset = Dataset.from_list(train_data)
        
        eval_dataset = None
        if os.path.exists(self.config.eval_data_path):
            eval_data = parse_file(self.config.eval_data_path)
            eval_dataset = Dataset.from_list(eval_data)
            
        return train_dataset, eval_dataset

    def _apply_lora(self):
        """Configure LoRA for CPU"""
        peft_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, peft_config)
        print(f"Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def train(self):
        self._apply_lora()
        train_dataset, eval_dataset = self._load_dataset()

        # Disable all GPU-related features
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            fp16=False,  # Disabled
            evaluation_strategy="no",  # Disabled eval for simplicity
            optim="adamw_torch",
            save_total_limit=1,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            report_to=[],  # Disabled TensorBoard
            use_cpu=True,  # Explicit CPU flag
            disable_tqdm=True  # Disable progress bars for cleaner output
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            formatting_func=self._format_instruction
        )

        print("🚀 Starting CPU training (this will take a while)...")
        trainer.train()
        
        print("✅ Training completed! Saving model...")
        trainer.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        print(f"💾 Model saved to {self.config.output_dir}")