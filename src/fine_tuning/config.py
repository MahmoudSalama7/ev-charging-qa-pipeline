from dataclasses import dataclass

@dataclass
class FineTuningConfig:
    # Model configuration
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    local_model_dir: str = "models/tinyllama-cache"
    
    # LoRA configuration (works on CPU)
    max_seq_length: int = 256  # Reduced for CPU memory
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules = ["q_proj", "v_proj"]
    
    # Training hyperparameters (optimized for CPU)
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 1
    learning_rate: float = 1e-5  # Lower for CPU stability
    logging_steps: int = 10
    save_steps: int = 200
    warmup_steps: int = 50
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Data parameters
    dataset_text_field: str = "text"
    packing: bool = False  # Must be False for CPU
    
    # Data paths
    train_data_path: str = "data/training/ev_qa_dataset.txt"
    eval_data_path: str = "data/validation/ev_qa_eval.txt"
    
    # Output directory
    output_dir: str = "models/finetuned_tinyllama_evqa_cpu"