# 🔌 EV Charging QA Pipeline

[![CI/CD](https://github.com/MahmoudSalama7/ev-charging-qa-system/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/MahmoudSalama7/ev-charging-qa-system/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

An end-to-end system for automated question answering about electric vehicle charging infrastructure, featuring data collection, model fine-tuning, and API deployment.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git LFS (for model files)
- [Poetry](https://python-poetry.org/) (recommended)

# Clone with LFS support
git lfs install
git clone https://github.com/MahmoudSalama7/ev-charging-qa-system.git
cd ev-charging-qa-system

# Install dependencies
poetry install  # or pip install -r requirements.txt

# Setup environment
cp .env.example .env
nano .env  # Add your API keys

# Run full pipeline
python src/orchestration/workflow.py


📂 Project Structure

ev-charging-qa-system/
├── config/               # Configuration files
│   └── settings.py       # Main configuration
├── data/                 # Data storage
│   ├── raw/              # Raw collected data
│   └── processed/        # Processed datasets
├── models/               # Model files (Git LFS)
├── src/                  # Source code
│   ├── data_collection/  # Data collection modules
│   ├── dataset_preparation/ # QA pair generation
│   ├── fine_tuning/      # Model training
│   ├── deployment/       # API serving
│   └── orchestration/    # Workflow management
├── tests/                # Test suite
├── .github/workflows/    # CI/CD pipelines
├── .env.example          # Environment template
├── pyproject.toml        # Poetry config
└── README.md             # This file

🔧 Configuration
Edit these files for setup:

1- config/settings.py - Core parameters:

class Config:
    BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    MAX_SEQ_LENGTH = 256  # For fine-tuning
    API_RATE_LIMIT = 100  # requests/minute

2- .env - Secrets (copy from .env.example):
API_KEYS="your_nrel_api_key"
MLFLOW_TRACKING_URI="http://localhost:5000" 

⚡ Key Commands
Task	Command
Run data collection	python src/data_collection/main.py
Generate QA pairs	python src/dataset_preparation/run_dataset_prep.py
Fine-tune model	python src/fine_tuning/run_finetuning.py
Start API	uvicorn src.deployment.api:app --reload
Run tests	pytest tests/
🌐 API Documentation
After starting the API:

Interactive docs: http://localhost:8000/docs

Endpoint: POST /ask

{
  "question": "What connectors do Tesla Superchargers use?",
  "max_length": 100
}


🤖 CI/CD Pipeline
Automated workflows:

CI Pipeline (on PR):

Unit tests

Data validation

Code formatting check

CD Pipeline (on main):

Model deployment

Smoke tests

Documentation update

📊 Evaluation Metrics
We track:

ROUGE-L: 0.85

BLEU-4: 0.78

Exact Match: 0.65

Latency: <500ms

📜 License
MIT License - See LICENSE for details.
