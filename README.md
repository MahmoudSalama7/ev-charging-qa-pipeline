# Electric Vehicle Charging Stations QA Pipeline

## Overview
This project implements an end-to-end AI pipeline for a QA system on electric vehicle charging stations, using Llama-3-7B fine-tuned with LoRA. It includes data collection, processing, dataset preparation, fine-tuning, evaluation, deployment, and orchestration.

## Setup
1. **Clone Repository**
   git clone https://github.com/your-username/ev-charging-qa-pipeline.git
   cd ev-charging-qa-pipeline


## Set Up Virtual Environment

    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate



## Install Dependencies

    pip install -r requirements.txt



## Configure Environment VariablesCreate .env:

    cp .env.example .env

Add your Hugging Face token and JWT secret.



## Create Project Structure

    chmod +x setup_project.sh
    ./setup_project.sh



- Add PDF FilesPlace PDF files in data/pdfs/ (e.g., download from https://afdc.energy.gov/).
- Run Pipeline
    python src/orchestration/workflow.py



## Start API

    uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000

### Test with:

    curl -H "Authorization: Bearer $(python -c 'import jwt; print(jwt.encode({\"user\": \"test\"}, \"your_jwt_secret_key\", algorithm=\"HS256\"))')" -X POST -d '{"question": "What are EV charging station types?"}' http://localhost:8000/qa



## Monitor
Logs:  logs/pipeline.log

Prometheus: http://localhost:9090 (run python src/deployment/monitoring.py)

## CI/CD

- GitHub Actions runs tests and deploys on push to main.
- Configure HUGGINGFACE_TOKEN and JWT_SECRET in GitHub Secrets.

## Requirements
- Python 3.8+
- GPU (recommended for fine-tuning)
- Hugging Face token
- PDF files in data/pdfs/