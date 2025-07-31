#api.py
from fastapi import FastAPI, HTTPException, Depends, status, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from typing import Optional
import logging
from pathlib import Path

# Import workflow components
from src.orchestration.workflow import EVQAWorkflow, TriggerType
from src.deployment.inference import EVQAInference
from src.deployment.model_registry import ModelRegistry

# Load environment variables
load_dotenv()

app = FastAPI(
    title="EV Charging QA API",
    description="API for EV charging station questions and pipeline control",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication Setup
security = HTTPBearer()
API_KEYS = set(os.getenv("API_KEYS", "").split(","))

# Initialize components
workflow = EVQAWorkflow()
model = EVQAInference()
model.load_model()
registry = ModelRegistry()
logger = logging.getLogger(__name__)

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key from Authorization header"""
    if credentials.credentials not in API_KEYS:
        logger.warning(f"Invalid API key attempt: {credentials.credentials}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return credentials.credentials

# Core API Routes
@app.get("/")
async def home():
    return {"message": "EV Charging QA API"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model.model is not None,
        "orchestrator": "active"
    }

@app.get("/model-info")
async def get_model_info(api_key: str = Depends(verify_api_key)):
    """Get current model information"""
    return registry.get_latest_model()

@app.post("/ask")
async def ask_question(
    question: str,
    max_length: int = 100,
    api_key: str = Depends(verify_api_key)
):
    """
    Get answers about EV charging stations
    Requires API key in Authorization header
    """
    logger.info(f"Question received: {question}")
    response = model.generate_response(question, max_length)
    
    if "error" in response:
        logger.error(f"Error processing question: {response['error']}")
        raise HTTPException(status_code=500, detail=response["error"])
    
    return response

# Orchestration Routes
orchestration_router = APIRouter(prefix="/orchestration", tags=["Orchestration"])

@orchestration_router.post("/trigger", status_code=status.HTTP_202_ACCEPTED)
async def trigger_pipeline(
    immediate: bool = False,
    api_key: str = Depends(verify_api_key)
):
    """
    Trigger the EVQA pipeline
    
    Parameters:
    - immediate: If True, runs immediately in foreground
    """
    if immediate:
        success = workflow.run_pipeline(trigger=TriggerType.API)
        return {
            "status": "completed" if success else "failed",
            "execution": "immediate"
        }
    else:
        # Run in background thread
        import threading
        thread = threading.Thread(
            target=workflow.run_pipeline,
            kwargs={"trigger": TriggerType.API},
            daemon=True
        )
        thread.start()
        return {
            "status": "started",
            "execution": "background"
        }

@orchestration_router.get("/status")
async def get_pipeline_status(api_key: str = Depends(verify_api_key)):
    """Get current pipeline status"""
    return {
        "is_running": workflow.is_running,
        "last_execution": workflow.last_execution_time,
        "next_scheduled": workflow.get_next_scheduled_run()
    }

app.include_router(orchestration_router)

# Model Management Routes
model_router = APIRouter(prefix="/models", tags=["Model Management"])

@model_router.post("/deploy")
async def deploy_model(
    version: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    """Deploy a specific model version"""
    try:
        from src.deployment.update_model import deploy_new_model
        deploy_new_model(version)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(model_router)