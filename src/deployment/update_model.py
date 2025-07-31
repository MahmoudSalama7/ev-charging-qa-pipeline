#update_model.py
import shutil
from pathlib import Path
import logging
from .model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deploy_new_model(version: str = None):
    """Deploy the specified model version or latest if None"""
    try:
        registry = ModelRegistry()
        
        # Get specified version or latest
        model_info = None
        if version:
            for model in registry.registry["models"]:
                if model["version"] == version:
                    model_info = model
                    break
            if not model_info:
                raise ValueError(f"Version {version} not found")
        else:
            model_info = registry.get_latest_model()
            if not model_info:
                raise ValueError("No models in registry")
        
        # Path setup
        src_path = Path(model_info["path"])
        dest_path = Path("models/deployed_model")
        
        # Clean and copy
        if dest_path.exists():
            shutil.rmtree(dest_path)
        shutil.copytree(src_path, dest_path)
        
        logger.info(f"Deployed model version {model_info['version']}")
        return True
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", help="Specific model version to deploy")
    args = parser.parse_args()
    
    deploy_new_model(args.version)