#model_registry.py
import json
from datetime import datetime
from pathlib import Path
import hashlib
import logging

class ModelRegistry:
    def __init__(self, registry_path: str = "model_registry.json"):
        self.registry_path = Path(registry_path)
        self.logger = logging.getLogger(__name__)
        self.registry = self._load_registry()
        self.logger.info(f"Model registry initialized at {self.registry_path}")

    def _load_registry(self):
        try:
            if self.registry_path.exists():
                with open(self.registry_path) as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading registry: {str(e)}")
        return {"models": []}

    def register_model(self, model_path: str, metadata: dict = None):
        """Register a new model version with metadata"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise ValueError("Model path does not exist")

            checksum = self._calculate_checksum(model_path)
            model_entry = {
                "path": str(model_path.absolute()),
                "timestamp": datetime.now().isoformat(),
                "checksum": checksum,
                "metadata": metadata or {},
                "version": f"v{len(self.registry['models']) + 1}"
            }

            self.registry["models"].append(model_entry)
            self._save_registry()
            self.logger.info(f"Registered new model: {model_entry['version']}")
            return model_entry
        except Exception as e:
            self.logger.error(f"Model registration failed: {str(e)}")
            raise

    def _calculate_checksum(self, model_path: Path) -> str:
        """Calculate SHA256 checksum of model files"""
        sha256 = hashlib.sha256()
        for file in model_path.glob("*"):
            if file.is_file():
                with open(file, "rb") as f:
                    while chunk := f.read(4096):
                        sha256.update(chunk)
        return sha256.hexdigest()

    def _save_registry(self):
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)

    def get_latest_model(self) -> dict:
        """Get the latest registered model"""
        if not self.registry["models"]:
            return None
        return self.registry["models"][-1]