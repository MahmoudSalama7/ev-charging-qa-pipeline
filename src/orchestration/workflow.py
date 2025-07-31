import schedule
import time
import subprocess
import logging
from datetime import datetime
from enum import Enum
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('orchestration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TriggerType(Enum):
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    API = "api"

class EVQAWorkflow:
    def __init__(self):
        self.is_running = False
        self.project_root = Path(__file__).parent.parent.parent
        self.lock = threading.Lock()

    def run_pipeline(self, trigger: TriggerType = TriggerType.MANUAL):
        """Main workflow executor with thread safety"""
        if self.is_running:
            logger.warning("Pipeline already running. Skipping...")
            return False

        with self.lock:
            self.is_running = True
            try:
                logger.info(f"Starting pipeline (Trigger: {trigger.value})")
                start_time = datetime.now()

                # 1. Data Collection Stage
                self._run_command("python src/data_collection/main.py")

                # 2. Training Stage
                self._run_command("python src/training/run_finetuning.py")

                # 3. Evaluation Stage
                self._run_command("python src/evaluation/comparator.py")

                # 4. Deployment Stage (if evaluation passes)
                self._run_command("python src/deployment/update_model.py")

                duration = datetime.now() - start_time
                logger.info(f"Pipeline completed successfully in {duration}")
                return True

            except Exception as e:
                logger.error(f"Pipeline failed: {str(e)}")
                return False
            finally:
                self.is_running = False

    def _run_command(self, command: str):
        """Execute a shell command with error handling"""
        logger.info(f"Executing: {command}")
        try:
            result = subprocess.run(
                command.split(),
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            logger.debug(f"Command output:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e.stderr}")
            raise

    def start_scheduler(self):
        """Start scheduled pipeline runs"""
        # Daily at 2 AM
        schedule.every().day.at("02:00").do(
            self.run_pipeline, 
            trigger=TriggerType.SCHEDULED
        )

        # Every Sunday at 3 AM
        schedule.every().sunday.at("03:00").do(
            self.run_pipeline,
            trigger=TriggerType.SCHEDULED
        )

        logger.info("Scheduler started with 2 jobs registered")
        while True:
            schedule.run_pending()
            time.sleep(60)

if __name__ == "__main__":
    workflow = EVQAWorkflow()

    # Start scheduler in background thread
    scheduler_thread = threading.Thread(
        target=workflow.start_scheduler,
        daemon=True
    )
    scheduler_thread.start()

    # Manual trigger example (can be called via API too)
    workflow.run_pipeline(trigger=TriggerType.MANUAL)