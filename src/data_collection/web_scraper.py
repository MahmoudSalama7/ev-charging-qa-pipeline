import os
import requests
import pandas as pd
import time
from datetime import datetime
import logging
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler

# Configuration
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://developer.nrel.gov/api/alt-fuel-stations/v1"
CSV_FALLBACK_URL = "https://data.nrel.gov/system/files/156/alt_fuel_stations.csv"
OUTPUT_DIR = "data/ev_stations"
LOG_FILE = "logs/ev_scraper.log"

# Setup directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()  # Also log to console
    ]
)

def run_scraper():
    """Wrapper function for the scheduler"""
    try:
        logging.info("=== Starting scraping job ===")
        scrape_ev_data()
        logging.info("=== Job completed ===")
    except Exception as e:
        logging.error(f"Job failed: {str(e)}")

def scrape_ev_data():
    """Main scraping function"""
    try:
        # Your existing scraping logic here
        params = {
            "api_key": API_KEY,
            "fuel_type": "ELEC",
            "country": "US",
            "limit": 200
        }
        
        logging.info("Fetching EV station data...")
        response = requests.get(f"{BASE_URL}.json", params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data["fuel_stations"])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{OUTPUT_DIR}/stations_{timestamp}.parquet"
        df.to_parquet(output_file)
        logging.info(f"Saved {len(df)} stations to {output_file}")
        
    except Exception as e:
        logging.error(f"Scraping failed: {str(e)}")
        # Add fallback logic if needed

def main():
    logging.info("===== Application Starting =====")
    
    # Immediate test run
    run_scraper()
    
    # Configure scheduler
    scheduler = BlockingScheduler()
    
    # Add job with explicit timezone and coalescing
    scheduler.add_job(
        run_scraper,
        'cron',
        hour=3,
        minute=0,
        timezone='UTC',
        misfire_grace_time=3600,
        coalesce=True,
        max_instances=1
    )
    
    try:
        logging.info("Starting scheduler (CTRL+C to exit)...")
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Scheduler shutdown requested")
    except Exception as e:
        logging.error(f"Scheduler crashed: {str(e)}")
    finally:
        if scheduler.running:
            scheduler.shutdown()
        logging.info("===== Application Stopped =====")

if __name__ == "__main__":
    main()