# src/data_processing/process.py
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple

from src.data_collection.pdf_extractor import PDFExtractor
from src.data_processing.station_processor import StationProcessor
from src.data_processing.tokenizer import TextTokenizer
from src.data_processing.storage import DataStorage

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("data_processing.log"),
            logging.StreamHandler()
        ]
    )

def run_pipeline(
    pdf_path: str = "D:/ev-charging-qa-pipeline/data/pdfs/sample.pdf",
    station_path: str = "D:/ev-charging-qa-pipeline/data/ev_stations/stations_20250729_195245.parquet"
) -> Tuple[bool, bool]:
    """Run complete processing pipeline"""
    logger = logging.getLogger(__name__)
    storage = DataStorage()
    
    # 1. Process PDF
    pdf_success = False
    try:
        logger.info("Processing PDF...")
        pdf_data = PDFExtractor(pdf_path).extract_text()
        
        if pdf_data['status'] == 'success':
            tokenized = TextTokenizer().batch_process([pdf_data])
            storage.save(tokenized, "processed_pdf", "json")
            pdf_success = True
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")

    # 2. Process Stations
    station_success = False
    try:
        logger.info("Processing station data...")
        stations = StationProcessor().process_stations(station_path)
        storage.save(stations, "processed_stations", "parquet")
        station_success = True
    except Exception as e:
        logger.error(f"Station processing failed: {str(e)}")

    logger.info(f"Pipeline completed - PDF: {pdf_success}, Stations: {station_success}")
    return pdf_success, station_success

if __name__ == "__main__":
    configure_logging()
    run_pipeline()