import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_preparation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import from local modules
from dataset_preparation import (
    EVQAGenerator,
    DatasetAugmentor, 
    EVQAFormatter,
    QAValidator
)

def main():
    try:
        logger.info("Initializing components...")
        
        # Initialize with GPT-2 (updated initialization)
        qa_generator = EVQAGenerator(model_name="gpt2-medium")
        augmentor = DatasetAugmentor(qa_generator)
        formatter = EVQAFormatter(format_template="alpaca")
        
        # Input sources - update these paths to your actual files
        sources = [
            {
                "path": r"data/processed/processed_pdf_20250730_150124.json",
                "type": "pdf"
            },
            {
                "path": r"data/processed/processed_stations_20250730_150125.parquet",
                "type": "stations"
            }
        ]
        
        logger.info("Starting data processing pipeline...")
        
        # 1. Generate Q&A pairs
        qa_df = augmentor.augment_dataset(sources)
        logger.info(f"Generated {len(qa_df)} QA pairs")
        
        # 2. Validate quality
        validation = QAValidator.validate_batch(qa_df.to_dict('records'))
        logger.info(f"Validation: {validation['valid_pairs']} valid, {validation['invalid_pairs']} invalid")
        
        # 3. Format for training
        formatted_data = formatter.format_dataset(qa_df)
        
        # 4. Save outputs
        output_dir = Path("data/training")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        training_path = output_dir / "ev_qa_dataset.txt"
        with open(training_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(formatted_data))
        logger.info(f"Saved training data to {training_path}")
        
        report_path = output_dir / "quality_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"QA Pair Quality Report\n{'='*30}\n")
            f.write(f"Valid Pairs: {validation['valid_pairs']}/{len(qa_df)}\n")
            if validation['error_breakdown']:
                f.write("\nError Breakdown:\n")
                for err, count in validation['error_breakdown'].items():
                    f.write(f"- {err}: {count}\n")
        logger.info(f"Saved quality report to {report_path}")
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()