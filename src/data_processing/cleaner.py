# src/data_processing/cleaner.py
import pandas as pd
import re
import logging
from typing import Dict, List
from src.config.paths import PDF_DIR

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self):
        self.pdf_dir = PDF_DIR

    def clean_station_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Clean EV station API data"""
        REQUIRED_COLS = [
            'id', 'station_name', 'latitude', 'longitude',
            'ev_connector_types', 'ev_network'
        ]
        
        # Validate input
        missing_cols = [col for col in REQUIRED_COLS if col not in raw_df]
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            raise ValueError("Input data missing required columns")
        
        df = raw_df[REQUIRED_COLS].copy()
        
        # Cleaning operations
        df = (df
              .pipe(self._clean_text_fields)
              .pipe(self._handle_missing_values)
              .pipe(self._standardize_connectors)
             )
        
        logger.info(f"Cleaned {len(df)} station records")
        return df

    def _clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean all text columns"""
        text_cols = ['station_name', 'ev_network']
        for col in text_cols:
            df[col] = df[col].str.strip().str[:500]  # Prevent overly long strings
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values consistently"""
        df['ev_network'] = df['ev_network'].fillna('UNKNOWN')
        df['ev_connector_types'] = df['ev_connector_types'].fillna('[]')
        return df

    def _standardize_connectors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize connector types"""
        df['connector_types'] = (
            df['ev_connector_types']
            .apply(lambda x: sorted(set(x)) if isinstance(x, list) else [])
        )
        return df