# src/data_processing/station_processor.py
import pandas as pd
import logging
from pathlib import Path
from typing import Optional

class StationProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def process_stations(self, parquet_path: str) -> pd.DataFrame:
        """Process the station data from your parquet file"""
        try:
            df = pd.read_parquet(parquet_path)
            
            # Validate required columns
            required_cols = {'station_name', 'latitude', 'longitude', 'ev_connector_types'}
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"Missing columns: {missing}")

            # Clean data
            df = (df
                .pipe(self._clean_names)
                .pipe(self._extract_connectors)
                .pipe(self._add_metadata)
            )
            
            self.logger.info(f"Processed {len(df)} stations")
            return df

        except Exception as e:
            self.logger.error(f"Station processing failed: {str(e)}")
            raise

    def _clean_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean station names"""
        df['station_name'] = (
            df['station_name']
            .str.strip()
            .str.replace(r'\s+', ' ', regex=True)
        )
        return df

    def _extract_connectors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process connector types"""
        df['connectors'] = (
            df['ev_connector_types']
            .apply(lambda x: list(set(x)) if isinstance(x, list) else ['Unknown'])
        )
        df['connector_count'] = df['connectors'].str.len()
        return df

    def _add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived fields"""
        df['has_fast_charging'] = (
            df['ev_connector_types']
            .apply(lambda x: any('DC' in c for c in x) if isinstance(x, list) else False)
        )
        return df