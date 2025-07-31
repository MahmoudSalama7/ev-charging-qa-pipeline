import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric

class DataDriftMonitor:
    def __init__(self):
        self.reference_data = pd.read_parquet("data/reference.parquet")
        self.column_mapping = ColumnMapping(
            target=None,
            numerical_features=['latitude', 'longitude'],
            categorical_features=['station_name', 'connector_types']
        )

    def check_drift(self, current_data):
        report = Report(metrics=[DatasetDriftMetric()])
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        return report.as_dict()['metrics'][0]['result']['dataset_drift']