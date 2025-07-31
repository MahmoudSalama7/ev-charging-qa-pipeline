import pandas as pd
from typing import List, Dict
from pathlib import Path
from .validator import QAValidator

class DatasetAugmentor:
    def __init__(self, qa_generator):
        self.qa_generator = qa_generator
        self.validator = QAValidator()

    def augment_dataset(self, sources: List[Dict]) -> pd.DataFrame:
        all_qa = []
        
        for source in sources:
            try:
                if source['type'] == 'pdf':
                    qa_pairs = self.qa_generator.generate_from_source(
                        Path(source['path']), 
                        'pdf'
                    )
                else:
                    df = pd.read_parquet(source['path'])
                    qa_pairs = self.qa_generator.generate_from_source(df, 'stations')
                
                valid_pairs = [qa for qa in qa_pairs if self.validator.validate_single(qa)['valid']]
                all_qa.extend(valid_pairs)
                
            except Exception as e:
                print(f"Failed processing {source['path']}: {str(e)}")
                continue
                
        return pd.DataFrame(all_qa)