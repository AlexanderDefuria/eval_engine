import datetime
from datetime import datetime
import os
from pathlib import Path
import pickle
import re
from typing import Dict, List, Optional, Pattern, Tuple

import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm

from runtime.model import Model
from runtime.storage import Storage


class Filter:
    def __init__(self, 
                 cross_validator: Optional[str] = None,
                 model: Optional[str] = None, 
                 resampler: Optional[str] = None, 
                 score: Optional[str] = None, tag: 
                 Optional[str] = None):
        self.cross_validator: Pattern = re.compile(cross_validator) if cross_validator else re.compile(r'(.*)')
        self.model: Pattern = re.compile(model) if model else re.compile(r'(.*)')
        self.resampler = re.compile(resampler) if resampler else re.compile(r'(.*)') 
        self.score = re.compile(score) if score else re.compile(r'(.*)')
        self.tag = re.compile(tag) if tag else re.compile(r'(.*)')
        

class ResultSource:
    def __init__(self, cross_validator_dir: str, resampler_dir: str, model_dir: str, score_dir: str, file: str, tag="base"):
        self.cross_validator_dir = cross_validator_dir
        self.resampler_dir = resampler_dir
        self.model_dir = model_dir
        self.score_dir = score_dir
        self.file = file
        self.tag = tag
        
    def __iter__(self):
        return iter([self.cross_validator_dir, 
                     self.resampler_dir, 
                     self.model_dir, 
                     self.score_dir, 
                     self.file, 
                     self.tag])

    def __str__(self) -> str:
        return "_".join(self.__iter__())


class Analysis:
    def __init__(self, 
                 load_models: bool = False, 
                 load_dataframes: bool = False, 
                 filter: Filter = Filter(), 
                 output: bool = True, 
                 records_path: Path = Path('records')) -> None:
        self.records_path: Path = records_path
        self.report_path: Path = Path('reports')
        self.raw_data: pd.DataFrame = pd.DataFrame()
        self.models: Dict[str, Model] = {}
        self.results: pd.DataFrame = pd.DataFrame()
        self.analytics_file: Path = Path('')
        self.load_models: bool = load_models
        self.load_dataframes: bool = load_dataframes
        self.filter: Filter = filter
        self.results_csv_regex = re.compile(r"results_cv_(\d+)\.csv")
        self.output: bool = output

    def load_data(self) -> None:
        """
        Load the data from the records directory and calculate the metrics.
        """
        job_list = self._discover_files()       

        # Load the data
        for job in tqdm(job_list, desc="Loading data"):
            self._handle_file(job)
            if not self.load_dataframes and job.file.suffix == '.csv':
                self._calculate_metrics(self.raw_data)
                self.raw_data = pd.DataFrame() # Clear the data to save memory and processing time
        
        # Save the results to a file
        if self.load_dataframes:
            try:
                self.raw_data.groupby(['cross_validator', 'model', 'resampler', 'score', 'tag', 'fold']).apply(self._calculate_metrics)
            except Exception as e:
                print(self.raw_data.head())
                raise e
        if self.output: 
            self._save()

    def _discover_files(self) -> List[Tuple[Path, Path, Path, Path, Path, str]]:
        job_list: List[Tuple[Path, Path, Path, Path, str]] = []
        for cross_validator_dir in self.records_path.iterdir():
            if not self.filter.cross_validator.match(cross_validator_dir.name): continue
            if not cross_validator_dir.is_dir(): continue
            for resampler_dir in cross_validator_dir.iterdir():
                if not self.filter.resampler.match(resampler_dir.name): continue
                if not resampler_dir.is_dir(): continue
                for model_dir in resampler_dir.iterdir():
                    if not self.filter.model.match(model_dir.name): continue
                    if not model_dir.is_dir(): continue
                    for score_dir in model_dir.iterdir():
                        if not self.filter.score.match(score_dir.name): continue
                        if not score_dir.is_dir(): continue
                        for record in score_dir.iterdir():
                            if record.is_dir():
                                if not self.filter.tag.match(record.name): continue
                                for record_file in record.iterdir():
                                    job_list.append(ResultSource(cross_validator_dir, 
                                                                 resampler_dir, 
                                                                 model_dir, 
                                                                 score_dir, 
                                                                 record_file, 
                                                                 record.name))
                            else:
                                job_list.append(ResultSource(cross_validator_dir, 
                                                             resampler_dir, 
                                                             model_dir, 
                                                             score_dir,
                                                             record))
        return job_list
 
    def _calculate_metrics(self, group: pd.DataFrame) -> None:
        """
        Calculate a variety of metrics and include them in the analytics data.
        """
        def safe_divide(a, b, default = 0) -> float:
            return a / b if b != 0 else default

        f1 = f1_score(group['actual'], group['predicted'])
        true_positives = group[(group['actual'] == 1) & (group['predicted'] == 1)]
        false_positives = group[(group['actual'] == 0) & (group['predicted'] == 1)]
        false_negatives = group[(group['actual'] == 1) & (group['predicted'] == 0)]
        true_negatives = group[(group['actual'] == 0) & (group['predicted'] == 0)]             
   
        sensitivity = safe_divide(len(true_positives), len(true_positives) + len(false_negatives))
        specificity = safe_divide(len(true_negatives), len(true_negatives) + len(false_positives))
        precision = safe_divide(len(true_positives), len(true_positives) + len(false_positives))
        accuracy = safe_divide(len(true_positives) + len(true_negatives), len(group))
        false_negative_rate = safe_divide(len(false_negatives), len(true_positives) + len(false_negatives))
        false_positive_rate = safe_divide(len(false_positives), len(true_negatives) + len(false_positives))
        false_omission_rate = safe_divide(len(false_negatives), len(true_negatives) + len(false_negatives))
        false_discovery_rate = safe_divide(len(false_positives), len(true_positives) + len(false_positives))
        fp_fn = safe_divide(1, false_positive_rate + false_negative_rate, default = 100)
       
        new_df = pd.DataFrame({
            'cross_validator': group['cross_validator'].unique(),
            'model': group['model'].unique(),
            'resampler': group['resampler'].unique(),
            'score': group['score'].unique(),
            'tag': group['tag'].unique(),
            'fold': group['fold'].unique(),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision (PPV)': precision,
            'recall (Sensitivity)': sensitivity,
            'accuracy': accuracy,
            'f1': f1,
            'false_negative_rate': false_negative_rate,
            'false_positive_rate': false_positive_rate,
            'false_omission_rate': false_omission_rate,
            'false_discovery_rate': false_discovery_rate,
            'actual_positives': len(true_positives) + len(false_negatives),
            'actual_negatives': len(true_negatives) + len(false_positives),
            'fp_fn': fp_fn,
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'true_positives': len(true_positives),
            'true_negatives': len(true_negatives),
            'total': len(group)
        })
        
        self.results = pd.concat([self.results, new_df])


    # def _handle_file(self, cross_validator_dir: Path, resampler_dir: Path, model_dir: Path, score_dir: Path, file: Path, tag = "") -> None:
    def _handle_file(self, source: ResultSource) -> None:
            if source.file.suffix == '.csv':
                record_data = pd.read_csv(source.file)
                record_data['cross_validator'] = source.cross_validator_dir.name
                record_data['resampler'] = source.resampler_dir.name
                record_data['model'] = source.model_dir.name
                record_data['score'] = source.score_dir.name
                record_data['tag'] = source.tag
                match = self.results_csv_regex.match(source.file.name)
                record_data['fold'] = int(match.group(1)) if match else -1
                self.raw_data = pd.concat([self.raw_data, record_data])
            elif '_model.pkl' in source.file.name and self.load_models: 
#                TODO: Fix this
#                self.models[model_dir.name] = Storage(file.parent).load_model()
                name = source.__str__()
                self.models[name] = pickle.load(open(source.file, 'rb'))


    def _load_model(self, model_path, model_name: Optional[str] = None) -> None:
        with open(model_path, 'rb') as file:
            self.models[model_name] = pickle.load(file)


    def _save(self) -> None:
        directory = self.report_path / Path("analysis") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(directory)
        self.analytics_file = directory / "analysis.csv"
        self.results.to_csv(self.analytics_file, index=False)       


