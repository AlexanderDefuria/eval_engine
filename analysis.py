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
                 model: Optional[str] = None, 
                 resampler: Optional[str] = None, 
                 score: Optional[str] = None, tag: 
                 Optional[str] = None):
        self.model: Pattern = re.compile(model) if model else re.compile(r'(.*)')
        self.resampler = re.compile(resampler) if resampler else re.compile(r'(.*)') 
        self.score = re.compile(score) if score else re.compile(r'(.*)')
        self.tag = re.compile(tag) if tag else re.compile(r'(.*)')


class Analysis:
    def __init__(self, load_models: bool = False, load_dataframes: bool = False, filter: Filter = Filter(), output: bool = True) -> None:
        self.records_path: Path = Path('records')
        self.report_path: Path = Path('reports')
        self.analytics_data: pd.DataFrame = pd.DataFrame()
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
        # Discover the files to load
        job_list = self._discover_files()       

        # Load the data
        for job in tqdm(job_list, desc="Loading data"):
            self._handle_file(*job)
            if not self.load_dataframes and job[3].suffix == '.csv':
                self._calculate_metrics(self.analytics_data)
                self.analytics_data = pd.DataFrame() # Clear the data to save memory and processing time
        
        # Save the results to a file
        if self.load_dataframes:
            self.analytics_data.groupby(['model', 'resampler', 'score', 'tag']).apply(self._calculate_metrics)
        if self.output: 
            self._save()

    def _discover_files(self) -> List[Tuple[Path, Path, Path, Path, str]]:
        job_list: List[Tuple[Path, Path, Path, Path, str]] = []
        for resampler_dir in self.records_path.iterdir():
            if not self.filter.resampler.match(resampler_dir.name): continue
            for model_dir in resampler_dir.iterdir():
                if not self.filter.model.match(model_dir.name): continue
                for score_dir in model_dir.iterdir():
                    if not self.filter.score.match(score_dir.name): continue
                    for record in score_dir.iterdir():
                        if record.is_dir():
                            if not self.filter.tag.match(record.name): continue
                            for record_file in record.iterdir():
                                job_list.append((resampler_dir, model_dir, score_dir, record_file, record.name))
                        else:
                            job_list.append((resampler_dir, model_dir, score_dir, record, "base"))
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
        recall = safe_divide(len(true_positives), len(true_positives) + len(false_negatives))
        accuracy = safe_divide(len(true_positives) + len(true_negatives), len(group))
        false_negative_rate = safe_divide(len(false_negatives), len(true_positives) + len(false_negatives))
        false_positive_rate = safe_divide(len(false_positives), len(true_negatives) + len(false_positives))
        fp_fn = safe_divide(1, false_positive_rate + false_negative_rate, default = 100)

        self.results = pd.concat([self.results, pd.DataFrame({
            'model': group['model'].unique(),
            'resampler': group['resampler'].unique(),
            'score': group['score'].unique(),
            'tag': group['tag'].unique(),
            'fold': group['fold'].unique(),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1': f1,
            'false_negative_rate': false_negative_rate,
            'false_positive_rate': false_positive_rate,
            'actual_positives': len(true_positives) + len(false_negatives),
            'fp_fn': fp_fn
        })])


    def _handle_file(self, resampler_dir: Path, model_dir: Path, score_dir: Path, file: Path, tag = "") -> None:
            if file.suffix == '.csv':
                record_data = pd.read_csv(file)
                record_data['resampler'] = resampler_dir.name
                record_data['model'] = model_dir.name
                record_data['score'] = score_dir.name
                record_data['tag'] = tag
                match = self.results_csv_regex.match(file.name)
                record_data['fold'] = int(match.group(1)) if match else -1
                self.analytics_data = pd.concat([self.analytics_data, record_data])
            elif '_model.pkl' in file.name and self.load_models: 
                self.models[model_dir.name] = Storage(file.parent).load_model()


    def _load_model(self, model_path, model_name: Optional[str] = None) -> None:
        with open(model_path, 'rb') as file:
            self.models[model_name] = pickle.load(file)


    def filter(self, model: Optional[str], resampler: Optional[str] = None, score: Optional[str] = None, tag: Optional[str] = None) -> pd.DataFrame:
        """
        Filter the analytics data based on the given parameters.
        """
        filtered = self.analytics_data
        if model:
            filtered = filtered[filtered['model'] == model]
        if resampler:
            filtered = filtered[filtered['resampler'] == resampler]
        if score:
            filtered = filtered[filtered['score'] == score]
        if tag:
            filtered = filtered[tag in filtered['tag']]
        return filtered


    def _save(self) -> None:
        directory = self.report_path / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(directory)
        self.analytics_file = directory / "analysis.csv"
        self.results.to_csv(self.analytics_file, index=False)       


