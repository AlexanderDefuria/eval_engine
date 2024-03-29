from pathlib import Path
import pickle
from typing import Optional
from pandas import DataFrame
from sklearn.model_selection import BaseCrossValidator, KFold, RepeatedStratifiedKFold, StratifiedKFold
from runtime.model import Model
import os


class Storage:

    def __init__(self, base_path: Path, cv_name: str, resampler: str, model: str, scorer: str, tag: Optional[str] = None) -> None:
        self.path = base_path / cv_name / resampler / model / scorer / (tag if tag else '')
        if not self.path.exists():
            os.makedirs(self.path, exist_ok=True)

    def save_dataframe(self, df: DataFrame, name: str) -> None:
        if '.' in name:
            raise ValueError('Name should not contain an extension, the Storage class will add it automatically')
        self.path = self.path / f'{name}_df.pkl'
        with open(self.path, 'wb') as f:
            pickle.dump(df, f)

    def load_dataframe(self, name: str) -> DataFrame:
        if '.' in name:
            raise ValueError('Name should not contain an extension, the Storage class will add it automatically')
        self.path = self.path / f'{name}_df.pkl'
        with open(self.path, 'rb') as f:
            return pickle.load(f)

    def save_model(self, model: Model) -> None:
        self.path = self.path / f'{model.name}_model.pkl'
        with open(self.path, 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, name: Optional[str] = None) -> Model:
        dir_content = os.listdir(self.path)
        model_files = [f for f in dir_content if f.endswith('_model.pkl')]
        model_file = ""
        if len(model_files) == 0:
            raise FileNotFoundError('No model files found in the given directory')
        if len(model_files) > 1:
            if name is None:
                raise ValueError('Multiple model files found in the given directory, please specify a name')
            if name + '_model.pkl' not in model_files: 
                raise ValueError(f'No model file found with the name {name}')
            model_file = name + '_model.pkl'
        if len(model_files) == 1:
            model_file = model_files[0]
        
        model = pickle.load(open(self.path / model_file, 'rb'))
        if not isinstance(model, Model):
            raise TypeError('The loaded model is not of the type Model')
        else:
            return model

    def save_csv(self, df: DataFrame, name: str) -> None:
        if '.' in name:
            raise ValueError('Name should not contain an extension, the Storage class will add it automatically')
        self.path = self.path / f'{name}.csv'
        df.to_csv(self.path, index=False)

    @classmethod
    def cross_validator_name(cls, cross_validator: BaseCrossValidator) -> str:
        if cross_validator is None:
            name = "None"
        elif type(cross_validator) == StratifiedKFold:
            name = f"StratifiedKFold_{cross_validator.n_splits}"
        elif type(cross_validator) == KFold:
            name = f"KFold_{cross_validator.n_splits}"
        elif type(cross_validator) == RepeatedStratifiedKFold:
            name = f"RepeatedStratifiedKFold_{cross_validator.n_repeats}x{cross_validator.cvargs['n_splits']}"
        else:
            name = cross_validator.__name__
        
        return name