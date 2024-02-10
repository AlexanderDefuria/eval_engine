
"""
    This file contains the engine of the project.
    It is what runs the models and the training, recording the entire process for later evaluation.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold
from runtime.model import Model, ModelDict
from runtime.resampling import *
from runtime.storage import Storage
from runtime.evaluation import *
from pandas import DataFrame
import requests
import pickle


class Engine:
    """
    The Engine.

    It supports multiple models and resamplers, and can run the training process in parallel using multiple processes.
    The results of the training are saved to a file in the HDF5 format. Models are simple and only need to be able to
    support the `fit` and `predict` methods. Resamplers are also simple and only need to be able to support the 
    `fit_resample` method.

    Parameters:
    models (List[Model] | ModelDict): The models to be used in the training.
    resamplers (List[Resampler]): The resamplers to be used in the training.
    scorers (Dict[str, str | Callable]): The scorers to be used in the training.
    data_file (Path): The path to the data input file. TODO - Add loading?
    X (DataFrame): The data to be used in the training.
    y (np.ndarray): The labels to be used in the training.
    verbosity (int): The verbosity level of the engine. 0 = silent, 1 = results.h5, 2 = verbose.
    records_dir (Path): The directory where the results.h5 file will be saved.
    max_workers (int): The maximum number of processes to be used in the training.
    output_name (str): The name of the output file `results`.
    output_format (str): The format of the output file. Can be `h5`, `json`, `csv`, or `pickle`(DataFrame).
    overwrite (bool): Whether to overwrite the output file if it already exists.
    disable_bayes_search (bool): Whether to disable the use of BayesSearchCV for hyperparameter optimization.
    kwargs: Additional keyword arguments.
    """
    output_lock = multiprocessing.Lock()

    def __init__(self, 
                models: List[Model] | ModelDict, 
                resamplers: Optional[List[Resampler] | List[None]] = None,
                scorers: Dict[str, str | Callable] = {"accuracy": accuracy},
                data_file: Path = Path("data"),
                X: np.ndarray = np.array([]),
                y: np.ndarray = np.array([]),
                verbosity: int = 1,
                records_dir: Path = Path("./records"),
                max_workers: int = 1,
                output_name: str = "results",
                output_format: str = "csv",
                overwrite: bool = False,
                tag: Optional[str] = None,
                disable_bayes_search: bool = False,
                cross_validate: bool = True,
                cross_validator: StratifiedKFold = StratifiedKFold(n_splits=5, random_state=None),
                **kwargs):

        # Handle data configuration tasks
        self.path = data_file
        self.X = X
        self.y = y
        self.logging = verbosity
        self.records_dir = records_dir
        self.max_workers = max_workers
        self.output_name = output_name
        self.output_format = output_format
        self.cross_validator = cross_validator
        self.cross_validate = cross_validate
        self.resamplers: List[Resampler] = []
        self.tag = tag
        try:
            # Handle model configuration tasks and creation of models Objects.
            self._load_models(models, scorers, disable_bayes_search=disable_bayes_search)
            # Handle loading of resamplers into Resampler objects
            self._load_resamplers(resamplers)
            # Handle output configuration tasks (create output file if it doesn't exist or overwrite if desired)
            self._setup_output(overwrite=overwrite, output_format=output_format)
        except Exception as e:
            print(f"Error while setting up engine: {e}")
            self.telegram(f"Error while setting up engine: {e}")
            raise e

    def run(self) -> None:
        """
        Initiates the training process for all combinations of models and resamplers.

        This method creates a ProcessPoolExecutor with a number of workers equal to `max_workers`.
        It then submits tasks to this executor for each combination of model and resampler.
        Each task is a call to the `_train` method with a specific model and resampler.

        Returns: 
        self: The engine instance.
        """
        executor = ProcessPoolExecutor(max_workers=self.max_workers)
        futures = []
        self.telegram(f"Started {self.tag}")
        for resampler in self.resamplers:
            for model in self.models:
                if self.max_workers == 1:
                    self._train(model, resampler)
                else:
                    futures.append(executor.submit(self._train, model, resampler))

        # TODO - Add progress bar and error checking here
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error while training: {e}")
                self.telegram("Error while training: {e}")
                raise e
        self.telegram(f"Done Training {self.output_file}")

    def telegram(self, msg: str):
        chat_id = "6556340412"
        try:
            token = os.environ.get('TELEGRAM_API_TOKEN')
        except:
            token = None
        
        if token: 
            url = f"https://api.telegram.org/bot{token}"
            params = {"chat_id": chat_id, "text": msg}
            if self.logging < 5 and msg != "Done Training.":
                return
            r = requests.get(url + "/sendMessage", params=params)

    def _train(self, model: Model, resampler: Resampler) -> None:
        """
        Trains the provided model using the provided resampler and the instance's data.

        Parameters:
        model (Model): The model to be trained.
        resampler (Resampler): The resampler to be used for resampling the training data.
        """
        iterator = self.cross_validator.split(self.X, self.y)
        if self.cross_validate == False:
            iterator = [(np.arange(len(self.X)), np.arange(len(self.X)))]
        for cv_round, (train_i, test_i) in enumerate(iterator):
            X_train, y_train = self.X[train_i,:], self.y[train_i]
            X_test, y_test = self.X[test_i,:], self.y[test_i]
            cv_round = f"cv_{cv_round}"
            
            X_train, y_train = resampler(X_train, y_train)
            model.train(X_train, y_train)
            predicted = model.predict(X_test)
            score = model.score(X_test, y_test)

            # TODO - This could be changed to store every fold in a single file, or every model in a single file.
            self._log(model, resampler, data=(test_i, y_test, predicted), cv_round=cv_round, score=score)
            self.telegram(f"""Model: {model}\nResampler: {resampler}\nCV Round: {cv_round}\nScorer: {model.scorer.__name__}\nScore: {score}\n""")

        self._save_model(model, resampler, model.scorer)

    def _log(self, model: Model, resampler: Resampler, data: Tuple, cv_round: str, score: float) -> None:
        """
        Logs the provided data to the output file. TODO - Maybe kill this method or improve it through a rewrite.

        Parameters:
        model (Model): The model used to generate the data (labelling purposes).
        resampler (Resampler): The resampler used to generate the data (labelling purposes).
        data (Tuple): The data to be saved.
        cv_round (int): The cross-validation round number.
        """
        if self.logging > 2:
            print(f"Saving data for {model}_{resampler}_{cv_round}")
        if self.logging > 1:
            print(f"{model}_{resampler}_{cv_round} = {score}") 
        if self.logging > 0:
            self._save_data(model, resampler, model.scorer, cv_round, data)


    def _save_data(self, model: Model, resampler: Resampler, scorer: callable, cv_round: str, data: Tuple, metadata: Optional[dict] = None) -> None:
        """
        Saves the provided data to a file. The file type is determined by the extension of the output file.

        Parameters:
        model (Model): The model used to generate the data (labelling purposes).
        resampler (Resampler): The resampler used to generate the data (labelling purposes).
        data (Tuple): The data to be saved. The first element is the round label, the rest are the data to be stored.

        Raises:
        AssertionError: If the data is not in the correct format.
        AssertionError: If the output file does not have a valid extension.
        """
        if self.logging <= 0:
            return # Don't save anything if verbosity is 0 or less.
        
        assert type(data) == tuple, "Data must be a tuple."
        assert len(data) > 1, "Data must contain at least 2 elements."
        assert self.output_format in ["csv", "pickle", "both"], "Output type must be valid."

        with self.output_lock:
            if self.tag:
                directory = self.records_dir / str(resampler) / str(model) / scorer.__name__ / str(self.tag) 
            else:
                directory = self.records_dir / str(resampler) / str(model) / scorer.__name__

            file_name = f"{self.output_name}_{cv_round}"
            df = DataFrame(data).transpose()
            df = df.set_axis(['index', 'actual', 'predicted'], axis=1)
            storage = Storage(directory)

            if self.output_format == "both":
                storage.save_dataframe(df, file_name)
                storage.save_csv(df, file_name)
            elif self.output_format == "pickle":
                storage.save_dataframe(df, file_name)
            else:
                storage.save_csv(df, file_name)

    def _save_model(self, model: Model, resampler: Resampler, scorer: callable) -> None:
        """
        Saves the provided model to a file.

        Parameters:
        model (Model): The model to be saved.
        resampler (Resampler): The resampler used to generate the model (labelling purposes).
        scorer (str): The scorer used to generate the model (labelling purposes).

        Raises:
        AssertionError: If the output file does not have a valid extension.
        """
        if self.logging <= 0:
            return
        with self.output_lock:
            print(f"Saving model for {model}_{resampler}_{scorer}")
            if self.tag:
                directory = self.records_dir / str(resampler) / str(model) / scorer.__name__ / str(self.tag)  
            else:
                directory = self.records_dir / str(resampler) / str(model) / scorer.__name__
            Storage(directory).save_model(model)

    def _load_models(self, models: List[Model] | ModelDict, scorers: Dict[str, str | Callable], disable_bayes_search: bool = False):
        """
        Loads the provided models into Model objects. This loads JSON/Dict from the script, not pickle objects.

        Parameters:
        models (List[Model] | ModelDict): The models to be used in the training.
        scorers (Dict[str, str | Callable]): The scorers to be used in the training.
        disable_bayes_search (bool): Whether to disable the use of BayesSearchCV for hyperparameter optimization.
        """
        self.scorers = scorers
        if type(models) == dict:
            self.models = []
            for scoring_method  in scorers.values():
                for name, json in models.items():
                    json["scorer"] = scoring_method
                    self.models.append(Model.from_json(name, json, disable_bayes_search=disable_bayes_search))
        elif type(models) == list:
            self.models = models


    def _load_resamplers(self, resamplers: Optional[List[Resampler | None]] = None) -> None:
        """
        Loads the provided resamplers into Resampler objects.
        """
        resamplers if resamplers else [Resampler("none")]
        for i, resampler in enumerate(resamplers):
            if resampler is None:
                resamplers[i] = Resampler("none")
            if type(resampler) != Resampler:
                resamplers[i] = Resampler(resampler=resampler)
            else: 
                resamplers[i] = resampler
        self.resamplers = resamplers


    def _setup_output(self, overwrite: bool = False, output_format: str = "h5"):
        """
        Sets up the output directory.
        """
        
        self.output_file = (self.records_dir / (str(self.output_name) + '.' + str(output_format))).resolve()
        if not self.records_dir.exists():
            self.records_dir.mkdir()
        if self.output_file.exists():
            if overwrite:
                os.remove(self.output_file)
            else:
                raise FileExistsError(f"File {self.output_file} already exists. Use overwrite=True to overwrite it.")




