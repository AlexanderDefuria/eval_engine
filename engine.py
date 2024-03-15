"""
    This file contains the engine of the project.
    It is what runs the models and the training, recording the entire process for later evaluation.
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
from typing import Dict, List, Tuple

import numpy as np
import requests
from pandas import DataFrame
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold

from runtime.evaluation import *
from runtime.model import Model, ModelDict
from runtime.resampling import *
from runtime.storage import Storage


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
    log_lock = multiprocessing.Lock()

    def __init__(self,
                models: List[Model] | ModelDict,
                resamplers: ResamplerList,
                scorers: Dict[str, str | Callable] | Metric,
                tag: str,
                X: np.ndarray = np.array([]),
                y: np.ndarray = np.array([]),
                verbosity: int = 1,
                records_dir: Path = Path("./records"),
                max_workers: int = 1,
                output_name: str = "results",
                output_format: str = "csv",
                disable_bayes_search: bool = False,
                cross_validate: bool = True,
                save_probabilities: bool = True,
                cross_validators: List[BaseCrossValidator] = [StratifiedKFold(n_splits=5, random_state=None)]):

        # Handle data configuration tasks
        self.X = X
        self.y = y
        self.verbosity = verbosity
        self.records_dir = records_dir
        self.max_workers = max_workers
        self.output_name = output_name
        self.output_format = output_format
        self.cross_validators = cross_validators
        self.cross_validate = cross_validate
        self.resamplers: List[Resampler] = []
        self.tag = tag
        self.save_probabilities = save_probabilities
        self.log_file = self.records_dir / "log.txt"

        try:
            # Handle model configuration tasks and creation of models Objects.
            self._load_models(models, scorers, disable_bayes_search=disable_bayes_search)
            # Handle loading of resamplers into Resampler objects
            self._load_resamplers(resamplers)
            # Handle output configuration tasks (create output file if it doesn't exist or overwrite if desired)
            self._setup_output()

        except Exception as e:
            self._log(f"Error while setting up engine: {e}")
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
        # Dispatch the training tasks.
        executor = ProcessPoolExecutor(max_workers=self.max_workers )
        futures = []
        for cross_validator in self.cross_validators:
            for resampler in self.resamplers:
                for model in self.models:
                    if self.max_workers == 1:
                        self._train(model, resampler, cross_validator)
                    else:
                        futures.append(executor.submit(self._train, model, resampler, cross_validator))

        # Wait for tasks to finish 
        # TODO Monitor progress and ensure progress is being made.
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                self._log(f"Error while training. {self.tag}: {e}")
                raise e

        executor.shutdown()
        self._log(f"Done Training {self.tag}.")


    def telegram(self, msg: str):
        try:
            chat_id = "6556340412"
            token = os.environ.get('TELEGRAM_API_TOKEN')
            url = f"https://api.telegram.org/bot{token}"
            params = {"chat_id": chat_id, "text": msg}
            r = requests.get(url + "/sendMessage", params=params)
        except:
            token = None
  
  
    def _train(self, model: Model, resampler: Resampler, cross_validator: BaseCrossValidator) -> None:
        """
        Trains the provided model using the provided resampler and the instance's data.

        Parameters:
        model (Model): The model to be trained.
        resampler (Resampler): The resampler to be used for resampling the training data.
        """
        try: 
            iterator = cross_validator.split(self.X, self.y)
            if self.cross_validate == False or self.cross_validators is None or self.cross_validators == []:
                iterator = [(np.arange(len(self.X)), np.arange(len(self.X)))]

            for cv_round, (train_i, test_i) in enumerate(iterator):
                X_train, y_train = self.X[train_i, :], self.y[train_i]
                X_test, y_test = self.X[test_i, :], self.y[test_i]
                cv_round = f"cv_{cv_round}"

                if not model.skip_resample:
                    X_train, y_train = resampler(X_train, y_train)
                    
                model.train(X_train, y_train)
                predicted = model.predict(X_test)
                score = model.score(X_test, y_test)

                if self.save_probabilities and hasattr(model.model, "predict_proba"):
                    proba = model.model.predict_proba(X_test)[:, 1]  # type: ignore
                self._save_data(model, cross_validator, resampler, model.scorer, cv_round, (test_i, y_test, predicted, proba)) # type: ignore
                self._log(f"""Model: {model}
                          CV:{cross_validator}
                          CV Round: {cv_round}
                          Resampler: {resampler}
                          Optimization: {model.scorer.__name__}
                          Score: {score}
                          Tag: {self.tag}\n""") 
                self._save_model(model, cross_validator, resampler, model.scorer) # type: ignore
                
        except Exception as e:
            error = f"Error while training {model}_{resampler}_{cross_validator}: {e}"
            raise Exception(error)


    def _log(self, msg: str) -> None:
        """
        Logs the provided data to the output file. TODO - Maybe kill this method or improve it through a rewrite.

        Parameters:
        model (Model): The model used to generate the data (labelling purposes).
        resampler (Resampler): The resampler used to generate the data (labelling purposes).
        data (Tuple): The data to be saved.
        cv_round (int): The cross-validation round number.
        """
        if self.verbosity >= 5:
            with self.log_lock:
                with open(self.log_file, "a") as f:
                    f.write(msg)
        if self.verbosity >= 4:
            self.telegram(msg)
        if self.verbosity >= 3:
            print(msg)


    def _save_data(self, model: Model, cross_validator: BaseCrossValidator, resampler: Resampler, scorer: Callable, cv_round: str, data: Tuple) -> None:
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
        assert type(data) is tuple, "Data must be a tuple."
        assert len(data) > 1, "Data must contain at least 2 elements."
        assert self.output_format in ["csv", "pickle", "both"], "Output type must be valid."

        with self.output_lock:
            cv_name = Storage.cross_validator_name(cross_validator)
            storage = Storage(self.records_dir, cv_name, str(resampler), str(model), scorer.__name__, self.tag)
            file_name = f"{self.output_name}_{cv_round}"
            df = DataFrame(data).transpose()
            df = df.set_axis(['index', 'actual', 'predicted', 'proba'], axis=1)

            if self.output_format == "both":
                storage.save_dataframe(df, file_name)
                storage.save_csv(df, file_name)
            elif self.output_format == "pickle":
                storage.save_dataframe(df, file_name)
            else:
                storage.save_csv(df, file_name)


    def _save_model(self, model: Model, cross_validator: BaseCrossValidator, resampler: Resampler, scorer: Callable) -> None:
        """
        Saves the provided model to a file.

        Parameters:
        model (Model): The model to be saved.
        resampler (Resampler): The resampler used to generate the model (labelling purposes).
        scorer (str): The scorer used to generate the model (labelling purposes).
        """
        with self.output_lock:
            print(f"Saving model for {model}_{resampler}_{scorer}")
            cv_name = Storage.cross_validator_name(cross_validator)
            Storage(self.records_dir, cv_name, str(resampler), str(model), scorer.__name__, self.tag).save_model(model)

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
            for scoring_method in scorers.values():
                for name, json in models.items():
                    json["scorer"] = scoring_method
                    assert " " not in name, "Model names cannot contain spaces."
                    assert "/" not in name or "\\" not in name, "Model names cannot contain slashes."
                    self.models.append(Model.from_json(name, json, disable_bayes_search=disable_bayes_search))
        elif type(models) == list:
            self.models = models

    def _load_resamplers(self, resampler_list: ResamplerList = None) -> None:
        """
        Loads the provided resamplers into Resampler objects.
        """
        resamplers: List[Resampler] = []
        if resampler_list is None:
            resampler_list = [None]
            
        for resampler in resampler_list:
            if resampler is None:
                resamplers.append(Resampler("none"))
            elif type(resampler) == Resampler:
                resamplers.append(Resampler(resampler=resampler.resampler, **resampler.params))
            else:
                resamplers.append(Resampler(resampler=resampler))
                
        self.resamplers = resamplers

    def _setup_output(self):
        """
        Sets up the output directory.
        """
        if not self.records_dir.exists():
            self.records_dir.mkdir()

        if not self.log_file.exists():
            self.log_file.touch()
            
    def _format_result(self, model, resampler, cross_validator, scorer, cv_round, score, tag):
            return f"""Model: {model}
                    CV: {cross_validator}
                    CV Round: {cv_round}
                    Resampler: {resampler}
                    Optimization: {scorer.__name__}
                    Score: {score}
                    Tag: {tag}
                    """
