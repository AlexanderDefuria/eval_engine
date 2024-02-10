from typing import Callable, Dict, Optional, Self
from skopt import BayesSearchCV
from skopt.space import Dimension
import numpy as np
from sklearn.base import BaseEstimator
from runtime.evaluation import accuracy


type SearchSpace = Dict[str, Dimension]
type ModelDict = dict[str, BaseEstimator | SearchSpace]


"""EXAMPLES

EXAMPLE_SEARCH_SPACE: SearchSpace = {
    "n_estimators": Integer(100, 1000),
    "max_depth": Integer(1, 10),
    "min_samples_split": Integer(2, 10),
    "min_samples_leaf": Integer(1, 10),
    "max_features": Categorical(["auto", "sqrt", "log2"]),
    "bootstrap": Categorical([True, False]),
}
EXAMPLE_MODEL_LIST = {
    "RandomForestClassifier": {
        "estimator": RandomForestClassifier,
        "search_spaces": EXAMPLE_SEARCH_SPACE,
    }
}
"""

class Model:
    """
    A model to be used in the training process.

    This is a wrapper around a scikit-learn estimator, and it contains the search space for the estimator's hyperparameters.
    It exists to make the training process more convenient, and to allow for serialization and deserialization of the model.
    
    Parameters:
    name (str): The name of the model (labelling purposes usually).
    estimator (BaseEstimator): The scikit-learn estimator to be used in the training process.
    search_spaces (SearchSpace): The search space for the estimator's hyperparameters.
    """
    def __init__(self, name: str, 
                 estimator: BaseEstimator, 
                 search_spaces: SearchSpace, 
                 scorer: Optional[Callable | str] = None,
                 **kwargs) -> None:
        self.estimator = estimator
        self.search_spaces = search_spaces
        self.name = name
        self.scorer = scorer if scorer else accuracy

        if self.search_spaces is not None:
            # print("Using BayesSearchCV")
           #  print("Scorer:", scorer)
            self.model = BayesSearchCV(estimator=self.estimator, 
                                    search_spaces=self.search_spaces,
                                    scoring=scorer, 
                                    cv=5,
                                    n_jobs=5,
                                    n_iter=10,
                                    **kwargs)
        else:
            self.model = self.estimator
            self.model.scorer_ = self.scorer


    def train(self, X, y, **kwargs) -> None:
        np.int = np.int64 # https://github.com/scikit-optimize/scikit-optimize/issues/1171#issuecomment-1584561856
        self.model.fit(X, y, **kwargs)


    def predict(self, X) -> np.array:
        np.int = np.int64 # https://github.com/scikit-optimize/scikit-optimize/issues/1171#issuecomment-1584561856
        return self.model.predict(X)
    

    def score(self, X, y) -> float:
        # TODO - Could be updated to use a instance of a `Scorer` class? This would mean we can just pass a scorer in similarly to `Resampler`
        # I don't think this is necessary in retrospect but it is possible.

        # Note the scorer can be a string or a callable. 
        # If it is a callable, it is assumed to be a function that takes the model, X, and y as parameters.
        if type(self.scorer) is not str:
            return self.scorer(self.model, X, y)
        return self.model.score(X, y)
    

    def __str__(self) -> str:
        return self.name
    

    def __repr__(self) -> str:
        """
        Returns a string representation of the model that can be used to reconstruct it later.
        This is useful to store in the results file metadata to allow for easy reconstruction of the model.
        """
        return str(self.__dict__)
    

    @classmethod
    def from_json(cls, name:str, json: ModelDict, disable_bayes_search: bool = False) -> Self:
        # Standardize the JSON

        if "search_spaces" not in json or disable_bayes_search:
            # Ensure that the search spaces are not used if they are not provided but they are defined.
            json["search_spaces"] = None

        if "scorer" not in json:
            json["scorer"] = None

        assert "estimator" in json, "Model JSON must contain an estimator"
        assert hasattr(json["estimator"], "fit") and callable(json["estimator"].fit), "Estimator must have a fit method"
        assert hasattr(json["estimator"], "predict") and callable(json["estimator"].predict), "Estimator must have a predict method"
        # assert hasattr(json["estimator"], "score") and callable(json["estimator"].score), "Estimator must have a score method"
        assert json.keys() == {"estimator", "search_spaces", "scorer"}, "Model JSON must only contain estimator, search_spaces, and scorer"

        return cls(name, json["estimator"], json["search_spaces"], json["scorer"])
    

class TestModel:
    def test_serde(self):
        # TODO - Add test for serialization and deserialization
        assert True
