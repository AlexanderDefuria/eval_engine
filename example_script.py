from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_breast_cancer
from skopt.space import Real, Integer, Categorical
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd
from engine import *

# df = pd.read_csv("/home/chml/whowe/training_all_July15.csv")
# X = df.drop("Definitive Diagnosis", axis=1).drop("Episode", axis=1).values
# y = df["Definitive Diagnosis"].values
X, y = load_breast_cancer(return_X_y=True)

MODELS = {
    "Balanced Bagging": {
        "estimator": BalancedBaggingClassifier(),
        "search_spaces": {
            "warm_start": Categorical([True, False]),
            "bootstrap": Categorical([True, False]),
            "n_estimators": Integer(5, 50),
#           "sampling_strategy": Real(0.01, 1, prior="log-uniform"), # Controls sampling rate for RUS.
        },
    },
}

RESAMPLERS = [
    None,
    SMOTE(),
]

METRICS = {
    "f1-pos": f1_score_pos,
    "f1-neg": f1_score_neg,
    "accuracy": accuracy,
}

engine = Engine(MODELS, 
                RESAMPLERS, 
                METRICS, 
                X=X,
                y=y,
                max_workers=40,
                verbosity=5,
                overwrite=True)

engine.run()
