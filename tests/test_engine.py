import shutil
import unittest

from skopt.space import Real, Integer, Categorical
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.datasets import load_breast_cancer
from skopt.space import Real, Categorical
from imblearn.over_sampling import SMOTE
from ..engine import *



class TestEngineIntegration(unittest.TestCase):
    def setUp(self):
        self.records_dir = Path(__file__).parent / "test_records"
        self.success = False
        
    def tearDown(self):
        shutil.rmtree(self.records_dir)

    def test_run(self):
        # Load a simple dataset
        iris = load_breast_cancer()
        X, y = iris['data'], iris['target']
        folds = 5

        # Define a simple model
        models = {
            "Balanced Bagging": {
                "estimator": BalancedBaggingClassifier(),
                "search_spaces": {
                    "warm_start": Categorical([True, False]),
                    "bootstrap": Categorical([True, False]),
                    "n_estimators": Integer(5, 50),
                    "sampling_strategy": Real(0.01, 1, prior="log-uniform"), # Controls sampling rate for RUS.
                },
            },
        }

        resamplers = [
            None,
            SMOTE()
        ]

        scorers = {
            "f1-pos": f1_score_pos,
        }


        # Create an instance of Engine
        engine = Engine(
            models=models,
            resamplers=resamplers,
            scorers=scorers,
            X=X,
            y=y,
            max_workers=1,
            records_dir=self.records_dir,
            output_name="test_results",
            output_format="pickle",
            overwrite=True,
            disable_bayes_search=True,
            cross_validator=StratifiedKFold(n_splits=folds, shuffle=True, random_state=42),
        )

        # Run the engine
        engine.run()

        # Check that the output file was created
        for i in range(folds):
            self.assertTrue((engine.records_dir / "None" / "Balanced Bagging" / "f1_score_pos" / f"test_results_cv_{i}.pkl").exists())
            self.assertTrue((engine.records_dir / "SMOTE" / "Balanced Bagging" / "f1_score_pos" / f"test_results_cv_{i}.pkl").exists())
        