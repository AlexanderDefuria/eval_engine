from pathlib import Path
from typing import Callable, Dict, Optional

from pandas import DataFrame


class Resampler:
    """
    A resampler to be used in the training process.

    This is a wrapper around an imblearn resampler, and it contains the search space for the resampler's hyperparameters.
    It exists to make the training process more convenient, and to allow for serialization and deserialization of the resampler.

    Parameters:
    name (str): The name of the resampler (labelling purposes usually).
    resampler (BaseResampler): The imblearn resampler to be used in the training process.
    params (Dict): The parameters for the resampler.
    """

    def __init__(self, 
                name: Optional[str] = None, 
                resampler: Optional[Callable] = None, 
                **kwargs):
        self.name = name
        self.resampler = resampler
        self.params = kwargs
            
        if not self.name:
            self.name = str(self.resampler).strip("() ").split(".")[-1]

    def __str__(self):
        return self.name

    def __repr__(self):
        """
        Returns a string representation of the resampler that can be used to reconstruct it later.
        This is useful to store in the results file metadata to allow for easy reconstruction of the resampler.
        """
        return str(self.__dict__)
    
    def __call__(self, X, y, output: Optional[Path] = None):
        # TODO ? - If using outlier detection classifier, must only train on positive class.
        resampled = self.resampler.fit_resample(X, y) if self.resampler else (X, y)

        if output:
            assert output.is_file, "Output path must be a file"
            output.mkdir(parents=True, exist_ok=True)
            data = DataFrame(resampled).transpose()
            data.columns = ["X", "y"]
            if ".pkl" in output.suffix or ".pickle" in output.suffix:
                data.pickle(output)
            else:
                data.to_csv(output)
        
        return resampled
    
    @classmethod
    def from_json(cls, json: Dict[str, str]):
        assert "name" in json, "Resampler JSON must contain a name"
        assert "params" in json, "Resampler JSON must contain params"
        assert json.keys() == {"name", "params"}, "Resampler JSON must only contain name and params"

        return cls(**json)
    

class TestResampler:

    def test_serde():
        # TODO - Implement
        pass
