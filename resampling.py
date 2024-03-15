from pathlib import Path
from typing import Callable, Optional, List

type ResamplerList = Optional[List[Resampler | None | Callable]]

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
        return self.name if self.name else str(self.resampler)

    def __repr__(self):
        """
        Returns a string representation of the resampler that can be used to reconstruct it later.
        This is useful to store in the results file metadata to allow for easy reconstruction of the resampler.
        """
        return str(self.__dict__)
    
    def __call__(self, X, y, output: Optional[Path] = None):
        # TODO ? - If using outlier detection classifier, must only train on positive class.

        # Make sure the resampler can call fit_resample if it is defined as an actual resampler
        if self.resampler is not None:
            assert hasattr(self.resampler, "fit_resample"), "Resampler must have a fit_resample method"
            assert callable(self.resampler.fit_resample), "Resampler's fit_resample method must be callable"
        # Make sure the resampler can be called with the given parameters
        resampled = self.resampler.fit_resample(X, y) if self.resampler else (X, y)

        return resampled
