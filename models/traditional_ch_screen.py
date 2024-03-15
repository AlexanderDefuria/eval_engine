import pandas as pd

class TraditionalScreen:
    def __init__(self, tsh_index: int, cutoff: float = 14.0) -> None:
        self.cutoff = cutoff
        self.tsh_index = tsh_index
        
    def predict(self, X):
        """
        Returns a dataframe of the patients that are predicted to have CH.
        
        Uses the TSH_I (RAW) column to determine if a patient has CH using the existing cutoff.
        """
        return [1 if x[self.tsh_index] >= self.cutoff else 0 for x in X]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        return None
