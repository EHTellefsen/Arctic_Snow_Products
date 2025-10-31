import pandas as pd
from .base import PixelPredictionModel

class MeanBaseline(PixelPredictionModel):
    def __init__(self, target_feature: str):
        super().__init__(input_features=[], target_feature=target_feature, weight_feature=None)

    def fit(self, data: pd.DataFrame):
        self.mean = data[self.target_feature].mean()
        self.is_fitted = True
        return self
    
    def _predict(self, X):
        return pd.Series(self.mean, index=X.index)
