import pandas as pd
from .base import PixelPredictionModel

class MeanBaselineModel(PixelPredictionModel):
    def __init__(self, target_feature: str, weight_feature: str = None):
        super().__init__(input_features=[], target_feature=target_feature, weight_feature=weight_feature)

    def fit(self, data: pd.DataFrame):
        if self.weight_feature is not None:
            self.mean = (data[self.target_feature]*data[self.weight_feature]).sum()/data[self.weight_feature].sum()
        else:
            self.mean = data[self.target_feature].mean()
        self.is_fitted = True
        return self
    
    def _predict(self, X):
        return pd.Series(self.mean, index=X.index)
