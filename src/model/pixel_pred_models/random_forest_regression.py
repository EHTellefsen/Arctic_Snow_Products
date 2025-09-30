import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .base import PixelPredictionModel


class RandomForestRegression(PixelPredictionModel):
    def __init__(self, grid=None, model_params=None, oob_score: bool = True, random_state: int = None):
        super().__init__(model_params=model_params, grid=grid)
        self.oob_score = oob_score
        self.random_state = random_state
    

    def train(self, X: pd.DataFrame, y: pd.Series, weights=None):

        self.model = RandomForestRegressor(
            oob_score=self.oob_score,
            random_state=self.random_state,
            **self.model_params
            )
        
        self.model.fit(X, y, sample_weight=weights)
        self.input_features = self.model.feature_names_in_.tolist()
        self.target_feature = y.name
        return self
