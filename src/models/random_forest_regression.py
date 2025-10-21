import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .base import PixelPredictionModel


class RandomForestRegression(PixelPredictionModel):
    def __init__(self, 
                 input_features: list, 
                 target_feature: str, 
                 weight_feature: str = None,
                 model_params: dict = None, 
                 oob_score: bool = False, 
                 random_state: int = None):

        super().__init__(input_features=input_features, 
                         target_feature=target_feature, 
                         weight_feature=weight_feature, 
                         model_params=model_params, 
                         )
        
        self.oob_score = oob_score
        self.random_state = random_state
    

    def fit(self, data: pd.DataFrame, model_params: dict = None):
        if model_params is not None:
            self.model_params = model_params

        X = data[self.input_features]
        y = data[self.target_feature]
        w = data[self.weight_feature] if self.weight_feature is not None else None

        self.model = RandomForestRegressor(
            oob_score=self.oob_score,
            random_state=self.random_state,
            **self.model_params
            )
        
        self.model.fit(X, y, sample_weight=w)
        return self
