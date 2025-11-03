# -- coding: utf-8 --
# linear_regression.py
"""Linear regression model implementation for snow depth prediction."""

# -- built-in modules --

# -- third-party modules --
import pandas as pd
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

# -- custom modules --
from .base import PixelPredictionModel

#######################################################
class LinearRegression(PixelPredictionModel):
    """Linear Regression model for snow depth prediction."""
    def __init__(self, 
                 input_features: list, 
                 target_feature: str, 
                 model_params: dict = {}, 
                 weight_feature: str = None):

        super().__init__(input_features=input_features, 
                         target_feature=target_feature, 
                         weight_feature=weight_feature, 
                         )
        
        self.model_params = model_params
        self.model = None
    

    def fit(self, data: pd.DataFrame):
        """Fit the Linear Regression model to the data."""
        X = data[self.input_features]
        y = data[self.target_feature]
        w = data[self.weight_feature] if self.weight_feature is not None else None

        self.model = SklearnLinearRegression(
            **self.model_params
            )
        
        self.model.fit(X, y, sample_weight=w)
        self.is_fitted = True
        return self
    

    def _predict(self, X):
        """Predict snow depth using the fitted model."""
        return self.model.predict(X)
