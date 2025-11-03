# --- coding: utf-8 ---
# kilic.py
"""Kilic et al. (2019) empirical snow depth model implementation."""

# -- built-in libraries --

# -- third-party libraries  --

# -- custom modules  --
from .base import PixelPredictionModel

########################################################################
class KilicModel(PixelPredictionModel):
    """Kilic et al. (2019) empirical snow depth model."""
    def __init__(self, target_feature, model_params: dict = {'a': 1.7701, 'b': 0.0175, 'c': -0.0280, 'd': 0.0041}):
        super().__init__(input_features=['6.9V','18V','36V'], target_feature=target_feature, weight_feature=None)
        self.is_fitted = True  # This model does not require fitting
        self.model_params = model_params
        
    def fit(self, data):
        """Fit the model to the data."""
        # No training needed for this empirical model
        return

    def _predict(self, X):
        """Predict snow depth using the Kilic et al. (2019) model."""
        a = self.model_params['a']
        b = self.model_params['b']
        c = self.model_params['c']
        d = self.model_params['d']
        return a + b*X['6.9V'] + c*X['18V'] + d*X['36V']

