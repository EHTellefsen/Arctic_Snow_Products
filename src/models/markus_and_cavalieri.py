# --- coding: utf-8 ---
# markus_and_cavalieri.py
"""Markus and Cavalieri empirical snow depth model implementation for AMSR."""

# -- built-in libraries --

# -- third-party libraries  --
import numpy as np

# -- custom modules  --
from .base import PixelPredictionModel

########################################################################
class MarkusAndCavalieriModel(PixelPredictionModel):
    """Markus and Cavalieri empirical snow depth model for AMSR."""
    def __init__(self, target_feature, model_params: dict = {'a':-2.34, 'b':-771}):
        super().__init__(input_features=['18V', '36V'], target_feature=target_feature, weight_feature=None)
        self.model_params = model_params
        self.is_fitted = True  # This model does not require fitting

    def fit(self, data):
        """Fit the model to the data."""
        # No training needed for this empirical model
        return

    def _predict(self, X):
        """Predict snow depth using the Markus and Cavalieri model."""
        a = self.model_params['a']
        b = self.model_params['b']

        sd = (a + b * (X['36V'] - X['18V']) / (X['36V'] + X['18V'])) * 1e-2  # [m]
        sd = np.where(sd < 0, 0, sd)
        return sd
    