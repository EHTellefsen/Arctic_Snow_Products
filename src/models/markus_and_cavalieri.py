import numpy as np

from .base import PixelPredictionModel

class MarkusAndCavalieriModel(PixelPredictionModel):
    def __init__(self, model_params: dict = {'a':-2.34, 'b':771}):
        super().__init__(input_features=['18V', '36V'], target_feature=None, weight_feature=None)
        self.model_params = model_params
        self.is_fitted = True  # This model does not require fitting

    def fit(self, data):
        # No training needed for this empirical model
        return

    def _predict(self, X):
        a = self.model_params['a']
        b = self.model_params['b']

        sd = (a + b * (X['36V'] - X['18V']) / (X['36V'] + X['18V'])) * 1e-2  # [m]
        sd = np.where(sd < 0, 0, sd)
        return sd
    