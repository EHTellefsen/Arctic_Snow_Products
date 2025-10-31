from .base import PixelPredictionModel

class KilicModel(PixelPredictionModel):
    def __init__(self, model_params: dict = {'a': 1.7701, 'b': 0.0175, 'c': -0.0280, 'd': 0.0041}):
        
        super().__init__(input_features=['6.9V','18V','36V'], target_feature=None, weight_feature=None)
        self.is_fitted = True  # This model does not require fitting
        self.model_params = model_params
        
    def fit(self, data):
        # No training needed for this empirical model
        return

    def _predict(self, X):
        a = self.model_params['a']
        b = self.model_params['b']
        c = self.model_params['c']
        d = self.model_params['d']
        return a + b*X['6.9V'] + c*X['18V'] + d*X['36V']

