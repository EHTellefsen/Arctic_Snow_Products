import pickle

from .base import PixelPredictionModel

class RandomForestRegression(PixelPredictionModel):
    def __init__(self):
        super().__init__()
    
    def load(self, model_path: str):
        self.model = pickle.load(open(model_path,'rb'))
        self.parameters = self.model.feature_names_in_ 

    def train(self, X, y, weights=None, model_params=None):
        pass

    def cross_validate(self, X, y, cv=5, weights=None, model_params=None):
        pass