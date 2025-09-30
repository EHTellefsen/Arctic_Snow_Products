from abc import ABC

class PixelPredictionModel(ABC):
    def __init__(self):
        self.parameters = None
        self.model = None
        self.grid = None

    def __repr__(self):
        print(f"PixelPredictionModel with parameters: {self.parameters}")
        return ""

