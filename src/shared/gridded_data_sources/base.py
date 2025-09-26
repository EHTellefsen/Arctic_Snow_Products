from abc import ABC, abstractmethod

class GriddedDataSource(ABC):
    def __init__(self):
        self.data = None
        self.grid = None

    @abstractmethod
    def load(self):
        """Load raw data from file"""
        pass

    @abstractmethod
    def interpolate(self, target_grid):
        """Interpolate to a common grid or resolution"""
        pass