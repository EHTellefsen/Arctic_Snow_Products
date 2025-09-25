from abc import ABC, abstractmethod

class GriddedDataSource(ABC):
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    @abstractmethod
    def load(self):
        """Load raw data from file"""
        pass

    @abstractmethod
    def interpolate(self, target_grid):
        """Interpolate to a common grid or resolution"""
        pass