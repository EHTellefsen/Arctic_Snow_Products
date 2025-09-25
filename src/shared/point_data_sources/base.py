from abc import ABC, abstractmethod

class PointDataSource(ABC):
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    @abstractmethod
    def load(self):
        """Load raw data from file"""
        pass

    @abstractmethod
    def resample_bucket(self, target_grid):
        """Bucket Resample to a common grid or resolution"""
        pass