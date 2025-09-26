from abc import ABC, abstractmethod
from ..utils.grid_utils import Grid

class GriddedDataSource(ABC):
    def __init__(self):
        self.data = None
        self.grid = Grid.from_predefined(self.grid_id)

    @abstractmethod
    def load(self):
        """Load raw data from file"""
        pass

    @abstractmethod
    def regrid(self, target_grid):
        """Regrid to a common grid or resolution"""
        pass