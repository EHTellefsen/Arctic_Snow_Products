import xarray as xr
from .base import GriddedDataSource

class CETBScene(GriddedDataSource):
    def __init__(self):
        super().__init__()

    def load(self, data_vars = None):
        self.data = xr.open_mfdataset(self.filepath, 
                                      combine='nested',
                                      data_vars = data_vars,
                                      engine='netcdf4')
        return self.data

    def interpolate(self, target_grid):
        pass

    @classmethod
    def from_files():
        pass