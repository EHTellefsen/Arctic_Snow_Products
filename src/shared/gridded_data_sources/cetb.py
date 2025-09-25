import xarray as xr
from .base import GriddedDataSource

class CETBScene(GriddedDataSource):
    def __init__(self, filepath):
        super().__init__(filepath)

    def load(self):
        self.data = xr.open_mfdataset(self.filepath, 
                                      combine='nested',
                                      chunks={"time": 50}, 
                                      data_vars = ['TB'],
                                      engine='netcdf4')
        return self.data

    def interpolate(self, target_grid):
        pass