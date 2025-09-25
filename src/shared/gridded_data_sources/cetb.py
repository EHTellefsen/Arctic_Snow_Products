import xarray as xr
from .base import DataSource

class CETBScene(DataSource):
    def load(self):
        self.data = xr.open_mfdataset(self.filepath, 
                                      combine='nested',
                                      chunks={"time": 50}, 
                                      data_vars = ['TB'],
                                      engine='netcdf4')
        return self.data

    def interpolate(self, target_grid):
        pass