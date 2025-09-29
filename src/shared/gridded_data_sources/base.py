from abc import ABC, abstractmethod

from pyproj import Transformer
import numpy as np
import xarray as xr
from IPython.display import display

from ..utils.grid_utils import Grid

class GriddedDataSource(ABC):
    def __init__(self):
        self.data = None
        self.grid = Grid.from_predefined(self.grid_id)
    

    # %% display methods
    def __repr__(self):
        return ""
    
    def _ipython_display_(self):
        display(self.data)


    # %% Grid transformation methods
    def regrid(self, target_grid, method='linear'):
        """Regrid data to target grid using specified interpolation method"""
        # create grid
        target_ds = target_grid.create_grid()

        # same grid for both source and target
        if self.grid == target_grid:
            return
        
        # same CRS, different grid
        elif self.grid.crs == target_grid.crs:
            self.data = self.data.interp({self.grid.coords[0]: target_ds[target_grid.coords[0]].values, self.grid.coords[1]: target_ds[target_grid.coords[1]].values}, method=method)

        # different CRS and grid
        else:
            # transform coordinates from target to source grid CRS
            transformer = Transformer.from_crs(target_grid.crs, self.grid.crs, always_xy=True)
            X, Y = np.meshgrid(target_ds[target_grid.coords[0]].values, target_ds[target_grid.coords[1]].values, indexing='ij')
            x, y = transformer.transform(X, Y)
            target_ds = target_ds.assign_coords({self.grid.coords[0]: (target_grid.coords, x), self.grid.coords[1]: (target_grid.coords, y)})

            # interpolate data onto the reprojected grid
            self.data = self.data.interp({self.grid.coords[0]: target_ds[self.grid.coords[0]], self.grid.coords[1]: target_ds[self.grid.coords[1]]}, method=method)
            self.data = self.data.drop_vars([self.grid.coords[0], self.grid.coords[1]])

        # update grid reference    
        self.grid = target_grid

    def modify_extent(self, new_extent):
        self.grid.modify_extent(new_extent)
        self.regrid(self.grid, method='linear')
    
    def modify_grid_cell_size(self, new_cell_size):
        self.grid.modify_grid_cell_size(new_cell_size)
        self.regrid(self.grid, method='linear')

    #%% Combination methods
    @classmethod
    def merge(cls, data_sources):
        for ds in data_sources[1:]:
            if data_sources[0].grid != ds.grid:
                raise ValueError("All datasets must have the same grid to be merged.")
            
        out = cls.__new__(cls)
        out.grid = data_sources[0].grid
        out.data = xr.merge(xr.align(*[ds.data for ds in data_sources], join='inner'))
        out.data.attrs = {}
        return out


    def __add__(self, other):
        return self.merge([self, other])
        

    # %% misc
    def copy(self):
        import copy
        return copy.deepcopy(self)
        

