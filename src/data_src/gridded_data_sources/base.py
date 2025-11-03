# -- coding: utf-8 --
# base.py
"""Base class for gridded data sources."""

# -- built-in modules --
import pickle

# -- third-party modules --
from pyproj import Transformer
import numpy as np
import xarray as xr
from IPython.display import display

# -- custom modules --
from src.utils.grid_utils import Grid

#######################################################
class GriddedDataSource:
    """Base class for gridded data sources."""
    def __init__(self, grid_id: str):
        self.data = None
        self.grid = Grid.from_predefined(grid_id)
    
    # %% display methods
    def __repr__(self):
        """String representation of the data source."""
        return ""
    
    def _ipython_display_(self):
        """Display the xarray dataset in Jupyter notebooks."""
        display(self.data)
    
    # %% loading and saving methods
    @classmethod
    def from_xarray(cls, data: xr.Dataset, grid: Grid):
        """Create a GriddedDataSource from an xarray Dataset and a Grid."""
        out = cls.__new__(cls)
        out.grid = grid
        out.data = data
        return out
    
    @classmethod
    def from_pickle(cls, path: str):
        """Load a GriddedDataSource from a pickle file."""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise ValueError(f"Loaded object is not of type {cls.__name__}")
        return model

        # %% Save and Load
    def save(self, path: str): 
        """Save the GriddedDataSource to a pickle file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def to_netcdf(self, path: str):
        """Save the xarray Dataset to a NetCDF file."""
        self.data.to_netcdf(path)

    # %% Grid transformation methods
    def regrid(self, target_grid: Grid, method: str = 'linear'):
        """Regrid dataset to target grid using xarray interpolation.
        !!! Note: method works for CETB and ERA5 data only as of now. Need to make more universal later. !!!
        """

        if self.grid == target_grid:
            return
        
        # same CRS, different grid
        target_ds = target_grid.create_grid()
        if self.grid.crs == target_grid.crs:
            self.data = self.data.interp_like(target_ds, method=method)

        # different CRS and grid
        else:
            # transform coordinates from target to source grid CRS
            transformer = Transformer.from_crs(target_grid.crs, self.grid.crs, always_xy=True)
            X, Y = np.meshgrid(target_ds[target_grid.coords[0]].values, target_ds[target_grid.coords[1]].values, indexing='ij')
            x, y = transformer.transform(X, Y)
            target_ds = target_ds.assign_coords({self.grid.coords[0]: (target_grid.coords, x), self.grid.coords[1]: (target_grid.coords, y)})

            # # fix longitude wrapping issue for global datasets
            if self.grid.coords[0] == 'longitude':
                lon = self.data[self.grid.coords[0]]
                new_lon = np.concatenate([lon, lon + 360])  # or lon + 360 if in 0–360 system

                # Concatenate data and assign new longitude coordinate
                self.data = xr.concat([self.data, self.data], dim="longitude")
                self.data = self.data.assign_coords(longitude=new_lon)

            # interpolate data onto the reprojected grid
            self.data = self.data.interp({self.grid.coords[0]: target_ds[self.grid.coords[0]], self.grid.coords[1]: target_ds[self.grid.coords[1]]}, method=method)
            self.data = self.data.drop_vars([self.grid.coords[0], self.grid.coords[1]])

        # update grid reference    
        self.grid = target_grid
        self.data = self.data.transpose('time', self.grid.coords[1], self.grid.coords[0])

    def modify_extent(self, new_extent):
        self.grid.modify_extent(new_extent)
        self.regrid(self.grid, method='linear')
    
    def modify_grid_cell_size(self, new_cell_size):
        self.grid.modify_grid_cell_size(new_cell_size)
        self.regrid(self.grid, method='linear')

    #%% Combination methods
    @classmethod
    def merge(cls, data_sources):
        """Merge multiple GriddedDataSource objects into one."""
        for ds in data_sources[1:]:
            if data_sources[0].grid != ds.grid:
                raise ValueError("All datasets must have the same grid to be merged.")
            
        out = cls.__new__(cls)
        out.grid = data_sources[0].grid
        out.data = xr.merge(xr.align(*[ds.data for ds in data_sources], join='inner'))
        out.data.attrs = {}
        return out


    def __add__(self, other):
        """Combine two GriddedDataSource objects by merging their datasets."""
        return self.merge([self, other])
        

    # %% misc
    def copy(self):
        """Create a deep copy of the GriddedDataSource."""
        import copy
        return copy.deepcopy(self)
        

