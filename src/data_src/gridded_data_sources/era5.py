# --- coding: utf-8 ---
# era5.py
"""ERA5 gridded data source implementation."""

# -- built-in modules --
from pathlib import Path
import zipfile
import tempfile

# -- third-party modules --
import xarray as xr

# -- custom modules --
from .base import GriddedDataSource

##############################################################################################
# %% ERA5 Scene class
class ERA5Scene(GriddedDataSource):
    """ERA5 Scene data source."""
    def __init__(self, filepaths, grid_id='ERA5_polar'):
        if isinstance(filepaths, str):
            filepaths = [filepaths]
        self.filepaths = filepaths
        self.grid_id = grid_id
        super().__init__(grid_id)

    @classmethod
    def from_files(cls, file_paths, grid_id='ERA5_polar'):
        """Create an ERA5Scene instance from file paths."""
        instance = cls(file_paths, grid_id=grid_id)
        instance.load()
        return instance
    
    @classmethod
    def from_dir(cls, dir_path, grid_id='ERA5_polar'):
        """Create an ERA5Scene instance from a directory containing NetCDF files."""
        if isinstance(dir_path, str):
            dir_path = [dir_path]
        
        file_paths = []
        for dp in dir_path:
            dp = Path(dp)
            if not dp.is_dir():
                raise ValueError(f"Provided path {dp} is not a directory.")
            
            file_paths.extend(list(Path(dp).rglob("*.nc")))
        instance = cls(file_paths, grid_id=grid_id)
        instance.load()
        return instance

    
    @classmethod
    def from_zip(cls, dir_path, grid_id='ERA5_polar'):
        """Create an ERA5Scene instance from a ZIP archive containing NetCDF files."""
        # !!!Experimental from GPT - needs testing!!!
        if isinstance(dir_path, str):
            dir_path = [dir_path]

        file_paths = []
        for dp in dir_path:
            dp = Path(dp)
            if not dp.is_file() or dp.suffix.lower() != ".zip":
                raise ValueError(f"Provided path {dp} is not a valid ZIP file.")

            # Handle ZIP archive
            with zipfile.ZipFile(dp, "r") as zf:
                with tempfile.TemporaryDirectory() as tmpdir:
                    zf.extractall(tmpdir)
                    file_paths.extend(list(Path(tmpdir).rglob("*.nc")))

        instance = cls(file_paths, grid_id=grid_id)
        instance.load()
        return instance
    

    def load(self):
        """Load ERA5 data from the specified file paths."""
        self.data = xr.open_mfdataset(self.filepaths, 
                                combine='by_coords',
                                data_vars = "minimal",
                                engine='netcdf4',
                                chunks={"valid_time": 50})
        self.data = self.data.rename({'valid_time': 'time'})
        self.data = self.data.drop_vars(['number'])  # drop unnecessary variable


##############################################################################################
# %% Loading utilities
def load_ERA5_data(files, grid):
    """Load and regrid ERA5 data from given files to the specified grid."""
    era5_ds = ERA5Scene.from_files(files)
    era5_ds.regrid(grid)
    return era5_ds