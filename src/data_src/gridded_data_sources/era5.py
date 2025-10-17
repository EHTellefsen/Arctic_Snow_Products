from pathlib import Path
import zipfile
import tempfile
import yaml
import os
from collections import defaultdict

import xarray as xr
from glob import glob
from datetime import date

from .base import GriddedDataSource

##############################################################################################
# %% ERA5 Scene class
class ERA5Scene(GriddedDataSource):
    def __init__(self, filepaths, grid_id='ERA5_polar'):
        if isinstance(filepaths, str):
            filepaths = [filepaths]
        self.filepaths = filepaths
        self.grid_id = grid_id
        super().__init__(grid_id)

    @classmethod
    def from_files(cls, file_paths, grid_id='ERA5_polar'):
        instance = cls(file_paths)
        instance.load()
        return instance
    
    @classmethod
    def from_dir(cls, dir_path, grid_id='ERA5_polar'):
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
        self.data = xr.open_mfdataset(self.filepaths, 
                                combine='by_coords',
                                data_vars = "minimal",
                                engine='netcdf4')
        self.data = self.data.rename({'valid_time': 'time'})
        self.data = self.data.drop_vars(['number'])  # drop unnecessary variable


##############################################################################################
# %% Loading utilities
def map_ERA5_file_dates(directory, channels = None):
    """
    Creates a dictionary mapping each date (from filenames like ERA5_YYYYMMDD.nc)
    to a list of all matching file paths within the directory and its subdirectories.
    """
    ERA5_dict = yaml.safe_load(open('configs/ERA5_variable_dictionary.yaml'))
    month_file_map = defaultdict(list)

    for root, _, files in os.walk(directory):
        # try to detect YYYY-MM in directory name
        dir_name = os.path.basename(root)

        try:
            year, month = map(int, dir_name.split('_'))
            for file in files:
                if file.endswith('.nc'):
                    if channels is not None:
                        if not any(ERA5_dict[var]['filename'] in file for var in channels):
                            continue
                    full_path = os.path.join(root, file)
                    month_file_map[(year, month)].append(full_path)
        except ValueError:
            # skip directories not following YYYY-MM format
            continue

    # make lookup by date easy
    class MonthLookup(dict):
        def __getitem__(self, key):
            if isinstance(key, date):
                return super().__getitem__((key.year, key.month))
            return super().__getitem__(key)

    return MonthLookup(month_file_map)


def load_ERA5_data(files, grid):
    era5_ds = ERA5Scene.from_files(files)
    era5_ds.regrid(grid)
    return era5_ds