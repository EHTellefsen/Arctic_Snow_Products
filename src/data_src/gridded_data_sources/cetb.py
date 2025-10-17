from pathlib import Path
import os
from collections import defaultdict

import xarray as xr
from datetime import datetime

from .base import GriddedDataSource

##############################################################################################
# %% CETB Scene class
class CETBScene(GriddedDataSource):
    def __init__(self, filepaths):
        if isinstance(filepaths, str):
            filepaths = [filepaths]
        self.filepaths = filepaths
        self._extract_name_metadata()
        super().__init__(grid_id=self.grid_id)


    @classmethod
    def from_files(cls, filepaths):
        instance = cls(filepaths)
        instance.load()
        return instance


    def load(self, keep_secondary_vars=False):
        self.data = xr.open_mfdataset(self.filepaths, 
                                      combine='nested',
                                      data_vars = "minimal",
                                      engine='netcdf4')
        
        self.data = self.data.rename({'TB': self.channel})
        self.data = self.data.drop_vars([var for var in self.data.data_vars if var != self.channel] if not keep_secondary_vars else [])


    def _extract_name_metadata(self):
        tokens = [Path(file).stem.split('_') for file in self.filepaths]
        if len({len(i) for i in tokens}) != 1:
            raise ValueError("Inconsistent token lengths in provided filepaths.")

        name_convention_dict = {
            "dataset_id": 0,
            "algorithm": 1,
            "grid_name": 2,
            "grid_resolution": 3,
            "platform": 4,
            "sensor": 5,
            "passing": 6,
            "channel": 7,
            "version": 10
        }

        for name, idx in name_convention_dict.items():
            if len({token[idx] for token in tokens}) != 1:
                raise ValueError(f"Inconsistent {name} in provided filepaths (token {idx}).")
            setattr(self, name, tokens[0][idx])
        self.grid_id = f"{self.grid_name}_{self.grid_resolution}"


##############################################################################################
# %% Loading utilities
def map_CETB_files(directory, channels=None):
    """
    Creates a dictionary mapping each date (from filenames like CETB_YYYYMMDD.nc)
    to a list of all matching file paths within the directory and its subdirectories.
    """
    date_file_map = defaultdict(list)

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.nc'):
                if channels is not None:
                    if Path(file).stem.split('_')[7] not in channels:
                        continue
                try:
                    # extract date assuming format CETB_YYYYMMDD.nc
                    date_str = file.split('_')[8]
                    date = datetime.strptime(date_str, '%Y%m%d').date()
                    full_path = os.path.join(root, file)
                    date_file_map[date].append(full_path)
                except (IndexError, ValueError):
                    continue

    return dict(date_file_map)


def load_CETB_data(files, grid):
    cetb_scenes = []
    for file in files:
        cetb_scene = CETBScene.from_files([file])
        cetb_scene.regrid(grid)
        cetb_scenes.append(cetb_scene)
    
    cetb_ds = GriddedDataSource.merge(cetb_scenes)
    return cetb_ds