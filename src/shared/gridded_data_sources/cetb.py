import os
from pathlib import Path

import xarray as xr

from .base import GriddedDataSource
from ..utils.grid_utils import Grid


class CETBScene(GriddedDataSource):

    def __init__(self, filepaths):
        if isinstance(filepaths, str):
            filepaths = [filepaths]
        self.filepaths = filepaths
        self._extract_name_metadata()
        super().__init__()


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



    def regrid(self, target_grid):
        target_ds = target_grid.create_grid()
        self.data = self.data.interp(x=target_ds.x, y=target_ds.y, method='linear')
        self.grid = target_grid



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


