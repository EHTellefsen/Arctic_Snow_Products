# --- coding: utf-8 ---
# cetb.py
"""CETB gridded data source implementation."""

# -- built-in modules --
from pathlib import Path

# -- third-party modules --
import xarray as xr

# -- custom modules --
from .base import GriddedDataSource

##############################################################################################
# %% CETB Scene class
class CETBScene(GriddedDataSource):
    """CETB Scene data source."""
    def __init__(self, filepaths):
        if isinstance(filepaths, str):
            filepaths = [filepaths]
        self.filepaths = filepaths
        self._extract_name_metadata()
        super().__init__(grid_id=self.grid_id)


    @classmethod
    def from_files(cls, filepaths):
        """Create a CETBScene instance from file paths."""
        instance = cls(filepaths)
        instance.load()
        return instance


    def load(self, keep_secondary_vars=False):
        """Load CETB data from the specified file paths."""
        self.data = xr.open_mfdataset(self.filepaths, 
                                      concat_dim='time', 
                                      combine='nested',
                                      data_vars = "minimal",
                                      chunks={"time": 50},
                                      engine='netcdf4')
        
        self.data = self.data.rename({'TB': self.channel})
        self.data = self.data.drop_vars([var for var in self.data.data_vars if var != self.channel] if not keep_secondary_vars else [])


    def _extract_name_metadata(self):
        """Extract metadata from CETB file naming convention."""
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
def load_CETB_data(files, grid):
    """Load and regrid CETB data from given files to the specified grid."""
    cetb_scenes = []
    for file in files:
        cetb_scene = CETBScene.from_files([file])
        cetb_scene.regrid(grid)
        cetb_scenes.append(cetb_scene)
    
    cetb_ds = GriddedDataSource.merge(cetb_scenes)
    return cetb_ds