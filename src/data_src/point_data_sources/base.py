from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from pyproj import Transformer
from tqdm import tqdm

from src.utils.bucket_utils import drop_in_bucket_resample
from src.utils.grid_utils import Grid
# %%
#########################################################################################################################
class PointDataSource(ABC):
    def __init__(self, primary_id=None, secondary_id=None):
        self.data = None
        self.param_dict = None
        self.primary_id = primary_id
        self.secondary_id = secondary_id

    @abstractmethod
    def load(self):
        """Load raw data from file"""
        pass

    # %%
    def _preprocess_SD_df(self, df, param_dict):
        """
        Function for preprocessing snow depth dataframe to standardize column names and types.
        """
        pp_df = df[[param_dict['datetime'], self.param_dict['lat'], param_dict['lon'], param_dict['snow_depth']]]
        pp_df = df.rename(columns={
            param_dict['datetime']: 'time',
            param_dict['lat']: 'lat',
            param_dict['lon']: 'lon',
            param_dict['snow_depth']: 'snow_depth',
        })
        
        if isinstance(param_dict['snow_depth_uncertainty'],str):
            pp_df['snow_depth_uncertainty'] = df[param_dict['snow_depth_uncertainty']]
        elif isinstance(param_dict['snow_depth_uncertainty'],float):
            pp_df['snow_depth_uncertainty'] = param_dict['snow_depth_uncertainty']
        elif param_dict['snow_depth_uncertainty'] is None:
            pp_df['snow_depth_uncertainty'] = np.nan
        else:
            raise ValueError('param_dict["snow_depth_uncertainty"] must be str, float, or None')
        
        pp_df['time'] = pd.to_datetime(pp_df['time'], errors='coerce')
        pp_df = pp_df.dropna(subset=['time', 'lat', 'lon', 'snow_depth'])
        pp_df = pp_df[pp_df['snow_depth'] > 0]
        return pp_df[['time', 'lat', 'lon', 'snow_depth', 'snow_depth_uncertainty']].reset_index(drop=True)
    

    def get_preprocessed_SD_df(self):
        if self.param_dict is None:
            raise ValueError('Param_dict is uninitialized')
        elif self.data is None:
            raise ValueError('Data is not intialized')
        
        return self._preprocess_SD_df(self.data, self.param_dict)


    # %%
    def resample_bucket(self, target_grid, input_crs = "EPSG:4326", daily=True):
        df = self._preprocess_SD_df(self.data, self.param_dict)

        if isinstance(target_grid, str):
            target_grid = Grid.from_predefined(target_grid)
        elif not isinstance(target_grid, Grid):
            raise ValueError("target_grid must be a Grid object or a predefined grid ID string.")

        if daily:
            dates = df['time'].dt.date.unique()
            dfs = []
            for date in tqdm(dates, desc="Processing dates", unit="date"):
                cdf = df[df['time'].dt.date == date].copy()
                dfs.append(drop_in_bucket_resample(cdf, target_grid))
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = drop_in_bucket_resample(df, target_grid, input_crs=input_crs)

        df_name = pd.DataFrame({
            'primary_id': [self.primary_id] * len(df),
            'secondary_id': [self.secondary_id] * len(df)
        })
        df = pd.concat([df_name, df.reset_index(drop=True)], axis=1)

        return GriddedPointDataSource(df, target_grid)


# %%
#########################################################################################################################
class GriddedPointDataSource:
    def __init__(self, data, grid):
        self.grid = grid
        self.data = data

    # %%
    def to_parquet(self, filepath):
        self.data.to_parquet(filepath, index=False)

    def to_csv(self, filepath):
        self.data.to_csv(filepath, index=False)
    
    @classmethod
    def load(cls, filepath, grid):
        if filepath.endswith('.parquet'):
            data = pd.read_parquet(filepath)
        elif filepath.endswith('.csv'):
            data = pd.read_csv(filepath)
        else:
            raise ValueError("Unsupported file format. Use .parquet or .csv")
        return cls(data, grid)


    # %%
    @classmethod
    def merge_sources(cls, sources):
        if not all(isinstance(source, GriddedPointDataSource) for source in sources):
            raise ValueError("All sources must be GriddedPointDataSource objects.")
        if not all(source.grid == sources[0].grid for source in sources):
            raise ValueError("All sources must have the same grid.")
        if not all(source.data.columns.tolist() == sources[0].data.columns.tolist() for source in sources):
            raise ValueError("All sources must have the same data columns.")
        
        combined_data = pd.concat([source.data for source in sources], ignore_index=True)
        return cls(combined_data, sources[0].grid)


    def __add__(self, other):
        return self.merge_sources([self, other])


