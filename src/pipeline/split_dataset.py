# -- coding: utf-8 --
# split_dataset.py
""""""

# -- built-in libraries --
import yaml
from pathlib import Path

# -- third-party libraries  --
import xarray as xr
from tqdm import tqdm
import pandas as pd
import numpy as np

#  -- custom modules  --
from src.utils.grid_utils import Grid
import src.data_src.point_data_sources as pds
import src.data_src.gridded_data_sources as gds
from src.utils.data_utils import DataMapping

#########################################################################
# %% splits
def split_by_period(dataset, start_date, end_date):
    """Split dataset by time period."""
    mask = (dataset.data['time'] >= start_date) & (dataset.data['time'] < end_date)
    subset = pds.GriddedPointDataSource(dataset.data[mask], dataset.grid)
    remaining = pds.GriddedPointDataSource(dataset.data[~mask], dataset.grid)
    return subset, remaining


def split_by_fraction(dataset, fraction, equalization_id):
    """Split dataset by fraction, optionally equalizing by secondary ID."""
    if equalization_id is not None:
        # Get unique secondary IDs
        ids = dataset.data[equalization_id].unique()
        val_indices = []
        for id in ids:
            id_mask = dataset.data[equalization_id] == id
            id_data = dataset.data[id_mask]
            id_val_size = int(len(id_data) * fraction)
            id_val_sample = id_data.sample(n=id_val_size, random_state=42)
            val_indices.extend(id_val_sample.index.tolist())
            
        subset = pds.GriddedPointDataSource(dataset.data.loc[val_indices], dataset.grid)
        remaining = pds.GriddedPointDataSource(dataset.data.drop(index=val_indices), dataset.grid)
    else:
        subset_size = int(len(dataset.data) * fraction)
        subset_data = dataset.data.sample(n=subset_size, random_state=42)
        subset = pds.GriddedPointDataSource(subset_data, dataset.grid)
        remaining = pds.GriddedPointDataSource(dataset.data.drop(index=subset_data.index), dataset.grid)

    return subset, remaining

#########################################################################
if __name__ == "__main__":

    config_path = "configs/pipeline_configs/split_dataset.yaml"
    config = yaml.safe_load(open(config_path, "r"))

    # load full dataset
    df_full = pd.read_parquet(Path(config['dataset']))

    # apply weights
    df_full['weights'] = np.exp(df_full['lat'] - 70) / np.exp(18)
    df_full['weights'] = 0.9 - 0.8 * df_full['weights']
    df_full['weights'] = df_full['weights'] * df_full['num_samples']

    # Iterate over datasets to create splits
    for dataset in config['datasets'].keys():
        if dataset not in ['train']:
            # split according to config
            ds = config['datasets'][dataset]
            if 'split_type' in ds and ds['split_type'] == 'periodic':
                subset, remaining = split_by_period(remaining, ds['period']['start'], ds['period']['end'])
            elif 'split_type' in ds and ds['split_type'] == 'fractional':
                subset, remaining = split_by_fraction(remaining, ds['split_params']['fraction'], equalization_id=ds['split_params']['equalization_id'])
            else:
                raise ValueError(f"Unsupported split type for dataset {dataset}")

            # save subset
            subset.to_parquet(Path(ds['save_directory']) / Path(ds['name']))

    # Save remaining as training dataset
    remaining.to_parquet(Path(config['datasets']['train']['save_directory']) / Path(config['datasets']['train']['name']))