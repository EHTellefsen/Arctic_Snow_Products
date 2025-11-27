# -- coding: utf-8 --
# split_dataset.py
""" Split dataset into training and testing sets based on time periods."""

# -- built-in libraries --
import yaml
from pathlib import Path

# -- third-party libraries  --
import pandas as pd
import numpy as np

#  -- custom modules  --

#########################################################################
# %% splits
def split_by_period(dataset, start_date, end_date):
    """Split dataset by time period."""
    mask = (dataset['time'] >= start_date) & (dataset['time'] < end_date)
    subset = dataset[mask]
    remaining = dataset[~mask]
    return subset, remaining

#########################################################################
if __name__ == "__main__":

    config_path = "configs/pipeline_configs/split_dataset.yaml"
    config = yaml.safe_load(open(config_path, "r"))

    # load full dataset
    df_full = pd.read_parquet(Path(config['input_directory']) / Path(config['input_dataset']))
    df_full = df_full[df_full['primary_id'].isin(config['allowed_primary_ids'])]
    df_full = df_full[df_full['secondary_id'].isin(config['allowed_secondary_ids'])]

    # apply weights
    df_full['weights'] = np.exp(df_full['lat'] - 70) / np.exp(18)
    df_full['weights'] = 0.9 - 0.8 * df_full['weights']
    df_full['weights'] = df_full['weights'] * df_full['num_samples']

    # Iterate over datasets to create splits

    remaining = df_full.copy()
    for test_set in config['test_sets'].keys():

        # split according to config
        ds = config['test_sets'][test_set]

        subset, remaining = split_by_period(remaining, ds['time_period']['start'], ds['time_period']['end'])

        # save subset
        subset.to_parquet(Path(config['output_directory']) / Path(ds['save_file_name']))

    # Save remaining as training dataset
    remaining.to_parquet(Path(config['output_directory']) / Path(config['train_set']['save_file_name']))