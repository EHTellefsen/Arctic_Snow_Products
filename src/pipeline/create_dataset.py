# -- coding: utf-8 --
# create_dataset.py
"""Script to create training dataset by merging point data sources with gridded data sources."""

# -- built-in libraries --
import yaml
from pathlib import Path
import logging

# -- third-party libraries  --
import xarray as xr
import pandas as pd
from tqdm import tqdm

#  -- custom modules  --
from src.utils.grid_utils import Grid
import src.data_src.point_data_sources as pds
import src.data_src.gridded_data_sources as gds
from src.utils.data_utils import DataMapping

# setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#########################################################################
# %% Point data sources
def load_point_data_source(filepath, secondary_id, retracker = 'LARM-smoothed'):
    """Load point data source based on secondary_id."""
    if secondary_id in ['2020-2021', '2021-2022']:
        return pds.C2I(filepath, secondary_id=secondary_id, retracker=retracker)
    elif secondary_id == 'IDCSI4':
        return pds.OIB_IDCSI4(filepath)
    elif secondary_id == 'QuickLook':
        return pds.OIB_QL(filepath)
    elif secondary_id == 'ICEBIRD':
        return pds.AEM_AWI_ICEBIRD(filepath)
    elif secondary_id == 'PAMARCMIP':
        return pds.AEM_AWI_PAMARCMIP(filepath)
    else:
        raise ValueError(f"Unsupported secondary_id: {secondary_id}")


def grid_and_merge_point_data_sources(sources, target_grid_id):
    """Resample point data sources to target grid and merge them."""
    gridded_sources = []
    for source in sources:
        gridded_sources.append(source.resample_bucket(target_grid_id))
    
    # Merge all gridded sources into one
    return pds.base.GriddedPointDataSource.merge_sources(gridded_sources)


# %% Sample from gridded sources
def sample_from_CETB(point_source, gridded_source):
    """Sample gridded CETB data at point source locations."""
    points = xr.Dataset({
        "time": (("points",), point_source.data['time'].astype('<M8[ns]').values),
        "x":   (("points",), point_source.data['x'].values),
        "y":  (("points",), point_source.data['y'].values),
    })

    match = gridded_source.data.interp(
        time=points["time"],
        x=points["x"],
        y=points["y"],
        method='linear'
    )

    sampled_df = point_source.data.copy()
    for var in match.data_vars:
        sampled_df[var] = match[var].values
    
    return pds.GriddedPointDataSource(sampled_df, point_source.grid)


def sample_from_ERA5(point_source, gridded_source):
    """Sample gridded ERA5 data at point source locations."""
    points = xr.Dataset({
        "time": (("points",), point_source.data['time'].astype('<M8[ns]').values),
        "latitude":   (("points",), point_source.data['lat'].values),
        "longitude":  (("points",), point_source.data['lon'].values),
    })

    match = gridded_source.data.interp(
        time=points["time"],
        latitude=points["latitude"],
        longitude=points["longitude"],
        method="linear"
    )

    sampled_df = point_source.data.copy()
    for var in match.data_vars:
        sampled_df[var] = match[var].values
    
    return pds.GriddedPointDataSource(sampled_df, point_source.grid)

#########################################################################
# %% run
if __name__ == "__main__":
    logger.info("Starting dataset creation...")
    config_path = "configs/pipeline_configs/create_dataset.yaml"
    config = yaml.safe_load(open(config_path, "r"))

    target_grid = Grid.from_predefined(config['grid'])

    # %% make point data dataframe
    # Load point data sources
    logger.info("Loading point data sources...")
    point_sources = []
    for source_cfg in tqdm(config['point_data_sources'], desc="Loading point data sources"):
        for sid_cfg in config['point_data_sources'][source_cfg]['secondary_id']:
            source = load_point_data_source(
                filepath=config['point_data_sources'][source_cfg]['secondary_id'][sid_cfg]['filepath'],
                secondary_id=config['point_data_sources'][source_cfg]['secondary_id'][sid_cfg]['id'],
                retracker=config['point_data_sources'][source_cfg]['retracker']
            )
            point_sources.append(source)
    
    # Grid and merge point data sources
    logger.info("Gridding and merging point data sources...")
    gridded_point_data = grid_and_merge_point_data_sources(point_sources, target_grid)

    # %% match to gridded data sources
    # CETB
    unique_dates = gridded_point_data.data['time'].dt.date.unique()
    needed_dates = list(set(list(unique_dates) + list(unique_dates+pd.Timedelta(days=1)) + list(unique_dates-pd.Timedelta(days=1))))

    logger.info("Matching to CETB data...")
    CETB_mapping = DataMapping(config['CETB']['directory'], dataset='CETB')
    for i, channel in enumerate(config['CETB']['channels']):
        logger.info(f"Processing CETB channel: {channel}" + f" ({i+1}/{len(config['CETB']['channels'])})")
        files = CETB_mapping.get_by_channel(channel)['filename'].tolist()
        files = CETB_mapping.get_by_channel(channel)

        needed_files = files[pd.to_datetime(files['date']).isin(pd.to_datetime(needed_dates))]['filename'].tolist()
        scene = gds.CETBScene.from_files(needed_files)
        match = sample_from_CETB(gridded_point_data, scene)
        gridded_point_data.data[channel] = match.data[channel]


    # ERA5
    logger.info("Matching to ERA5 data...")
    ERA5_mapping = DataMapping(config['ERA5']['directory'], dataset='ERA5')
    for i, channel in enumerate(config['ERA5']['channels']):
        logger.info(f"Processing ERA5 channel:{channel}" + f" ({i+1}/{len(config['ERA5']['channels'])})")
        files = ERA5_mapping.get_by_channel(channel)['filename'].tolist()
        scene = gds.ERA5Scene.from_files(files)
        match = sample_from_ERA5(gridded_point_data, scene)
        gridded_point_data.data[channel] = match.data[channel]

    # save full dataset
    logger.info("saving full dataset...")
    full_dataset = gridded_point_data.data
    full_dataset = full_dataset.dropna(subset=config['CETB']['channels'][0])
    full_dataset = full_dataset[(full_dataset['time']>=pd.to_datetime(config['time_period']['start'])) & (full_dataset['time']<pd.to_datetime(config['time_period']['end']))]
    full_dataset = full_dataset.reset_index(drop=True)
    
    full_dataset.to_parquet(Path(config['output_dataset']['save_directory']) / Path(config['output_dataset']['name']))
