# -- coding: utf-8 --
# create_training_dataset.py
"""Script to create training dataset by merging point data sources with gridded data sources."""

# -- built-in libraries --
import yaml
from pathlib import Path
import logging

# -- third-party libraries  --
import xarray as xr
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
        return pds.OIB(filepath)
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
# %% run
if __name__ == "__main__":
    logger.info("Starting dataset creation...")
    config_path = "configs/pipeline_configs/create_training_dataset.yaml"
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
    logger.info("Matching to CETB data...")
    CETB_mapping = DataMapping(config['CETB']['directory'], dataset='CETB')
    for i, channel in enumerate(config['CETB']['channels']):
        logger.info(f"Processing CETB channel: {channel}" + f" ({i+1}/{len(config['CETB']['channels'])})")
        files = CETB_mapping.get_by_channel(channel)['filename'].tolist()
        scene = gds.CETBScene.from_files(files)
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

    logger.info("saving full dataset...")
    full_dataset = gridded_point_data
    full_dataset.to_parquet(Path(config['datasets']['full']['save_directory']) / Path(config['datasets']['full']['name']))

    # %% dataset splits
    logger.info("Splitting dataset...")

    remaining = full_dataset
    # Iterate over datasets to create splits
    for dataset in config['datasets'].keys():
        if dataset not in ['full', 'train']:
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