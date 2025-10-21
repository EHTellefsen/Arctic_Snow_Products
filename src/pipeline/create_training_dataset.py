import xarray as xr
from tqdm import tqdm

import yaml
from pathlib import Path
import logging

from src.utils.grid_utils import Grid
import src.data_src.point_data_sources as pds
import src.data_src.gridded_data_sources as gds
from src.utils.data_utils import DataMapping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# %% Point data sources
def load_point_data_source(filepath, secondary_id):
    if secondary_id in ['2020-2021', '2021-2022']:
        return pds.C2I(filepath, secondary_id=secondary_id)
    elif secondary_id == 'IDCSI4':
        return pds.OIB(filepath)
    elif secondary_id == 'ICEBIRD':
        return pds.AEM_AWI_ICEBIRD(filepath)
    elif secondary_id == 'PAMARCMIP':
        return pds.AEM_AWI_PAMARCMIP(filepath)
    else:
        raise ValueError(f"Unsupported secondary_id: {secondary_id}")


def grid_and_merge_point_data_sources(sources, target_grid_id):
    gridded_sources = []
    for source in sources:
        gridded_sources.append(source.resample_bucket(target_grid_id))
    
    # Merge all gridded sources into one
    return pds.base.GriddedPointDataSource.merge_sources(gridded_sources)


# %% Sample from gridded sources
def sample_from_CETB(point_source, gridded_source):
    points = xr.Dataset({
        "time": (("points",), point_source.data['time'].astype('<M8[ns]').values),
        "x":   (("points",), point_source.data['x'].values),
        "y":  (("points",), point_source.data['y'].values),
    })

    match = gridded_source.data.interp(
        time=points["time"],
        x=points["x"],
        y=points["y"],
        method = 'linear'
    )

    sampled_df = point_source.data.copy()
    for var in match.data_vars:
        sampled_df[var] = match[var].values
    
    return pds.GriddedPointDataSource(sampled_df, point_source.grid)


def sample_from_ERA5(point_source, gridded_source):
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
def split_dataset(dataset, val_frac, val_equalization, test_period):
    
    # Split dataset into train_val and test sets - test set is held out based on time period
    if test_period is not None:
        test_set, train_val_set = split_by_period(dataset, test_period['start'], test_period['end'])
    else:
        test_set = None
        train_val_set = dataset
    
    # Split train_val_set into train and validation sets - validation set is randomly sampled based on val_frac and equalization
    if val_frac > 0:
        train_set, val_set = split_by_fraction(train_val_set, val_frac, equalization_id=val_equalization['equalization_id'])
    else:
        train_set = train_val_set
        val_set = None

    return train_set, val_set, test_set


def split_by_period(dataset, start_date, end_date):
    mask = (dataset.data['time'] >= start_date) & (dataset.data['time'] < end_date)
    subset = pds.GriddedPointDataSource(dataset.data[mask], dataset.grid)
    remaining = pds.GriddedPointDataSource(dataset.data[~mask], dataset.grid)
    return subset, remaining

def split_by_fraction(dataset, fraction, equalization_id):
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
            
        val_set = pds.GriddedPointDataSource(dataset.data.loc[val_indices], dataset.grid)
        train_set = pds.GriddedPointDataSource(dataset.data.drop(index=val_indices), dataset.grid)
    else:
        val_size = int(len(dataset.data) * fraction)
        val_data = dataset.data.sample(n=val_size, random_state=42)
        val_set = pds.GriddedPointDataSource(val_data, dataset.grid)
        train_set = pds.GriddedPointDataSource(dataset.data.drop(index=val_data.index), dataset.grid)

    return train_set, val_set

# %% run
if __name__ == "__main__":
    logger.info("Starting dataset creation...")

    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs/pipeline_configs/create_training_dataset.yaml"
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
                secondary_id=config['point_data_sources'][source_cfg]['secondary_id'][sid_cfg]['id']
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
    full_dataset.to_parquet(config['full_dataset']['directory'] / config['full_dataset']['name'])

    # %% dataset splits
    logger.info("Splitting dataset...")
    train_set, val_set, test_set = split_dataset(
        full_dataset,
        val_frac=config['split_params']['val_fraction'],
        val_equalization=config['split_params']['val_equalization'],
        test_period=config['split_params']['test_period']
    )
    
    logger.info("Saving datasets...")
    train_set.to_parquet(Path(config['train_set']['directory']) / config['train_set']['name'])
    if val_set is not None:
        val_set.to_parquet(Path(config['val_set']['directory']) / config['val_set']['name'])
    if test_set is not None:
        test_set.to_parquet(Path(config['test_set']['directory']) / config['test_set']['name'])