import yaml
import pickle
from datetime import timedelta
import logging

from tqdm import tqdm
import numpy as np
import pandas as pd

from src.utils.grid_utils import Grid
from src.utils.data_utils import DataMapping
from src.data_src.gridded_data_sources import load_ERA5_data, load_CETB_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Start data processing...")
    with open("./configs/pipeline_configs/predict_data.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    logger.info("Initializing...")

    # loading and modifying target grid
    target_grid = Grid.from_predefined(config['grid'])
    target_grid.modify_extent(config['output_extent'])

    # loading model
    with open(config['model_checkpoint'], 'rb') as model_file:
        model = pickle.load(model_file)

    # channels to process
    cetb_channels = config['CETB']['channels']
    era5_channels = config['ERA5']['channels']

    # mapping CETB and ERA5 files
    logger.info("Mapping CETB and ERA5 files...")
    cetb_mapping = DataMapping(config['CETB']['directory'], 'CETB')
    era5_mapping = DataMapping(config['ERA5']['directory'], 'ERA5')

    # date range to process
    date_range = np.arange(config['dates']['start'], config['dates']['end'], timedelta(days=1))

    # processing data
    logger.info("Processing data...")
    for date in tqdm(date_range, desc="Predicting over date range", unit="dates"):
        date = pd.Timestamp(date)
        
        # Identify available data for given date
        cetb_data = cetb_mapping.get_by_date(date)
        era5_data = era5_mapping.get_by_date(date)
        if len(cetb_data) < len(cetb_channels) or len(era5_data) < len(era5_channels):
            continue

        # loading data
        cetb_data = cetb_data[cetb_data['channel'].isin(cetb_channels)]
        cetb_scene = load_CETB_data(cetb_data['filename'], grid=target_grid)        
        era5_data = era5_data[era5_data['channel'].isin(era5_channels)]
        era5_scene = load_ERA5_data(era5_data['filename'], grid=target_grid)

        # merging data
        scene = cetb_scene + era5_scene

        # masking land and ocean areas
        mask = scene.data[config['mask']['channel']] > config['mask']['threshold']

        # predicting
        prediction = model.predict(scene, mask=mask)
        
        # saving output
        prediction.to_netcdf(f"{config['output']['directory']}/{config['output']['name'].format(date=str(date.date()))}")
