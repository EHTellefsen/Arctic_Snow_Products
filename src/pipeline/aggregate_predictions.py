# -- coding: utf-8 --
# aggregate_predictions.py
"""Script to aggregate daily prediction NetCDF files into a single monthly mean NetCDF file."""

# -- built-in libraries --
import logging
import yaml

# -- third party libraries --
import xarray as xr
from tqdm import tqdm
import pandas as pd

#  -- custom modules  --
from src.utils.data_utils import DataMapping

logger = logging.getLogger(__name__)

########################################################################
if __name__ == "__main__":
    with open("./configs/pipeline_configs/aggregate_predictions.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    logger.info("Aggregating predictions...")

    # mapping prediction files
    prediction_mapping = DataMapping(config['predictions_directory'], 'ASP')
    date_range = pd.date_range(start=config['time_period']['start'], end=config['time_period']['end'], freq='D')

    for year in tqdm(range(date_range.year.min(), date_range.year.max()+1), desc="Aggregating yearly data", unit="years"):
        for month in range(1, 13):
            monthly_files = prediction_mapping.mapping[
                (prediction_mapping.mapping['date'].dt.year == year) &
                (prediction_mapping.mapping['date'].dt.month == month)
            ]['filename'].tolist()

            if not monthly_files:
                logger.warning(f"No prediction files found for {year}-{month:02d}. Skipping...")
                continue

            # load and concatenate daily files
            monthly_data = xr.open_mfdataset(monthly_files, combine='by_coords')
            monthly_mean = monthly_data.mean(dim='time')
            monthly_mean = monthly_mean.expand_dims(time=[pd.Timestamp(year=year, month=month, day=15)])

            # save to NetCDF
            output_filename = f"{config['output_directory']}/ASP_snow_depth_prediction_{year}-{month:02d}.nc"
            monthly_mean.to_netcdf(output_filename)
            logger.info(f"Saved aggregated file: {output_filename}")

    
    


