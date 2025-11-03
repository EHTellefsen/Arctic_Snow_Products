# -- coding: utf-8 --
# bucket_utils.py
"""Utility functions for bucket resampling and statistics calculation."""

# -- built-in libraries --

# -- third-party libraries  --
import numpy as np
import pandas as pd
from pyproj import Transformer

#  -- custom modules  --

##########################################################################
def xy_to_grid(x, y, cell_size, x_min, y_max):
    """
    Convert x, y coordinates to grid row, col indices.

    Parameters
    ----------
    x, y : float or np.ndarray
        Coordinates in the same CRS as the grid.
    cell_size : (float, float)
        Resolution of each cell (square).
    x_min : float
        Minimum x (left edge of the grid).
    y_max : float
        Maximum y (top edge of the grid).

    Returns
    -------
    row, col : int or np.ndarray
        Grid indices.
    """
    col = np.floor((x - x_min) / cell_size[0]).astype(int)
    row = np.floor((y_max - y) / cell_size[1]).astype(int)  # invert y-axis
    return row, col


def distance_to_cell_center(x, y, cell_size):
    """Calculate distance from point to center of its grid cell."""
    x_center = (np.floor(x / cell_size[0]) * cell_size[0]) + (cell_size[0] / 2)
    y_center = (np.floor(y / cell_size[1]) * cell_size[1]) + (cell_size[1] / 2)
    distances = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    return distances

def median_abs_dev(series):
    """Calculate the median absolute deviation of a pandas Series."""
    return np.median(np.abs(series - np.median(series)))

def standard_error(series):
    """Calculate the standard error of a pandas Series."""
    return np.std(series)/np.sqrt(len(series))


def drop_in_bucket_resample(df,  target_grid, input_crs = "EPSG:4326"):
    """
    Function for transforming snow depth dataframe to EASE2 grid using drop-in-bucket method.

    !!! only works for EASE2 grid for now !!!
    """
    if target_grid.crs != "EPSG:6931":
        raise ValueError('Target grid CRS must be EPSG:6931')
    
    # prepare transformers
    transformer = Transformer.from_crs(input_crs, target_grid.crs, always_xy=True)
    inv_transformer = Transformer.from_crs(target_grid.crs, input_crs, always_xy=True)
    cell_size = target_grid.grid_cell_size
    xmin = target_grid.extent[0]
    ymax = target_grid.extent[3]

    # prepare grid parameters
    x, y = transformer.transform(df['lon'], df['lat'])
    row, col = xy_to_grid(x, y, cell_size, xmin, ymax)
    distances = distance_to_cell_center(x, y, cell_size)
    df['x'] = x
    df['y'] = y                    
    df['row'] = row
    df['col'] = col
    df['distance_to_cell_center'] = distances
    
    # Calculate statistics
    stats = df.groupby(['row', 'col']).agg(
        time=('time', 'mean'),
        samples_mean_lon=('lon', 'mean'),
        samples_mean_lat=('lat', 'mean'),
        samples_mean_x=('x', 'mean'),
        samples_mean_y=('y', 'mean'),            
        num_samples=('snow_depth', 'size'),
        mean_distance_to_cell_center=('distance_to_cell_center', 'mean'),
        SD_mean=('snow_depth', 'mean'),
        SD_median=('snow_depth', 'median'),
        SD_min=('snow_depth', 'min'),
        SD_max=('snow_depth', 'max'),
        SD_std=('snow_depth', 'std'),
        SD_MAD=('snow_depth', median_abs_dev),
        SD_SE=('snow_depth',standard_error),
        SD_obs_unc=('snow_depth_uncertainty', 'mean'),
    ).reset_index()

    # Additional calculations
    stats['cluster_distance_to_cell_center'] = distance_to_cell_center(stats['samples_mean_x'], stats['samples_mean_y'], cell_size)
    stats['x'] = (stats['col'] * cell_size[0]) + xmin + (cell_size[0] / 2) 
    stats['y'] =  ymax - (stats['row'] * cell_size[1]) - (cell_size[1] / 2)
    stats['lon'], stats['lat'] = inv_transformer.transform(stats['x'], stats['y'])
    stats['doy'] = stats['time'].dt.dayofyear

    # Return selected columns
    return stats.loc[:, ['time','doy','x', 'y','row', 'col', 'lon', 'lat' ,'num_samples',
                            'samples_mean_lon', 'samples_mean_lat','samples_mean_x', 'samples_mean_y',
                            'mean_distance_to_cell_center', 'cluster_distance_to_cell_center',
                            'SD_mean', 'SD_median', 'SD_min', 'SD_max', 'SD_std', 'SD_MAD','SD_SE','SD_obs_unc']]