import yaml
from pathlib import Path

import src.data_src.point_data_sources as pds
import src.data_src.gridded_data_sources as gds
import src.utils.data_utils as data_utils
from src.utils.grid_utils import Grid


# %% Point data sources
def load_point_data_source(filepath, secondary_id):
    if secondary_id in ['2020-2021', '2021-2022']:
        return pds.C2I(filepath, secondary_id=secondary_id)
    elif secondary_id == 'IDCSI4':
        return pds.OIB(filepath)
    elif secondary_id == 'ICEBIRD':
        return pds.AEM_AWI_ICEBIRD(filepath)
    elif secondary_id == 'PARARCMIP':
        return pds.AEM_AWI_PARARCMIP(filepath)
    else:
        raise ValueError(f"Unsupported secondary_id: {secondary_id}")


def grid_and_merge_point_data_sources(sources, target_grid_id):
    gridded_sources = []
    for source in sources:
        gridded_sources.append(source.resample_bucket(target_grid_id))
    
    # Merge all gridded sources into one
    return pds.base.GriddedPointDataSource.merge_sources(gridded_sources)


# %% Sample from gridded sources
def sample_from_gridded_sources(point_source, gridded_source):
    pass

# %% splits
def split_dataset(dataset, val_frac, val_equalization, test_period):
    pass

# %% run
if __name__ == "__main__":
    pass