from .base import PointDataSource
import pandas as pd
from pathlib import Path

class C2I(PointDataSource):
    def __init__(self, files, secondary_id, retracker = 'LARM_smoothed'):
        self.files = files
        self.retracker = retracker

        if secondary_id not in ['2020-2021','2021-2022']:
            raise ValueError('Secondary id does not represent relevant time periods')

        super().__init__(primary_id='C2I', secondary_id=secondary_id)
        self.param_dict = {
            'datetime': 'time',
            'lat': 'lat',
            'lon': 'lon',
            'snow_depth': 'snow_depth_original_MSS_{}'.format(retracker),
            'snow_depth_uncertainty': 0.0899 if secondary_id=='2020-2021' else 0.0911
        }
        self.load()


    def load(self):
        """Load raw data from file"""
        if isinstance(self.files, str):
            if self.files.endswith('.csv'):
                self.files = [self.files]
            elif Path(self.files).is_dir():
                self.files = list(Path(self.files).glob('*.csv'))
        elif not isinstance(self.files, list):
            raise ValueError("files should be a string or a list of strings.")

        self.data = pd.concat([pd.read_table(file, delimiter=',') for file in self.files])
    
