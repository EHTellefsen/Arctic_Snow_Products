from .base import PointDataSource
import pandas as pd
from pathlib import Path

class OIB(PointDataSource):
    def __init__(self, files):
        self.files = files

        super().__init__(primary_id='OIB', secondary_id='IDCSI4')
        self.param_dict = {
            'datetime': 'date',
            'lat': 'lat',
            'lon': 'lon',
            'snow_depth': 'snow_depth',
            'snow_depth_uncertainty': 'snow_depth_unc',
        }
        self.load()


    def load(self):
        """Load raw data from file"""
        if isinstance(self.files, str):
            if self.files.endswith('.txt'):
                self.files = [self.files]
            elif Path(self.files).is_dir():
                self.files = list(Path(self.files).glob('*.txt'))
        elif not isinstance(self.files, list):
            raise ValueError("files should be a string or a list of strings.")
        
        df = pd.concat([pd.read_table(file, delimiter=',', na_values=[-99999.0000, -99999.0], dtype={'date': str, 'ATM_file_name': str}) for file in self.files])
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        self.data = df