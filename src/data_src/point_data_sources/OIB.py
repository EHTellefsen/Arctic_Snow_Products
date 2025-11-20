# -- coding: utf-8 --
# OIB.py
"""OIB point data source implementation."""

# -- built-in modules --
from pathlib import Path

# -- third-party modules --
import pandas as pd

# -- custom modules --
from .base import PointDataSource

##############################################################################################
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

class OIB_IDCSI4(PointDataSource):
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


class OIB_QL(PointDataSource):
    def __init__(self, files):
        self.files = files
        secondary_id = self._identify_secondary_id()

        super().__init__(primary_id='OIB', secondary_id=secondary_id)
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
                self.files = list(Path(self.files).rglob('OIB_*.txt'))
        elif not isinstance(self.files, list):
            raise ValueError("files should be a string or a list of strings.")
        
        df = pd.concat([pd.read_table(file, delimiter=',', na_values=[-99999.0000, -99999.0], dtype={'date': str, 'ATM_file_name': str}) for file in self.files])
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        self.data = df

    def _identify_secondary_id(self):
        years = []
        for f in self.files:
            f_name = Path(f).stem
            year = f_name.split('_')[1][:4]
            years.append(int(year))

        if all(y < 2015 for y in years):
            return 'QL-pre-2015'
        elif all(y >= 2015 for y in years):
            return 'QL-post-2015'
        else:
            raise ValueError("Mixed years in files; cannot determine secondary_id.")