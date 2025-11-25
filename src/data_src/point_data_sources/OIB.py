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
        df = pd.concat([pd.read_table(file, delimiter=',', na_values=[-99999], dtype={ 'date': str}, usecols = ['date', 'lat', 'lon',  'snow_depth', 'snow_depth_unc']) for file in self.files])
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        self.data = df


class OIB_QL(PointDataSource):
    def __init__(self, files):
        self.files = files
        
        super().__init__(primary_id='OIB', secondary_id='QuickLook')
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
                self.files = list(Path(self.files).rglob('*.txt'))
        elif isinstance(self.files, list):
            out = []
            for f in self.files:
                if Path(f).is_dir():
                    out.extend(list(Path(f).rglob('*.txt')))
                else:
                    out.append(f)
            self.files = list(out)
        elif not isinstance(self.files, list):
            raise ValueError("files should be a string or a list of strings or directories with OIB_*.txt files.")
        
        df = pd.concat([pd.read_table(file, delimiter=',', na_values=[-99999], dtype={ 'date': str}, usecols = ['date', 'lat', 'lon',  'snow_depth', 'snow_depth_unc']) for file in self.files])
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        self.data = df
