from .base import PointDataSource
import pandas as pd
from pathlib import Path

class AEM_AWI_ICEBIRD(PointDataSource):
    def __init__(self, files):
        self.files = files

        super().__init__(primary_id='AEM_AWI', secondary_id='ICEBIRD')
        self.param_dict = {
            'datetime': 'Date/Time',
            'lat': 'Latitude (Internal Navigation System)',
            'lon': 'Longitude (Internal Navigation System)',
            'snow_depth': 'Snow thick [m] (Calculated)',
            'snow_depth_uncertainty': 'Snow thick unc [m] (0.044 m (Jutila et al., 2021)...)',
            'weight': None            
        }
        self.load()

    def load(self):
        """Load raw data from file"""
        if isinstance(self.files, str):
            if self.files.endswith('.tab'):
                self.files = [self.files]
            elif Path(self.files).is_dir():
                self.files = list(Path(self.files).glob('*.tab'))
        elif not isinstance(self.files, list):
            raise ValueError("files should be a string or a list of strings.")

        self.data = pd.concat([pd.read_table(file,skiprows=37, delimiter='\t') for file in self.files])


class AEM_AWI_PAMARCMIP(PointDataSource):
    def __init__(self, files):
        self.files = files

        super().__init__(primary_id='AEM_AWI', secondary_id='PAMARCMIP')
        self.param_dict = {
            'datetime': 'Date/Time (UTC, Airborne electromagnetic...)',
            'lat': 'Latitude (Airborne electromagnetic (EM)...)',
            'lon': 'Longitude (Airborne electromagnetic (EM)...)',
            'snow_depth': 'Snow thick [m] (mean, within the EM-Bird foot...)',
            'snow_depth_uncertainty': 'Snow thick unc [m] (uncertainty of mean, within t...)',
            'weight': 'NOBS [#] (number of snow depth estimate...)'
        }
        self.load()

    def load(self):
        """Load raw data from file"""
        if isinstance(self.files, str):
            if self.files.endswith('.tab'):
                self.files = [self.files]
            elif Path(self.files).is_dir():
                self.files = list(Path(self.files).glob('*.tab'))
        elif not isinstance(self.files, list):
            raise ValueError("files should be a string or a list of strings.")

        df = pd.concat([pd.read_table(file,skiprows=47, delimiter='\t') for file in self.files])
        self.data = df.loc[df.index.repeat(df[self.param_dict['weight']])].reset_index(drop=True)