import os
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime

class DataMapping:
    def __init__(self, directory, dataset):
        self.mapping = pd.DataFrame({'date':[], 'channel': [], 'filename': []})
        
        self.dataset=dataset

        if dataset=='ERA5':
            self._map_ERA5(directory)
        elif dataset=='CETB':
            self._map_CETB(directory)
        else:
            raise ValueError('dateset needs to be either ERA5 or CETB')

    # %% Mapping utilities
    def _map_ERA5(self, directory):
        ERA5_dict = yaml.safe_load(open('configs/ERA5_variable_dictionary.yaml'))
        ERA5_dict_df = pd.DataFrame(ERA5_dict).T
        ERA5_dict_df.index.name = 'short_name'
        ERA5_dict_df=ERA5_dict_df.reset_index()
        ERA5_dict_df = ERA5_dict_df.set_index('filename')

        for root, _, files in os.walk(directory):
            # try to detect YYYY-MM in directory name
            dir_name = os.path.basename(root)

            try:
                year, month = map(int, dir_name.split('_'))
                for file in files:
                    if file.endswith('.nc'):
                        try:
                            self.mapping.loc[len(self.mapping)] = [
                                pd.Timestamp(year=year, month=month, day=1),
                                ERA5_dict_df.loc[file]['short_name'],
                                os.path.join(root, file)
                                ]
                        except KeyError:
                            continue
                    
            except ValueError:
                # skip directories not following YYYY-MM format
                continue


    def _map_CETB(self, directory):
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.nc'):
                    channel = Path(file).stem.split('_')[7]
                    try:
                        # extract date assuming format CETB_YYYYMMDD.nc
                        date_str = file.split('_')[8]
                        date = datetime.strptime(date_str, '%Y%m%d').date()
                        full_path = os.path.join(root, file)
                        self.mapping.loc[len(self.mapping)] = [pd.Timestamp(date),
                                                                channel,
                                                                full_path]
                    except (IndexError, ValueError):
                        continue

    # %% Querying utilities
    def get_by_channel(self, channel):
        if isinstance(channel, list):
            return self.mapping[self.mapping['channel'].isin(channel)]
        elif isinstance(channel, str):
            return self.mapping[self.mapping['channel'] == channel]
        else:
            raise ValueError('channel needs to be either a string or a list of strings')

    def get_by_date(self, date):
        
        # list of dates
        if isinstance(date, list):
            date = [d.date() if isinstance(d, datetime) else datetime.strptime(d, '%Y-%m-%d').date() for d in date]

            if self.dataset=='CETB':
                return self.mapping[self.mapping['date'].isin([pd.Timestamp(d) for d in date])]
            elif self.dataset=='ERA5':
                date = [pd.Timestamp(year=d.year, month=d.month, day=1) for d in date]
                return self.mapping[self.mapping['date'].isin(date)]

        # time period
        elif isinstance(date, slice):
            start_date = date.start
            end_date = date.stop

            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

            if self.dataset=='CETB':
                return self.mapping[(self.mapping['date'] >= pd.Timestamp(start_date)) & (self.mapping['date'] <= pd.Timestamp(end_date))]
            elif self.dataset=='ERA5':
                start_date = pd.Timestamp(year=start_date.year, month=start_date.month, day=1)
                end_date = pd.Timestamp(year=end_date.year, month=end_date.month, day=1)
                return self.mapping[(self.mapping['date'] >= start_date) & (self.mapping['date'] <= end_date)]

        # single date
        elif isinstance(date, datetime):
            date = date.date()

        elif isinstance(date, str):
            try:
                date = datetime.strptime(date, '%Y%m%d').date()
            except ValueError:
                date = datetime.strptime(date, '%Y-%m-%d').date()

        elif isinstance(date, pd.Timestamp):
            date = date

        else:
            raise ValueError('date needs to be either a datetime object, a string in YYYY-MM-DD or YYYYMMDD format, a list of dates, or a slice object representing a time period')

        if self.dataset=='CETB':
            return self.mapping[self.mapping['date'] == pd.Timestamp(date)]
        elif self.dataset=='ERA5':
            date = pd.Timestamp(year=date.year, month=date.month, day=1)
            return self.mapping[self.mapping['date'] == date]


        