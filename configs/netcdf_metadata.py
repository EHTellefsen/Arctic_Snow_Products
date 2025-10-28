from datetime import datetime
attrs = {
    'x': {
        'standard_name': 'projection_x_coordinate',
        'coverage_content_type': 'coordinate',
        'long_name': 'x',
        'units': 'meters',
        'axis': 'X',
        'valid_range_min': -9000000.0, 
        'valid_range_max': 9000000.0
    },
    'y': {
        'standard_name': 'projection_y_coordinate',
        'coverage_content_type': 'coordinate',
        'long_name': 'y',
        'units': 'meters',
        'axis': 'Y',
        'valid_range_min': -9000000.0, 
        'valid_range_max': 9000000.0
    },
    'lat': {
        'standard_name': 'latitude',
        'long_name': 'latitude',
        'units': 'degrees_north',
        'valid_range_min': -90.0, 
        'valid_range_max': 90.0
    },
    'lon': {
        'standard_name': 'longitude',
        'long_name': 'longitude',
        'units': 'degrees_east',
        'valid_range_min': -180.0,
        'valid_range_max': 180.0
    },
    'time': {
        'standard_name': 'time',
        'coverage_content_type': 'coordinate',
        'long_name': 'time',
        'axis': 'T',
        #'units': 'seconds since 1970-01-01T00:00:00Z',
    },
    'sd': {
        'standard_name': 'snow_depth',
        'long_name': 'snow depth',
        'description': 'Snow depth data derived using model utilizing inputs AMSR2 measurements from NSIDC-0630 (CETB) and ERA5 daily reanalysis data, and trained using CRYO2ICE (C2I) data from doi:10.1029/2023EA003313',
        'units': 'meters',
        'valid_range_min': 0.0,
        'valid_range_max': 10.0,
        'RMSE_cross_validation': 0.059,
        'RMSE_test_set': 0.072,
        'R2_test_set': 0.27,
        'units': 'meters',
        'grid_mapping' : 'crs',
        'coordinates': "lon lat"
    },
    'crs' : {
        'grid_mapping_name': 'lambert_azimuthal_equal_area',
        'longitude_of_projection_origin': 0.0,
        'latitude_of_projection_origin': 90.0,
        'false_easting': 0.0,
        'false_northing': 0.0,
        'semi_major_axis': 6378137.0,
        'inverse_flattening': 298.257223563,
        'proj4_string': '+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m',
        'epsg_code': 6931,
        'srid': 'urn:ogc:def:crs:EPSG::6931',
        'coverage_content_type' : 'auxiliaryInformation',
        'crs_wkt' : 'PROJCRS["WGS 84 / NSIDC EASE-Grid 2.0 North", BASEGEODCRS["WGS 84", DATUM["World Geodetic System 1984", ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1.0]]]], CONVERSION["US NSIDC EASE-Grid 2.0 North", METHOD["Lambert Azimuthal Equal Area",ID["EPSG",9820]], PARAMETER["Latitude of natural origin",90,ANGLEUNIT["degree",0.01745329252]], PARAMETER["Longitude of natural origin",0,ANGLEUNIT["degree",0.01745329252]], PARAMETER["False easting",0,LENGTHUNIT["metre",1.0]], PARAMETER["False northing",0,LENGTHUNIT["metre",1.0]]], CS[cartesian,2], AXIS["easting (X)",south,MERIDIAN[90,ANGLEUNIT["degree",0.01745329252]],ORDER[1]], AXIS["northing (Y)",south,MERIDIAN[180,ANGLEUNIT["degree",0.01745329252]],ORDER[2]], LENGTHUNIT["metre",1.0], ID["EPSG",6931]]',
        'long_name' : 'EASE2_N3.125km',
        'GeoTransform' : '-4500000.00000  3125.00000 0.00000 4500000.00000 0.00000 -3125.00000'
    },
    'global': {
        'title': 'Automatic Snow Products - Snow Depth V1.0',
        'summary': 'This dataset contains snow depth (SD) estimates over the Arctic region derived from AMSR2 brightness temperature measurements and ERA5 reanalysis data using a machine learning model (Random Forest Regression) trained on snow depth observations from the CRYO2ICE campaign during the winter season 2020-2021 and 2021-2022.',
        'contributer_name': 'Emil H. Tellefsen, Mai Winstrup, Henriette Skourup, Renée M. Fredensborg Hansen',
        'contributer_role': 'principal_investigator, co_investigator, co_investigator, co_investigator',
        'software_repository': 'https://github.com/EHTellefsen/Automatic_Snow_Products',
        'data_sources': 'CETB AMSR2 NSIDC-0630 (https://nsidc.org/data/nsidc-0630/versions/2), \nERA5 reanalysis (https://cds.climate.copernicus.eu/datasets/derived-era5-single-levels-daily-statistics?tab=overview), \nCRYO2ICE snow depth data (https://data.dtu.dk/articles/dataset/CRYO2ICE_radar_laser_freeboards_snow_depth_on_sea_ice_and_comparison_against_auxiliary_data_during_winter_season_2020-2021/21369129)',
        'Conventions': 'CF-1.9, ACDD-1.3',
        'institution': 'DTU Space',
        'geospatial_lat_min': 0.0,
        'geospatial_lat_max': 90.0,
        'geospatial_lon_min': -180.0,
        'geospatial_lon_max': 180.0,
        'geospatial_lat_units': 'degrees_north',
        'geospatial_lon_units': 'degrees_east',
        'time_coverage_start': '2020-01-01T00:00:00Z',
        'time_coverage_end': '2024-12-31T23:59:59Z',
        'keywords': 'Snow Depth, AMSR2, ERA5, Machine Learning, Random Forest, Arctic, Remote Sensing',
        'email': 'maiwin@space.dtu.dk',
        'date_created': datetime.today().strftime('%Y-%m-%d')
    }
}