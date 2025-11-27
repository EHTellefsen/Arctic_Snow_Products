Although the scripts allows for great flexibility as to the exact location by which raw data is stored, and processed data is saved, code failure can arise from inconsistent storage of data.
To ensure import of data is as smooth as possible, we have tried to write the code s.t. inclusion of raw data requires minimal change to its original format. Yet some changes were needed, as data were either (a) downloaded file by file, (b) did not follow a consistent naming convention or (c) had dates inaccessible in file name. This document illustrates the repository structure for the ./data folder, as it was durring the original ASP v1.0 processing.

Origin of the various datasets are shown here (access date: 27-11-2025):

AEM-AWI-ICEBIRD_2019: https://doi.pangaea.de/10.1594/PANGAEA.932790
AEM-AWI-PAMARCMIP_2017: https://doi.pangaea.de/10.1594/PANGAEA.933883
C2I_SnowDepths: https://data.dtu.dk/articles/dataset/CRYO2ICE_radar_laser_freeboards_snow_depth_on_sea_ice_and_comparison_against_auxiliary_data_during_winter_season_2020-2021/21369129
CETB_AMSR2: https://nsidc.org/data/nsidc-0630/versions/2
ERA5_all-param_daily_perMonth_1hourly-sample: https://cds.climate.copernicus.eu/datasets/derived-era5-single-levels-daily-statistics?tab=overview
IceBridge_Sea_Ice_Freeboard_SnowDepth_and_Thickness_QuickLook: https://nsidc.org/data/nsidc-0708/versions/1
OIB_IDCSI4: https://nsidc.org/data/idcsi4/versions/1

A few notes:
* CETB_AMSR2 does not have to be temporally sorted in folders- code is adapted to be indifferent to subfolder name - it only looks at file names.
* ERA5 data needs to be sorted in monthly folders (2013-11,...) and each variable within folders needs its own netCDF file. NetCDF files needs to contain data for the entire month. 
* ERA5 variable files need to follow naming convention of "configs/era5_variable_dictionary.yaml".
* Renaming/restructering is unneeded for snow depth datasets.

Except for data folders in "raw" all folders shown in tree will be created when setup.py is run.

data
	├───fig
	│
	├───processed
	│   ├───models
	│   ├───predictions
	│   └───predictions_aggregated
	│
	├───intermediate
	│   ├───CV_results
	│   └───datasets
	│
	└───raw
	    ├───AEM-AWI-ICEBIRD_2019
	    │   └───datasets
	    │       ├─── P6_217_ICEBIRD_2019_1904020901_high_snowdepth_V1.tab
	    │       └───...
	    │   
	    ├───AEM-AWI-PAMARCMIP_2017
	    │   └───datasets
	    │       ├─── P5_205_PAMARCMIP_2017_1704021401_sea-ice.tab
	    │       └───...
	    │
	    ├───C2I_SnowDepths
	    │   └───CRYO2ICE_individul_comparison_BaselineE
	    │       ├───2020-2021
	    │       │   ├─── CRYO2ICE_CRYO2ICE_CS_LTA__SIR_SAR_2__20201102T123414_20201102T123933_E001_original_MSS_smooth_BaselineE_LARM_AMSR2_SMLG_MERRA5_mW99_smooth_BaselineE.csv
	    │       │   └───...
	    │       └───2021-2022
	    │   
	    ├───CETB_AMSR2
	    │   ├───2012-2013
	    │   │   ├─── NSIDC0630_SIR_EASE2_N3.125km_GCOMW1_AMSR2_M_36H_20121031_2505230315_v2.0.nc
	    │   │   └───...
	    │   ├───2013-2014
	    │   └───...
	    │   
	    ├───ERA5_all-param_daily_perMonth_1hourly-sample
	    │   ├───2012_11
	    │   │   ├─── 2m_temperature_0_daily-mean.nc
	    │   │   ├─── sea_ice_cover_0_daily-mean.nc
	    │   │   └───...
	    │   ├───2012_12
	    │   └───...
	    │   
	    ├───IceBridge_Sea_Ice_Freeboard_SnowDepth_and_Thickness_QuickLook
	    │   ├───2012_GR_NASA
	    │   │   ├─── OIB_20120314_IDCSI2_ql.txt
	    │   │   └───...
	    │   ├───2013_GR_NASA
	    │   └───...
	    │   
	    └───OIB_IDCSI4
		├───IDCSI4_20090331.txt
		└───...
