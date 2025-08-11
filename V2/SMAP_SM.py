import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import datetime
import h5py
import calendar
# Ignore runtime warning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import xarray as xr
import rasterio as rio
import rioxarray
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, mapping
import sys

#Monthly Average Bin's 1km SMAP soil moisture products for Limpopo
filepaths = sorted(glob.glob(r'D:\processed-data\SMAP\UVA_1km\*.tif'))
daterange = pd.date_range(start='201504__',end='20151126',freq='D').strftime('%Y%m%d')
files_selected = [[file for file in filepaths if file.find(id) > 0] for id in sorted(daterange)]
files_selected = [file for files in files_selected for file in files]
shpname = r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
shapefile = gpd.read_file(shpname)
path=r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\SMAP_SM_1km\south_africa_monthly\south_africa_daily'
clipped_xr = [xr.open_mfdataset(file, parallel=True, chunks={'x':500,'y':500}).rio.write_crs('epsg:6933').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False).band_data.mean(axis=0).rio.to_raster(path+'\{}.tif'.format(date))
        for file,date in zip(files_selected,daterange)]


#Create NETCDF
clipped_filepaths = sorted(glob.glob(path+'\*.tif'))
clipped_xrds = [xr.open_mfdataset(file, parallel=True, chunks={'x':500,'y':500}).rio.reproject('epsg:4326') for file in clipped_filepaths]
dataset_daily = xr.concat(clipped_xrds, dim=pd.date_range(start='20151126',end='20220731',freq='D')).rename({'concat_dim':'time'})
dataset_monthly = dataset_daily.band_data.resample(time='1M').mean()

dataset_daily.to_netcdf(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\SMAP_SM_1km\south_africa_monthly\SMAP_2016_2021_daily.nc')
dataset_monthly.to_netcdf(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\SMAP_SM_1km\south_africa_monthly\SMAP_2016_2021_monthly.nc')


#SMAP NETCDF Files
SMAP_files = sorted(glob.glob(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\SMAP_SM_1km\south_africa_monthly\*.nc'))
sm_2015_2020_monthly = xr.open_mfdataset([SMAP_files[0]],parallel=True,chunks={'x':500,'y':500}).SM_vwc
sm_2017_2018_monthly = xr.open_mfdataset([SMAP_files[2]],parallel=True,chunks={'x':500,'y':500}).band_data
sm_2021_2022_monthly = xr.open_mfdataset([SMAP_files[4]],parallel=True,chunks={'x':500,'y':500}).band_data


sm_2015_2020_monthly.mean(dim={'x','y'}).plot()
#sm_2017_2018_monthly.mean(dim={'x','y'}).plot()
sm_2021_2022_monthly.mean(dim={'x','y'})[:,0].plot()

sm_ts = xr.concat([sm_2015_2020_monthly.mean(dim={'x','y'}),(sm_2021_2022_monthly.mean(dim={'x','y'})[:,0])],dim='time')

# Generate sequence of string between start and end dates (Year + DOY)
start_date = '2015-04-01'
end_date = '2021-10-28'

start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
delta_date = end_date - start_date

date_seq = []
for i in range(delta_date.days + 1):
    date_str = start_date + datetime.timedelta(days=i)
    date_seq.append(str(date_str.timetuple().tm_year) + str(date_str.timetuple().tm_yday).zfill(3))

# Count how many days for a specific year
yearname = np.linspace(2015, 2021, 7, dtype='int')
monthnum = np.linspace(1, 12, 12, dtype='int')
monthname = np.arange(1, 13)
monthname = [str(i).zfill(2) for i in monthname]

daysofyear = []
for idt in range(len(yearname)):
    if idt == 0:
        f_date = datetime.date(yearname[idt], monthnum[3], 1)
        l_date = datetime.date(yearname[idt], monthnum[-1], 31)
        delta_1y = l_date - f_date
        daysofyear.append(delta_1y.days + 1)
    else:
        f_date = datetime.date(yearname[idt], monthnum[0], 1)
        l_date = datetime.date(yearname[idt], monthnum[-1], 31)
        delta_1y = l_date - f_date
        daysofyear.append(delta_1y.days + 1)

daysofyear = np.asarray(daysofyear)

# Find the indices of each month in the list of days between 2015 - 2018
nlpyear = 1999 # non-leap year
lpyear = 2000 # leap year
daysofmonth_nlp = np.array([calendar.monthrange(nlpyear, x)[1] for x in range(1, len(monthnum)+1)])
ind_nlp = [np.arange(daysofmonth_nlp[0:x].sum(), daysofmonth_nlp[0:x+1].sum()) for x in range(0, len(monthnum))]
daysofmonth_lp = np.array([calendar.monthrange(lpyear, x)[1] for x in range(1, len(monthnum)+1)])
ind_lp = [np.arange(daysofmonth_lp[0:x].sum(), daysofmonth_lp[0:x+1].sum()) for x in range(0, len(monthnum))]
ind_iflpr = np.array([int(calendar.isleap(yearname[x])) for x in range(len(yearname))]) # Find out leap years
# Generate a sequence of the days of months for all years
daysofmonth_seq = np.array([np.tile(daysofmonth_nlp[x], len(yearname)) for x in range(0, len(monthnum))])
daysofmonth_seq[1, :] = daysofmonth_seq[1, :] + ind_iflpr # Add leap days to February


# Path of SMAP data
path_smap = r'D:\raw-data\SMAP\L3_9km_V4'

smap_files = glob.glob(path_smap+'\*.h5')


for iyr in range(0, len(daysofyear)):

    os.chdir(path_smap + '/' + str(yearname[iyr]))
    smap_files_year = sorted(glob.glob('*.h5'))
    print(smap_files_year)

# Group SMAP data by month
for imo in range(len(monthnum)):

    os.chdir(path_smap + '/' + str(yearname[iyr]))
    smap_files_group_1month = [smap_files_year.index(i) for i in smap_files_year if str(yearname[iyr]) + monthname[imo] in i]

# Process each month
if len(smap_files_group_1month) != 0:
    smap_files_month = [smap_files_year[smap_files_group_1month[i]] for i in range(len(smap_files_group_1month))]

    # Create initial empty matrices for monthly SMAP final output data
    matsize_smap = [matsize_smap_1day[0], matsize_smap_1day[1], daysofmonth_seq[imo, iyr]]
    smap_mat_month_am = np.empty(matsize_smap, dtype='float32')
    smap_mat_month_am[:] = np.nan
    smap_mat_month_pm = np.copy(smap_mat_month_am)

# Extract SMAP data layers and rebind to daily
for idt in range(daysofmonth_seq[imo, iyr]):
    smap_files_group_1day = [smap_files_month.index(i) for i in smap_files_month if
                                str(yearname[iyr]) + monthname[imo] + str(idt+1).zfill(2) in i]
    smap_files_1day = [smap_files_month[smap_files_group_1day[i]] for i in
                        range(len(smap_files_group_1day))]
    smap_files_group_1day_am = [smap_files_1day.index(i) for i in smap_files_1day if
                                'D_' + str(yearname[iyr]) + monthname[imo] + str(idt+1).zfill(2) in i]
    smap_files_group_1day_pm = [smap_files_1day.index(i) for i in smap_files_1day if
                                'A_' + str(yearname[iyr]) + monthname[imo] + str(idt+1).zfill(2) in i]
    smap_mat_group_1day = \
        np.empty([matsize_smap_1day[0], matsize_smap_1day[1], len(smap_files_group_1day)], dtype='float32')
    smap_mat_group_1day[:] = np.nan

# Read swath files within a day and stack
for ife in range(len(smap_files_1day)):
    smap_mat_1file = np.copy(smap_mat_init_1day)
    fe_smap = h5py.File(smap_files_1day[ife], "r")
    group_list_smap = list(fe_smap.keys())
    smap_data_group = fe_smap[group_list_smap[1]]
    varname_list_smap = list(smap_data_group.keys())
    # Extract variables
    col_ind = smap_data_group[varname_list_smap[41]][()]
    row_ind = smap_data_group[varname_list_smap[10]][()]
    sm_flag = smap_data_group[varname_list_smap[20]][()]
    sm = smap_data_group[varname_list_smap[25]][()]
    sm[np.where(sm == -9999)] = np.nan
    sm[np.where((sm_flag == 7) & (sm_flag == 15))] = np.nan # Refer to the results of np.binary_repr

    smap_mat_1file[row_ind, col_ind] = sm
    smap_mat_group_1day[:, :, ife] = smap_mat_1file
    print(smap_files_1day[ife])
    fe_smap.close()

    del(smap_mat_1file, fe_smap, group_list_smap, smap_data_group, varname_list_smap, col_ind, row_ind,
        sm_flag, sm)


#Bin's GITHUB:

########################################################################################################################

# 0. Input variables
# Specify file paths
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_CONUS/codes_py'
# Path of source MODIS data
path_modis = '/Users/binfang/Downloads/Processing/HDF'
# # Path of output MODIS data
# path_modis_op = '/Volumes/MyPassport/SMAP_Project/NewData/MODIS/Output'
# Path of MODIS data for SM downscaling model input
path_modis_model = '/Users/binfang/Downloads/Processing/Model_Input'
# Path of SM model output
path_model_op = '/Users/binfang/Downloads/Processing/Model_Output'
# Path of SMAP data
path_smap = '/Volumes/MyPassport/SMAP_Project/NewData/SMAP'
# Path of processed data
path_procdata = '/Users/binfang/Downloads/Processing'
# Path of Land mask
path_lmask = '/Volumes/MyPassport/SMAP_Project/Datasets/Lmask'
lst_folder = '/MYD11A1/'
ndvi_folder = '/MYD13A2/'
subfolders = np.arange(2015, 2019+1, 1)
subfolders = [str(i).zfill(4) for i in subfolders]

# Load in variables
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_world_max', 'lat_world_min', 'lon_world_max', 'lon_world_min',
                'lat_world_ease_9km', 'lon_world_ease_9km', 'lat_world_ease_1km', 'lon_world_ease_1km',
                'lat_world_geo_1km', 'lon_world_geo_1km', 'row_world_ease_1km_from_geo_1km_ind',
                'col_world_ease_1km_from_geo_1km_ind']

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
f.close()

# Generate sequence of string between start and end dates (Year + DOY)
start_date = '2015-04-01'
end_date = '2019-10-31'

start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
delta_date = end_date - start_date


########################################################################################################################
# 2. Process SMAP enhanced L2 radiometer half-orbit SM 9 km data

matsize_smap_1day = [len(lat_world_ease_9km), len(lon_world_ease_9km)]
smap_mat_init_1day = np.empty(matsize_smap_1day, dtype='float32')
smap_mat_init_1day[:] = np.nan


for iyr in range(4, len(daysofyear)):

    os.chdir(path_smap + '/' + str(yearname[iyr]))
    smap_files_year = sorted(glob.glob('*.h5'))

    # Group SMAP data by month
    for imo in range(len(monthnum)):

        os.chdir(path_smap + '/' + str(yearname[iyr]))
        smap_files_group_1month = [smap_files_year.index(i) for i in smap_files_year if str(yearname[iyr]) + monthname[imo] in i]

        # Process each month
        if len(smap_files_group_1month) != 0:
            smap_files_month = [smap_files_year[smap_files_group_1month[i]] for i in range(len(smap_files_group_1month))]

            # Create initial empty matrices for monthly SMAP final output data
            matsize_smap = [matsize_smap_1day[0], matsize_smap_1day[1], daysofmonth_seq[imo, iyr]]
            smap_mat_month_am = np.empty(matsize_smap, dtype='float32')
            smap_mat_month_am[:] = np.nan
            smap_mat_month_pm = np.copy(smap_mat_month_am)

            # Extract SMAP data layers and rebind to daily
            for idt in range(daysofmonth_seq[imo, iyr]):
                smap_files_group_1day = [smap_files_month.index(i) for i in smap_files_month if
                                         str(yearname[iyr]) + monthname[imo] + str(idt+1).zfill(2) in i]
                smap_files_1day = [smap_files_month[smap_files_group_1day[i]] for i in
                                    range(len(smap_files_group_1day))]
                smap_files_group_1day_am = [smap_files_1day.index(i) for i in smap_files_1day if
                                         'D_' + str(yearname[iyr]) + monthname[imo] + str(idt+1).zfill(2) in i]
                smap_files_group_1day_pm = [smap_files_1day.index(i) for i in smap_files_1day if
                                         'A_' + str(yearname[iyr]) + monthname[imo] + str(idt+1).zfill(2) in i]
                smap_mat_group_1day = \
                    np.empty([matsize_smap_1day[0], matsize_smap_1day[1], len(smap_files_group_1day)], dtype='float32')
                smap_mat_group_1day[:] = np.nan

                # Read swath files within a day and stack
                for ife in range(len(smap_files_1day)):
                    smap_mat_1file = np.copy(smap_mat_init_1day)
                    fe_smap = h5py.File(smap_files_1day[ife], "r")
                    group_list_smap = list(fe_smap.keys())
                    smap_data_group = fe_smap[group_list_smap[1]]
                    varname_list_smap = list(smap_data_group.keys())
                    # Extract variables
                    col_ind = smap_data_group[varname_list_smap[41]][()]
                    row_ind = smap_data_group[varname_list_smap[10]][()]
                    sm_flag = smap_data_group[varname_list_smap[20]][()]
                    sm = smap_data_group[varname_list_smap[25]][()]
                    sm[np.where(sm == -9999)] = np.nan
                    sm[np.where((sm_flag == 7) & (sm_flag == 15))] = np.nan # Refer to the results of np.binary_repr

                    smap_mat_1file[row_ind, col_ind] = sm
                    smap_mat_group_1day[:, :, ife] = smap_mat_1file
                    print(smap_files_1day[ife])
                    fe_smap.close()

                    del(smap_mat_1file, fe_smap, group_list_smap, smap_data_group, varname_list_smap, col_ind, row_ind,
                        sm_flag, sm)

                smap_mat_1day_am = np.nanmean(smap_mat_group_1day[:, :, smap_files_group_1day_am], axis=2)
                smap_mat_1day_pm = np.nanmean(smap_mat_group_1day[:, :, smap_files_group_1day_pm], axis=2)
                # plt.imshow(np.nanmean(np.concatenate((np.atleast_3d(smap_mat_1day_am),
                # np.atleast_3d(smap_mat_1day_pm)), axis=2), axis=2))
                del(smap_mat_group_1day, smap_files_group_1day, smap_files_1day)

                smap_mat_month_am[:, :, idt] = smap_mat_1day_am
                smap_mat_month_pm[:, :, idt] = smap_mat_1day_pm
                del(smap_mat_1day_am, smap_mat_1day_pm)

            # Save file
            os.chdir(path_procdata)
            var_name = ['smap_sm_9km_am_' + str(yearname[iyr]) + monthname[imo],
                        'smap_sm_9km_pm_' + str(yearname[iyr]) + monthname[imo]]
            data_name = ['smap_mat_month_am', 'smap_mat_month_pm']

            with h5py.File('smap_sm_9km_' + str(yearname[iyr]) + monthname[imo] + '.hdf5', 'w') as f:
                for idv in range(len(var_name)):
                    f.create_dataset(var_name[idv], data=eval(data_name[idv]))
            f.close()
            del(smap_mat_month_am, smap_mat_month_pm)

        else:
            pass