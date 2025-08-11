import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy.ma as ma
import xarray as xr
import rioxarray as rxr
from shapely.geometry import mapping, box
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from pathlib import Path
import glob
import rasterio as rio

def read_file(file):
    with rio.open(file) as src:
        return(src.read())

##############################
#TEMPERATURE CONDITION INDEX (TCI) -- 7 minutes to run

dates= pd.date_range('2002-07-01','2022-01-01' , freq='1M')
monthpath_lst = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_LST'
lst_month_path = monthpath_lst+'\*.tif'
lst_month_files = sorted(list(glob.glob(lst_month_path)))

os.chdir(monthpath_lst)
lst_mins = []
lst_maxs = []
for file in lst_month_files:
    raster = read_file(file)
    lst_mins.append(np.nanmin(raster))
    lst_maxs.append(np.nanmax(raster))
lst_min = np.nanmin(lst_mins)
lst_max = np.nanmax(lst_maxs)

monthly_avgs_tci = []
for file in lst_month_files:
    raster = read_file(file)
    tci = (lst_max-raster) / (lst_max - lst_min) * 100
    monthly_avgs_tci.append(np.nanmean(tci))

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(dates,monthly_avgs_tci,color='red')
ax.set_ylabel('TCI',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('TCI Time Series',weight='bold',fontsize=15)

##############################
#VEGETATION CONDITION INDEX (VCI) -- 5 minutes to run

#09/03/22: Sampled with Aqua NDVI; requested longer dataset for NDVI via AppEEARS for Aqua & Terra
#          Update when ready

dates= pd.date_range('2003-02-01','2022-01-01' , freq='1M')
monthpath_ndvi = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_NDVI\MODMYD'
ndvi_month_path = monthpath_ndvi+'\*.tif'
ndvi_month_files = sorted(list(glob.glob(ndvi_month_path)))[7:-8]

os.chdir(monthpath_ndvi)
ndvi_mins = []
ndvi_maxs = []
for file in ndvi_month_files:
    raster = read_file(file)
    ndvi_mins.append(np.nanmin(raster))
    ndvi_maxs.append(np.nanmax(raster))

ndvi_min = np.nanmin(ndvi_mins)
ndvi_max = np.nanmax(ndvi_maxs)

monthly_avgs_vci = []
for file in ndvi_month_files:
    raster = read_file(file)
    vci = (raster-ndvi_min) / (ndvi_max - ndvi_min) * 100
    monthly_avgs_vci.append(np.nanmean(vci))

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(dates,monthly_avgs_vci,color='green')
ax.set_ylabel('VCI',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('VCI Time Series',weight='bold',fontsize=15)

##############################
#VEGETATION HEALTH INDEX (VHI) -- 30 s to run

#09/03/22: edited index for files to adjust date discrepency; wait until NDVI is all downloaded

#MONTHLY
monthly_avgs_vhi = []
for file1,file2 in zip(lst_month_files,ndvi_month_files):
    os.chdir(monthpath_lst)
    raster1 = read_file(file1)
    os.chdir(monthpath_ndvi)
    raster2 = read_file(file2)
    tci = (lst_max-raster1) / (lst_max - lst_min) * 100
    vci = (raster2-ndvi_min) / (ndvi_max - ndvi_min) * 100
    vhi = 0.5*vci + ((1-0.5)*tci)
    monthly_avgs_vhi.append(np.nanmean(vhi))

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(dates,monthly_avgs_vhi,color='orange')
ax.set_ylabel('VHI',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('VHI Time Series',weight='bold',fontsize=15)



################################
#ALL INDICES
fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(dates,monthly_avgs_vhi,color='orange')
ax.plot(dates,monthly_avgs_vci,color='green')
ax.plot(dates,monthly_avgs_tci,color='red')
ax.set_ylabel('Index',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)
ax.legend()

