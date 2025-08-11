import h5py
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
import earthpy as et
import earthpy.plot as ep
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from pathlib import Path
from glob import glob
import rasterio as rio
import csv

def read_file(file):
    with rio.open(file) as src:
        return(src.read())

path_smap = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\SMAP_SM_1km\south_africa_monthly\south_africa_monthly'
path_gldas = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_SOILMOISTURE\SMAP_Comparison'

smap_files = path_smap+ '\*.tif'
gldas_files = path_gldas+ '\*.tif'

sm_smap = sorted(list(glob(smap_files)))
sm_gldas =  sorted(list(glob(gldas_files)))

dates= pd.date_range('2015-04-01','2021-01-01' , freq='1M')

shapefile=r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
limpopo = gpd.read_file(shapefile)

#Spatially Averaged Values
monthly_avgs_smap = ['means']
os.chdir(path_smap)
for file in sm_smap:
    array = rxr.open_rasterio(file,masked=True).squeeze()
    clipped = array.rio.clip(limpopo.geometry.apply(mapping), limpopo.crs, drop=True,all_touched=True)
    monthly_avgs_smap.append(np.nanmean(clipped))


monthly_avgs_gldas = ['means']
os.chdir(path_gldas)
for file in sm_gldas:
    array = rxr.open_rasterio(file,masked=True).squeeze()
    monthly_avgs_gldas.append(np.nanmean(array))


#Monthly Time Series
fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(dates,monthly_avgs_smap[1::],color='blue')
ax.set_ylabel('Volumetric Water Content (cm^3/cm^3)',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('SMAP (5cm) Time Series',weight='bold',fontsize=15)

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(dates,monthly_avgs_gldas[1::],'--',color='blue')
ax.set_ylabel('Soil Moisture (kg/m^2)',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('GLDAS (10cm) Time Series',weight='bold',fontsize=15)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(monthly_avgs_smap[1::],monthly_avgs_gldas[1::],'*',color='blue')
ax.set_ylabel('Soil Moisture (kg/m^2)',weight='bold',fontsize=12)
ax.set_xlabel('Volumetric Water Content (cm^3/cm^3)',weight='bold',fontsize=12)
ax.set_title('GLDAS vs SMAP Comparison',weight='bold',fontsize=15)


#Monthly Spatial Plots 

for file1,file2,date in zip(sm_smap,sm_gldas,dates):

    #SMAP
    os.chdir(path_smap)
    array = rxr.open_rasterio(file1,masked=True).squeeze()
    clipped = array.rio.clip(limpopo.geometry.apply(mapping), limpopo.crs, drop=True,all_touched=True)

    lon = np.array(clipped.x)
    lat = np.array(clipped.y)
    count_ticks = 500

    fig,ax = plt.subplots(figsize=(10, 5))
    plt.grid(color='black', linestyle='-', linewidth=0.05)
    image=ax.imshow(clipped, cmap='RdBu',vmin=0,vmax=0.3)

    ax.set_xticks([i for i in range(0,len(lon),count_ticks)]) 
    ax.set_xticklabels(np.int_(np.around(lon[0::count_ticks])))
    ax.set_yticks([i for i in range(0,len(lat),count_ticks)])
    ax.set_yticklabels(reversed((np.int_(np.around(lat[0::count_ticks])))))

    degree_sign = u"\N{DEGREE SIGN}"
    ax.set_xlabel("Longitude ({0} E)".format(degree_sign))
    ax.set_ylabel("Latitude ({0} N)".format(degree_sign))

    cbar = fig.colorbar(image,ax=ax,shrink=0.5)
    cbar.ax.set_ylabel("Volumetric Water Content".format(degree_sign),rotation=270, labelpad=45)
    ax.set_title('Monthly SMAP {}'.format(date), pad=10)

    #GLDAS
    os.chdir(path_gldas)
    array = rxr.open_rasterio(file2,masked=True).squeeze()
    clipped = array.rio.clip(limpopo.geometry.apply(mapping), limpopo.crs, drop=True,all_touched=True)

    lon = np.array(clipped.x)
    lat = np.array(clipped.y)
    count_ticks = 500

    fig,ax = plt.subplots(figsize=(10, 5))
    plt.grid(color='black', linestyle='-', linewidth=0.05)
    image=ax.imshow(clipped, cmap='RdBu',vmin=5,vmax=25)

    ax.set_xticks([i for i in range(0,len(lon),count_ticks)]) 
    ax.set_xticklabels(np.int_(np.around(lon[0::count_ticks])))
    ax.set_yticks([i for i in range(0,len(lat),count_ticks)])
    ax.set_yticklabels(reversed((np.int_(np.around(lat[0::count_ticks])))))

    degree_sign = u"\N{DEGREE SIGN}"
    ax.set_xlabel("Longitude ({0} E)".format(degree_sign))
    ax.set_ylabel("Latitude ({0} N)".format(degree_sign))

    cbar = fig.colorbar(image,ax=ax,shrink=0.5)
    cbar.ax.set_ylabel("Volumetric Water Content".format(degree_sign),rotation=270, labelpad=45)
    ax.set_title('Monthly GLDAS Soil Moisture {}'.format(date), pad=10)
