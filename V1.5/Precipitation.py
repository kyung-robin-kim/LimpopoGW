import pandas as pd
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import h5py
from shapely.geometry import box, mapping
from rasterio import features
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from rasterstats import zonal_stats
import csv
import os
from pathlib import Path
import glob
import seaborn as sns
import netCDF4
import salem
import skimage
from pydap.client import open_url
import netCDF4
from itertools import chain
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import re
import requests
import time
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import rasterstats as rs
import gc
from rasterio.enums import Resampling
from datetime import datetime
import statsmodels as sm 
import statsmodels.graphics.tsaplots as tsaplots
import statsmodels.api as smapi
from sklearn.linear_model import LinearRegression

f, ax = plt.subplots(figsize=(90,50))
gpm[0].where(gpm[0]<16).plot(ax=ax)
precip_points_filtered.iloc[[0]].plot(ax=ax,color='black',markersize=70)
plt.show()

def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)


def plot_np(array,vmin,vmax,title):
    array = np.where(array==0,np.nan,array)
    fig1, ax1 = plt.subplots(figsize=(20,16))
    image = ax1.imshow(array,cmap='RdBu',vmin=vmin,vmax=vmax)
    cbar = fig1.colorbar(image,ax=ax1)
    ax1.set_title('{}'.format(title))


############################################
#MONTHLY Spatial Statistics

path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\PRECIPITATION'
files = sorted(glob.glob(path+'\*.nc'))

p_gpm = xr.open_mfdataset(files[1],parallel=True,chunks={"lat": 100,"lon":100})
p_chirps = xr.open_mfdataset(files[0],parallel=True,chunks={"latitude": 100,"longitude":100})

dates = p_chirps.time #dates.dt.strftime('%Y-%m')

mean_gpm = p_gpm.P_mm.mean(dim=['x','y']).to_dataframe()
mean_chirps = p_chirps.P_mm.mean(dim=['x','y']).to_dataframe()

sum_gpm = p_gpm.P_mm.sum(dim=['x','y']).to_dataframe()
sum_chirps = p_chirps.P_mm.sum(dim=['x','y']).to_dataframe()


#Mean
fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(dates,p_gpm.P_mm.mean(dim=['x','y']),color='C1')
ax.plot(dates,p_chirps.P_mm.mean(dim=['x','y']),color='C0')
ax.set_ylabel('Precipitation (mm)',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)

#Sum
fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(dates,p_gpm.P_mm.sum(dim=['x','y']),color='C1')
ax.plot(dates,p_chirps.P_mm.sum(dim=['x','y']),color='C0')
ax.set_ylabel('Precipitation (mm)',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)

#################
#Sophia's InSitu Rainfall Data
filepath = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\sophia\Limpopo In-Situ Data\Precipitation'
xl_files = sorted(glob.glob(filepath+'//*.xlsx'))
csv_files = sorted(glob.glob(filepath+'//*.csv'))

#Select which stations have data AND are in LRB
station_id = pd.read_excel(xl_files[0], sheet_name='Stationdetails')
precip_data = pd.read_excel(xl_files[0], sheet_name='_2000_2021Rainfall')
station_codes = sorted(precip_data['STATION'].unique()) #Not all are in Limpopo River Basin

filtered_station_id = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\validation_points\precip_sites_filtered.shp'
station_codes_lrb = sorted(gpd.read_file(filtered_station_id)['CODE'])

filter = np.isin(station_codes_lrb,station_codes)
station_codes_lrb = np.array(station_codes_lrb)[filter]


#Extract precip data by station ID and replace ID to datetime
precip_by_station = [precip_data[precip_data.STATION == i] for i in station_codes_lrb]
for i in range(0,len(precip_by_station)):
    precip_by_station[i].index = pd.to_datetime(precip_by_station[i]['DATE'])


#Plot timeseries
#for i in range(0,len(precip_by_station)):
    plt.figure()
    precip_by_station[i]['MM'].plot(title=precip_by_station[i]['STATION'][0])
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\sophia\Limpopo In-Situ Data\Precipitation\plots\{}'.format(i))


#Sum up by month accumulations
precip_by_station_month_sum = [station.groupby(pd.Grouper(freq='M')).sum() for station in precip_by_station]
#For filtering out months with too many NaNs
NAN_count_precip_month_sum = [pd.isna(station['MM']).groupby(pd.Grouper(freq='M')).sum() for station in precip_by_station]

#Filter out data that is too old
last_date_data = [data.index[-1] for data in precip_by_station_month_sum]
valid_sites = [last_date > pd.to_datetime('2001-01') for last_date in last_date_data]
valid_sites = np.where(valid_sites)

precip_monthly_filtered = [precip_by_station_month_sum[i] for i in valid_sites[0]]


#Plot CHIRPS & GPM comparisons with point validations
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\PRECIPITATION'
files = sorted(glob.glob(path+"/*.nc"))

chirps = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 25,"lon":25}).P_mm
gpm = xr.open_mfdataset(files[1],parallel=True,chunks={"lat": 5,"lon":5}).P_mm

precip_points = gpd.read_file(filtered_station_id)
precip_points_filtered = precip_points[precip_points.isin(station_codes_lrb)['CODE']].reset_index()


#################################################################################################
#Extract point values from raster
#OPTION A -- use rasterstats point_query function -- use nearest (vs. bilinear)
monthly_chirps = [rs.point_query(precip_points_filtered, np.array(chirps[i]), affine= chirps[i].rio.transform(), geojson_out=True,interpolate='nearest') for i in range(0,len(chirps))]
time_series_by_month_chirps = np.array([ [monthly_chirps[month][site]['properties']['value'] for site in range(0,len(precip_points_filtered))] for month in range(0,len(monthly_chirps))])
time_series_by_site_CHIRPS = time_series_by_month_chirps.T

monthly_gpm = [rs.point_query(precip_points_filtered, np.array(gpm[i]), affine= gpm[i].rio.transform(), geojson_out=True, interpolate='nearest') for i in range(0,len(gpm))]
time_series_by_month_gpm = np.array([ [monthly_gpm[month][site]['properties']['value'] for site in range(0,len(precip_points_filtered))] for month in range(0,len(monthly_gpm))])
time_series_by_site_GPM = time_series_by_month_gpm.T


#OPTION B -- use rio.open.read and .index based on gpd.xy
#extract x,y points of validation sites
x = [point.xy[0][0] for point in precip_points_filtered['geometry']]
y = [point.xy[1][0] for point in precip_points_filtered['geometry']]
#call rio.open()
chirps_rio = rio.open(files[0])
gpm_rio = rio.open(files[1])
#map function to get row/col indices of x,y points
rows_chirps,cols_chirps = map(list,zip(*[chirps_rio.index(lon,lat) for lon,lat in zip(x,y)]))
rows_gpm,cols_gpm = map(list,zip(*[gpm_rio.index(lon,lat) for lon,lat in zip(x,y)]))

#loop through indices of each month in .nc file to get point value in each dataset - last 
chirps_values = [[chirps_rio.read(i)[row,col] for row,col in zip(rows_chirps,cols_chirps)] for i in range(1,241)]
gpm_values = [[gpm_rio.read(i)[row,col] for row,col in zip(rows_gpm,cols_gpm)] for i in range(1,241)]

monthly_chirps = np.array([[chirps_values[month][site] for site in range(0,len(precip_points_filtered))] for month in range(0,len(chirps_values))])
monthly_chirps_site = monthly_chirps.T

monthly_gpm = np.array([[gpm_values[month][site] for site in range(0,len(precip_points_filtered))] for month in range(0,len(gpm_values))])
monthly_gpm_site = monthly_gpm.T

#2000 and after
plt.rc('font', size = 10)
for i,site in zip(range(0,len(precip_monthly_filtered)),precip_points_filtered['CODE']):
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time,time_series_by_site_CHIRPS[i],label='CHIRPS')
    ax2.plot(gpm.time,time_series_by_site_GPM[i],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_monthly_filtered[i].index,precip_monthly_filtered[i]['MM'],'*',color='C2')
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,500)
    ax1.set_title('{}'.format(site))
    ax2.legend()
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\precip_validation\{}_A'.format(site))


#Determine common time window frame for comparison
prestart_dates_insitu_missing = [gpm.time.to_index().difference(precip.index) for precip in precip_monthly_filtered]
postend_dates_insitu_missing = [gpm.time.to_index().difference(precip.index) for precip in precip_monthly_filtered]

prestart_dates_gpm_missing = [precip.index.difference(gpm.time.to_index()) for precip in precip_monthly_filtered]
postend_dates_gpm_missing = [precip.index.difference(gpm.time.to_index()) for precip in precip_monthly_filtered]

def linear_plot(independent,dependent,ind_label,d_label,color_choice):
    y = np.array(dependent)
    x = np.array(independent)
    X = smapi.add_constant(x)
    est = smapi.OLS(y, X)
    model = est.fit()
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot()
    ax.scatter(independent,dependent,color=color_choice)
    ax.plot([], [], ' ', label='R{} = {}'.format(get_super('2'),round(model.rsquared, 3)))
    ax.plot([], [], ' ', label='p-val = {}'.format(round(model.pvalues[-1], 3)))
    ax.legend(loc='upper right')
    ax.set_xlabel(ind_label)
    ax.set_ylabel(d_label)


#PRECIPITATION VALIDATION
#Plots different indices are: GIR (0), NYA (6), and SHM (12); others are consistent
#Others are the same: GPM/CHIRPS end 9 months earlier, In-Situ start 24 months late

#NOT A REAL FOR LOOP -- just to collapse all code...
for validation in precipitation:
    
    #GIR - 0
    #OPTION A
    i=0
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time[60:60+len(precip_monthly_filtered[i].index)],time_series_by_site_CHIRPS[i][60:60+len(precip_monthly_filtered[i].index)],label='CHIRPS')
    ax2.plot(gpm.time[60:60+len(precip_monthly_filtered[i].index)],time_series_by_site_GPM[i][60:60+len(precip_monthly_filtered[i].index)],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_monthly_filtered[i].index,precip_monthly_filtered[i]['MM'],color='C2',linewidth=2)
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,400)
    ax2.legend()

    linear_plot(time_series_by_site_GPM[i][60:60+len(precip_monthly_filtered[i].index)],precip_monthly_filtered[i]['MM'],'GPM','In-Situ','blue')
    linear_plot(time_series_by_site_CHIRPS[i][60:60+len(precip_monthly_filtered[i].index)],precip_monthly_filtered[i]['MM'],'CHIRPS','In-Situ','blue')

    #OPTION B
    i=0
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time[60:60+len(precip_monthly_filtered[i].index)],monthly_chirps_site[i][60:60+len(precip_monthly_filtered[i].index)],label='CHIRPS')
    ax2.plot(gpm.time[60:60+len(precip_monthly_filtered[i].index)],monthly_gpm_site[i][60:60+len(precip_monthly_filtered[i].index)],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_monthly_filtered[i].index,precip_monthly_filtered[i]['MM'],color='C2',linewidth=2)
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,400)
    ax2.legend()

    linear_plot(monthly_gpm_site[i][60:60+len(precip_monthly_filtered[i].index)],precip_monthly_filtered[i]['MM'],'GPM','In-Situ','blue')
    linear_plot(monthly_chirps_site[i][60:60+len(precip_monthly_filtered[i].index)],precip_monthly_filtered[i]['MM'],'CHIRPS','In-Situ','blue')


    #HOU
    i=1
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time[:-9],time_series_by_site_CHIRPS[i][:-9],label='CHIRPS')
    ax2.plot(gpm.time[:-9],time_series_by_site_GPM[i][:-9],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_monthly_filtered[i].index[24:],precip_monthly_filtered[i]['MM'][24:],color='C2',linewidth=2)
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,400)
    ax2.legend()

    linear_plot(time_series_by_site_GPM[i][:-9],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(time_series_by_site_CHIRPS[i][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')


    i=1
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time[:-9],monthly_chirps_site[0][:-9],label='CHIRPS')
    ax2.plot(gpm.time[:-9],monthly_gpm_site[0][:-9],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_monthly_filtered[i].index[24:],precip_monthly_filtered[i]['MM'][24:],color='C2',linewidth=2)
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,400)
    ax2.legend()

    linear_plot(monthly_gpm_site[0][:-9],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(monthly_chirps_site[0][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')




    #KFI
    i=2
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time[:-9],time_series_by_site_CHIRPS[i][:-9],label='CHIRPS')
    ax2.plot(gpm.time[:-9],time_series_by_site_GPM[i][:-9],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_monthly_filtered[i].index[24:],precip_monthly_filtered[i]['MM'][24:],color='C2',linewidth=2)
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,400)
    ax2.legend()


    linear_plot(time_series_by_site_CHIRPS[i][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')

    i=2
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time[:-9],monthly_chirps_site[0][:-9],label='CHIRPS')
    ax2.plot(gpm.time[:-9],monthly_gpm_site[0][:-9],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_monthly_filtered[i].index[24:],precip_monthly_filtered[i]['MM'][24:],color='C2',linewidth=2)
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,400)
    ax2.legend()

    linear_plot(monthly_gpm_site[0][:-9],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(monthly_chirps_site[0][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')


    #LET
    i=3
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time[:-9],time_series_by_site_CHIRPS[i][:-9],label='CHIRPS')
    ax2.plot(gpm.time[:-9],time_series_by_site_GPM[i][:-9],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_monthly_filtered[i].index[24:],precip_monthly_filtered[i]['MM'][24:],color='C2',linewidth=2)
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,400)
    ax2.legend()

    linear_plot(time_series_by_site_GPM[i][:-9],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(time_series_by_site_CHIRPS[i][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')

    i=3
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time[:-9],monthly_chirps_site[0][:-9],label='CHIRPS')
    ax2.plot(gpm.time[:-9],monthly_gpm_site[0][:-9],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_monthly_filtered[i].index[24:],precip_monthly_filtered[i]['MM'][24:],color='C2',linewidth=2)
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,400)
    ax2.legend()

    linear_plot(monthly_gpm_site[0][:-9],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(monthly_chirps_site[0][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')



    #MAH
    i=4
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time[:-9],time_series_by_site_CHIRPS[i][:-9],label='CHIRPS')
    ax2.plot(gpm.time[:-9],time_series_by_site_GPM[i][:-9],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_monthly_filtered[i].index[24:],precip_monthly_filtered[i]['MM'][24:],color='C2',linewidth=2)
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,400)
    ax2.legend()

    linear_plot(time_series_by_site_GPM[i][:-9],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(time_series_by_site_CHIRPS[i][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')

    i=4
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time[:-9],monthly_chirps_site[0][:-9],label='CHIRPS')
    ax2.plot(gpm.time[:-9],monthly_gpm_site[0][:-9],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_monthly_filtered[i].index[24:],precip_monthly_filtered[i]['MM'][24:],color='C2',linewidth=2)
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,400)
    ax2.legend()

    linear_plot(monthly_gpm_site[0][:-9],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(monthly_chirps_site[0][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')


    #MOO
    i=5
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time[:-9],time_series_by_site_CHIRPS[i][:-9],label='CHIRPS')
    ax2.plot(gpm.time[:-9],time_series_by_site_GPM[i][:-9],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_monthly_filtered[i].index[24:],precip_monthly_filtered[i]['MM'][24:],color='C2',linewidth=2)
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,400)
    ax2.legend()

    linear_plot(time_series_by_site_GPM[i][:-9],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(time_series_by_site_CHIRPS[i][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')

    i=5
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time[:-9],monthly_chirps_site[0][:-9],label='CHIRPS')
    ax2.plot(gpm.time[:-9],monthly_gpm_site[0][:-9],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_monthly_filtered[i].index[24:],precip_monthly_filtered[i]['MM'][24:],color='C2',linewidth=2)
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,400)
    ax2.legend()

    linear_plot(monthly_gpm_site[0][:-9],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(monthly_chirps_site[0][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')


    #NYA
    i=6
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time[60:60+len(precip_monthly_filtered[i].index)],time_series_by_site_CHIRPS[i][60:60+len(precip_monthly_filtered[i].index)],label='CHIRPS')
    ax2.plot(gpm.time[60:60+len(precip_monthly_filtered[i].index)],time_series_by_site_GPM[i][60:60+len(precip_monthly_filtered[i].index)],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_monthly_filtered[i].index,precip_monthly_filtered[i]['MM'],color='C2',linewidth=2)
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,400)
    ax2.legend()

    linear_plot(time_series_by_site_GPM[i][60:60+len(precip_monthly_filtered[i].index)],precip_monthly_filtered[i]['MM'],'GPM','In-Situ','blue')
    linear_plot(time_series_by_site_CHIRPS[i][60:60+len(precip_monthly_filtered[i].index)],precip_monthly_filtered[i]['MM'],'CHIRPS','In-Situ','blue')

    i=6
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time[60:60+len(precip_monthly_filtered[i].index)],monthly_chirps_site[0][60:60+len(precip_monthly_filtered[i].index)],label='CHIRPS')
    ax2.plot(gpm.time[60:60+len(precip_monthly_filtered[i].index)],monthly_gpm_site[0][60:60+len(precip_monthly_filtered[i].index)],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_monthly_filtered[i].index,precip_monthly_filtered[i]['MM'],color='C2',linewidth=2)
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,400)
    ax2.legend()

    linear_plot(monthly_gpm_site[0][60:60+len(precip_monthly_filtered[i].index)],precip_monthly_filtered[i]['MM'],'GPM','In-Situ','blue')
    linear_plot(monthly_chirps_site[0][60:60+len(precip_monthly_filtered[i].index)],precip_monthly_filtered[i]['MM'],'CHIRPS','In-Situ','blue')


    #OLI
    i=7
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time[:-9],time_series_by_site_CHIRPS[i][:-9],label='CHIRPS')
    ax2.plot(gpm.time[:-9],time_series_by_site_GPM[i][:-9],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_monthly_filtered[i].index[24:],precip_monthly_filtered[i]['MM'][24:],color='C2',linewidth=2)
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,400)
    ax2.legend()

    linear_plot(time_series_by_site_GPM[i][:-9],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(time_series_by_site_CHIRPS[i][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')

    linear_plot(monthly_gpm_site[i][:-9],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(monthly_chirps_site[i][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')


    #PAP
    i=8
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time[:-9],time_series_by_site_CHIRPS[i][:-9],label='CHIRPS')
    ax2.plot(gpm.time[:-9],time_series_by_site_GPM[i][:-9],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_monthly_filtered[i].index[24:],precip_monthly_filtered[i]['MM'][24:],color='C2',linewidth=2)
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,400)
    ax2.legend()

    linear_plot(time_series_by_site_GPM[i][:-9],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(time_series_by_site_CHIRPS[i][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')

    linear_plot(monthly_gpm_site[i][:-9],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(monthly_chirps_site[i][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')

    #PHA
    i=9
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time[:-9],time_series_by_site_CHIRPS[i][:-9],label='CHIRPS')
    ax2.plot(gpm.time[:-9],time_series_by_site_GPM[i][:-9],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_monthly_filtered[i].index[24:],precip_monthly_filtered[i]['MM'][24:],color='C2',linewidth=2)
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,400)
    ax2.legend()

    linear_plot(time_series_by_site_GPM[i][:-9],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(time_series_by_site_CHIRPS[i][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')

    linear_plot(monthly_gpm_site[i][:-9],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(monthly_chirps_site[i][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')

    #PUN
    i=10
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time[:-9],time_series_by_site_CHIRPS[i][:-9],label='CHIRPS')
    ax2.plot(gpm.time[:-9],time_series_by_site_GPM[i][:-9],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_monthly_filtered[i].index[24:],precip_monthly_filtered[i]['MM'][24:],color='C2',linewidth=2)
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,400)
    ax1.set_title('{}'.format(site))
    ax2.legend()

    linear_plot(time_series_by_site_GPM[i][:-9],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(time_series_by_site_CHIRPS[i][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')

    linear_plot(monthly_gpm_site[i][:-9],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(monthly_chirps_site[i][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')

    #SHA
    i=11
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time[:-9],time_series_by_site_CHIRPS[i][:-9],label='CHIRPS')
    ax2.plot(gpm.time[:-9],time_series_by_site_GPM[i][:-9],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_monthly_filtered[i].index[24:],precip_monthly_filtered[i]['MM'][24:],color='C2',linewidth=2)
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,400)
    ax2.legend()

    linear_plot(time_series_by_site_GPM[i][:-9],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(time_series_by_site_CHIRPS[i][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')

    linear_plot(monthly_gpm_site[i][:-9],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(monthly_chirps_site[i][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')

    #SHM
    i=12
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time[:-187],time_series_by_site_CHIRPS[i][:-187],label='CHIRPS')
    ax2.plot(gpm.time[:-187],time_series_by_site_GPM[i][:-187],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_monthly_filtered[i].index[24:],precip_monthly_filtered[i]['MM'][24:],color='C2',linewidth=2)
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,400)
    ax2.legend()

    linear_plot(time_series_by_site_GPM[i][:-187],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(time_series_by_site_CHIRPS[i][:-187],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')

    linear_plot(monthly_gpm_site[i][:-187],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(monthly_chirps_site[i][:-187],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')

    #VLA
    i=13
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time[:-9],time_series_by_site_CHIRPS[i][:-9],label='CHIRPS')
    ax2.plot(gpm.time[:-9],time_series_by_site_GPM[i][:-9],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_monthly_filtered[i].index[24:],precip_monthly_filtered[i]['MM'][24:],color='C2',linewidth=2)
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,400)
    ax2.legend()

    linear_plot(time_series_by_site_GPM[i][:-9],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(time_series_by_site_CHIRPS[i][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')

    linear_plot(monthly_gpm_site[i][:-9],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(monthly_chirps_site[i][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')


    #WOO
    i=14
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time[:-9],time_series_by_site_CHIRPS[i][:-9],label='CHIRPS')
    ax2.plot(gpm.time[:-9],time_series_by_site_GPM[i][:-9],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_monthly_filtered[i].index[24:],precip_monthly_filtered[i]['MM'][24:],color='C2',linewidth=2)
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,400)
    ax2.legend()

    linear_plot(time_series_by_site_GPM[i][:-9],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(time_series_by_site_CHIRPS[i][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')

    linear_plot(monthly_gpm_site[i][:-9],precip_monthly_filtered[i]['MM'][24:],'GPM','In-Situ','blue')
    linear_plot(monthly_chirps_site[i][:-9],precip_monthly_filtered[i]['MM'][24:],'CHIRPS','In-Situ','blue')


    #Xai-Xai in MZ
    precip_MZ_data = pd.read_csv(csv_files[-2])
    precip_MZ_data.index = pd.to_datetime(precip_MZ_data['DATE'])
    precip_MZ_data_monthly = precip_MZ_data.groupby(pd.Grouper(freq='M')).sum()['PRCP']
    NaN_count_precip_MZ_data_monthly = pd.isna(precip_MZ_data['PRCP']).groupby(pd.Grouper(freq='M')).sum()

    #plt.figure()
    #precip_MZ_data_monthly.plot(title='XaiXai')
    #plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\sophia\Limpopo In-Situ Data\Precipitation\plots\{}'.format('XaiXai'))

    MZ_station_id = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\validation_points\precip_sites_MZ.shp'
    MZ_precip_points = gpd.read_file(MZ_station_id)

    x = [point.xy[0][0] for point in MZ_precip_points['geometry']]
    y = [point.xy[1][0] for point in MZ_precip_points['geometry']]
    #map function to get row/col indices of x,y points
    rows_chirps,cols_chirps = map(list,zip(*[chirps_rio.index(lon,lat) for lon,lat in zip(x,y)]))
    rows_gpm,cols_gpm = map(list,zip(*[gpm_rio.index(lon,lat) for lon,lat in zip(x,y)]))
    #loop through indices of each month in .nc file to get point value in each dataset - last 
    chirps_values = [[chirps_rio.read(i)[row,col] for row,col in zip(rows_chirps,cols_chirps)] for i in range(1,241)]
    gpm_values = [[gpm_rio.read(i)[row,col] for row,col in zip(rows_gpm,cols_gpm)] for i in range(1,241)]

    monthly_chirps = np.array([[chirps_values[month][site] for site in range(0,len(MZ_precip_points))] for month in range(0,len(chirps_values))])
    monthly_chirps_site = monthly_chirps.T

    monthly_gpm = np.array([[gpm_values[month][site] for site in range(0,len(MZ_precip_points))] for month in range(0,len(gpm_values))])
    monthly_gpm_site = monthly_gpm.T


    site='XaiXai'
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time,monthly_chirps_site[0],label='CHIRPS')
    ax2.plot(gpm.time,monthly_gpm_site[0],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_MZ_data_monthly.index,precip_MZ_data_monthly,'*',color='C2')
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,500)
    ax1.set_title('{}'.format(site))
    ax2.legend()
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\precip_validation\{}_B_only'.format(site))


    #MZB - XaiXai
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(chirps.time[95:95+len(precip_MZ_data_monthly.index)],time_series_by_site_CHIRPS[i][95:95+len(precip_MZ_data_monthly.index)],label='CHIRPS')
    ax2.plot(gpm.time[95:95+len(precip_MZ_data_monthly.index)],time_series_by_site_GPM[i][95:95+len(precip_MZ_data_monthly.index)],label='GPM')
    ax2.set_ylabel('Gridded Precip (mm)')
    ax2.set_ylim(0,400)
    ax1.plot(precip_MZ_data_monthly.index,precip_MZ_data_monthly,color='C2',linewidth=2)
    ax1.set_ylabel('In-Situ (mm)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0,400)
    ax2.legend()

    linear_plot(monthly_chirps_site[i][95:95+len(precip_MZ_data_monthly.index)],precip_MZ_data_monthly,'CHIRPS','In-Situ','blue')
    linear_plot(monthly_gpm_site[i][95:95+len(precip_MZ_data_monthly.index)],precip_MZ_data_monthly,'GPM','In-Situ','blue')



#####################
#Date Reference
days_count = {'month': ['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], 
            'month_no': ['01', '02', '03','04','05','06','07','08','09','10','11','12'],
            'non_leap': [31,28,31,30,31,30,31,31,30,31,30,31], 
            'leap': [31,29,31,30,31,30,31,31,30,31,30,31]}
days_df = pd.DataFrame(data=days_count)

#Use the dataframe to multiply no. of days & 24 hours to get accumulated monthly precipitation 

#days_df.loc[days_df['month_no']==file[24:26]]['non_leap'] #number of days in a given month for non-leap year
#days_df.loc[days_df['month_no']==file[24:26]]['leap'] #number of days in a given month for leap year

#####################
#Shapefiles
shpname = r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
shapefile = gpd.read_file(shpname)

#Data
path_gpm = r'D:\raw-data\GPM_IMERG'
#gpm_files = sorted(glob.glob(path+"/*.nc4"))

path_chirps = r'D:\raw-data\CHIRPS'
#chirps_files = sorted(glob.glob(path+"/*.nc"))

############################################
#DAILY Spatial Statistics

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(ti_chirps.time,precip_shapefile_means[0],color='C1')
ax.plot(ti_gpm,precip_shapefile_means[1],color='C0')
ax.set_ylabel('Precipitation (mm)',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Mean Time Series',weight='bold',fontsize=15)

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(ti_chirps.time,precip_shapefile_sums[0],color='C1')
ax.plot(ti_gpm,precip_shapefile_sums[1],color='C0')
ax.set_ylabel('Precipitation (mm)',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Sum Time Series',weight='bold',fontsize=15)

for year in range(2002,2022):

    year=2021

    gpm_files = sorted(glob.glob(path_gpm+"//*{}????-*.nc4".format(year)))
    chirps_files= sorted(glob.glob(path_chirps+"//*{}*.nc".format(year)))

    p_gpm = xr.open_mfdataset(gpm_files,parallel=True,chunks={"lat": 100,"lon":100}).precipitationCal.transpose('time', 'lat', 'lon')
    p_chirps = xr.open_mfdataset(chirps_files,parallel=True,chunks={"latitude": 100,"longitude":100}).rename({'latitude':'lat','longitude':'lon'}).precip

    precips_shapefile = [p.rio.set_spatial_dims('lon','lat',inplace=True).rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True) 
    for p in [p_chirps,p_gpm]]
    
    precip_shapefile_sums = [ precip.sum(dim={'lon','lat'}) for precip in precips_shapefile]
    precip_shapefile_means = [precip.mean(dim={'lon','lat'}) for precip in precips_shapefile]

    ti_chirps = precip_shapefile_sums[0].time
    ti_gpm = precip_shapefile_means[1].indexes['time'].to_datetimeindex()

    times = [ti_chirps,ti_gpm]
    data_names = ['CHIRPS', 'GPM IMERG']

    #SUMS
    plt.rcParams.update({'font.size': 20})
    fig,ax = plt.subplots(figsize=(9,6))
    for sums,ti,data in zip(precip_shapefile_sums,times,data_names):

        if data == 'CHIRPS':
            ax.bar(ti,sums,  linewidth=2, color='#56B4E9', label='{}'.format(data)) #blue

        if data == 'GPM IMERG':
            ax.bar(ti,sums, linewidth = 2, color='#D55E00', label='{}'.format(data)) #orange

    ax.set_xlim(times[0][0],times[0][-1])
    ax.tick_params(axis='x', rotation=45)
    #ax.set_title('Limpopo River Basin')
    #ax.set_ylim(0,20000)
    ax.set_ylabel('Daily Precip Accum. (mm)', fontsize=18,weight='bold',labelpad=2)

    #ax.legend(bbox_to_anchor=(0.05, 1.05),loc='center left',fontsize=12)
    fig.tight_layout()
    fig.suptitle('{} Spatial Sums'.format(year))
    fig.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\precipitation\Daily\SUM\{}.png'.format(year),dpi=200)

    #MEANS
    plt.rcParams.update({'font.size': 20})
    fig,ax = plt.subplots(figsize=(9,6))
    for means,ti,data in zip(precip_shapefile_means,times,data_names):

        if data == 'CHIRPS':
            ax.plot(ti,means,  linewidth=2, color='#56B4E9', label='{}'.format(data)) #blue

        if data == 'GPM IMERG':
            ax.plot(ti,means, linewidth = 2, color='#D55E00', label='{}'.format(data))  #orange

    ax.set_xlim(times[0][0],times[0][-1])
    ax.tick_params(axis='x', rotation=45)
    #ax.set_title('Limpopo River Basin')
    #ax.set_ylim(0,50)
    ax.set_ylabel('Daily Precip Accum. (mm)', fontsize=18,weight='bold',labelpad=2)

    #ax.legend(bbox_to_anchor=(0.05, 1.05),loc='center left',fontsize=12)
    fig.tight_layout()
    fig.suptitle('{} Spatial Means'.format(year))
    fig.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\precipitation\Daily\MEAN\{}.png'.format(year),dpi=200)


############################################
#Extract Precipitation Centroids 
maindir= r'C:\Users\robin\Box\HMA\Data\Climatology'

#Shapefiles
shppath = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\validation_points\*.shp'
files = sorted(glob.glob(shppath))

myagdi_centroids = r'C:\Users\robin\Desktop\HMA_SWAT\KaliGandaki\00dataPreparation\centroids\data_points_myagdi.shp'
modi_centroids = r'C:\Users\robin\Desktop\HMA_SWAT\KaliGandaki\00dataPreparation\centroids\data_points_modi.shp'
catchments = [myagdi_centroids, modi_centroids]
centroids = [gpd.read_file(centroid) for centroid in catchments]

myagdi_centroids = centroids[0]
modi_centroids = centroids[1]


#DID NOT CORRECT daily CHELSA precip w/ MONTHLY PBCOR (10-05-22) -- apply as needed later on for monthly estimates
#Must TRANSPOSE GPM_IMERG data with .transpose('time','lat','lon')
for year in range(2000,2017):

    gpm_files = sorted(glob.glob(path_gpm+"//*{}????-*.nc4".format(year)))
    chirps_files= sorted(glob.glob(path_chirps+"//*{}*.nc".format(year)))
    chelsa_files = sorted(glob.glob(path_chelsa+"//*pr*{}*.nc".format(year)))
    pbcor_file = sorted(glob.glob(path_PBCOR+"//CHELSA_V12.nc"))

    p_gpm = xr.open_mfdataset(gpm_files,parallel=True,chunks={"lat": 100,"lon":100}).precipitationCal.transpose('time', 'lat', 'lon')
    p_chirps = xr.open_mfdataset(chirps_files,parallel=True,chunks={"latitude": 100,"longitude":100}).rename({'latitude':'lat','longitude':'lon'}).precip
    p_chelsa = xr.open_mfdataset(chelsa_files,parallel=True,chunks={"lat": 100,"lon":100}).pr * 86400 #convert to mm/day from sec
    pbcor = xr.open_mfdataset(pbcor_file,parallel=True,chunks={"lat": 100,"lon":100})

    precips_myagdi = [p.rio.set_spatial_dims('lon','lat',inplace=True).rio.write_crs('WGS84').rio.clip(myagdi.geometry.apply(mapping), myagdi.crs, drop=True,all_touched=True).rio.reproject(myagdi.crs)
    for p in [p_chelsa,p_chirps,p_gpm]]
    precips_modi = [p.rio.set_spatial_dims('lon','lat',inplace=True).rio.write_crs('WGS84').rio.clip(modi.geometry.apply(mapping), modi.crs, drop=True,all_touched=True).rio.reproject(modi.crs) 
    for p in [p_chelsa,p_chirps,p_gpm]]

    ##Extract Data
    #Myagdi 
    dps_chelsa_mya = [rs.point_query(myagdi_centroids,np.array(precips_myagdi[0][i]), affine= precips_myagdi[0].rio.transform(), geojson_out=True) for i in range(0,len(precips_myagdi[0]))]
    dps_chirps_mya =[rs.point_query(myagdi_centroids,np.array(precips_myagdi[1][i]), affine= precips_myagdi[1].rio.transform(), geojson_out=True) for i in range(0,len(precips_myagdi[1]))]
    dps_gpm_mya = [rs.point_query(myagdi_centroids, np.array(precips_myagdi[2][i]), affine= precips_myagdi[2].rio.transform(), geojson_out=True) for i in range(0,len(precips_myagdi[2]))]
    #Modi
    dps_chelsa_mod = [rs.point_query(modi_centroids, np.array(precips_modi[0][i]), affine= precips_modi[0].rio.transform(), geojson_out=True) for i in range(0,len(precips_modi[0]))]
    dps_chirps_mod =[rs.point_query(modi_centroids, np.array(precips_modi[1][i]), affine= precips_modi[1].rio.transform(), geojson_out=True) for i in range(0,len(precips_modi[1]))]
    dps_gpm_mod = [rs.point_query(modi_centroids, np.array(precips_modi[2][i]), affine= precips_modi[2].rio.transform(), geojson_out=True) for i in range(0,len(precips_modi[2]))]

    ##Convert and Save to CSV Dataframes
    dps_chelsa_mya_dfs = [gpd.GeoDataFrame.from_features(datapoints).to_csv(maindir+'/Myagdi/Precip/{}/CHELSA/{:03}.csv'.format(year,day))
    for datapoints,day in zip(dps_chelsa_mya,range(0,len(precips_myagdi[0])))]
    dps_chirps_mya_dfs = [gpd.GeoDataFrame.from_features(datapoints).to_csv(maindir+'/Myagdi/Precip/{}/CHIRPS/{:03}.csv'.format(year,day))
    for datapoints,day in zip(dps_chirps_mya,range(0,len(precips_myagdi[1])))]
    dps_gpm_mya_dfs = [gpd.GeoDataFrame.from_features(datapoints).to_csv(maindir+'/Myagdi/Precip/{}/GPM_IMERG/{:03}.csv'.format(year,day))
    for datapoints,day in zip(dps_gpm_mya,range(0,len(precips_myagdi[2])))]

    dps_chelsa_mod_dfs = [gpd.GeoDataFrame.from_features(datapoints).to_csv(maindir+'/Modi/Precip/{}/CHELSA/{:03}.csv'.format(year,day))
    for datapoints,day in zip(dps_chelsa_mod,range(0,len(precips_modi[0])))]
    dps_chirps_mod_dfs = [gpd.GeoDataFrame.from_features(datapoints).to_csv(maindir+'/Modi/Precip/{}/CHIRPS/{:03}.csv'.format(year,day))
    for datapoints,day in zip(dps_chirps_mod,range(0,len(precips_modi[1])))]
    dps_gpm_mod_dfs = [gpd.GeoDataFrame.from_features(datapoints).to_csv(maindir+'/Modi/Precip/{}/GPM_IMERG/{:03}.csv'.format(year,day))
    for datapoints,day in zip(dps_gpm_mod,range(0,len(precips_modi[2])))]

    del dps_chelsa_mya_dfs, dps_chirps_mya_dfs, dps_gpm_mya_dfs, dps_chelsa_mod_dfs, dps_chirps_mod_dfs, dps_gpm_mod_dfs
    del dps_chelsa_mya, dps_chirps_mya, dps_gpm_mya, dps_chelsa_mod, dps_chirps_mod, dps_gpm_mod
    del precips_myagdi, precips_modi
    del p_gpm, p_chirps, p_chelsa
    gc.collect()
