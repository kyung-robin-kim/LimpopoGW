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

def plot_np(array,vmin,vmax,title):
    array = np.where(array==0,np.nan,array)
    fig1, ax1 = plt.subplots(figsize=(20,16))
    image = ax1.imshow(array,cmap='RdBu',vmin=vmin,vmax=vmax)
    cbar = fig1.colorbar(image,ax=ax1)
    ax1.set_title('{}'.format(title))

path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_RUNOFF\RUNOFF_2000_2021.nc'
GLDAS_monthly_runoff = xr.open_mfdataset(path,parallel=True,chunks={"y": 10,"x":10}).R_kg_m2


###############################################
#GRDC DATA

#Daily Discharge
GRDC_path_daily = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\validation_points\GRDC\2022-10-15_21-18\*Day*.txt'
GRDC_daily_discharge = sorted(glob.glob(GRDC_path_daily))

d_data = [pd.read_csv(i,encoding='latin1', sep=';', header = 36) for i in GRDC_daily_discharge]
for i in range(0,len(d_data)):
    d_data[i].index = pd.to_datetime(d_data[i]['YYYY-MM-DD'])

#Accumulated monthly runoff (m3s-1)
d_data_datetime = [d_data[i].groupby(pd.Grouper(freq='M')).sum() for i in range(0,len(d_data))]

#Monthly Discharge
GRDC_path_monthly = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\validation_points\GRDC\2022-10-15_21-18\*Month*.txt'
GRDC_monthly_discharge = sorted(glob.glob(GRDC_path_monthly))

m_data = [pd.read_csv(i,encoding='latin1', sep=';', header = 38) for i in GRDC_monthly_discharge]
for i in range(0,len(m_data)):
    m_data[i].index = d_data_datetime[i].index

data_flag = [m_data.iloc[:,4] for m_data in m_data] #Percentage of daily data used per month

d_data_flagged = [monthly_accum[flag==100] for monthly_accum,flag in zip(d_data_datetime,data_flag)]

#Metadata
meta_data = [pd.read_csv(i,encoding='latin1', header = None, skiprows = 8, nrows = 9) for i in GRDC_daily_discharge]

gauge = [meta.loc[0][0][12::].strip() for meta in meta_data] #GRDC ID
river = [meta.loc[1][0][8::].strip() for meta in meta_data] #river name
lat =  [float(meta.loc[4][0][16::].strip()) for meta in meta_data] #DD
lon =  [float(meta.loc[5][0][17::].strip()) for meta in meta_data] #DD
area =  [float(meta.loc[6][0][23::].strip()) for meta in meta_data] #km2
elev =  [float(meta.loc[7][0][19::].strip()) for meta in meta_data] #meters ASL

#meta_data_df = pd.DataFrame([gauge,river,lat,lon,area,elev]).T.rename({0:'StationID',1:'River',2:'Lat',3:'Lon',4:'Area_km2',5:'Elev_m'},axis=1)
#meta_data_df.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\validation_points\GRDC\discharge_gauges.csv')

'''
for data, id in zip(d_data_flagged,gauge):
    plt.figure()
    plt.scatter(data.index, data,s=1)
    plt.title('{}'.format(id))
    #data.plot.scatter(x='YYYY-MM-DD', y='monthly discharge accum (m3s-1)', title='{}'.format(id))
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\validation_points\GRDC\2022-10-15_21-18\plots\{}'.format(id))
'''

#Filter out for valid gauge sites (newer than 2001-01)
last_date_data = [data.index[-1] for data in d_data_flagged]
valid_sites = [last_date > pd.to_datetime('2001-01') for last_date in last_date_data]
valid_sites = np.where(valid_sites)

d_data_filtered = [d_data_flagged[i] for i in valid_sites[0]]
#meta_data_df_filtered = meta_data_df.iloc[valid_sites[0]]
#meta_data_df_filtered.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\validation_points\GRDC\discharge_gauges_valid.csv')

path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_RUNOFF'
files = sorted(glob.glob(path+"/*.nc"))
r = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 10,"lon":10}).rio.write_crs('WGS84').R_kg_m2
discharge_points = gpd.read_file(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\validation_points\discharge_sites.shp')

monthly = [rs.point_query(discharge_points, np.array(r[i]), affine= r[i].rio.transform(), geojson_out=True, interpolate='nearest') for i in range(0,len(r))]
time_series_by_month = np.array([ [monthly[month][site]['properties']['value'] for site in range(0,len(discharge_points))] for month in range(0,len(monthly))])
time_series_by_site = time_series_by_month.T

#2001 and after
for i,site in zip(range(0,len(d_data_filtered)),gauge):
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax2.plot(r.time[12::],time_series_by_site[i][12::],label='GLDAS')
    ax2.set_ylabel('GLDAS ()')
    ax1.plot(d_data_filtered[i].index,d_data_filtered[i],color='C1',label='in-situ')
    ax1.set_ylabel('GRDC (m3s1 monthly accum)')
    ax1.set_xlabel('Date')
    ax1.set_title('{}'.format(site))
    ax1.legend()
    ax2.legend()
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\grdc_gldas_runoff\{}'.format(site))