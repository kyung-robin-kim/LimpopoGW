import pandas as pd
import xarray as xr
import rioxarray
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
import skimage
import rasterstats as rs
from rasterio.enums import Resampling
from datetime import datetime

'''
#Sophia's InSitu Borehole Data
#Borehole water levels - 227 sites
filepath = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\sophia\Limpopo In-Situ Data\Boreholes\LP\LP_WaterLevels\*.csv'
csv_files = sorted(list(glob.glob(filepath)))
waterlevel_dfs = [pd.read_csv(file, names=['site', 'date','time','level','?','??'],index_col=False) for file in csv_files]
waterlevel_dates_df =  [np.stack([datetime.strptime(str(date), '%Y%m%d') for date in waterlevel_dfs[i].date]) for i in range(0,len(waterlevel_dfs))]
waterlevel_dfs = [pd.concat([waterlevel_dfs[i]['site'], pd.Series(waterlevel_dates_df[i]).rename('date'), waterlevel_dfs[i]['time'],waterlevel_dfs[i]['level']],axis=1) for i in range(0,len(waterlevel_dfs))] 
waterlevel_sites = [(waterlevel_dfs[i]['site'][0]).strip() for i in range(0,len(waterlevel_dfs)) ]

#Resampled to day-average 
waterlevel_dfs_daily = [waterlevel_dfs[i].groupby('date').mean() for i in range(0,len(waterlevel_dfs))]
waterlevel_dfs_daily = [waterlevel_dfs_daily[i][(waterlevel_dfs_daily[i]>-100) & (waterlevel_dfs_daily[i]!=0)] for i in range(0,len(waterlevel_dfs_daily))]

#Borehole shapefile - 655 sites (therefore filter out for those with water level data)
shpname = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\sophia\Limpopo In-Situ Data\Boreholes\LP\LP_BOREHOLES.shp'
boreholes = gpd.read_file(shpname).to_crs({'init': 'epsg:4326'})

boreholes_boolean = [boreholes.loc[i].F_STATION in waterlevel_sites for i in range(0,len(boreholes))]
boreholes = boreholes[boreholes_boolean]
'''

filepath = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\validation_points\LULC\boreholes\*.shp'
borehole_files = sorted(list(glob.glob(filepath)))

boreholes = gpd.read_file(borehole_files[0])
