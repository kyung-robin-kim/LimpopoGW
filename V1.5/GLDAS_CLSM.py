import pandas as pd
import xarray as xr
import rioxarray
import geopandas as gpd
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

#####################
#10-16-22

path = r'D:\raw-data\GLDAS\CLSM025_L4'
files = sorted(glob.glob(path+"/*.nc4"))

shpname = r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
shapefile = gpd.read_file(shpname)

clsm = xr.open_mfdataset(files[0:3],parallel=True,chunks={"lat": 200,"lon":400})

surface_r = clsm.Qs_tavg[:,:,:].rio.write_crs('epsg:4329').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)
baseflow_r = clsm.Qsb_tavg[:,:,:].rio.write_crs('epsg:4329').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)
sm_surface = clsm.SoilMoist_S_tavg[:,:,:].rio.write_crs('epsg:4329').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)
sm_rz = clsm.SoilMoist_RZ_tavg[:,:,:].rio.write_crs('epsg:4329').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)
sm_profile = clsm.SoilMoist_P_tavg[:,:,:].rio.write_crs('epsg:4329').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)
e_canopy = clsm.ECanop_tavg[:,:,:].rio.write_crs('epsg:4329').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)
t_veg = clsm.TVeg_tavg[:,:,:].rio.write_crs('epsg:4329').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)
interc_canopy = clsm.CanopInt_tavg[:,:,:].rio.write_crs('epsg:4329').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)
e_soil = clsm.ESoil_tavg[:,:,:].rio.write_crs('epsg:4329').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)
tws = clsm.TWS_tavg[:,:,:].rio.write_crs('epsg:4329').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)
gws = clsm.GWS_tavg[:,:,:].rio.write_crs('epsg:4329').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)

#https://stackoverflow.com/questions/54776283/how-to-call-the-xarrays-groupby-function-to-group-data-by-a-combination-of-year
year_month_idx = pd.MultiIndex.from_arrays([np.array(clsm['time.year']), np.array(clsm['time.month'])])

savepath = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_CLSM'
filepaths = ['GROUNDWATER','RUNOFF','SOIL_MOISTURE','TWS']


gws.coords['year_month'] = ('time', year_month_idx)
gws_monthly = gws.groupby('year_month').mean()
[gws_monthly[i].rio.to_raster(savepath+'//{}/MONTHLY/{}_{:02d}.tif'.format(filepaths[0],np.array(year),np.array(month))) for i,year,month in zip(range(0,len(gws_monthly)),gws_monthly.year_month_level_0,gws_monthly.year_month_level_1)]

tws.coords['year_month'] = ('time', year_month_idx)
tws_monthly = tws.groupby('year_month').mean()
[tws_monthly[i].rio.to_raster(savepath+'//{}/MONTHLY/{}_{:02d}.tif'.format(filepaths[3],np.array(year),np.array(month))) for i,year,month in zip(range(0,len(tws_monthly)),tws_monthly.year_month_level_0,tws_monthly.year_month_level_1)]


sm_surface.coords['year_month'] = ('time', year_month_idx)
sm_surface_monthly = sm_surface.groupby('year_month').mean()
[sm_surface_monthly[i].rio.to_raster(savepath+'//{}/Surface/MONTHLY/{}_{:02d}.tif'.format(filepaths[2],np.array(year),np.array(month))) for i,year,month in zip(range(0,len(sm_surface_monthly)),sm_surface_monthly.year_month_level_0,sm_surface_monthly.year_month_level_1)]

sm_rz.coords['year_month'] = ('time', year_month_idx)
sm_rz_monthly = sm_rz.groupby('year_month').mean()
[sm_rz_monthly[i].rio.to_raster(savepath+'//{}/RZ/MONTHLY/{}_{:02d}.tif'.format(filepaths[2],np.array(year),np.array(month))) for i,year,month in zip(range(0,len(sm_rz_monthly)),sm_rz_monthly.year_month_level_0,sm_rz_monthly.year_month_level_1)]