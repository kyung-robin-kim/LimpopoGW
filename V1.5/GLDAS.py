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
#08-12-21
#10-06-22 -- edit code for final analyses -- index out [24:-6] for 2002-01 through 2021-12

path = r'C:\Users\robin\Box\Data\Modeled\GLDAS_NOAH_V2_1'
files = sorted(glob.glob(path+"/*.nc4"))

shpname = r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
shapefile = gpd.read_file(shpname)

sample_text = 'GLDAS_NOAH025_M.A200001.021.nc4'
# year --> sample_text[-14:-10]
# month --> sample_text[-10:-8]

days_count = {'month': ['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], 
            'month_no': ['01', '02', '03','04','05','06','07','08','09','10','11','12'],
            'non_leap': [31,28,31,30,31,30,31,31,30,31,30,31], 
            'leap': [31,29,31,30,31,30,31,31,30,31,30,31]}
days_df = pd.DataFrame(data=days_count)

# days_df.loc[days_df['month_no']==files[0][-10:-8]]['non_leap']

###################################
#RUNOFF (mm)
###################################

#3-hour monthly accumulation; therefore --> example. Qs_acc (April){kg/m2} = Qs_acc (April){kg/m2/3hr} * 8{3hr/day} * 30{days}
#Add runoff sources to get total modeled runoff

savepath = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_RUNOFF'

for file in files[-15:-6]: #index to continue 2021-04 through 2021-12
    if int(files[0][-14:-10])%4 == 0:
        storm_runoff = xr.open_mfdataset(file,parallel=True,chunks={"lat": 10,"lon":10}).Qs_acc * 8 * int(days_df.loc[days_df['month_no']==file[-10:-8]]['leap'])
        subsurface_runoff = xr.open_mfdataset(file,parallel=True,chunks={"lat": 10,"lon":10}).Qsb_acc * 8 * int(days_df.loc[days_df['month_no']==file[-10:-8]]['leap'])
    else:
        storm_runoff = xr.open_mfdataset(file,parallel=True,chunks={"lat": 10,"lon":10}).Qs_acc * 8 * int(days_df.loc[days_df['month_no']==file[-10:-8]]['non_leap'])
        subsurface_runoff = xr.open_mfdataset(file,parallel=True,chunks={"lat": 10,"lon":10}).Qsb_acc * 8 * int(days_df.loc[days_df['month_no']==file[-10:-8]]['non_leap'])
            
    runoff = storm_runoff + subsurface_runoff
    runoff_limpopo = runoff.rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)
    array_tif = xr.DataArray(runoff_limpopo[0], dims=("lat", "lon"), coords={"lat": runoff_limpopo.lat, "lon": runoff_limpopo.lon}, name="Monthly_Runoff")
    array_tif.rio.set_crs("epsg:4326")
    array_tif.rio.set_spatial_dims('lon','lat',inplace=True)
    os.chdir(savepath)
    array_tif.rio.to_raster('GLDAS_Runoff_{}_{}.tif'.format(file[-14:-10],file[-10:-8]))


###################################
#SOIL MOISTURE - instantaneous (mm) 
###################################

savepath = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_SOILMOISTURE\0_200cm'
#0-200cm
for file in files[-15:-6]:
    sm_0_10 = xr.open_mfdataset(file,parallel=True,chunks={"lat": 100,"lon":100}).SoilMoi0_10cm_inst
    sm_10_40  = xr.open_mfdataset(file,parallel=True,chunks={"lat": 100,"lon":100}).SoilMoi10_40cm_inst
    sm_40_100  = xr.open_mfdataset(file,parallel=True,chunks={"lat": 100,"lon":100}).SoilMoi40_100cm_inst
    sm_100_200  = xr.open_mfdataset(file,parallel=True,chunks={"lat": 100,"lon":100}).SoilMoi100_200cm_inst
    sm_total = sm_0_10 + sm_10_40 + sm_40_100 + sm_100_200

    sm_limpopo = sm_total.rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)

    array_tif = xr.DataArray(sm_limpopo[0], dims=("lat", "lon"), coords={"lat": sm_limpopo.lat, "lon": sm_limpopo.lon}, name="Monthly_SoilMoisture")
    array_tif.rio.set_crs("epsg:4326")
    array_tif.rio.set_spatial_dims('lon','lat',inplace=True)
    os.chdir(savepath)
    array_tif.rio.to_raster('GLDAS_SM_{}_{}.tif'.format(file[-14:-10],file[-10:-8]))


#0-10cm
savepath = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_SOILMOISTURE\0_10cm'
for file in files[:-6]:
    sm_0_10 = xr.open_mfdataset(file,parallel=True,chunks={"lat": 100,"lon":100}).SoilMoi0_10cm_inst

    sm_limpopo = sm_0_10.rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)

    array_tif = xr.DataArray(sm_limpopo[0], dims=("lat", "lon"), coords={"lat": sm_limpopo.lat, "lon": sm_limpopo.lon}, name="Monthly_SoilMoisture")
    array_tif.rio.set_crs("epsg:4326")
    array_tif.rio.set_spatial_dims('lon','lat',inplace=True)
    os.chdir(savepath)
    array_tif.rio.to_raster('GLDAS_SM_{}_{}.tif'.format(file[-14:-10],file[-10:-8]))


#0-100cm
savepath = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_SOILMOISTURE\0_100cm'
for file in files[:-6]:
    sm_0_10 = xr.open_mfdataset(file,parallel=True,chunks={"lat": 100,"lon":100}).SoilMoi0_10cm_inst
    sm_10_40  = xr.open_mfdataset(file,parallel=True,chunks={"lat": 100,"lon":100}).SoilMoi10_40cm_inst
    sm_40_100  = xr.open_mfdataset(file,parallel=True,chunks={"lat": 100,"lon":100}).SoilMoi40_100cm_inst

    sm_total = sm_0_10 + sm_10_40 + sm_40_100 

    sm_limpopo = sm_total.rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)

    array_tif = xr.DataArray(sm_limpopo[0], dims=("lat", "lon"), coords={"lat": sm_limpopo.lat, "lon": sm_limpopo.lon}, name="Monthly_SoilMoisture")
    array_tif.rio.set_crs("epsg:4326")
    array_tif.rio.set_spatial_dims('lon','lat',inplace=True)
    os.chdir(savepath)
    array_tif.rio.to_raster('GLDAS_SM_{}_{}.tif'.format(file[-14:-10],file[-10:-8]))


#Root Zone Soil Moisture
savepath = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_SOILMOISTURE\RootZone'
for file in files[:-6]:
    sm = xr.open_mfdataset(file,parallel=True,chunks={"lat": 100,"lon":100}).RootMoist_inst
    sm_limpopo = sm.rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)

    array_tif = xr.DataArray(sm_limpopo[0], dims=("lat", "lon"), coords={"lat": sm_limpopo.lat, "lon": sm_limpopo.lon}, name="Monthly_SoilMoisture")
    array_tif.rio.set_crs("epsg:4326")
    array_tif.rio.set_spatial_dims('lon','lat',inplace=True)
    os.chdir(savepath)
    array_tif.rio.to_raster('GLDAS_SM_{}_{}.tif'.format(file[-14:-10],file[-10:-8]))
