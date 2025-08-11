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
#from rasterstats import zonal_stats
import csv
import earthpy.mask as em
import os
from pathlib import Path
from glob import glob
import seaborn as sns

# TIFS for analysis
path_myd=r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\LST_MODIS\MYD'
path_mod=r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\LST_MODIS\MOD'

#2002-07 through 2021-12
years=range(2022,2023)

for year in years:

    os.chdir(path_myd)
    myd = xr.open_mfdataset('{}.nc'.format(year),parallel=True,chunks={"lat": 100,"lon":200})
    n_myd = myd.LST_Night_1km
    d_myd = myd.LST_Day_1km

    os.chdir(path_mod)
    mod = xr.open_mfdataset('{}.nc'.format(year),parallel=True,chunks={"lat": 100,"lon":200})
    n_mod = mod.LST_Night_1km
    d_mod = mod.LST_Day_1km

    lon = np.array(n_myd.lon)
    lat = np.array(n_myd.lat)

    months = np.arange(1,13)
    for month in months:
        print('{}-'f"{month:02d}".format(year))

        month_n_lst_myd = n_myd.sel(time='{}-'f"{month:02d}".format(year)).mean(dim='time')
        month_d_lst_myd = d_myd.sel(time='{}-'f"{month:02d}".format(year)).mean(dim='time')
        month_n_lst_mod = n_mod.sel(time='{}-'f"{month:02d}".format(year)).mean(dim='time')
        month_d_lst_mod = d_mod.sel(time='{}-'f"{month:02d}".format(year)).mean(dim='time')
        month_modmyd = np.nanmean([month_n_lst_myd,month_d_lst_myd,month_n_lst_mod,month_d_lst_mod],axis=0)
        del month_n_lst_myd,month_d_lst_myd,month_n_lst_mod,month_d_lst_mod

        savepath = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_LST'
        array_tif = xr.DataArray(month_modmyd, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name="LST_K")
        array_tif.rio.set_crs("epsg:4326")
        array_tif.rio.set_spatial_dims('lon','lat',inplace=True)
        os.chdir(savepath)
        array_tif.rio.to_raster('MODMYD_LST_{}_'f"{month:02d}.tif".format(year))
        del month_modmyd