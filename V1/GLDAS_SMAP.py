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

path = r'C:\Users\robin\Box\Data\Modeled\GLDAS_NOAH_V2_1'
files = sorted(glob.glob(path+"/*.nc4"))[183:252]

shpname = r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
shapefile = gpd.read_file(shpname)

savepath = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_SOILMOISTURE\SMAP_Comparison'

for file in files:

    sm_0_10 = xr.open_mfdataset(file,parallel=True,chunks={"lat": 10,"lon":10}).SoilMoi0_10cm_inst

    sm_limpopo = sm_0_10.rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)

    array_tif = xr.DataArray(sm_limpopo[0], dims=("lat", "lon"), coords={"lat": sm_limpopo.lat, "lon": sm_limpopo.lon}, name="Monthly_Runoff")
    array_tif.rio.set_crs("epsg:4326")
    array_tif.rio.set_spatial_dims('lon','lat',inplace=True)
    os.chdir(savepath)
    array_tif.rio.to_raster('GLDAS_SM_10cm_{}_{}.tif'.format(file[-14:-10],file[-10:-8]))