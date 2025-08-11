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
#10-04-22

path = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\NDVI_MODIS\Update'
files = sorted(glob.glob(path+"/*.nc"))

shpname = r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
shapefile = gpd.read_file(shpname)

mod = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 500,"lon":500})
myd = xr.open_mfdataset(files[1],parallel=True,chunks={"lat": 500,"lon":500})

#4 minutes to run
ndvi_mod = mod.rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)
ndvi_myd = myd.rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)

lon = np.array(ndvi_mod.lon)
lat = np.array(ndvi_mod.lat)


years=range(2022,2023)

for year in years:

    mod = ndvi_mod.sel(time='{}'.format(year))._1_km_monthly_NDVI
    myd = ndvi_myd.sel(time='{}'.format(year))._1_km_monthly_NDVI

    months = np.arange(1,13)
    for month in months:
        print('{}-'f"{month:02d}".format(year))

        month_mod = mod.sel(time='{}-'f"{month:02d}".format(year)).mean(dim='time')
        month_myd = myd.sel(time='{}-'f"{month:02d}".format(year)).mean(dim='time')
        month_modmyd = np.nanmean([month_mod,month_myd],axis=0)
        del month_mod, month_myd

        savepath = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_NDVI\MODMYD'
        array_tif = xr.DataArray(month_modmyd, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name="NDVI")
        array_tif.rio.set_crs("epsg:4326")
        array_tif.rio.set_spatial_dims('lon','lat',inplace=True)
        os.chdir(savepath)
        array_tif.rio.to_raster('MODMYD_NDVI_{}_'f"{month:02d}.tif".format(year))
        del month_modmyd

