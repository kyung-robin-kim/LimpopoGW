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

#####################
#12-03-21
path = r'C:\Users\robin\Box\Data\StudyRegion\Limpopo\CHIRPS_V2\200001_202112_daily'
files = sorted(glob.glob(path+"/*.tif"))
shpname = r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
shapefile = gpd.read_file(shpname)

date_strings = []
for file in files:
    date_strings.append(str(file[-12:-6]))
unique_months = np.unique(date_strings)


for date in unique_months:
    monthly_sum = []
    for file in files:
        if file[-12:-6] == date:
            da = xr.open_mfdataset(file)
            da = da.salem.roi(shape=shapefile,all_touched=False).band_data[0].rename('P').expand_dims(dim='time')

            monthly_sum.append(da)

    dataset = xr.concat(monthly_sum,dim='time')
    p_accumulation = dataset.sum(dim='time')
    print(date)
    p_accumulation.rio.to_raster(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\CHIRPS\CHIRPS_{}.tif'.format(date))

    del dataset
    del da
    del p_accumulation
    del monthly_sum
