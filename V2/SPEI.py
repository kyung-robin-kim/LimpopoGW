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
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from pathlib import Path
import glob
import rasterio as rio

def read_file(file):
    with rio.open(file) as src:
        return(src.read())


#RESAMPLE TO SAME RESOLUTION
datasets = sorted(glob.glob(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\climate_indices\native_res\*.nc'))
p_gpm = xr.open_mfdataset(datasets[0],parallel=True,chunks={"y": 100,"x":100}).P_mm
pet_modis = xr.open_mfdataset(datasets[-1],parallel=True,chunks={"y": 100,"x":100}).PET_kg_m2

from rasterio.enums import Resampling
p_gpm_resample = p_gpm.rio.write_crs('epsg:4326').rio.reproject_match(pet_modis.rio.write_crs('epsg:4326'),resampling=Resampling.bilinear)
if 'grid_mapping' in p_gpm_resample.attrs:
    # Remove the 'grid_mapping' attribute
    del p_gpm_resample.attrs['grid_mapping']
p_gpm_resample.to_netcdf(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\climate_indices\PRECIP_input.nc')


pet_modis.to_netcdf(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\climate_indices\PET_input.nc')
