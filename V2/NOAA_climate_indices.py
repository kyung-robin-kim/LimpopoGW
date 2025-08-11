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
from rasterio.enums import Resampling

def read_file(file):
    with rio.open(file) as src:
        return(src.read())

#example SPEI outputs
path = r'C:\Users\robin\Box\SouthernAfrica\example_climate_indices\example\input'
files = sorted(glob.glob(path+'\*.nc'))
input_ppt = xr.open_mfdataset(files[3],parallel=True,chunks={"lat": 100,"lon":100}).prcp
input_pet = xr.open_mfdataset(files[2],parallel=True,chunks={"lat": 100,"lon":100}).pet
#example outputs
path = r'C:\Users\robin\Box\SouthernAfrica\example_climate_indices\example\output_spei_ex'
files = sorted(glob.glob(path+'\*spei*.nc'))
outputs = xr.open_mfdataset(files,parallel=True,chunks={"lat": 100,"lon":100})




#Resample to same resolutions (ex. CHIRPS to MODIS PET):
files = sorted(glob.glob(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\climate_indices\native_res\*.nc'))
pet_ref = xr.open_mfdataset(files[-1],parallel=True,chunks={"lat": 100,"lon":100})
ppt_resamp = xr.open_mfdataset(files[1],parallel=True,chunks={"lat": 100,"lon":100}).rio.write_crs('epsg:4326').rio.reproject_match(pet_ref.rio.write_crs('epsg:4326'), resampling=rio.enums.Resampling.average)
ppt_resamp = ppt_resamp.rename({'y':'lat','x':'lon'}).transpose('lat','lon','time')
del ppt_resamp.P_mm.attrs['grid_mapping']
ppt_resamp.P_mm.attrs['units'] = 'mm'
ppt_resamp.to_netcdf(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\climate_indices\PPT_GPM_preinput.nc')


pet = pet_ref.rename({'y':'lat','x':'lon'}).transpose('lat','lon','time')
del pet.PET_kg_m2.attrs['grid_mapping']
pet.PET_kg_m2.attrs['units'] = 'mm'
pet.to_netcdf(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\climate_indices\PET_MODIS_preinput.nc')


#Follow instructions on: 
# https://climate-indices.readthedocs.io/en/latest/
# change conda env to env where package is installed & run code as copied for index

#C:/Users/robin/Desktop/Limpopo_VegetationHydrology/Data/climate_indices
#C:/Users/robin/Desktop/Limpopo_VegetationHydrology/Data/climate_indices/SPEI/CHIRPS

#Single line of code
process_climate_indices --index spei --periodicity monthly --netcdf_precip C:/Users/robin/Desktop/Limpopo_VegetationHydrology/Data/climate_indices/PPT_GPM_preinput.nc --var_name_precip P_mm --netcdf_pet C:/Users/robin/Desktop/Limpopo_VegetationHydrology/Data/climate_indices/PET_MODIS_preinput.nc --var_name_pet PET_kg_m2 --output_file_base C:/Users/robin/Desktop/Limpopo_VegetationHydrology/Data/climate_indices/limpopo --scales 12 --calibration_start_year 2002 --calibration_end_year 2022 --multiprocessing all