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
import csv
import time
import seaborn as sns
from pathlib import Path
from rasterio.merge import merge
import rasterio
from rasterio.plot import show
import glob,os
import pandas as pd
import numpy as np 
import gc

def read_file(file):
    with rio.open(file) as src:
        return(src.read())

##################################
#General Code Structure:
#1. Call all TIF files
#2. Create datetime array based on frequency and length of data
#3. Xarray concatenate into NETCDF file

shpname = r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
shapefile = gpd.read_file(shpname)

'''
#In String format
#dates = pd.date_range('2002-04-30','2021-05-31', freq='MS').strftime("%Y-%b").tolist()
#In Datetime format
dates = pd.date_range('2002-07-30','2021-06-30',  freq='1M') 
'''

#10-09-2022 UPDATE
#################################
#PRECIPITATION
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\PRECIPITATION'
dates = pd.date_range('2002-01','2022-11',  freq='1M') 

#CHIRPS
files = sorted(list(glob.glob(path+'/CHIRPS/*.tif')))
da = [xr.open_rasterio(file)[0].rename(new_name_or_name_dict='P_mm').rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
            for file in files]
dataset = xr.concat(da, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'\CHIRPS_2002_2022.nc')

#GPM_IMERG
dates = pd.date_range('2002-01','2022-11',  freq='1M') 
files = sorted(list(glob.glob(path+'/GPM_IMERG/*.tif')))
da = [xr.open_rasterio(file)[0].rename(new_name_or_name_dict='P_mm').rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
            for file in files]
dataset = xr.concat(da, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'\GPM_IMERG_2002_2022.nc')
del dataset
gc.collect()

#################################
#MODMYD

#NDVI
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_NDVI'
dates = pd.date_range('2002-07','2022-09',  freq='1M') 

files = sorted(glob.glob(path+"/MODMYD/*.tif"))
da = [xr.open_rasterio(file)[0].rename(new_name_or_name_dict='NDVI').rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
            for file in files]
dataset = xr.concat(da, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'\MODMYD_NDVI_2002_2022.nc')
del dataset
gc.collect()


#LST
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_LST'
dates = pd.date_range('2002-07','2022-06',  freq='1M') 

files = sorted(glob.glob(path+"/*.tif"))
da = [xr.open_rasterio(file)[0].rename(new_name_or_name_dict='LST_K').rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
            for file in files]
dataset = xr.concat(da, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'\MODMYD_LST_2002_2022.nc')
del dataset
gc.collect()


#################################
#GLDAS - NOAH

#SOIL MOISTURE
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_SOILMOISTURE'

#SM: 0 - 200cm
files = sorted(glob.glob(path+"/0_200cm/*.tif"))
dates = pd.date_range('2000-01','2022-01',  freq='1M') 
da = [xr.open_rasterio(file)[0].rename(new_name_or_name_dict='SM_kg_m2').rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
            for file in files]
dataset = xr.concat(da, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'\SM_0_200cm_2000_2021.nc')

#SM: 0 - 100cm
files = sorted(glob.glob(path+"/0_100cm/*.tif"))
dates = pd.date_range('2000-01','2022-01',  freq='1M')  
da = [xr.open_rasterio(file)[0].rename(new_name_or_name_dict='SM_kg_m2').rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
            for file in files]
dataset = xr.concat(da, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'\SM_0_100cm_2000_2021.nc')

#SM: 0 - 10cm
files = sorted(glob.glob(path+"/0_10cm/*.tif"))
dates = pd.date_range('2000-01','2022-01',  freq='1M') 
da = [xr.open_rasterio(file)[0].rename(new_name_or_name_dict='SM_kg_m2').rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
            for file in files]
dataset = xr.concat(da, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'\SM_0_10cm_2000_2021.nc')

#SM: Root Zone
files = sorted(glob.glob(path+"/RootZone/*.tif"))
dates = pd.date_range('2000-01','2022-01',  freq='1M') 
da = [xr.open_rasterio(file)[0].rename(new_name_or_name_dict='SM_kg_m2').rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
            for file in files]
dataset = xr.concat(da, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'\SM_RootZone_2000_2021.nc')


#RUNOFF
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_RUNOFF'
files = sorted(glob.glob(path+"/*.tif"))
dates = pd.date_range('2000-01','2022-01',  freq='1M') 
da = [xr.open_rasterio(file)[0].rename(new_name_or_name_dict='R_kg_m2').rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
            for file in files]
dataset = xr.concat(da, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'\RUNOFF_2000_2021.nc')

#################################
#GLDAS - CLSM

path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_CLSM'
dates = pd.date_range('2003-02','2022-08',  freq='1M') 

#GROUNDWATER
files = sorted(glob.glob(path+'/GROUNDWATER/MONTHLY/*.tif'))
da = [xr.open_rasterio(file)[0].rename(new_name_or_name_dict='mm').rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
            for file in files]
dataset = xr.concat(da, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'\CLSM_GW_2003_2022.nc')

#Terrestrial Water Storage
files = sorted(glob.glob(path+'/TWS/MONTHLY/*.tif'))
da = [xr.open_rasterio(file)[0].rename(new_name_or_name_dict='mm').rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
            for file in files]
dataset = xr.concat(da, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'\CLSM_TWS_2003_2022.nc')

#Soil Moisture
files = sorted(glob.glob(path+'/SOIL_MOISTURE/RZ/MONTHLY/*.tif'))
da = [xr.open_rasterio(file)[0].rename(new_name_or_name_dict='kgm-2').rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
            for file in files]
dataset = xr.concat(da, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'\CLSM_SM_RZ_2003_2022.nc')

files = sorted(glob.glob(path+'/SOIL_MOISTURE/Surface/MONTHLY/*.tif'))
da = [xr.open_rasterio(file)[0].rename(new_name_or_name_dict='kgm-2').rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
            for file in files]
dataset = xr.concat(da, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'\CLSM_SM_Surface_2003_2022.nc')

#Runoff
files = sorted(glob.glob(path+'/RUNOFF/Surface/MONTHLY/*.tif'))
da = [xr.open_rasterio(file)[0].rename(new_name_or_name_dict='kgm-2s-1').rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
            for file in files]
dataset = xr.concat(da, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'\CLSM_R_Surface_2003_2022.nc')

files = sorted(glob.glob(path+'/RUNOFF/Base/MONTHLY/*.tif'))
da = [xr.open_rasterio(file)[0].rename(new_name_or_name_dict='kgm-2s-1').rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
            for file in files]
dataset = xr.concat(da, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'\CLSM_R_Baseflow_2003_2022.nc')



#################################
#SMAP

path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\SMAP_SM_1km\south_africa_monthly\south_africa_monthly'
files = sorted(glob.glob(path+"/*.tif"))
dates = pd.date_range('2015-04','2021-01',  freq='1M') 
da = [xr.open_rasterio(file)[0].rename(new_name_or_name_dict='SM_vwc').rio.reproject(shapefile.crs).rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
            for file in files]
dataset = xr.concat(da, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'\SMAP_2015_2020.nc')




#################################
#Sentinel LULC
path = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\Sentinel_LULC'
files = sorted(glob.glob(path+"/*.tif"))
dates = ['2017','2021']
da = [xr.open_rasterio(file)[0].rename(new_name_or_name_dict='lulc') for file in files]
dataset = xr.concat(da, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'\sentinel_LULC.nc')





#OLD METHOD (PRE-2022)


###################################
#CHIRPS PRECIP
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\CHIRPS'
files = sorted(glob.glob(path+"/*.tif"))
dates = pd.date_range('2001-01-31','2021-09-30',  freq='1M') 

arrays = []
for file in files:
    da = xr.open_rasterio(file)[0].rename(new_name_or_name_dict='P')
    da = da.rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
    arrays.append(da)

dataset = xr.concat(arrays, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'\CHIRPS_P_2001_2021_120321.nc')

##################################
#GPM IMERG
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GPM_IMERG'
files = sorted(glob.glob(path+"/*.tif"))

#In String format
#dates = pd.date_range('2002-04-30','2021-05-31', freq='MS').strftime("%Y-%b").tolist()
#In Datetime format
dates = pd.date_range('2002-10-31','2020-12-31',  freq='1M') 

arrays = []
for file in files:
    
    if file[-11:-4] not in ('2020_10', '2020_11', '2020_12'):
        da = xr.open_rasterio(file)[0].rename(new_name_or_name_dict='P')
        arrays.append(da)

    else:
        da = xr.open_rasterio(file)[0].rename(new_name_or_name_dict='P')
        da = da.rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
        arrays.append(da)

dataset = xr.concat(arrays, dim=dates).rename({'concat_dim':'time'})

dataset.to_netcdf(path+'\GPM_P_2002_2020_120321.nc')

###################################
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_ET\MODMYD'
files = sorted(glob.glob(path+"/*.tif"))

da = xr.open_rasterio(files[0])


###################################
#TERRA MODIS ET
#Created second NC dataset for ET < 100000 
#12-04-21

path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_ET\TERRA'
files = sorted(glob.glob(path+"/*.tif"))
dates = pd.date_range('2000-01-31','2020-12-31',  freq='1M') 

arrays = []
for file in files:
    da = xr.open_rasterio(file)[0].rename(new_name_or_name_dict='ET')
    da = da.rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
    da = da.where(da<10000,np.nan) 
    arrays.append(da)

dataset = xr.concat(arrays, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'\TMODIS_ET_2000_2020_120421.nc')

##################################
#NDVI
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_NDVI'
Terras = sorted(glob.glob(path+"/TERRA/*.tif"))[29:] #Start from same time as Aqua (2002-04)
Aquas = sorted(glob.glob(path+"/AQUA/*.tif"))[:-1]
dates = pd.date_range('2002-07-30','2021-06-30',  freq='1M') 

arrays = []
for fileT,fileA,day in zip(Terras,Aquas,dates):
    daMODIS = []
    daT = xr.open_rasterio(fileT)[0].rename(new_name_or_name_dict='NDVI').expand_dims(dim='time')
    daMODIS.append(daT)
    daA = xr.open_rasterio(fileA)[0].rename(new_name_or_name_dict='NDVI').expand_dims(dim='time')
    daMODIS.append(daA)

    dataset = xr.concat(daMODIS, dim='time')
    dataset = dataset.mean(dim='time')
    arrays.append(dataset)
    print(day)

dataset = xr.concat(arrays, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'MODIS_NDVI_2002_2021_120321.nc')

###################################
#GLDAS SM
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_SOILMOISTURE'
files = sorted(glob.glob(path+"/*.tif"))
dates = pd.date_range('2000-01-31','2021-04-30',  freq='1M') 

arrays = []
for file in files:
    da = xr.open_rasterio(file)[0].rename(new_name_or_name_dict='SM')
    da = da.rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
    arrays.append(da)

dataset = xr.concat(arrays, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'\GLDAS_SM_2000_2021_120321.nc')

#GLDAS RUNOFF
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_RUNOFF'
files = sorted(glob.glob(path+"/*.tif"))
dates = pd.date_range('2000-01-31','2021-04-30',  freq='1M') 

arrays = []
for file in files:
    da = xr.open_rasterio(file)[0].rename(new_name_or_name_dict='R')
    da = da.rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
    arrays.append(da)

dataset = xr.concat(arrays, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'\GLDAS_R_2000_2021_120321.nc')


###################################
#AIRS
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\AIRS'
temperatures = ['\AIR_SURFACE','\SKIN', '\TROP']
locations = ['\ENSO','\IOD','\LIMPOPO']
dates = pd.date_range('2002-09-30','2020-12-31',  freq='1M') 

for temp in temperatures:
    for loc in locations:
        files = sorted(glob.glob(path+temp+loc+"/*.tif"))
        
        if temp == '\AIR_SURFACE':
            arrays = []
            for file in files:
                da = xr.open_rasterio(file)[0].rename(new_name_or_name_dict='AIR') - 273.15
                arrays.append(da)
        
            dataset = xr.concat(arrays, dim=dates).rename({'concat_dim':'time'})
            dataset.to_netcdf(path+temp+loc+r'\{}_AIRTEMP_2002_2020_120321.nc'.format(loc[1::]))
            
        if temp == '\SKIN':
            arrays = []    
            for file in files:
                da = xr.open_rasterio(file)[0].rename(new_name_or_name_dict='SKIN') - 273.15
                arrays.append(da)
        
            dataset = xr.concat(arrays, dim=dates).rename({'concat_dim':'time'})
            dataset.to_netcdf(path+temp+loc+r'\{}_SKINTEMP_2002_2020_120321.nc'.format(loc[1::]))


        if temp == '\TROP':
            arrays = []    
            for file in files:
                da = xr.open_rasterio(file)[0].rename(new_name_or_name_dict='TROP') - 273.15
                arrays.append(da)
        
            dataset = xr.concat(arrays, dim=dates).rename({'concat_dim':'time'})
            dataset.to_netcdf(path+temp+loc+r'\{}_TROPTEMP_2002_2020_120321.nc'.format(loc[1::]))
