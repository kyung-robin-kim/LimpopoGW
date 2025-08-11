import h5py
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
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from pathlib import Path
import glob
import rasterio as rio
import csv
import salem
import skimage
import statsmodels as sm 
import statsmodels.graphics.tsaplots as tsaplots
import statsmodels.api as smapi

def read_file(file):
    with rio.open(file) as src:
        return(src.read())

def anomaly(dataset):
    ds_anomaly = dataset - dataset.mean(dim='time')
    d_anomaly_df = ds_anomaly.mean(['x','y']).to_dataframe()

    return ds_anomaly, d_anomaly_df

def deseason(df):
    decomp = smapi.tsa.seasonal_decompose(df)
    decomp_df = pd.DataFrame(decomp.seasonal)
    decomp_df.columns=[df.name]
    deseas_df = df - decomp_df.squeeze()

    return decomp, decomp_df, deseas_df

def month_anomaly(dataset):
    month_idxs=dataset.groupby('time.month').groups

    dataset_month_anomalies = []
    months=range(1,13)
    for month in months:
        idxs = month_idxs[month]
        ds_month_avg = dataset.isel(time=idxs).mean(dim='time')
        dataset_month_anomaly = dataset.isel(time=idxs) - ds_month_avg
        dataset_month_anomalies.append(dataset_month_anomaly)
        print(str(month))

        del dataset_month_anomaly

    monthly_anomalies_ds = xr.merge(dataset_month_anomalies)

    del dataset_month_anomalies

    return(monthly_anomalies_ds)


def month_anomaly_means_NDVI(dataset):

    month_idxs=dataset.groupby('time.month').groups

    dataset_month_anomalies = []
    months=range(1,13)
    for month in months:
        idxs = month_idxs[month]
        ds_month_avg = dataset.isel(time=idxs).mean(dim='time')
        dataset_month_anomaly = dataset.isel(time=idxs) - ds_month_avg
        dataset_month_anomalies.append(dataset_month_anomaly.NDVI.mean(dim=['x','y']))
        print(str(month))

        del dataset_month_anomaly

    monthly_anomalies_ds = xr.merge(dataset_month_anomalies)

    del dataset_month_anomalies

    return(monthly_anomalies_ds)

########################################
#Precipitation
#GPM
p = xr.open_mfdataset(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GPM_IMERG/*.nc')
p_df = p.P.mean(dim=['x','y']).to_dataframe()
p_df = p_df.P

p_anomaly = month_anomaly(p)
p_anomaly_df = p_anomaly.P.mean(dim=['x','y']).to_dataframe()
p_anomaly_df_index = p_anomaly_df['P']/p_df.std()

#CHIRPS (USE CHIRPS NEXT TIME)
pc = xr.open_mfdataset(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\CHIRPS/*.nc')
pc_df = pc.P.mean(dim=['x','y']).to_dataframe()
pc_df = pc_df[20:240].P

p_anomaly, p_anomaly_df = anomaly(p)
p_decomp, p_decomp_df, p_deseas_df = deseason(p_df)
########################################
#Evapotranspiration
#et = xr.open_mfdataset(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_ET/*.nc',parallel=True,chunks={"x": 100,"y":100})
#et_df = et.ET.mean(dim=['lat','lon']).to_dataframe()

#Terra Only (Aqua is missing 6 weeks in 2016 February/March)
et = xr.open_mfdataset(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_ET\TERRA\TMODIS_ET_2000_2020_120421.nc',parallel=True,chunks={"x": 100,"y": 100})
et_df = et.ET.mean(dim=['x','y']).to_dataframe()
et_df = et_df[33::].ET

et_anomaly = month_anomaly(et)
et_anomaly.to_netcdf(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_ET\TERRA\TMODIS_ET_2000_2020_120421_anomalies.nc')
#et_anomaly = xr.open_mfdataset(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_ET\TERRA\TMODIS_ET_2000_2020_120421_anomalies.nc')
et_anomaly_df = et_anomaly.mean(dim=['x','y']).to_dataframe()
et_anomaly_df_index = et_anomaly_df['ET']/et_df.std()

et_decomp, et_decomp_df, et_deseas_df = deseason(et_df)
########################################
#Runoff
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_RUNOFF'
files = sorted(glob.glob(path+"/*.nc"))
r = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 10,"lon":10})

r_df = r.mean(['x','y'])
r_df = r_df.rename({'SM':'R'}).to_dataframe()
r_df = r_df[33:252].R

r_anomaly = month_anomaly(r)
r_anomaly_df = r_anomaly.SM.mean(dim=['x','y']).to_dataframe()
r_anomaly_df_index = r_anomaly_df['SM']/r_df.std()

r_decomp, r_decomp_df, r_deseas_df = deseason(r_df)
########################################
#Soil Moisture
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_SOILMOISTURE'
files = sorted(glob.glob(path+"/*.nc"))
sm = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 10,"lon":10})

sm_df = sm.mean(['x','y']).to_dataframe()
sm_df = sm_df.SM[33:252]

sm_anomaly = month_anomaly(sm)
sm_anomaly_df = sm_anomaly.SM.mean(dim=['x','y']).to_dataframe()
sm_anomaly_df_index = sm_anomaly_df['SM']/sm_df.std()

sm_decomp, sm__decomp_df, sm__deseas_df = deseason(sm_df)
########################################
#NDVI
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_NDVI'
files = sorted(glob.glob(path+"/*.nc"))
ndvi = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 100,"lon":100})

ndvi_df = ndvi.mean(['x','y']).to_dataframe()
ndvi_df = ndvi_df.NDVI[3:222]

ndvi_anomaly = month_anomaly_means_NDVI(ndvi)
ndvi_anomaly.to_netcdf(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_NDVI\NDVI_ANOMALIES.nc')
ndvi_anomaly = xr.open_mfdataset(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_NDVI\NDVI_ANOMALIES.nc')
ndvi_anomaly_df = ndvi_anomaly.to_dataframe()
ndvi_anomaly_df_index = ndvi_anomaly_df['NDVI']/ndvi_df.std()

ndvi_decomp, ndvi__decomp_df, ndvi_deseas_df = deseason(ndvi_df)
########################################
#AIRS TEMPERATURES

#ENSO
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\AIRS'
file = sorted(glob.glob(path+"\SKIN\ENSO\*.nc"))
enso_skin = xr.open_mfdataset(file,parallel=True,chunks={"lat": 1,"lon":1})
enso_skin_df = enso_skin.mean(['x','y']).to_dataframe()
enso_skin_df = enso_skin_df.SKIN[1::]

enso_skin_anomaly = month_anomaly(file)
enso_skin_anomaly_df = enso_skin_anomaly.SKIN.mean(dim=['x','y']).to_dataframe()
enso_skin_anomaly_df_index = enso_skin_anomaly_df['SKIN']/enso_skin_df.std()

file = sorted(glob.glob(path+"\AIR_SURFACE\ENSO\*.nc"))
enso_air = xr.open_mfdataset(file,parallel=True,chunks={"lat": 1,"lon":1})
enso_air_df = enso_air.mean(['x','y']).to_dataframe()
enso_air_df = enso_air_df.AIR[1::]

file = sorted(glob.glob(path+"\TROP\ENSO\*.nc"))
enso_trop = xr.open_mfdataset(file,parallel=True,chunks={"lat": 1,"lon":1})
enso_trop_df = enso_trop.mean(['x','y']).to_dataframe()
enso_trop_df = enso_trop_df.TROP[1::]

enso_decomp, enso_decomp_df, enso_deseas_df = deseason(enso_skin_df)


#IOD
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\AIRS'
file = sorted(glob.glob(path+"\SKIN\IOD\*.nc"))
iod_skin = xr.open_mfdataset(file,parallel=True,chunks={"lat": 1,"lon":1})
iod_skin_df = iod_skin.mean(['x','y']).to_dataframe()
iod_skin_df = iod_skin_df.SKIN[1::]

iod_skin_anomaly = month_anomaly(file)
iod_skin_anomaly_df = iod_skin_anomaly.SKIN.mean(dim=['x','y']).to_dataframe()
iod_skin_anomaly_df_index = iod_skin_anomaly_df['SKIN']/iod_skin_df.std()

file = sorted(glob.glob(path+"\AIR_SURFACE\IOD\*.nc"))
iod_air = xr.open_mfdataset(file,parallel=True,chunks={"lat": 1,"lon":1})
iod_air_df = iod_air.mean(['x','y']).to_dataframe()
iod_air_df = iod_air_df.AIR[1::]

file = sorted(glob.glob(path+"\TROP\IOD\*.nc"))
iod_trop = xr.open_mfdataset(file,parallel=True,chunks={"lat": 1,"lon":1})
iod_trop_df = iod_trop.mean(['x','y']).to_dataframe()
iod_trop_df = iod_trop_df.TROP[1::]

iod_decomp, iod_decomp_df, iod_deseas_df = deseason(iod_skin_df)


#LIMPOPO
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\AIRS'
file = sorted(glob.glob(path+"\SKIN\LIMPOPO\*.nc"))
limpopo_skin = xr.open_mfdataset(file,parallel=True,chunks={"lat": 1,"lon":1})
limpopo_skin_df = limpopo_skin.mean(['x','y']).to_dataframe()
limpopo_skin_df = limpopo_skin_df.SKIN[1::]

file = sorted(glob.glob(path+"\AIR_SURFACE\LIMPOPO\*.nc"))
limpopo_air = xr.open_mfdataset(file,parallel=True,chunks={"lat": 1,"lon":1})
limpopo_air_df = limpopo_air.mean(['x','y']).to_dataframe()
limpopo_air_df = limpopo_air_df.AIR[1::]

limpopo_air_anomaly = month_anomaly(file)
limpopo_air_anomaly_df = limpopo_air_anomaly.AIR.mean(dim=['x','y']).to_dataframe()
limpopo_air_anomaly_df_index = limpopo_air_anomaly_df['AIR']/limpopo_air_df.std()

file = sorted(glob.glob(path+"\TROP\LIMPOPO\*.nc"))
limpopo_trop = xr.open_mfdataset(file,parallel=True,chunks={"lat": 1,"lon":1})
limpopo_trop_df = limpopo_trop.mean(['x','y']).to_dataframe()
limpopo_trop_df = limpopo_trop_df.TROP[1::]

limpopo_decomp, limpopo_decomp_df, limpopo_deseas_df = deseason(limpopo_air_df)




fig1, ax = plt.subplots()
plt.rc('font', size = 8)
ax.plot(limpopo_air_df,color='blue')
ax.plot(limpopo_skin_df,color='red')
ax.plot(et_deseas_df,color='green')
ax.legend(['Observed','Seasonality','Observed without seasonality'])


########################################
#Change in Total Water Storage

#Analyzed the code below in Excel to make dataset continuous
#Need to code faster method later (12/3/21)
'''
path = r'C:\Users\robin\Box\Data\Groundwater\GRACE_MASCON_RL06_V2'
files = sorted(glob.glob(path+"/*.nc"))
grace = xr.open_mfdataset(files[1],parallel=True,chunks={"lat": 10,"lon":10}).lwe_thickness.salem.roi(shape=shapefile)
scale_factor = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 10,"lon":10}).salem.roi(shape=shapefile)
land_mask = xr.open_mfdataset(files[3],parallel=True,chunks={"lat": 10,"lon":10})

shpname = r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
shapefile = gpd.read_file(shpname)

scaled_grace = grace * scale_factor.scale_factor
twsa = scaled_grace - scaled_grace.mean(dim='time')
twsa_df = twsa.mean(['lat','lon']).to_dataframe('Values')
twsa_df = twsa_df[3:192]

os.chdir(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED')
twsa_df.to_csv('TWSA_grace.csv')
'''

twsa_df = pd.read_csv(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\TWSA_grace.csv')
twsa_df.interpolate(method='linear',inplace=True)

twsa_df_values = twsa_df.Values[1::]
twsa_df_dates = pd.date_range(start='10-01-2002',end='01-01-2021',freq='1M')
twsa_df = pd.DataFrame({"Values": list(twsa_df_values)},index=twsa_df_dates)

decomp_twsa = smapi.tsa.seasonal_decompose(twsa_df)
#decomp_twsa.plot()
decomp_twsa_df = pd.DataFrame(decomp_twsa.seasonal)
decomp_twsa_df.columns=['Values']
twas_deseas = twsa_df - decomp_twsa_df

fig1, ax = plt.subplots()
plt.rc('font', size = 8)
ax.plot(twsa_df,color='blue')
ax.plot(decomp_twsa_df,color='red')
ax.plot(twas_deseas,color='green')
ax.legend(['Observed','Seasonality','Observed without seasonality'])

#Water Balance Equation
print(len(p_df))
print(len(et_df))
print(len(r_df))
lhs = p_df - et_df - r_df
dates = pd.date_range(start='10-01-2002',end='01-01-2021',freq='1M')

plt.rc('font', size = 20)
fig = plt.figure(figsize=(14,7))
ax = fig.add_subplot()
ax2 = ax.twinx()
ax.grid(True)
ax.plot(dates,lhs,linewidth=3,color='black',label='P-ET-R')
ax2.plot(dates,twsa_df,linewidth=2,color='gray',label='TWS Anomaly')
ax2.set_ylim(-15,20)
ax.set_ylim(-112.5,150)
ax.set_ylabel('P-ET-R (mm)')
ax2.set_ylabel('Total Water Storage Anomaly (mm)',rotation=-90, labelpad=25)
ax.legend(loc='upper right')
ax2.legend(loc='upper right',bbox_to_anchor=(1, 0.9))
ax.set_xlabel('Year')

plt.rc('font', size = 20)
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot()
ax.scatter(lhs,twsa_df,color='c',label='TWS Anomaly')
ax.set_xlabel('P-ET-R (mm)')
ax.set_ylabel('Total Water Storage Anomaly (mm)')

#Precipitation, Evapotranspiration, Runoff, Soil Moisture, NDVI
plt.rc('font', size = 10)
fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1,figsize=(11,10))

p_df.plot.bar(ax=ax1,color='blue')
ax1.set_xticks(range(4,224,12))
ax1.set_xticklabels([i for i in range(2003,2022,1)], rotation=0)

et_df.plot.bar(ax=ax2,color='blue')
ax2.set_xticks(range(4,224,12))
ax2.set_xticklabels([i for i in range(2003,2022,1)], rotation=0)

r_df.plot.bar(ax=ax3,color='blue')
ax3.set_xticks(range(4,224,12))
ax3.set_xticklabels([i for i in range(2003,2022,1)], rotation=0)

sm_df.plot.bar(ax=ax4,color='blue')
ax3.set_xticks(range(4,224,12))
ax3.set_xticklabels([i for i in range(2003,2022,1)], rotation=0)

ndvi_df.plot.bar(ax=ax5,color='blue')
ax3.set_xticks(range(4,224,12))
ax3.set_xticklabels([i for i in range(2003,2022,1)], rotation=0)

ax1.grid(True)
ax1.set_xlabel(None)
ax1.set_ylabel('Precipitation (mm)')

ax2.grid(True)
ax2.set_xlabel(None)
ax2.set_ylabel('Evapotranspiration (mm)')

ax3.grid(True)
ax3.set_xlabel(None)
ax3.set_ylabel('Runoff (kg/m^2)')

ax4.grid(True)
ax4.set_xlabel(None)
ax4.set_ylabel('Soil Moisture (kg/m^2)')

ax5.grid(True)
ax5.set_xlabel(None)
ax5.set_ylabel('NDVI')


fig.text(0.95,0.93,'a')
fig.text(0.95,0.45,'b')
fig.legend(['Nile','Upper Blue Nile'],loc='upper left',bbox_to_anchor=(0.1,0.97))
fig.tight_layout()