import pandas as pd
import xarray as xr
import rioxarray
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
import skimage
import rasterstats as rs
from rasterio.enums import Resampling
from datetime import datetime

#############
# #OLD DATA (V2)
# path = r'C:\Users\robin\Box\Data\Groundwater\GRACE_MASCON_RL06_V2'
# files = sorted(glob.glob(path+"/*.nc"))
# shpname = r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
# limpopo = gpd.read_file(shpname)

# grace = xr.open_mfdataset(files[1],parallel=True,chunks={"lat": 100,"lon":100}).lwe_thickness.rio.set_spatial_dims('lon','lat',inplace=True).rio.write_crs('WGS84').rio.clip(limpopo.geometry.apply(mapping), limpopo.crs, drop=True,all_touched=True)
# grace_uncertainty = xr.open_mfdataset(files[1],parallel=True,chunks={"lat": 100,"lon":100}).uncertainty.rio.set_spatial_dims('lon','lat',inplace=True).rio.write_crs('WGS84').rio.clip(limpopo.geometry.apply(mapping), limpopo.crs, drop=True,all_touched=True)
# scale_factor = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 100,"lon":100}).rio.set_spatial_dims('lon','lat',inplace=True).rio.write_crs('WGS84').rio.clip(limpopo.geometry.apply(mapping), limpopo.crs, drop=True,all_touched=True)
# land_mask = xr.open_mfdataset(files[3],parallel=True,chunks={"lat": 100,"lon":100}).rio.set_spatial_dims('lon','lat',inplace=True).rio.write_crs('WGS84').rio.clip(limpopo.geometry.apply(mapping), limpopo.crs, drop=True,all_touched=True)
# grace_scaled_old = grace * scale_factor.scale_factor

# #twsa_da = grace_scaled - grace_scaled.mean(dim='time')
# grace_scaled_old = grace_scaled_old.resample(time='1M').mean() #GAP FILL


#####################
#10-21-22
path = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\validation_points'
files = sorted(glob.glob(path+"/*.shp"))

shapefile = gpd.read_file(files[0])

#unique_vals = [np.unique(shapefile[i]) for i in ['sent_17','sent_21','mod_05','mod_10','mod_15','mod_20','SA_20','gw_prod','gw_stora','depth_gw','hygeo_aqfr']]
sentinel_key = {2:'trees',5:'crops',7:'urban',11:'rangeland'}
modis_key = {6:'closed_shrub',7:'open_shrub',8:'woody_sav',9:'sav',10:'grass',12:'crops',13:'urban'}
sa_key = {2:'forest_low',3:'forest_dense',4:'open_woodland',5:'forest_dense_plantation',13:'grass_natural',22:'wetland_herb',23:'wetland_herb',25:'barren_rock',27:'barren_eroded',31:'barren_other',
            38:'crop_annual_pivot_irrig',40:'crop_annual_rainfed',41:'crop_subsist',42:'trees_cultiv',43:'bush_cultiv',44:'grass_cultiv',
            47:'urban_formal_tree',48:'urban_formal_bush',50:'urban_formal_bare',52:'urban_informal_bush',55:'urban_village',
            58:'urban_remote_bush',66:'industrial',67:'roads_rails'}
gwprod_key = {1:'M_1to5_L/s',5:'LM_0.5to1_L/s',6:'L_0.1to0.5_L/s'}
gwstor_key = {1:'LM_1kto10k_mm',5:'L_<1k_mm'}
gwdepth_key = {1:'SM_25to50_mbgl',2:'VS_0to7mbgl',4:'M_50to100mbgl',5:'S_7to25mbgl'}
aqfr_recharge_key = {12:'MAJOR_low_2to20mm/yr',22:'COMPLEX_low_vlow_<20mm/yr',23:'COMPLEX_medium_20to100mm/yr',33:'LOCAL_SHALLOW_med_vlow_<100mm/yr'}

values = [shapefile[i] for i in ['sent_17','sent_21','mod_05','mod_10','mod_15','mod_20','SA_20','gw_prod','gw_stora','depth_gw','hygeo_aqfr']]
lulc_index = ['sent_17','sent_21','mod_05','mod_10','mod_15','mod_20','SA_20','gw_prod','gw_stora','depth_gw','hygeo_aqfr']

#####################
#10-10-2022 -- why does calling all data corrupt the file?? -- individual is fine

path = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\GRACE_Mascons\JPL'
files = sorted(glob.glob(path+"/*.nc"))
shpname = r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
shapefile = gpd.read_file(shpname)

grace_fo = [xr.open_mfdataset(file,parallel=True,chunks={"lat": 100,"lon":100}).lwe_thickness for file in files]
grace_limpopo = [grace.rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)
                    for grace in grace_fo]
grace_limpopo_new_xr = xr.concat(grace_limpopo,dim='time')


path = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\GRACE_Mascons'
files = sorted(glob.glob(path+"/*.nc"))

grace_old = xr.open_mfdataset(files[2],parallel=True,chunks={"lat": 100,"lon":100}).lwe_thickness
scale_factor = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 100,"lon":100}).scale_factor
scaled_grace_old = grace_old * scale_factor

grace_limpopo_old = scaled_grace_old.rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)
grace_limpopo_old.mean(dim={'lat','lon'})

plt.figure
plt.plot(grace_limpopo_old.time,grace_limpopo_old.mean(dim={'lat','lon'}))
plt.plot(grace_limpopo_new_xr.time,grace_limpopo_new_xr.mean(dim={'lat','lon'})*100,'--')



#####################
#10-09-2022
#Print TIFs & dataframes

path = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\GRACE Mascons'
files = sorted(glob.glob(path+"/*.nc"))

grace_old = xr.open_mfdataset(files[1],parallel=True,chunks={"lat": 10,"lon":10}).lwe_thickness
grace_new = xr.open_mfdataset(files[2],parallel=True,chunks={"lat": 10,"lon":10}).lwe_thickness
scale_factor = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 10,"lon":10}).scale_factor
land_mask = xr.open_mfdataset(files[4],parallel=True,chunks={"lat": 10,"lon":10})

shpname = r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
shapefile = gpd.read_file(shpname)

#grace_limpopo = grace.rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)
#index for time slice (197 months): 04/2002 [0] through 05/2021 [196]

scaled_grace_new = grace_new * scale_factor
scaled_grace_old = grace_old * scale_factor


scaled_grace = scaled_grace - scaled_grace.mean(dim='time')
twsa = scaled_grace.salem.roi(shape=shapefile)

twsa_df = twsa.mean(['lat','lon']).to_dataframe('Values')





lon = np.array(grace_limpopo[1][0].lon)
lat = np.array(grace_limpopo[1][0].lat)

new_array = xr.DataArray(grace_limpopo[1][0], dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name="LiqWaterEquiv_cm")
new_array.rio.set_spatial_dims('lon','lat',inplace=True)
new_array.rio.set_crs("epsg:4326")
new_array.rio.to_raster(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\GRACE_Mascons\tif\v04.tif')




#####################
#10-02-22
#Validation of GRACE anomalies

path = r'C:\Users\robin\Box\Data\StudyRegion\Limpopo'
files = sorted(glob.glob(path+"/*.csv"))

boreholes = pd.read_csv(files[1])
boreholes_data = boreholes[~pd.isna(boreholes['WaterLevel_WaterLevel'])]
boreholes_data.WaterLevel_MeasurementDateAndTime = pd.to_datetime(boreholes_data.WaterLevel_MeasurementDateAndTime)

#203 sites in the LRB have at least one observation at one time (2000/10 - 2022/10)
sites = boreholes_data.GeositeInfo_Identifier.unique()
sites_data = [(boreholes_data[boreholes_data['GeositeInfo_Identifier'] == sites[i]]) for i in range(0,len(sites))]
days_range = [pd.date_range((sites_data[i].WaterLevel_MeasurementDateAndTime).dt.date.iloc[0],
                           (sites_data[i].WaterLevel_MeasurementDateAndTime).dt.date.iloc[-1],freq='D')
                           for i in range(0,len(sites))]

#93 sites in LRB with at two month observations
val_sites = sites[np.array([len(days) for days in days_range])>62]
val_sites_data = [(boreholes_data[boreholes_data['GeositeInfo_Identifier'] == val_sites[i]]) for i in range(0,len(val_sites))]

days_range = [pd.date_range((val_sites_data[i].WaterLevel_MeasurementDateAndTime).dt.date.iloc[0],
                           (val_sites_data[i].WaterLevel_MeasurementDateAndTime).dt.date.iloc[-1],freq='D')
                           for i in range(0,len(val_sites))]

val_sites_gdf = [gpd.GeoDataFrame(val_sites_data[i],geometry=gpd.points_from_xy(val_sites_data[i]['GeositeInfo_Longitude'],val_sites_data[i]['GeositeInfo_Latitude']))
                    for i in range(0,len(val_sites))]

#Most recent observation is from 01/2005 
#Therefore, limit TWS observations from 04/2002 to 01/2005 (indices 0:31) !!!
# Until 2007, not 2005, but 2007 is single point

#plt.plot(val_sites_gdf[0].WaterLevel_MeasurementDateAndTime,val_sites_gdf[0].WaterLevel_WaterLevel,'*')
for data in val_sites_gdf:
    fig,ax = plt.subplots()
    ax.plot(data.WaterLevel_MeasurementDateAndTime,data.WaterLevel_WaterLevel,'*')
    plt.xticks(rotation=45)
    ax.set_xlabel('Date')

#GRACE TWSA 
path = r'C:\Users\robin\Box\Data\Groundwater\GRACE_MASCON_RL06_V2'
files = sorted(glob.glob(path+"/*.nc"))
shpname = r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
limpopo = gpd.read_file(shpname)

grace = xr.open_mfdataset(files[1],parallel=True,chunks={"lat": 100,"lon":100}).lwe_thickness.rio.set_spatial_dims('lon','lat',inplace=True).rio.write_crs('WGS84').rio.clip(limpopo.geometry.apply(mapping), limpopo.crs, drop=True,all_touched=True)
grace_uncertainty = xr.open_mfdataset(files[1],parallel=True,chunks={"lat": 100,"lon":100}).uncertainty.rio.set_spatial_dims('lon','lat',inplace=True).rio.write_crs('WGS84').rio.clip(limpopo.geometry.apply(mapping), limpopo.crs, drop=True,all_touched=True)
scale_factor = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 100,"lon":100}).rio.set_spatial_dims('lon','lat',inplace=True).rio.write_crs('WGS84').rio.clip(limpopo.geometry.apply(mapping), limpopo.crs, drop=True,all_touched=True)
land_mask = xr.open_mfdataset(files[3],parallel=True,chunks={"lat": 100,"lon":100}).rio.set_spatial_dims('lon','lat',inplace=True).rio.write_crs('WGS84').rio.clip(limpopo.geometry.apply(mapping), limpopo.crs, drop=True,all_touched=True)
grace_scaled = grace * scale_factor.scale_factor

twsa_da = grace_scaled - grace_scaled.mean(dim='time')







#Runoff (incorrectly labeled as "SM", but still runoff)
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_RUNOFF'
files = sorted(glob.glob(path+"/*.nc"))
r = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 10,"lon":10}).rio.write_crs('WGS84')

path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_SOILMOISTURE'
files = sorted(glob.glob(path+"/*.nc"))
sm = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 10,"lon":10}).rio.write_crs('WGS84')

scaled_grace_interp = grace_scaled.rio.reproject_match(r,resampling=Resampling.nearest)

#OPTIONS
#A - reindex/resample to monthly for TWS and choose common dates
#https://stackoverflow.com/questions/67052852/how-to-drop-unmatching-time-series-from-two-xarray-time-series-dataset
#B - resample by frequency 
scaled_grace_interp = scaled_grace_interp.resample(time='1M').mean() #GAP FILL


dGW = scaled_grace_interp[0:229] - sm.SM[27::]/10 - r.SM[27::]/10

dGW_mean = dGW.mean(dim={'x','y'})

plt.figure
plt.plot(dGW_mean.time,scaled_grace_interp.mean(dim={'x','y'})[:-1])
plt.plot(dGW_mean.time,dGW_mean)
plt.plot(dGW_mean.time,np.diff((sm.SM[26::]/10).mean(dim={'x','y'})))
plt.plot(dGW_mean.time,np.diff((r.SM[26::]/10).mean(dim={'x','y'})))



#GLDAS CLSM
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_CLSM'
files = sorted(glob.glob(path+"/*.nc"))

gw = xr.open_mfdataset(files[0],parallel=True).mm
tws = xr.open_mfdataset(files[-1],parallel=True).mm

sm_rz = xr.open_mfdataset(files[3],parallel=True)['kgm-2']
sm_surf = xr.open_mfdataset(files[4],parallel=True)['kgm-2']

gw_sm = sm_rz+sm_surf+gw

#need to average on a raster by raster basis for new GRACE version (concatenate/xarray corrupts it)
grace_limpopo_new = np.empty([len(grace_limpopo),8,10])
for i in range(0,len(grace_limpopo)):
    grace_limpopo_new[i,:,:] = np.array(grace_limpopo[i])

#grace_limpopo_new = np.nanmean(grace_limpopo_new,axis=0)
grace_limpopo_new = np.nanmean(np.nanmean(grace_limpopo_new,axis=1),axis=1)

fig,ax = plt.subplots()
ax1 = ax.twinx()
gw.mean(['x','y']).plot(ax=ax,color='red')
tws.mean(['x','y']).plot(ax=ax,color='green')
(grace_limpopo_old.mean(['lon','lat'])*10).plot(ax=ax1,color='blue')
ax1.plot(grace_limpopo_new_xr.time, grace_limpopo_new*1000)




#################
#Sophia's InSitu Borehole Data
#Borehole water levels - 227 sites
filepath = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\sophia\Limpopo In-Situ Data\Boreholes\LP\LP_WaterLevels\*.csv'
csv_files = sorted(list(glob.glob(filepath)))
waterlevel_dfs = [pd.read_csv(file, names=['site', 'date','time','level','?','??'],index_col=False) for file in csv_files]
waterlevel_dates_df =  [np.stack([datetime.strptime(str(date), '%Y%m%d') for date in waterlevel_dfs[i].date]) for i in range(0,len(waterlevel_dfs))]
waterlevel_dfs = [pd.concat([waterlevel_dfs[i]['site'], pd.Series(waterlevel_dates_df[i]).rename('date'), waterlevel_dfs[i]['time'],waterlevel_dfs[i]['level']],axis=1) for i in range(0,len(waterlevel_dfs))] 
waterlevel_sites = [(waterlevel_dfs[i]['site'][0]).strip() for i in range(0,len(waterlevel_dfs)) ]

#Resampled to day-average 
waterlevel_dfs_daily = [waterlevel_dfs[i].groupby('date').mean() for i in range(0,len(waterlevel_dfs))]
waterlevel_dfs_daily = [waterlevel_dfs_daily[i][(waterlevel_dfs_daily[i]>-100) & (waterlevel_dfs_daily[i]!=0)] for i in range(0,len(waterlevel_dfs_daily))]

#Borehole shapefile - 655 sites (therefore filter out for those with water level data)
shpname = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\sophia\Limpopo In-Situ Data\Boreholes\LP\LP_BOREHOLES.shp'
boreholes = gpd.read_file(shpname).to_crs({'init': 'epsg:4326'})

boreholes_boolean = [boreholes.loc[i].F_STATION in waterlevel_sites for i in range(0,len(boreholes))]
boreholes = boreholes[boreholes_boolean]
#boreholes.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\validation_points\boreholes.csv')


#################
#Which data? 1)dGW  2)TWSA
#1 -- dGW
monthly = [rs.point_query(boreholes, np.array(dGW[i]), affine= dGW[i].rio.transform(), geojson_out=True, interpolate='nearest') for i in range(0,len(dGW))]
time_series_by_month = np.array([ [monthly[month][site]['properties']['value'] for site in range(0,len(boreholes))] for month in range(0,len(monthly))])
time_series_by_site = time_series_by_month.T

for i,site in zip(range(0,len(waterlevel_dfs_daily)),waterlevel_sites):
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax1.plot(waterlevel_dfs_daily[i].index,waterlevel_dfs_daily[i].level,'*',color='C1')
    ax1.set_ylabel('Water Level')
    ax2.plot(dGW.time,time_series_by_site[i])
    ax2.set_ylabel('dGW')
    ax1.set_xlabel('Date')
    ax1.set_title('{}'.format(site))
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\\boreholes_dGW\{}'.format(site))

#################
#2 -- v02 TWSA
monthly = [rs.point_query(boreholes, np.array(twsa_da[i]), affine= twsa_da[i].rio.transform(), geojson_out=True,interpolate='nearest') for i in range(0,len(twsa_da))]
time_series_by_month = np.array([ [monthly[month][site]['properties']['value'] for site in range(0,len(boreholes))] for month in range(0,len(monthly))])
time_series_by_site = time_series_by_month.T

for i,site in zip(range(0,len(waterlevel_dfs_daily)),waterlevel_sites):
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax1.plot(waterlevel_dfs_daily[i].index,waterlevel_dfs_daily[i].level,'*',color='C1')
    ax1.set_ylabel('Water Level')
    ax2.plot(grace_scaled.time,time_series_by_site[i])
    ax2.set_ylabel('TWSA')
    ax1.set_xlabel('Date')
    ax1.set_title('{}'.format(site))
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\\boreholes_TWSA_v02\{}'.format(site))

#################
#3 -- v04 TWSA
monthly = [rs.point_query(boreholes, np.array(grace_limpopo[i][0]), affine= grace_limpopo[i][0].rio.transform(), geojson_out=True,interpolate='nearest') for i in range(0,len(grace_limpopo_new))]
time_series_by_month = np.array([ [monthly[month][site]['properties']['value'] for site in range(0,len(boreholes))] for month in range(0,len(monthly))])
time_series_by_site = time_series_by_month.T

for i,site in zip(range(0,len(waterlevel_dfs_daily)),waterlevel_sites):
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax1.plot(waterlevel_dfs_daily[i].index,waterlevel_dfs_daily[i].level,'*',color='C1')
    ax1.set_ylabel('Water Level')
    ax2.plot(grace_limpopo_new_xr.time,time_series_by_site[i])
    ax2.set_ylabel('TWSA')
    ax1.set_xlabel('Date')
    ax1.set_title('{}'.format(site))
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\\boreholes_TWSA_v04\{}'.format(site))

fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
for i,site in zip(range(0,len(waterlevel_dfs_daily)),waterlevel_sites):
    plt.xticks(rotation=45)
    #ax1.plot(waterlevel_dfs_daily[i].index,waterlevel_dfs_daily[i].level,color='C1')
    ax1.set_ylabel('Water Level')
    ax2.plot(grace_limpopo_new_xr.time,time_series_by_site[i])
    ax2.set_ylabel('TWSA')
    ax1.set_xlabel('Date')
    ax1.set_title('{}'.format(site))


#################
#4 -- CLSM GW
monthly = [rs.point_query(boreholes, np.array(gw[i]), affine= gw[i].rio.transform(), geojson_out=True, interpolate='nearest') for i in range(0,len(gw))]
time_series_by_month = np.array([ [monthly[month][site]['properties']['value'] for site in range(0,len(boreholes))] for month in range(0,len(monthly))])
time_series_by_site = time_series_by_month.T

fig,ax1 = plt.subplots()
for i,site,sent,mod05,mod20,SA,prod,stor,depth,aqfr in zip(range(0,len(waterlevel_dfs_daily)),waterlevel_sites,values[0],values[2],values[5],values[6],values[7],values[8],values[9],values[10]):
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax1.plot(waterlevel_dfs_daily[i].index,waterlevel_dfs_daily[i].level,'*',color='C1')
    ax1.set_ylabel('Water Level')
    ax2.plot(gw.time,time_series_by_site[i])
    ax2.set_ylabel('GW')
    ax1.set_xlabel('Date')
    ax2.set_ylim(0,1500)

for i,site,sent,mod05,mod20,SA,prod,stor,depth,aqfr in zip(range(0,len(waterlevel_dfs_daily)),waterlevel_sites,values[0],values[2],values[5],values[6],values[7],values[8],values[9],values[10]):
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax1.plot(waterlevel_dfs_daily[i].index,waterlevel_dfs_daily[i].level,'*',color='C1')
    ax1.set_ylabel('Water Level')
    ax2.plot(gw.time,time_series_by_site[i])
    ax2.set_ylabel('GW')
    ax1.set_xlabel('Date')
    ax1.set_title('{}_LULC: sent17 {},modis_05 {}, mod20 {}, SA20 {}, prod {}, stor {}, dep {}, aqfr {}'.format(site,sent,mod05,mod20,SA,prod,stor,depth,aqfr))
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\\boreholes_gw_clsm\{}'.format(site))



#################
#5 -- CLSM TWS
monthly = [rs.point_query(boreholes, np.array(tws[i]), affine= tws[i].rio.transform(), geojson_out=True, interpolate='nearest') for i in range(0,len(tws))]
time_series_by_month = np.array([ [monthly[month][site]['properties']['value'] for site in range(0,len(boreholes))] for month in range(0,len(monthly))])
time_series_by_site = time_series_by_month.T

fig,ax1 = plt.subplots()
for i,site in zip(range(0,len(waterlevel_dfs_daily)),waterlevel_sites):
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax1.plot(waterlevel_dfs_daily[i].index,waterlevel_dfs_daily[i].level,'*',color='C1')
    ax1.set_ylabel('Water Level')
    ax2.plot(tws.time,time_series_by_site[i])
    ax2.set_ylabel('TWS')
    ax1.set_xlabel('Date')
    ax1.set_title('{}'.format(site))
    ax2.set_ylim(0,1500)

for i,site in zip(range(0,len(waterlevel_dfs_daily)),waterlevel_sites):
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax1.plot(waterlevel_dfs_daily[i].index,waterlevel_dfs_daily[i].level,'*',color='C1')
    ax1.set_ylabel('Water Level')
    ax2.plot(tws.time,time_series_by_site[i])
    ax2.set_ylabel('dGW')
    ax1.set_xlabel('Date')
    ax1.set_title('{}'.format(site))
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\\boreholes_tws_clsm\{}'.format(site))


#################
#6 -- CLSM GW + SM (Surface + RootZone)
monthly = [rs.point_query(boreholes, np.array(gw_sm[i]), affine= gw_sm[i].rio.transform(), geojson_out=True, interpolate='nearest') for i in range(0,len(tws))]
time_series_by_month = np.array([ [monthly[month][site]['properties']['value'] for site in range(0,len(boreholes))] for month in range(0,len(monthly))])
time_series_by_site = time_series_by_month.T

fig,ax1 = plt.subplots()
for i,site in zip(range(0,len(waterlevel_dfs_daily)),waterlevel_sites):
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax1.plot(waterlevel_dfs_daily[i].index,waterlevel_dfs_daily[i].level,'*',color='C1')
    ax1.set_ylabel('Water Level')
    ax2.plot(gw_sm.time,time_series_by_site[i])
    ax2.set_ylabel('GW + SM')
    ax1.set_xlabel('Date')
    ax1.set_title('{}'.format(site))
    ax2.set_ylim(0,1500)


#################
#7 -- CLSM SM Surface
monthly = [rs.point_query(boreholes, np.array(sm_surf[i]), affine= sm_surf[i].rio.transform(), geojson_out=True, interpolate='nearest') for i in range(0,len(tws))]
time_series_by_month = np.array([ [monthly[month][site]['properties']['value'] for site in range(0,len(boreholes))] for month in range(0,len(monthly))])
time_series_by_site = time_series_by_month.T

fig,ax1 = plt.subplots()
for i,site in zip(range(0,len(waterlevel_dfs_daily)),waterlevel_sites):
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax1.plot(waterlevel_dfs_daily[i].index,waterlevel_dfs_daily[i].level,'*',color='C1')
    ax1.set_ylabel('Water Level')
    ax2.plot(sm_surf.time,time_series_by_site[i])
    ax2.set_ylabel('Surface SM')
    ax1.set_xlabel('Date')
    ax1.set_title('{}'.format(site))
    ax2.set_ylim(0,100)


#################
#7 -- CLSM SM Root Zone
monthly = [rs.point_query(boreholes, np.array(sm_rz[i]), affine= sm_rz[i].rio.transform(), geojson_out=True, interpolate='nearest') for i in range(0,len(tws))]
time_series_by_month = np.array([ [monthly[month][site]['properties']['value'] for site in range(0,len(boreholes))] for month in range(0,len(monthly))])
time_series_by_site = time_series_by_month.T

fig,ax1 = plt.subplots()
for i,site in zip(range(0,len(waterlevel_dfs_daily)),waterlevel_sites):
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax1.plot(waterlevel_dfs_daily[i].index,waterlevel_dfs_daily[i].level,'*',color='C1')
    ax1.set_ylabel('Water Level')
    ax2.plot(sm_rz.time,time_series_by_site[i])
    ax2.set_ylabel('Root Zone SM')
    ax1.set_xlabel('Date')
    ax1.set_title('{}'.format(site))
    ax2.set_ylim(0,1500)

#################
#validation_points = gpd.GeoDataFrame([val_sites.iloc[0] for val_sites in val_sites_gdf]).to_csv(r'C:\Users\robin\Box\Data\StudyRegion\Limpopo\SA_val_sites.csv')
validation_points = gpd.read_file(r'C:\Users\robin\Box\Data\StudyRegion\Limpopo\SA_val_sites.shp')

twsa_dates = dGW.time[0:31]
twsa = dGW[0:31]
monthly = [rs.point_query(validation_points, np.array(twsa[i]), affine= twsa[i].rio.transform(), geojson_out=True, interpolate='nearest') for i in range(0,len(twsa_dates))]
#1 hour to process; infinite time to open...

#93 points, 31 months
time_series_by_month = np.array([ [monthly[month][site]['properties']['value'] for site in range(0,len(validation_points))] for month in range(0,len(monthly))])
time_series_by_site = time_series_by_month.T

for i in range(0,93):
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax1.plot(val_sites_gdf[i].WaterLevel_MeasurementDateAndTime,val_sites_gdf[i].WaterLevel_WaterLevel,'*',color='C1')
    ax1.set_ylabel('Water Level')
    ax2.plot(twsa_dates,time_series_by_site[i])
    ax2.set_ylabel('TWSA')
    ax.set_xlabel('Date')

#References
#https://stackoverflow.com/questions/47691228/how-to-fill-missing-date-in-timeseries


#REDO WATER BALANCE EQ

p = xr.open_mfdataset(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GPM_IMERG/*.nc')
p_df = p.P.mean(dim=['x','y']).to_dataframe()
p_df = p_df.P

et = xr.open_mfdataset(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_ET\TERRA\TMODIS_ET_2000_2020_120421.nc',parallel=True,chunks={"x": 500,"y":500})
et_df = et.ET.mean(dim=['x','y']).to_dataframe()
et_std = et_df.std().ET
et_df = et_df[33::].ET

path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_RUNOFF'
files = sorted(glob.glob(path+"/*.nc"))
r = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 10,"lon":10})

r_df = r.mean(['x','y'])
r_std = r_df.std().SM
r_df = r_df.rename({'SM':'R'}).to_dataframe()
r_df = r_df[33:252].R

path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_SOILMOISTURE'
files = sorted(glob.glob(path+"/*.nc"))
sm = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 10,"lon":10})

sm_df = sm.mean(['x','y']).to_dataframe()
sm_std = sm_df.std().SM
sm_df = sm_df.SM[33:252]

print(len(p_df))
print(len(et_df))
print(len(r_df))
lhs = p_df - et_df - r_df
dates = pd.date_range(start='10-01-2002',end='01-01-2021',freq='1M')

dS = scaled_grace_interp.mean(dim={'x','y'}).to_dataframe()

dGW = p_df - et_df - r_df - sm_df scaled_grace_interp[0:229] - sm.SM[27::]/10 - r.SM[27::]/10





#####################
#08-12-21
#Print TIFs & dataframes

path = r'C:\Users\robin\Box\Data\Groundwater\GRACE_MASCON_RL06_V2'
files = sorted(glob.glob(path+"/*.nc"))

grace = xr.open_mfdataset(files[1],parallel=True,chunks={"lat": 10,"lon":10}).lwe_thickness
scale_factor = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 10,"lon":10})
land_mask = xr.open_mfdataset(files[3],parallel=True,chunks={"lat": 10,"lon":10})

shpname = r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
shapefile = gpd.read_file(shpname)

#grace_limpopo = grace.rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)
#index for time slice (197 months): 04/2002 [0] through 05/2021 [196]

scaled_grace = grace * scale_factor.scale_factor
scaled_grace = scaled_grace - scaled_grace.mean(dim='time')
twsa = scaled_grace.salem.roi(shape=shapefile)

twsa_df = twsa.mean(['lat','lon']).to_dataframe('Values')

lon = np.array(grace.lon)
lat = np.array(grace.lat)

new_array = xr.DataArray(grace[0], dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name="LiqWaterEquiv_cm")
new_array.rio.set_spatial_dims('lon','lat',inplace=True)
new_array.rio.set_crs("epsg:4326")
new_array.rio.to_raster(path+'\\LWE_2002_04_global.tif')


