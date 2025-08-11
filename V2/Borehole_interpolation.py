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
import csv
import os
from pathlib import Path
import glob
import skimage
from rasterio.enums import Resampling
from datetime import datetime
import geopandas as gpd

import rasterio.mask
from rasterio.plot import show
from rasterio.transform import Affine
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import box
from shapely.geometry import Polygon, Point
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

degree_sign = u"\N{DEGREE SIGN}"
plt.rcParams["font.family"] = "Times New Roman"
def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

#Borehole data
filepath = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\sophia\Limpopo In-Situ Data\Boreholes\LP\LP_WaterLevels\*.csv'
csv_files = sorted(list(glob.glob(filepath)))
waterlevel_dfs = [pd.read_csv(file, names=['site', 'date','time','level','?','??'],index_col=False) for file in csv_files]
waterlevel_dates_df =  [np.stack([datetime.strptime(str(date), '%Y%m%d') for date in waterlevel_dfs[i].date]) for i in range(0,len(waterlevel_dfs))]
waterlevel_dfs = [pd.concat([waterlevel_dfs[i]['site'], pd.Series(waterlevel_dates_df[i]).rename('Date'), waterlevel_dfs[i]['time'],waterlevel_dfs[i]['level']],axis=1) for i in range(0,len(waterlevel_dfs))] 
waterlevel_sites = [(waterlevel_dfs[i]['site'][0]).strip() for i in range(0,len(waterlevel_dfs)) ]
#Resampled to day-average
waterlevel_dfs_daily = [waterlevel_dfs[i].groupby('Date').mean() for i in range(0,len(waterlevel_dfs))]
waterlevel_dfs_daily = [waterlevel_dfs_daily[i][(waterlevel_dfs_daily[i]>-100) & (waterlevel_dfs_daily[i]!=0)] for i in range(0,len(waterlevel_dfs_daily))]
#Borehole shapefile - 655 sites (therefore filter out for those with water level data)
shpname = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\sophia\Limpopo In-Situ Data\Boreholes\LP\LP_BOREHOLES.shp'
boreholes = gpd.read_file(shpname).to_crs({'init': 'epsg:4326'})
boreholes_boolean = [boreholes.loc[i].F_STATION in waterlevel_sites for i in range(0,len(boreholes))]
boreholes = boreholes[boreholes_boolean]


waterlevel_dfs_daily

#Resample & interpolate (gap-fill) monthly borehole data
resample_monthly = [waterlevel.resample('MS').mean() for waterlevel in waterlevel_dfs_daily]
resample_monthly_interp = [waterlevel.resample('MS').mean().interpolate() for waterlevel in waterlevel_dfs_daily]

#Select only long records (>180 months)
resample_boolean_lens =  np.array([ len(df) for df in resample_monthly_interp])
resample_boolean = np.array([ len(df) > 180 for df in resample_monthly_interp])
boolean_index = np.where(resample_boolean)[0]

resample_monthly_interp_long = [resample_monthly_interp[i] for i in boolean_index]
waterlevel_sites_long = [waterlevel_sites[i] for i in boolean_index]

#Calculate anomalies for either all or only long records
resample_monthly_interp_anom = [(df - df.mean())/df.std() for df in resample_monthly_interp]
resample_monthly_interp_long_anom = [(df - df.mean())/df.std() for df in resample_monthly_interp_long]


for i,site in zip(resample_monthly_interp_long_anom,waterlevel_sites_long):
    fig,ax1 = plt.subplots()
    plt.xticks(rotation=45)
    ax1.plot(i.index,i.level,color='C0')
    ax1.set_ylabel('Water Level')
    ax1.set_xlabel('Date')
    ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
    ax1.set_ylim([-10,10])
    ax1.set_title('{}'.format(site))
    #ax1.invert_yaxis()
    #plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\in_situ_boreholes\long_record\anomalies\{}_interp'.format(site))

sites = set(waterlevel_sites_long)
indexed = np.array([i for i, site in enumerate(boreholes.iloc[:,0]) if site in sites])


#ALL RECORD
fig,ax1 = plt.subplots()
for i,site in zip(resample_monthly_interp,waterlevel_sites):
    plt.xticks(rotation=45)
    ax1.plot(i.index,i.level,color='C0')
    ax1.set_ylabel('Water Level')
    ax1.set_xlabel('Date')
    ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
    ax1.set_ylim([-100,1])
    ax1.set_title('{}'.format(site))

fig,ax1 = plt.subplots()
for i,site in zip(resample_monthly_interp_anom,waterlevel_sites):
    plt.xticks(rotation=45)
    ax1.plot(i.index,i.level,color='C0',linewidth=0.5)
    ax1.set_ylabel('Water Level')
    ax1.set_xlabel('Date')
    ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
    ax1.set_ylim([-10,10])
    ax1.set_title('{}'.format(site))
    #ax1.invert_yaxis()


#LONG RECORDS ONLY
fig,ax1 = plt.subplots()
for i,site in zip(resample_monthly_interp_long,waterlevel_sites_long):
    plt.xticks(rotation=45)
    ax1.plot(i.index,i.level,color='C0')
    ax1.set_ylabel('Water Level')
    ax1.set_xlabel('Date')
    ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
    ax1.set_ylim([-100,1])
    ax1.set_title('{}'.format(site))

fig,ax1 = plt.subplots()
for i,site in zip(resample_monthly_interp_long_anom,waterlevel_sites_long):
    plt.xticks(rotation=45)
    ax1.plot(i.index,i.level,color='C0')
    ax1.set_ylabel('Water Level')
    ax1.set_xlabel('Date')
    ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
    ax1.set_ylim([-10,10])
    ax1.set_title('{}'.format(site))
    #ax1.invert_yaxis()


#Concatenate by Time (monthly)
concat_interp = pd.concat(resample_monthly_interp,axis=1)
concat_interp_anom = pd.concat(resample_monthly_interp_anom,axis=1)
concat_interp_long = pd.concat(resample_monthly_interp_long,axis=1)
concat_interp_long_anom = pd.concat(resample_monthly_interp_long_anom,axis=1)

fig,ax1 = plt.subplots()
for i,site in zip(resample_monthly_interp_anom,waterlevel_sites):
    plt.xticks(rotation=45)
    ax1.plot(i.index,i.level,color='C0',linewidth=0.5)
    ax1.set_ylabel('Water Level')
    ax1.set_xlabel('Date')
    ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
    ax1.set_ylim([-10,10])
    ax1.set_title('{}'.format(site))
    #ax1.invert_yaxis()
concat_interp_anom.mean(axis=1).plot(ax=ax1,linewidth=2,color='black')
(concat_interp_anom.mean(axis=1)+3*concat_interp_anom.std(axis=1)).plot(ax=ax1,linewidth=2)
(concat_interp_anom.mean(axis=1)-3*concat_interp_anom.std(axis=1)).plot(ax=ax1,linewidth=2)

fig,ax1 = plt.subplots()
for i,site in zip(resample_monthly_interp_long_anom,waterlevel_sites_long):
    plt.xticks(rotation=45)
    ax1.plot(i.index,i.level,color='C0')
    ax1.set_ylabel('Water Level')
    ax1.set_xlabel('Date')
    ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
    ax1.set_ylim([-10,10])
concat_interp_long_anom.mean(axis=1).plot(ax=ax1)
(concat_interp_long_anom.mean(axis=1)+3*concat_interp_long_anom.std(axis=1)).plot(ax=ax1)
(concat_interp_long_anom.mean(axis=1)-3*concat_interp_long_anom.std(axis=1)).plot(ax=ax1)

#####################################################################################################################################################
#####################################################################################################################################################
#FILTERED FOR Aquifer Characteristics (recharge & depth)
path = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\Hydrogeology_maps'
raster_files = sorted(glob.glob(path+"/*.tif"))
csv_files = sorted(glob.glob(path+"/*.csv"))

shpname = r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
limpopo = gpd.read_file(shpname)


#####################
#GROUNDWATER DEPTH
gw_depth = xr.open_rasterio(raster_files[0],parallel=True,chunks={"x": 100,"y":100}).rio.set_spatial_dims('x','y',inplace=True).rio.write_crs('WGS84').rio.clip(limpopo.geometry.apply(mapping), limpopo.crs, drop=True,all_touched=True)
gw_depth_key = pd.read_csv(csv_files[0]) #1,2,4,5 [0,1,3,4]
gw_depth_points = rs.point_query(boreholes, np.array(gw_depth[0]), affine= gw_depth[0].rio.transform(), geojson_out=True, interpolate='nearest')
gw_depth_values = np.array([int(gw_depth_points[site]['properties']['value']) for site in range(0,len(boreholes))])

boolean_filter_gw_depth = [gw_depth_values == value for value in np.unique(gw_depth_values)]
boolean_filter_gw_depth_long = [(gw_depth_values == value) & resample_boolean for value in np.unique(gw_depth_values)]
gw_depth_indices  = [np.where(boolean_filter)[0] for boolean_filter in boolean_filter_gw_depth]
gw_depth_indices_long  = [np.where(boolean_filter)[0] for boolean_filter in boolean_filter_gw_depth_long]

#Resampled monthly-interpolated GW_DEPTH filtered - WATER LEVEL
resample_monthly_interp_gwdepth_filtered = [[resample_monthly_interp[i] for i in gw_depth_index] for gw_depth_index in gw_depth_indices]
resample_monthly_interp_long_gwdepth_filtered = [[resample_monthly_interp[i] for i in gw_depth_index] for gw_depth_index in gw_depth_indices_long]
#Resampled monthly-interpolated GW_DEPTH filtered - ANOMALIES
resample_monthly_interp_anom_gwdepth_filtered = [[(df - df.mean())/df.std() for df in df_gw_depth_filtered] for df_gw_depth_filtered in resample_monthly_interp_gwdepth_filtered]
resample_monthly_interp_long_anom_gwdepth_filtered = [[(df - df.mean())/df.std()  for df in df_gw_depth_filtered] for df_gw_depth_filtered in resample_monthly_interp_long_gwdepth_filtered]
#Borehole Site names -- filtered for GW_DEPTH
waterlevel_sites_gwdepth_filtered = [[waterlevel_sites[i] for i in gw_depth_index] for gw_depth_index in gw_depth_indices]
waterlevel_sites_long_gwdepth_filtered = [[waterlevel_sites[i] for i in gw_depth_index] for gw_depth_index in gw_depth_indices_long]


#GROUNDWATER_DEPTH FILTERED
#ALL RECORD
for level,waterlevel_name,gw_depth in zip(resample_monthly_interp_gwdepth_filtered,waterlevel_sites_gwdepth_filtered,['25-50 m','0-7 m', '50-100 m','7-25 m']):
    fig,ax1 = plt.subplots()
    for i in level:
        plt.xticks(rotation=45)
        ax1.plot(i.index,i.level,color='C0',linewidth=0.5)
        ax1.set_ylabel('Water Level')
        ax1.set_xlabel('Date')
        ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
        ax1.set_ylim([-100,1])
        ax1.set_title('{}'.format(gw_depth))

for anom,waterlevel_name,gw_depth in zip(resample_monthly_interp_anom_gwdepth_filtered,waterlevel_sites_gwdepth_filtered,['25-50 m','0-7 m', '50-100 m','7-25 m']):
    fig,ax1 = plt.subplots()
    for i in anom:
        plt.xticks(rotation=45)
        ax1.plot(i.index,i.level,color='C0',linewidth=0.5)
        ax1.set_ylabel('Water Level Anomaly')
        ax1.set_xlabel('Date')
        ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
        ax1.set_ylim([-10,10])
        ax1.set_title('{}'.format(gw_depth))

#LONG RECORDS ONLY
for level,waterlevel_name,gw_depth in zip(resample_monthly_interp_long_gwdepth_filtered,waterlevel_sites_long_gwdepth_filtered,['25-50 m','0-7 m', '50-100 m','7-25 m']):
    fig,ax1 = plt.subplots()
    for i in level:
        plt.xticks(rotation=45)
        ax1.plot(i.index,i.level,color='black',linewidth=0.5)
        ax1.set_ylabel('Water Level')
        ax1.set_xlabel('Date')
        ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
        ax1.set_ylim([-100,1])
        ax1.set_title('{}'.format(gw_depth))

for anom,waterlevel_name,gw_depth in zip(resample_monthly_interp_long_anom_gwdepth_filtered,waterlevel_sites_long_gwdepth_filtered,['25-50 m','0-7 m', '50-100 m','7-25 m']):
    fig,ax1 = plt.subplots()
    for i in anom:
        plt.xticks(rotation=45)
        ax1.plot(i.index,i.level,color='black',linewidth=0.5)
        ax1.set_ylabel('Water Level Anomaly')
        ax1.set_xlabel('Date')
        ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
        ax1.set_ylim([-5,5])
        ax1.set_title('{}'.format(gw_depth))

#Concatenate by Time (monthly)
concat_interp = [pd.concat(level,axis=1) for level in resample_monthly_interp_gwdepth_filtered]
concat_interp_anom = [pd.concat(anom,axis=1) for anom in resample_monthly_interp_anom_gwdepth_filtered]
concat_interp_long = [pd.concat(level,axis=1) for level in resample_monthly_interp_long_gwdepth_filtered]
concat_interp_long_anom = [pd.concat(anom,axis=1) for anom in resample_monthly_interp_long_anom_gwdepth_filtered]

for concat,anom,gw_depth in zip(concat_interp_long_anom,resample_monthly_interp_long_anom_gwdepth_filtered,['25 - 50 m','0 - 7 m', '50 - 100 m','7 - 25 m']):
    fig,ax1 = plt.subplots()
    for i in anom:
        plt.xticks(rotation=45)
        ax1.plot(i.index,i.level,color='gray',linewidth=0.5)
        ax1.set_ylabel('Anomaly')
        ax1.set_xlabel('Date')
        ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
        ax1.set_ylim([-5,5])
        ax1.set_title('{}'.format(gw_depth))
        #ax1.invert_yaxis()
    concat.mean(axis=1).plot(ax=ax1,linewidth=2,color='black')
    #(concat.mean(axis=1)+concat.std(axis=1)).plot(ax=ax1,linewidth=2)
    #(concat.mean(axis=1)-concat.std(axis=1)).plot(ax=ax1,linewidth=2)
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\FinalFigures\Plots\boreholes_by_depth_{}.pdf'.format(gw_depth))


#####################
#RECHARGE RATE
recharge_rate = xr.open_rasterio(raster_files[-1],parallel=True,chunks={"x": 100,"y":100}).rio.set_spatial_dims('x','y',inplace=True).rio.write_crs('WGS84').rio.clip(limpopo.geometry.apply(mapping), limpopo.crs, drop=True,all_touched=True)
recharge_rate_key = pd.read_csv(csv_files[-1]) #12,22,23,33 [3,8,7,10]
recharge_rate_points = rs.point_query(boreholes, np.array(recharge_rate[0]), affine= recharge_rate[0].rio.transform(), geojson_out=True, interpolate='nearest')
recharge_rate_values = np.array([int(recharge_rate_points[site]['properties']['value']) for site in range(0,len(boreholes))])

boolean_filter_recharge_rate = [recharge_rate_values == value for value in np.unique(recharge_rate_values)]
boolean_filter_recharge_rate_long = [(recharge_rate_values == value) & resample_boolean for value in np.unique(recharge_rate_values)]
recharge_rate_indices  = [np.where(boolean_filter)[0] for boolean_filter in boolean_filter_recharge_rate]
recharge_rate_indices_long  = [np.where(boolean_filter)[0] for boolean_filter in boolean_filter_recharge_rate_long]

#Resampled monthly-interpolated RECHARGE RATE filtered - WATER LEVEL
resample_monthly_interp_recharge_filtered = [[resample_monthly_interp[i] for i in recharge_rate_index] for recharge_rate_index in recharge_rate_indices]
resample_monthly_interp_long_recharge_filtered = [[resample_monthly_interp[i] for i in recharge_rate_index] for recharge_rate_index in recharge_rate_indices_long]
#Resampled monthly-interpolated RECHARGE RATE filtered - ANOMALIES
resample_monthly_interp_anom_recharge_filtered = [[(df - df.mean())/df.std() for df in df_recharge_rate_filtered] for df_recharge_rate_filtered in resample_monthly_interp_recharge_filtered]
resample_monthly_interp_long_anom_recharge_filtered = [[(df - df.mean())/df.std()  for df in df_recharge_rate_filtered] for df_recharge_rate_filtered in resample_monthly_interp_long_recharge_filtered]
#Borehole Sites names -- filtered for RECHARGE RATE
waterlevel_sites_recharge_filtered = [[waterlevel_sites[i] for i in recharge_rate_index] for recharge_rate_index in recharge_rate_indices]
waterlevel_sites_long_recharge_filtered = [[waterlevel_sites[i] for i in recharge_rate_index] for recharge_rate_index in recharge_rate_indices_long]


#In-situ
concat_interp_long_anom = [pd.concat(anom,axis=1) for anom in resample_monthly_interp_long_anom_recharge_filtered]

#Monthly CLM-GW ANOMALIES
monthly = [rs.point_query(boreholes, np.array(variables[0][i]), affine= variables[0][i].rio.transform(), geojson_out=True, interpolate='nearest') for i in range(0,len(variables[0]))]
time_series_by_month = np.array([ [monthly[month][site]['properties']['value'] for site in range(0,len(boreholes))] for month in range(0,len(monthly))])
time_series_by_site = time_series_by_month.T

time_series_by_site_long_lulc_filtered = [[time_series_by_site[i] for i in lulc_index] for lulc_index in lulc_indices_long]
time_series_by_site_long_anom_lulc_filtered_clmgw = [[(df - np.mean(df))/np.std(df,ddof=1) for df in time_series]for time_series in time_series_by_site_long_lulc_filtered]


for concat,anom,lulc,anom_situ in zip(concat_interp_long_anom,time_series_by_site_long_anom_lulc_filtered_clmgw,['Forested','Cropland','Built Area','Rangeland'],resample_monthly_interp_long_anom_recharge_filtered):
    fig,ax1 = plt.subplots()
    for i in anom_situ:
        plt.rc('font', size = 14)
        ax1.plot(i.index,i.level,color='gray',linewidth=0.3)
        ax1.set_ylabel('Anomaly')
        ax1.set_xlabel('Date')
        ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
        ax1.set_ylim([-5,5])
        ax1.set_title('{}'.format(lulc))
        #ax1.invert_yaxis()    
    #ax2 = ax1.twinx()
    ax1.plot(concat.index,concat.level.mean(axis=1),color='black',linewidth=2)
    #(concat.mean(axis=1)+concat.std(axis=1)).plot(ax=ax1,linewidth=1,color='C0')
    #(concat.mean(axis=1)-concat.std(axis=1)).plot(ax=ax1,linewidth=1,color='C0')
    ax1.plot(variables[v].time,np.mean(anom,axis=0),color='C0')
    ax1.set_ylabel('Anomaly')
    #ax2.set_ylabel('CLM GW Anomaly')
    ax1.set_xlabel('Date')
    ax1.set_ylim([-5,5])
    #ax2.set_ylim([-3,3])
    ax1.set_xlim([datetime(2002, 12, 31), datetime(2023, 1, 8)])
    ax1.set_title('{}'.format(lulc))
    plt.axhline(0,color='black',linewidth=0.4)


#RECHARGE RATE FILTERED
#ALL RECORD
for level,rate in zip(resample_monthly_interp_recharge_filtered,['Major 2-20 mm/yr','Complex <20 mm/yr', 'Complex 20-100 mm/yr', 'Local/Shallow <100 mm/yr']):
    fig,ax1 = plt.subplots()
    for i in level:
        plt.xticks(rotation=45)
        ax1.plot(i.index,i.level,color='C0',linewidth=0.5)
        ax1.set_ylabel('Water Level')
        ax1.set_xlabel('Date')
        ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
        ax1.set_ylim([-100,1])
        ax1.set_title('{}'.format(rate))

for anom,rate in zip(resample_monthly_interp_anom_recharge_filtered,['Major 2-20 mm/yr','Complex <20 mm/yr', 'Complex 20-100 mm/yr', 'Local/Shallow <100 mm/yr']):
    fig,ax1 = plt.subplots()
    for i in anom:
        plt.xticks(rotation=45)
        ax1.plot(i.index,i.level,color='C0',linewidth=0.5)
        ax1.set_ylabel('Water Level Anomaly')
        ax1.set_xlabel('Date')
        ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
        ax1.set_ylim([-10,10])
        ax1.set_title('{}'.format(rate))

#LONG RECORDS ONLY
for level,rate in zip(resample_monthly_interp_long_recharge_filtered,['Major 2-20 mm/yr','Complex <20 mm/yr', 'Complex 20-100 mm/yr', 'Local/Shallow <100 mm/yr']):
    fig,ax1 = plt.subplots()
    for i in level:
        plt.xticks(rotation=45)
        ax1.plot(i.index,i.level,color='C0',linewidth=0.5)
        ax1.set_ylabel('Water Level')
        ax1.set_xlabel('Date')
        ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
        ax1.set_ylim([-100,1])
        ax1.set_title('{}'.format(rate))

for anom,rate in zip(resample_monthly_interp_long_anom_recharge_filtered,['Major 2-20 mm/yr','Complex <20 mm/yr', 'Complex 20-100 mm/yr', 'Local/Shallow <100 mm/yr']):
    fig,ax1 = plt.subplots()
    for i in anom:
        plt.xticks(rotation=45)
        ax1.plot(i.index,i.level,color='C0',linewidth=0.5)
        ax1.set_ylabel('Water Level Anomaly')
        ax1.set_xlabel('Date')
        ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
        ax1.set_ylim([-10,10])
        ax1.set_title('{}'.format(rate))


#Concatenate by Time (monthly)
concat_interp = [pd.concat(level,axis=1) for level in resample_monthly_interp_recharge_filtered]
concat_interp_anom = [pd.concat(anom,axis=1) for anom in resample_monthly_interp_anom_recharge_filtered]
concat_interp_long = [pd.concat(level,axis=1) for level in resample_monthly_interp_long_recharge_filtered]
concat_interp_long_anom = [pd.concat(anom,axis=1) for anom in resample_monthly_interp_long_anom_recharge_filtered]

for concat,anom,recharge,save_name in zip(concat_interp_long_anom,resample_monthly_interp_long_anom_recharge_filtered,['Major 2-20 mm/yr','Complex <20 mm/yr', 'Complex 20-100 mm/yr', 'Local/Shallow <100 mm/yr'],['major','complex_less20','complex_20_100','shallow']):
    fig,ax1 = plt.subplots()
    for i in anom:
        plt.xticks(rotation=45)
        ax1.plot(i.index,i.level,color='gray',linewidth=0.5)
        ax1.set_ylabel('Anomaly')
        ax1.set_xlabel('Date')
        ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
        ax1.set_ylim([-5,5])
        ax1.set_title('{}'.format(recharge))
        #ax1.invert_yaxis()
    concat.mean(axis=1).plot(ax=ax1,linewidth=2,color='black')
    #(concat.mean(axis=1)+concat.std(axis=1)).plot(ax=ax1,linewidth=2)
    #(concat.mean(axis=1)-concat.std(axis=1)).plot(ax=ax1,linewidth=2)
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\FinalFigures\Plots\boreholes_by_recharge_{}.pdf'.format(save_name))

#####################################################################################################################################################
#####################################################################################################################################################
#FILTER FOR LAND USE LAND COVER
path_class = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\Sentinel_LULC\classification.csv'
classes = pd.read_csv(path_class)
water_class = classes['Raster'][classes['Class']=='Water'].iloc[0] #1
tree_class = classes['Raster'][classes['Class']=='Trees'].iloc[0] #2
range_class = classes['Raster'][classes['Class']=='Rangeland'].iloc[0] #11
crops_class = classes['Raster'][classes['Class']=='Crops'].iloc[0] #5
urban_class = classes['Raster'][classes['Class']=='Built Area'].iloc[0] #7
bare_class = classes['Raster'][classes['Class']=='Bare Ground'].iloc[0] #8

path = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\Sentinel_LULC'
raster_files = sorted(glob.glob(path+"/*.tif"))

#####################
#Select LULC year

lulc_year = 0 #2017
#lulc_year = 1 #2021

lulc_points = rs.point_query(boreholes, raster_files[lulc_year], geojson_out=True, interpolate='nearest')
lulc_values = np.array([int(lulc_points[site]['properties']['value']) for site in range(0,len(boreholes))]) #2,5,7,11 [1,3,4,8]

boolean_filter_lulc = [lulc_values == value for value in np.unique(lulc_values)]
boolean_filter_lulc_long = [(lulc_values == value) & resample_boolean for value in np.unique(lulc_values)]
lulc_indices  = [np.where(boolean_filter)[0] for boolean_filter in boolean_filter_lulc]
lulc_indices_long  = [np.where(boolean_filter)[0] for boolean_filter in boolean_filter_lulc_long]

#Resampled monthly-interpolated LULC filtered - WATER LEVEL
resample_monthly_interp_lulc_filtered = [[resample_monthly_interp[i] for i in lulc_index] for lulc_index in lulc_indices]
resample_monthly_interp_long_lulc_filtered = [[resample_monthly_interp[i] for i in lulc_index] for lulc_index in lulc_indices_long]
#Resampled monthly-interpolated LULC filtered - ANOMALIES
resample_monthly_interp_anom_lulc_filtered = [[(df - df.mean())/df.std() for df in df_lulc_filtered] for df_lulc_filtered in resample_monthly_interp_lulc_filtered]
resample_monthly_interp_long_anom_lulc_filtered = [[(df - df.mean())/df.std()  for df in df_lulc_filtered] for df_lulc_filtered in resample_monthly_interp_long_lulc_filtered]
#Borehole Site names -- filtered for LULC
waterlevel_sites_lulc_filtered = [[waterlevel_sites[i] for i in lulc_index] for lulc_index in lulc_indices]
waterlevel_sites_long_lulc_filtered = [[waterlevel_sites[i] for i in lulc_index] for lulc_index in lulc_indices_long]


#LULC FILTERED
#ALL RECORD
for level,cover in zip(resample_monthly_interp_lulc_filtered,['Forested','Cropland','Built Area','Rangeland']):
    fig,ax1 = plt.subplots()
    for i in level:
        plt.xticks(rotation=45)
        ax1.plot(i.index,i.level,color='C0',linewidth=0.5)
        ax1.set_ylabel('Water Level')
        ax1.set_xlabel('Date')
        ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
        ax1.set_ylim([-100,1])
        ax1.set_title('{}'.format(cover))

for anom,cover in zip(resample_monthly_interp_anom_lulc_filtered,['Forested','Cropland','Built Area','Rangeland']):
    fig,ax1 = plt.subplots()
    for i in anom:
        plt.xticks(rotation=45)
        ax1.plot(i.index,i.level,color='C0',linewidth=0.5)
        ax1.set_ylabel('Water Level Anomaly')
        ax1.set_xlabel('Date')
        ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
        ax1.set_ylim([-10,10])
        ax1.set_title('{}'.format(cover))

#LONG RECORDS ONLY
for level,cover in zip(resample_monthly_interp_long_lulc_filtered,['Forested','Cropland','Built Area','Rangeland']):
    fig,ax1 = plt.subplots()
    for i in level:
        plt.xticks(rotation=45)
        ax1.plot(i.index,i.level,color='C0',linewidth=0.5)
        ax1.set_ylabel('Water Level')
        ax1.set_xlabel('Date')
        ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
        ax1.set_ylim([-100,1])
        ax1.set_title('{}'.format(cover))

for anom,cover in zip(resample_monthly_interp_long_anom_lulc_filtered,['Forested','Cropland','Built Area','Rangeland']):
    fig,ax1 = plt.subplots()
    for i in anom:
        plt.xticks(rotation=45)
        ax1.plot(i.index,i.level,color='C0',linewidth=0.5)
        ax1.set_ylabel('Water Level Anomaly')
        ax1.set_xlabel('Date')
        ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
        ax1.set_ylim([-10,10])
        ax1.set_title('{}'.format(cover))

#Concatenate by Time (monthly)
concat_interp = [pd.concat(level,axis=1) for level in resample_monthly_interp_lulc_filtered]
concat_interp_anom = [pd.concat(anom,axis=1) for anom in resample_monthly_interp_anom_lulc_filtered]
concat_interp_long = [pd.concat(level,axis=1) for level in resample_monthly_interp_long_lulc_filtered]
concat_interp_long_anom = [pd.concat(anom,axis=1) for anom in resample_monthly_interp_long_anom_lulc_filtered]

for concat,anom,lulc in zip(concat_interp_long_anom,resample_monthly_interp_long_anom_lulc_filtered,['Forested','Cropland','Built Area','Rangeland']):
    fig,ax1 = plt.subplots()
    for i in anom:
        plt.xticks(rotation=45)
        ax1.plot(i.index,i.level,color='gray',linewidth=0.5)
        ax1.set_ylabel('Anomaly')
        ax1.set_xlabel('Date')
        ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
        ax1.set_ylim([-5,5])
        ax1.set_title('{}'.format(lulc))
        #ax1.invert_yaxis()
    concat.mean(axis=1).plot(ax=ax1,linewidth=2,color='black')
    #(concat.mean(axis=1)+concat.std(axis=1)).plot(ax=ax1,linewidth=2)
    #(concat.mean(axis=1)-concat.std(axis=1)).plot(ax=ax1,linewidth=2)
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\FinalFigures\Plots\boreholes_by_lulc_{}.pdf'.format(lulc))

#####################################################################################################################################################
#####################################################################################################################################################
#Label classifications
gw_depth_key #1,2,4,5
gw_depth_names = {1:'25 - 50',2:'0 - 7',4:'50 - 100',5:'7 - 25'}
recharge_rate_key #12,22,23,33
recharge_names = {12:'Major (2-10)', 22:'Complex (<20)', 23:'Complex (20-100)', 33:'Shallow/Local (<100)'}
classes #2,5,7,11
lulc_names = {2:'Forest',5:'Cropland',7:'Built Area',11:'Rangeland'}

classifications = [pd.DataFrame(character) for character in [gw_depth_values, recharge_rate_values, lulc_values]]
for i in gw_depth_names:
    classifications[0] = classifications[0].where(classifications[0]!=i, gw_depth_names[i])
for i in recharge_names:
    classifications[1] = classifications[1].where(classifications[1]!=i, recharge_names[i])
for i in lulc_names:
    classifications[2] = classifications[2].where(classifications[2]!=i, lulc_names[i])
classifications = pd.concat(classifications,axis=1)
classifications.columns = ['gwdepth','recharge','lulc']

boreholes
labeled_boreholes = pd.concat([boreholes,classifications.set_index(boreholes.index)],axis=1)
filtered_labeled_boreholes = labeled_boreholes.iloc[boolean_index]
filtered_labeled_boreholes.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\validation_points\borehole_long_record_update.csv')
#####################################################################################################################################################
#####################################################################################################################################################






#FILTERED FOR Elevation & Slope
#Elevation Data
path = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\SRTM'
files = sorted(glob.glob(path+"/*.tif"))
dem_file = files[-2]
slope_file = files[-1]

shpname = r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
limpopo = gpd.read_file(shpname)


#####################
#Elevation
elevation_points = rs.point_query(boreholes, dem_file, geojson_out=True, interpolate='nearest')
elevation_values = np.array([int(elevation_points[site]['properties']['value']) for site in range(0,len(boreholes))])

boolean_filter_elev = [elevation_values == value for value in np.unique(elevation_values)]







boolean_filter_gw_depth_long = [(gw_depth_values == value) & resample_boolean for value in np.unique(gw_depth_values)]
gw_depth_indices  = [np.where(boolean_filter)[0] for boolean_filter in boolean_filter_gw_depth]
gw_depth_indices_long  = [np.where(boolean_filter)[0] for boolean_filter in boolean_filter_gw_depth_long]

#Resampled monthly-interpolated GW_DEPTH filtered - WATER LEVEL
resample_monthly_interp_gwdepth_filtered = [[resample_monthly_interp[i] for i in gw_depth_index] for gw_depth_index in gw_depth_indices]
resample_monthly_interp_long_gwdepth_filtered = [[resample_monthly_interp[i] for i in gw_depth_index] for gw_depth_index in gw_depth_indices_long]
#Resampled monthly-interpolated GW_DEPTH filtered - ANOMALIES
resample_monthly_interp_anom_gwdepth_filtered = [[(df - df.mean())/df.std() for df in df_gw_depth_filtered] for df_gw_depth_filtered in resample_monthly_interp_gwdepth_filtered]
resample_monthly_interp_long_anom_gwdepth_filtered = [[(df - df.mean())/df.std()  for df in df_gw_depth_filtered] for df_gw_depth_filtered in resample_monthly_interp_long_gwdepth_filtered]
#Borehole Site names -- filtered for GW_DEPTH
waterlevel_sites_gwdepth_filtered = [[waterlevel_sites[i] for i in gw_depth_index] for gw_depth_index in gw_depth_indices]
waterlevel_sites_long_gwdepth_filtered = [[waterlevel_sites[i] for i in gw_depth_index] for gw_depth_index in gw_depth_indices_long]


#####################
#Slope
slope_points = rs.point_query(boreholes, slope_file, geojson_out=True, interpolate='nearest')
slope_values = np.array([int(slope_points[site]['properties']['value']) for site in range(0,len(boreholes))])

boolean_filter_slope = [slope_values == value for value in np.unique(slope_values)]





#Average all anomalies // water levels --
# for all & long
# for filtered: gw_depth, recharge_rate, LULC
boreholes_long = boreholes[resample_boolean]

###################################################
###################################################
#Compare sampled data with in-situ data for only long-term records (anomalies)
#CLSM
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_CLSM'
files = sorted(glob.glob(path+'\*.nc'))
#GW
clm_gw = xr.open_mfdataset(files[0],parallel=True).mm
#SOIL MOISTURE
clm_sm_rz = xr.open_mfdataset(files[3],parallel=True)['kgm-2']
clm_sm_surface = xr.open_mfdataset(files[4],parallel=True)['kgm-2']


###################################################
#SMAP
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\SMAP_SM_1km\south_africa_monthly\south_africa_monthly'
files = sorted(glob.glob(path+'\*.nc'))
smap = xr.open_mfdataset(files[0],parallel=True,chunks={"x": 100,"y":100}).SM_vwc
smap_dates = pd.date_range('2015-04','2021-01',  freq='1M') 

############################################
#NDVI & LST
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_NDVI'
files = sorted(glob.glob(path+'\*.nc'))
ndvi = xr.open_mfdataset(files[0],parallel=True,chunks={"y": 100,"x":100})

path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_LST'
files = sorted(glob.glob(path+'\*.nc'))
lst = xr.open_mfdataset(files[0],parallel=True,chunks={"y": 100,"x":100})

############################################
#PRECIPITATIION

path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\PRECIPITATION'
files = sorted(glob.glob(path+'\*.nc'))
p_gpm = xr.open_mfdataset(files[1],parallel=True,chunks={"y": 100,"x":100})
p_chirps = xr.open_mfdataset(files[0],parallel=True,chunks={"y": 100,"x":100})


#################################################
#EVAPOTRANSPIRATION

path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_ET'
files = sorted(glob.glob(path+'\*.nc'))
et_modis = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 500,"lon":500})


variables = [clm_gw,clm_sm_rz,clm_sm_surface,smap,ndvi.NDVI,lst.LST_K,p_gpm.P_mm,p_chirps.P_mm,et_modis.ET_kg_m2]
var_names = ['CLM GW','CLM RZ SM','CLM Surface SM','SMAP','NDVI','LST','GPM','CHIRPS','ET']



borehole_csv_data = pd.read_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\sophia\Limpopo In-Situ Data\Boreholes\LP\LPBOREHOLEINFO.csv')

#################################################
######################
#CLM GW // SM RZ - GW

#In-situ
resample_monthly_interp_long_anom = [(df - df.mean())/df.std() for df in resample_monthly_interp_long]
concat_interp_long_anom = pd.concat(resample_monthly_interp_long_anom,axis=1)

#CLM GW
v = 0 #CLM GW
monthly = [rs.point_query(boreholes_long, np.array(variables[v][i]), affine= variables[v][i].rio.transform(), geojson_out=True, interpolate='nearest') for i in range(0,len(variables[v]))]
time_series_by_month = np.array([ [monthly[month][site]['properties']['value'] for site in range(0,len(boreholes_long))] for month in range(0,len(monthly))])
time_series_by_site = time_series_by_month.T
time_series_by_site_anomalies_clmgw = [(df - np.mean(df))/np.std(df,ddof=1) for df in time_series_by_site]

v = 1 #CLM SM RZ
monthly = [rs.point_query(boreholes_long, np.array(variables[v][i]), affine= variables[v][i].rio.transform(), geojson_out=True, interpolate='nearest') for i in range(0,len(variables[v]))]
time_series_by_month = np.array([ [monthly[month][site]['properties']['value'] for site in range(0,len(boreholes_long))] for month in range(0,len(monthly))])
time_series_by_site = time_series_by_month.T
time_series_by_site_anomalies_clmsm_rz = [(df - np.mean(df))/np.std(df,ddof=1) for df in time_series_by_site]

v = 2 #CLM SM
monthly = [rs.point_query(boreholes_long, np.array(variables[v][i]), affine= variables[v][i].rio.transform(), geojson_out=True, interpolate='nearest') for i in range(0,len(variables[v]))]
time_series_by_month = np.array([ [monthly[month][site]['properties']['value'] for site in range(0,len(boreholes_long))] for month in range(0,len(monthly))])
time_series_by_site = time_series_by_month.T
time_series_by_site_anomalies_clmsm = [(df - np.mean(df))/np.std(df,ddof=1) for df in time_series_by_site]

#Averaged for all sites:
plt.rc('font', size = 20)
fig,ax1 = plt.subplots(figsize=(12,7))
#for i,site in zip(range(0,len(resample_monthly_interp_long_anom)),waterlevel_sites):
ax1.plot(concat_interp_long_anom.index,concat_interp_long_anom.mean(axis=1),color='black',label='In-Situ')
ax1.set_ylabel('Anomaly')
#ax2.plot(variables[0].time,time_series_by_site_anomalies[i])
#ax2.set_ylabel('Anomaly')
ax1.set_title('Borehole Groundwater Levels')
ax1.set_ylim(-3,3)
concat_interp_long_anom.mean(axis=1).plot(ax=ax1,linewidth=3,color='black')
(concat_interp_long_anom.mean(axis=1)+1*concat_interp_long_anom.std(axis=1)).plot(ax=ax1,linewidth=1.5,linestyle='--',color='black')
(concat_interp_long_anom.mean(axis=1)-1*concat_interp_long_anom.std(axis=1)).plot(ax=ax1,linewidth=1.5,linestyle='--',color='black')
ax1.plot(variables[v].time,np.mean(time_series_by_site_anomalies_clmgw,axis=0),color='C0',linewidth=3, label='CLSM')
#ax2.plot(variables[v].time,np.mean(time_series_by_site_anomalies_clmsm_rz,axis=0),color='C2',label='Root Zone Soil Moisture')
#ax2.plot(variables[v].time,np.mean(time_series_by_site_anomalies_clmsm,axis=0),color='C1',label='Surface Soil Moisture')
#ax2.plot(variables[-5].time,np.mean(time_series_by_site_anomalies_ndvi,axis=0),color='C2')
#ax1.legend()
plt.axhline(0,color='black',linewidth=0.4)
ax1.set_xlim([datetime(2002, 12, 31), datetime(2023, 1, 8)])
plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\FinalFigures\Plots\boreholes.pdf')

'''
#For each site:
for i in range(0,len(boreholes_long)):
    fig,ax1 = plt.subplots()
    #for i,site in zip(range(0,len(resample_monthly_interp_long_anom)),waterlevel_sites):
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax1.plot(concat_interp_long_anom.index,concat_interp_long_anom.iloc[:,i],color='C1')
    ax1.set_ylabel('Water Level Anomaly')
    #ax2.plot(variables[0].time,time_series_by_site_anomalies[i])
    ax2.set_ylabel('Anomaly')
    ax1.set_xlabel('Date')
    ax1.set_title('{}'.format(var_names[v]))
    ax1.set_ylim(-3,3)
    ax2.set_ylim(-3,3)
    ax2.plot(variables[v].time,(time_series_by_site_anomalies_clmgw[i]),color='C0')
    ax2.legend()
'''
import statsmodels.api as smapi
def linear_plot(independent,dependent,ind_label,d_label):

    y = np.array(dependent)
    x = np.array(independent)
    X = smapi.add_constant(x)

    est = smapi.OLS(y, X)
    model = est.fit()

    plt.rc('font', size = 16)
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot()
    ax.scatter(independent,dependent,color='0')
    ax.plot([], [], ' ', label='R{} = {}'.format(get_super('2'),round(model.rsquared, 3)))
    ax.legend(loc='upper right')
    ax.set_xlabel(ind_label)
    ax.set_ylabel(d_label)
    ax.set_ylim(-3,3)
    ax.set_xlim(-3,3)

variables[v].time[23:-19]

linear_plot(np.mean(time_series_by_site_anomalies_clmgw,axis=0)[23:-19],concat_interp_long_anom.mean(axis=1),'clsm','in-situ')

######################
## CLM - GW by different LULC (2017 reference)
boolean_filter_lulc_long = [(lulc_values == value) & resample_boolean for value in np.unique(lulc_values)]
lulc_indices_long  = [np.where(boolean_filter)[0] for boolean_filter in boolean_filter_lulc_long]

#In-situ
concat_interp_long_anom = [pd.concat(anom,axis=1) for anom in resample_monthly_interp_long_anom_lulc_filtered]

#Monthly CLM-GW ANOMALIES
monthly = [rs.point_query(boreholes, np.array(variables[0][i]), affine= variables[0][i].rio.transform(), geojson_out=True, interpolate='nearest') for i in range(0,len(variables[0]))]
time_series_by_month = np.array([ [monthly[month][site]['properties']['value'] for site in range(0,len(boreholes))] for month in range(0,len(monthly))])
time_series_by_site = time_series_by_month.T

time_series_by_site_long_lulc_filtered = [[time_series_by_site[i] for i in lulc_index] for lulc_index in lulc_indices_long]
time_series_by_site_long_anom_lulc_filtered_clmgw = [[(df - np.mean(df))/np.std(df,ddof=1) for df in time_series]for time_series in time_series_by_site_long_lulc_filtered]


for concat,anom,lulc,anom_situ in zip(concat_interp_long_anom,time_series_by_site_long_anom_lulc_filtered_clmgw,['Forested','Cropland','Built Area','Rangeland'],resample_monthly_interp_long_anom_lulc_filtered):
    fig,ax1 = plt.subplots()
    for i in anom_situ:
        plt.rc('font', size = 14)
        ax1.plot(i.index,i.level,color='gray',linewidth=0.3)
        ax1.set_ylabel('Anomaly')
        ax1.set_xlabel('Date')
        ax1.set_xlim([datetime(2003, 6,30), datetime(2022, 1,1)])
        ax1.set_ylim([-5,5])
        ax1.set_title('{}'.format(lulc))
        #ax1.invert_yaxis()    
    #ax2 = ax1.twinx()
    ax1.plot(concat.index,concat.level.mean(axis=1),color='black',linewidth=2)
    #(concat.mean(axis=1)+concat.std(axis=1)).plot(ax=ax1,linewidth=1,color='C0')
    #(concat.mean(axis=1)-concat.std(axis=1)).plot(ax=ax1,linewidth=1,color='C0')
    ax1.plot(variables[v].time,np.mean(anom,axis=0),color='C0')
    ax1.set_ylabel('Anomaly')
    #ax2.set_ylabel('CLM GW Anomaly')
    ax1.set_xlabel('Date')
    ax1.set_ylim([-5,5])
    #ax2.set_ylim([-3,3])
    ax1.set_xlim([datetime(2002, 12, 31), datetime(2023, 1, 8)])
    ax1.set_title('{}'.format(lulc))
    plt.axhline(0,color='black',linewidth=0.4)
    


##########################################
############

#MODIS NDVI
v = -5 #NDVI

#In-situ
resample_monthly_interp_long_anom = [(df - df.mean())/df.std() for df in resample_monthly_interp_long]
concat_interp_long_anom = pd.concat(resample_monthly_interp_long_anom,axis=1)

monthly = [rs.point_query(boreholes_long, np.array(variables[v][i]), affine= variables[v][i].rio.transform(), geojson_out=True, interpolate='nearest') for i in range(0,len(variables[v]))]
time_series_by_month = np.array([ [monthly[month][site]['properties']['value'] for site in range(0,len(boreholes_long))] for month in range(0,len(monthly))])
time_series_by_site = time_series_by_month.T
time_series_by_site_anomalies_ndvi = [(df - np.mean(df))/np.std(df,ddof=1) for df in time_series_by_site]

fig,ax1 = plt.subplots()
#for i,site in zip(range(0,len(resample_monthly_interp_long_anom)),waterlevel_sites):
ax2 = ax1.twinx()
plt.xticks(rotation=45)
ax1.plot(concat_interp_long_anom.index,concat_interp_long_anom.mean(axis=1),color='C1')
ax1.set_ylabel('Water Level Anomaly')
#ax2.plot(variables[0].time,time_series_by_site_anomalies[i])
ax2.set_ylabel('NDVI Anomaly')
ax1.set_xlabel('Date')
ax1.set_title('{}'.format(var_names[v]))
ax1.set_ylim(-5,5)
ax2.set_ylim(-5,5)

concat_interp_long_anom.mean(axis=1).plot(ax=ax1,linewidth=3,color='black')
(concat_interp_long_anom.mean(axis=1)+1*concat_interp_long_anom.std(axis=1)).plot(ax=ax1,linewidth=1,color='black')
(concat_interp_long_anom.mean(axis=1)-1*concat_interp_long_anom.std(axis=1)).plot(ax=ax1,linewidth=1,color='black')
ax2.plot(variables[v].time,np.mean(time_series_by_site_anomalies_ndvi,axis=0),color='C0')


## NDVI by different LULC (2017 reference)
boolean_filter_lulc_long = [(lulc_values == value) & resample_boolean for value in np.unique(lulc_values)]
lulc_indices_long  = [np.where(boolean_filter)[0] for boolean_filter in boolean_filter_lulc_long]

#In-situ
concat_interp_long_anom = [pd.concat(anom,axis=1) for anom in resample_monthly_interp_long_anom_lulc_filtered]

#Monthly ANOMALIES
monthly = [rs.point_query(boreholes, np.array(variables[v][i]), affine= variables[v][i].rio.transform(), geojson_out=True, interpolate='nearest') for i in range(0,len(variables[v]))]
time_series_by_month = np.array([ [monthly[month][site]['properties']['value'] for site in range(0,len(boreholes))] for month in range(0,len(monthly))])
time_series_by_site = time_series_by_month.T

time_series_by_site_long_lulc_filtered = [[time_series_by_site[i] for i in lulc_index] for lulc_index in lulc_indices_long]
time_series_by_site_long_anom_lulc_filtered_ndvi = [[(df - np.mean(df))/np.std(df,ddof=1) for df in time_series]for time_series in time_series_by_site_long_lulc_filtered]

for concat,anom,lulc in zip(concat_interp_long_anom,time_series_by_site_long_anom_lulc_filtered_ndvi,['Forested','Cropland','Built Area','Rangeland']):
    fig,ax1 = plt.subplots()
    plt.xticks(rotation=45)
    ax2 = ax1.twinx()
    ax1.plot(concat.index,concat.level.mean(axis=1),color='C0',linewidth=2)
    (concat.mean(axis=1)+concat.std(axis=1)).plot(ax=ax1,linewidth=1,color='C0')
    (concat.mean(axis=1)-concat.std(axis=1)).plot(ax=ax1,linewidth=1,color='C0')
    ax2.plot(variables[v].time,np.mean(anom,axis=0),color='C1')
    ax1.set_ylabel('Water Level Anomaly')
    ax2.set_ylabel('MODIS NDVI Anomaly')
    ax1.set_xlabel('Date')
    ax1.set_ylim([-5,5])
    ax2.set_ylim([-5,5])
    ax1.set_title('{}'.format(lulc))



##########################################
############

#IMERG (-3) vs CHIRPS (-2) PPT
v = -2 #Precip GPM

#In-situ
resample_monthly_interp_long_anom = [(df - df.mean())/df.std() for df in resample_monthly_interp_long]
concat_interp_long_anom = pd.concat(resample_monthly_interp_long_anom,axis=1)

monthly = [rs.point_query(boreholes_long, np.array(variables[v][i]), affine= variables[v][i].rio.transform(), geojson_out=True, interpolate='nearest') for i in range(0,len(variables[v]))]
time_series_by_month = np.array([ [monthly[month][site]['properties']['value'] for site in range(0,len(boreholes_long))] for month in range(0,len(monthly))])
time_series_by_site = time_series_by_month.T
time_series_by_site_anomalies_gpm = [(df - np.mean(df))/np.std(df,ddof=1) for df in time_series_by_site]

fig,ax1 = plt.subplots()
#for i,site in zip(range(0,len(resample_monthly_interp_long_anom)),waterlevel_sites):
ax2 = ax1.twinx()
plt.xticks(rotation=45)
ax1.plot(concat_interp_long_anom.index,concat_interp_long_anom.mean(axis=1),color='C1')
ax1.set_ylabel('Water Level Anomaly')
#ax2.plot(variables[0].time,time_series_by_site_anomalies[i])
ax2.set_ylabel('NDVI Anomaly')
ax1.set_xlabel('Date')
ax1.set_title('{}'.format(var_names[v]))
ax1.set_ylim(-5,5)
ax2.set_ylim(-5,5)

concat_interp_long_anom.mean(axis=1).plot(ax=ax1,linewidth=3,color='black')
(concat_interp_long_anom.mean(axis=1)+1*concat_interp_long_anom.std(axis=1)).plot(ax=ax1,linewidth=1,color='black')
(concat_interp_long_anom.mean(axis=1)-1*concat_interp_long_anom.std(axis=1)).plot(ax=ax1,linewidth=1,color='black')
ax2.plot(variables[v].time,np.mean(time_series_by_site_anomalies_gpm,axis=0),color='C0')


## PPT by different LULC (2017 reference)
boolean_filter_lulc_long = [(lulc_values == value) & resample_boolean for value in np.unique(lulc_values)]
lulc_indices_long  = [np.where(boolean_filter)[0] for boolean_filter in boolean_filter_lulc_long]

#In-situ
concat_interp_long_anom = [pd.concat(anom,axis=1) for anom in resample_monthly_interp_long_anom_lulc_filtered]

#Monthly ANOMALIES
monthly = [rs.point_query(boreholes, np.array(variables[v][i]), affine= variables[v][i].rio.transform(), geojson_out=True, interpolate='nearest') for i in range(0,len(variables[v]))]
time_series_by_month = np.array([ [monthly[month][site]['properties']['value'] for site in range(0,len(boreholes))] for month in range(0,len(monthly))])
time_series_by_site = time_series_by_month.T

time_series_by_site_long_lulc_filtered = [[time_series_by_site[i] for i in lulc_index] for lulc_index in lulc_indices_long]
time_series_by_site_long_anom_lulc_filtered_gpm = [[(df - np.mean(df))/np.std(df,ddof=1) for df in time_series]for time_series in time_series_by_site_long_lulc_filtered]

for concat,anom,lulc in zip(concat_interp_long_anom,time_series_by_site_long_anom_lulc_filtered_gpm,['Forested','Cropland','Built Area','Rangeland']):
    fig,ax1 = plt.subplots()
    plt.xticks(rotation=45)
    ax2 = ax1.twinx()
    ax1.plot(concat.index,concat.level.mean(axis=1),color='C0',linewidth=2)
    (concat.mean(axis=1)+concat.std(axis=1)).plot(ax=ax1,linewidth=1,color='C0')
    (concat.mean(axis=1)-concat.std(axis=1)).plot(ax=ax1,linewidth=1,color='C0')
    ax2.plot(variables[v].time,np.mean(anom,axis=0),color='C1')
    ax1.set_ylabel('Water Level Anomaly')
    ax2.set_ylabel('Precipitation Anomaly')
    ax1.set_xlabel('Date')
    ax1.set_ylim([-5,5])
    ax2.set_ylim([-5,5])
    ax1.set_title('{}'.format(lulc))


########################
#######
#LST (-4) // ET (-1) // SMAP (4)

v = 2

#In-situ
resample_monthly_interp_long_anom = [(df - df.mean())/df.std() for df in resample_monthly_interp_long]
concat_interp_long_anom = pd.concat(resample_monthly_interp_long_anom,axis=1)

monthly = [rs.point_query(boreholes_long, np.array(variables[v][i]), affine= variables[v][i].rio.transform(), geojson_out=True, interpolate='nearest') for i in range(0,len(variables[v]))]
time_series_by_month = np.array([ [monthly[month][site]['properties']['value'] for site in range(0,len(boreholes_long))] for month in range(0,len(monthly))])
time_series_by_site = time_series_by_month.T
time_series_by_site_anomalies = [(df - np.mean(df))/np.std(df,ddof=1) for df in time_series_by_site]

fig,ax1 = plt.subplots()
#for i,site in zip(range(0,len(resample_monthly_interp_long_anom)),waterlevel_sites):
ax2 = ax1.twinx()
plt.xticks(rotation=45)
ax1.plot(concat_interp_long_anom.index,concat_interp_long_anom.mean(axis=1),color='C1')
ax1.set_ylabel('Water Level Anomaly')
#ax2.plot(variables[0].time,time_series_by_site_anomalies[i])
ax2.set_ylabel('{} Anomaly'.format(var_names[v]))
ax1.set_xlabel('Date')
ax1.set_title('{}'.format(var_names[v]))
ax1.set_ylim(-5,5)
ax2.set_ylim(-5,5)
#ax2.set_ylim(-5,5)
concat_interp_long_anom.mean(axis=1).plot(ax=ax1,linewidth=3,color='black')
(concat_interp_long_anom.mean(axis=1)+1*concat_interp_long_anom.std(axis=1)).plot(ax=ax1,linewidth=1,color='black')
(concat_interp_long_anom.mean(axis=1)-1*concat_interp_long_anom.std(axis=1)).plot(ax=ax1,linewidth=1,color='black')
ax2.plot(variables[v].time,np.nanmean(time_series_by_site_anomalies,axis=0),color='C0')


## by different LULC (2017 reference)
boolean_filter_lulc_long = [(lulc_values == value) & resample_boolean for value in np.unique(lulc_values)]
lulc_indices_long  = [np.where(boolean_filter)[0] for boolean_filter in boolean_filter_lulc_long]

#In-situ
concat_interp_long_anom = [pd.concat(anom,axis=1) for anom in resample_monthly_interp_long_anom_lulc_filtered]

#Monthly ANOMALIES
monthly = [rs.point_query(boreholes, np.array(variables[v][i]), affine= variables[v][i].rio.transform(), geojson_out=True, interpolate='nearest') for i in range(0,len(variables[v]))]
time_series_by_month = np.array([ [monthly[month][site]['properties']['value'] for site in range(0,len(boreholes))] for month in range(0,len(monthly))])
time_series_by_site = time_series_by_month.T

time_series_by_site_long_lulc_filtered = [[time_series_by_site[i] for i in lulc_index] for lulc_index in lulc_indices_long]
time_series_by_site_long_anom_lulc_filtered_gpm = [[(df - np.mean(df))/np.std(df,ddof=1) for df in time_series]for time_series in time_series_by_site_long_lulc_filtered]

for concat,anom,lulc in zip(concat_interp_long_anom,time_series_by_site_long_anom_lulc_filtered_gpm,['Forested','Cropland','Built Area','Rangeland']):
    fig,ax1 = plt.subplots()
    plt.xticks(rotation=45)
    ax2 = ax1.twinx()
    ax1.plot(concat.index,concat.level.mean(axis=1),color='C0',linewidth=2)
    (concat.mean(axis=1)+concat.std(axis=1)).plot(ax=ax1,linewidth=1,color='C0')
    (concat.mean(axis=1)-concat.std(axis=1)).plot(ax=ax1,linewidth=1,color='C0')
    ax2.plot(variables[v].time,np.mean(anom,axis=0),color='C1')
    ax1.set_ylabel('Water Level Anomaly')
    ax2.set_ylabel('{} Anomaly'.format(var_names[v]))
    ax1.set_xlabel('Date')
    ax1.set_ylim([-5,5])
    ax2.set_ylim([-5,5])
    ax1.set_title('{}'.format(lulc))




# Get X and Y coordinates of points
x_gw = boreholes["geometry"].x
y_gw = boreholes["geometry"].y

# Create list of XY coordinate pairs
coords_gw = [list(xy) for xy in zip(x_gw, y_gw)]
min_x_ws, min_y_ws, max_x_ws, max_y_ws = limpopo.total_bounds



#Kriging Tutorial from: -- not necessary for purposes of this paper
#https://guillaumeattard.com/geostatistics-applied-to-hydrogeology-with-scikit-gstat/
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# We import the folium library to make interactive maps
import folium

# We need to import branca.colormap to give pretty colors to our points according 
# to groundwater table elevation
import branca.colormap as cm

# We need requests to get our dataset and zipfile to unzip our dataset
import requests
import zipfile


url = "https://github.com/guiattard/geoscience-notebooks/raw/master/geostatistics-applied-to-hydrogeology-with-scikit-gstat/hydraulic-head-lyon-sample.zip"
file = "hydraulic-head-lyon-sample.zip"
# Command to donwload the file at the given url
r = requests.get(url)

# Then we open the file
open(file, 'wb').write(r.content)

# We extract the content of the .zip file
with zipfile.ZipFile(file, 'r') as unzip: unzip.extractall("./dat")

# we finally read the shapefile and make some cleaning
gdf = gpd.read_file("./dat/hydraulic-head-sample-lyon.shp")

# We rename the hydraulic name column by hh
gdf = gdf.rename(columns = {'hydraulic_' : "hh"})
gdf.head()

# The url where we can fide the shapefile
url_dep = "http://osm13.openstreetmap.fr/~cquest/openfla/export/departements-20180101-shp.zip"

# The name of the zip file
file_dep = "departements-20180101-shp.zip"

# Command to donwload the file at the given url
r = requests.get(url_dep)

# Then we open the file
open(file_dep, 'wb').write(r.content)

# We extract the content of the .zip file
with zipfile.ZipFile(file_dep, 'r') as unzip: unzip.extractall("dep.shp")

# we finally read the shapefile and make some cleaning
dep = gpd.read_file("dep.shp")

# We remove the zipfile
os.remove(file_dep)

# we print the head of our geodataframe
dep.head()

# We plot the result
fig, ax = plt.subplots(figsize=(10,8))

# We add the location of borehole with hydraulic head
gdf.plot(ax = ax, cmap = "viridis_r",
            column = "hh",
            markersize=10, 
            legend = True,
            legend_kwds={'label': "hydraulic head [m a.s.l]"})

plt.show()