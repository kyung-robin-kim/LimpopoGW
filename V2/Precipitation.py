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
import skimage
from pydap.client import open_url
import netCDF4
from itertools import chain
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import re
import requests
import time
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import rasterstats as rs
import gc
from rasterio.enums import Resampling
from datetime import datetime
import statsmodels as sm 
import statsmodels.graphics.tsaplots as tsaplots
import statsmodels.api as smapi
from sklearn.linear_model import LinearRegression
from scipy import stats
from shapely.geometry import Point


def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)


def plot_np(array,vmin,vmax,title):
    array = np.where(array==0,np.nan,array)
    fig1, ax1 = plt.subplots(figsize=(20,16))
    image = ax1.imshow(array,cmap='RdBu',vmin=vmin,vmax=vmax)
    cbar = fig1.colorbar(image,ax=ax1)
    ax1.set_title('{}'.format(title))


############################################
#MONTHLY Spatial Statistics

path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\PRECIPITATION'
files = sorted(glob.glob(path+'\*.nc'))

p_gpm = xr.open_mfdataset(files[1],parallel=True,chunks={"lat": 100,"lon":100})
p_chirps = xr.open_mfdataset(files[0],parallel=True,chunks={"latitude": 100,"longitude":100})

dates = p_chirps.time #dates.dt.strftime('%Y-%m')

mean_gpm = p_gpm.P_mm.mean(dim=['x','y']).to_dataframe()
mean_chirps = p_chirps.P_mm.mean(dim=['x','y']).to_dataframe()

sum_gpm = p_gpm.P_mm.sum(dim=['x','y']).to_dataframe()
sum_chirps = p_chirps.P_mm.sum(dim=['x','y']).to_dataframe()


#Mean
fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(dates,p_gpm.P_mm.mean(dim=['x','y']),color='C1')
ax.plot(dates,p_chirps.P_mm.mean(dim=['x','y']),color='C0')
ax.set_ylabel('Precipitation (mm)',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)

#################
#Sophia's InSitu Rainfall Data
filepath = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\sophia\Limpopo In-Situ Data\Precipitation'
xl_files = sorted(glob.glob(filepath+'//*.xlsx'))
csv_files = sorted(glob.glob(filepath+'//*.csv'))

#Select which stations have data AND are in LRB
station_id = pd.read_excel(xl_files[0], sheet_name='Stationdetails')
precip_data = pd.read_excel(xl_files[0], sheet_name='_2000_2021Rainfall')
station_codes = sorted(precip_data['STATION'].unique()) #Not all are in Limpopo River Basin

filtered_station_id = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\validation_points\precip_sites_filtered.shp'
station_codes_lrb = sorted(gpd.read_file(filtered_station_id)['CODE'])

filter = np.isin(station_codes_lrb,station_codes)
station_codes_lrb = np.array(station_codes_lrb)[filter]


#Extract precip data by station ID and replace ID to datetime
precip_by_station = [precip_data[precip_data.STATION == i] for i in station_codes_lrb]
for i in range(0,len(precip_by_station)):
    precip_by_station[i].index = pd.to_datetime(precip_by_station[i]['DATE'])


#Sum up by month accumulations
precip_by_station_month_sum = [station.groupby(pd.Grouper(freq='M')).sum() for station in precip_by_station]
#For filtering out months with too many NaNs
NAN_count_precip_month_sum = [pd.isna(station['MM']).groupby(pd.Grouper(freq='M')).sum() for station in precip_by_station]

#Filter out data that is too old
last_date_data = [data.index[-1] for data in precip_by_station_month_sum]
valid_sites = [last_date > pd.to_datetime('2001-01') for last_date in last_date_data]
valid_sites = np.where(valid_sites)

precip_monthly_filtered = [precip_by_station_month_sum[i] for i in valid_sites[0]]


#Plot CHIRPS & GPM comparisons with point validations
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\PRECIPITATION'
files = sorted(glob.glob(path+"/*.nc"))

chirps = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 25,"lon":25}).P_mm
gpm = xr.open_mfdataset(files[1],parallel=True,chunks={"lat": 5,"lon":5}).P_mm

precip_points = gpd.read_file(filtered_station_id)
precip_points_filtered = precip_points[precip_points.isin(station_codes_lrb)['CODE']].reset_index()

'''
#Plot timeseries
#for i in range(0,len(precip_by_station)):
    plt.figure()
    precip_by_station[i]['MM'].plot(title=precip_by_station[i]['STATION'][0])
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\sophia\Limpopo In-Situ Data\Precipitation\plots\{}'.format(i))
'''
#################################################################################################
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

'''
#VALIDATION
queries_ppt = [[(find_nearest(raster_array.y,lat)[1],find_nearest(raster_array.x,lon)[1]) for lat,lon in zip(y,x)] for raster_array in [chirps[0],gpm[0]]]


for points,data,source in zip(queries_ppt,[chirps,gpm],['CHIRPS','GPM']):
    all_timeseries = []
    for point,name in zip(points,precip_points_filtered.CODE):
        timeseries = data[:,point[0],point[1]].to_dataframe().iloc[:,-1].rename(name)
        all_timeseries.append(timeseries)

    all_df = pd.concat(all_timeseries,axis=1)
    all_df.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\in_situ\validation\ppt_{}.csv'.format(source))

'''

precip_by_station_month_sum
precip_products = [pd.read_csv(file) for file in sorted(glob.glob(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\in_situ\validation\ppt*.csv'))]
precip_products = [product.set_index(pd.to_datetime(product.time)).iloc[:,1:] for product in precip_products]

merged_ids = [[precip.iloc[:,i].index.intersection(precip_by_station_month_sum[i].index) for i in range(0,len(precip_by_station_month_sum))] for precip in precip_products]

def linear_stats(model,insitu,title):
    try:
        nanmask = ~np.isnan(model) & ~np.isnan(insitu)
        model = model[nanmask]
        insitu = insitu[nanmask]

        slope, intercept, r_value, p_value, std_err = stats.linregress(insitu,model)
        modeled_y = intercept + slope*insitu

        rmse = np.sqrt(np.mean((modeled_y - insitu)**2))
        bias = np.mean(modeled_y - insitu)
        ub_rmse = np.sqrt(np.sum((modeled_y - insitu)**2)/(len(insitu)-1))
        std_err

        metrics = pd.DataFrame([round(r_value,3),round(ub_rmse,4),round(bias,4),round(p_value,4),title]).rename({0:'R',1:'ub-rmse',2:'bias',3:'pval',4:'label'}).T

        return metrics
    
    except ValueError:
        print('error {}'.format(title))


all_metrics = []
all_data = []

for product,id_set,name in zip(precip_products,merged_ids,['CHIRPS','GPM']):

    set_all_metrics = []
    set_all_data = []

    for i in range(0,len(precip_by_station_month_sum)):

        model_set = product.loc[id_set[i]].iloc[:,i]
        insitu_set = (precip_by_station_month_sum[i].loc[id_set[i]].MM)
        set_all_metrics.append(linear_stats(model_set,insitu_set,precip_points_filtered.CODE[i]+'_{}'.format(name)))
        set_all_data.append(pd.DataFrame([model_set,insitu_set]).T.rename(columns={precip_points_filtered.CODE[i]:'model','MM':'insitu'}))

    all_metrics.append(set_all_metrics)
    all_data.append(set_all_data)

all_data_concat_chirps = pd.concat(all_data[0],axis=0)
all_data_concat_gpm = pd.concat(all_data[1],axis=0)

from scipy.stats import gaussian_kde
def linear_plot(model,insitu,title):

    insitu = (all_data_concat_gpm.insitu)[(all_data_concat_gpm.model)>-10]
    model = (all_data_concat_gpm.model)[(all_data_concat_gpm.model)>-10]
    title = 'GPM IMERG'



    try:
        nanmask = ~np.isneginf(model) & ~np.isneginf(insitu)
        model = model[nanmask]
        insitu = insitu[nanmask]

        slope, intercept, r_value, p_value, std_err = stats.linregress(insitu,model)
        modeled_y = intercept + slope*insitu

        rmse = np.sqrt(np.mean((modeled_y - insitu)**2))
        bias = np.mean(modeled_y - insitu)
        ub_rmse = np.sqrt(np.sum((modeled_y - insitu)**2)/(len(insitu)-1))
        std_err

        # Calculate the point density
        xy = np.vstack([insitu,model])
        z = gaussian_kde(xy)(xy)

        fig,ax = plt.subplots()
        ax.scatter(insitu, model, c=z, s=2)
        ax.plot(insitu, intercept + slope*insitu, label='r: {}'.format(round(r_value,3)),color='b')
        ax.set_ylabel('Precipitation Product (mm)', fontsize=13)
        ax.set_xlabel('InSitu Precipitation (mm)', fontsize=13)
        ax.legend()
        
        ax.set_xlim(0,600)
        ax.set_ylim(0,600)
        ax.plot([0,600],[0,600],'--',color='black')
        ax.text(400,140,s="rmse: {}".format(round(rmse,3)), fontsize=11, ha="left", va="top")
        ax.text(400,100,s="bias: {}".format(round(bias,3)), fontsize=11, ha="left", va="top")
        ax.text(400,60,s="p-val < 0.01", fontsize=11, ha="left", va="top") #{}".format(round(p_value,3)), fontsize=11, ha="left", va="top")
        ax.set_title('(a)',weight='bold',fontsize=15)

        #ax.set_xlim(-8,8)
        #ax.set_ylim(-8,8)
        #ax.plot([-8,8],[-8,8],'--',color='black')
        #ax.text(4,-4,s="rmse: {}".format(round(rmse,3)), fontsize=11, ha="left", va="top")
        #ax.text(4,-5,s="bias: {}".format(round(bias,3)), fontsize=11, ha="left", va="top")
        #ax.text(4,-6,s="p-val < 0.01", fontsize=11, ha="left", va="top") #{}".format(round(p_value,3)), fontsize=11, ha="left", va="top")
        #ax.set_title('(b)',weight='bold',fontsize=15)
        plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\{}'.format(title),dpi=300)

        print(title,round(r_value,3))

    except ValueError:
        print('error {}'.format(title))

plt.rcParams["font.family"] = "Times New Roman"
linear_plot(model=np.log(all_data_concat_gpm.model)[np.log(all_data_concat_gpm.model)>-10],insitu=np.log(all_data_concat_gpm.insitu)[np.log(all_data_concat_gpm.model)>-10],title='GPM IMERG log')
linear_plot((all_data_concat_gpm.insitu),(all_data_concat_gpm.model),'GPM IMERG')

linear_plot(np.log(all_data_concat_gpm.insitu)[np.log(all_data_concat_gpm.model)>-10],np.log(all_data_concat_gpm.model)[np.log(all_data_concat_gpm.model)>-10],'GPM IMERG')
linear_plot((all_data_concat_chirps.insitu),(all_data_concat_chirps.model),'CHIRPS')

ii=1
for i in range(0,len(all_data[ii])):
    linear_plot(all_data[ii][i].insitu,all_data[ii][i].model,precip_points_filtered.CODE[i]+' {}'.format(['CHIRPS','GPM'][ii]))




######################################################################################################################################################
######################################################################################################################################################
#################################################################################################
growing_months = [10,11,12,1,2,3]
start_year = 2004
end_year = 2021
from scipy.stats import boxcox
precip_by_station_month_sum


precip_by_station_month_sum
all_dates = precip_by_station_month_sum[-1].index
all_precip = pd.DataFrame(np.empty(len(all_dates))).set_index(all_dates)
all_precip_lambda = []
for i in range(0,15):
    plt.figure()
    precip_points_filtered.CODE[i]
    precip_boxcox = pd.DataFrame(boxcox(precip_by_station_month_sum[i].MM+0.00001)[0]).set_index(precip_by_station_month_sum[i].index)
    precip_lambda = (boxcox(precip_by_station_month_sum[i].MM+0.00001)[1])
    #plt.plot((precip_ln-precip_ln.mean())/precip_ln.std(ddof=1))

    all_precip['{}'.format(precip_points_filtered.CODE[i])] = precip_boxcox
    all_precip_lambda.append(precip_lambda)

all_precip = all_precip.iloc[:,1:]
all_precip.mean(axis=1).plot()





precip_by_station_month_sum
all_dates = precip_by_station_month_sum[-1].index
all_precip = pd.DataFrame(np.empty(len(all_dates))).set_index(all_dates)

for i in range(0,15):
    plt.figure()
    precip_points_filtered.CODE[i]
    precip_ln = np.log(precip_by_station_month_sum[i].MM).where(np.log(precip_by_station_month_sum[i].MM)>-100,np.nan)
    #plt.plot((precip_ln-precip_ln.mean())/precip_ln.std(ddof=1))
    precip_by_station_month_sum[i].MM.plot()
    all_precip['{}'.format(precip_points_filtered.CODE[i])] = precip_boxcox

all_precip = all_precip.iloc[:,1:]
all_precip.mean(axis=1).plot()

#########################################
#Choose Log_transformed OR raw discharge
#Log-transformed
precip_ln_seasonal = all_precip
#Convert to water year for growing season means / year
precip_ln_seasonal['water_year'] = precip_ln_seasonal.index.year.where(precip_ln_seasonal.index.month < 10, precip_ln_seasonal.index.year + 1)


all_precip = all_precip.interpolate('linear')
all_precip = all_precip.iloc[:,[0,1,2,3,4,5,7,8,9,10,11,13,14]]

#OPTION A: No seasonal component -- just normalize for all record (post log-transformation)

runoff_anomaly_mean = pd.DataFrame(((all_precip - all_precip.mean(axis=0))/(all_precip.std(axis=0,ddof=1))).mean(axis=1)).set_index(pd.to_datetime(all_precip.index))
runoff_anomaly_std = pd.DataFrame(((all_precip - all_precip.mean(axis=0))/(all_precip.std(axis=0,ddof=1))).std(axis=1,ddof=1)).set_index(pd.to_datetime(all_precip.index))

plt.plot(-2* runoff_anomaly_std + runoff_anomaly_mean)
plt.plot(2* runoff_anomaly_std+ runoff_anomaly_mean)

#OPTION B: Filter and normalize for Growing Season only
#QUESTIONS: Could pre or post growing season be important?
runoff_level_wet_monthly = precip_ln_seasonal[precip_ln_seasonal.index.month.isin(growing_months)]
runoff_anomaly_wet_monthly = runoff_level_wet_monthly - runoff_level_wet_monthly.mean(axis=0)#/ runoff_level_wet_monthly.std(axis=0,ddof=1)
runoff_anomaly_wet_monthly.iloc[:,:-1].mean(axis=1).plot()
#(-2* runoff_anomaly_wet_monthly.std(axis=1,ddof=1) + runoff_anomaly_wet_monthly.mean(axis=1)).plot()
#(2* runoff_anomaly_wet_monthly.std(axis=1,ddof=1) + runoff_anomaly_wet_monthly.mean(axis=1)).plot()

#OPTION C: Seasonally Averaged Anomalies for Growing season only
runoff_level_wet_monthly = precip_ln_seasonal[precip_ln_seasonal.index.month.isin(growing_months)]
annual_data =  pd.concat([runoff_level_wet_monthly[runoff_level_wet_monthly.water_year== i].iloc[:,:-1].mean(axis=0) for i in range(start_year,end_year+1)],axis=1).T
annual_std = annual_data.std(axis=0,ddof=1)
dataset_anomaly_wet_annual = (annual_data - annual_data.mean(axis=0))/annual_std 
dataset_anomaly_wet_annual_df_r = pd.DataFrame(dataset_anomaly_wet_annual.mean(axis=1)).set_index(pd.date_range(start=str(start_year), end=str(end_year), freq='YS')).rename(columns={0:'Growing Season'})
dataset_anomaly_wet_annual_df_r_std = pd.DataFrame(dataset_anomaly_wet_annual.std(axis=1,ddof=1)).set_index(pd.date_range(start=str(start_year), end=str(end_year), freq='YS')).rename(columns={0:'Growing Season'})




#Seasonal Anomalies (Growing Season)
netcdf_anom_path = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies_V2\netcdfs'
vhi_anom = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies_V2\netcdfs\VHI\VHI_anom.nc'
files_spei = sorted(glob.glob(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies_V2\netcdfs\SPEI\*.nc'))
files_indicators = [vhi_anom,files_spei[1],files_spei[2],files_spei[0]]
indices_datasets = [xr.open_mfdataset(file,parallel=True) for file in files_indicators]
inidices_keys = [list(ds.data_vars)[0] for ds in indices_datasets]
indices_datasets = [var['{}'.format(key)] for var,key in zip(indices_datasets[0:1],inidices_keys[0:1])] + [var['{}'.format(key)].rename({'lat':'y','lon':'x'}).transpose('year_i','y','x') for var,key in zip(indices_datasets[1:],inidices_keys[1:])]
indices_names = ['VHI','SPEI-3','SPEI-12','SPEI-1']
files = sorted(glob.glob(netcdf_anom_path+'\*.nc'))
var_datasets = [xr.open_mfdataset(file,parallel=True) for file in files]
var_keys = [list(ds.data_vars)[0] for ds in var_datasets]
var_datasets = [var['{}'.format(key)] for var,key in zip(var_datasets,var_keys)]
var_names = ['ET', 'GW','LST','NDVI','PPT_CHIRPS','PPT_GPM','RZ','Runoff','SMAP','SM_Surf']

spatial_mean_anomalies_vars_growing = [var.mean(dim={'x','y'}).assign_coords(year_i=pd.date_range(start='2003-12-31',end='2022-12-31',freq='YS')) for var in var_datasets]
spatial_mean_anomalies_indic_growing = [var.mean(dim={'x','y'}).assign_coords(year_i=pd.date_range(start='2003-12-31',end='2022-12-31',freq='YS')) for var in indices_datasets]

plt.figure(figsize=(6,4))
#spatial_mean_anomalies_indic_growing[0].plot(label='VHI')
#spatial_mean_anomalies_indic_growing[1].plot(label='SPEI-3')
#spatial_mean_anomalies_indic_growing[2].plot(label='SPEI-12')
#spatial_mean_anomalies_indic_growing[3].plot(label='SPEI-1')
#spatial_mean_anomalies_vars_growing[0].plot(label='MODIS ET')
#spatial_mean_anomalies_vars_growing[1].plot(label='CLSM GW')


plt.figure()
x = pd.to_datetime(all_precip.index)
#dataset = runoff_ln
#monthly_data = [dataset.loc[dataset.index.month == i] for i in range(1,13)]
#monthly_mean = [dataset.loc[dataset.index.month == i].mean(axis=0) for i in range(1,13)]
#monthly_std = [dataset.loc[dataset.index.month == i].std(axis=0,ddof=1) for i in range(1,13)]
#dataset_month_anomalies = [ (monthly_data[i] - monthly_mean[i])/monthly_std[i] for i in range(0,12)]
#monthly_anomalies_runoff = pd.concat(dataset_month_anomalies).sort_index()
plt.plot(dataset_anomaly_wet_annual_df_r,linewidth=2)
#plt.plot(runoff_anomaly_mean,color='C0',linestyle='--',linewidth=1,label='Monthly')
#plt.plot(-1* dataset_anomaly_wet_annual_df_r_std + dataset_anomaly_wet_annual_df_r,linewidth=0.5,color='C0')
#plt.plot(1* dataset_anomaly_wet_annual_df_r_std+ dataset_anomaly_wet_annual_df_r,linewidth=0.5,color='C0')
#spatial_mean_anomalies_vars_growing[5].plot(label='GPM PPT')
#spatial_mean_anomalies_vars_growing[6].plot(label='CHIRPS PPT')
#plt.plot(runoff_anomaly_mean)
plt.plot(-1* runoff_anomaly_std + runoff_anomaly_mean,linewidth=0.5,color='C0')
plt.plot(1* runoff_anomaly_std+ runoff_anomaly_mean,linewidth=0.5,color='C0')
plt.fill_between(x,(-1* runoff_anomaly_std  + runoff_anomaly_mean).iloc[:,0], (1* runoff_anomaly_std + runoff_anomaly_mean).iloc[:,0], alpha=0.2) 
#monthly_anomalies_runoff.mean(axis=1).plot(linewidth=0.5,label='Monthly (per Month) Anomalies',legend=True)\
plt.axhline(0,color='black',linestyle='--',linewidth=0.5)
plt.ylim(-3,3)
plt.xlabel('Date')
plt.legend()
plt.ylabel('ln(Discharge)  Anomalies')
plt.title('Basin-Averaged In-Situ Gauges')





#OPTION D: Add (not average runoff -- ET -- precip)
#Log-transformed
runoff_ln = (velocity.resample({'time':'M'}).sum())
#Convert to water year for growing season means / year
runoff_ln['water_year'] = runoff_ln.time.dt.year.where(runoff_ln.time.dt.month < 10, runoff_ln.time.dt.year + 1)



runoff = pd.DataFrame(np.log(runoff_ln))
runoff = runoff.where(runoff>-100,np.nan)
runoff_ln_seasonal = np.log(runoff_ln).where(np.log(runoff_ln)>-100,np.nan)

runoff_level_wet_monthly = runoff_ln[runoff_ln.time.dt.month.isin(growing_months)]
annual_data =  xr.concat([runoff_level_wet_monthly[runoff_level_wet_monthly.water_year== i].sum(dim='time') for i in range(start_year,end_year+1)],dim='q_growing')
annual_std = annual_data.std(dim='id',ddof=1)
dataset_anomaly_wet_annual = (annual_data.T - annual_data.mean(dim='q_growing'))/annual_std 
dataset_anomaly_wet_annual_df_r = pd.DataFrame(dataset_anomaly_wet_annual.mean(dim='id')).set_index(pd.date_range(start=str(start_year), end=str(end_year), freq='YS')).rename(columns={0:'Growing Season'})

plt.figure()
x = pd.to_datetime(runoff_ln.time)
#dataset = runoff_ln
#monthly_data = [dataset.loc[dataset.index.month == i] for i in range(1,13)]
#monthly_mean = [dataset.loc[dataset.index.month == i].mean(axis=0) for i in range(1,13)]
#monthly_std = [dataset.loc[dataset.index.month == i].std(axis=0,ddof=1) for i in range(1,13)]
#dataset_month_anomalies = [ (monthly_data[i] - monthly_mean[i])/monthly_std[i] for i in range(0,12)]
#monthly_anomalies_runoff = pd.concat(dataset_month_anomalies).sort_index()
plt.plot(dataset_anomaly_wet_annual_df_r,linewidth=2)
#plt.plot(runoff_anomaly_mean,color='C0',linestyle='--',linewidth=1,label='Monthly')
plt.plot(-1* runoff_anomaly_std + runoff_anomaly_mean,linewidth=0.5,color='C0')
plt.plot(1* runoff_anomaly_std + runoff_anomaly_mean,linewidth=0.5,color='C0')
plt.fill_between(x,(-1* runoff_anomaly_std + runoff_anomaly_mean)[0], (1* runoff_anomaly_std + runoff_anomaly_mean)[0], alpha=0.2) 
#monthly_anomalies_runoff.mean(axis=1).plot(linewidth=0.5,label='Monthly (per Month) Anomalies',legend=True)\
plt.axhline(0,color='black',linestyle='--',linewidth=0.5)
plt.ylim(-3,3)
plt.xlabel('Date')
plt.ylabel('ln(Discharge)  Anomalies')
plt.title('Basin-Averaged In-Situ Gauges')
