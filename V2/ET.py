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

path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_ET'
files = sorted(glob.glob(path+'\*.nc'))

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


#Plot timeseries
#for i in range(0,len(precip_by_station)):
    plt.figure()
    precip_by_station[i]['MM'].plot(title=precip_by_station[i]['STATION'][0])
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\sophia\Limpopo In-Situ Data\Precipitation\plots\{}'.format(i))


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
        ax.set_ylabel('Precipitation Product Ln(mm)', fontsize=13)
        ax.set_xlabel('In-Situ Precipitation Ln(mm)', fontsize=13)
        ax.legend()
        ax.set_xlim(-8,8)
        ax.set_ylim(-8,8)
        ax.plot([-10,600],[-10,600],'--',color='black')
        ax.text(4,-4,s="rmse: {}".format(round(rmse,3)), fontsize=11, ha="left", va="top")
        ax.text(4,-5,s="bias: {}".format(round(bias,3)), fontsize=11, ha="left", va="top")
        ax.text(4,-6,s="p-val: {}".format(round(p_value,3)), fontsize=11, ha="left", va="top")
        ax.set_title('Monthly '+title,weight='bold')

        print(title,round(r_value,3))

    except ValueError:
        print('error {}'.format(title))

plt.rcParams["font.family"] = "Times New Roman"
linear_plot(np.log(all_data_concat_gpm.insitu)[np.log(all_data_concat_gpm.model)>-10],np.log(all_data_concat_gpm.model)[np.log(all_data_concat_gpm.model)>-10],'GPM IMERG')
linear_plot(np.log(all_data_concat_chirps.insitu),np.log(all_data_concat_chirps.model),'CHIRPS')

ii=1
for i in range(0,len(all_data[ii])):
    linear_plot(all_data[ii][i].insitu,all_data[ii][i].model,precip_points_filtered.CODE[i]+' {}'.format(['CHIRPS','GPM'][ii]))
