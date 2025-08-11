import rasterio as rio
import os
import glob
from scipy import special
import math
import matplotlib.pyplot as plt 
import matplotlib
from pathlib import Path
import numpy as np
import seaborn as sns
import xarray as xr
import rioxarray
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from rasterstats import zonal_stats
import geopandas as gpd
import pandas as pd
from scipy import special
from shapely.geometry import box, mapping
from rasterio.enums import Resampling
import gc
import statsmodels.api as smapi
import rasterstats as rs
from datetime import datetime

def read_file(file):
    with rio.open(file) as src:
        return(src.read(1))

degree_sign = u"\N{DEGREE SIGN}"
plt.rcParams["font.family"] = "Times New Roman"

def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

def plot_np(array,vmin,vmax,title):
    array = np.where(array==0,np.nan,array)
    fig1, ax1 = plt.subplots(figsize=(20,16))
    image = ax1.imshow(array,cmap = 'RdBu_r',vmin=vmin,vmax=vmax)
    cbar = fig1.colorbar(image,ax=ax1)
    ax1.set_title('{}'.format(title))

def unique_vals(array,roundoff):
    unique_values = np.unique(np.round(array,roundoff))

    return unique_values

def linear_plot(independent,dependent,ind_label,d_label):

    y = np.array(dependent)
    x = np.array(independent)
    X = smapi.add_constant(x)

    est = smapi.OLS(y, X)
    model = est.fit()

    plt.rc('font', size = 20)
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot()
    ax.scatter(independent,dependent,color='0')
    ax.plot([], [], ' ', label='R2 = {}'.format(round(model.rsquared, 3)))
    ax.legend(loc='upper right')
    ax.set_xlabel(ind_label)
    ax.set_ylabel(d_label)


shpname = r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
limpopo = gpd.read_file(shpname)

path = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\SRTM'
files = sorted(glob.glob(path+"/*.tif"))
dem = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 500,"lon":500}).band_data

path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_NDVI'
files = sorted(glob.glob(path+"/*.nc"))
ndvi = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 100,"lon":100})
ndvi_df = ndvi.mean(['x','y']).to_dataframe()
ndvi_std = ndvi_df.std().NDVI

lat = np.array(ndvi.y)
lon = np.array(ndvi.x)

dem_interp = dem.rio.set_crs("epsg:4326").rio.reproject_match(ndvi.rio.set_crs("epsg:4326"),resampling=Resampling.average)

path = r'C:\Users\robin\Box\Data\Groundwater\GRACE_MASCON_RL06_V2'
files = sorted(glob.glob(path+"/*.nc"))
grace = xr.open_mfdataset(files[1],parallel=True,chunks={"lat": 100,"lon":100}).lwe_thickness.rio.set_spatial_dims('lon','lat',inplace=True).rio.write_crs('WGS84').rio.clip(limpopo.geometry.apply(mapping), limpopo.crs, drop=True,all_touched=True)
grace_uncertainty = xr.open_mfdataset(files[1],parallel=True,chunks={"lat": 100,"lon":100}).uncertainty.rio.set_spatial_dims('lon','lat',inplace=True).rio.write_crs('WGS84').rio.clip(limpopo.geometry.apply(mapping), limpopo.crs, drop=True,all_touched=True)
scale_factor = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 100,"lon":100}).rio.set_spatial_dims('lon','lat',inplace=True).rio.write_crs('WGS84').rio.clip(limpopo.geometry.apply(mapping), limpopo.crs, drop=True,all_touched=True)
land_mask = xr.open_mfdataset(files[3],parallel=True,chunks={"lat": 100,"lon":100}).rio.set_spatial_dims('lon','lat',inplace=True).rio.write_crs('WGS84').rio.clip(limpopo.geometry.apply(mapping), limpopo.crs, drop=True,all_touched=True)
grace_scaled = grace * scale_factor.scale_factor
twsa_da = grace_scaled - grace_scaled.mean(dim='time')
scaled_grace_interp = grace_scaled.rio.reproject_match(ndvi.rio.set_crs("epsg:4326"),resampling=Resampling.nearest)
scaled_grace_interp = scaled_grace_interp.resample(time='1M').mean() #GAP FILL



##################################################################
#Elevation vs. NDVI, ET, LST, PPT, SM?, GW?
def linearmodel_plot_elev(predictor, outcome, labelx, labely,title):

    independent = predictor
    dependent = outcome

    nanmask = ~np.isnan(independent) & ~np.isnan(dependent)
    independent_mask = np.array(independent)[nanmask]
    dependent_mask = np.array(dependent)[nanmask]

    x=np.array(np.array(independent_mask).reshape(-1,1))
    y=np.array(np.array(dependent_mask).reshape(-1,1))

    model=LinearRegression()
    model.fit(x,y)
    print("linear model: y = {:.5}x + {:.5}".format(model.coef_[0][0], model.intercept_[0]))
    bf_line=model.predict(x)

    mod=sm.OLS(y,x)
    fii=mod.fit()
    p_values = fii.summary2().tables[1]['P>|t|']
    r_values = fii.summary2().tables[1]['Coef.']
    print('p= ', p_values, '\nr2=', r_values)

    axis = [(0,0), (2400,0)]
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot()
    ax.plot([axis[0][0],axis[1][0]],[axis[0][1],axis[1][1]],'--',color='black',linewidth=1)
    if int(title[5:7]) in [4,5,6,7,8,9]: #DRY WINTER
        ax.scatter(independent_mask[0::1],dependent_mask[0::1],color='C1',s=5)
    else: #WET SUMMER
        ax.scatter(independent_mask[0::1],dependent_mask[0::1],color='C0',s=5)
    ax.plot(independent_mask[0::1],bf_line[0::1], label='y = {:.3}x + {:.3}'.format(model.coef_[0][0], model.intercept_[0]), color='black',linewidth=1)
    ax.set_ylabel(labely,weight='bold',fontsize=14)
    ax.set_xlabel(labelx,weight='bold',fontsize=14)
    ax.set_xlim(0,2500)
    ax.set_ylim(-1,1)
    ax.set_title('{}'.format(title),weight='bold',fontsize=18)  
    ax.legend(loc='lower right',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\elev_ndvi\{}.png'.format(title[0:10]))


[linearmodel_plot_elev(dem_interp[0],ndvi.NDVI[i],'elevation','ndvi monthly','{}'.format(date)) for i,date in zip(range(0,len(ndvi.NDVI)),ndvi_df.index)]

#################################################################

def linearmodel_plot_ndvi(predictor, outcome, labelx, labely,title):

    independent = predictor
    dependent = outcome

    nanmask = ~np.isnan(independent) & ~np.isnan(dependent)
    independent_mask = np.array(independent)[nanmask]
    dependent_mask = np.array(dependent)[nanmask]

    x=np.array(np.array(independent_mask).reshape(-1,1))
    y=np.array(np.array(dependent_mask).reshape(-1,1))

    model=LinearRegression()
    model.fit(x,y)
    print("linear model: y = {:.5}x + {:.5}".format(model.coef_[0][0], model.intercept_[0]))
    bf_line=model.predict(x)

    mod=sm.OLS(y,x)
    fii=mod.fit()
    p_values = fii.summary2().tables[1]['P>|t|']
    r_values = fii.summary2().tables[1]['Coef.']
    print('p= ', p_values, '\nr2=', r_values)

    axis = [(0,0), (1,0)]
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot()
    ax.plot([axis[0][0],axis[1][0]],[axis[0][1],axis[1][1]],'--',color='black',linewidth=1)
    if int(title[5:7]) in [4,5,6,7,8,9]: #DRY WINTER
        ax.scatter(independent_mask[0::1],dependent_mask[0::1],color='C1',s=5)
    else: #WET SUMMER
        ax.scatter(independent_mask[0::1],dependent_mask[0::1],color='C0',s=5)
    ax.plot(independent_mask[0::1],bf_line[0::1], label='y = {:.3}x + {:.3}'.format(model.coef_[0][0], model.intercept_[0]), color='black',linewidth=1)
    ax.set_ylabel(labely,weight='bold',fontsize=14)
    ax.set_xlabel(labelx,weight='bold',fontsize=14)
    ax.set_xlim(-0.5,1)
    ax.set_ylim(-10,12)
    ax.set_title('{}'.format(title),weight='bold',fontsize=18)  
    ax.legend(loc='lower right',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\ndvi_twsa\{}.png'.format(title[0:10]))



#NDVI vs. TWSA
scaled_grace_interp_filtered = scaled_grace_interp[3::]
[linearmodel_plot_ndvi(ndvi.NDVI[2:80][i],scaled_grace_interp_filtered[2:80][i],'ndvi monthly','twsa monthly','{}'.format(date)) for i,date in zip(range(0,len(ndvi.NDVI)),ndvi_df.index[2:80])]



#NDVI vs. Boreholes
#Borehole water levels - 227 sites
filepath = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\sophia\Limpopo In-Situ Data\Boreholes\LP\LP_WaterLevels\*.csv'
csv_files = sorted(list(glob.glob(filepath)))
waterlevel_dfs = [pd.read_csv(file, names=['site', 'date','time','level','?','??'],index_col=False) for file in csv_files]
waterlevel_dates_df =  [np.stack([datetime.strptime(str(date), '%Y%m%d') for date in waterlevel_dfs[i].date]) for i in range(0,len(waterlevel_dfs))]
waterlevel_dfs = [pd.concat([waterlevel_dfs[i]['site'], pd.Series(waterlevel_dates_df[i]).rename('date'), waterlevel_dfs[i]['time'],waterlevel_dfs[i]['level']],axis=1) for i in range(0,len(waterlevel_dfs))] 
waterlevel_sites = [(waterlevel_dfs[i]['site'][0]).strip() for i in range(0,len(waterlevel_dfs)) ]

#Resampled to day-average 
waterlevel_dfs_daily = [waterlevel_dfs[i].groupby('date').mean() for i in range(0,len(waterlevel_dfs))]

#Borehole shapefile - 655 sites (therefore filter out for those with water level data)
shpname = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\sophia\Limpopo In-Situ Data\Boreholes\LP\LP_BOREHOLES.shp'
boreholes = gpd.read_file(shpname).to_crs({'init': 'epsg:4326'})

boreholes_boolean = [boreholes.loc[i].F_STATION in waterlevel_sites for i in range(0,len(boreholes))]
boreholes = boreholes[boreholes_boolean]


monthly = [rs.point_query(boreholes, np.array(ndvi.NDVI[i]), affine= ndvi.NDVI[i].rio.transform(), geojson_out=True) for i in range(0,len(ndvi_df.index))]

time_series_by_month = np.array([ [monthly[month][site]['properties']['value'] for site in range(0,len(boreholes))] for month in range(0,len(monthly))])
time_series_by_site = time_series_by_month.T


for i,site in zip(range(0,len(waterlevel_dfs_daily)),waterlevel_sites):
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xticks(rotation=45)
    ax1.plot(waterlevel_dfs_daily[i].index,waterlevel_dfs_daily[i].level,'*',color='C1')
    ax1.set_ylabel('Water Level')
    ax2.plot(ndvi_df.index,time_series_by_site[i])
    ax2.set_ylim(0,1)
    ax2.set_ylabel('NDVI')
    ax2.set_xlabel('Date')
    ax1.set_title('{}'.format(site))
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\boreholes_NDVI\fixed_water_depth\{}'.format(site))



###########################################################################
#SOIL MOISTURE vs. NDVI vs. dTWS

def linearmodel_plot_sm(predictor, outcome, labelx, labely,title):

    independent = predictor
    dependent = outcome

    nanmask = ~np.isnan(np.array(independent)) & ~np.isnan(np.array(dependent))
    independent_mask = np.array(independent)[nanmask]
    dependent_mask = np.array(dependent)[nanmask]

    x=np.array(np.array(independent_mask).reshape(-1,1))
    y=np.array(np.array(dependent_mask).reshape(-1,1))

    model=LinearRegression()
    model.fit(x,y)
    print("linear model: y = {:.5}x + {:.5}".format(model.coef_[0][0], model.intercept_[0]))
    bf_line=model.predict(x)

    mod=sm.OLS(y,x)
    fii=mod.fit()
    p_values = fii.summary2().tables[1]['P>|t|']
    r_values = fii.summary2().tables[1]['Coef.']
    print('p= ', p_values, '\nr2=', r_values)

    axis = [(0,0), (1,0)]
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot()
    ax.plot([axis[0][0],axis[1][0]],[axis[0][1],axis[1][1]],'--',color='black',linewidth=1)
    if int(title[5:7]) in [4,5,6,7,8,9]: #DRY WINTER
        ax.scatter(independent_mask[0::1],dependent_mask[0::1],color='C1',s=5)
    else: #WET SUMMER
        ax.scatter(independent_mask[0::1],dependent_mask[0::1],color='C0',s=5)
    ax.plot(independent_mask[0::1],bf_line[0::1], label='y = {:.3}x + {:.3}'.format(model.coef_[0][0], model.intercept_[0]), color='black',linewidth=1)
    ax.set_ylabel(labely,weight='bold',fontsize=14)
    ax.set_xlabel(labelx,weight='bold',fontsize=14)
    ax.set_xlim(0,0.5)
    ax.set_ylim(-0.1,1)
    ax.set_title('{}'.format(title),weight='bold',fontsize=18)  
    ax.legend(loc='lower right',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\NDVI_SMAP\{}'.format(title[0:7]))




path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\SMAP_SM_1km\south_africa_monthly\south_africa_monthly'
files = sorted(glob.glob(path+'\*.nc'))
smap = xr.open_mfdataset(files[0],parallel=True,chunks={"x": 100,"y":100}).SM_vwc
smap_dates = pd.date_range('2015-04','2021-01',  freq='1M') 


path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_NDVI'
files = sorted(glob.glob(path+"/*.nc"))
ndvi = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 100,"lon":100}).NDVI
scaled_ndvi = ndvi.rio.write_crs('epsg:4326').rio.reproject_match(smap.rio.write_crs('epsg:4326'),resampling=Resampling.average)
scaled_ndvi = scaled_ndvi[153:222] #SMAP timeline\

[linearmodel_plot_sm(smap[i],scaled_ndvi[i],'SM','NDVI','{}'.format(date)) for i,date in zip(range(0,len(smap)),smap_dates)]


