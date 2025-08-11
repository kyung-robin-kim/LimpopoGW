import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from shapely.geometry import Point
import datetime
import glob
import xarray as xr
import pymannkendall as mk
from scipy.stats import spearmanr
import rasterio as rio
import rioxarray
from scipy import stats
from shapely.geometry import mapping, box


plt.rcParams["font.family"] = "Times New Roman"
def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

def plot_np(array,vmin,vmax,title):
    array = np.where(array==0,np.nan,array)
    fig1, ax1 = plt.subplots(figsize=(10,8))
    image = ax1.imshow(array,cmap = 'RdBu_r',vmin=vmin,vmax=vmax)
    cbar = fig1.colorbar(image,ax=ax1)
    ax1.set_title('{}'.format(title))

growing_months = [10,11,12,1,2,3]

######################################################################################################################################################
#GROUNDWATER INSITU
SA_gw_data = pd.read_csv(glob.glob(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\sophia\Limpopo In-Situ Data\Boreholes\*.csv')[-2])
insitu_points_gw = [Point(xy) for xy in zip(SA_gw_data['Longitude'], SA_gw_data['Latitude'])]
insitu_gpd_gw = gpd.GeoDataFrame(SA_gw_data, geometry=insitu_points_gw).set_crs('EPSG:4326')

limpopo_watershed = gpd.read_file(r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp')
limpopo_gw_insitu = insitu_gpd_gw[insitu_gpd_gw.within(limpopo_watershed.loc[0,'geometry'])]

start = 40-4 #October 1999:0 .... Feb 2003: 40  January 2011: 135
years = 21

end = start + 12*years-1
#10 years = 132 months

limpopo_gw_insitu_filtered = limpopo_gw_insitu[limpopo_gw_insitu.filter(like='lev').iloc[:,start+1:end+1].T.count()>120]
limpopo_gw_insitu_filtered.plot()
for i, label in enumerate(limpopo_gw_insitu_filtered.index):
    plt.annotate(label, (limpopo_gw_insitu_filtered.Longitude.iloc[i], limpopo_gw_insitu_filtered.Latitude.iloc[i]), textcoords="offset points", xytext=(0,0), ha='center')

limpopo_gw_insitu_filtered.iloc[:,585] #rocks
points = np.array([geom.xy for geom in limpopo_gw_insitu_filtered.geometry])
geology = limpopo_gw_insitu_filtered.iloc[:,584:586]
site_elevation= limpopo_gw_insitu_filtered.iloc[:,10].T
gw_masl = limpopo_gw_insitu_filtered.filter(like='masl').iloc[:,start:(end)].T
gw_level = limpopo_gw_insitu_filtered.filter(like='lev').iloc[:,start+1:(end+1)].T

#LULC from Sentinel-2
SA_gw_data_LULC = pd.read_csv(glob.glob(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\sophia\Limpopo In-Situ Data\Boreholes\lulc_boreholes\*.csv')[0]).iloc[:,[1,-1,-2]]

#Create datetime index for easy formatting
dt_index = [pd.to_datetime(year_month[0:8],format='%Y-%b') for year_month in gw_level.index]
gw_level.index = dt_index
#Choose to interpolate or not for averaging purposes
gw_level = gw_level.interpolate('linear')
#Filter out boreholes near urban areas (Pretoria/Joberg)   #(SA_gw_data_LULC.Sentinel17!=5) & (SA_gw_data_LULC.Sentinel17!=7) & (SA_gw_data_LULC.MODIS17<12) & 
gw_level = gw_level.loc[:,((limpopo_gw_insitu_filtered.Latitude>-25.5)).values]
#Convert to water year for growing season means / year
gw_level['water_year'] = gw_level.index.year.where(gw_level.index.month < 10, gw_level.index.year + 1)




start_year = 2004
end_year = 2022

#OPTION A: No seasonal component -- just normalize for all record
gw_anomaly = (gw_level - gw_level.mean(axis=0))/gw_level.std(axis=0,ddof=1)
gw_anomaly.mean(axis=1).plot()
(-2* gw_anomaly.std(axis=1,ddof=1) + gw_anomaly.mean(axis=1)).plot()
(2* gw_anomaly.std(axis=1,ddof=1) + gw_anomaly.mean(axis=1)).plot()

#OPTION B: Filter and normalize for Growing Season only
#QUESTIONS: Could pre or post growing season be important?
gw_level_wet_monthly = gw_level[gw_level.index.month.isin(growing_months)]
gw_anomaly_wet_monthly = (gw_level_wet_monthly - gw_level_wet_monthly.mean(axis=0))#/ gw_level_wet_monthly.std(axis=0,ddof=1)
gw_anomaly_wet_monthly.mean(axis=1).plot()
(-2* gw_anomaly_wet_monthly.std(axis=1,ddof=1) + gw_anomaly_wet_monthly.mean(axis=1)).plot()
(2* gw_anomaly_wet_monthly.std(axis=1,ddof=1) + gw_anomaly_wet_monthly.mean(axis=1)).plot()

#OPTION C: Seasonally Averaged Anomalies for Growing season only
gw_level_wet_monthly = gw_level[gw_level.index.month.isin(growing_months)]
annual_data =  pd.concat([gw_level_wet_monthly.loc[gw_level_wet_monthly.water_year== i].mean() for i in range(start_year,end_year+1)],axis=1)
annual_std = annual_data.T.iloc[:,:-1].std(ddof=1)
dataset_anomaly_wet_annual = ((annual_data.T - annual_data.mean(axis=1))/annual_std).mean(axis=1)
dataset_anomaly_wet_annual_df_gw = pd.DataFrame(dataset_anomaly_wet_annual).set_index(pd.date_range(start=str(start_year), end=str(end_year), freq='YS')).rename(columns={0:'Growing Season'})



#dataset = gw_level
#monthly_data = [dataset.loc[dataset.index.month == i] for i in range(1,13)]
#monthly_mean = [dataset.loc[dataset.index.month == i].mean(axis=0) for i in range(1,13)]
#monthly_std = [dataset.loc[dataset.index.month == i].std(axis=0,ddof=1) for i in range(1,13)]
#dataset_month_anomalies = [ (monthly_data[i] - monthly_mean[i])/monthly_std[i] for i in range(0,12)]
#monthly_anomalies_gw = pd.concat(dataset_month_anomalies).sort_index()
dataset_anomaly_wet_annual_df_gw.plot(linewidth=2)
gw_anomaly.mean(axis=1).plot(color='C0',linestyle='--',linewidth=1,label='Monthly',legend=True)
(-1* gw_anomaly.std(axis=1,ddof=1) + gw_anomaly.mean(axis=1)).plot(linewidth=0.5,color='C0')
(1* gw_anomaly.std(axis=1,ddof=1) + gw_anomaly.mean(axis=1)).plot(linewidth=0.5,color='C0')
#monthly_anomalies_gw.mean(axis=1).plot(linewidth=0.5,label='Monthly (per Month) Anomalies',legend=True)
plt.axhline(0,color='black',linestyle='--',linewidth=0.5)
plt.ylim(-3,3)
plt.xlabel('Date')
plt.ylabel('Groundwater Level Anomalies')
plt.title('Basin-Averaged In-Situ Boreholes')


######################################################################################################################################################
#DISCHARGE INSITU FROM GRDC
q_data = glob.glob(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\GRDC\2023-05-03_00-21\*nc')[0]
insitu_q = xr.open_mfdataset(q_data)

runoff = insitu_q.runoff_mean[(insitu_q.time>pd.to_datetime('2002-10')) & (insitu_q.time<pd.to_datetime('2022-08'))].transpose('id','time')
runoff_filtered = runoff[runoff.count(dim='time')>6200].transpose('time','id')
station_id = insitu_q.station_name[runoff.count(dim='time')>6200]
geo_x = insitu_q.geo_x[runoff.count(dim='time')>6200]
geo_y = insitu_q.geo_y[runoff.count(dim='time')>6200]
geo_z = insitu_q.geo_z[runoff.count(dim='time')>6200]
areas_filtered = insitu_q.area[runoff.count(dim='time')>6200]

runoff_filtered = runoff_filtered[:,(geo_y>-25.5).values]
area_filtered = areas_filtered[(geo_y>-25.5).values]*(1000000) #Convert km2 to m2
velocity = (runoff_filtered/area_filtered)*3600*24 #units of m/d
monthly_runoff_mean = velocity.resample({'time':'M'}).sum().mean(axis=0)
monthly_runoff_std = velocity.resample({'time':'M'}).sum().std(axis=0,ddof=1)
#monthly_runoff_anomaly = np.log(monthly_runoff/monthly_runoff.mean())


insitu_points_q = [Point(xy) for xy in zip(geo_x, geo_y)]
insitu_gpd_q = gpd.GeoDataFrame(station_id, geometry=insitu_points_q).set_crs('EPSG:4326')
dataframe_q = pd.concat([station_id.to_dataframe(),geo_x.to_dataframe(),geo_y.to_dataframe(),geo_z.to_dataframe()],axis=1)
#q_anomaly = (runoff_ln - runoff_ln.mean(dim='time'))/ runoff_ln.std(dim='time',ddof=1)

#########################################
#Choose Log_transformed OR raw discharge
#Raw
runoff_raw =  pd.DataFrame(runoff_filtered).set_index(pd.to_datetime(runoff_filtered.time)).resample('1MS').mean()
runoff_raw['water_year'] = runoff_raw.index.year.where(runoff_raw.index.month < 10, runoff_raw.index.year + 1)

#Log-transformed
runoff_ln = (velocity.resample({'time':'M'}).sum())
#Convert to water year for growing season means / year
runoff_ln['water_year'] = runoff_ln.time.dt.year.where(runoff_ln.time.dt.month < 10, runoff_ln.time.dt.year + 1)


runoff = pd.DataFrame(np.log(runoff_ln))
runoff = runoff.where(runoff>-100,np.nan)
runoff_ln_seasonal = np.log(runoff_ln).where(np.log(runoff_ln)>-100,np.nan)

#OPTION A: No seasonal component -- just normalize for all record (post log-transformation)
runoff_anomaly_mean = pd.DataFrame(((runoff - runoff.mean(axis=0))/(runoff.std(axis=0,ddof=1))).mean(axis=1)).set_index(pd.to_datetime(runoff_ln.time))
runoff_anomaly_std = pd.DataFrame(((runoff - runoff.mean(axis=0))/(runoff.std(axis=0,ddof=1))).std(axis=1,ddof=1)).set_index(pd.to_datetime(runoff_ln.time))

#(-2* runoff_anomaly.std(axis=1,ddof=1) + runoff_anomaly.mean(axis=1)).plot()
#(2* runoff_anomaly.std(axis=1,ddof=1) + runoff_anomaly.mean(axis=1)).plot()

#OPTION B: Filter and normalize for Growing Season only
#QUESTIONS: Could pre or post growing season be important?
runoff_level_wet_monthly = runoff_ln_seasonal[runoff_ln_seasonal.time.dt.month.isin(growing_months)]
runoff_anomaly_wet_monthly = runoff_level_wet_monthly - runoff_level_wet_monthly.mean(axis=0)#/ runoff_level_wet_monthly.std(axis=0,ddof=1)
runoff_anomaly_wet_monthly.mean(dim='id').plot()
#(-2* runoff_anomaly_wet_monthly.std(axis=1,ddof=1) + runoff_anomaly_wet_monthly.mean(axis=1)).plot()
#(2* runoff_anomaly_wet_monthly.std(axis=1,ddof=1) + runoff_anomaly_wet_monthly.mean(axis=1)).plot()

#OPTION C: Seasonally Averaged Anomalies for Growing season only
runoff_level_wet_monthly = runoff_ln_seasonal[runoff_ln_seasonal.time.dt.month.isin(growing_months)]
annual_data =  xr.concat([runoff_level_wet_monthly[runoff_level_wet_monthly.water_year== i].mean(dim='time') for i in range(start_year,end_year+1)],dim='q_growing')
annual_std = annual_data.std(dim='id',ddof=1)
dataset_anomaly_wet_annual = (annual_data.T - annual_data.mean(dim='q_growing'))/annual_std 
dataset_anomaly_wet_annual_df_r = pd.DataFrame(dataset_anomaly_wet_annual.mean(dim='id')).set_index(pd.date_range(start=str(start_year), end=str(end_year), freq='YS')).rename(columns={0:'Growing Season'})

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




###########################################################################
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


fit_params = stats.pearson3.fit(p_gpm.P_mm.mean(dim={'x','y'}))
shape, loc, scale = fit_params
transformed_p_gpm = stats.pearson3.logpdf(p_gpm.P_mm.mean(dim={'x','y'}), shape, loc, scale)

#################################################
#EVAPOTRANSPIRATION
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_ET'
files = sorted(glob.glob(path+'\*.nc'))
et_modis = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 500,"lon":500})

#path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_PET'
#files = sorted(glob.glob(path+'\*.nc'))
#pet_modis = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 500,"lon":500})

#################################################
#P-ET
LHS = p_gpm.P_mm.mean(dim={'x','y'}) -  et_modis.ET_kg_m2.mean(dim={'x','y'})

###################################################
#GLDAS - CLSM
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_CLSM'
files = sorted(glob.glob(path+'\*.nc'))
#GW
clm_gw = xr.open_mfdataset(files[0],parallel=True).mm
clm_gw_anom = (clm_gw.mean(dim={'x','y'}) - clm_gw.mean(dim={'x','y'}).mean())/clm_gw.mean(dim={'x','y'}).std(ddof=1)


def runoff_accum(runoff_data):
    #convert kg/m2/s to mm lwe
    limpopo_watershed
    monthly_runoff = ((runoff_data*3600*24/1000).resample({'time':'M'}).sum()*1000).rio.write_crs('epsg:4326').rio.clip(limpopo_watershed.geometry.apply(mapping), limpopo_watershed.crs, drop=True,all_touched=False)
    return monthly_runoff

#Runoff Surface
clm_r = runoff_accum(xr.open_mfdataset(files[2],parallel=True)['kgm-2s-1']).rename('mm')
clm_r_anom = (np.log(clm_r.mean(dim={'x','y'})) - np.log(clm_r.mean(dim={'x','y'}).mean()))

clm_bq = xr.open_mfdataset(files[1],parallel=True)['kgm-2s-1']
clm_bq_anom = (np.log(clm_bq.mean(dim={'x','y'})) - np.log(clm_bq.mean(dim={'x','y'}).mean()))

#SOIL MOISTURE
clm_sm_rz = xr.open_mfdataset(files[3],parallel=True)['kgm-2']
clm_sm_surface = xr.open_mfdataset(files[4],parallel=True)['kgm-2']


###################################################
#SMAP
SMAP_files = sorted(glob.glob(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\SMAP_SM_1km\south_africa_monthly\*.nc'))
sm_2015_2020_monthly = xr.open_mfdataset([SMAP_files[0]],parallel=True,chunks={'x':500,'y':500}).SM_vwc
sm_2016_2021_monthly = xr.open_mfdataset([SMAP_files[2]],parallel=True,chunks={'x':500,'y':500}).band_data
sm_2017_2018_monthly = xr.open_mfdataset([SMAP_files[4]],parallel=True,chunks={'x':500,'y':500}).band_data
sm_2021_2022_monthly = xr.open_mfdataset([SMAP_files[6]],parallel=True,chunks={'x':500,'y':500}).band_data

sm_2015_2020_monthly.mean(dim={'x','y'}).plot()
#sm_2017_2018_monthly.mean(dim={'x','y'}).plot()
sm_2016_2021_monthly.mean(dim={'x','y'})[:,0].plot()
sm_2021_2022_monthly.mean(dim={'x','y'})[:,0].plot()
sm_ts = xr.concat([sm_2015_2020_monthly,(sm_2021_2022_monthly[:,0])],dim='time')

fig,ax = plt.subplots()
ax2 = ax.twinx()
#ndvi.mean(dim={'x','y'}).rolling(time=12).mean().NDVI.plot(ax=ax,color='C1')
#et_modis.mean(dim={'x','y'}).ET_kg_m2.rolling(time=12).mean().plot(ax=ax2)
#lst.mean(dim={'x','y'}).LST_K.rolling(time=12).mean().plot(ax=ax,color='C3')
sm_ts.rolling(time=12).mean().plot(ax=ax,color='C2')
clm_gw.mean(dim={'x','y'}).rolling(time=12).mean().plot(color='C0')


'''
start_year = 2004
end_year = 2023

#Select only growing season (oct - mar):
def growing_annual(dataset):
    dataset['water_year'] = dataset.time.dt.year.where(dataset.time.dt.month < 10, dataset.time.dt.year + 1)
    dataset = dataset[dataset.time.dt.month.isin(growing_months)]
    annual_data = [dataset.loc[dataset.water_year == i].mean(axis=0) for i in range(start_year,end_year)]
    return annual_data

def growing_anom(annual_data):
    annual_mean = xr.concat(annual_data,dim='year_i').mean(dim='year_i')
    annual_std = xr.concat(annual_data,dim='year_i').std(dim='year_i',ddof=1)
    anomaly_raster = xr.concat([(data - annual_mean)/annual_std  for data in annual_data],dim='year_i')
    return anomaly_raster

netcdf_anom_path = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies_V2\netcdfs'

#LST, NDVI, GW, SM
variables = [lst.LST_K, ndvi.NDVI, clm_gw, clm_sm_rz, clm_sm_surface, sm_ts]
[print(var.dims) for var in variables]
variable_names = ['LST','NDVI','GW','RZ','Surface SM', 'SMAP']
variables = [growing_anom(growing_annual(var)) for var in variables]
[var.to_netcdf(netcdf_anom_path+r'\{}.nc'.format(name)) for var,name in zip(variables,variable_names)]

#ET - resampled for memory (500m to 1km)
variables = [et_modis.ET_kg_m2]
[print(var.dims) for var in variables]
variable_names = ['ET_1km']
[growing_anom(growing_annual(var.rio.write_crs('epsg:4326').rio.reproject_match(lst.rio.write_crs('epsg:4326'), resampling=rio.enums.Resampling.bilinear))).to_netcdf(netcdf_anom_path+r'\{}.nc'.format(name))
               for var,name in zip(variables,variable_names)]

#PPT
precip_variables = [np.log(p_gpm.P_mm).where(np.log(p_gpm.P_mm)>-100,np.nan), np.log(p_chirps.P_mm).where(np.log(p_chirps.P_mm)>-100,np.nan)]
[print(var.dims) for var in precip_variables]
precip_variable_names = ['PPT_GPM','PPT_CHIRPS']
[growing_anom(growing_annual(var)).to_netcdf(netcdf_anom_path+r'\{}.nc'.format(name)) for var,name in zip(precip_variables,precip_variable_names)]

#VHI
file_vhi = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\netcdfs\VHI\VHI.nc' #only applied VHI calculations
vhi_dataset = xr.open_mfdataset(file_vhi,parallel=True)
growing_anom(growing_annual(vhi_dataset.VHI)).to_netcdf(netcdf_anom_path+r'\VHI\VHI_anom.nc')

#SPEI
def growing_annual_SPEI(dataset):
    dataset['water_year'] = dataset.time.dt.year.where(dataset.time.dt.month < 10, dataset.time.dt.year + 1)
    dataset = dataset[dataset.time.dt.month.isin(growing_months)]
    annual_data = [dataset.loc[dataset.water_year == i].mean(axis=0) for i in range(start_year,end_year)]
    return annual_data

files_spei = sorted(glob.glob(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\climate_indices\SPEI\*.nc'))
files_indicators = [xr.open_mfdataset(file).transpose('time','lat','lon')['{}'.format(variable)] for file,variable in zip([files_spei[5],files_spei[7],files_spei[4]],['spei_pearson_03','spei_pearson_12','spei_pearson_01'])]
[xr.concat(growing_annual_SPEI(xr.concat(spei,dim='year_i')),dim='year_i').to_netcdf(netcdf_anom_path+r'\SPEI\SPEI_{:02d}_anom.nc'.format(value)) for spei,value in zip(files_indicators,[3,12,1])]

#RUNOFF
clm_r
growing_anom(growing_annual(clm_r)).to_netcdf(netcdf_anom_path+r'\Runoff.nc')

'''
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



##############################################################################################################################
#Seasonal Trends (MK // Spearman's Rank)

climate_shapefile_paths = sorted(glob.glob(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\climatezone_shapefiles\*.shp'))
climate_shapefiles = [gpd.read_file(shp) for shp in climate_shapefile_paths]
climatezones = ['Arid','Coastal Plains','Escarpment','Semi-arid']

var_datasets_clipped = [[var.rio.write_crs('WGS84').rio.clip(shp.geometry.apply(mapping), shp.crs, drop=True,all_touched=True) for var in var_datasets] for shp in climate_shapefiles]
indices_datasets_clipped = [[var.rio.write_crs('WGS84').rio.clip(shp.geometry.apply(mapping), shp.crs, drop=True,all_touched=True) for var in indices_datasets] for shp in climate_shapefiles]


spatial_mean_anomalies_vars_growing_zoned = [[var.mean(dim={'x','y'}).assign_coords(year_i=pd.date_range(start='2003-12-31',end='2022-12-31',freq='YS')) for var in vars]
                                                for vars in var_datasets_clipped]
spatial_mean_anomalies_indic_growing_zoned = [[var.mean(dim={'x','y'}).assign_coords(year_i=pd.date_range(start='2003-12-31',end='2022-12-31',freq='YS')) for var in vars]
                                                for vars in indices_datasets_clipped]

spatial_mean_anomalies_vars_growing_zoned[3][3].plot()

for zone in range(0,4):
    print('\n')
    for i in range(0,10):
        print(var_names[i], 'growing for {}'.format(climatezones[zone]))
        print((mk.original_test(spatial_mean_anomalies_vars_growing_zoned[zone][i])))
        print((spearmanr(range(0,len(spatial_mean_anomalies_vars_growing_zoned[zone][i])),spatial_mean_anomalies_vars_growing_zoned[zone][i])))
        
        
for zone in range(0,4):
    print('\n')
    for i in range(0,4):
        print(indices_names[i],'growing for {}'.format(climatezones[zone]))
        print((mk.original_test(spatial_mean_anomalies_indic_growing_zoned[zone][i])))
        print((spearmanr(range(0,len(spatial_mean_anomalies_indic_growing_zoned[zone][i])),spatial_mean_anomalies_indic_growing_zoned[zone][i])))



##############################################################################################################################

spatial_mean_anomalies_vars_growing = [var.mean(dim={'x','y'}).assign_coords(year_i=pd.date_range(start='2003-12-31',end='2022-12-31',freq='YS')) for var in var_datasets]
spatial_mean_anomalies_indic_growing = [var.mean(dim={'x','y'}).assign_coords(year_i=pd.date_range(start='2003-12-31',end='2022-12-31',freq='YS')) for var in indices_datasets]

#Referenced Anomalies
def anomaly_index(dataset):
     anomalies = (dataset.mean(dim={'x','y'}) - dataset.mean(dim={'x','y'}).mean())/(dataset.mean(dim={'x','y'}).std(ddof=1))
     return anomalies
def anomaly_std(dataset):
     anomalies = ((dataset - dataset.mean(dim={'x','y'})))/dataset.std(dim={'x','y'})
     return anomalies
original_datasets = [(np.log(et_modis['ET_kg_m2'])).where(np.log(et_modis['ET_kg_m2'])>-100,np.nan),clm_gw,lst['LST_K'],ndvi['NDVI'],np.log(p_chirps['P_mm']).where(np.log(p_chirps['P_mm'])>-100,np.nan),np.log(p_gpm['P_mm']).where(np.log(p_gpm['P_mm'])>-100,np.nan),clm_sm_rz,np.log(clm_r).where(np.log(clm_r)>-100,np.nan),sm_ts,clm_sm_surface]
anomaly_datasets_means = [anomaly_index(data) for data in original_datasets]

'''
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

#VALIDATION
#Filter out lat/lon for point value extraction
gw_lat = np.array(limpopo_gw_insitu_filtered.Latitude[((limpopo_gw_insitu_filtered.Latitude>-25.5)).values])
gw_lon = np.array(limpopo_gw_insitu_filtered.Longitude[((limpopo_gw_insitu_filtered.Latitude>-25.5)).values])
raster_array = original_datasets[1]
queries_gw = [(find_nearest(raster_array.y,lat)[1],find_nearest(raster_array.x,lon)[1]) for lat,lon in zip(gw_lat,gw_lon)] 

#Filter out lat/lon for point value extraction
q_lat = np.array(geo_y[(geo_y>-25.5).values])
q_lon = np.array(geo_x[(geo_y>-25.5).values])
raster_array = original_datasets[-3]
queries_q = [(find_nearest(raster_array.y,lat)[1],find_nearest(raster_array.x,lon)[1]) for lat,lon in zip(q_lat,q_lon)] 

for points,data,variable,sites in zip([queries_gw,queries_q],[clm_gw,clm_r],['GW','R'],[limpopo_gw_insitu_filtered.Station[((limpopo_gw_insitu_filtered.Latitude>-25.5)).values],insitu_gpd_q.iloc[:,0]]):
    all_timeseries = []
    for point,name in zip(points,sites):
        timeseries = data[:,point[0],point[1]].to_dataframe().iloc[:,-1].rename(name)
        all_timeseries.append(timeseries)

    all_df = pd.concat(all_timeseries,axis=1)
    all_df.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\in_situ\validation\{}.csv'.format(variable))
'''


#P - ET - R - SM_RZ - SM_Surf
var_names = ['ET', 'GW','LST','NDVI','PPT_CHIRPS','PPT_GPM','RZ','Runoff','SMAP','SM_Surf']
clm_sm_rz,clm_r,sm_ts,clm_sm_surface

RHS_GPM = p_gpm['P_mm'].mean(dim={'x','y'}) - et_modis['ET_kg_m2'].mean(dim={'x','y'}) - clm_r.mean(dim={'x','y'})
RHS_CHIRPS = p_chirps['P_mm'].mean(dim={'x','y'}) - et_modis['ET_kg_m2'].mean(dim={'x','y'}) - clm_r.mean(dim={'x','y'})
 
CLSM_GW = (original_datasets[1].mean(dim={'x','y'}) - original_datasets[1].mean(dim={'x','y'}).mean())


plt.plot(RHS_CHIRPS[:-1],CLSM_GW[1:],'*')



#RHS_GPM.plot()
grace_limpopo_new_xr = xr.concat(grace_limpopo,dim='time').resample(time='1M').mean()

plt.figure(figsize=(6,4))
CLSM_GW.plot(label='CLSM')
plt.plot(grace_limpopo_new_xr.time, grace_limpopo_new_xr.mean(dim={'lat','lon'})*1000,label='GRACE')
plt.xlim(pd.to_datetime('2003-02-01'),pd.to_datetime('2022-07-01'))
plt.axhline(0,color='black',linestyle='--',linewidth=0.5)
plt.title('Monthly Groundwater Anomalies',weight='bold',fontsize=15)
plt.xlabel('Date',weight='bold',fontsize=12)
plt.ylabel('LWE (mm)',weight='bold',fontsize=12)
plt.tight_layout()
plt.legend()



#NEW DATA (V4)
path = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\GRACE_Mascons\JPL'
files = sorted(glob.glob(path+"/*.nc"))
shpname = r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
shapefile = gpd.read_file(shpname)

grace_fo = [xr.open_mfdataset(file,parallel=True,chunks={"lat": 100,"lon":100}).lwe_thickness for file in files]
grace_limpopo = [grace.rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)
                    for grace in grace_fo]
grace_limpopo_new_xr = xr.concat(grace_limpopo,dim='time') #CONCAT CORRUPTS GRACE v04 FILES

grace_limpopo_new = np.empty([len(grace_limpopo),8,10])
for i in range(0,len(grace_limpopo)):
    grace_limpopo_new[i,:,:] = np.array(grace_limpopo[i])

#grace_limpopo_new = np.nanmean(grace_limpopo_new,axis=0)
grace_limpopo_new = np.nanmean(np.nanmean(grace_limpopo_new,axis=1),axis=1)

fig,ax = plt.subplots()
ax.plot(grace_limpopo_new_xr.time, grace_limpopo_new*1000)
ax.set_ylabel('TWSA (mm)',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)


#CORRUPTED DATA, but GAP FILLED
#twsa_da = grace_scaled - grace_scaled.mean(dim='time')
grace_scaled_monthly = grace_limpopo_new_xr.resample(time='1M').mean() #GAP FILL

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(grace_scaled_monthly.time,grace_scaled_monthly.mean(dim=['lat','lon'])*100,'--',color='C1',linewidth=2)
ax.plot(grace_limpopo_new_xr.time, grace_limpopo_new*100)
#ax.plot(grace_scaled_old.time,grace_scaled_old.mean(dim=['lat','lon']),color='C0')
ax.set_ylabel('TWSA (cm)',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)


#Plotting All Datasets & Variables
clm_validation = glob.glob(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\in_situ\validation\*.csv')
clm_dfs = [pd.read_csv(file) for file in clm_validation]
[df.set_index(pd.to_datetime(df.time),inplace=True) for df in clm_dfs]
clm_dfs = [df.iloc[:,1:] for df in clm_dfs]

var_names = ['MODIS ET', 'CLSM GW','MODIS LST','MODIS NDVI','CHIRPS P','GPM P','CLSM RZ SM','CLSM Surface Runoff','SMAP Surface SM','CLSM Surface SM']
for i in range(0,10):
    plt.figure(figsize=(6,4))
    plt.plot(anomaly_datasets_means[i].time,anomaly_datasets_means[i],linewidth=0.4,color='black')
    plt.plot(spatial_mean_anomalies_vars_growing[i].year_i,spatial_mean_anomalies_vars_growing[i],label='{}'.format(var_names[i]),color='black')
    plt.axhline(0,color='black',linestyle='--',linewidth=0.5)
    plt.ylabel('z-score')
    plt.xlabel('Date')
    plt.ylim(-3,3)
    plt.grid(axis='x')
    plt.xlim(pd.to_datetime('2003-02-01'),pd.to_datetime('2022-07-01'))
    plt.title(var_names[i])
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\timeseries\{}.png'.format(var_names[i]))


#############################################################################
#Plots of variables called above:

#Mean Precip & Soil Moisture
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot()
ax.plot(p_gpm.time,p_gpm.P_mm.mean(dim=['x','y']),color='black',linewidth=1,label='GPM IMERG Precipitation')
#ax.plot(p_chirps.time,p_chirps.P_mm.mean(dim=['x','y']),color='C1',label='CHIRPS P')
ax.fill_between(p_gpm.time,0,p_gpm.P_mm.mean(dim=['x','y']), color='black', alpha=0.1) 
#ax.fill_between(p_chirps.time,0,p_chirps.P_mm.mean(dim=['x','y']), color='C1', alpha=0.1) 
ax.set_ylim(0,300)
#ax.invert_yaxis()
ax.set_ylabel('Precipitation (mm)',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_xlim(pd.to_datetime('2003-02-01'),pd.to_datetime('2022-07-01'))
plt.legend(loc='upper left')

ax2 = ax.twinx()
ax2.plot(sm_ts.time,sm_ts.mean(dim={'x','y'})*5,color='navy',label='SMAP Soil Moisture (5 cm)',linewidth=1.5) #approximately 1/5 of CLSM
ax2.set_ylim(0,6)
ax2.plot(clm_sm_surface.time,clm_sm_surface.mean(dim={'x','y'}),label='CLSM Soil Moisture (2 cm)',linewidth=1.5)
ax2.set_ylabel('Soil Moisture (mm)',weight='bold',fontsize=12)
plt.legend(loc='upper right')
ax.grid(axis='x')
ax2.grid(axis='x')
ax.set_title('(a)',weight='bold',fontsize=15)
plt.tight_layout()
plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\timeseries\TSA.png',dpi=300)


#ET & NDVI & LST
degree_sign = u"\N{DEGREE SIGN}"
fig = plt.figure(figsize=(8.65,4))
ax = fig.add_subplot()
ax.plot(et_modis.time,et_modis.ET_kg_m2.mean(dim=['x','y']),color='black',linewidth=1,label='ET')
ax.fill_between(et_modis.time,et_modis.ET_kg_m2.mean(dim=['x','y']), color='black', alpha=0.1) 
ax.set_ylim(0,150)
#ax.invert_yaxis()
ax.set_ylabel('Evapotranspiration (mm)',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_xlim(pd.to_datetime('2003-02-01'),pd.to_datetime('2022-07-01'))
plt.legend(loc='upper left')

ax2 = ax.twinx()
ax2.plot(ndvi.time,ndvi.NDVI.mean(dim={'x','y'}),color='darkgreen',label='NDVI',linewidth=2) #approximately 1/5 of CLSM
ax2.set_ylim(0.2,0.75)

ax2.set_ylabel('Norm. Difference Vegetation Index',weight='bold',fontsize=11)
plt.legend(loc='upper center')

ax3 = ax.twinx()
ax3.plot(lst.time,lst.LST_K.mean(dim={'x','y'})-273.15,label='LST',color='C3',linewidth=0.7)
ax3.set_ylabel('Land Surface Temperature ({}C)'.format(degree_sign),weight='bold',fontsize=11)
ax3.set_ylim(10,35)
ax3.spines['right'].set_position(('outward', 40))
plt.legend(loc='upper right')

ax.grid(axis='x')
ax2.grid(axis='x')
ax3.grid(axis='x')
ax.set_title('(b)',weight='bold',fontsize=15)
plt.tight_layout()
plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\timeseries\TSB.png',dpi=300)



#Mean GW & Runoff
fig = plt.figure(figsize=(8.2,4))
ax = fig.add_subplot()
ax.plot(clm_gw.time,clm_gw.mean(dim=['x','y']),color='C0',linewidth=1.5,label='CLSM GW')
ax.set_ylim(500,690)
ax.set_ylabel('Groundwater (mm)',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_xlim(pd.to_datetime('2003-02-01'),pd.to_datetime('2022-07-01'))
#plt.legend(loc='upper left')

ax2 = ax.twinx()
ax2.plot(clm_r.time,clm_r.mean(dim={'x','y'}),color='navy',label='CLSM Q',linewidth=1.5)
ax2.set_ylim(0,2.2)
ax2.set_ylabel('Surface Runoff (mm)',weight='bold',fontsize=12)
plt.legend(loc='upper center')

ax.grid(axis='x')
ax2.grid(axis='x')
ax.set_title('(c)',weight='bold',fontsize=15)
plt.tight_layout()
plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\timeseries\TSC.png',dpi=300)


#Validation
start_year = 2004
end_year = 2023

def growing_annual(dataset):
    dataset['water_year'] = dataset.index.year.where(dataset.index.month < 10, dataset.index.year + 1)
    dataset = dataset[dataset.index.month.isin(growing_months)]
    annual_data = pd.concat([dataset.loc[dataset.water_year == i].mean(axis=0) for i in range(start_year,end_year)],axis=1)
    return annual_data

def growing_anom(annual_data):
    annual_mean = annual_data.mean(axis=1)
    annual_std = annual_data.std(axis=1,ddof=1)
    anomalies = pd.concat([((annual_data.iloc[:,i] - annual_mean)/annual_std) for i in range(0,len(annual_data.keys()))],axis=1).iloc[:-1,:]
    return anomalies

clm_gw_sites = (growing_anom(growing_annual(clm_dfs[0])))
clm_r_sites = (growing_anom(growing_annual(clm_dfs[1])))


i=1
for ii in range(0,len(gw_level.keys())):
    fig,ax = plt.subplots()
    ax.plot(spatial_mean_anomalies_vars_growing[i].year_i,spatial_mean_anomalies_vars_growing[i],linewidth=2,c='black',label='Growing Season')
    #ax.plot(spatial_mean_anomalies_vars_growing[i].year_i,clm_gw_sites.mean(axis=0),linewidth=2,c='black',label='CLSM GW Levels')
    #ax.plot(clm_gw_anom.time.resample({'time':'MS'}).first().time,clm_gw_anom,linewidth=0.5,c='black')
    ax.plot((gw_level.iloc[:,ii]-gw_level.iloc[:,ii].mean())/(gw_level.iloc[:,ii].std(ddof=1)),linewidth=2,c='C0',label='InSitu Boreholes')
    #ax.plot(gw_anomaly.mean(axis=1),linewidth=0.7)
    #(-1* gw_anomaly.std(axis=1,ddof=1) + gw_anomaly.mean(axis=1)).plot(linewidth=0.5,color='C0')
    #(1* gw_anomaly.std(axis=1,ddof=1) + gw_anomaly.mean(axis=1)).plot(linewidth=0.5,color='C0')
    plt.title('Groundwater Anomalies')
    plt.legend()
    #plt.ylim(-3.2,3.2)
    plt.axhline(0,color='black',linestyle='--',linewidth=0.5)
    plt.grid(axis='x')
    plt.xlim(pd.to_datetime('2003-02-01'),pd.to_datetime('2022-07-01'))
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\seasonal_insitu\final\individual_bh\below_-25\{:03d}.png'.format(ii))

#GROUNDWATER PLOT
i=1
fig,ax = plt.subplots(figsize=(6,4))
ax.plot(spatial_mean_anomalies_vars_growing[i].year_i,spatial_mean_anomalies_vars_growing[i],linewidth=2,c='black',label='CLSM GW')
ax.plot(dataset_anomaly_wet_annual_df_gw,linewidth=2,c='C0',label='InSitu Boreholes')
plt.plot(-1* gw_anomaly.std(axis=1,ddof=1) + gw_anomaly.mean(axis=1),linewidth=0.1,color='C0')
plt.plot(1* gw_anomaly.std(axis=1,ddof=1) + gw_anomaly.mean(axis=1),linewidth=0.1,color='C0')
plt.fill_between(gw_anomaly.index,(-1* gw_anomaly.std(axis=1,ddof=1) + gw_anomaly.mean(axis=1)), (1* gw_anomaly.std(axis=1,ddof=1) + gw_anomaly.mean(axis=1)), alpha=0.2) 
plt.legend()
plt.ylim(-3,3)
plt.axhline(0,color='black',linestyle='--',linewidth=0.5)
plt.grid(axis='x')
plt.xlim(pd.to_datetime('2003-02-01'),pd.to_datetime('2022-07-01'))
plt.title('(a) Groundwater Anomalies',weight='bold',fontsize=15)
plt.xlabel('Date',weight='bold',fontsize=12)
plt.ylabel('z-score',weight='bold',fontsize=12)
plt.tight_layout()
plt.legend()
plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\timeseries\insitu_GW.png',dpi=300)


#RUNOFF PLOT
i=-3 #Runoff 
#3 # NDVI
fig,ax = plt.subplots(figsize=(6,4))
ax.plot(spatial_mean_anomalies_vars_growing[i].year_i,spatial_mean_anomalies_vars_growing[i],linewidth=2,c='black',label='CLSM Surface Runoff')
#ax.plot(spatial_mean_anomalies_vars_growing[i].year_i,clm_r_sites.mean(axis=0),linewidth=2,c='black',label='CLSM Surface Runoff')
#ax.plot(clm_bq_anom.time,clm_bq_anom,linewidth=0.5,c='black')
plt.plot(dataset_anomaly_wet_annual_df_r,linewidth=2,label='InSitu Gauges Discharge')
#plt.plot(runoff_anomaly_mean,color='C0',linestyle='--',linewidth=1,label='Monthly')
plt.plot(-1* runoff_anomaly_std + runoff_anomaly_mean,linewidth=0.1,color='C0')
plt.plot(1* runoff_anomaly_std + runoff_anomaly_mean,linewidth=0.1,color='C0')
plt.fill_between(x,(-1* runoff_anomaly_std + runoff_anomaly_mean)[0], (1* runoff_anomaly_std + runoff_anomaly_mean)[0], alpha=0.2) 
plt.legend()
plt.ylim(-3,3)
plt.axhline(0,color='black',linestyle='--',linewidth=0.5)
plt.grid(axis='x')
plt.xlim(pd.to_datetime('2003-02-01'),pd.to_datetime('2022-07-01'))
plt.title('(b) Runoff/Discharge Anomalies',weight='bold',fontsize=15)
plt.xlabel('Date',weight='bold',fontsize=12)
plt.ylabel('z-score',weight='bold',fontsize=12)
plt.tight_layout()
plt.legend()
plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\timeseries\insitu_Q.png',dpi=300)


'''
#Monthly Anomalies
netcdf_anom_path = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\netcdfs'
files_vhi = sorted(glob.glob(netcdf_anom_path+'\VHI\*.nc'))
files_spei = sorted(glob.glob(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\climate_indices\SPEI\V1\*.nc'))
files_indicators = [files_vhi[1],files_spei[1],files_spei[3],files_spei[0]]
indices_datasets = [xr.open_mfdataset(file,parallel=True) for file in files_indicators]
inidices_keys = [list(ds.data_vars)[0] for ds in indices_datasets]
indices_datasets = [var['{}'.format(key)] for var,key in zip(indices_datasets[0:1],inidices_keys[0:1])] + [var['{}'.format(key)].rename({'lat':'y','lon':'x'}).transpose('time','y','x') for var,key in zip(indices_datasets[1:],inidices_keys[1:])]
indices_names = ['VHI','SPEI-3','SPEI-12','SPEI-1']
files = sorted(glob.glob(netcdf_anom_path+'\*.nc'))
var_datasets = [xr.open_mfdataset(file,parallel=True) for file in files]
var_keys = [list(ds.data_vars)[0] for ds in var_datasets]
var_datasets = [var['{}'.format(key)] for var,key in zip(var_datasets,var_keys)]
var_names = ['ET', 'GW','LST','NDVI','PPT_GPM','RZ','Runoff','SMAP','SM_Surf']

spatial_mean_anomalies_vars_monthly = [var.mean(dim={'x','y'}) for var in var_datasets]
spatial_mean_anomalies_indic_monthly = [var.mean(dim={'x','y'}) for var in indices_datasets]
'''

vhi_anom = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies_V2\netcdfs\VHI\VHI_anom.nc'
files_spei = sorted(glob.glob(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies_V2\netcdfs\SPEI\*.nc'))
files_indicators = [vhi_anom,files_spei[1],files_spei[2],files_spei[0]]

#STATISTICAL TEST
#insitu
(mk.original_test(dataset_anomaly_wet_annual_df_gw)) #groundwater is depleteing? based on in-situ data (non-irrigated lands)
(spearmanr(range(0,len(dataset_anomaly_wet_annual_df_gw)), (dataset_anomaly_wet_annual_df_gw))) 

(mk.original_test(dataset_anomaly_wet_annual_df_r)) #no real trend with streamflow
(spearmanr(range(0,len(dataset_anomaly_wet_annual_df_r)), (dataset_anomaly_wet_annual_df_r)))

#modeled
(mk.original_test(spatial_mean_anomalies_vars_growing[1])) #questionably low with groundwater
(spearmanr(range(0,len(spatial_mean_anomalies_vars_growing[1])), spatial_mean_anomalies_vars_growing[1]))

(mk.original_test(spatial_mean_anomalies_vars_growing[7])) #no trend in streamflow, again
(spearmanr(range(0,len(spatial_mean_anomalies_vars_growing[7])),spatial_mean_anomalies_vars_growing[7]))

(mk.original_test(spatial_mean_anomalies_vars_growing[3])) #ndvi again
(spearmanr(range(0,len(spatial_mean_anomalies_vars_growing[3])),spatial_mean_anomalies_vars_growing[3]))

(mk.original_test(spatial_mean_anomalies_vars_growing[2])) #lst again
(spearmanr(range(0,len(spatial_mean_anomalies_vars_growing[2])),spatial_mean_anomalies_vars_growing[2]))

for i in range(0,10):
    print(var_names[i])
    print((mk.original_test(spatial_mean_anomalies_vars_growing[i])))
    print((spearmanr(range(0,len(spatial_mean_anomalies_vars_growing[i])),spatial_mean_anomalies_vars_growing[i])))
    print(var_names[i],' regular')
    print((mk.seasonal_test(anomaly_datasets_means[i],period=12)))
    print((spearmanr(range(0,len(anomaly_datasets_means[i])),anomaly_datasets_means[i])))
        
for i in range(0,4):
    print(indices_names[i],' growing')
    print((mk.original_test(spatial_mean_anomalies_indic_growing[i])))
    print((spearmanr(range(0,len(spatial_mean_anomalies_indic_growing[i])),spatial_mean_anomalies_indic_growing[i])))
    print(indices_names_monthly[i],' regular')
    print((mk.original_test(spatial_mean_anomalies_indic_monthly[i])))
    print((spearmanr(range(0,len(spatial_mean_anomalies_indic_monthly[i][11:])),spatial_mean_anomalies_indic_monthly[i][11:])))


#VHI
path = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies_V2\netcdfs\VHI'
files = sorted(glob.glob(path+'\*.nc'))
vhi = xr.open_mfdataset(files[0],parallel=True).VHI
vhi_index = (vhi.mean(dim={'x','y'}) - vhi.mean(dim={'x','y'}).mean())/vhi.mean(dim={'x','y'}).std(ddof=1)

plt.figure(figsize=(8,4))
plt.plot(vhi_index.time,vhi_index,color='C1',label='VHI')
plt.plot(spatial_mean_anomalies_indic_monthly[2].time,spatial_mean_anomalies_indic_monthly[2],color='black',label='SPEI-12')
#plt.plot(vhi_index.time,vhi_index-spatial_mean_anomalies_indic_monthly[2],color='black',linestyle=':',label='diff')
plt.title('Drought Indices',weight='bold',fontsize=15)
plt.legend()
plt.ylim(-3,3)
plt.axhline(0,color='black',linestyle='--',linewidth=0.5)
plt.grid(axis='x')
plt.ylabel('z-score',weight='bold',fontsize=12)
plt.xlim(pd.to_datetime('2003-02-01'),pd.to_datetime('2022-07-01'))
plt.xlabel('Date',weight='bold',fontsize=12)
plt.tight_layout()
plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\DroughtIndex.png')

#CLSM vs. NDVI vs. PPT Growing Season Anomalies
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot()
ax.plot(spatial_mean_anomalies_vars_growing[1].year_i,spatial_mean_anomalies_vars_growing[1],color='black',label='CLSM GW')
ax.plot(spatial_mean_anomalies_vars_growing[7].year_i,spatial_mean_anomalies_vars_growing[7],color='black',linestyle=':',label='CLSM Q')
#ax.plot(spatial_mean_anomalies_vars_growing[6].year_i,spatial_mean_anomalies_vars_growing[6],color='red',label='CLSM RZ')
ax.plot(spatial_mean_anomalies_vars_growing[9].year_i,spatial_mean_anomalies_vars_growing[9],color='black',linestyle='--',label='CLSM SM')
ax.set_ylim(-3,3)
ax.legend(loc='upper center')

ax2 = ax.twinx()
ax2.plot(spatial_mean_anomalies_vars_growing[3].year_i,spatial_mean_anomalies_vars_growing[3],color='C2',linewidth=1,label='MODIS NDVI')
#ax2.plot(spatial_mean_anomalies_vars_growing[0].year_i,spatial_mean_anomalies_vars_growing[0],color='C3',label='MODIS ET')
ax2.plot(spatial_mean_anomalies_vars_growing[5].year_i,spatial_mean_anomalies_vars_growing[5],color='C0',linewidth=1,label='GPM IMERG P')
#ax2.plot(spatial_mean_anomalies_vars_growing[4].year_i,spatial_mean_anomalies_vars_growing[4],color='navy',linewidth=2,label='CHIRPS P')

ax2.set_yticks([])
ax2.set_ylim(-3,3)
ax2.legend(loc='lower center')

plt.title('(b) Growing Season Anomalies',weight='bold',fontsize=15)
plt.axhline(0,color='black',linestyle='--',linewidth=0.5)
plt.tight_layout()
ax.set_ylabel('z-score',weight='bold',fontsize=12)
ax.grid(axis='x')
ax2.grid(axis='x')
ax.set_xlabel('Date',weight='bold',fontsize=12)
plt.xlim(pd.to_datetime('2003-02-01'),pd.to_datetime('2022-07-01'))
plt.tight_layout()
plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\timeseries\GrowingSeason_v3.png',dpi=300)
#plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\GrowingSeason_v2.pdf')


var_names_monthly

def monthly_anomaly(dataset):
    dataset=clm_r
    month_idxs=dataset.groupby('time.month').groups
    dataset_month_anomalies = [dataset.isel(time=month_idxs[i]) - dataset.isel(time=month_idxs[i]).mean(dim='time') for i in range(1,13)]
    monthly_anomalies_ds = xr.merge(dataset_month_anomalies)

    d_anomaly_df = (monthly_anomalies_ds.mean(['x','y'])).to_dataframe()
    d_std = d_anomaly_df.iloc[:-1].std()
    
    anomaly_df = (d_anomaly_df.iloc[:,-1]/float(d_std))
    return anomaly_df


def monthly_anom(dataset):
    #dataset = et_modis.ET_kg_m2
    month_idxs=dataset.groupby('time.month').groups
    dataset_month_anomalies = [(dataset.isel(time=month_idxs[i]) - dataset.isel(time=month_idxs[i]).mean(dim='time')) for i in range(1,13)]
    var_name = dataset_month_anomalies[0].var().name
    monthly_anomalies_ds = xr.merge(dataset_month_anomalies)[var_name]

    return monthly_anomalies_ds

clm_r_anom_month = monthly_anom(np.log(clm_r+0.01)).mean(dim={'x','y'})
gpm_p_anom_month = monthly_anom(np.log(p_gpm.P_mm+1)).mean(dim={'x','y'})[1:]

fig = plt.figure(figsize=(8,4))
plt.plot(clm_r_anom_month.mean(dim={'x','y'}))
plt.plot(gpm_p_anom_month[1:].mean(dim={'x','y'}))

#ppt_anomalies = ((np.log(p_gpm.P_mm+0.1).mean(dim={'x','y'}) - np.log(p_gpm.P_mm+0.1).mean(dim={'x','y'}).mean()))/(np.log(p_gpm.P_mm+0.1).mean(dim={'x','y'}).std())
#ppt_anomalies = (np.log((p_gpm.P_mm).mean(dim={'x','y'}) - (p_gpm.P_mm).mean(dim={'x','y'}).mean()))

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot()
ax.plot(spatial_mean_anomalies_vars_monthly[2].time,spatial_mean_anomalies_vars_monthly[2],color='black',linewidth=1,label='CLSM GW')
ax.plot(clm_r_anom_month.time,(clm_r_anom_month),color='black',linestyle=':',label='CLSM Q')
#ax.plot(spatial_mean_anomalies_vars_monthly[8].time,spatial_mean_anomalies_vars_monthly[8],color='black',linestyle=':',linewidth=1,label='CLSM Q')
ax.plot(spatial_mean_anomalies_vars_monthly[11].time,spatial_mean_anomalies_vars_monthly[11],color='black',linestyle='--',label='CLSM SM')
ax.set_ylim(-3,3)
ax.legend(loc='upper center',fontsize=9)

ax2 = ax.twinx()
ax2.plot(spatial_mean_anomalies_vars_monthly[4].time,spatial_mean_anomalies_vars_monthly[4],color='C2',linewidth=1,label='MODIS NDVI')
#ax2.plot(spatial_mean_anomalies_vars_monthly[0].time,spatial_mean_anomalies_vars_monthly[0],color='C3',linewidth=1,label='MODIS ET')
ax2.plot(spatial_mean_anomalies_vars_monthly[6].time,gpm_p_anom_month,color='C0',linewidth=1,label='GPM IMERG P')
#ax2.plot(spatial_mean_anomalies_vars_monthly[6].time,spatial_mean_anomalies_vars_monthly[6],color='C0',linewidth=1,label='GPM IMERG P')
#ax2.plot(spatial_mean_anomalies_vars_monthly[4].time,spatial_mean_anomalies_vars_growing[4],color='navy',linewidth=2,label='CHIRPS P')

ax2.set_yticks([])
ax2.set_ylim(-3,3)
ax2.legend(loc='lower center')
ax.set_ylabel('z-score',weight='bold',fontsize=12)
ax.grid(axis='x')
ax2.grid(axis='x')
ax.set_xlabel('Date',weight='bold',fontsize=12)
plt.xlim(pd.to_datetime('2003-02-01'),pd.to_datetime('2022-07-01'))
plt.title('(a) Monthly Anomalies',weight='bold',fontsize=15)
plt.tight_layout()
plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\timeseries\MonthlyAnomalies_v3.png',dpi=300)
#plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\MonthlyAnomalies_v2.pdf')





#CORRELATIONS
linear_plot(spatial_mean_anomalies_vars_monthly[2],spatial_mean_anomalies_vars_monthly[6][:-3],'GW','P')
linear_plot(spatial_mean_anomalies_vars_monthly[2],spatial_mean_anomalies_vars_monthly[4],'GW','NDVI')
linear_plot(spatial_mean_anomalies_vars_monthly[4][:-2],spatial_mean_anomalies_vars_monthly[6][5:-3],'NDVI','P')



plt.figure(figsize=(8,4))
plt.plot(spatial_mean_anomalies_vars_monthly[7].time,spatial_mean_anomalies_vars_monthly[7],color='black',label='CLSM Q',linewidth=1,linestyle='--')
plt.plot(spatial_mean_anomalies_vars_monthly[11].time,spatial_mean_anomalies_vars_monthly[11],color='black',label='CLSM SM',linewidth=1,linestyle=':')
plt.plot(spatial_mean_anomalies_vars_monthly[4].time,spatial_mean_anomalies_vars_monthly[4],color='C2',label='MODIS NDVI')
plt.title('Monthly Anomalies',weight='bold',fontsize=15)
plt.legend(loc='lower center')
plt.ylim(-3,3)
plt.axhline(0,color='black',linestyle='--',linewidth=0.5)
plt.grid(axis='x')
plt.ylabel('z-score',weight='bold',fontsize=12)
plt.xlim(pd.to_datetime('2003-02-01'),pd.to_datetime('2022-07-01'))
plt.xlabel('Date',weight='bold',fontsize=12)
plt.tight_layout()


plt.figure(figsize=(6,4))
#spatial_mean_anomalies_indic_growing[0].plot(label='VHI')
#spatial_mean_anomalies_indic_growing[1].plot(label='SPEI-3')
#spatial_mean_anomalies_indic_growing[2].plot(label='SPEI-12')
#spatial_mean_anomalies_indic_growing[3].plot(label='SPEI-1')
#spatial_mean_anomalies_vars_growing[0].plot(label='MODIS ET')
#spatial_mean_anomalies_vars_growing[7].plot(label='CLSM Runoff')
#spatial_mean_anomalies_vars_growing[1].plot(label='CLSM GW')
anomaly_datasets_means[1].plot(label='CLSM GW')

#spatial_mean_anomalies_indic_monthly[2].plot(label='SPEI-12')
spatial_mean_anomalies_vars_growing[2].plot(label='MODIS LST')
spatial_mean_anomalies_vars_growing[3].plot(label='MODIS NDVI')
spatial_mean_anomalies_vars_growing[5].plot(label='GPM PPT')
spatial_mean_anomalies_vars_growing[6].plot(label='CHIRPS PPT')
#spatial_mean_anomalies_vars_growing[9].plot(label='CLSM SM')
#spatial_mean_anomalies_vars_growing[8].plot(label='SMAP SM')
plt.title('Drought Indices')
plt.legend()
plt.ylim(-3.2,3.2)
plt.axhline(0,color='black',linestyle='--',linewidth=0.5)
plt.grid(axis='x')
plt.xlim(pd.to_datetime('2003-02-01'),pd.to_datetime('2022-07-01'))

fig,ax = plt.subplots(figsize=(6,4))
ax2 = ax.twinx()
p_gpm.P_mm.mean(dim={'x','y'}).plot(ax=ax2,linestyle='--',linewidth=0.5,color='C1')
spatial_mean_anomalies_vars_monthly[4].plot(ax=ax,label='GPM PPT')
spatial_mean_anomalies_vars_growing[4].plot(ax=ax,label='GPM PPT')
plt.xlim(pd.to_datetime('2003-02-01'),pd.to_datetime('2022-07-01'))


indices_names = ['VHI','SPEI-3','SPEI-12','SPEI-1']
var_names = ['ET','GW','LST','NDVI','PPT_GPM','RZ','SMAP','SM_Surf']

def linear_plot(independent,dependent,var1,var2):
    slope, intercept, r_value, p_value, std_err = stats.linregress(independent, dependent)
    y = np.array(dependent)
    x = np.array(independent)

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    ax.scatter(independent,dependent,s=4)
    ax.plot([], [], ' ', label='p-val = {}'.format(round(p_value, 3)))
    ax.plot([], [], ' ', label='r: {}'.format(round(r_value,5)))
    #ax.set_xlim(-2,2)
    #ax.set_ylim(-2,2)
    ax.legend(loc='upper right')
    ax.set_title('{} vs. {}'.format(var1,var2))

linear_plot(spatial_mean_anomalies_vars_growing[1],spatial_mean_anomalies_vars_growing[0],'GW','ET')
linear_plot(spatial_mean_anomalies_vars_growing[1],spatial_mean_anomalies_vars_growing[2],'GW','LST')
linear_plot(spatial_mean_anomalies_vars_growing[1],spatial_mean_anomalies_vars_growing[3],'GW','NDVI')
linear_plot(spatial_mean_anomalies_vars_growing[1],spatial_mean_anomalies_vars_growing[7],'GW','Q')

linear_plot(spatial_mean_anomalies_vars_growing[4],spatial_mean_anomalies_vars_growing[5],'PPT_GPM','PPT_CHIRPS')
linear_plot(spatial_mean_anomalies_vars_growing[0],spatial_mean_anomalies_vars_growing[3],'ET','NDVI')




#Insitu GW & Runoff
plt.plot(dataset_anomaly_wet_annual_df_gw)
plt.plot(dataset_anomaly_wet_annual_df_r)

#NDVI & GW
(spatial_mean_anomalies_vars_growing[1]).plot(label='CLSM GW')
anomaly_datasets[1].plot()
plt.legend()

spatial_mean_anomalies_vars_growing[3].plot(label='NDVI')
plt.legend()

spatial_mean_anomalies_vars_growing[4].plot(label='PPT')
plt.legend()

spatial_mean_anomalies_vars_growing[0].plot(label='ET')
plt.legend()

spatial_mean_anomalies_vars_growing[2].plot(label='LST')
plt.legend()

#GW
plt.plot(dataset_anomaly_wet_annual_df_gw) #insitu
(spatial_mean_anomalies_vars_growing[1]/10).plot() #CLSM
(spatial_mean_anomalies_vars_growing[3]*100).plot()
(spatial_mean_anomalies_indic_growing[0]).plot()

plt.plot(dataset_anomaly_wet_annual_df_r)

plt.figure()
clm_r_anom.plot()
dataset_anomaly_wet_annual_df_r.plot()


#SPEI-12 and GW
spatial_mean_anomalies_vars[1].plot()
dataset_anomaly_wet_annual_df_gw.plot(linewidth=2)
spatial_mean_anomalies_indic[2].plot()

monthly_anomalies_gw.mean(axis=1).plot()
monthly_anomalies_q.mean(axis=1).plot(linewidth=1)

#VHI and NDVI
spatial_mean_anomalies_vars[3].plot(c='orange')
dataset_anomaly_wet_annual_df_r.plot()

spatial_mean_anomalies_indic[0].plot()
monthly_anomalies_gw.mean(axis=1).plot()
monthly_anomalies_q.mean(axis=1).plot(linewidth=1)

#SM & VHI
spatial_mean_anomalies_vars[-1].plot() #CLSM SM 
#spatial_mean_anomalies_vars[-2].plot() #SMAP
spatial_mean_anomalies_indic[0].plot() #VHI
monthly_anomalies_gw.mean(axis=1).plot()
monthly_anomalies_q.mean(axis=1).plot(linewidth=1)

#SM & SPEI
spatial_mean_anomalies_vars[-1].plot() #CLSM SM
spatial_mean_anomalies_vars[-2].plot() #SMAP
spatial_mean_anomalies_indic[2].plot() #SPEI-12
spatial_mean_anomalies_indic[1].plot() #SPEI-3
monthly_anomalies_gw.mean(axis=1).plot()
monthly_anomalies_q.mean(axis=1).plot(linewidth=1)




#3 minutes to run below
indices_names_monthly = ['VHI','SPEI-1','SPEI-6','SPEI-12']
var_names_monthly = ['ET', 'ET_P', 'GW','LST','NDVI','PPT_CHIRPS','PPT_GPM','RZ','Runoff', 'Runoff base', 'SMAP','SM_Surf']

#Q: 2003-02 thru 2022-07
clm_r_anom_month = monthly_anom(np.log(clm_r+0.01)).mean(dim={'x','y'})
#P: 2002-02 thru 2022-10
gpm_p_anom_month = monthly_anom(np.log(p_gpm.P_mm+1)).mean(dim={'x','y'})[1:]

linear_plot(gpm_p_anom_month[12:-3],clm_r_anom_month,'P','Q')    

#ET: 2002-02 thru 2022-09
linear_plot(spatial_mean_anomalies_vars_monthly[0][12:-2],clm_r_anom_month,'ET','Q')    
linear_plot(spatial_mean_anomalies_vars_monthly[0],gpm_p_anom_month[:-1],'ET','P')  

#VHI: 2002-07 thru 2022-05
linear_plot(spatial_mean_anomalies_indic_monthly[0],spatial_mean_anomalies_vars_monthly[0][5:-4],'VHI','ET')
linear_plot(spatial_mean_anomalies_indic_monthly[0],gpm_p_anom_month[5:-5],'VHI','P')
linear_plot(spatial_mean_anomalies_indic_monthly[0][7:],clm_r_anom_month[:-2],'VHI','Q')

#SPEI: 2004-02 thru 2021-12 (but must start 12:)
linear_plot(spatial_mean_anomalies_vars_monthly[0][24:-9],spatial_mean_anomalies_indic_monthly[2][12:],'ET','SPEI-12')
linear_plot(gpm_p_anom_month[12+12:-10],spatial_mean_anomalies_indic_monthly[2][12:],'P','SPEI-12')
linear_plot(clm_r_anom_month[12:-7],spatial_mean_anomalies_indic_monthly[2][12:],'Q','SPEI-12')

#GW: 2002-02 thru 2022-04
linear_plot(spatial_mean_anomalies_vars_monthly[2],clm_r_anom_month[:-3],'GW','Q')  
linear_plot(spatial_mean_anomalies_vars_monthly[2],gpm_p_anom_month[12:-6],'GW','P')
linear_plot(spatial_mean_anomalies_vars_monthly[2],spatial_mean_anomalies_vars_monthly[0][12:-5],'GW','ET')

#SM: 2002-02 thru 2022-04
linear_plot(spatial_mean_anomalies_vars_monthly[-1],clm_r_anom_month[:-3],'SM','Q')  
linear_plot(spatial_mean_anomalies_vars_monthly[-1],gpm_p_anom_month[12:-6],'SM','P')
linear_plot(spatial_mean_anomalies_vars_monthly[-1],spatial_mean_anomalies_vars_monthly[0][12:-5],'SM','ET')

#RZ: 2002-02 thru 2022-04
linear_plot(spatial_mean_anomalies_vars_monthly[-5],clm_r_anom_month[:-3],'RZ','Q')  
linear_plot(spatial_mean_anomalies_vars_monthly[-5],gpm_p_anom_month[12:-6],'RZSM','P')
linear_plot(spatial_mean_anomalies_vars_monthly[-5],spatial_mean_anomalies_vars_monthly[0][12:-5],'RZ','ET')

#NDVI: 2002-07 thru 2022-05
linear_plot(spatial_mean_anomalies_vars_monthly[4],spatial_mean_anomalies_vars_monthly[0][5:-4],'NDVI','ET')
linear_plot(spatial_mean_anomalies_vars_monthly[4],gpm_p_anom_month[5:-5],'NDVI','P')
linear_plot(spatial_mean_anomalies_vars_monthly[4][7:],clm_r_anom_month[:-2],'NDVI','Q')
linear_plot(spatial_mean_anomalies_vars_monthly[4][8:],clm_r_anom_month[:-3],'NDVI','Q')







linear_plot(spatial_mean_anomalies_indic_monthly[0][7+12:-5],spatial_mean_anomalies_indic_monthly[2][12:],'VHI','SPEI-12')
linear_plot(spatial_mean_anomalies_vars_monthly[4][19:-5],spatial_mean_anomalies_indic_monthly[2][12:],'NDVI','SPEI-12')
linear_plot(spatial_mean_anomalies_vars_monthly[2][12:-4],spatial_mean_anomalies_indic_monthly[2][12:],'CLSM GW','SPEI-12')
linear_plot(spatial_mean_anomalies_vars_monthly[11][12:-4],spatial_mean_anomalies_indic_monthly[2][12:],'CLSM SM','SPEI-12')


linear_plot(spatial_mean_anomalies_indic_monthly[0][7:-5],spatial_mean_anomalies_vars_monthly[2][:-4],'VHI','CLSM GW')
linear_plot(spatial_mean_anomalies_vars_monthly[4][7:-5],spatial_mean_anomalies_vars_monthly[2][:-4],'NDVI','CLSM GW')
linear_plot(spatial_mean_anomalies_vars_monthly[11],spatial_mean_anomalies_vars_monthly[2],'CLSM SM','CLSM GW')

linear_plot(spatial_mean_anomalies_vars_monthly[11][:-4],spatial_mean_anomalies_indic_monthly[0][7:-5],'CLSM SM','VHI')
linear_plot(spatial_mean_anomalies_vars_monthly[4][7:-5],spatial_mean_anomalies_vars_monthly[11][:-4],'NDVI','CLSM SM')
linear_plot(spatial_mean_anomalies_vars_monthly[4][7:-5],spatial_mean_anomalies_indic_monthly[0],'NDVI','VHI')


linear_plot(spatial_mean_anomalies_vars_monthly[7],spatial_mean_anomalies_vars_monthly[2],'CLSM RZ','CLSM GW')
linear_plot(spatial_mean_anomalies_vars_monthly[7],spatial_mean_anomalies_vars_monthly[11],'CLSM RZ','CLSM SM')
linear_plot(spatial_mean_anomalies_vars_monthly[4][7:-5],spatial_mean_anomalies_vars_monthly[7][:-4],'NDVI','CLSM RZ')
linear_plot(spatial_mean_anomalies_vars_monthly[7][:-4],spatial_mean_anomalies_indic_monthly[0],'CLSM RZ','VHI')
linear_plot(spatial_mean_anomalies_vars_monthly[7][12:-4],spatial_mean_anomalies_indic_monthly[2][12:],'CLSM RZ','SPEI-12')


linear_plot(spatial_mean_anomalies_vars_growing[1],spatial_mean_anomalies_vars_growing[2],'GW','LST')
linear_plot(spatial_mean_anomalies_vars_growing[1],spatial_mean_anomalies_vars_growing[3],'GW','NDVI')
linear_plot(spatial_mean_anomalies_vars_growing[1],spatial_mean_anomalies_vars_growing[7],'GW','Q')

linear_plot(spatial_mean_anomalies_vars_growing[4],spatial_mean_anomalies_vars_growing[5],'PPT_GPM','PPT_CHIRPS')
linear_plot(spatial_mean_anomalies_vars_growing[0],spatial_mean_anomalies_vars_growing[3],'ET','NDVI')





#Mann-Kendall

def MK_pix (dataset):
    dataset = dataset.where(~np.isnan(dataset),0).interpolate_na(dim='time')
    y = dataset.y
    x = dataset.x

    '''
    0: trend?
    1: True or False (trend)
    2: p-value
    3: normalized statistic
    4: Kendall tau parameter
    5: MK score
    6: variance of MK score
    7: slope (Theil-Sen)
    8: intercept of Kendall-Theil robust line
    '''
    mk_dataset = xr.DataArray(np.apply_along_axis(mk.seasonal_test, axis=0, arr=dataset),
                                dims=('MK_stat','y','x'),
                                coords={'x':x,'y':y})
    #mk_dataset = mk_dataset.transpose('MK_stat','y','x')

    return mk_dataset

# hours... to run
indices_names = ['VHI','SPEI-3','SPEI-12','SPEI-1']
MK_test_indices = [MK_pix(ds) for ds in indices_datasets]

var_names = ['ET', 'ET_P','GW','LST','NDVI','PPT_CHIRPS','PPT_GPM','RZ','SMAP','SM_Surf']
MK_test_vars = [MK_pix(ds) for ds in var_datasets]

g_mk = MK_pix(var_datasets[2])
ndvi_mk = MK_pix(var_datasets[4])

sm_mk = MK_pix(var_datasets[-1])
sm_mk.to_netcdf(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\MK_tests\sm_mk.np')
lst_mk = MK_pix(var_datasets[3])
lst_mk.to_netcdf(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\MK_tests\lst_mk.np')
ppt_gpm_mk = MK_pix(var_datasets[6])
ppt_gpm_mk.to_netcdf(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\MK_tests\ppt_gpm_mk.np')
vhi_mk = MK_pix(indices_datasets[0])
vhi_mk.to_netcdf(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\MK_tests\vhi_mk.np')

del sm_mk,lst_mk,ppt_gpm_mk,vhi_mk
import gc
gc.collect()

spei12_mk = MK_pix(indices_datasets[2])
spei12_mk.to_netcdf(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\MK_tests\spei12_mk.np')

