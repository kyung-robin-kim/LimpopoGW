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


########################
#GRACE
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


#RHS_GPM.plot()
grace_limpopo_new_xr = xr.concat(grace_limpopo,dim='time').resample(time='1M').mean()
var_names = ['ET', 'GW','LST','NDVI','PPT_CHIRPS','PPT_GPM','RZ','Runoff','SMAP','SM_Surf']
clm_sm_rz,clm_r,sm_ts,clm_sm_surface

RHS_GPM = p_gpm['P_mm'].mean(dim={'x','y'}) - et_modis['ET_kg_m2'].mean(dim={'x','y'}) - clm_r.mean(dim={'x','y'})
RHS_CHIRPS = p_chirps['P_mm'].mean(dim={'x','y'}) - et_modis['ET_kg_m2'].mean(dim={'x','y'}) - clm_r.mean(dim={'x','y'})
 
CLSM_GW = (original_datasets[1].mean(dim={'x','y'}) - original_datasets[1].mean(dim={'x','y'}).mean())

CLSM_GW.plot()

grace_limpopo_new_xr = xr.concat(grace_limpopo,dim='time').resample(time='1M').mean()


plt.figure(figsize=(6,4))
CLSM_GW.plot(label='CLSM')
plt.plot(grace_limpopo_new_xr.time, grace_limpopo_new_xr.mean(dim={'lat','lon'})*1000,label='GRACE',color='darkblue')
plt.xlim(pd.to_datetime('2003-02-01'),pd.to_datetime('2022-07-01'))
plt.axhline(0,color='black',linestyle='--',linewidth=0.5)
plt.title('Monthly Groundwater Anomalies',weight='bold',fontsize=15)
plt.xlabel('Date',weight='bold',fontsize=12)
plt.ylabel('LWE (mm)',weight='bold',fontsize=12)
plt.tight_layout()
plt.legend()
plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\anomalies\GRACE_CLSM.png',dpi=300)



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

spatial_mean_anomalies_vars_monthly = [var.mean(dim={'x','y'})[:-3] for var in var_datasets]
spatial_mean_anomalies_indic_monthly = [var.mean(dim={'x','y'}) for var in indices_datasets]
'''

#PLOT MONTHS 
vhi = xr.open_mfdataset(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies_V2\netcdfs\VHI\VHI.nc').VHI
original_datasets = [(et_modis['ET_kg_m2']),clm_gw,lst['LST_K'],ndvi['NDVI'],(p_gpm['P_mm']),clm_sm_rz,(clm_r),sm_ts,clm_sm_surface]

p_gpm = p_chirps

degree_sign = u"\N{DEGREE SIGN}"
#Summer - January 2016: 155
#Summer - January 2021: 215


i=155
smap_da = sm_2016_2021_monthly[2][0]
plt.rc('font', size = 40)
fig, (axA, axB) = plt.subplots(2,3,figsize=(45,20))
clm_gw[i].rename('mm').plot(ax=axA[0],cmap='YlGnBu', levels=[i for i in range(400,1000)])
ndvi['NDVI'][i+7].rename(' ').plot(ax=axA[1],cmap='YlGn',levels=[i for i in np.arange(0.25,0.95,0.05)])
smap_da.attrs['long_name'] = ('m{} m{}'.format(get_super('3'),get_super('-3')))
smap_da.plot(ax=axA[2],cmap='GnBu',levels=[i for i in np.arange(0.1,0.35,0.01)])
[axA[i].set_title('{}'.format(variable),fontsize=50,fontweight='bold') for i,variable in zip(range(0,3),['CLSM GW','MODIS NDVI', 'SMAP SM'])]
[axA[i].set_xlabel('') for i in (range(0,3))]
axA[0].set_ylabel('Latitude ({})'.format(degree_sign),weight='bold',fontsize=40)
axA[1].set_ylabel('')
axA[2].set_ylabel('')
[axA[i].grid(linewidth=0.5) for i in (range(0,3))]

spei12 = indices_datasets[2][i]
spei12.attrs['long_name'] = ' '
spei12.plot(ax=axB[0],cmap='RdBu', levels=[i for i in np.arange(-3.1,3.1,0.1)])
vhi[i+7].rename(' ').plot(ax=axB[1],cmap='BrBG',levels=[i for i in range(0,100)])
p_gpm['P_mm'][i+13].rename('mm').plot(ax=axB[2],cmap='Blues', levels=[i for i in range(0,250)])
[axB[i].set_title('{}'.format(variable),fontsize=50,fontweight='bold') for i,variable in zip(range(0,3),['SPEI-12','VHI', 'CHIRPS P'])]
[axB[i].set_xlabel('Longitude ({})'.format(degree_sign),weight='bold',fontsize=40) for i in range(0,3)]
axB[0].set_ylabel('Latitude ({})'.format(degree_sign),weight='bold',fontsize=40)
axB[1].set_ylabel('')
axB[2].set_ylabel('')
[axB[i].grid(linewidth=0.5) for i in (range(0,3))]
plt.tight_layout()
plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\maps\multiplot_16_chirps.png',dpi=200)


i=215
smap_da = sm_2021_2022_monthly[0][0]
plt.rc('font', size = 40)
fig, (axA, axB) = plt.subplots(2,3,figsize=(45,20))
clm_gw[i].rename('mm').plot(ax=axA[0],cmap='YlGnBu', levels=[i for i in range(400,1000)])
ndvi['NDVI'][i+7].rename(' ').plot(ax=axA[1],cmap='YlGn',levels=[i for i in np.arange(0.25,0.95,0.05)])
smap_da.attrs['long_name'] = ('m{} m{}'.format(get_super('3'),get_super('-3')))
smap_da.plot(ax=axA[2],cmap='GnBu',levels=[i for i in np.arange(0.1,0.35,0.01)])
[axA[i].set_title('{}'.format(variable),fontsize=50,fontweight='bold') for i,variable in zip(range(0,3),['CLSM GW','MODIS NDVI', 'SMAP SM'])]
[axA[i].set_xlabel('') for i in (range(0,3))]
axA[0].set_ylabel('Latitude ({})'.format(degree_sign),weight='bold',fontsize=40)
axA[1].set_ylabel('')
axA[2].set_ylabel('')
[axA[i].grid(linewidth=0.5) for i in (range(0,3))]

spei12 = indices_datasets[2][i]
spei12.attrs['long_name'] = ' '
spei12.plot(ax=axB[0],cmap='RdBu', levels=[i for i in np.arange(-3.1,3.1,0.1)])
vhi[i+7].rename(' ').plot(ax=axB[1],cmap='BrBG',levels=[i for i in range(0,100)])
p_gpm['P_mm'][i+13].rename('mm').plot(ax=axB[2],cmap='Blues', levels=[i for i in range(0,250)])
[axB[i].set_title('{}'.format(variable),fontsize=50,fontweight='bold') for i,variable in zip(range(0,3),['SPEI-12','VHI', 'CHIRPS P'])]
[axB[i].set_xlabel('Longitude ({})'.format(degree_sign),weight='bold',fontsize=40) for i in range(0,3)]
axB[0].set_ylabel('Latitude ({})'.format(degree_sign),weight='bold',fontsize=40)
axB[1].set_ylabel('')
axB[2].set_ylabel('')
[axB[i].grid(linewidth=0.5) for i in (range(0,3))]
plt.tight_layout()
plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\maps\multiplot_21_chirps.png',dpi=200)


i=155+12+12+12+12
smap_da = sm_2016_2021_monthly[2+12+12+12+12][0]
plt.rc('font', size = 40)
fig, (axA, axB) = plt.subplots(2,3,figsize=(45,20))
clm_gw[i].rename('mm').plot(ax=axA[0],cmap='YlGnBu', levels=[i for i in range(400,1000)])
ndvi['NDVI'][i+7].rename(' ').plot(ax=axA[1],cmap='YlGn',levels=[i for i in np.arange(0.25,0.95,0.05)])
smap_da.attrs['long_name'] = ('m{} m{}'.format(get_super('3'),get_super('-3')))
smap_da.plot(ax=axA[2],cmap='GnBu',levels=[i for i in np.arange(0.1,0.35,0.01)])
[axA[i].set_title('{}'.format(variable),fontsize=50,fontweight='bold') for i,variable in zip(range(0,3),['CLSM GW','MODIS NDVI', 'SMAP SM'])]
[axA[i].set_xlabel('') for i in (range(0,3))]
axA[0].set_ylabel('Latitude ({})'.format(degree_sign),weight='bold',fontsize=40)
axA[1].set_ylabel('')
axA[2].set_ylabel('')
[axA[i].grid(linewidth=0.5) for i in (range(0,3))]

spei12 = indices_datasets[2][i]
spei12.attrs['long_name'] = ' '
spei12.plot(ax=axB[0],cmap='RdBu', levels=[i for i in np.arange(-3.1,3.1,0.1)])
vhi[i+7].rename(' ').plot(ax=axB[1],cmap='BrBG',levels=[i for i in range(0,100)])
p_gpm['P_mm'][i+13].rename('mm').plot(ax=axB[2],cmap='Blues', levels=[i for i in range(0,250)])
[axB[i].set_title('{}'.format(variable),fontsize=50,fontweight='bold') for i,variable in zip(range(0,3),['SPEI-12','VHI', 'CHIRPS P'])]
[axB[i].set_xlabel('Longitude ({})'.format(degree_sign),weight='bold',fontsize=40) for i in range(0,3)]
axB[0].set_ylabel('Latitude ({})'.format(degree_sign),weight='bold',fontsize=40)
axB[1].set_ylabel('')
axB[2].set_ylabel('')
[axB[i].grid(linewidth=0.5) for i in (range(0,3))]
plt.tight_layout()
plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\maps\multiplot_20_chirps.png',dpi=200)



i=155 - (12*8)
plt.rc('font', size = 40)
fig, (axA, axB) = plt.subplots(2,3,figsize=(45,20))
clm_gw[i].rename('mm').plot(ax=axA[0],cmap='YlGnBu', levels=[i for i in range(400,1000)])
ndvi['NDVI'][i+7].rename(' ').plot(ax=axA[1],cmap='YlGn',levels=[i for i in np.arange(0.25,0.95,0.05)])
clm_sm_surface[i].plot(ax=axA[2],cmap='GnBu',levels=[i for i in np.arange(0.1,9,0.5)])
[axA[i].set_title('{}'.format(variable),fontsize=50,fontweight='bold') for i,variable in zip(range(0,3),['CLSM GW','MODIS NDVI', 'CLSM SM'])]
[axA[i].set_xlabel('') for i in (range(0,3))]
axA[0].set_ylabel('Latitude ({})'.format(degree_sign),weight='bold',fontsize=40)
axA[1].set_ylabel('')
axA[2].set_ylabel('')
[axA[i].grid(linewidth=0.5) for i in (range(0,3))]

spei12 = indices_datasets[2][i]
spei12.attrs['long_name'] = ' '
spei12.plot(ax=axB[0],cmap='RdBu', levels=[i for i in np.arange(-3.1,3.1,0.1)])
vhi[i+7].rename(' ').plot(ax=axB[1],cmap='BrBG',levels=[i for i in range(0,100)])
p_gpm['P_mm'][i+13].rename('mm').plot(ax=axB[2],cmap='Blues', levels=[i for i in range(0,250)])
[axB[i].set_title('{}'.format(variable),fontsize=50,fontweight='bold') for i,variable in zip(range(0,3),['SPEI-12','VHI', 'CHIRPS P'])]
[axB[i].set_xlabel('Longitude ({})'.format(degree_sign),weight='bold',fontsize=40) for i in range(0,3)]
axB[0].set_ylabel('Latitude ({})'.format(degree_sign),weight='bold',fontsize=40)
axB[1].set_ylabel('')
axB[2].set_ylabel('')
[axB[i].grid(linewidth=0.5) for i in (range(0,3))]
plt.tight_layout()
plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\maps\multiplot_08_chirps.png',dpi=200)


i=155 - (12*10)
plt.rc('font', size = 40)
fig, (axA, axB) = plt.subplots(2,3,figsize=(45,20))
clm_gw[i].rename('mm').plot(ax=axA[0],cmap='YlGnBu', levels=[i for i in range(400,1000)])
ndvi['NDVI'][i+7].rename(' ').plot(ax=axA[1],cmap='YlGn',levels=[i for i in np.arange(0.25,0.95,0.05)])
clm_sm_surface[i].plot(ax=axA[2],cmap='GnBu',levels=[i for i in np.arange(0.1,9,0.5)])
[axA[i].set_title('{}'.format(variable),fontsize=50,fontweight='bold') for i,variable in zip(range(0,3),['CLSM GW','MODIS NDVI', 'CLSM SM'])]
[axA[i].set_xlabel('') for i in (range(0,3))]
axA[0].set_ylabel('Latitude ({})'.format(degree_sign),weight='bold',fontsize=40)
axA[1].set_ylabel('')
axA[2].set_ylabel('')
[axA[i].grid(linewidth=0.5) for i in (range(0,3))]

spei12 = indices_datasets[2][i]
spei12.attrs['long_name'] = ' '
spei12.plot(ax=axB[0],cmap='RdBu', levels=[i for i in np.arange(-3.1,3.1,0.1)])
vhi[i+7].rename(' ').plot(ax=axB[1],cmap='BrBG',levels=[i for i in range(0,100)])
p_gpm['P_mm'][i+13].rename('mm').plot(ax=axB[2],cmap='Blues', levels=[i for i in range(0,250)])
[axB[i].set_title('{}'.format(variable),fontsize=50,fontweight='bold') for i,variable in zip(range(0,3),['SPEI-12','VHI', 'CHIRPS P'])]
[axB[i].set_xlabel('Longitude ({})'.format(degree_sign),weight='bold',fontsize=40) for i in range(0,3)]
axB[0].set_ylabel('Latitude ({})'.format(degree_sign),weight='bold',fontsize=40)
axB[1].set_ylabel('')
axB[2].set_ylabel('')
[axB[i].grid(linewidth=0.5) for i in (range(0,3))]
plt.tight_layout()
plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\maps\multiplot_06_chirps.png',dpi=200)


i=215+12
smap_da = sm_2021_2022_monthly[0+12][0]
plt.rc('font', size = 40)
fig, (axA, axB) = plt.subplots(2,3,figsize=(45,20))
clm_gw[i].rename('mm').plot(ax=axA[0],cmap='YlGnBu', levels=[i for i in range(400,1000)])
ndvi['NDVI'][i+7].rename(' ').plot(ax=axA[1],cmap='YlGn',levels=[i for i in np.arange(0.25,0.95,0.05)])
smap_da.attrs['long_name'] = ('m{} m{}'.format(get_super('3'),get_super('-3')))
smap_da.plot(ax=axA[2],cmap='GnBu',levels=[i for i in np.arange(0.1,0.35,0.01)])
[axA[i].set_title('{}'.format(variable),fontsize=50,fontweight='bold') for i,variable in zip(range(0,3),['CLSM GW','MODIS NDVI', 'SMAP SM'])]
[axA[i].set_xlabel('') for i in (range(0,3))]
axA[0].set_ylabel('Latitude ({})'.format(degree_sign),weight='bold',fontsize=40)
axA[1].set_ylabel('')
axA[2].set_ylabel('')
[axA[i].grid(linewidth=0.5) for i in (range(0,3))]

spei12 = indices_datasets[2][-1]
spei12.attrs['long_name'] = ' '
spei12.plot(ax=axB[0],cmap='RdBu', levels=[i for i in np.arange(-3.1,3.1,0.1)])
vhi[i+7].rename(' ').plot(ax=axB[1],cmap='BrBG',levels=[i for i in range(0,100)])
p_gpm['P_mm'][i+13].rename('mm').plot(ax=axB[2],cmap='Blues', levels=[i for i in range(0,250)])
[axB[i].set_title('{}'.format(variable),fontsize=50,fontweight='bold') for i,variable in zip(range(0,3),['SPEI-12','VHI', 'CHIRPS P'])]
[axB[i].set_xlabel('Longitude ({})'.format(degree_sign),weight='bold',fontsize=40) for i in range(0,3)]
axB[0].set_ylabel('Latitude ({})'.format(degree_sign),weight='bold',fontsize=40)
axB[1].set_ylabel('')
axB[2].set_ylabel('')
[axB[i].grid(linewidth=0.5) for i in (range(0,3))]
plt.tight_layout()
plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\maps\multiplot_22_chirps.png',dpi=200)












#PLOT Mann Kendall (Seasonal Trends)
files = sorted(glob.glob(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\mk\*.nc'))
mk_test_files = [xr.open_mfdataset(file).__xarray_dataarray_variable__ for file in files]


#Show p-values
for i in range(0,len(mk_test_files)):
    plt.figure()
    mk_test_files[i][2].astype(float).plot(vmax=0.051)


#Show s-statistic
for i in range(0,len(mk_test_files)):
    plt.figure()
    mk_test_files[i][5].astype(float).plot()




def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

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





#P - ET - R
RHS_GPM = original_datasets[4].mean(dim={'x','y'}) - original_datasets[0].mean(dim={'x','y'}) - original_datasets[7].mean(dim={'x','y'}) - original_datasets[-3].mean(dim={'x','y'})
RHS_CHIRPS = original_datasets[5].mean(dim={'x','y'}) - original_datasets[0].mean(dim={'x','y'}) - original_datasets[7].mean(dim={'x','y'}) - original_datasets[-3].mean(dim={'x','y'})

CLSM_GW = (original_datasets[1].mean(dim={'x','y'}) - original_datasets[1].mean(dim={'x','y'}).mean())


#Plotting All Datasets & Variables

for i in range(0,9):
    plt.figure()
    spatial_mean_anomalies_vars_growing[i].plot(label='{}'.format(var_names[i]))
    anomaly_datasets[i].plot()
    plt.legend()


var_names = ['MODIS ET', 'CLSM GW','MODIS LST','MODIS NDVI','CHIRPS P','GPM IMERG P','CLSM RZ SM','CLSM Q','SMAP Surface SM','CLSM SM']
for i in range(0,10):
    plt.figure(figsize=(6,4))
    plt.plot(anomaly_datasets_means[i].time,anomaly_datasets_means[i],linewidth=0.4,color='black')
    plt.plot(spatial_mean_anomalies_vars_growing[i].year_i,spatial_mean_anomalies_vars_growing[i],label='{}'.format(var_names[i]),color='black')
    plt.axhline(0,color='black',linestyle='--',linewidth=0.5)
    plt.ylabel('z-score',weight='bold')
    plt.xlabel('Date')
    plt.ylim(-3.2,3.2)
    plt.grid(axis='x')
    plt.xlim(pd.to_datetime('2003-02-01'),pd.to_datetime('2022-07-01'))
    plt.title(var_names[i],weight='bold')
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\anomalies\{}.png'.format(var_names[i]),dpi=300)




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
plt.title('Groundwater Anomalies',weight='bold',fontsize=15)
plt.xlabel('Date',weight='bold',fontsize=12)
plt.ylabel('z-score',weight='bold',fontsize=12)
plt.tight_layout()
plt.legend()

plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\gw_anomalies.png',dpi=300)



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
plt.title('Runoff/Discharge Anomalies',weight='bold',fontsize=15)
plt.xlabel('Date',weight='bold',fontsize=12)
plt.ylabel('z-score',weight='bold',fontsize=12)
plt.tight_layout()
plt.legend()
plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\runoff_anomalies_vD.png',dpi=300)




path = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies_V2\netcdfs\VHI'
files = sorted(glob.glob(path+'\*.nc'))
vhi = xr.open_mfdataset(files[0],parallel=True).VHI
vhi_index = (vhi.mean(dim={'x','y'}) - vhi.mean(dim={'x','y'}).mean())/vhi.mean(dim={'x','y'}).std(ddof=1)


plt.figure(figsize=(8,4))
#plt.plot(vhi_index.time,vhi_index,color='C1',label='VHI')
#plt.plot(vhi_index.rolling(time=12).mean().time,vhi_index.rolling(time=12).mean(),color='C1',linestyle='--',linewidth=1)
plt.plot(spatial_mean_anomalies_vars_monthly[2].time,spatial_mean_anomalies_vars_monthly[2],color='C0',label='CLSM GW')
#plt.plot(spatial_mean_anomalies_vars_monthly[4].time,spatial_mean_anomalies_vars_monthly[4],color='green',label='MODIS NDVI')
#plt.plot(spatial_mean_anomalies_vars_monthly[8].time,spatial_mean_anomalies_vars_monthly[8],color='red',label='CLSM Q')
#plt.plot(spatial_mean_anomalies_vars_monthly[6].time,spatial_mean_anomalies_vars_monthly[6],color='green',label='GPM P')
plt.plot(spatial_mean_anomalies_vars_monthly[7].time,spatial_mean_anomalies_vars_monthly[7],color='green',label='CLSM RZ')
plt.plot(spatial_mean_anomalies_indic_monthly[0].time,spatial_mean_anomalies_indic_monthly[0],color='C1',label='VHI')
#plt.plot(spatial_mean_anomalies_indic_monthly[1].time,spatial_mean_anomalies_indic_monthly[1],color='black',linewidth=0.5)
plt.plot(spatial_mean_anomalies_indic_monthly[2].time,spatial_mean_anomalies_indic_monthly[2],color='black',label='SPEI')
#plt.plot(vhi_index.time,vhi_index-spatial_mean_anomalies_indic_monthly[2],color='black',linestyle=':',label='diff')
plt.title('Drought Indices',weight='bold',fontsize=15)
plt.legend(loc='lower center')
plt.ylim(-3,3)
plt.axhline(0,color='black',linestyle='--',linewidth=0.5)
plt.grid(axis='x')
plt.ylabel('z-score',weight='bold',fontsize=12)
plt.xlim(pd.to_datetime('2003-02-01'),pd.to_datetime('2022-07-01'))
plt.xlabel('Date',weight='bold',fontsize=12)
plt.tight_layout()



plt.figure(figsize=(6,4))
spatial_mean_anomalies_indic_growing[0].plot(label='VHI')
#spatial_mean_anomalies_indic_growing[1].plot(label='SPEI-3')
spatial_mean_anomalies_indic_growing[2].plot(label='SPEI-12')
#spatial_mean_anomalies_indic_growing[3].plot(label='SPEI-1')
plt.title('Drought Indices')
plt.legend()
plt.ylim(-3.2,3.2)
plt.axhline(0,color='black',linestyle='--',linewidth=0.5)
plt.grid(axis='x')
plt.xlim(pd.to_datetime('2003-02-01'),pd.to_datetime('2022-07-01'))



plt.figure(figsize=(6,4))
#spatial_mean_anomalies_indic_growing[0].plot(label='VHI')
#spatial_mean_anomalies_indic_growing[1].plot(label='SPEI-3')
#spatial_mean_anomalies_indic_growing[2].plot(label='SPEI-12')
#spatial_mean_anomalies_indic_growing[3].plot(label='SPEI-1')
#spatial_mean_anomalies_vars_growing[0].plot(label='MODIS ET')
#spatial_mean_anomalies_vars_growing[1].plot(label='CLSM GW')
spatial_mean_anomalies_vars_growing[2].plot(label='MODIS LST')
spatial_mean_anomalies_vars_growing[3].plot(label='MODIS NDVI')
spatial_mean_anomalies_vars_growing[5].plot(label='GPM PPT')
spatial_mean_anomalies_vars_growing[6].plot(label='CHIRPS PPT')

#spatial_mean_anomalies_vars_growing[7].plot(label='CLSM Runoff')
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
    ax.plot([], [], ' ', label='p-val = {}'.format(round(p_value, 5)))
    ax.plot([], [], ' ', label='r: {}'.format(round(r_value,5)))
    #ax.set_xlim(-2,2)
    #ax.set_ylim(-2,2)
    ax.legend(loc='upper right')
    ax.set_title('{} vs. {}'.format(var1,var2))

#Monthly-monthly anomalies
'''
['C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\ET_1km.nc',
 'C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\ET_P_1km.nc',
 'C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\GW.nc',
 'C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\LST.nc',
 'C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\NDVI.nc',
 'C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\PPT_CHIRPS.nc',
 'C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\PPT_GPM.nc',
 'C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\RZ.nc',
 'C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\Runoff.nc',
 'C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\Runoff_base.nc',
 'C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\SMAP.nc',
 'C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\Surface SM.nc']
'''

linear_plot(spatial_mean_anomalies_indic_monthly[0][12:],spatial_mean_anomalies_indic_monthly[2][12:],'VHI','SPEI-12')
linear_plot(spatial_mean_anomalies_vars_monthly[4][19:-5],spatial_mean_anomalies_indic_monthly[2][12:],'NDVI','SPEI-12')
linear_plot(spatial_mean_anomalies_vars_monthly[2][12:-4],spatial_mean_anomalies_indic_monthly[2][12:],'CLSM GW','SPEI-12')
linear_plot(spatial_mean_anomalies_vars_monthly[11][12:-4],spatial_mean_anomalies_indic_monthly[2][12:],'CLSM SM','SPEI-12')


linear_plot(spatial_mean_anomalies_indic_monthly[0],spatial_mean_anomalies_vars_monthly[2][:-4],'VHI','CLSM GW')
linear_plot(spatial_mean_anomalies_vars_monthly[4][7:-5],spatial_mean_anomalies_vars_monthly[2][:-4],'NDVI','CLSM GW')
linear_plot(spatial_mean_anomalies_vars_monthly[11],spatial_mean_anomalies_vars_monthly[2],'CLSM SM','CLSM GW')

linear_plot(spatial_mean_anomalies_vars_monthly[11][:-4],spatial_mean_anomalies_indic_monthly[0],'CLSM SM','VHI')
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
indices_names = ['VHI','SPEI-3','SPEI-12','SPEI-1']
var_names = ['ET','GW','LST','NDVI','PPT_GPM','RZ','SMAP','SM_Surf']


linear_plot(spatial_mean_anomalies_vars[3],spatial_mean_anomalies_indic[2],'NDVI','SPEI-12') #NDVI vs. SPEI-12    
linear_plot(spatial_mean_anomalies_vars[3],spatial_mean_anomalies_indic[1],'NDVI','SPEI-3')  #NDVI vs. SPEI-3
linear_plot(spatial_mean_anomalies_vars[3],spatial_mean_anomalies_indic[3],'NDVI', 'SPEI-1')  #NDVI vs. SPEI-1
linear_plot(spatial_mean_anomalies_vars[3],spatial_mean_anomalies_indic[0],'NDVI','VHI') #NDVI vs. VHI (most)

linear_plot(spatial_mean_anomalies_vars[-1],spatial_mean_anomalies_indic[2],'CLSM SM','SPEI-12')  #CLSM SM vs. SPEI-12
linear_plot(spatial_mean_anomalies_vars[-1],spatial_mean_anomalies_indic[1],'CLSM SM', 'SPEI-3')  #CLSM SM vs. SPEI-3
linear_plot(spatial_mean_anomalies_vars[-1],spatial_mean_anomalies_indic[0], 'CLSM SM', 'VHI') #CLSM SM vs. VHI (most)
linear_plot(spatial_mean_anomalies_vars[-1],spatial_mean_anomalies_indic[3],'CLSM SM', 'SPEI-1')  #CLSM SM vs. SPEI-1

linear_plot(spatial_mean_anomalies_vars[1],spatial_mean_anomalies_indic[2], 'GW','SPEI-12') #GW vs. SPEI-12 (most)
linear_plot(spatial_mean_anomalies_vars[1],spatial_mean_anomalies_indic[1], 'GW',' SPEI-3')  #GW vs. SPEI-3
linear_plot(spatial_mean_anomalies_vars[1],spatial_mean_anomalies_indic[0],'GW','VHI')  #GW vs. VHI
linear_plot(spatial_mean_anomalies_vars[1],spatial_mean_anomalies_indic[3],'GW','SPEI-1')  #GW vs. SPEI-1

linear_plot(spatial_mean_anomalies_indic[0],spatial_mean_anomalies_indic[2], 'VHI','SPEI-12') #VHI vs. SPEI-12 
linear_plot(spatial_mean_anomalies_indic[0],spatial_mean_anomalies_indic[1],'VHI','SPEI-3')  #VHI vs. SPEI-3 (most)
linear_plot(spatial_mean_anomalies_indic[0],spatial_mean_anomalies_indic[3],'VHI','SPEI-1')  #VHI vs. SPEI-1
linear_plot(spatial_mean_anomalies_indic[1],spatial_mean_anomalies_indic[2],'SPEI-3','SPEI-12')  #SPEI-3 vs. SPEI-12 (less correlation than expected)

linear_plot(spatial_mean_anomalies_vars[4],spatial_mean_anomalies_indic[2],'GPM','SPEI-12') 
linear_plot(spatial_mean_anomalies_vars[4],spatial_mean_anomalies_indic[1],'GPM', 'SPEI-3') 
linear_plot(spatial_mean_anomalies_vars[4],spatial_mean_anomalies_indic[0], 'GPM', 'VHI')
linear_plot(spatial_mean_anomalies_vars[4],spatial_mean_anomalies_indic[3],'GPM', 'SPEI-1') 


linear_plot(spatial_mean_anomalies_vars_growing[1],spatial_mean_anomalies_vars_growing[3],'CLSM GW','NDVI')

indices_names = ['VHI','SPEI-3','SPEI-12','SPEI-1']
var_names = ['ET','GW','LST','NDVI','PPT_GPM','RZ','SMAP','SM_Surf']

#In-Situ Comparisons
dataset_anomaly_wet_annual_df_gw.plot()
dataset_anomaly_wet_annual_df_r.plot()

linear_plot(spatial_mean_anomalies_vars[1],spatial_mean_anomalies_vars[3],'CLSM GW','NDVI')



linear_plot(dataset_anomaly_wet_annual_df_gw['Growing Season Anomalies'][1:],spatial_mean_anomalies_vars[1],'InSitu GW','CLSM GW')
linear_plot(dataset_anomaly_wet_annual_df_r['Growing Season Anomalies'][1:],spatial_mean_anomalies_vars[3],'InSitu R','NDVI')
linear_plot(dataset_anomaly_wet_annual_df_r['Growing Season Anomalies'][1:],spatial_mean_anomalies_vars[4],'InSitu R','GPM P')
linear_plot(dataset_anomaly_wet_annual_df_r['Growing Season Anomalies'][1:],spatial_mean_anomalies_vars[0],'InSitu R','MODIS ET')


linear_plot(dataset_anomaly_wet_annual_df_gw['Growing Season Anomalies'],dataset_anomaly_wet_annual_df_r['Growing Season Anomalies'],'InSitu GW','InSitu Q')

linear_plot(monthly_anomalies_gw.mean(axis=1),spatial_mean_anomalies_vars[4],'InSitu GW','NDVI')
linear_plot(monthly_anomalies_q.mean(axis=1),spatial_mean_anomalies_vars[4],'InSitu Q','NDVI')
linear_plot(monthly_anomalies_q.mean(axis=1),spatial_mean_anomalies_vars[-1],'InSitu Q','CLSM SM')
linear_plot(monthly_anomalies_gw.mean(axis=1)[11:],spatial_mean_anomalies_indic[2][11:],'InSitu GW','SPEI-12')
linear_plot(monthly_anomalies_q.mean(axis=1),spatial_mean_anomalies_indic[0],'InSitu Q','VHI')
linear_plot(monthly_anomalies_q.mean(axis=1)[0:],spatial_mean_anomalies_indic[3][0:],'InSitu Q','SPEI-1')

spatial_mean_anomalies_indic[3].plot() #SPEI-1
spatial_mean_anomalies_vars[-1].plot() #CLSM SM
spatial_mean_anomalies_indic[2].plot() #SPEI-12
spatial_mean_anomalies_vars[4].plot() #NDVI
spatial_mean_anomalies_indic[3].plot() #SPEI-1
monthly_anomalies_q.mean(axis=1).plot() #Q



#Mann-Kendall

def MK_pix (dataset):
    dataset = var_datasets[2]
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

MK_test_vars = [MK_pix(ds) for ds in var_datasets]
MK_test_indices = [MK_pix(ds) for ds in indices_datasets]

indices_names = ['VHI','SPEI-3','SPEI-12','SPEI-1']
MK_test_indices = [MK_pix(ds) for ds in indices_datasets]

var_names = ['ET', 'ET_P','GW','LST','NDVI','PPT_CHIRPS','PPT_GPM','RZ','SMAP','SM_Surf']
MK_test_vars = [MK_pix(ds) for ds in var_datasets]

g_mk = MK_pix(var_datasets[2])
ndvi_mk = MK_pix(var_datasets[4])
ppt_gpm_mk = MK_pix(var_datasets[6])
vhi_mk = MK_pix(indices_datasets[0])
spei12_mk = MK_pix(indices_datasets[2])

#gw_mk.to_netcdf(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\MK_tests\gw_mk.nc')
#ndvi_mk.to_netcdf(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\MK_tests\ndvi_mk.nc')
#vhi_mk.to_netcdf(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\MK_tests\vhi_mk.nc')
#spei12_mk.to_netcdf(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\MK_tests\spei12_mk.nc')
#ppt_gpm_mk.to_netcdf(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\MK_tests\ppt_gpm_mk.nc')

files = sorted(glob.glob(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\MK_tests\*.nc'))

mk_test_files = [xr.open_mfdataset(file).__xarray_dataarray_variable__ for file in files]

#Vectorizing doesn't work .......
#xr.apply_ufunc(mk.seasonal_test, var_datasets[2], input_core_dims=[['time','y','x']], output_core_dims=[['x','y']], vectorize=True,dask='parallelized')
#[(mk.seasonal_test(var,period=12),name) for var,name in zip(var_datasets,var_names)]




#NonParametric
#Mann-Kendall
[print(mk.seasonal_test(var,period=12),name) for var,name in zip(spatial_mean_anomalies_vars,var_names)]
[print(mk.seasonal_test(var,period=12),name) for var,name in zip(spatial_mean_anomalies_indic,indices_names)]
(mk.seasonal_test(monthly_anomalies_q.mean(axis=1),period=12))
(mk.seasonal_test(monthly_anomalies_gw.mean(axis=1),period=12))

#Spearman's rank correlation
[print(spearmanr(range(0,len(data)), data),var) for data,var in zip(spatial_mean_anomalies_vars,var_names)]
[print(spearmanr(range(0,len(data)), data),var) for data,var in zip(spatial_mean_anomalies_indic,indices_names)]

#spearmanr(range(0,len(spatial_mean_anomalies_vars[8])), spatial_mean_anomalies_vars[8].resample('1M').mean()) #SMAP
spearmanr(range(3,len(spatial_mean_anomalies_indic[1])), spatial_mean_anomalies_indic[1][3:]) #SPEI-3
spearmanr(range(12,len(spatial_mean_anomalies_indic_monthly[2])), spatial_mean_anomalies_indic_monthly[2][12:]) #SPEI-12
spearmanr(range(1,len(spatial_mean_anomalies_indic[3])), spatial_mean_anomalies_indic[3][1:]) #SPEI-1
(spearmanr(range(0,len(monthly_anomalies_q)), (monthly_anomalies_q.mean(axis=1)))) #insitu Q
(spearmanr(range(0,len(monthly_anomalies_gw)), (monthly_anomalies_gw.mean(axis=1)))) #insitu GW


#STATISTICAL TEST
#insitu
(mk.original_test(dataset_anomaly_wet_annual_df_gw)) #groundwater is depleteing? based on in-situ data (non-irrigated lands)
(spearmanr(range(0,len(dataset_anomaly_wet_annual_df_gw)), (dataset_anomaly_wet_annual_df_gw))) 
(mk.original_test(dataset_anomaly_wet_annual_df_r)) #no real trend with streamflow
(spearmanr(range(0,len(dataset_anomaly_wet_annual_df_r)), (dataset_anomaly_wet_annual_df_r)))


'''
['C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\ET_1km.nc',
 'C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\ET_P_1km.nc',
 'C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\GW.nc',
 'C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\LST.nc',
 'C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\NDVI.nc',
 'C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\PPT_CHIRPS.nc',
 'C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\PPT_GPM.nc',
 'C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\RZ.nc',
 'C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\Runoff.nc',
 'C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\Runoff_base.nc',
 'C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\SMAP.nc',
 'C:\\Users\\robin\\Desktop\\Limpopo_VegetationHydrology\\Data\\anomalies\\monthly\\netcdfs\\Surface SM.nc']
'''

monthly_anom_vars = spatial_mean_anomalies_vars_monthly
monthly_anom_vars = [monthly_anom_vars[ii] for ii in [0,2,3,4,6,7,8,11]]

for i in range(0,8):
    print(var_names[i])
    print((mk.original_test(spatial_mean_anomalies_vars_growing[i])))
    print((spearmanr(range(0,len(spatial_mean_anomalies_vars_growing[i])),spatial_mean_anomalies_vars_growing[i])))
    print(var_names[i],' regular')
    print((mk.original_test(spatial_mean_anomalies_vars_monthly[i])))
    print((spearmanr(range(0,len(spatial_mean_anomalies_vars_monthly[i])),spatial_mean_anomalies_vars_monthly[i])))
        
for i in range(0,4):
    print(indices_names[i],' growing')
    print((mk.original_test(spatial_mean_anomalies_indic_growing[i])))
    print((spearmanr(range(0,len(spatial_mean_anomalies_indic_growing[i])),spatial_mean_anomalies_indic_growing[i])))
    print(indices_names[i],' regular')
    print((mk.original_test(spatial_mean_anomalies_indic_monthly[i])))
    print((spearmanr(range(0,len(spatial_mean_anomalies_indic_monthly[i])),spatial_mean_anomalies_indic_monthly[i])))



plt.figure(figsize=(8,4))
plt.plot(spatial_mean_anomalies_vars_monthly[7].time,spatial_mean_anomalies_vars_monthly[7],color='black',label='CLSM RZ',linewidth=1)
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
plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\V2\NDVI_SoilMoisture.png',dpi=300)


#POINT-to-POINT (SPATIAL AVERAGE) COMPARISON
#https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

insitu_points_gw = [Point(xy) for xy in zip(limpopo_gw_insitu_filtered['Longitude'], limpopo_gw_insitu_filtered['Latitude'])]
insitu_gpd_gw = gpd.GeoDataFrame(limpopo_gw_insitu_filtered, geometry=insitu_points_gw).set_crs('EPSG:4326')
gw_lats = [point.y for point in insitu_points_gw]
gw_lons =  [point.x for point in insitu_points_gw]

[print(var.dims) for var in var_datasets]
var_names = ['ET', 'ET_P','GW','LST','NDVI','PPT_CHIRPS','PPT_GPM','RZ','SMAP','SM_Surf']

#In-Situ vs. Boreholes
query_points_gw = [[(find_nearest(var.y,lat)[1],find_nearest(var.x,lon)[1]) for lat,lon in zip(gw_lats,gw_lons)] for var in var_datasets]
var_points = np.array([[var[:,point[0],point[1]] for point in var_points] for var, var_points in zip(var_datasets,query_points_gw)])
var_points_avg = [xr.concat(all_points,dim='x').mean(dim='x') for all_points in var_points]

#....takes >10 minutes...why???
query_points_gw_indices = [[(find_nearest(var.y,lat)[1],find_nearest(var.x,lon)[1]) for lat,lon in zip(gw_lats,gw_lons)] for var in indices_datasets]
indic_points = np.array([[var[:,point[0],point[1]] for point in var_points] for var, var_points in zip(indices_datasets,query_points_gw_indices)])
indic_points_avg = [xr.concat(all_points,dim='x').mean(dim='x') for all_points in indic_points]


#NonParametric
#Mann-Kendall
[print(mk.seasonal_test(var,period=12),name) for var,name in zip(var_points_avg,var_names)]
(mk.seasonal_test(monthly_anomalies_q.mean(axis=1),period=12))
(mk.seasonal_test(monthly_anomalies_gw.mean(axis=1),period=12))

#Spearman's rank correlation
[print(spearmanr(range(0,len(data)), data),var) for data,var in zip(var_points_avg,var_names)]
(spearmanr(range(0,len(monthly_anomalies_q)), (monthly_anomalies_q.mean(axis=1))))
(spearmanr(range(0,len(monthly_anomalies_gw)), (monthly_anomalies_gw.mean(axis=1))))


#In-Situ Comparisons
linear_plot(monthly_anomalies_gw.mean(axis=1),var_points_avg[2],'InSitu GW','CLSM GW')
linear_plot(monthly_anomalies_gw.mean(axis=1),monthly_anomalies_q.mean(axis=1),'InSitu GW','InSitu Q')
linear_plot(monthly_anomalies_gw.mean(axis=1),var_points_avg[4],'InSitu GW','NDVI')
linear_plot(monthly_anomalies_q.mean(axis=1),var_points_avg[4],'InSitu Q','NDVI')
linear_plot(monthly_anomalies_q.mean(axis=1),var_points_avg[-1],'InSitu Q','CLSM SM')

var_points_avg[4].plot() #NDVI
var_points_avg[2].plot() #GW
var_points_avg[-1].plot() #SM
monthly_anomalies_gw.mean(axis=1).plot()
monthly_anomalies_q.mean(axis=1).rolling(4).mean().plot(linewidth=1)


import seaborn as sns
from scipy import stats

#Correlation Plots
#Takes about 2 minutes to run correlation_pix plots
#2 minutes to run KDE

def correlation_pix(independent, dependent, labelx, labely,title,savepath):

    nanmask = ~np.isnan(independent) & ~np.isnan(dependent)
    independent_mask = np.array(independent)[nanmask]
    dependent_mask = np.array(dependent)[nanmask]

    x=np.array(independent_mask)
    y=np.array(dependent_mask)

    '''
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    axis = [(-5,-5), (5,5)]
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    ax.plot([axis[0][0],axis[1][0]],[axis[0][1],axis[1][1]],'--',color='black',linewidth=1)
    if int(title[41:43]) in [4,5,6,7,8,9]: #DRY WINTER
        ax.scatter(independent_mask[0::1],dependent_mask[0::1],color='C1',s=5)
    else: #WET SUMMER
        ax.scatter(independent_mask[0::1],dependent_mask[0::1],color='C0',s=5)
    ax.plot([], [], ' ', label='p-val = {}'.format(round(p_value, 3)))
    ax.plot([], [], ' ', label='r: {}'.format(round(r_value,5)))
    ax.set_ylabel(labely,weight='bold')
    ax.set_xlabel(labelx,weight='bold')
    ax.set_xlim(-3.5,3.5)
    ax.set_ylim(-3.5,3.5)
    ax.set_title('{}'.format(title[36:46]),weight='bold')
    ax.legend(loc='lower right')
    plt.savefig(savepath+r'\{}.png'.format(title[36:46]))
    '''

    return x,y

def KDE_plot(x,y,days,label):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    plt.figure(figsize=(8,6))
    sns.kdeplot(x=x,y=y, cmap='Blues',  shade=True,cbar=True,thresh=0.01)
    plt.plot([-5,5],[-5,5],'--',color='black')
    plt.xlim(-3.5,3.5)
    plt.ylim(-3.5,3.5)
    plt.plot([], [], ' ', label='p-val = {}'.format(round(p_value, 3)))
    plt.plot([], [], ' ', label='r: {}'.format(round(r_value,5)))
    plt.legend()
    plt.savefig(savepath+r'\KDE_plot_{}.png'.format(label))

ndvi_4plot = var_datasets[4].rio.write_crs('epsg:4326').rio.reproject_match(var_datasets[2].rio.write_crs('epsg:4326'), resampling=rio.enums.Resampling.average)
vhi_4plot = indices_datasets[0].rio.write_crs('epsg:4326').rio.reproject_match(var_datasets[2].rio.write_crs('epsg:4326'), resampling=rio.enums.Resampling.average)
spei12_4plot = indices_datasets[2].rio.write_crs('epsg:4326').rio.reproject_match(var_datasets[2].rio.write_crs('epsg:4326'), resampling=rio.enums.Resampling.average)
et_4plot = var_datasets[0].rio.write_crs('epsg:4326').rio.reproject_match(var_datasets[2].rio.write_crs('epsg:4326'), resampling=rio.enums.Resampling.average)
ppt_4plot = var_datasets[6].rio.write_crs('epsg:4326').rio.reproject_match(var_datasets[2].rio.write_crs('epsg:4326'), resampling=rio.enums.Resampling.average)

savepath = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\anomalies_correlation\GW_SM'
points_GW_SM = [correlation_pix(var_datasets[2][i],var_datasets[-1][i],'CLSM GW','CLSM SM','{}'.format(date),savepath) for i,date in zip(range(0,len(ndvi_4plot)),ndvi_4plot.time)]
points_GW_SM = np.array(points_GW_SM)
points_GW_SM_x = np.array([points_GW_SM[i,0,:] for i in range(0,227)]).flatten()
points_GW_SM_y = np.array([points_GW_SM[i,1,:] for i in range(0,227)]).flatten()
KDE_plot(points_GW_SM_x,points_GW_SM_y,227,'GW_vs_SM')

savepath = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\anomalies_correlation\NDVI_SM'
points_NDVI_SM = [correlation_pix(ndvi_4plot[i],var_datasets[-1][i],'NDVI','CLSM SM','{}'.format(date),savepath) for i,date in zip(range(0,len(ndvi_4plot)),ndvi_4plot.time)]
points_NDVI_SM = np.array(points_NDVI_SM)
points_NDVI_SM_x = np.concatenate([points_NDVI_SM[i,0] for i in range(0,227)])
points_NDVI_SM_y = np.concatenate([points_NDVI_SM[i,1] for i in range(0,227)])
KDE_plot(points_NDVI_SM_x,points_NDVI_SM_y,227,'NDVI_vs_SM')

savepath = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\anomalies_correlation\NDVI_GW'
points_NDVI_GW = [correlation_pix(ndvi_4plot[i],var_datasets[2][i],'NDVI','CLSM GW','{}'.format(date),savepath) for i,date in zip(range(0,len(ndvi_4plot)),ndvi_4plot.time)]
points_NDVI_GW = np.array(points_NDVI_GW)
points_NDVI_GW_x = np.concatenate([points_NDVI_GW[i,0] for i in range(0,227)])
points_NDVI_GW_y = np.concatenate([points_NDVI_GW[i,1] for i in range(0,227)])
KDE_plot(points_NDVI_GW_x,points_NDVI_GW_y,227,'NDVI_vs_GW')

savepath = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\anomalies_correlation\SM_VHI'
points_SM_VHI = [correlation_pix(var_datasets[-1][i],vhi_4plot[i],'CLSM SM','VHI','{}'.format(date),savepath) for i,date in zip(range(0,len(vhi_4plot)),vhi_4plot.time)]
points_SM_VHI = np.array(points_SM_VHI)
points_SM_VHI_x = np.concatenate([points_SM_VHI[i,0] for i in range(0,227)])
points_SM_VHI_y = np.concatenate([points_SM_VHI[i,1] for i in range(0,227)])
KDE_plot(points_SM_VHI_x,points_SM_VHI_y,227,'SM_vs_VHI')

savepath = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\anomalies_correlation\GW_SPEI12'
points_GW_SPEI12 = [correlation_pix(var_datasets[2][i],spei12_4plot[i],'GW','SPEI-12','{}'.format(date),savepath) for i,date in zip(range(11,len(spei12_4plot)),spei12_4plot.time[11:])]
points_GW_SPEI12 = np.array(points_GW_SPEI12)
points_GW_SPEI12_x = np.concatenate([points_GW_SPEI12[i,0] for i in range(0,216)])
points_GW_SPEI12_y = np.concatenate([points_GW_SPEI12[i,1] for i in range(0,216)])
KDE_plot(points_GW_SPEI12_x,points_GW_SPEI12_y,227-11,'GW_vs_SPEI-12')

savepath = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\anomalies_correlation'
points = [correlation_pix(ndvi_4plot[i],vhi_4plot[i],'NDVI','VHI','{}'.format(date),savepath) for i,date in zip(range(0,len(vhi_4plot)),vhi_4plot.time)]
points = np.array(points)
x = np.concatenate([points[i,0] for i in range(0,227)])
y = np.concatenate([points[i,1] for i in range(0,227)])
KDE_plot(x,y,227,'NDVI_vs_VHI')

points = [correlation_pix(et_4plot[i],ndvi_4plot[i],'ET','NDVI','{}'.format(date),savepath) for i,date in zip(range(0,len(ndvi_4plot)),ndvi_4plot.time)]
points = np.array(points)
x = np.concatenate([points[i,0] for i in range(0,227)])
y = np.concatenate([points[i,1] for i in range(0,227)])
KDE_plot(x,y,227,'ET_vs_NDVI')

points = [correlation_pix(ppt_4plot[i],ndvi_4plot[i],'PPT','NDVI','{}'.format(date),savepath) for i,date in zip(range(0,len(ndvi_4plot)),ndvi_4plot.time)]
points = np.array(points)
x = np.concatenate([points[i,0] for i in range(0,227)])
y = np.concatenate([points[i,1] for i in range(0,227)])
KDE_plot(x,y,227,'PPT_vs_NDVI')

points = [correlation_pix(ppt_4plot[i],var_datasets[2][i],'PPT','GW','{}'.format(date),savepath) for i,date in zip(range(0,len(ppt_4plot)),ppt_4plot.time)]
points = np.array(points)
x = np.concatenate([points[i,0] for i in range(0,227)])
y = np.concatenate([points[i,1] for i in range(0,227)])
KDE_plot(x,y,227,'PPT_vs_GW')

points = [correlation_pix(ppt_4plot[i],var_datasets[-1][i],'PPT','SM','{}'.format(date),savepath) for i,date in zip(range(0,len(ppt_4plot)),ppt_4plot.time)]
points = np.array(points)
x = np.concatenate([points[i,0] for i in range(0,227)])
y = np.concatenate([points[i,1] for i in range(0,227)])
KDE_plot(x,y,227,'PPT_vs_SM')

#Ignore for now....
#By geology #18 categories
gw_anomaly_by_geology = [monthly_anomalies_gw.T[(geology.iloc[:,0]==index)] for index in np.unique(geology.iloc[:,0])]
gw_by_geology_df = [limpopo_gw_insitu_filtered[(geology.iloc[:,0]==index)] for index in np.unique(geology.iloc[:,0])]
geo_names = [geology.iloc[:,1][geology.iloc[:,0]==np.unique(geology.iloc[:,0])[i]].iloc[0] for i in range(0,18)]

insitu_points_gw_geo = [[Point(xy) for xy in zip(geology['Longitude'], geology['Latitude'])] for geology in gw_by_geology_df]
#insitu_gpd_gw_geo = [gpd.GeoDataFrame(geology, geometry=geo_points).set_crs('EPSG:4326') for geology,geo_points in zip(gw_anomaly_by_geology,insitu_points_gw_geo)]
gw_lats = [[point.y for point in geology_points] for geology_points in insitu_points_gw_geo]
gw_lons =  [[point.x for point in geology_points] for geology_points in insitu_points_gw_geo]

#In-Situ vs. Boreholes - by geology
# point --> points --> geology --> dataset
query_points_gw = [[[(find_nearest(var.y,lat)[1],find_nearest(var.x,lon)[1]) for lat,lon in zip(lats,lons)]
                    for var in var_datasets] 
                    for lats,lons in zip(gw_lats,gw_lons)]

var_points = np.array([[[var[:,point[0],point[1]] for point in geopoints] for var,geopoints in zip(var_datasets,var_points)]
                      for var_points in query_points_gw])
var_points_avg = [[xr.concat(all_points,dim='x').mean(dim='x') for all_points in geopoints] for geopoints in var_points]

# avg_points --> geology [i] --> dataset [ii]
#GW
for ii in [2]: #datasets
        for i in range(0,len(np.unique(geology.iloc[:,0]))): #geology
            figure,ax = plt.subplots(figsize=(10,7))
            var_points_avg[i][ii].plot(ax=ax)
            gw_anomaly_by_geology[i].T.plot(ax=ax,legend=False)
            plt.title(geology.iloc[:,1][geology.iloc[:,0]==np.unique(geology.iloc[:,0])[i]].iloc[0])

#Mann-Kendall
model_results_gw = [[(mk.seasonal_test(var_points_avg[i][ii],period=12)) for i in range(0,18)] for ii in [2]]
in_situ_results_gw = [mk.seasonal_test(gw_anomaly_by_geology[i].T.mean(axis=1),period=12) for i in range(0,18)]

#Increasing (in-situ) or no trend
geo_names[8:10] #in-situ - increasing; model - no trend
geo_names[-4] #in-situ only no trend
geo_names[-1] 

#NDVI
for ii in [4]: #datasets
        for i in range(0,len(np.unique(geology.iloc[:,0]))): #geology
            figure,ax = plt.subplots(figsize=(10,7))
            var_points_avg[i][ii].plot(ax=ax)
            gw_anomaly_by_geology[i].T.mean(axis=1).plot(ax=ax,legend=False)
            plt.title(geology.iloc[:,1][geology.iloc[:,0]==np.unique(geology.iloc[:,0])[i]].iloc[0])

#Mann-Kendall
model_results_ndvi = [[(mk.seasonal_test(var_points_avg[i][ii],period=12)) for i in range(0,18)] for ii in [4]]
