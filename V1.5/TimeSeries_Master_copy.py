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
#dates.dt.strftime('%Y-%m')

#Mean
fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(p_chirps.time,p_gpm.P_mm.mean(dim=['x','y']),color='C1')
ax.plot(p_chirps.time,p_chirps.P_mm.mean(dim=['x','y']),color='C0')
ax.set_ylabel('Precipitation (mm)',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)


#################################################
#EVAPOTRANSPIRATION

path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_ET'
files = sorted(glob.glob(path+'\*.nc'))
et_modis = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 500,"lon":500})

path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_PET'
files = sorted(glob.glob(path+'\*.nc'))
pet_modis = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 500,"lon":500})
#dates.dt.strftime('%Y-%m')

#Mean
fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(et_modis.time,et_modis.ET_kg_m2.mean(dim=['x','y']),color='C3',label='ET')
ax.plot(pet_modis.time,pet_modis.ET_kg_m2.mean(dim=['x','y']),color='C2',label='PET')
ax.set_ylabel('ET (mm)',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)
ax.legend()

P_PET = (p_gpm.P_mm.mean(dim=['x','y']))- (pet_modis.ET_kg_m2.mean(dim=['x','y']))
P_ET = (p_gpm.P_mm.mean(dim=['x','y'])) - (et_modis.ET_kg_m2.mean(dim=['x','y']))

#Mean
fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(pet_modis.time,P_PET,color='C1',label='P-PET')
ax.plot(et_modis.time,P_ET,color='C0',label='P-ET')
ax.set_ylabel('P - (P)ET (mm)',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)
ax.legend()


###################################################
#TWSA: GRACE

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

#############
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


###################################################
#GLDAS - CLSM

path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_CLSM'
files = sorted(glob.glob(path+'\*.nc'))

#GW
clm_gw = xr.open_mfdataset(files[0],parallel=True).mm

#SOIL MOISTURE
clm_sm_rz = xr.open_mfdataset(files[3],parallel=True)['kgm-2']
clm_sm_surface = xr.open_mfdataset(files[4],parallel=True)['kgm-2']

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax_twin = ax.twinx()
ax.plot(clm_sm_rz.time,clm_sm_rz.mean(dim=['x','y']),color='C0')
ax.plot(clm_gw.time,clm_gw.mean(dim=['x','y']),color='C1')
ax_twin.plot(clm_sm_surface.time,clm_sm_surface.mean(dim=['x','y']),color='C2')
ax.set_ylabel('Kg_m2',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax_twin = ax.twinx()
ax_twin.plot(clm_sm_rz.time,clm_sm_rz.mean(dim=['x','y']),color='C0')
ax.plot(clm_gw.time,clm_gw.mean(dim=['x','y']),color='C1')
ax.set_ylabel('mm',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)

#RUNOFF
#clm_r_surf = xr.open_mfdataset(files[2],parallel=True)['kgm-2s-1']
#clm_r_base = xr.open_mfdataset(files[1],parallel=True)['kgm-2s-1']
#dates.dt.strftime('%Y-%m')

###################################################
#SMAP
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\SMAP_SM_1km\south_africa_monthly\south_africa_monthly'
files = sorted(glob.glob(path+'\*.nc'))
smap = xr.open_mfdataset(files[0],parallel=True,chunks={"x": 100,"y":100}).SM_vwc
smap_dates = pd.date_range('2015-04','2021-01',  freq='1M') 

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax_twin = ax.twinx()
ax_twin.plot(clm_sm_surface.time,clm_sm_surface.mean(dim=['x','y']),color='C0')
ax.plot(smap_dates,smap.mean(dim=['x','y']),color='C1')
ax.set_ylabel('m3/m3',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)



###################################################
# #GLDAS - NOAH

# #RUNOFF
# path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_RUNOFF'
# files = sorted(glob.glob(path+'\*.nc'))
# gldas_r = xr.open_mfdataset(files[0],parallel=True).R_kg_m2
# #dates.dt.strftime('%Y-%m')

# fig = plt.figure(figsize=(15,4))
# ax = fig.add_subplot()
# ax.plot(gldas_r.time,gldas_r.mean(dim=['x','y']),color='C0')
# ax.set_ylabel('Kg_m2 (mm)',weight='bold',fontsize=12)
# ax.set_xlabel('Date',weight='bold',fontsize=12)
# ax.set_title('Time Series',weight='bold',fontsize=15)


# #SOIL MOISTURE
# path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_SOILMOISTURE'
# files = sorted(glob.glob(path+'\*.nc'))
# gldas_sm_0_10 = xr.open_mfdataset(files[1],parallel=True,chunks={"x": 100,"y":100}).SM_kg_m2
# gldas_sm_0_100 = xr.open_mfdataset(files[0],parallel=True,chunks={"x": 100,"y":100}).SM_kg_m2
# gldas_sm_0_200 = xr.open_mfdataset(files[2],parallel=True,chunks={"x": 100,"y":100}).SM_kg_m2
# gldas_sm_RZ = xr.open_mfdataset(files[3],parallel=True,chunks={"x": 100,"y":100}).SM_kg_m2
# #dates.dt.strftime('%Y-%m')

# fig = plt.figure(figsize=(15,4))
# ax = fig.add_subplot()
# ax.plot(gldas_sm_0_200.time,gldas_sm_0_10.mean(dim=['x','y']),color='C0')
# ax.set_ylabel('Kg_m2 (mm)',weight='bold',fontsize=12)
# ax.set_xlabel('Date',weight='bold',fontsize=12)
# ax.set_title('Time Series',weight='bold',fontsize=15)


# 2 minutes to run above code
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
##################################
#LAND USE LAND COVER RELATIONSHIPS
##################################

### NEED TO INCLUDE SPEI

#Mask all datasets for surface water before averaging spatially to get time series && LULC MASKS

#Sentinel-2 LULC (10 meter)
file_paths = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\Sentinel_LULC'
files = sorted(glob.glob(file_paths+'\*.nc'))
lulc = xr.open_mfdataset(files,parallel=True,chunks={"lat": 100,"lon":100}).lulc

variables_to_mask = [lst.LST_K, ndvi.NDVI, clm_gw, clm_sm_rz, clm_sm_surface, p_gpm.P_mm, et_modis.ET_kg_m2, pet_modis.ET_kg_m2]
xy_variables_to_mask = [[variable.x, variable.y] for variable in variables_to_mask]

path_class = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\Sentinel_LULC\classification.csv'
classes = pd.read_csv(path_class)
water_class = classes['Raster'][classes['Class']=='Water'].iloc[0]
tree_class = classes['Raster'][classes['Class']=='Trees'].iloc[0]
range_class = classes['Raster'][classes['Class']=='Rangeland'].iloc[0]
crops_class = classes['Raster'][classes['Class']=='Crops'].iloc[0]
urban_class = classes['Raster'][classes['Class']=='Built Area'].iloc[0]
bare_class = classes['Raster'][classes['Class']=='Bare Ground'].iloc[0]

class_values = [tree_class, range_class, crops_class, urban_class, bare_class]
class_names = ['trees','rangeland','crops','urban','bare']
variable_names = ['01_LST','02_NDVI','03_CLM_GW','04_CLM_RZ_SM','05_CLM_Surface_SM','06_P_GPM','07_ET','08_PET']
years = [2017,2021]

'''
#Resample LULC to resolution of dataset (14 minutes)
resampled_lulc = [lulc.rio.write_crs('epsg:4328').rio.reproject_match(variable.rio.write_crs('epsg:4328'),resampling=Resampling.mode) for variable in variables_to_mask]

path_mask = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\Sentinel_LULC\masks'
#Convert xarrays to numpy arrays for TIFs
resampled_lulc_arrays = [[np.array(resampled_lulc[ii][i]) for ii in range(0,len(variable_names))] for i in range(0,len(years))]

#Create a new data array with dimensions names, appropriate coordinate variable names and arrays, and variable name
new_arrays = [[xr.DataArray(resampled_lulc_arrays[i][ii], dims=("y", "x"), coords={"y": coord[1], "x": coord[0]}, name='{}'.format(variable_name)) 
for i in range(0,len(years))] 
for ii,variable_name,coord in zip(range(0,len(variables_to_mask)),variable_names,xy_variables_to_mask)]

[[new_arrays[ii][i].rio.set_crs("epsg:4326").rio.set_spatial_dims('x','y',inplace=True).rio.to_raster(path_mask+'/{}/{}.tif'.format(year,variable_name))
for ii,variable_name in zip(range(0,len(variables_to_mask)),variable_names)]
for i,year in zip(range(0,len(years)),years)] 
'''


path_mask = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\Sentinel_LULC\masks'
files_2017 = sorted(glob.glob(path_mask+'/2017/*.tif'))
files_2021 = sorted(glob.glob(path_mask+'/2021/*.tif'))

lulc_2017 = [xr.open_rasterio(file)[0] for file in files_2017]
lulc_2021 = [xr.open_rasterio(file)[0] for file in files_2021]

#2017 masks
water_masked_variables_17 = [variable.where(lulc!=water_class) for variable,lulc in zip(variables_to_mask,lulc_2017)]
#2021 masks
water_masked_variables_21 = [variable.where(lulc!=water_class) for variable,lulc in zip(variables_to_mask,lulc_2021)]

#2017 spatial averages
WMV_sp_avg_17 = [variable.mean(dim={'x','y'}) for variable in water_masked_variables_17]
#2021 spatial averages
WMV_sp_avg_21 = [variable.mean(dim={'x','y'}) for variable in water_masked_variables_21]


############################################################
variable_names = ['01_LST','02_NDVI','03_CLM_GW','04_CLM_RZ_SM','05_CLM_Surface_SM','06_P_GPM','07_ET','08_PET']
class_names = ['trees','rangeland','crops','urban','bare']
#Limit dates from 02/2003 through 12/2021 to match with CLSM (starts at 02/2003) and MODIS ET (ends at 12/2021) files
[lst.LST_K[7:], ndvi.NDVI[7:], clm_gw[:-7], clm_sm_rz[:-7], clm_sm_surface[:-7]] #include smap when time series is ready (MEANS)
[p_gpm.P_mm[13:], et_modis.ET_kg_m2[13:], pet_modis.ET_kg_m2[13:]] # (SUMS)

#WATER MASKED (primary)
#[0][7:] LST
#[1][7:] NDVI
#[2][:-7] CLM GW
#[3][:-7] CLM SM RZ
#[4][:-7] CLM SM Surface
#[5][13:] GPM PPT
#[6][13:] ET MODIS
#[7][13:] PET MODIS

#LULC MASKS (secondary)
#[0][0:8][different dates] trees
#[1][0:8][different dates] rangeland
#[2][0:8][different dates] crops
#[3][0:8][different dates] urban
#[4][0:8][different dates] bare


def condition_index_tci(variable):
    max = variable.max(dim='time')
    min = variable.min(dim='time')

    ci = [ (max-month)/(max-min)*100 for month in variable  ]
    ci = xr.concat(ci,dim='time')
    return ci

def condition_index_vci(variable):
    max = variable.max(dim='time')
    min = variable.min(dim='time')

    ci = [ (month-min)/(max-min)*100 for month in variable  ]
    ci = xr.concat(ci,dim='time')
    return ci

def health_index_vhi(vci,tci,alpha):
    vhi = [ (vci_month*alpha) + ((1-alpha)*tci_month) for vci_month,tci_month in zip(vci,tci)]
    vhi = xr.concat(vhi,dim='time')
    return vhi


##############################
#TEMPERATURE CONDITION INDEX (TCI) - 20 seconds
lst_wm = water_masked_variables_17[0][7:]
dates= pd.date_range('2003-02-01','2022-01-01' , freq='1M') #to match with CLSM & ET data period
tci = condition_index_tci(lst_wm)

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(tci.time,tci.mean(dim={'x','y'}),color='red')
ax.set_ylabel('TCI',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('TCI Time Series',weight='bold',fontsize=15)

#2 minutes to run
masked_tci = [tci.where(lulc_2017[1]==value) for value in class_values]
for i, name in zip(range(0,5),class_names):
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot()
    ax.plot(masked_tci[i].time,masked_tci[i].mean(dim={'x','y'}),color='red')
    ax.set_ylabel('TCI',weight='bold',fontsize=12)
    ax.set_xlabel('Date',weight='bold',fontsize=12)
    ax.set_title('TCI {} Time Series'.format(name),weight='bold',fontsize=15)

##############################
#VEGETATION CONDITION INDEX (VCI) - 35 seconds
ndvi_wm = water_masked_variables_17[1][7:]
dates= pd.date_range('2003-02-01','2022-01-01' , freq='1M') #to match with CLSM & ET data period
vci = condition_index_vci(ndvi_wm)

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(vci.time,vci.mean(dim={'x','y'}),color='green')
#ax.plot(tci.time,tci.mean(dim={'x','y'}),color='red')
ax.set_ylabel('VCI',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('VCI Time Series',weight='bold',fontsize=15)

#3 minutes to run
masked_vci = [vci.where(lulc_2017[1]==value) for value in class_values]
for i, name in zip(range(0,5),class_names):
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot()
    ax.plot(masked_vci[i].time,masked_vci[i].mean(dim={'x','y'}),color='green')
    ax.plot(masked_tci[i].time,masked_tci[i].mean(dim={'x','y'}),color='red')
    ax.set_ylabel('VCI',weight='bold',fontsize=12)
    ax.set_xlabel('Date',weight='bold',fontsize=12)
    ax.set_title('VCI {} Time Series'.format(name),weight='bold',fontsize=15)

##############################
#VEGETATION HEALTH INDEX (VHI) - 40 seconds

# low alpha TEMPERATURE DEPENDENT
# high alpha MOISTURE DEPENDENT
alpha = 0.5
vhi = health_index_vhi(vci,tci,alpha)

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(vhi.time,vhi.mean(dim={'x','y'}),color='orange')
ax.set_ylabel('VHI',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('VHI Time Series',weight='bold',fontsize=15)

#6 minutes to run
masked_vhi = [health_index_vhi(vci,tci,alpha) for vci,tci in zip(masked_vci,masked_tci)]
for i, name in zip(range(0,5),class_names):
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot()
    ax.plot(masked_vhi[i].time,masked_vhi[i].mean(dim={'x','y'}),color='orange')
    ax.plot(masked_tci[i].time,masked_tci[i].mean(dim={'x','y'}),color='red')
    ax.set_ylabel('VHI',weight='bold',fontsize=12)
    ax.set_xlabel('Date',weight='bold',fontsize=12)
    ax.set_title('VHI {} Time Series'.format(name),weight='bold',fontsize=15)

############################################################
#MASKED LULC OR AQUIFER -- need to do SPEI

#WATER MASKED (primary)
#[0][7:] LST
#[1][7:] NDVI
#[2][:-7] CLM GW
#[3][:-7] CLM SM RZ
#[4][:-7] CLM SM Surface
#[5][13:] GPM PPT
#[6][13:] ET MODIS
#[7][13:] PET MODIS

lst_wm = water_masked_variables_17[0][7:]
ndvi_wm = water_masked_variables_17[1][7:]
clm_gw_wm = water_masked_variables_17[2][:-7]
clm_sm_rz_wm = water_masked_variables_17[3][:-7]
clm_sm_surface_wm = water_masked_variables_17[4][:-7]
p_gpm_wm = water_masked_variables_17[5][13:]
et_modis_wm = water_masked_variables_17[6][13:]
pet_modis_wm = water_masked_variables_17[7][13:]

variables = [lst_wm,ndvi_wm,clm_gw_wm,clm_sm_rz_wm,clm_sm_surface_wm,p_gpm_wm,et_modis_wm,pet_modis_wm]
colors = ['#E69F00','#009E73','#0072B2','#0072B2','#0072B2','#56B4E9','#6E8FBB','#6E8FBB']
variable_names = ['01_LST','02_NDVI','03_CLM_GW','04_CLM_RZ_SM','05_CLM_Surface_SM','06_P_GPM','07_ET','08_PET']
unit_names = ['LST (K)','NDVI','GW Level (mm)','RZ SM Level (mm)','Surface SM Level (mm)','Precipitation (mm)','ET (kgm-2)','ET (kgm-2)']
variable_ylims = [[285,310],[0.15,0.70],[400,1000],[120,320],[1.5,6.5],[0,275],[0,105],[100,280]]

#1.5 min to run
for i, name,unit,ylims in zip(range(0,8),variable_names,unit_names,variable_ylims):
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot()
    ax.plot(variables[i].time,variables[i].mean(dim={'x','y'}),color=colors[i])
    ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
    ax.set_ylim(ylims)
    ax.set_xlabel('Date',weight='bold',fontsize=12)
    ax.set_title('{} Time Series'.format(name),weight='bold',fontsize=15)


#2.5 minutes to run
masked_variables = [[variable.where(lulc==value) for value in class_values] for variable,lulc in zip(variables,lulc_2017)]
class_names = ['trees','rangeland','crops','urban','bare']

for ii,var_name,unit,ylims in zip(range(0,8),variable_names,unit_names,variable_ylims): #8 variables
    for i, name in zip(range(0,5),class_names): #5 classes
        fig = plt.figure(figsize=(15,4))
        ax = fig.add_subplot()
        ax.plot(masked_variables[ii][i].time,masked_variables[ii][i].mean(dim={'x','y'}),color=colors[ii])
        ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
        ax.set_ylim(ylims)
        ax.set_xlabel('Date',weight='bold',fontsize=12)
        ax.set_title('{} {} Time Series'.format(var_name,name),weight='bold',fontsize=15)

############################################################
#SEASONAL AVERAGE

#include SPEI
variables = [tci,vci,vhi,lst_wm,ndvi_wm,clm_gw_wm,clm_sm_rz_wm,clm_sm_surface_wm,p_gpm_wm,et_modis_wm,pet_modis_wm]

variables_mean = variables[0:8] #include smap when time series is ready
variables_sum = variables[8:11]

#Wet: Oct - April 
#Dry: May - Sept

#Start date: 2003-02
#End date: 2021-12

season_array = np.where( (np.array(variables_mean[0]['time.month']) == 2) |
(np.array(variables_mean[0]['time.month']) == 3) |
(np.array(variables_mean[0]['time.month']) == 4) |
(np.array(variables_mean[0]['time.month']) == 10) |
(np.array(variables_mean[0]['time.month']) == 11) |
(np.array(variables_mean[0]['time.month']) == 12) | 
(np.array(variables_mean[0]['time.month']) == 1), 'WET','DRY')

Y_M_season_idxs = [pd.MultiIndex.from_arrays([np.array(variable['time.year']),season_array]) for variable in variables_mean]
for i in range(0,len(variables_mean)):
    variables_mean[i].coords['year_month_S'] = ('time', Y_M_season_idxs[i])
variables_seasonally_mean = [variable.groupby('year_month_S').mean() for variable in variables_mean]

Y_M_season_idxs = [pd.MultiIndex.from_arrays([np.array(variable['time.year']),season_array]) for variable in variables_sum]
for i in range(0,len(variables_sum)):
    variables_sum[i].coords['year_month_S'] = ('time', Y_M_season_idxs[i])
variables_seasonally_sum = [variable.groupby('year_month_S').sum() for variable in variables_sum]


variables_mean = [variable_seasoned.mean(dim={'x','y'}) for variable_seasoned in variables_seasonally_mean]
variables_sum = [variable_seasoned.mean(dim={'x','y'}) for variable_seasoned in variables_seasonally_sum]


###########
#MEAN VARIABLES - 2 min
colors = ['red','green','orange','#E69F00','#009E73','#0072B2','#0072B2','#0072B2']
variable_names = ['TCI','VCI','VHI','01_LST','02_NDVI','03_CLM_GW','04_CLM_RZ_SM','05_CLM_Surface_SM']
unit_names = ['TCI','VCI','VHI','LST (K)','NDVI','GW Level (mm)','RZ SM Level (mm)','Surface SM Level (mm)']
variable_ylims = [[0,100],[0,100],[0,100],[285,310],[0.15,0.70],[400,1000],[120,320],[1.5,6.5]]

for i, name,unit,ylims in zip(range(0,8),variable_names,unit_names,variable_ylims):
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot()
    ax.plot(variables_mean[i][0::2].year_month_S_level_0,variables_mean[i][0::2],color=colors[i]) #WET SEASON
    ax.plot(variables_mean[i][1::2].year_month_S_level_0,variables_mean[i][1::2],color=colors[i], alpha=0.5) #DRY SEASON
    ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
    ax.set_ylim(ylims)
    ax.set_xlabel('Year',weight='bold',fontsize=12)
    ax.set_xlim(2003,2021)
    ax.set_title('{} Time Series'.format(name),weight='bold',fontsize=15)

###########
#SUM VARIABLES - 30s
colors = ['#56B4E9','#6E8FBB','#6E8FBB']
variable_names = ['06_P_GPM','07_ET','08_PET']
unit_names = ['Precipitation (mm)','ET (kgm-2)','ET (kgm-2)']
variable_ylims = [[0,400],[0,300],[0,1000]]

for i, name,unit,ylims in zip(range(0,3),variable_names,unit_names,variable_ylims):
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot()
    ax.plot(variables_sum[i][0::2].year_month_S_level_0,variables_sum[i][0::2],color=colors[i]) #WET SEASON
    ax.plot(variables_sum[i][1::2].year_month_S_level_0,variables_sum[i][1::2],color=colors[i], alpha=0.5) #DRY SEASON
    ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
    ax.set_ylim(ylims)
    ax.set_xlabel('Year',weight='bold',fontsize=12)
    ax.set_xlim(2003,2021)
    ax.set_title('{} Time Series'.format(name),weight='bold',fontsize=15)


############################################################
#SEASONAL AVERAGE BY LULC MASKED DATASETS
all_masked_variables = [masked_tci,masked_vci,masked_vhi,masked_variables[0],masked_variables[1],masked_variables[2],
masked_variables[3],masked_variables[4],masked_variables[5],masked_variables[6],masked_variables[7]]

variables_mean = all_masked_variables[0:8] #include smap when time series is ready
variables_sum = all_masked_variables[8:11]


season_arrays = [np.where( (np.array(variables_mean[i][0]['time.month']) == 2) |
(np.array(variables_mean[i][0]['time.month']) == 3) |
(np.array(variables_mean[i][0]['time.month']) == 4) |
(np.array(variables_mean[i][0]['time.month']) == 10) |
(np.array(variables_mean[i][0]['time.month']) == 11) |
(np.array(variables_mean[i][0]['time.month']) == 12) | 
(np.array(variables_mean[i][0]['time.month']) == 1), 'WET','DRY') for i in range(0,len(variables_mean[0]))]


#MEAN VARIABLES
Y_M_season_idxs = [[pd.MultiIndex.from_arrays([np.array(variable[i]['time.year']),season_array]) for variable in variables_mean] 
                    for i,season_array in zip(range(0,len(variables_mean[0])),season_arrays)]
for ii in range(0,len(variables_mean[0])):
    for i in range(0,len(variables_mean)):
        variables_mean[i][ii].coords['year_month_S'] = ('time', Y_M_season_idxs[ii][i])
variables_seasonally_mean_lulc = [[variable[i].groupby('year_month_S').mean() for variable in variables_mean] for i in range(0,len(variables_mean[0]))]

#SUM VARIABLES
Y_M_season_idxs = [[pd.MultiIndex.from_arrays([np.array(variable[i]['time.year']),season_array]) for variable in variables_sum]
                     for i,season_array in zip(range(0,len(variables_sum[0])),season_arrays)]
for ii in range(0,len(variables_sum[0])):
    for i in range(0,len(variables_sum)):
        variables_sum[i][ii].coords['year_month_S'] = ('time', Y_M_season_idxs[ii][i])
variables_seasonally_sum_lulc = [[variable[i].groupby('year_month_S').sum() for variable in variables_sum] for i in range(0,len(variables_sum[0]))]



variables_mean = [[variable_seasoned[i].mean(dim={'x','y'}) for variable_seasoned in variables_seasonally_mean_lulc] for i in range(0,len(variables_seasonally_mean_lulc[0]))]
variables_sum = [[variable_seasoned[i].mean(dim={'x','y'}) for variable_seasoned in variables_seasonally_sum_lulc] for i in range(0,len(variables_seasonally_sum_lulc[0]))]

###########
#MEAN VARIABLES - >10 minutes
colors = ['red','green','orange','#E69F00','#009E73','#0072B2','#0072B2','#0072B2']
variable_names = ['TCI','VCI','VHI','01_LST','02_NDVI','03_CLM_GW','04_CLM_RZ_SM','05_CLM_Surface_SM']
unit_names = ['TCI','VCI','VHI','LST (K)','NDVI','GW Level (mm)','RZ SM Level (mm)','Surface SM Level (mm)']
variable_ylims = [[0,100],[0,100],[0,100],[285,310],[0.15,0.70],[400,1000],[120,320],[1.5,6.5]]
class_names = ['trees','rangeland','crops','urban','bare']

for ii,var_name,unit,ylims in zip(range(0,8),variable_names,unit_names,variable_ylims): #8 variables
    for i, name in zip(range(0,5),class_names): #5 classes
        fig = plt.figure(figsize=(15,4))
        ax = fig.add_subplot()
        ax.plot(variables_mean[ii][i][0::2].year_month_S_level_0,variables_mean[ii][i][0::2],color=colors[ii])
        ax.plot(variables_mean[ii][i][1::2].year_month_S_level_0,variables_mean[ii][i][1::2],color=colors[ii], alpha=0.5)
        ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
        ax.set_ylim(ylims)
        ax.set_xlabel('Year',weight='bold',fontsize=12)
        ax.set_xlim(2003,2021)
        ax.set_title('{} {} Time Series'.format(var_name,name),weight='bold',fontsize=15)


###########
#SUM VARIABLES - > 3 minutes
colors = ['#56B4E9','#6E8FBB','#6E8FBB']
variable_names = ['06_P_GPM','07_ET','08_PET']
unit_names = ['Precipitation (mm)','ET (kgm-2)','ET (kgm-2)']
variable_ylims = [[0,400],[0,300],[0,1000]]
class_names = ['trees','rangeland','crops','urban','bare']

for ii,var_name,unit,ylims in zip(range(0,3),variable_names,unit_names,variable_ylims): #3 variables
    for i, name in zip(range(0,5),class_names): #5 classes
        fig = plt.figure(figsize=(15,4))
        ax = fig.add_subplot()
        ax.plot(variables_sum[ii][i][0::2].year_month_S_level_0,variables_sum[ii][i][0::2],color=colors[ii])
        ax.plot(variables_sum[ii][i][1::2].year_month_S_level_0,variables_sum[ii][i][1::2],color=colors[ii], alpha=0.5)
        ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
        ax.set_ylim(ylims)
        ax.set_xlabel('Year',weight='bold',fontsize=12)
        ax.set_xlim(2003,2021)
        ax.set_title('{} {} Time Series'.format(var_name,name),weight='bold',fontsize=15)



########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
##################################
#AQUIFER CHARACTER RELATIONSHIPS
##################################
file_paths = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\Hydrogeology_maps'
files = sorted(glob.glob(file_paths+'\*.tif'))
hydrogeo = [xr.open_rasterio(file) for file in files]

### NEED TO INCLUDE SPEI

#Mask all datasets for surface water before averaging spatially to get time series && LULC MASKS

#Sentinel-2 LULC (10 meter)
file_paths = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\Sentinel_LULC'
files = sorted(glob.glob(file_paths+'\*.nc'))
lulc = xr.open_mfdataset(files,parallel=True,chunks={"lat": 100,"lon":100}).lulc

variables_to_mask = [lst.LST_K, ndvi.NDVI, clm_gw, clm_sm_rz, clm_sm_surface, p_gpm.P_mm, et_modis.ET_kg_m2, pet_modis.ET_kg_m2]
xy_variables_to_mask = [[variable.x, variable.y] for variable in variables_to_mask]

path_class = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\Sentinel_LULC\classification.csv'
classes = pd.read_csv(path_class)
water_class = classes['Raster'][classes['Class']=='Water'].iloc[0]
tree_class = classes['Raster'][classes['Class']=='Trees'].iloc[0]
range_class = classes['Raster'][classes['Class']=='Rangeland'].iloc[0]
crops_class = classes['Raster'][classes['Class']=='Crops'].iloc[0]
urban_class = classes['Raster'][classes['Class']=='Built Area'].iloc[0]
bare_class = classes['Raster'][classes['Class']=='Bare Ground'].iloc[0]

class_values = [tree_class, range_class, crops_class, urban_class, bare_class]
class_names = ['trees','rangeland','crops','urban','bare']
variable_names = ['01_LST','02_NDVI','03_CLM_GW','04_CLM_RZ_SM','05_CLM_Surface_SM','06_P_GPM','07_ET','08_PET']
years = [2017,2021]
'''
#Resample LULC to resolution of dataset (14 minutes)
resampled_lulc = [lulc.rio.write_crs('epsg:4328').rio.reproject_match(variable.rio.write_crs('epsg:4328'),resampling=Resampling.mode) for variable in variables_to_mask]

path_mask = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\Sentinel_LULC\masks'
#Convert xarrays to numpy arrays for TIFs
resampled_lulc_arrays = [[np.array(resampled_lulc[ii][i]) for ii in range(0,len(variable_names))] for i in range(0,len(years))]

#Create a new data array with dimensions names, appropriate coordinate variable names and arrays, and variable name
new_arrays = [[xr.DataArray(resampled_lulc_arrays[i][ii], dims=("y", "x"), coords={"y": coord[1], "x": coord[0]}, name='{}'.format(variable_name)) 
for i in range(0,len(years))] 
for ii,variable_name,coord in zip(range(0,len(variables_to_mask)),variable_names,xy_variables_to_mask)]

[[new_arrays[ii][i].rio.set_crs("epsg:4326").rio.set_spatial_dims('x','y',inplace=True).rio.to_raster(path_mask+'/{}/{}.tif'.format(year,variable_name))
for ii,variable_name in zip(range(0,len(variables_to_mask)),variable_names)]
for i,year in zip(range(0,len(years)),years)] 
'''


path_mask = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\Sentinel_LULC\masks'
files_2017 = sorted(glob.glob(path_mask+'/2017/*.tif'))
files_2021 = sorted(glob.glob(path_mask+'/2021/*.tif'))

lulc_2017 = [xr.open_rasterio(file)[0] for file in files_2017]
lulc_2021 = [xr.open_rasterio(file)[0] for file in files_2021]

#2017 masks
water_masked_variables_17 = [variable.where(lulc!=water_class) for variable,lulc in zip(variables_to_mask,lulc_2017)]
#2021 masks
water_masked_variables_21 = [variable.where(lulc!=water_class) for variable,lulc in zip(variables_to_mask,lulc_2021)]



############################################################
variable_names = ['01_LST','02_NDVI','03_CLM_GW','04_CLM_RZ_SM','05_CLM_Surface_SM','06_P_GPM','07_ET','08_PET']
class_names = ['trees','rangeland','crops','urban','bare']
#Limit dates from 02/2003 through 12/2021 to match with CLSM (starts at 02/2003) and MODIS ET (ends at 12/2021) files
[lst.LST_K[7:], ndvi.NDVI[7:], clm_gw[:-7], clm_sm_rz[:-7], clm_sm_surface[:-7]] #include smap when time series is ready (MEANS)
[p_gpm.P_mm[13:], et_modis.ET_kg_m2[13:], pet_modis.ET_kg_m2[13:]] # (SUMS)

#WATER MASKED (primary)
#[0][7:] LST
#[1][7:] NDVI
#[2][:-7] CLM GW
#[3][:-7] CLM SM RZ
#[4][:-7] CLM SM Surface
#[5][13:] GPM PPT
#[6][13:] ET MODIS
#[7][13:] PET MODIS

#LULC MASKS (secondary)
#[0][0:8][different dates] trees
#[1][0:8][different dates] rangeland
#[2][0:8][different dates] crops
#[3][0:8][different dates] urban
#[4][0:8][different dates] bare


def condition_index_tci(variable):
    max = variable.max(dim='time')
    min = variable.min(dim='time')

    ci = [ (max-month)/(max-min)*100 for month in variable  ]
    ci = xr.concat(ci,dim='time')
    return ci

def condition_index_vci(variable):
    max = variable.max(dim='time')
    min = variable.min(dim='time')

    ci = [ (month-min)/(max-min)*100 for month in variable  ]
    ci = xr.concat(ci,dim='time')
    return ci

def health_index_vhi(vci,tci,alpha):
    vhi = [ (vci_month*alpha) + ((1-alpha)*tci_month) for vci_month,tci_month in zip(vci,tci)]
    vhi = xr.concat(vhi,dim='time')
    return vhi


##############################
#TEMPERATURE CONDITION INDEX (TCI) - 20 seconds
lst_wm = water_masked_variables_17[0][7:]
dates= pd.date_range('2003-02-01','2022-01-01' , freq='1M') #to match with CLSM & ET data period
tci = condition_index_tci(lst_wm)

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(tci.time,tci.mean(dim={'x','y'}),color='red')
ax.set_ylabel('TCI',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('TCI Time Series',weight='bold',fontsize=15)

#2 minutes to run
masked_tci = [tci.where(lulc_2017[1]==value) for value in class_values]
for i, name in zip(range(0,5),class_names):
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot()
    ax.plot(masked_tci[i].time,masked_tci[i].mean(dim={'x','y'}),color='red')
    ax.set_ylabel('TCI',weight='bold',fontsize=12)
    ax.set_xlabel('Date',weight='bold',fontsize=12)
    ax.set_title('TCI {} Time Series'.format(name),weight='bold',fontsize=15)

##############################
#VEGETATION CONDITION INDEX (VCI) - 35 seconds
ndvi_wm = water_masked_variables_17[1][7:]
dates= pd.date_range('2003-02-01','2022-01-01' , freq='1M') #to match with CLSM & ET data period
vci = condition_index_vci(ndvi_wm)

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(vci.time,vci.mean(dim={'x','y'}),color='green')
#ax.plot(tci.time,tci.mean(dim={'x','y'}),color='red')
ax.set_ylabel('VCI',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('VCI Time Series',weight='bold',fontsize=15)

#3 minutes to run
masked_vci = [vci.where(lulc_2017[1]==value) for value in class_values]
for i, name in zip(range(0,5),class_names):
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot()
    ax.plot(masked_vci[i].time,masked_vci[i].mean(dim={'x','y'}),color='green')
    ax.plot(masked_tci[i].time,masked_tci[i].mean(dim={'x','y'}),color='red')
    ax.set_ylabel('VCI',weight='bold',fontsize=12)
    ax.set_xlabel('Date',weight='bold',fontsize=12)
    ax.set_title('VCI {} Time Series'.format(name),weight='bold',fontsize=15)

##############################
#VEGETATION HEALTH INDEX (VHI) - 40 seconds

# low alpha TEMPERATURE DEPENDENT
# high alpha MOISTURE DEPENDENT
alpha = 0.5
vhi = health_index_vhi(vci,tci,alpha)

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(vhi.time,vhi.mean(dim={'x','y'}),color='orange')
ax.set_ylabel('VHI',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('VHI Time Series',weight='bold',fontsize=15)

#6 minutes to run
masked_vhi = [health_index_vhi(vci,tci,alpha) for vci,tci in zip(masked_vci,masked_tci)]
for i, name in zip(range(0,5),class_names):
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot()
    ax.plot(masked_vhi[i].time,masked_vhi[i].mean(dim={'x','y'}),color='orange')
    ax.plot(masked_tci[i].time,masked_tci[i].mean(dim={'x','y'}),color='red')
    ax.set_ylabel('VHI',weight='bold',fontsize=12)
    ax.set_xlabel('Date',weight='bold',fontsize=12)
    ax.set_title('VHI {} Time Series'.format(name),weight='bold',fontsize=15)

############################################################
#MASKED LULC OR AQUIFER -- need to do SPEI

#WATER MASKED (primary)
#[0][7:] LST
#[1][7:] NDVI
#[2][:-7] CLM GW
#[3][:-7] CLM SM RZ
#[4][:-7] CLM SM Surface
#[5][13:] GPM PPT
#[6][13:] ET MODIS
#[7][13:] PET MODIS

lst_wm = water_masked_variables_17[0][7:]
ndvi_wm = water_masked_variables_17[1][7:]
clm_gw_wm = water_masked_variables_17[2][:-7]
clm_sm_rz_wm = water_masked_variables_17[3][:-7]
clm_sm_surface_wm = water_masked_variables_17[4][:-7]
p_gpm_wm = water_masked_variables_17[5][13:]
et_modis_wm = water_masked_variables_17[6][13:]
pet_modis_wm = water_masked_variables_17[7][13:]

variables = [lst_wm,ndvi_wm,clm_gw_wm,clm_sm_rz_wm,clm_sm_surface_wm,p_gpm_wm,et_modis_wm,pet_modis_wm]
colors = ['#E69F00','#009E73','#0072B2','#0072B2','#0072B2','#56B4E9','#6E8FBB','#6E8FBB']
variable_names = ['01_LST','02_NDVI','03_CLM_GW','04_CLM_RZ_SM','05_CLM_Surface_SM','06_P_GPM','07_ET','08_PET']
unit_names = ['LST (K)','NDVI','GW Level (mm)','RZ SM Level (mm)','Surface SM Level (mm)','Precipitation (mm)','ET (kgm-2)','ET (kgm-2)']
variable_ylims = [[285,310],[0.15,0.70],[400,1000],[120,320],[1.5,6.5],[0,275],[0,105],[100,280]]

#1.5 min to run
for i, name,unit,ylims in zip(range(0,8),variable_names,unit_names,variable_ylims):
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot()
    ax.plot(variables[i].time,variables[i].mean(dim={'x','y'}),color=colors[i])
    ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
    ax.set_ylim(ylims)
    ax.set_xlabel('Date',weight='bold',fontsize=12)
    ax.set_title('{} Time Series'.format(name),weight='bold',fontsize=15)


#2.5 minutes to run
masked_variables = [[variable.where(lulc==value) for value in class_values] for variable,lulc in zip(variables,lulc_2017)]
class_names = ['trees','rangeland','crops','urban','bare']

for ii,var_name,unit,ylims in zip(range(0,8),variable_names,unit_names,variable_ylims): #8 variables
    for i, name in zip(range(0,5),class_names): #5 classes
        fig = plt.figure(figsize=(15,4))
        ax = fig.add_subplot()
        ax.plot(masked_variables[ii][i].time,masked_variables[ii][i].mean(dim={'x','y'}),color=colors[ii])
        ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
        ax.set_ylim(ylims)
        ax.set_xlabel('Date',weight='bold',fontsize=12)
        ax.set_title('{} {} Time Series'.format(var_name,name),weight='bold',fontsize=15)

############################################################
#SEASONAL AVERAGE

#include SPEI
variables = [tci,vci,vhi,lst_wm,ndvi_wm,clm_gw_wm,clm_sm_rz_wm,clm_sm_surface_wm,p_gpm_wm,et_modis_wm,pet_modis_wm]

variables_mean = variables[0:8] #include smap when time series is ready
variables_sum = variables[8:11]

#Wet: Oct - April 
#Dry: May - Sept

#Start date: 2003-02
#End date: 2021-12

season_array = np.where( (np.array(variables_mean[0]['time.month']) == 2) |
(np.array(variables_mean[0]['time.month']) == 3) |
(np.array(variables_mean[0]['time.month']) == 4) |
(np.array(variables_mean[0]['time.month']) == 10) |
(np.array(variables_mean[0]['time.month']) == 11) |
(np.array(variables_mean[0]['time.month']) == 12) | 
(np.array(variables_mean[0]['time.month']) == 1), 'WET','DRY')

Y_M_season_idxs = [pd.MultiIndex.from_arrays([np.array(variable['time.year']),season_array]) for variable in variables_mean]
for i in range(0,len(variables_mean)):
    variables_mean[i].coords['year_month_S'] = ('time', Y_M_season_idxs[i])
variables_seasonally_mean = [variable.groupby('year_month_S').mean() for variable in variables_mean]

Y_M_season_idxs = [pd.MultiIndex.from_arrays([np.array(variable['time.year']),season_array]) for variable in variables_sum]
for i in range(0,len(variables_sum)):
    variables_sum[i].coords['year_month_S'] = ('time', Y_M_season_idxs[i])
variables_seasonally_sum = [variable.groupby('year_month_S').sum() for variable in variables_sum]


variables_mean = [variable_seasoned.mean(dim={'x','y'}) for variable_seasoned in variables_seasonally_mean]
variables_sum = [variable_seasoned.mean(dim={'x','y'}) for variable_seasoned in variables_seasonally_sum]


###########
#MEAN VARIABLES - 2 min
colors = ['red','green','orange','#E69F00','#009E73','#0072B2','#0072B2','#0072B2']
variable_names = ['TCI','VCI','VHI','01_LST','02_NDVI','03_CLM_GW','04_CLM_RZ_SM','05_CLM_Surface_SM']
unit_names = ['TCI','VCI','VHI','LST (K)','NDVI','GW Level (mm)','RZ SM Level (mm)','Surface SM Level (mm)']
variable_ylims = [[0,100],[0,100],[0,100],[285,310],[0.15,0.70],[400,1000],[120,320],[1.5,6.5]]

for i, name,unit,ylims in zip(range(0,8),variable_names,unit_names,variable_ylims):
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot()
    ax.plot(variables_mean[i][0::2].year_month_S_level_0,variables_mean[i][0::2],color=colors[i]) #WET SEASON
    ax.plot(variables_mean[i][1::2].year_month_S_level_0,variables_mean[i][1::2],color=colors[i], alpha=0.5) #DRY SEASON
    ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
    ax.set_ylim(ylims)
    ax.set_xlabel('Year',weight='bold',fontsize=12)
    ax.set_xlim(2003,2021)
    ax.set_title('{} Time Series'.format(name),weight='bold',fontsize=15)

###########
#SUM VARIABLES - 30s
colors = ['#56B4E9','#6E8FBB','#6E8FBB']
variable_names = ['06_P_GPM','07_ET','08_PET']
unit_names = ['Precipitation (mm)','ET (kgm-2)','ET (kgm-2)']
variable_ylims = [[0,400],[0,300],[0,1000]]

for i, name,unit,ylims in zip(range(0,3),variable_names,unit_names,variable_ylims):
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot()
    ax.plot(variables_sum[i][0::2].year_month_S_level_0,variables_sum[i][0::2],color=colors[i]) #WET SEASON
    ax.plot(variables_sum[i][1::2].year_month_S_level_0,variables_sum[i][1::2],color=colors[i], alpha=0.5) #DRY SEASON
    ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
    ax.set_ylim(ylims)
    ax.set_xlabel('Year',weight='bold',fontsize=12)
    ax.set_xlim(2003,2021)
    ax.set_title('{} Time Series'.format(name),weight='bold',fontsize=15)


############################################################
#SEASONAL AVERAGE BY LULC MASKED DATASETS
all_masked_variables = [masked_tci,masked_vci,masked_vhi,masked_variables[0],masked_variables[1],masked_variables[2],
masked_variables[3],masked_variables[4],masked_variables[5],masked_variables[6],masked_variables[7]]

variables_mean = all_masked_variables[0:8] #include smap when time series is ready
variables_sum = all_masked_variables[8:11]


season_arrays = [np.where( (np.array(variables_mean[i][0]['time.month']) == 2) |
(np.array(variables_mean[i][0]['time.month']) == 3) |
(np.array(variables_mean[i][0]['time.month']) == 4) |
(np.array(variables_mean[i][0]['time.month']) == 10) |
(np.array(variables_mean[i][0]['time.month']) == 11) |
(np.array(variables_mean[i][0]['time.month']) == 12) | 
(np.array(variables_mean[i][0]['time.month']) == 1), 'WET','DRY') for i in range(0,len(variables_mean[0]))]


#MEAN VARIABLES
Y_M_season_idxs = [[pd.MultiIndex.from_arrays([np.array(variable[i]['time.year']),season_array]) for variable in variables_mean] 
                    for i,season_array in zip(range(0,len(variables_mean[0])),season_arrays)]
for ii in range(0,len(variables_mean[0])):
    for i in range(0,len(variables_mean)):
        variables_mean[i][ii].coords['year_month_S'] = ('time', Y_M_season_idxs[ii][i])
variables_seasonally_mean_lulc = [[variable[i].groupby('year_month_S').mean() for variable in variables_mean] for i in range(0,len(variables_mean[0]))]

#SUM VARIABLES
Y_M_season_idxs = [[pd.MultiIndex.from_arrays([np.array(variable[i]['time.year']),season_array]) for variable in variables_sum]
                     for i,season_array in zip(range(0,len(variables_sum[0])),season_arrays)]
for ii in range(0,len(variables_sum[0])):
    for i in range(0,len(variables_sum)):
        variables_sum[i][ii].coords['year_month_S'] = ('time', Y_M_season_idxs[ii][i])
variables_seasonally_sum_lulc = [[variable[i].groupby('year_month_S').sum() for variable in variables_sum] for i in range(0,len(variables_sum[0]))]



variables_mean = [[variable_seasoned[i].mean(dim={'x','y'}) for variable_seasoned in variables_seasonally_mean_lulc] for i in range(0,len(variables_seasonally_mean_lulc[0]))]
variables_sum = [[variable_seasoned[i].mean(dim={'x','y'}) for variable_seasoned in variables_seasonally_sum_lulc] for i in range(0,len(variables_seasonally_sum_lulc[0]))]

###########
#MEAN VARIABLES - 2 min
colors = ['red','green','orange','#E69F00','#009E73','#0072B2','#0072B2','#0072B2']
variable_names = ['TCI','VCI','VHI','01_LST','02_NDVI','03_CLM_GW','04_CLM_RZ_SM','05_CLM_Surface_SM']
unit_names = ['TCI','VCI','VHI','LST (K)','NDVI','GW Level (mm)','RZ SM Level (mm)','Surface SM Level (mm)']
variable_ylims = [[0,100],[0,100],[0,100],[285,310],[0.15,0.70],[400,1000],[120,320],[1.5,6.5]]
class_names = ['trees','rangeland','crops','urban','bare']

for ii,var_name,unit,ylims in zip(range(0,8),variable_names,unit_names,variable_ylims): #8 variables
    for i, name in zip(range(0,5),class_names): #5 classes
        fig = plt.figure(figsize=(15,4))
        ax = fig.add_subplot()
        ax.plot(variables_mean[ii][i][0::2].year_month_S_level_0,variables_mean[ii][i][0::2],color=colors[ii])
        ax.plot(variables_mean[ii][i][1::2].year_month_S_level_0,variables_mean[ii][i][1::2],color=colors[ii], alpha=0.5)
        ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
        ax.set_ylim(ylims)
        ax.set_xlabel('Year',weight='bold',fontsize=12)
        ax.set_xlim(2003,2021)
        ax.set_title('{} {} Time Series'.format(var_name,name),weight='bold',fontsize=15)


###########
#SUM VARIABLES - 30s
colors = ['#56B4E9','#6E8FBB','#6E8FBB']
variable_names = ['06_P_GPM','07_ET','08_PET']
unit_names = ['Precipitation (mm)','ET (kgm-2)','ET (kgm-2)']
variable_ylims = [[0,400],[0,300],[0,1000]]
class_names = ['trees','rangeland','crops','urban','bare']

for ii,var_name,unit,ylims in zip(range(0,3),variable_names,unit_names,variable_ylims): #3 variables
    for i, name in zip(range(0,5),class_names): #5 classes
        fig = plt.figure(figsize=(15,4))
        ax = fig.add_subplot()
        ax.plot(variables_sum[ii][i][0::2].year_month_S_level_0,variables_sum[ii][i][0::2],color=colors[ii])
        ax.plot(variables_sum[ii][i][1::2].year_month_S_level_0,variables_sum[ii][i][1::2],color=colors[ii], alpha=0.5)
        ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
        ax.set_ylim(ylims)
        ax.set_xlabel('Year',weight='bold',fontsize=12)
        ax.set_xlim(2003,2021)
        ax.set_title('{} {} Time Series'.format(var_name,name),weight='bold',fontsize=15)


############################################################
############################################################
############################################################
#MULTIPLOT

#CLM - GW, SM (RZ/SURFACE) + SMAP + NDVI + PPT(?)
#Based on Aquifer // Vegetation -- breakdown by season as well

fig,ax1 = plt.subplots(figsize=(15,5))
ax2 = ax1.twinx()
ax3 = ax2.twinx()
plt.xticks(rotation=45)
ax1.plot(clm_gw.time,clm_gw.mean(dim=['x','y']),color='C0')
ax1.set_ylabel('(mm)')
ax1.set_xlabel('Date')

ax2.plot(lst.time,lst.LST_K.mean(dim=['x','y']),'--',color='C1')
ax2.set_ylabel('K')

ax3.plot(ndvi.time,ndvi.NDVI.mean(dim=['x','y']),color='C2')
ax3.set_ylabel('NDVI')


####
#High TCI -- Low Temperature
#Low TCI -- High Temperature
fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax2 = ax.twinx()
ax3 = ax2.twinx()
ax4 = ax3.twinx()
ax2.plot(clm_gw.time,clm_gw.mean(dim=['x','y']),color='C1')
ax3.plot(P_PET.time,P_PET,'--',color='C5')
#ax.plot(dates,monthly_avgs_vci,color='C1')
ax.plot(vhi.time,vhi.mean(dim={'x','y'}),color='C3')
ax3.plot(ndvi.time,ndvi.NDVI.mean(dim=['x','y']),color='C2')
ax4.plot(p_gpm.time,p_gpm.P_mm.mean(dim=['x','y']),color='C0')
ax.set_ylabel('TCI',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)


# ---> make above time series (cleaner) for different land covers/aquifers & seasons
# GW/SM + LST + PPT/ET ++ NDVI
# SPEI + VCI ++ NDVI












###################################################
#SOIL MOISTURE COMPARISON

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
import earthpy as et
import earthpy.plot as ep
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from pathlib import Path
from glob import glob
import rasterio as rio
import csv

def read_file(file):
    with rio.open(file) as src:
        return(src.read())

path_smap = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\SMAP_SM_1km\south_africa_monthly\south_africa_monthly'
path_gldas = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_SOILMOISTURE\SMAP_Comparison'

smap_files = path_smap+ '\*.tif'
gldas_files = path_gldas+ '\*.tif'

sm_smap = sorted(list(glob(smap_files)))
sm_gldas =  sorted(list(glob(gldas_files)))

dates= pd.date_range('2015-04-01','2021-01-01' , freq='1M')

shapefile=r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
limpopo = gpd.read_file(shapefile)

#Spatially Averaged Values
monthly_avgs_smap = ['means']
os.chdir(path_smap)
for file in sm_smap:
    array = rxr.open_rasterio(file,masked=True).squeeze()
    clipped = array.rio.clip(limpopo.geometry.apply(mapping), limpopo.crs, drop=True,all_touched=True)
    monthly_avgs_smap.append(np.nanmean(clipped))


monthly_avgs_gldas = ['means']
os.chdir(path_gldas)
for file in sm_gldas:
    array = rxr.open_rasterio(file,masked=True).squeeze()
    monthly_avgs_gldas.append(np.nanmean(array))


#Monthly Time Series
fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(dates,monthly_avgs_smap[1::],color='blue')
ax.set_ylabel('Volumetric Water Content (cm^3/cm^3)',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('SMAP (5cm) Time Series',weight='bold',fontsize=15)

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(dates,monthly_avgs_gldas[1::],'--',color='blue')
ax.set_ylabel('Soil Moisture (kg/m^2)',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('GLDAS (10cm) Time Series',weight='bold',fontsize=15)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(monthly_avgs_smap[1::],monthly_avgs_gldas[1::],'*',color='blue')
ax.set_ylabel('Soil Moisture (kg/m^2)',weight='bold',fontsize=12)
ax.set_xlabel('Volumetric Water Content (cm^3/cm^3)',weight='bold',fontsize=12)
ax.set_title('GLDAS vs SMAP Comparison',weight='bold',fontsize=15)


##############################
#COMBINED

#MONTHLY

dates_sm= pd.date_range('2015-04-01','2021-01-01' , freq='1M')

fig = plt.figure(figsize=(18,6))
ax = fig.add_subplot()
ax2 = ax.twinx()
ax3 = ax.twinx()

ax.grid(color='black', linestyle='-', linewidth=0.05)
ax.plot(dates,monthly_avgs_vci,color='green',label='Vegetation Condition (VCI)')
ax.plot(dates,monthly_avgs_tci,color='red',label='Temperature Condition (TCI)')
ax.plot(dates,monthly_avgs_vhi,'--',color='orange',label='Vegetation Health (VHI)')
ax2.plot(dates_sm,monthly_avgs_smap[1::],color='blue',label='VWC (cm3/cm3) SMAP')

#ax3.plot(gldas_sm_0_200.time,gldas_sm_RZ.mean(dim=['x','y']),'--',color='blue',label='SM (kg/m^2) GLDAS')
ax3.plot(grace_scaled.time,grace_scaled.mean(dim=['lat','lon']),'--',color='black',linewidth=2)
ax.legend(loc='upper left')
ax2.legend(loc='lower right')
ax3.legend(loc='upper right')
ax.set_ylabel('Percent',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Limpopo Basin Indices (2015-2021)',weight='bold',fontsize=15)
ax.set_ylim(0,100)
