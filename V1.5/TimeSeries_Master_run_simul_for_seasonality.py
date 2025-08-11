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
import statsmodels as sm 
import statsmodels.graphics.tsaplots as tsaplots
import statsmodels.api as smapi
from sklearn.linear_model import LinearRegression
import datetime

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

path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_PET'
files = sorted(glob.glob(path+'\*.nc'))
pet_modis = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 500,"lon":500})


###################################################
#GLDAS - CLSM

path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_CLSM'
files = sorted(glob.glob(path+'\*.nc'))
#GW
clm_gw = xr.open_mfdataset(files[0],parallel=True).mm
#SOIL MOISTURE
clm_sm_rz = xr.open_mfdataset(files[3],parallel=True)['kgm-2']
clm_sm_surface = xr.open_mfdataset(files[4],parallel=True)['kgm-2']
#RUNOFF
#clm_r_surf = xr.open_mfdataset(files[2],parallel=True)['kgm-2s-1']
#clm_r_base = xr.open_mfdataset(files[1],parallel=True)['kgm-2s-1']


###################################################
#SMAP
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\SMAP_SM_1km\south_africa_monthly\south_africa_monthly'
files = sorted(glob.glob(path+'\*.nc'))
smap = xr.open_mfdataset(files[0],parallel=True,chunks={"x": 100,"y":100}).SM_vwc
smap_dates = pd.date_range('2015-04','2021-01',  freq='1M') 



#10 MINUTES TO RUN ABOVE





###################################################
#TWSA: GRACE

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


#############################################################################
#Plots of variables called above:
#Mean Precip
fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(p_chirps.time,p_gpm.P_mm.mean(dim=['x','y']),color='C1')
ax.plot(p_chirps.time,p_chirps.P_mm.mean(dim=['x','y']),color='C0')
ax.set_ylabel('Precipitation (mm)',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)

#Mean ET//PET
fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(et_modis.time,et_modis.ET_kg_m2.mean(dim=['x','y']),color='C3',label='ET')
ax.plot(pet_modis.time,pet_modis.ET_kg_m2.mean(dim=['x','y']),color='C2',label='PET')
ax.set_ylabel('ET (mm)',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)
ax.legend()

#CLM GW, SM RZ, SM Surface
fig, (axA, axB) = plt.subplots(2,1,figsize=(13,6))
ax_twinA = axA.twinx()
ax_twinB = axB.twinx()
axA.plot(pd.date_range('2003-02-01','2022-07-31', freq='1MS'),(clm_gw.mean(dim=['x','y'])),color='C0',linewidth=2,label='GW')
ax_twinA.plot(pd.date_range('2003-02-01','2022-07-31', freq='1MS'),(clm_sm_rz.mean(dim=['x','y'])),color='C2',label='SM RZ')
axB.plot(pd.date_range('2003-02-01','2022-07-31', freq='1MS'),(clm_sm_surface.mean(dim=['x','y'])),color='C1',label='SM Surf')
ax_twinB.plot(smap.time,smap.mean(dim=['x','y']),color='C3',linewidth=2,label='SM SMAP')
axA.set_ylabel('Groundwater (mm)',fontsize=12)
axB.set_ylabel('Surface SM (kg m{})'.format(get_super('-2')),fontsize=12)
ax_twinA.set_ylabel('Root Zone SM (kg m{})'.format(get_super('-2')),fontsize=12)
ax_twinB.set_ylabel('Volumetric Water Content (m{}m{})'.format(get_super('3'),get_super('-3')),fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)
ax.set_xlim([datetime.date(2003, 6,30), datetime.date(2022, 1,1)])
major_ticks = pd.date_range(start='01-01-2002',end='01-01-2022',freq='Y')
axA.set_xticks(major_ticks)
axA.set_xticklabels([])
axA.grid(which='major', axis='x', alpha=1)
axB.set_xticks(major_ticks)
axB.set_xticklabels([i for i in range(2003,2023)],fontsize=15)
axB.set_xlabel('Date')
axB.grid(which='major', axis='x', alpha=1)
ax.legend()
ax_twinA.legend()
ax_twinB.legend()
axA.legend(loc='right')
axB.legend()
plt.tight_layout()
plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\FinalFigures\Plots\sm_gw.pdf')

#LST, NDVI, ET, PPT
fig, (axA, axB) = plt.subplots(2,1,figsize=(13,6))
ax_twinA = axA.twinx()
ax_twinB = axB.twinx()
axA.plot(pd.date_range('2002-07-01','2021-12-31', freq='1MS'),(ndvi.NDVI.mean(dim=['x','y'])),'--',color='C2',linewidth=2,label='NDVI')
ax_twinA.plot(pd.date_range('2002-07-01','2021-12-31', freq='1MS'),(lst.LST_K.mean(dim=['x','y'])),color='C3',linewidth=1,label='LST')
axB.plot(pd.date_range('2002-01-01','2021-12-31', freq='1MS'),(et_modis.ET_kg_m2.mean(dim=['x','y'])),color='C0',linewidth=2,label='ET')
ax_twinB.bar(pd.date_range('2002-01-01','2021-12-31', freq='1MS'),p_gpm.P_mm.mean(dim=['x','y']),color='black',width=17,label='P')
axA.set_ylabel('NDVI',fontsize=12)
ax_twinA.set_ylabel('Temperature (K)',fontsize=12)
axB.set_ylabel('Evapotranspiration (kg m{})'.format(get_super('-2')),fontsize=12)
ax_twinB.set_ylabel('Precipitation (mm)',fontsize=12)
major_ticks = pd.date_range(start='01-01-2001',end='01-01-2022',freq='Y')
axA.set_xticks(major_ticks)
axA.set_xticklabels([])
axA.grid(which='major', axis='x', alpha=1)
axB.set_ylim(0,110)
axB.set_xticks(major_ticks)
axB.set_xticklabels([i for i in range(2002,2023)],fontsize=15)
axB.set_xlabel('Date')
axB.grid(which='major', axis='x', alpha=1)
ax.legend()
ax_twinA.legend()
ax_twinB.legend()
axA.legend(loc='right')
axB.legend()
plt.tight_layout()
plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\FinalFigures\Plots\ndvi_lst_et_ppt.pdf')


fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax_twin = ax.twinx()
ax_twin.plot(clm_sm_rz.time,clm_sm_rz.mean(dim=['x','y']),color='C0')
ax.plot(clm_gw.time,clm_gw.mean(dim=['x','y']),color='C1')
ax.set_ylabel('mm',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)

#SMAP SM
fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax_twin = ax.twinx()
ax_twin.plot(clm_sm_surface.time,clm_sm_surface.mean(dim=['x','y']),color='C0')
ax.plot(smap_dates,smap.mean(dim=['x','y']),color='C1')
ax.set_ylabel('m3/m3',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)

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
#Mask all datasets for surface water before averaging spatially to get time series && LULC MASKS
variables_to_mask = [lst.LST_K, ndvi.NDVI, clm_gw, clm_sm_rz, clm_sm_surface, p_gpm.P_mm, et_modis.ET_kg_m2, pet_modis.ET_kg_m2]
#xy_variables_to_mask = [[variable.x, variable.y] for variable in variables_to_mask]

#Sentinel-2 LULC (10 meter)
file_paths = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\Sentinel_LULC'
files = sorted(glob.glob(file_paths+'\*.nc'))
lulc = xr.open_mfdataset(files,parallel=True,chunks={"lat": 100,"lon":100}).lulc

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

path_mask = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\Sentinel_LULC\masks'
files_2017 = sorted(glob.glob(path_mask+'/2017/*.tif'))
#files_2021 = sorted(glob.glob(path_mask+'/2021/*.tif'))

lulc_2017 = [xr.open_rasterio(file)[0] for file in files_2017]
#lulc_2021 = [xr.open_rasterio(file)[0] for file in files_2021]

#2017 masks
water_masked_variables_17 = [variable.where(lulc!=water_class) for variable,lulc in zip(variables_to_mask,lulc_2017)]
#2021 masks
#water_masked_variables_21 = [variable.where(lulc!=water_class) for variable,lulc in zip(variables_to_mask,lulc_2021)]

### NEED TO INCLUDE SPEI (5 MINUTES TO OPEN)
spei_path = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\climate_indices\SPEI'
spei_files = sorted(glob.glob(spei_path+'\*spei*.nc'))


'''
resampled_lulc = lulc[0].rio.write_crs('epsg:4328').rio.reproject_match(spei_ds.rio.write_crs('epsg:4328'),resampling=Resampling.mode)
path_mask = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\Sentinel_LULC\masks'
resampled_lulc_array = np.array(resampled_lulc)
new_array = xr.DataArray(resampled_lulc_array, dims=("y", "x"), coords={"y": spei_ds.lat.rename({'lat':'y'}), "x": spei_ds.lon.rename({'lon':'x'})}, name='09_SPEI')
new_array.rio.set_crs("epsg:4326").rio.set_spatial_dims('x','y',inplace=True).rio.to_raster(path_mask+'/2017/09_SPEI.tif')

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


#2017 spatial averages
#WMV_sp_avg_17 = [variable.mean(dim={'x','y'}) for variable in water_masked_variables_17]
#2021 spatial averages
#WMV_sp_avg_21 = [variable.mean(dim={'x','y'}) for variable in water_masked_variables_21]


############################################################
#Limit dates from 02/2003 through 12/2021 to match with CLSM (starts at 02/2003) and MODIS ET (ends at 12/2021) files

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
dates= pd.date_range('2003-02-01','2022-01-01' , freq='1MS') #to match with CLSM & ET data period
tci = condition_index_tci(lst_wm)
masked_tci = [tci.where(lulc_2017[1]==value) for value in class_values]

#VEGETATION CONDITION INDEX (VCI) - 35 seconds
ndvi_wm = water_masked_variables_17[1][7:]
dates= pd.date_range('2003-02-01','2022-01-01' , freq='1MS') #to match with CLSM & ET data period
vci = condition_index_vci(ndvi_wm)
masked_vci = [vci.where(lulc_2017[1]==value) for value in class_values]


#VEGETATION HEALTH INDEX (VHI) - 40 seconds
# low alpha TEMPERATURE DEPENDENT
# high alpha MOISTURE DEPENDENT
alpha = 0.5
vhi = health_index_vhi(vci,tci,alpha)
masked_vhi = [health_index_vhi(vci,tci,alpha) for vci,tci in zip(masked_vci,masked_tci)]


##############################
#TCI PLOTS
fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(tci.time,tci.mean(dim={'x','y'}),color='red')
ax.set_ylabel('TCI',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('TCI Time Series',weight='bold',fontsize=15)

#2 minutes to run
for i, name in zip(range(0,5),class_names):
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot()
    ax.plot(masked_tci[i].time,masked_tci[i].mean(dim={'x','y'}),color='red')
    ax.set_ylabel('TCI',weight='bold',fontsize=12)
    ax.set_xlabel('Date',weight='bold',fontsize=12)
    ax.set_title('TCI {} Time Series'.format(name),weight='bold',fontsize=15)

#VCI PLOTS
fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(vci.time,vci.mean(dim={'x','y'}),color='green')
#ax.plot(tci.time,tci.mean(dim={'x','y'}),color='red')
ax.set_ylabel('VCI',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('VCI Time Series',weight='bold',fontsize=15)

#3 minutes to run
for i, name in zip(range(0,5),class_names):
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot()
    ax.plot(masked_vci[i].time,masked_vci[i].mean(dim={'x','y'}),color='green')
    ax.plot(masked_tci[i].time,masked_tci[i].mean(dim={'x','y'}),color='red')
    ax.set_ylabel('VCI',weight='bold',fontsize=12)
    ax.set_xlabel('Date',weight='bold',fontsize=12)
    ax.set_title('VCI {} Time Series'.format(name),weight='bold',fontsize=15)

#VHI PLOTS
fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(vhi.time,vhi.mean(dim={'x','y'}),color='orange')
ax.set_ylabel('VHI',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('VHI Time Series',weight='bold',fontsize=15)

#6 minutes to run
for i, name in zip(range(0,5),class_names):
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot()
    ax.plot(masked_vhi[i].time,masked_vhi[i].mean(dim={'x','y'}),color='orange')
    ax.plot(masked_tci[i].time,masked_tci[i].mean(dim={'x','y'}),color='red')
    ax.set_ylabel('VHI',weight='bold',fontsize=12)
    ax.set_xlabel('Date',weight='bold',fontsize=12)
    ax.set_title('VHI {} Time Series'.format(name),weight='bold',fontsize=15)

##############################
def anomaly_df(dataset):
    anomaly = dataset - dataset.mean(dim='time')
    d_anomaly_df = anomaly.mean(['x','y']).to_dataframe()
    d_std = d_anomaly_df.iloc[:,1].std()

    anomaly_df = (d_anomaly_df.iloc[:,1]/float(d_std))
    return anomaly_df


def monthly_anomaly(dataset):
    month_idxs=dataset.groupby('time.month').groups
    dataset_month_anomalies = [dataset.isel(time=month_idxs[i]) - dataset.isel(time=month_idxs[i]).mean(dim='time') for i in range(1,13)]
    monthly_anomalies_ds = xr.merge(dataset_month_anomalies)

    d_anomaly_df = monthly_anomalies_ds.mean(['x','y']).to_dataframe()
    d_std = d_anomaly_df.iloc[:,1].std()
    
    anomaly_df = (d_anomaly_df.iloc[:,1]/float(d_std))
    return anomaly_df
###############################


lst_wm = water_masked_variables_17[0][7:]
ndvi_wm = water_masked_variables_17[1][7:]
clm_gw_wm = water_masked_variables_17[2][:-7]
clm_sm_rz_wm = water_masked_variables_17[3][:-7]
clm_sm_surface_wm = water_masked_variables_17[4][:-7]
p_gpm_wm = water_masked_variables_17[5][13:]
et_modis_wm = water_masked_variables_17[6][13:]
pet_modis_wm = water_masked_variables_17[7][13:]


variables = [lst_wm,ndvi_wm,clm_gw_wm,clm_sm_rz_wm,clm_sm_surface_wm,p_gpm_wm]
anomalies_df = [anomaly_df(variable) for variable in variables]
monthly_anomalies_df = [monthly_anomaly(variable) for variable in variables]
variable_names = ['01_LST','02_NDVI','03_CLM_GW','04_CLM_RZ_SM','05_CLM_Surface_SM','06_P_GPM']
[monthly_df.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\{}.csv'.format(name)) 
 for monthly_df,name in zip(monthly_anomalies_df,variable_names)]
[df.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\{}.csv'.format(name)) 
 for df,name in zip(anomalies_df,variable_names)]

#run separately for ET/PET -- cannot get anomaly csv for ET/PET (too much memory??)
et_anomaly_month_df = monthly_anomaly(et_modis_wm)
et_anomaly_month_df.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\07_ET.csv')
pet_anomaly_month_df = monthly_anomaly(pet_modis_wm)
pet_anomaly_month_df.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\08_PET.csv')

#run separately for TCI/VCI/VHI
TCI_anomaly_month_df = monthly_anomaly(tci)
TCI_anomaly_month_df.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\09_TCI.csv')
VCI_anomaly_month_df = monthly_anomaly(vci)
VCI_anomaly_month_df.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\10_VCI.csv')
VHI_anomaly_month_df = monthly_anomaly(vhi.rename('vhi'))
VHI_anomaly_month_df.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\11_VHI.csv')


#[0].iloc[:,1] LST
#[1].iloc[:,1] NDVI
#[2].iloc[:,1] CLM GW
#[3].iloc[:,1] CLM SM RZ
#[4].iloc[:,1] CLM SM Surface
#[5].iloc[:,1] GPM PPT
#[6].iloc[:,1] ET MODIS
#[7].iloc[:,1] PET MODIS


#LEFT OFF on 10/26 12:28am --- dataframes for all lulc masked variables? --> anomalies? --> lags/trends?
#still need to process VHI anomalies csv & ET/PET csv; masked SPEI with 08_PET lulc2017?
#still need to get aquifer map masks -- tricky because of multiple different values
#average borehole data based on proximity & aquifer type???

##############################
#DATAFRAME ANALYSES

#Lag Analysis
#calculate cross correlation
def analyze_lag(var1,var2):
    p_vals = []
    r_sqs = []
    for i in range(0,25):
        if i == 0:
            variable1 = var1
            variable2 = var2
        else:
            variable1 = var1[:-i]
            variable2 = var2[i::]

        x = np.array(variable1)
        X = smapi.add_constant(x)
        y = np.array(variable2)
        est = smapi.OLS(y, X)
        model = est.fit()
        r_sqs.append(round(model.rsquared, 3))
        p_vals.append(round(model.pvalues[-1], 3))

    lag_dataframe = pd.DataFrame({'R_sq':r_sqs,'P-Val':p_vals})

    return lag_dataframe

def cross_correlation(var1,var2):
    variable1 = var1
    variable2 = var2
    cross_correlations = smapi.tsa.stattools.ccf(variable1, variable2, adjusted=False)

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot()
    ax.scatter(range(0,len(cross_correlations)),cross_correlations)

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot()
    ax.scatter(variable1,variable2)

    index_lag_max = np.where(cross_correlations==np.max(cross_correlations))
    index_lag_min = np.where(cross_correlations==np.min(cross_correlations))
    print(np.max(cross_correlations), index_lag_max, np.min(cross_correlations), index_lag_min)

    return cross_correlations

def linear_plot(independent,dependent,ind_label,d_label,color_choice):
    y = np.array(dependent)
    x = np.array(independent)
    X = smapi.add_constant(x)
    est = smapi.OLS(y, X)
    model = est.fit()
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot()
    ax.scatter(independent,dependent,color=color_choice)
    ax.plot([], [], ' ', label='R{} = {}'.format(get_super('2'),round(model.rsquared, 3)))
    ax.plot([], [], ' ', label='p-val = {}'.format(round(model.pvalues[-1], 3)))
    ax.legend(loc='upper right')
    ax.set_xlabel(ind_label)
    ax.set_ylabel(d_label)

###########################SPEI vs. VHI
spei_path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\climate_indices_data\climate_indices'
spei_files = sorted(glob.glob(spei_path+'\*spei*.nc'))
spei_ds = xr.open_mfdataset(spei_files,parallel=True,chunks={"lat": 100,"lon":100})

spei_1_df = spei_ds.spei_gamma_01.mean(dim=['lat','lon'])[0:].to_dataframe()
spei_3_df = spei_ds.spei_gamma_03.mean(dim=['lat','lon'])[2:].to_dataframe()
spei_6_df = spei_ds.spei_gamma_06.mean(dim=['lat','lon'])[5:].to_dataframe()
spei_12_df = spei_ds.spei_gamma_12.mean(dim=['lat','lon'])[11:].to_dataframe()
spei_48_df = spei_ds.spei_gamma_48.mean(dim=['lat','lon'])[47:].to_dataframe()


linear_plot(vhi_df.vhi[5:],spei_6_df,'VHI','SPEI-6','black')
linear_plot(vhi_df.vhi[2:],spei_3_df,'VHI','SPEI-3','black')
linear_plot(vhi_df.vhi[0:],spei_1_df,'VHI','SPEI-1','black')
linear_plot(vhi_df.vhi[11:],spei_12_df,'VHI','SPEI-12','black')
linear_plot(vhi_df.vhi[47:],spei_48_df,'VHI','SPEI-48','black')

#############################################################################################
#SPEI - 1,3,6,12
plt.rc('font', size = 22)
speis = [spei_1_df.spei_gamma_01, spei_3_df.spei_gamma_03,spei_6_df.spei_gamma_06,spei_12_df.spei_gamma_12 ]
speis_dates = [pd.date_range('2003-02-01','2021-12-31' , freq='1MS'),pd.date_range('2003-04-01','2021-12-31' , freq='1MS'),pd.date_range('2003-07-01','2021-12-31' , freq='1MS'),pd.date_range('2004-01-01','2021-12-31' , freq='1MS')]
fig, (ax, axA, axB, axC, axD) = plt.subplots(5,1,figsize=(17,20))
ax_twin = ax.twinx()
[axis.plot(date,spei,color='black') for axis,date,spei in zip([axA,axB,axC,axD],speis_dates,speis)]
[axis.fill_between(date,spei, 0, where=(spei<0), color='C3', alpha=.5) for axis,date,spei in zip([axA,axB,axC,axD],speis_dates,speis)]

major_ticks = pd.date_range(start='01-01-2001',end='01-01-2022',freq='Y')
[axis.set_xticks(major_ticks) for axis in [ax,ax_twin,axA,axB,axC,axD]]
[axis.set_xticklabels([]) for axis in [ax,ax_twin,axA,axB,axC]]
[axis.grid(which='major', axis='x', alpha=1) for axis in [ax,ax_twin,axA,axB,axC,axD]]
[axis.axhline(0,color='black',linewidth=0.4) for axis in [ax_twin,axA,axB,axC,axD]]
[axis.set_ylim(-3,3) for axis in [axA,axB,axC,axD]]
axD.set_xlabel('Date')
axD.set_xticks(major_ticks)
axD.set_xticklabels([i for i in range(2002,2023)],fontsize=18)

[axis.set_ylabel(spei_title) for axis,spei_title in zip([axA,axB,axC,axD],['SPEI-1','SPEI-3','SPEI-6','SPEI-12'])]
ax.set_ylabel('VHI')
ax_twin.set_ylabel('Groundwater Anomaly')
ax.yaxis.label.set_color('C1')
ax_twin.yaxis.label.set_color('C0')

ax.plot(pd.date_range('2003-02-01','2021-12-31' , freq='1MS'),vhi_df.vhi,color='C1',label='VHI')
ax.set_ylim(0,100)
ax_twin.plot(pd.date_range('2003-02-01','2021-12-31' , freq='1MS'),anoms[2].mm,color='C0',linewidth=2,label='GW')
#ax.legend()
#ax_twin.legend(loc='upper left')
plt.tight_layout()
plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\FinalFigures\Plots\monthly_indices.pdf')

lag_6SPEI_vhi = analyze_lag(vhi_df.vhi[5:],spei_6_df)
lag_3SPEI_vhi = analyze_lag(vhi_df.vhi[2:],spei_3_df)
lag_1SPEI_vhi = analyze_lag(vhi_df.vhi[0:],spei_1_df)

cross_6SPEI_vhi = cross_correlation(vhi_df.vhi[5:],spei_6_df)
cross_3SPEI_vhi = cross_correlation(vhi_df.vhi[2:],spei_3_df)
cross_1SPEI_vhi = cross_correlation(vhi_df.vhi[0:],spei_1_df)


#ANOMALIES ANOMALIES ANOMALIES ANOMALIES ANOMALIES ANOMALIES ANOMALIES ANOMALIES ANOMALIES
################################################################################################
################################################################################################
################################################################################################
################################################################################################

#ANOMALIES (monthly) -- need to calculate ET/PET manually 
anom_path = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\*.csv'
anom_files = sorted(glob.glob(anom_path))

anoms = [pd.read_csv(file) for file in anom_files]

#TO DATAFRAMES (ENTIRE BASIN)
tci_df = tci.rename('tci').mean(dim={'x','y'}).to_dataframe()
vci_df = vci.rename('vci').mean(dim={'x','y'}).to_dataframe()
vhi_df = vhi.rename('vhi').mean(dim={'x','y'}).to_dataframe()
#p_chirps_df = p_chirps.mean(dim={'x','y'}).to_dataframe()
lst_df = lst_wm.mean(dim={'x','y'}).to_dataframe()
ndvi_df = ndvi_wm.mean(dim={'x','y'}).to_dataframe()
p_gpm_df = p_gpm_wm.mean(dim={'x','y'}).to_dataframe()
clm_gw_df = clm_gw_wm.mean(dim={'x','y'}).to_dataframe()
clm_rz_df = clm_sm_rz_wm.mean(dim={'x','y'}).to_dataframe()
clm_sm_df = clm_sm_surface_wm.mean(dim={'x','y'}).to_dataframe()
et_modis_df = et_modis_wm.mean(dim={'x','y'}).to_dataframe()

variables = [lst_wm,ndvi_wm,clm_gw_wm,clm_sm_rz_wm,clm_sm_surface_wm,p_gpm_wm,et_modis_wm,pet_modis_wm]
masked_variables = [[variable.where(lulc==value) for value in class_values] for variable,lulc in zip(variables,lulc_2017)]
class_names = ['trees','rangeland','crops','urban','bare']

all_masked_variables = [masked_spei, masked_tci,masked_vci,masked_vhi,masked_variables[0],masked_variables[1],masked_variables[2],
masked_variables[3],masked_variables[4],masked_variables[5],masked_variables[6],masked_variables[7]]


###########################NDVI vs. GW // PPT
lag_gw_ndvi = analyze_lag(ndvi_df.NDVI,clm_gw_df.mm)
lag_gw_ndvi_anom = analyze_lag(anoms[1].NDVI,anoms[2].mm)
#no lag
linear_plot(ndvi_df.NDVI,clm_gw_df.mm,'NDVI','GW','black')
linear_plot(anoms[1].NDVI,anoms[2].mm,'NDVI Anomaly','GW Anomaly','blue')
#lags
[linear_plot(ndvi_df.NDVI[:-i],clm_gw_df.mm[i:],'NDVI','GW','black') for i in range(1,13)]
[linear_plot(anoms[1].NDVI[:-i],anoms[2].mm[i:],'NDVI Anomaly','GW Anomaly','blue') for i in range(1,13)]
#Based on above plots, there is a one-month lag between vegetation condition (moisture) and GW levels for the basin
#positive relationship


fig = plt.figure(figsize=(15,4))
plt.rc('font', size = 15)
ax = fig.add_subplot()
ax_twin = ax.twinx()
ax_twin.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),ndvi_df.NDVI,color='C2')
ax.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),clm_gw_df.mm,color='C0')
#ax.bar(p_chirps_df.index,p_chirps_df.P_mm,color='C3',label='P',width=14)
ax.set_ylabel('GW (mm)',weight='bold',fontsize=12)
#ax.set_ylim([0,1000])
ax_twin.set_ylabel('NDVI',weight='bold',fontsize=12)
#ax_twin.set_ylim([0,1])
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.yaxis.label.set_color('C0')
ax_twin.yaxis.label.set_color('C2')
ax.set_title('Time Series',weight='bold',fontsize=15)
ax.set_xlim([datetime.date(2003, 6,30), datetime.date(2022, 1,1)])
major_ticks = pd.date_range(start='01-01-2002',end='01-01-2022',freq='Y')
ax.set_xticks(major_ticks)
ax.set_xticklabels([i for i in range(2003,2023)],rotation=35,fontsize=10)
ax.grid(which='major', axis='x', alpha=1)

fig = plt.figure(figsize=(15,4))
plt.rc('font', size = 15)
ax = fig.add_subplot()
ax_twin = ax.twinx()
ax_twin.plot(anoms[1].index,anoms[1].NDVI,color='C2',label='NDVI')
ax.plot(anoms[2].index,anoms[2].mm,color='C0',label='GW')
ax.set_ylabel('GW Anomaly',weight='bold',fontsize=12)
ax_twin.set_ylabel('NDVI Anomaly',weight='bold',fontsize=12)
ax.set_ylim([-3.5,3.5])
ax_twin.set_ylim([-3.5,3.5])
#ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_xlim(0,226)
ax.yaxis.label.set_color('C0')
plt.xticks([])
ax_twin.yaxis.label.set_color('C2')
ax.set_title('GW vs. NDVI',weight='bold',fontsize=15)


lag_ppt_ndvi = analyze_lag(ndvi_df.NDVI,p_gpm_df.P_mm)
lag_ppt_ndvi_anom = analyze_lag(anoms[1].NDVI,anoms[5].P_mm)
#no lag
linear_plot(ndvi_df.NDVI,p_gpm_df.P_mm,'NDVI','P','black')
linear_plot(anoms[1].NDVI,anoms[5].P_mm,'NDVI Anomaly','P Anomaly','blue')
#lags
[linear_plot(ndvi_df.NDVI[:-i],p_gpm_df.P_mm[i:],'NDVI','P','black') for i in range(1,13)]
[linear_plot(anoms[1].NDVI[:-i],anoms[5].P_mm[i:],'NDVI Anomaly','P Anomaly','blue') for i in range(1,13)]
#Based on above plots, there is a one-month lag between vegetation condition (moisture) and GW levels for the basin
#positive relationship


fig = plt.figure(figsize=(15,4))
plt.rc('font', size = 15)
ax = fig.add_subplot()
ax_twin = ax.twinx()
ax_twin.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),ndvi_df.NDVI,color='C2')
ax.bar(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),p_gpm_df.P_mm,color='C0',width=14)
#ax.bar(p_chirps_df.index,p_chirps_df.P_mm,color='C3',label='P',width=14)
ax.set_ylabel('P (mm)',weight='bold',fontsize=12)
ax.set_ylim([0,230])
#ax.set_ylim([0,1000])
ax_twin.set_ylabel('NDVI',weight='bold',fontsize=12)
#ax_twin.set_ylim([0,1])
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.yaxis.label.set_color('C0')
ax_twin.yaxis.label.set_color('C2')
ax.set_title('Time Series',weight='bold',fontsize=15)
ax.set_xlim([datetime.date(2003, 6,30), datetime.date(2022, 1,1)])
major_ticks = pd.date_range(start='01-01-2002',end='01-01-2022',freq='Y')
ax.set_xticks(major_ticks)
ax.set_xticklabels([i for i in range(2003,2023)],rotation=35,fontsize=10)
ax.grid(which='major', axis='x', alpha=1)

fig = plt.figure(figsize=(15,4))
plt.rc('font', size = 15)
ax = fig.add_subplot()
ax_twin = ax.twinx()
ax_twin.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),anoms[1].NDVI,color='C2',label='NDVI')
ax.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),anoms[5].P_mm,color='C0',label='P')
ax.set_ylabel('P Anomaly',weight='bold',fontsize=12)
ax_twin.set_ylabel('NDVI Anomaly',weight='bold',fontsize=12)
ax.set_ylim([-4,4])
ax_twin.set_ylim([-4,4])
#ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.yaxis.label.set_color('C0')
plt.xticks([])
ax_twin.yaxis.label.set_color('C2')


###########################LST vs. GW // ET
lag_gw_lst = analyze_lag(lst_df.LST_K,clm_gw_df.mm)
lag_gw_lst_anom = analyze_lag(anoms[0].LST_K,anoms[2].mm)
#no lag
linear_plot(lst_df.LST_K,clm_gw_df.mm,'LST','GW','black')
linear_plot(anoms[0].LST_K,anoms[2].mm,'LST Anomaly','GW Anomaly','blue')
#lags
[linear_plot(lst_df.LST_K[:-i],clm_gw_df.mm[i:],'LST','GW','black') for i in range(1,13)]
[linear_plot(anoms[0].LST_K[:-i],anoms[2].mm[i:],'LST Anomaly','GW Anomaly','blue') for i in range(1,13)]
#Based on above plots, there is a one-month lag between temperture condition (T) and GW levels for the basin
#negative relationship


fig = plt.figure(figsize=(15,4))
plt.rc('font', size = 15)
ax = fig.add_subplot()
ax_twin = ax.twinx()
ax_twin.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),lst_df.LST_K,color='C3')
ax.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),clm_gw_df.mm,color='C0')
#ax.bar(p_chirps_df.index,p_chirps_df.P_mm,color='C3',label='P',width=14)
ax.set_ylabel('GW (mm)',weight='bold',fontsize=12)
#ax.set_ylim([0,1000])
ax_twin.set_ylabel('LST (K)',weight='bold',fontsize=12)
#ax_twin.set_ylim([0,1])
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.yaxis.label.set_color('C0')
ax_twin.yaxis.label.set_color('C3')
ax.set_title('Time Series',weight='bold',fontsize=15)
ax.set_xlim([datetime.date(2003, 6,30), datetime.date(2022, 1,1)])
major_ticks = pd.date_range(start='01-01-2002',end='01-01-2022',freq='Y')
ax.set_xticks(major_ticks)
ax.set_xticklabels([i for i in range(2003,2023)],rotation=35,fontsize=10)
ax.grid(which='major', axis='x', alpha=1)

fig = plt.figure(figsize=(15,4))
plt.rc('font', size = 15)
ax = fig.add_subplot()
ax_twin = ax.twinx()
ax_twin.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),anoms[0].LST_K,color='C3')
ax.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),anoms[2].mm,color='C0')
ax.set_ylabel('GW Anomaly',weight='bold',fontsize=12)
ax_twin.set_ylabel('LST Anomaly',weight='bold',fontsize=12)
ax.set_ylim([-3.5,3.5])
ax_twin.set_ylim([-3.5,3.5])
#ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.yaxis.label.set_color('C0')
plt.xticks([])
ax_twin.yaxis.label.set_color('C3')



lag_et_lst = analyze_lag(lst_df.LST_K,et_modis_df.ET_kg_m2)
lag_et_lst_anom = analyze_lag(anoms[0].LST_K,anoms[5].ET_kg_m2)
#no lag
linear_plot(lst_df.LST_K,et_modis_df.ET_kg_m2,'LST','ET','black')
linear_plot(anoms[0].LST_K,anoms[5].ET_kg_m2,'NDVI Anomaly','P Anomaly','blue')
#lags
[linear_plot(lst_df.LST_K[:-i],et_modis_df.ET_kg_m2[i:],'LST','ET','black') for i in range(1,13)]
[linear_plot(anoms[0].NDVI[:-i],anoms[5].ET_kg_m2[i:],'NDVI Anomaly','P Anomaly','blue') for i in range(1,13)]
#Based on above plots, there is a one-month lag between vegetation condition (moisture) and GW levels for the basin
#positive relationship

fig = plt.figure(figsize=(15,4))
plt.rc('font', size = 15)
ax = fig.add_subplot()
ax_twin = ax.twinx()
ax_twin.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),lst_df.LST_K,color='C3')
ax.bar(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),et_modis_df.ET_kg_m2,color='C0',width=14)
#ax.bar(p_chirps_df.index,p_chirps_df.P_mm,color='C3',label='P',width=14)
ax.set_ylabel('ET (mm)',weight='bold',fontsize=12)
ax.set_ylim([0,150])
#ax.set_ylim([0,1000])
ax_twin.set_ylabel('LST',weight='bold',fontsize=12)
#ax_twin.set_ylim([0,1])
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.yaxis.label.set_color('C0')
ax_twin.yaxis.label.set_color('C3')
ax.set_title('Time Series',weight='bold',fontsize=15)
ax.set_xlim([datetime.date(2003, 6,30), datetime.date(2022, 1,1)])
major_ticks = pd.date_range(start='01-01-2002',end='01-01-2022',freq='Y')
ax.set_xticks(major_ticks)
ax.set_xticklabels([i for i in range(2003,2023)],rotation=35,fontsize=10)
ax.grid(which='major', axis='x', alpha=1)

fig = plt.figure(figsize=(15,4))
plt.rc('font', size = 15)
ax = fig.add_subplot()
ax_twin = ax.twinx()
ax_twin.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),anoms[1].NDVI,color='C2',label='NDVI')
ax.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),anoms[5].P_mm,color='C0',label='P')
ax.set_ylabel('GW Anomaly',weight='bold',fontsize=12)
ax_twin.set_ylabel('NDVI Anomaly',weight='bold',fontsize=12)
ax.set_ylim([-4,4])
ax_twin.set_ylim([-4,4])
#ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.yaxis.label.set_color('C0')
plt.xticks([])
ax_twin.yaxis.label.set_color('C2')


###########################NDVI vs. LST // VCI vs. TCI
lag_lst_ndvi = analyze_lag(ndvi_df.NDVI,lst_df.LST_K)
lag_lst_ndvi_anom = analyze_lag(anoms[1].NDVI,anoms[0].LST_K)
#no lag
linear_plot(ndvi_df.NDVI,lst_df.LST_K,'NDVI','LST','black')
linear_plot(anoms[1].NDVI,anoms[0].LST_K,'NDVI Anomaly','LST Anomaly','blue')
#lags
[linear_plot(ndvi_df.NDVI[:-i],lst_df.LST_K[i:],'NDVI','LST','black') for i in range(1,13)]
[linear_plot(anoms[1].NDVI[:-i],anoms[0].LST_K[i:],'NDVI Anomaly','LST Anomaly','blue') for i in range(1,13)]
#Based on above plots, there is a one-month lag between vegetation condition (moisture) and GW levels for the basin

lag_lst_ndvi = analyze_lag(vci_df.vci,tci_df.tci)
#no lag
linear_plot(vci_df.vci,tci_df.tci,'VCI','TCI','blue')
#lags
[linear_plot(vci_df.vci[:-i],tci_df.tci[i:],'VCI','TCI','blue') for i in range(1,13)]

fig = plt.figure(figsize=(15,4))
plt.rc('font', size = 15)
ax = fig.add_subplot()
ax_twin = ax.twinx()
ax_twin.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS') ,vci_df.vci,color='C2')
ax.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS') ,tci_df.tci,color='C3')
#ax.bar(p_chirps_df.index,p_chirps_df.P_mm,color='C3',label='P',width=14)
ax.set_ylabel('TCI',weight='bold',fontsize=12)
ax.set_ylim([0,100])
ax_twin.set_ylabel('VCI',weight='bold',fontsize=12)
ax_twin.set_ylim([0,100])
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax_twin.yaxis.label.set_color('C2')
ax.yaxis.label.set_color('C3')
ax.set_xlim([datetime.date(2003, 6,30), datetime.date(2022, 1,1)])
major_ticks = pd.date_range(start='01-01-2002',end='01-01-2022',freq='Y')
ax.set_xticks(major_ticks)
ax.set_xticklabels([i for i in range(2003,2023)],rotation=35,fontsize=10)
ax.grid(which='major', axis='x', alpha=1)

lag_lst_ndvi = analyze_lag(ndvi_df.NDVI,lst_df.LST_K)
lag_ppt_ndvi = analyze_lag(ndvi_df.NDVI,p_gpm_df.P_mm)
lag_et_ndvi = analyze_lag(ndvi_df.NDVI,et_modis_df.ET_kg_m2)

#Cross Correlations

fig = plt.figure(figsize=(15,4))
plt.rc('font', size = 15)
ax = fig.add_subplot()
ax_twin = ax.twinx()
ax.plot(pd.date_range('2003-02-01','2021-12-31' , freq='1MS'),vci_df.vci,color='C1',linewidth=2)
#ax2.plot(pd.date_range('2003-07-01','2021-12-31' , freq='1MS'),vhi_df.vhi[5:],color='C2')
ax_twin.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),anoms[0].LST_K,color='C3', label='LST')
ax_twin.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),anoms[1].NDVI,color='C2',label = 'NDVI')
#ax_twin.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),anoms[2].mm,color='C0', label = 'GW')
#ax.bar(p_chirps_df.index,p_chirps_df.P_mm,color='C3',label='P',width=14)
ax.set_ylabel('VCI',weight='bold',fontsize=12)
#ax.set_ylim([0,1000])
ax_twin.set_ylabel('Z-Score',weight='bold',fontsize=12)
#ax_twin.set_ylim([0,1])
ax.set_xlabel('Date',weight='bold',fontsize=12)
#ax.yaxis.label.set_color('C0')
ax.set_ylim([0,100])
ax_twin.set_ylim([-3.5,3.5])
ax.set_xlim([datetime.date(2003, 6,30), datetime.date(2022, 1,1)])
major_ticks = pd.date_range(start='01-01-2002',end='01-01-2022',freq='Y')
ax.set_xticks(major_ticks)
ax.set_xticklabels([i for i in range(2003,2023)],rotation=35,fontsize=10)
ax.grid(which='major', axis='x', alpha=1)
ax_twin.legend(fontsize=12)

fig = plt.figure(figsize=(15,4))
plt.rc('font', size = 15)
ax = fig.add_subplot()
ax_twin = ax.twinx()
ax_twin.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),lst_df.LST_K,color='C3',label='NDVI')
ax.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),clm_gw_df.mm,color='C0',label='GW')
#ax.bar(p_chirps_df.index,p_chirps_df.P_mm,color='C3',label='P',width=14)
ax.set_ylabel('GW (mm)',weight='bold',fontsize=12)
#ax.set_ylim([0,1000])
ax_twin.set_ylabel('LST (K)',weight='bold',fontsize=12)
#ax_twin.set_ylim([0,1])
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.yaxis.label.set_color('C0')
ax_twin.yaxis.label.set_color('C3')
ax.set_xlim([datetime.date(2003, 6,30), datetime.date(2022, 1,1)])
major_ticks = pd.date_range(start='01-01-2002',end='01-01-2022',freq='Y')
ax.set_xticks(major_ticks)
ax.set_xticklabels([i for i in range(2003,2023)],rotation=35,fontsize=10)
ax.grid(which='major', axis='x', alpha=1)


fig = plt.figure(figsize=(15,4))
plt.rc('font', size = 15)
ax = fig.add_subplot()
ax_twin = ax.twinx()
ax_twin.plot(spei_6_df.index,spei_6_df,color='C1',label='SPEI')
ax.bar(p_gpm_df.P_mm[5:].index,p_gpm_df.P_mm[5:],color='C0',label='P',width=12)
#ax.bar(p_chirps_df.index,p_chirps_df.P_mm,color='C3',label='P',width=14)
ax.set_ylabel('P (mm)',weight='bold',fontsize=12)
ax.set_ylim([0,250])
ax_twin.set_ylabel('SPEI',weight='bold',fontsize=12)
ax_twin.set_ylim([-2.5,2.5])
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)

cross_ppt_vhi = cross_correlation(p_gpm_df.P_mm,vhi_df.vhi)
cross_ppt_6spei = cross_correlation(p_gpm_df.P_mm[5:],spei_6_df)

fig = plt.figure(figsize=(15,4))
plt.rc('font', size = 15)
ax = fig.add_subplot()
ax_twin = ax.twinx()
ax_twin.plot(spei_6_df.index,spei_6_df,color='C1',label='SPEI')
ax.bar(p_gpm_df.P_mm[5:].index,p_gpm_df.P_mm[5:],color='C0',label='P',width=12)
#ax.bar(p_chirps_df.index,p_chirps_df.P_mm,color='C3',label='P',width=14)
ax.set_ylabel('P (mm)',weight='bold',fontsize=12)
ax.set_ylim([0,250])
ax_twin.set_ylabel('SPEI',weight='bold',fontsize=12)
ax_twin.set_ylim([-2.5,2.5])
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)



############################SM vs. GW vs. GRACE TWSA

#Surface
lag_gw_sm = analyze_lag(clm_sm_df['kgm-2'],clm_gw_df.mm)
lag_gw_sm_anom = analyze_lag(anoms[4]['kgm-2'],anoms[2].mm)
#no lag
linear_plot(clm_sm_df['kgm-2'],clm_gw_df.mm,'SM','GW','black')
linear_plot(anoms[4]['kgm-2'],anoms[2].mm,'SM Anomaly','GW Anomaly','blue')
#lags
[linear_plot(clm_sm_df['kgm-2'][:-i],clm_gw_df.mm[i:],'SM','GW','black') for i in range(1,13)]
[linear_plot(anoms[4]['kgm-2'][:-i],anoms[2].mm[i:],'SM Anomaly','GW Anomaly','blue') for i in range(1,13)]
#1 month lag is strongest


#Root Zone
lag_gw_sm = analyze_lag(clm_rz_df['kgm-2'],clm_gw_df.mm)
lag_gw_sm_anom = analyze_lag(anoms[3]['kgm-2'],anoms[2].mm)
#no lag
linear_plot(clm_rz_df['kgm-2'],clm_gw_df.mm,'SM','GW','black')
linear_plot(anoms[3]['kgm-2'],anoms[2].mm,'SM Anomaly','GW Anomaly','blue')
#lags
[linear_plot(clm_rz_df['kgm-2'][:-i],clm_gw_df.mm[i:],'SM','GW','black') for i in range(1,13)]
[linear_plot(anoms[3]['kgm-2'][:-i],anoms[2].mm[i:],'SM Anomaly','GW Anomaly','blue') for i in range(1,13)]


fig = plt.figure(figsize=(15,4))
plt.rc('font', size = 15)
ax = fig.add_subplot()
ax_twin = ax.twinx()
ax_twin.plot(clm_rz_df.index,clm_rz_df['kgm-2'],color='blue',label='SM')
ax.plot(clm_gw_df.index,clm_gw_df.mm,color='C0',label='GW')
#ax.bar(p_chirps_df.index,p_chirps_df.P_mm,color='C3',label='P',width=14)
ax.set_ylabel('GW (mm)',weight='bold',fontsize=12)
#ax.set_ylim([0,1000])
ax_twin.set_ylabel('SM',weight='bold',fontsize=12)
#ax_twin.set_ylim([0,1])
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)

fig = plt.figure(figsize=(15,4))
plt.rc('font', size = 15)
ax = fig.add_subplot()
ax_twin = ax.twinx()
ax_twin.plot(clm_sm_df.index,clm_sm_df['kgm-2'],color='blue',label='SM')
ax.plot(clm_gw_df.index,clm_gw_df.mm,color='C0',label='GW')
#ax.bar(p_chirps_df.index,p_chirps_df.P_mm,color='C3',label='P',width=14)
ax.set_ylabel('GW (mm)',weight='bold',fontsize=12)
#ax.set_ylim([0,1000])
ax_twin.set_ylabel('SM',weight='bold',fontsize=12)
#ax_twin.set_ylim([0,1])
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)

fig = plt.figure(figsize=(15,4))
plt.rc('font', size = 15)
ax = fig.add_subplot()
ax_twin = ax.twinx()
ax_twin.plot(anoms[3].index,anoms[3]['kgm-2'],color='blue',label='SM')
ax.plot(anoms[3].index,anoms[2].mm,color='C0',label='GW')
#ax.bar(p_chirps_df.index,p_chirps_df.P_mm,color='C3',label='P',width=14)
ax.set_ylabel('GW Anomaly',weight='bold',fontsize=12)
ax_twin.set_ylabel('SM',weight='bold',fontsize=12)
ax.set_ylim([-3.5,3.5])
ax_twin.set_ylim([-3.5,3.5])
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)

fig = plt.figure(figsize=(15,4))
plt.rc('font', size = 15)
ax = fig.add_subplot()
ax_twin = ax.twinx()
ax_twin.plot(anoms[3].index,anoms[4]['kgm-2'],color='blue',label='SM')
ax.plot(anoms[3].index,anoms[2].mm,color='C0',label='GW')
#ax.bar(p_chirps_df.index,p_chirps_df.P_mm,color='C3',label='P',width=14)
ax.set_ylabel('GW Anomaly',weight='bold',fontsize=12)
ax_twin.set_ylabel('SM',weight='bold',fontsize=12)
ax.set_ylim([-3.5,3.5])
ax_twin.set_ylim([-3.5,3.5])
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)

fig,ax = plt.subplots()
ax.plot(grace_limpopo_new_xr.time, grace_limpopo_new*1000)
ax.set_ylabel('TWSA (mm)',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)

fig = plt.figure(figsize=(15,4))
plt.rc('font', size = 15)
ax = fig.add_subplot()
ax_twin = ax.twinx()
ax_twin.plot(grace_limpopo_new_xr.time, grace_limpopo_new*1000)
ax.plot(pd.date_range('2003-02','2021-11',  freq='1M') ,anoms[2].mm,color='C0',label='GW')
#ax.bar(p_chirps_df.index,p_chirps_df.P_mm,color='C3',label='P',width=14)
ax.set_ylabel('GW Anomaly',weight='bold',fontsize=12)
ax_twin.set_ylabel('TWSA',weight='bold',fontsize=12)
ax.set_ylim([-3.5,3.5])
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)



cross_lst_ndvi = cross_correlation(ndvi_df.NDVI,lst_df.LST_K)












#ACF & PACF
variable1 = 
variable2 = 

smapi.graphics.tsa.plot_acf(variable1, lags=24)
plt.show()
smapi.graphics.tsa.plot_pacf(variable1, lags=24)
plt.show()

smapi.graphics.tsa.plot_acf(variable2, lags=24)
plt.show()
smapi.graphics.tsa.plot_pacf(variable2, lags=24)
plt.show()




##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
#SPATIALLY VARIABLE ANOMALIES -- MASKED by LAND COVER & AQUIFER CHARACTERISTICS
#MONTHLY & SEASONAL

#CHOOSE 12 (GW) or 6 (VHI)
spei_ds_06 = xr.open_mfdataset(spei_files,parallel=True,chunks={"lat": 100,"lon":100}).spei_gamma_06.rename({'lat':'y','lon':'x'})
masked_spei_06 = [spei_ds_06.where(lulc_2017[-1]==value) for value in class_values]

spei_ds_12 = xr.open_mfdataset(spei_files,parallel=True,chunks={"lat": 100,"lon":100}).spei_gamma_12.rename({'lat':'y','lon':'x'})
masked_spei_12 = [spei_ds_12.where(lulc_2017[-1]==value) for value in class_values]

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

masked_variables = [[variable.where(lulc==value) for value in class_values] for variable,lulc in zip(variables,lulc_2017)]
class_names = ['trees','rangeland','crops','urban','bare']


#Only water-masked: 1.5 min to run
for i, name,unit,ylims in zip(range(0,8),variable_names,unit_names,variable_ylims):
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot()
    ax.plot(variables[i].time,variables[i].mean(dim={'x','y'}),color=colors[i])
    ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
    ax.set_ylim(ylims)
    ax.set_xlabel('Date',weight='bold',fontsize=12)
    ax.set_title('{} Time Series'.format(name),weight='bold',fontsize=15)


plt.rc('font', size = 12)
#CLM GW, SM RZ, SM Surface
fig, (axA, axB) = plt.subplots(2,1,figsize=(13,6))
ax_twinA = axA.twinx()
ax_twinB = axB.twinx()
axA.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),(variables[2].mean(dim=['x','y'])),color='C0',linewidth=2,label='GW')
ax_twinA.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),(variables[3].mean(dim=['x','y'])),color='C2',label='SM RZ')
axB.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),(variables[4].mean(dim=['x','y'])),color='C1',label='SM Surf')
ax_twinB.plot(smap.time,smap.mean(dim=['x','y']),color='C3',linewidth=2,label='SM SMAP')
axA.set_ylabel('Groundwater (mm)',fontsize=12)
axB.set_ylabel('Surface SM (kg m{})'.format(get_super('-2')),fontsize=12)
ax_twinA.set_ylabel('Root Zone SM (kg m{})'.format(get_super('-2')),fontsize=12)
ax_twinB.set_ylabel('Volumetric Water Content (m{}m{})'.format(get_super('3'),get_super('-3')),fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)
ax.set_xlim([datetime.date(2003, 6,30), datetime.date(2022, 1,1)])
major_ticks = pd.date_range(start='01-01-2002',end='01-01-2022',freq='Y')
axA.set_xticks(major_ticks)
axA.set_xticklabels([])
axA.grid(which='major', axis='x', alpha=1)
axB.set_xticks(major_ticks)
axB.set_xticklabels([i for i in range(2003,2023)],fontsize=15)
axB.set_xlabel('Date')
axB.grid(which='major', axis='x', alpha=1)
axA.yaxis.label.set_color('C0')
ax_twinA.yaxis.label.set_color('C2')
axB.yaxis.label.set_color('C1')
ax_twinB.yaxis.label.set_color('C3')
plt.tight_layout()
plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\FinalFigures\Plots\sm_gw_v2.pdf')

#LST, NDVI, ET, PPT
fig, (axA, axB) = plt.subplots(2,1,figsize=(13,6))
ax_twinA = axA.twinx()
ax_twinB = axB.twinx()
axA.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),(variables[1].mean(dim=['x','y'])),'--',color='C2',linewidth=2,label='NDVI')
ax_twinA.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),(variables[0].mean(dim=['x','y'])),color='C3',linewidth=1,label='LST')
axB.plot(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),(variables[6].mean(dim=['x','y'])),color='C0',linewidth=2,label='ET')
ax_twinB.bar(pd.date_range('2003-02-01','2021-12-31', freq='1MS'),variables[5].mean(dim=['x','y']),color='black',width=17,label='P')
axA.set_ylabel('NDVI',fontsize=12)
ax_twinA.set_ylabel('Temperature (K)',fontsize=12)
axB.set_ylabel('Evapotranspiration (kg m{})'.format(get_super('-2')),fontsize=12)
ax_twinB.set_ylabel('Precipitation (mm)',fontsize=12)
major_ticks = pd.date_range(start='01-01-2001',end='01-01-2022',freq='Y')
axA.set_xticks(major_ticks)
axA.set_xticklabels([])
axA.grid(which='major', axis='x', alpha=1)
axB.set_ylim(0,110)
axB.set_xticks(major_ticks)
axB.set_xticklabels([i for i in range(2002,2023)],fontsize=15)
axB.set_xlabel('Date')
axB.grid(which='major', axis='x', alpha=1)
axA.yaxis.label.set_color('C2')
ax_twinA.yaxis.label.set_color('C3')
axB.yaxis.label.set_color('C0')
ax_twinB.yaxis.label.set_color('black')
plt.tight_layout()
plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\FinalFigures\Plots\ndvi_lst_et_ppt_v2.pdf')


all_masked_variables = [masked_spei,masked_tci,masked_vci,masked_vhi,masked_variables[0],masked_variables[1],masked_variables[2],
masked_variables[3],masked_variables[4],masked_variables[5],masked_variables[6],masked_variables[7]]

colors = ['black','red','green','orange','#E69F00','#009E73','#0072B2','#0072B2','#0072B2']
variable_names = ['SPEI-6','TCI','VCI','VHI','01_LST','02_NDVI','03_CLM_GW','04_CLM_RZ_SM','05_CLM_Surface_SM']
unit_names = ['SPEI-6','TCI','VCI','VHI','LST (K)','NDVI','GW Level (mm)','RZ SM Level (mm)','Surface SM Level (mm)']
variable_ylims = [[-3,3],[0,100],[0,100],[0,100],[285,310],[0.15,0.70],[400,1000],[120,320],[1.5,6.5]]

for ii,var_name,unit,ylims in zip([3,6],['VHI','GW'],['VHI','GW'],[[0,100],[400,1000]]): #8 variables
    for i, name in zip(range(0,5),class_names): #5 classes
        fig = plt.figure(figsize=(15,4))
        ax = fig.add_subplot()
        ax.plot(all_masked_variables[ii][i].time,(all_masked_variables[ii][i].mean(dim={'x','y'})),color=colors[ii])
        ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
        ax.set_ylim(ylims[0],ylims[1])
        ax.set_xlabel('Date',weight='bold',fontsize=12)
        ax.set_title('{} {} Time Series'.format(var_name,name),weight='bold',fontsize=15)
        plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\time_series\lulc\monthly\{}_{}.png'.format(var_name,name))

#VHI LULC MONTHLY DF
for i, name in zip(range(0,5),class_names): #5 classes
    df = all_masked_variables[3][i].mean(dim={'x','y'}).rename('VHI').to_dataframe()
    df.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\lulc\SPEI_VHI\{}_{}.csv'.format(variable_names[3],name))

#SPEI LULC MONTHLY DF
for i, name in zip(range(0,5),class_names): #5 classes
    df = masked_spei_12[i].mean(dim={'x','y'}).to_dataframe()
    df.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\lulc\SPEI_VHI\{}_{}.csv'.format('SPEI-12',name))
    #fig = plt.figure(figsize=(15,4))
    #ax = fig.add_subplot()
    #ax.plot(all_masked_variables[0][i].time,all_masked_variables[0][i].mean(dim={'x','y'}),color=colors[0])
    #ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
    #ax.set_ylim(ylims)
    #ax.set_xlabel('Year',weight='bold',fontsize=12)
    #ax.set_title('{} {} Time Series'.format(var_name,name),weight='bold',fontsize=15)
    #plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\time_series\lulc\monthly\{}_{}.png'.format(variable_names[0],name))


###################

for ii,var_name,unit,ylims in zip(range(0,8),variable_names,unit_names,variable_ylims): #8 variables
    for i, name in zip(range(0,5),class_names): #5 classes
        #monthly_anom = (monthly_anomaly(masked_variables[ii][i]))
        #anom_df = (anomaly_df(masked_variables[ii][i]))
        #monthly_anom.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\lulc\monthly\{}_{}.csv'.format(var_name,name)
        #anom_df.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\lulc\{}_{}.csv'.format(var_name,name)


        fig = plt.figure(figsize=(15,4))
        ax = fig.add_subplot()
        ax.plot(masked_variables[ii][i].time,(masked_variables[ii][i].mean(dim={'x','y'})),color=colors[ii])
        ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
        ax.set_xlabel('Date',weight='bold',fontsize=12)
        ax.set_title('{} {} Time Series'.format(var_name,name),weight='bold',fontsize=15)
        plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\anomalies_LULC\{}_{}.pdf'.format(var_name,name))




[spei_ds_06[:,:,i].rio.set_spatial_dims(y_dim='y',x_dim='x').rio.to_raster(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\TIFS_FOR_MAPS\SPEI\6\{:03}.tif'.format(i)) for i in range(0,227)]
[spei_ds_12[:,:,i].rio.set_spatial_dims(y_dim='y',x_dim='x').rio.to_raster(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\TIFS_FOR_MAPS\SPEI\12\{:03}.tif'.format(i)) for i in range(0,227)]
[vhi[i].rio.set_spatial_dims(y_dim='y',x_dim='x').rio.to_raster(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\TIFS_FOR_MAPS\VHI\{:03}.tif'.format(i)) for i in range(0,227)]
##################################
#AQUIFER CHARACTER RELATIONSHIPS
##################################

#CALCULATE ANOMALIES with anomaly_df & monthly_anomaly for variables[3:]

all_variables = [spei_ds_06, spei_ds_12,vhi.rename('VHI'),lst_wm,ndvi_wm,clm_gw_wm,clm_sm_rz_wm,clm_sm_surface_wm]

variable_names = ['12_SPEI-6','13_SPEI-12','11_VHI','01_LST','02_NDVI','03_CLM_GW','04_CLM_RZ_SM','05_CLM_Surface_SM']
unit_names = ['SPEI-6','SPEI-12','VHI','LST (K)','NDVI','GW Level','RZ SM Level','Surface SM Level']


############################################################
#Resample Aquifer Maps to resolution of dataset (2 sec)
file_paths = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\Hydrogeology_maps'
files = sorted(glob.glob(file_paths+'\*.tif'))
hydrogeo_maps = [xr.open_rasterio(file) for file in files]
aquifer_chars = ['depth_GW','productivity','storage','recharge_rate']

resampled_hydrogeo = [[hydrogeo.rio.write_crs('epsg:4328').rio.reproject_match(variable.rio.write_crs('epsg:4328'),resampling=Resampling.nearest) 
                        for hydrogeo in hydrogeo_maps]
                        for variable in all_variables]

path_mask = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\Hydrogeology_maps\masks'
[[resampled_hydrogeo[ii][i].rio.set_crs("epsg:4326").rio.set_spatial_dims('x','y',inplace=True).rio.to_raster(path_mask+'/{}/{}.tif'.format(aquifer_char,variable_name))
for ii,variable_name in zip(range(0,len(all_variables)),variable_names)]
for i,aquifer_char in zip(range(0,len(hydrogeo_maps)),aquifer_chars)] 
############################################################



path_mask = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\Hydrogeology_maps\masks'
aquifer_chars = ['depth_GW','recharge_rate']

files_mask = [sorted(glob.glob(path_mask+'/{}/*.tif'.format(char))) for char in aquifer_chars ]
aquifer_masks = [[xr.open_rasterio(file[i]) for i in range(0,len(all_variables))] for file in files_mask]

csv_files = sorted(glob.glob(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\Hydrogeology_maps\*.csv'))
csv_aqfr = [pd.read_csv(csv) for csv in csv_files]
aqfr_class_names = [csv_aqfr[i].iloc[:,0] for i in range(0,len(csv_aqfr))]
aqfr_class_values = [csv_aqfr[i].iloc[:,1] for i in range(0,len(csv_aqfr))]

depth_GW_masks = aquifer_masks[0]
recharge_masks = aquifer_masks[-1]

depth_GW_class_names = aqfr_class_names[0]
recharge_class_names = aqfr_class_names[-1]

depth_GW_class_values = aqfr_class_values[0]
recharge_class_values = aqfr_class_values[-1]

#NON-index variables
variables = all_variables[3:]
#Depth_to_GW Masks
masked_variables_GWdepth_variables = [[variable.where(depth[0]==value) for value in depth_GW_class_values] for variable,depth in zip(variables,depth_GW_masks[0:5])]
#Recharge Masks
masked_variables_recharge_variables = [[variable.where(depth[0]==value) for value in recharge_class_values] for variable,depth in zip(variables,recharge_masks[0:5])]

################################################
#GW_Depth/Recharge

#VARIABLES - ANOMALIES?

colors = ['#E69F00','#009E73','#0072B2','#0072B2','#0072B2']
variable_names = ['01_LST','02_NDVI','03_CLM_GW','04_CLM_RZ_SM','05_CLM_Surface_SM']
unit_names = ['LST (K)','NDVI','GW Level (mm)','RZ SM Level (mm)','Surface SM Level (mm)']
variable_ylims =  [[285,310],[0.15,0.70],[400,1000],[120,320],[1.5,6.5]]

for ii,var_name,unit,ylims in zip(range(0,5),variable_names,unit_names,variable_ylims):
    for i, name in zip(range(0,6),depth_GW_class_names):
        fig = plt.figure(figsize=(15,4))
        ax = fig.add_subplot()
        ax.plot(masked_variables_GWdepth_variables[ii][i].time,(masked_variables_GWdepth_variables[ii][i].mean(dim={'x','y'})),color=colors[ii])
        ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
        ax.set_ylim(ylims)
        ax.set_xlabel('Date',weight='bold',fontsize=12)
        ax.set_title('{} {} Time Series'.format(var_name,name),weight='bold',fontsize=15)
        plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\time_series\aquifer\monthly\variables\GW_DEPTH\{}_{}.png'.format(var_name,name))

        monthly_anom = (monthly_anomaly(masked_variables_GWdepth_variables[ii][i]))
        anom_df = (anomaly_df(masked_variables_GWdepth_variables[ii][i]))
        monthly_anom.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\aquifer\DEPTH_GW\monthly\{}_{}.csv'.format(var_name,name))
        anom_df.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\aquifer\DEPTH_GW\{}_{}.csv'.format(var_name,name))

for ii,var_name,unit,ylims in zip(range(0,5),variable_names,unit_names,variable_ylims):
    for i, name in zip(range(0,11),recharge_class_names):
        fig = plt.figure(figsize=(15,4))
        ax = fig.add_subplot()
        ax.plot(masked_variables_recharge_variables[ii][i].time,(masked_variables_recharge_variables[ii][i].mean(dim={'x','y'})),color=colors[ii])
        ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
        ax.set_ylim(ylims)
        ax.set_xlabel('Date',weight='bold',fontsize=12)
        ax.set_title('{} {} Time Series'.format(var_name,name),weight='bold',fontsize=15)
        plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\time_series\aquifer\monthly\variables\RECHARGE\{}_{}.png'.format(var_name,name))

        monthly_anom = (monthly_anomaly(masked_variables_recharge_variables[ii][i]))
        anom_df = (anomaly_df(masked_variables_recharge_variables[ii][i]))
        monthly_anom.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\aquifer\RECHARGE\monthly\{}_{}.csv'.format(var_name,name))
        anom_df.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\aquifer\RECHARGE\{}_{}.csv'.format(var_name,name))

import gc

del masked_variables_GWdepth_variables,masked_variables_recharge_variables
gc.collect()



#MONTHLY INDICES --- VHI, SPEI-6, SPEI-12
#INDICES
colors = ['orange','black','black']
variable_names = ['VHI','SPEI-6','SPEI-12']
unit_names = ['VHI','SPEI-6','SPEI-12']
variable_ylims = [[0,100],[-3,3],[-3,3]]


###################
#VHI LULC MONTHLY DF
masked_vhi_GWdepth = [vhi.rename('VHI').where(depth_GW_masks[-3]==value) for value in depth_GW_class_values]
for i, name in zip(range(0,6),depth_GW_class_names): #6 classes
    df = masked_vhi_GWdepth[i].mean(dim={'x','y'}).to_dataframe()
    df.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\lulc\SPEI_VHI\{}_{}.csv'.format(variable_names[0],name))
del masked_vhi_GWdepth
gc.collect()


masked_vhi_recharge = [vhi.rename('VHI').where(recharge_masks[-3]==value) for value in recharge_class_values]
for i, name in zip(range(0,11),recharge_class_names): #11 classes
    df = masked_vhi_recharge[i].mean(dim={'x','y'}).to_dataframe()
    df.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\lulc\SPEI_VHI\{}_{}.csv'.format(variable_names[0],name))
del masked_vhi_recharge
gc.collect()

####################
#SPEI LULC MONTHLY DF
#6
masked_spei_06_GWdepth = [[np.where(depth_GW_masks[-2][0]==value,spei_ds_06[:,:,i],np.nan) for value in depth_GW_class_values] for i in range(0,227)]
masked_spei_06_GWdepths = [pd.concat(masked_spei_06_GWdepth[i]) for i in range(0,6)]
for i, name in zip(range(0,6),class_names): #6 classes
    df = masked_spei_06_GWdepths[i].mean(dim={'x','y'}).to_dataframe()
    df.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\lulc\SPEI_VHI\{}_{}.csv'.format(variable_names[1],name))
del masked_spei_06_GWdepth,masked_spei_06_GWdepths
gc.collect()


masked_spei_06_recharge = [[np.where(recharge_masks[-2][0]==value,spei_ds_06[:,:,i],np.nan) for value in recharge_class_values] for i in range(0,227)]
masked_spei_06_recharges = [pd.concat(masked_spei_06_recharge[i]) for i in range(0,11)]
for i, name in zip(range(0,11),class_names): #11 classes
    df = masked_spei_06_recharges[i].mean(dim={'x','y'}).to_dataframe()
    df.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\lulc\SPEI_VHI\{}_{}.csv'.format(variable_names[1],name))
del masked_spei_06_recharge,masked_spei_06_recharges
gc.collect()

#12
masked_spei_12_GWdepth = [[np.where(depth_GW_masks[-1][0]==value,spei_ds_12[:,:,i],np.nan) for value in depth_GW_class_values] for i in range(0,227)]
masked_spei_12_GWdepths = [pd.concat(masked_spei_12_GWdepth[i]) for i in range(0,6)]
for i, name in zip(range(0,6),depth_GW_class_names): #6 classes
    df = masked_spei_12_GWdepths[i].mean(dim={'x','y'}).to_dataframe()
    df.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\lulc\SPEI_VHI\{}_{}.csv'.format(variable_names[2],name))
del masked_spei_12_GWdepth, masked_spei_12_GWdepths
gc.collect()

masked_spei_12_recharge = [[np.where(recharge_masks[-1][0]==value,spei_ds_12[:,:,i],np.nan) for value in recharge_class_values] for i in range(0,227)]
masked_spei_12_recharges = [pd.concat(masked_spei_12_recharge[i]) for i in range(0,11)]
for i, name in zip(range(0,11),recharge_class_names): #11 classes
    df = masked_spei_12_recharges[i].mean(dim={'x','y'}).to_dataframe()
    df.to_csv(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\lulc\SPEI_VHI\{}_{}.csv'.format(variable_names[2],name))
del masked_spei_12_recharge, masked_spei_12_recharges
gc.collect()



### CALL CSV DATA AND MAKE PLOTS FOR FIGURES (11-26-22 03:21)
for ii,var_name,unit,ylims in zip(range(0,3),variable_names,unit_names,variable_ylims): #3 variables
    for i, name in zip(range(0,6),depth_GW_class_names): #5 classes
        fig = plt.figure(figsize=(15,4))
        ax = fig.add_subplot()
        ax.plot(masked_variables_GWdepth_indices[ii][i].time,(masked_variables_GWdepth_indices[ii][i].mean(dim={'x','y'})),color=colors[ii])
        ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
        ax.set_ylim(ylims[0],ylims[1])
        ax.set_xlabel('Date',weight='bold',fontsize=12)
        ax.set_title('{} {} Time Series'.format(var_name,name),weight='bold',fontsize=15)
        plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\time_series\aquifer\monthly\{}_{}.png'.format(var_name,name))


############################################################
#SEASONAL LULC & WATER MASKED
############################################################
#MASKED WATER: SEASONAL

variables_monthly = [spei_ds.rename({'lat':'y','lon':'x'}),tci,vci,vhi,lst_wm,ndvi_wm,clm_gw_wm,clm_sm_rz_wm,clm_sm_surface_wm,p_gpm_wm,et_modis_wm,pet_modis_wm]

variables_mean_monthly = variables_monthly[0:9] #include smap when time series is ready
variables_sum_monthly = variables_monthly[9:12]

#Wet: Oct - April 
#Dry: May - Sept

#Start date: 2003-02
#End date: 2021-12

season_array = np.where( (np.array(variables_mean_monthly[0]['time.month']) == 2) |
(np.array(variables_mean_monthly[0]['time.month']) == 3) |
(np.array(variables_mean_monthly[0]['time.month']) == 4) |
(np.array(variables_mean_monthly[0]['time.month']) == 10) |
(np.array(variables_mean_monthly[0]['time.month']) == 11) |
(np.array(variables_mean_monthly[0]['time.month']) == 12) | 
(np.array(variables_mean_monthly[0]['time.month']) == 1), 'WET','DRY')

Y_M_season_idxs = [pd.MultiIndex.from_arrays([np.array(variable['time.year']),season_array]) for variable in variables_mean_monthly]
for i in range(0,len(variables_mean_monthly)):
    variables_mean_monthly[i].coords['year_month_S'] = ('time', Y_M_season_idxs[i])
variables_seasonally_mean = [variable.groupby('year_month_S').mean() for variable in variables_mean_monthly]

Y_M_season_idxs = [pd.MultiIndex.from_arrays([np.array(variable['time.year']),season_array]) for variable in variables_sum_monthly]
for i in range(0,len(variables_sum_monthly)):
    variables_sum_monthly[i].coords['year_month_S'] = ('time', Y_M_season_idxs[i])
variables_seasonally_sum = [variable.groupby('year_month_S').sum() for variable in variables_sum_monthly]


variables_mean_seasonal = [variable_seasoned.mean(dim={'x','y'}) for variable_seasoned in variables_seasonally_mean]
variables_sum_seasonal = [variable_seasoned.mean(dim={'x','y'}) for variable_seasoned in variables_seasonally_sum]


#MEAN VARIABLES - 2 min
colors = ['black','red','green','orange','#E69F00','#009E73','#0072B2','#0072B2','#0072B2']
variable_names = ['SPEI-6','TCI','VCI','VHI','01_LST','02_NDVI','03_CLM_GW','04_CLM_RZ_SM','05_CLM_Surface_SM']
unit_names = ['SPEI-6','TCI','VCI','VHI','LST (K)','NDVI','GW Level (mm)','RZ SM Level (mm)','Surface SM Level (mm)']
variable_ylims = [[-3,3],[0,100],[0,100],[0,100],[285,310],[0.15,0.70],[400,1000],[120,320],[1.5,6.5]]

for i, name,unit,ylims in zip(range(0,8),variable_names,unit_names,variable_ylims):
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot()
    ax.plot(variables_mean_seasonal[i][0::2].year_month_S_level_0,variables_mean_seasonal[i][0::2],color=colors[i]) #WET SEASON
    ax.plot(variables_mean_seasonal[i][1::2].year_month_S_level_0,variables_mean_seasonal[i][1::2],color=colors[i], alpha=0.5) #DRY SEASON
    ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
    ax.set_ylim(ylims)
    ax.set_xlabel('Year',weight='bold',fontsize=12)
    ax.set_xlim(2003,2021)
    ax.set_title('{} Time Series'.format(name),weight='bold',fontsize=15)
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\time_series\seasonal\{}'.format(name))

#SUM VARIABLES - 30s
colors = ['#56B4E9','#6E8FBB','#6E8FBB']
variable_names = ['06_P_GPM','07_ET','08_PET']
unit_names = ['Precipitation (mm)','ET (kgm-2)','ET (kgm-2)']
variable_ylims = [[0,400],[0,300],[0,1000]]

for i, name,unit,ylims in zip(range(0,3),variable_names,unit_names,variable_ylims):
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot()
    ax.plot(variables_sum_seasonal[i][0::2].year_month_S_level_0,variables_sum_seasonal[i][0::2],color=colors[i]) #WET SEASON
    ax.plot(variables_sum_seasonal[i][1::2].year_month_S_level_0,variables_sum_seasonal[i][1::2],color=colors[i], alpha=0.5) #DRY SEASON
    ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
    ax.set_ylim(ylims)
    ax.set_xlabel('Year',weight='bold',fontsize=12)
    ax.set_xlim(2003,2021)
    ax.set_title('{} Time Series'.format(name),weight='bold',fontsize=15)
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\time_series\seasonal\{}'.format(name))


############################################################
#LULC MASKED: SEASONAL
all_masked_variables = [masked_spei,masked_tci,masked_vci,masked_vhi,masked_variables[0],masked_variables[1],masked_variables[2],
masked_variables[3],masked_variables[4],masked_variables[5],masked_variables[6],masked_variables[7]]

variables_mean_monthly_masked = all_masked_variables[0:9] #include smap when time series is ready
variables_sum_monthly_masked = all_masked_variables[9:12]


season_arrays = [np.where( (np.array(variables_mean_monthly_masked[i][0]['time.month']) == 2) |
(np.array(variables_mean_monthly_masked[i][0]['time.month']) == 3) |
(np.array(variables_mean_monthly_masked[i][0]['time.month']) == 4) |
(np.array(variables_mean_monthly_masked[i][0]['time.month']) == 10) |
(np.array(variables_mean_monthly_masked[i][0]['time.month']) == 11) |
(np.array(variables_mean_monthly_masked[i][0]['time.month']) == 12) | 
(np.array(variables_mean_monthly_masked[i][0]['time.month']) == 1), 'WET','DRY') for i in range(0,len(variables_mean_monthly_masked[0]))]

Y_M_season_idxs = [[pd.MultiIndex.from_arrays([np.array(variable[i]['time.year']),season_array]) for variable in variables_mean_monthly_masked] 
                    for i,season_array in zip(range(0,len(variables_mean_monthly_masked[0])),season_arrays)]
for ii in range(0,len(variables_mean_monthly_masked[0])):
    for i in range(0,len(variables_mean_monthly_masked)):
        variables_mean_monthly_masked[i][ii].coords['year_month_S'] = ('time', Y_M_season_idxs[ii][i])
variables_seasonally_mean_lulc = [[variable[i].groupby('year_month_S').mean() for variable in variables_mean_monthly_masked] for i in range(0,len(variables_mean_monthly_masked[0]))]

Y_M_season_idxs = [[pd.MultiIndex.from_arrays([np.array(variable[i]['time.year']),season_array]) for variable in variables_sum_monthly_masked]
                     for i,season_array in zip(range(0,len(variables_sum_monthly_masked[0])),season_arrays)]
for ii in range(0,len(variables_sum_monthly_masked[0])):
    for i in range(0,len(variables_sum_monthly_masked)):
        variables_sum_monthly_masked[i][ii].coords['year_month_S'] = ('time', Y_M_season_idxs[ii][i])
variables_seasonally_sum_lulc = [[variable[i].groupby('year_month_S').sum() for variable in variables_sum_monthly_masked] for i in range(0,len(variables_sum_monthly_masked[0]))]


variables_mean_seasonal_masked = [[variable_seasoned[i].mean(dim={'x','y'}) for variable_seasoned in variables_seasonally_mean_lulc] for i in range(0,len(variables_seasonally_mean_lulc[0]))]
variables_sum_seasonal_masked = [[variable_seasoned[i].mean(dim={'x','y'}) for variable_seasoned in variables_seasonally_sum_lulc] for i in range(0,len(variables_seasonally_sum_lulc[0]))]


#MEAN VARIABLES - 12 minutes
colors = ['black','red','green','orange','#E69F00','#009E73','#0072B2','#0072B2','#0072B2']
variable_names = ['SPEI-6','TCI','VCI','VHI','01_LST','02_NDVI','03_CLM_GW','04_CLM_RZ_SM','05_CLM_Surface_SM']
unit_names = ['SPEI-6','TCI','VCI','VHI','LST (K)','NDVI','GW Level (mm)','RZ SM Level (mm)','Surface SM Level (mm)']
variable_ylims = [[-3,3],[0,100],[0,100],[0,100],[285,310],[0.15,0.70],[400,1000],[120,320],[1.5,6.5]]
class_names = ['trees','rangeland','crops','urban','bare']

for ii,var_name,unit,ylims in zip(range(1,9),variable_names[1:],unit_names[1:],variable_ylims[1:]): #8 variables
    for i, name in zip(range(0,5),class_names): #5 classes
        fig = plt.figure(figsize=(15,4))
        ax = fig.add_subplot()
        ax.plot(variables_mean_seasonal_masked[ii][i][0::2].year_month_S_level_0,variables_mean_seasonal_masked[ii][i][0::2],color=colors[ii])
        ax.plot(variables_mean_seasonal_masked[ii][i][1::2].year_month_S_level_0,variables_mean_seasonal_masked[ii][i][1::2],color=colors[ii], alpha=0.5)
        ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
        ax.set_ylim(ylims)
        ax.set_xlabel('Year',weight='bold',fontsize=12)
        ax.set_xlim(2003,2021)
        ax.set_title('{} {} Time Series'.format(var_name,name),weight='bold',fontsize=15)
        plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\time_series\lulc\seasonal\{}_{}.png'.format(var_name,name))

#JUST FOR SPEI-06
for i, name in zip(range(0,5),class_names): #5 classes
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot()
    ax.plot(variables_mean_seasonal_masked[0][i][:,:][0::2].year_month_S_level_0,variables_mean_seasonal_masked[0][i][:,:][0::2],color=colors[ii])
    ax.plot(variables_mean_seasonal_masked[0][i][:,:][1::2].year_month_S_level_0,variables_mean_seasonal_masked[0][i][:,:][1::2],color=colors[ii], alpha=0.5)
    ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
    ax.set_ylim(ylims)
    ax.set_xlabel('Year',weight='bold',fontsize=12)
    ax.set_xlim(2003,2021)
    ax.set_title('{} {} Time Series'.format(var_name,name),weight='bold',fontsize=15)
    plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\time_series\lulc\seasonal\{}_{}.png'.format(variable_names[0],name))
    
#SUM VARIABLES - 3 minutes
colors = ['#56B4E9','#6E8FBB','#6E8FBB']
variable_names = ['06_P_GPM','07_ET','08_PET']
unit_names = ['Precipitation (mm)','ET (kgm-2)','ET (kgm-2)']
variable_ylims = [[0,400],[0,300],[0,1000]]
class_names = ['trees','rangeland','crops','urban','bare']

for ii,var_name,unit,ylims in zip(range(0,3),variable_names,unit_names,variable_ylims): #3 variables
    for i, name in zip(range(0,5),class_names): #5 classes
        fig = plt.figure(figsize=(15,4))
        ax = fig.add_subplot()
        ax.plot(variables_sum_seasonal_masked[ii][i][0::2].year_month_S_level_0,variables_sum_seasonal_masked[ii][i][0::2],color=colors[ii])
        ax.plot(variables_sum_seasonal_masked[ii][i][1::2].year_month_S_level_0,variables_sum_seasonal_masked[ii][i][1::2],color=colors[ii], alpha=0.5)
        ax.set_ylabel('{}'.format(unit),weight='bold',fontsize=12)
        ax.set_ylim(ylims)
        ax.set_xlabel('Year',weight='bold',fontsize=12)
        ax.set_xlim(2003,2021)
        ax.set_title('{} {} Time Series'.format(var_name,name),weight='bold',fontsize=15)
        plt.savefig(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\DraftFigures\time_series\lulc\seasonal\{}_{}.png'.format(var_name,name))


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
ax1.plot(clm_gw.time,clm_gw.mean(dim=['x','y']),color='C0')
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



############################################################
############################################################
############################################################
#Resample datasets for climate indices -- resolution of ET/PET (500m)

lst_wm = water_masked_variables_17[0][7:].rename({'x': 'lon','y': 'lat'})
p_gpm_wm = water_masked_variables_17[5][13:].rename({'x': 'lon','y': 'lat'})
et_modis_wm = water_masked_variables_17[6][13:].rename({'x': 'lon','y': 'lat'})
pet_modis_wm = water_masked_variables_17[7][13:].rename({'x': 'lon','y': 'lat'})



lst_wm_rs = lst_wm.rio.write_crs('epsg:4328').rio.reproject_match(et_modis_wm.rio.write_crs('epsg:4328'),resampling=Resampling.nearest).rename({'x': 'lon','y': 'lat'})
p_gpm_wm_rs = p_gpm_wm.rio.write_crs('epsg:4328').rio.reproject_match(et_modis_wm.rio.write_crs('epsg:4328'),resampling=Resampling.nearest).rename({'x': 'lon','y': 'lat'})

ds_lat = et_modis_wm.lat
ds_lon = et_modis_wm.lon
ds_time = et_modis_wm.time

path = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\climate_indices'
new_array = xr.DataArray(p_gpm_wm_rs.transpose('lat','lon','time').assign_attrs({'units':'millimeter'}), dims=("lat", "lon","time"), coords={"lat": ds_lat, "lon": ds_lon,"time":ds_time}, name="P_mm")
new_array.rio.set_crs("epsg:4326").to_netcdf(path+'\GPM_PPT_2002_2021.nc')

new_array = xr.DataArray(lst_wm_rs.transpose('lat','lon','time').assign_attrs({'units':'kelvin'}), dims=("lat", "lon","time"), coords={"lat": ds_lat, "lon": ds_lon,"time":ds_time}, name="LST_K")
new_array.rio.set_crs("epsg:4326").to_netcdf(path+'\MODMYD_LST_2002_2021.nc')

new_array = xr.DataArray(et_modis_wm.transpose('lat','lon','time').assign_attrs({'units':'millimeter'}), dims=("lat", "lon","time"), coords={"lat": ds_lat, "lon": ds_lon,"time":ds_time}, name="ET_kg_m2")
new_array.rio.set_crs("epsg:4326").to_netcdf(path+'\MODMYD_ET_2002_2021.nc')

new_array = xr.DataArray(pet_modis_wm.transpose('lat','lon','time').assign_attrs({'units':'millimeter'}), dims=("lat", "lon","time"), coords={"lat": ds_lat, "lon": ds_lon,"time":ds_time}, name="ET_kg_m2")
new_array.rio.set_crs("epsg:4326").to_netcdf(path+'\MODMYD_PET_2002_2021.nc')

path = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\climate_indices'
files = sorted(glob.glob(path+'\*.nc'))
ds = xr.open_mfdataset(files[1],parallel=True,chunks={"y": 100,"x":100})

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

dates= pd.date_range('2015-04-01','2021-12-31' , freq='1MS')

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
