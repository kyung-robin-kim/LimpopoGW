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

######################################################################################################################################################
#GROUNDWATER INSITU
SA_gw_data = pd.read_csv(glob.glob(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\sophia\Limpopo In-Situ Data\Boreholes\*.csv')[-1])
insitu_points_gw = [Point(xy) for xy in zip(SA_gw_data['Longitude'], SA_gw_data['Latitude'])]
insitu_gpd_gw = gpd.GeoDataFrame(SA_gw_data, geometry=insitu_points_gw).set_crs('EPSG:4326')

limpopo_watershed = gpd.read_file(r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp')
limpopo_gw_insitu = insitu_gpd_gw[insitu_gpd_gw.within(limpopo_watershed.loc[0,'geometry'])]

start = 40 #October 1999:0 .... Feb 2003: 40  January 2011: 135
years = 19
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

dt_index = [pd.to_datetime(year_month[1:9],format='%Y_%b') for year_month in gw_level.index]
gw_level.index = dt_index
gw_level = gw_level.interpolate('linear')

gw_anomaly = (gw_level - gw_level.mean(axis=0))/ gw_level.std(axis=0,ddof=1)
gw_anomaly.mean(axis=1).plot()
(-2* gw_anomaly.std(axis=1,ddof=1) + gw_anomaly.mean(axis=1)).plot()
(2* gw_anomaly.std(axis=1,ddof=1) + gw_anomaly.mean(axis=1)).plot()

dataset = gw_level
monthly_data = [dataset.loc[dataset.index.month == i] for i in range(1,13)]
monthly_mean = [dataset.loc[dataset.index.month == i].mean(axis=0) for i in range(1,13)]
monthly_std = [dataset.loc[dataset.index.month == i].std(axis=0,ddof=1) for i in range(1,13)]
dataset_month_anomalies = [ (monthly_data[i] - monthly_mean[i])/monthly_std[i] for i in range(0,12)]
monthly_anomalies_gw = pd.concat(dataset_month_anomalies).sort_index()

gw_anomaly.mean(axis=1).plot()
monthly_anomalies_gw.mean(axis=1).plot(legend=False)

######################################################################################################################################################
#DISCHARGE INSITU FROM GRDC
q_data = glob.glob(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\GRDC\2023-05-03_00-21\*nc')[0]
insitu_q = xr.open_mfdataset(q_data)

runoff = insitu_q.runoff_mean[(insitu_q.time>pd.to_datetime('2003-02')) & (insitu_q.time<pd.to_datetime('2022-01'))].transpose('id','time')
runoff_filtered = runoff[runoff.count(dim='time')>6200].transpose('time','id')

station_id = insitu_q.station_name[runoff.count(dim='time')>6200]
geo_x = insitu_q.geo_x[runoff.count(dim='time')>6200]
geo_y = insitu_q.geo_y[runoff.count(dim='time')>6200]
geo_z = insitu_q.geo_z[runoff.count(dim='time')>6200]

insitu_points_q = [Point(xy) for xy in zip(geo_x, geo_y)]
insitu_gpd_q = gpd.GeoDataFrame(station_id, geometry=insitu_points_q).set_crs('EPSG:4326')

dataframe_q = pd.concat([station_id.to_dataframe(),geo_x.to_dataframe(),geo_y.to_dataframe(),geo_z.to_dataframe()],axis=1)


runoff_ln = np.log(runoff_filtered)
runoff_ln = runoff_ln.where(runoff_ln>-100, np.nan)
runoff_sqrt = np.sqrt(runoff)
runoff_sqrt = runoff_sqrt.where(runoff_sqrt>-100, np.nan)

transformed = runoff_ln
q_anomaly = (transformed - transformed.mean(dim='time'))/ transformed.std(dim='time',ddof=1)


runoff_ln = pd.DataFrame(transformed).set_index(pd.to_datetime(transformed.time))
dataset = runoff_ln.resample('1M').mean()
monthly_data = [dataset.loc[dataset.index.month == i] for i in range(1,13)]
monthly_mean = [dataset.loc[dataset.index.month == i].mean(axis=0) for i in range(1,13)]
monthly_std = [dataset.loc[dataset.index.month == i].std(axis=0,ddof=1) for i in range(1,13)]
dataset_month_anomalies = [ (monthly_data[i] - monthly_mean[i])/monthly_std[i] for i in range(0,12)]
monthly_anomalies_q = pd.concat(dataset_month_anomalies).sort_index()

'''
for i in range(0,len(runoff_filtered.id)):
    plt.figure()
    q_anomaly[:,i].plot()
'''
#q_anomaly.rolling(time=30, min_periods=1).mean().mean(dim='id').plot()
#gw_anomaly.mean(axis=1).plot()
#(-2* gw_anomaly.std(axis=1,ddof=1) + gw_anomaly.mean(axis=1)).plot()
#(2* gw_anomaly.std(axis=1,ddof=1) + gw_anomaly.mean(axis=1)).plot()
monthly_anomalies_q.mean(axis=1).plot()
monthly_anomalies_gw.mean(axis=1).plot()




(mk.seasonal_test(q_anomaly.rolling(time=30, min_periods=1).mean().mean(dim='id'),period=365))
(mk.seasonal_test(gw_anomaly.mean(axis=1),period=12))


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
#GW
clm_q = xr.open_mfdataset(files[2],parallel=True)['kgm-2s-1']
clm_bq = xr.open_mfdataset(files[1],parallel=True)['kgm-2s-1']

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


#STATISTICAL ANALYSES - Feb 2003 through Dec 2021
def monthly_anom(dataset):
    #dataset = et_modis.ET_kg_m2
    month_idxs=dataset.groupby('time.month').groups
    dataset_month_anomalies = [(dataset.isel(time=month_idxs[i]) - dataset.isel(time=month_idxs[i]).mean(dim='time'))/(dataset.isel(time=month_idxs[i]).std(ddof=1,dim='time')) for i in range(1,13)]
    var_name = dataset_month_anomalies[0].var().name
    monthly_anomalies_ds = xr.merge(dataset_month_anomalies)[var_name]

    return monthly_anomalies_ds

#STATISTICAL ANALYSES UPDATED - Feb 2002 through Dec 2022
#removed standard deviation for standardization to keep same units
def monthly_anom(dataset):
    #dataset = et_modis.ET_kg_m2
    month_idxs=dataset.groupby('time.month').groups
    dataset_month_anomalies = [(dataset.isel(time=month_idxs[i]) - dataset.isel(time=month_idxs[i]).mean(dim='time')) for i in range(1,13)]
    var_name = dataset_month_anomalies[0].var().name
    monthly_anomalies_ds = xr.merge(dataset_month_anomalies)[var_name]

    return monthly_anomalies_ds

#Create Monthly Anomalies Datasets
'''
netcdf_anom_path = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\netcdfs'

#CLSM Runoff
clm_q_log = np.log(clm_bq)
clm_q_log = clm_q_log[(clm_q_log.time>pd.to_datetime('2002-02')) & (clm_q_log.time<pd.to_datetime('2023-01'))] 
monthly_anom(clm_q_log).to_netcdf(netcdf_anom_path+r'\Runoff_base.nc')

#LST, NDVI, GW, SM
variables = [lst.LST_K, ndvi.NDVI, clm_gw, clm_sm_rz, clm_sm_surface, sm_ts]
[print(var.dims) for var in variables]
variable_names = ['LST','NDVI','GW','RZ','Surface SM', 'SMAP']
variables = [var[(var.time>pd.to_datetime('2002-02')) & (var.time<pd.to_datetime('2023-01'))] for var in variables]
[monthly_anom(var).to_netcdf(netcdf_anom_path+r'\{}.nc'.format(name)) for var,name in zip(variables,variable_names)]

#ET - resampled for memory (500m to 1km)
variables = [et_modis.ET_kg_m2, pet_modis.PET_kg_m2]
[print(var.dims) for var in variables]
variable_names = ['ET_1km','ET_P_1km']
variables = [var[(var.time>pd.to_datetime('2002-02')) & (var.time<pd.to_datetime('2023-01'))] for var in variables]
[monthly_anom(var.rio.write_crs('epsg:4326').rio.reproject_match(lst.rio.write_crs('epsg:4326'), resampling=rio.enums.Resampling.bilinear)).to_netcdf(netcdf_anom_path+r'\{}.nc'.format(name))
               for var,name in zip(variables,variable_names)]

#PPT
precip_variables = [np.log(p_gpm.P_mm).where(np.log(p_gpm.P_mm)>-100,np.nan), np.log(p_chirps.P_mm).where(np.log(p_chirps.P_mm)>-100,np.nan)]
[print(var.dims) for var in precip_variables]
precip_variable_names = ['PPT_GPM','PPT_CHIRPS']
precip_variables = [var[(var.time>pd.to_datetime('2002-02')) & (var.time<pd.to_datetime('2023-11'))] for var in precip_variables]
[monthly_anom(var).to_netcdf(netcdf_anom_path+r'\{}.nc'.format(name)) for var,name in zip(precip_variables,precip_variable_names)]

#VHI
files_vhi = sorted(glob.glob(netcdf_anom_path+'\VHI\*.nc'))
vhi_dataset = xr.open_mfdataset(files_vhi[0],parallel=True)
monthly_anom(vhi_dataset.VHI.rio.write_crs('epsg:4326').rio.reproject_match(ndvi.rio.write_crs('epsg:4326'), resampling=rio.enums.Resampling.bilinear)).to_netcdf(netcdf_anom_path+r'\VHI\VHI_anom.nc')


'''
netcdf_anom_path = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\netcdfs'

files_vhi = sorted(glob.glob(netcdf_anom_path+'\VHI\*.nc'))
files_spei = sorted(glob.glob(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\climate_indices\SPEI\*.nc'))
files_indicators = [files_vhi[1],files_spei[1],files_spei[3],files_spei[0]]
indices_datasets = [xr.open_mfdataset(file,parallel=True) for file in files_indicators]
inidices_keys = [list(ds.data_vars)[0] for ds in indices_datasets]
indices_datasets = [var['{}'.format(key)] for var,key in zip(indices_datasets[0:1],inidices_keys[0:1])] + [var['{}'.format(key)].rename({'lat':'y','lon':'x'}).transpose('time','y','x') for var,key in zip(indices_datasets[1:],inidices_keys[1:])]
indices_names = ['VHI','SPEI-3','SPEI-12','SPEI-1']

files = sorted(glob.glob(netcdf_anom_path+'\*.nc'))
var_datasets = [xr.open_mfdataset(file,parallel=True) for file in files]
var_keys = [list(ds.data_vars)[0] for ds in var_datasets]
var_datasets = [var['{}'.format(key)] for var,key in zip(var_datasets,var_keys)]
var_names = ['ET', 'ET_P','GW','LST','NDVI','PPT_CHIRPS','PPT_GPM','RZ','SMAP','SM_Surf']

spatial_mean_anomalies_vars = [var.mean(dim={'x','y'}) for var in var_datasets]
spatial_mean_anomalies_indic = [var.mean(dim={'x','y'}) for var in indices_datasets]


#Vectorizing doesn't work .......
#xr.apply_ufunc(mk.seasonal_test, var_datasets[2], input_core_dims=[['time','y','x']], output_core_dims=[['x','y']], vectorize=True,dask='parallelized')
#[(mk.seasonal_test(var,period=12),name) for var,name in zip(var_datasets,var_names)]


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



#Trend? (boolean)
ds_mk = 
ds_mk[1].where(ds_mk[1]!='False',0).where(ds_mk[1]!='True',1).astype(int).plot()

#p-val
ds_mk[2].astype(float).plot()

#z-statistic
ds_mk[3].astype(float).plot()

#MK-statistic
ds_mk[5].astype(float).plot()

#MK-statistic
ds_mk[5].astype(float).plot()

#NonParametric
#Mann-Kendall

def mk_df(data,varname):
    stats = mk.seasonal_test(data,period=12)
    df = pd.DataFrame([stat for stat in stats]).rename(columns={0:'{}'.format(varname)})
    return df

def sprank_df(data,varname):
    stats = spearmanr(range(0,len(data)),data)
    df = pd.DataFrame([stat for stat in stats]).rename(columns={0:'{}'.format(varname)})
    return df

mk_vars_df = pd.concat([mk_df(var,name) for var,name in zip(spatial_mean_anomalies_vars,var_names)],axis=1)
mk_indic_df = pd.concat([mk_df(var,name) for var,name in zip(spatial_mean_anomalies_indic,indices_names)],axis=1)
mk_q_insitu_df = mk_df(monthly_anomalies_q.mean(axis=1),'insitu Q')
mk_gw_insitu_df = mk_df(monthly_anomalies_gw.mean(axis=1),'insitu GW')

#Spearman's rank correlation
sprank_vars_df = pd.concat([sprank_df(data,var) for data,var in zip(spatial_mean_anomalies_vars,var_names)],axis=1)
sprank_indic_df = pd.concat([sprank_df(data,var) for data,var in zip(spatial_mean_anomalies_indic,indices_names)],axis=1)

#spearmanr(range(0,len(spatial_mean_anomalies_vars[8])), spatial_mean_anomalies_vars[8].resample('1M').mean()) #SMAP
spearmanr(range(3,len(spatial_mean_anomalies_indic[1])), spatial_mean_anomalies_indic[1][3:]) #SPEI-3
spearmanr(range(12,len(spatial_mean_anomalies_indic[2])), spatial_mean_anomalies_indic[2][12:]) #SPEI-12
spearmanr(range(1,len(spatial_mean_anomalies_indic[3])), spatial_mean_anomalies_indic[3][1:]) #SPEI-1
spearmanr(range(0,len(monthly_anomalies_q)), (monthly_anomalies_q.mean(axis=1))) #insitu Q
spearmanr(range(0,len(monthly_anomalies_gw)), (monthly_anomalies_gw.mean(axis=1))) #insitu GW


#SPEI-12 and GW
spatial_mean_anomalies_vars[2].plot()
spatial_mean_anomalies_indic[2].plot()
monthly_anomalies_gw.mean(axis=1).plot()
monthly_anomalies_q.mean(axis=1).plot(linewidth=1)

#VHI and NDVI
spatial_mean_anomalies_vars[3].plot() #LST
spatial_mean_anomalies_vars[4].plot()
spatial_mean_anomalies_indic[1].plot()
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

def corr_df(independent,dependent_arrays,offset):
    rs = []
    ps = []
    for dependent in dependent_arrays:
        slope, intercept, r_value, p_value, std_err = stats.linregress(independent[offset:], dependent[offset:])
        rs.append(r_value)
        ps.append(p_value)
    corr_dataframe = pd.DataFrame({'R':rs,'P-Val':ps})
    return corr_dataframe

def analyze_lag(var1,var2):
    from scipy import stats
    rs = []
    ps = []
    for i in range(0,25):
        if i == 0:
            variable1 = var1
            variable2 = var2
        else:
            variable1 = var1[:-i]
            variable2 = var2[i:]
        slope, intercept, r_value, p_value, std_err = stats.linregress(variable1, variable2)
        rs.append(round(r_value, 5))
        ps.append(round(p_value, 5))
    lag_dataframe = pd.DataFrame({'R':rs,'P-Val':ps})
    return lag_dataframe

def linear_plot(independent,dependent,var1,var2):
    slope, intercept, r_value, p_value, std_err = stats.linregress(independent, dependent)

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    ax.scatter(independent,dependent,s=4)
    ax.plot([], [], ' ', label='p-val = {}'.format(round(p_value, 3)))
    ax.plot([], [], ' ', label='r: {}'.format(round(r_value,5)))
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.legend(loc='lower right')
    ax.set_title('{} vs. {}'.format(var1,var2))


#Correlations
dependent_vars = spatial_mean_anomalies_vars
del dependent_vars[-2]
#Variable correlations
et_corr_df = corr_df(spatial_mean_anomalies_vars[0],dependent_vars)
gw_corr_df = corr_df(spatial_mean_anomalies_vars[2],dependent_vars)
lst_corr_df = corr_df(spatial_mean_anomalies_vars[3],dependent_vars)
ndvi_corr_df = corr_df(spatial_mean_anomalies_vars[4],dependent_vars)
gpm_corr_df = corr_df(spatial_mean_anomalies_vars[6],dependent_vars)
sm_corr_df = corr_df(spatial_mean_anomalies_vars[-1],dependent_vars)
spei12_corr_df = corr_df(spatial_mean_anomalies_indic[2],dependent_vars,11)
spei3_corr_df = corr_df(spatial_mean_anomalies_indic[1],dependent_vars,2)

#Index correlations
et_corr_df = corr_df(spatial_mean_anomalies_vars[0],spatial_mean_anomalies_indic)
gw_corr_df = corr_df(spatial_mean_anomalies_vars[2],spatial_mean_anomalies_indic)
lst_corr_df = corr_df(spatial_mean_anomalies_vars[3],spatial_mean_anomalies_indic)
ndvi_corr_df = corr_df(spatial_mean_anomalies_vars[4],spatial_mean_anomalies_indic)
gpm_corr_df = corr_df(spatial_mean_anomalies_vars[6],spatial_mean_anomalies_indic)
sm_corr_df = corr_df(spatial_mean_anomalies_vars[-1],spatial_mean_anomalies_indic)
vhi_corr_df = corr_df(spatial_mean_anomalies_indic[0],spatial_mean_anomalies_indic)
spei12_corr_df = corr_df(spatial_mean_anomalies_indic[2],spatial_mean_anomalies_indic,11)
spei3_corr_df = corr_df(spatial_mean_anomalies_indic[1],spatial_mean_anomalies_indic,3)

#InSitu
et_corr_df = corr_df(spatial_mean_anomalies_vars[0],[monthly_anomalies_gw.mean(axis=1),monthly_anomalies_q.mean(axis=1)])
gw_corr_df = corr_df(spatial_mean_anomalies_vars[2],[monthly_anomalies_gw.mean(axis=1),monthly_anomalies_q.mean(axis=1)])
lst_corr_df = corr_df(spatial_mean_anomalies_vars[3],[monthly_anomalies_gw.mean(axis=1),monthly_anomalies_q.mean(axis=1)])
ndvi_corr_df = corr_df(spatial_mean_anomalies_vars[4],[monthly_anomalies_gw.mean(axis=1),monthly_anomalies_q.mean(axis=1)])
gpm_corr_df = corr_df(spatial_mean_anomalies_vars[6],[monthly_anomalies_gw.mean(axis=1),monthly_anomalies_q.mean(axis=1)])
sm_corr_df = corr_df(spatial_mean_anomalies_vars[-1],[monthly_anomalies_gw.mean(axis=1),monthly_anomalies_q.mean(axis=1)])
vhi_corr_df = corr_df(spatial_mean_anomalies_indic[0],[monthly_anomalies_gw.mean(axis=1),monthly_anomalies_q.mean(axis=1)])
spei1_corr_df = corr_df(spatial_mean_anomalies_indic[-1],[monthly_anomalies_gw.mean(axis=1),monthly_anomalies_q.mean(axis=1)])
spei3_corr_df = corr_df(spatial_mean_anomalies_indic[1],[monthly_anomalies_gw.mean(axis=1),monthly_anomalies_q.mean(axis=1)],2)
spei12_corr_df = corr_df(spatial_mean_anomalies_indic[2],[monthly_anomalies_gw.mean(axis=1),monthly_anomalies_q.mean(axis=1)],11)

#Lags
var_names = ['ET', 'ET_P','GW','LST','NDVI','PPT_CHIRPS','PPT_GPM','RZ','SMAP','SM_Surf']
indices_names = ['VHI','SPEI-3','SPEI-12','SPEI-1'] 

var1 = 2
var2 = 0

#Variable vs. Variable
print(var_names[var1])
print(var_names[var2])
lag = analyze_lag(spatial_mean_anomalies_vars[var1],spatial_mean_anomalies_vars[var2])
print(lag)

#Variable vs. Index
print(var_names[var1])
print(indices_names[var2])
lag = analyze_lag(spatial_mean_anomalies_vars[var1],spatial_mean_anomalies_indic[var2])
print(lag)

#Index vs. Index
print(indices_names[var1])
print(indices_names[var2])
lag = analyze_lag(spatial_mean_anomalies_indic[var1],spatial_mean_anomalies_indic[var2])
print(lag)


#Cross-Correlation
corr_matrix = np.correlate(spatial_mean_anomalies_vars[0],spatial_mean_anomalies_vars[0],mode='full')

#3 minutes to run below
linear_plot(spatial_mean_anomalies_vars[4],spatial_mean_anomalies_vars[3],'NDVI','LST') #NDVI vs. LST

linear_plot(spatial_mean_anomalies_vars[4][11:],spatial_mean_anomalies_indic[2][11:],'NDVI','SPEI-12') #NDVI vs. SPEI-12    
linear_plot(spatial_mean_anomalies_vars[4][2:],spatial_mean_anomalies_indic[1][2:],'NDVI','SPEI-3')  #NDVI vs. SPEI-3
linear_plot(spatial_mean_anomalies_vars[4][0:],spatial_mean_anomalies_indic[3][0:],'NDVI', 'SPEI-1')  #NDVI vs. SPEI-1
linear_plot(spatial_mean_anomalies_vars[4],spatial_mean_anomalies_indic[0],'NDVI','VHI') #NDVI vs. VHI (most)

linear_plot(spatial_mean_anomalies_vars[-1][11:],spatial_mean_anomalies_indic[2][11:],'CLSM SM','SPEI-12')  #CLSM SM vs. SPEI-12
linear_plot(spatial_mean_anomalies_vars[-1][2:],spatial_mean_anomalies_indic[1][2:],'CLSM SM', 'SPEI-3')  #CLSM SM vs. SPEI-3
linear_plot(spatial_mean_anomalies_vars[-1],spatial_mean_anomalies_indic[0], 'CLSM SM', 'VHI') #CLSM SM vs. VHI (most)
linear_plot(spatial_mean_anomalies_vars[-1][0:],spatial_mean_anomalies_indic[3][0:],'CLSM SM', 'SPEI-1')  #CLSM SM vs. SPEI-1

linear_plot(spatial_mean_anomalies_vars[2][11:],spatial_mean_anomalies_indic[2][11:], 'GW','SPEI-12') #GW vs. SPEI-12 (most)
linear_plot(spatial_mean_anomalies_vars[2][2:],spatial_mean_anomalies_indic[1][2:], 'GW',' SPEI-3')  #GW vs. SPEI-3
linear_plot(spatial_mean_anomalies_vars[2],spatial_mean_anomalies_indic[0],'GW','VHI')  #GW vs. VHI
linear_plot(spatial_mean_anomalies_vars[2][0:],spatial_mean_anomalies_indic[3][0:],'GW','SPEI-1')  #GW vs. SPEI-1

linear_plot(spatial_mean_anomalies_indic[0][11:],spatial_mean_anomalies_indic[2][11:], 'VHI','SPEI-12') #VHI vs. SPEI-12 
linear_plot(spatial_mean_anomalies_indic[0][2:],spatial_mean_anomalies_indic[1][2:],'VHI','SPEI-3')  #VHI vs. SPEI-3 (most)
linear_plot(spatial_mean_anomalies_indic[0][1:],spatial_mean_anomalies_indic[3][1:],'VHI','SPEI-1')  #VHI vs. SPEI-1
linear_plot(spatial_mean_anomalies_indic[1][11:],spatial_mean_anomalies_indic[2][11:],'SPEI-3','SPEI-12')  #SPEI-3 vs. SPEI-12 (less correlation than expected)

linear_plot(spatial_mean_anomalies_vars[6][11:],spatial_mean_anomalies_indic[2][11:],'GPM','SPEI-12') 
linear_plot(spatial_mean_anomalies_vars[6][2:],spatial_mean_anomalies_indic[1][2:],'GPM', 'SPEI-3') 
linear_plot(spatial_mean_anomalies_vars[6],spatial_mean_anomalies_indic[0], 'GPM', 'VHI')
linear_plot(spatial_mean_anomalies_vars[6][0:],spatial_mean_anomalies_indic[3][0:],'GPM', 'SPEI-1') 

#In-Situ Comparisons
linear_plot(monthly_anomalies_gw.mean(axis=1),spatial_mean_anomalies_vars[2],'InSitu GW','CLSM GW')
linear_plot(monthly_anomalies_gw.mean(axis=1),monthly_anomalies_q.mean(axis=1),'InSitu GW','InSitu Q')
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
