import datetime
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
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from pathlib import Path
import glob
import rasterio as rio
import csv
import salem
import skimage
import statsmodels as sm 
import statsmodels.graphics.tsaplots as tsaplots
import statsmodels.api as smapi
from sklearn.linear_model import LinearRegression


def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

def read_file(file):
    with rio.open(file) as src:
        return(src.read())

def anomaly(dataset):
    ds_anomaly = dataset - dataset.mean(dim='time')
    d_anomaly_df = ds_anomaly.mean(['x','y']).to_dataframe()
    return ds_anomaly, d_anomaly_df

def deseason(df):
    decomp = smapi.tsa.seasonal_decompose(df, model='additive', period=12)
    decomp_df = pd.DataFrame(decomp.seasonal)
    decomp_df.columns=[df.name]
    deseas_df = df - decomp_df.squeeze()
    return decomp, decomp_df, deseas_df


def month_anomaly(dataset):
    month_idxs=dataset.groupby('time.month').groups

    dataset_month_anomalies = []
    months=range(1,13)
    for month in months:
        idxs = month_idxs[month]
        ds_month_avg = dataset.isel(time=idxs).mean(dim='time')
        dataset_month_anomaly = dataset.isel(time=idxs) - ds_month_avg
        dataset_month_anomalies.append(dataset_month_anomaly)

    monthly_anomalies_ds = xr.merge(dataset_month_anomalies)
    return(monthly_anomalies_ds)

def month_anomaly(dataset):
    month_idxs=dataset.groupby('time.month').groups

    dataset_month_anomalies = []
    months=range(1,13)
    for month in months:
        idxs = month_idxs[month]
        ds_month_avg = dataset.isel(time=idxs).mean(dim='time')
        dataset_month_anomaly = dataset.isel(time=idxs) - ds_month_avg
        dataset_month_anomalies.append(dataset_month_anomaly)
    monthly_anomalies_ds = xr.merge(dataset_month_anomalies)

    return(monthly_anomalies_ds)


########################################
#Precipitation
#GPM
p = xr.open_mfdataset(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GPM_IMERG/*.nc')
p_df = p.P.mean(dim=['x','y']).to_dataframe()
p_df = p_df.P

p_anomaly = month_anomaly(p)
p_anomaly_df = p_anomaly.P.mean(dim=['x','y']).to_dataframe()
p_anomaly_df_index = p_anomaly_df['P']/p_df.std()

#CHIRPS (USE CHIRPS NEXT TIME)
pc = xr.open_mfdataset(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\CHIRPS/*.nc')
pc_df = pc.P.mean(dim=['x','y']).to_dataframe()
pc_df = pc_df[20:240].P

p_anomaly, p_anomaly_df = anomaly(p)
p_decomp, p_decomp_df, p_deseas_df = deseason(p_df)
########################################
#Evapotranspiration
#et = xr.open_mfdataset(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_ET/*.nc',parallel=True,chunks={"x": 100,"y":100})
#et_df = et.ET.mean(dim=['lat','lon']).to_dataframe()

#Terra Only (Aqua is missing 6 weeks in 2016 February/March)
et = xr.open_mfdataset(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_ET\TERRA\TMODIS_ET_2000_2020_120421.nc',parallel=True,chunks={"x": 100,"y":100})
et_df = et.ET.mean(dim=['x','y']).to_dataframe()
et_std = et_df.std().ET
et_df = et_df[33::].ET

et_anomaly = xr.open_mfdataset(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_ET\TERRA\TMODIS_ET_2000_2020_120421_anomalies.nc')
et_anomaly_df = et_anomaly.to_dataframe()
et_anomaly_df_index = (et_anomaly_df['ET']/et_std)[33::]

et_decomp, et_decomp_df, et_deseas_df = deseason(et_df)
########################################
#Runoff (incorrectly labeled as "SM", but still runoff)
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_RUNOFF'
files = sorted(glob.glob(path+"/*.nc"))
r = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 10,"lon":10})

r_df = r.mean(['x','y'])
r_std = r_df.std().SM
r_df = r_df.rename({'SM':'R'}).to_dataframe()
r_df = r_df[33:252].R

r_anomaly = month_anomaly(r)
r_anomaly_df = r_anomaly.SM.mean(dim=['x','y']).to_dataframe()
r_anomaly_df_index = (r_anomaly_df['SM']/float(r_std))[33:252]

r_decomp, r_decomp_df, r_deseas_df = deseason(r_df)
########################################
#Soil Moisture
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GLDAS_SOILMOISTURE'
files = sorted(glob.glob(path+"/*.nc"))
sm = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 10,"lon":10})

sm_df = sm.mean(['x','y']).to_dataframe()
sm_std = sm_df.std().SM
sm_df = sm_df.SM[33:252]

sm_anomaly = month_anomaly(sm)
sm_anomaly_df = sm_anomaly.SM.mean(dim=['x','y']).to_dataframe()
sm_anomaly_df_index = (sm_anomaly_df['SM']/float(sm_std))[33:252]

sm_decomp, sm_decomp_df, sm_deseas_df = deseason(sm_df)
########################################
#NDVI
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_NDVI'
files = sorted(glob.glob(path+"/*.nc"))
ndvi = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 100,"lon":100})

ndvi_df = ndvi.mean(['x','y']).to_dataframe()
ndvi_std = ndvi_df.std().NDVI
ndvi_df = ndvi_df.NDVI[3:222]

ndvi_anomaly = xr.open_mfdataset(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_NDVI\NDVI_ANOMALIES.nc')
ndvi_anomaly_df = ndvi_anomaly.to_dataframe()
ndvi_anomaly_df_index = (ndvi_anomaly_df['NDVI']/ndvi_std)[3:222]

ndvi_decomp, ndvi_decomp_df, ndvi_deseas_df = deseason(ndvi_df)
########################################
#AIRS TEMPERATURES

#ENSO
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\AIRS'
file = sorted(glob.glob(path+"\SKIN\ENSO\*.nc"))
enso_skin = xr.open_mfdataset(file,parallel=True,chunks={"lat": 1,"lon":1})
enso_skin_df = enso_skin.mean(['x','y']).to_dataframe()
enso_skin_std = enso_skin_df.std().SKIN
enso_skin_df = enso_skin_df.SKIN[1::]

enso_skin_anomaly = month_anomaly(enso_skin)
enso_skin_anomaly_df = enso_skin_anomaly.SKIN.mean(dim=['x','y']).to_dataframe()[1::]
enso_skin_anomaly_df_index = (enso_skin_anomaly_df['SKIN']/(float(enso_skin_std)))

file = sorted(glob.glob(path+"\AIR_SURFACE\ENSO\*.nc"))
enso_air = xr.open_mfdataset(file,parallel=True,chunks={"lat": 1,"lon":1})
enso_air_df = enso_air.mean(['x','y']).to_dataframe()
enso_air_std = enso_air_df.std().AIR
enso_air_df = enso_air_df.AIR[1::]

enso_air_anomaly = month_anomaly(enso_air)
enso_air_anomaly_df = enso_air_anomaly.AIR.mean(dim=['x','y']).to_dataframe()[1::]
enso_air_anomaly_df_index = (enso_air_anomaly_df['AIR']/float(enso_air_std))

file = sorted(glob.glob(path+"\TROP\ENSO\*.nc"))
enso_trop = xr.open_mfdataset(file,parallel=True,chunks={"lat": 1,"lon":1})
enso_trop_df = enso_trop.mean(['x','y']).to_dataframe()
enso_trop_df = enso_trop_df.TROP[1::]

enso_decomp, enso_decomp_df, enso_deseas_df = deseason(enso_skin_df)

#IOD
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\AIRS'
file = sorted(glob.glob(path+"\SKIN\IOD\*.nc"))
iod_skin = xr.open_mfdataset(file,parallel=True,chunks={"lat": 1,"lon":1})
iod_skin_df = iod_skin.mean(['x','y']).to_dataframe()
iod_skin_std = iod_skin_df.std().SKIN
iod_skin_df = iod_skin_df.SKIN[1::]

iod_skin_anomaly = month_anomaly(iod_skin)
iod_skin_anomaly_df = iod_skin_anomaly.SKIN.mean(dim=['x','y']).to_dataframe()[1::]
iod_skin_anomaly_df_index = (iod_skin_anomaly_df['SKIN']/float(iod_skin_std))

file = sorted(glob.glob(path+"\AIR_SURFACE\IOD\*.nc"))
iod_air = xr.open_mfdataset(file,parallel=True,chunks={"lat": 1,"lon":1})
iod_air_df = iod_air.mean(['x','y']).to_dataframe()
iod_air_std = iod_air_df.std().AIR
iod_air_df = iod_air_df.AIR[1::]

iod_air_anomaly = month_anomaly(iod_air)
iod_air_anomaly_df = iod_air_anomaly.AIR.mean(dim=['x','y']).to_dataframe()[1::]
iod_air_anomaly_df_index = (iod_air_anomaly_df['AIR']/float(iod_air_std))


file = sorted(glob.glob(path+"\TROP\IOD\*.nc"))
iod_trop = xr.open_mfdataset(file,parallel=True,chunks={"lat": 1,"lon":1})
iod_trop_df = iod_trop.mean(['x','y']).to_dataframe()
iod_trop_df = iod_trop_df.TROP[1::]

iod_decomp, iod_decomp_df, iod_deseas_df = deseason(iod_skin_df)

#LIMPOPO
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\AIRS'
file = sorted(glob.glob(path+"\SKIN\LIMPOPO\*.nc"))
limpopo_skin = xr.open_mfdataset(file,parallel=True,chunks={"lat": 1,"lon":1})
limpopo_skin_df = limpopo_skin.mean(['x','y']).to_dataframe()
limpopo_skin_std = limpopo_skin_df.std().SKIN
limpopo_skin_df = limpopo_skin_df.SKIN[1::]

limpopo_skin_anomaly = month_anomaly(limpopo_skin)
limpopo_skin_anomaly_df = limpopo_skin_anomaly.SKIN.mean(dim=['x','y']).to_dataframe()[1::]
limpopo_skin_anomaly_df_index = (limpopo_skin_anomaly_df['SKIN']/float(limpopo_skin_std))

file = sorted(glob.glob(path+"\AIR_SURFACE\LIMPOPO\*.nc"))
limpopo_air = xr.open_mfdataset(file,parallel=True,chunks={"lat": 1,"lon":1})
limpopo_air_df = limpopo_air.mean(['x','y']).to_dataframe()
limpopo_air_std = limpopo_air.std().AIR
limpopo_air_df = limpopo_air_df.AIR[1::]

limpopo_air_anomaly = month_anomaly(limpopo_air)
limpopo_air_anomaly_df = limpopo_air_anomaly.AIR.mean(dim=['x','y']).to_dataframe()[1::]
limpopo_air_anomaly_df_index = (limpopo_air_anomaly_df['AIR']/float(limpopo_air_std))

file = sorted(glob.glob(path+"\TROP\LIMPOPO\*.nc"))
limpopo_trop = xr.open_mfdataset(file,parallel=True,chunks={"lat": 1,"lon":1})
limpopo_trop_df = limpopo_trop.mean(['x','y']).to_dataframe()
limpopo_trop_df = limpopo_trop_df.TROP[1::]

limpopo_decomp, limpopo_decomp_df, limpopo_deseas_df = deseason(limpopo_air_df)

########################################
#Change in Total Water Storage

#Analyzed the code below in Excel to make dataset continuous
#Need to code faster method later (12/3/21)

'''
path = r'C:\Users\robin\Box\Data\Groundwater\GRACE_MASCON_RL06_V2'
files = sorted(glob.glob(path+"/*.nc"))
grace = xr.open_mfdataset(files[1],parallel=True,chunks={"lat": 10,"lon":10}).lwe_thickness.salem.roi(shape=shapefile)
scale_factor = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 10,"lon":10}).salem.roi(shape=shapefile)
land_mask = xr.open_mfdataset(files[3],parallel=True,chunks={"lat": 10,"lon":10})

shpname = r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
shapefile = gpd.read_file(shpname)

scaled_grace = grace * scale_factor.scale_factor
twsa = scaled_grace - scaled_grace.mean(dim='time')
twsa_df = twsa.mean(['lat','lon']).to_dataframe('Values')
twsa_df = twsa_df[3:192]

os.chdir(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED')
twsa_df.to_csv('TWSA_grace.csv')  #need to manually introduce time gap for interpolation
'''

twsa_df = pd.read_csv(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\TWSA_grace.csv')
twsa_df.interpolate(method='linear',inplace=True)

twsa_df_values = twsa_df.Values[1::] #remove september 2002
twsa_df_dates = pd.date_range(start='10-01-2002',end='01-01-2021',freq='1M')
twsa_df = pd.DataFrame({"Values": list(twsa_df_values)},index=twsa_df_dates)

decomp_twsa = smapi.tsa.seasonal_decompose(twsa_df,model='additive',period=12)
#decomp_twsa.plot()
decomp_twsa_df = pd.DataFrame(decomp_twsa.seasonal)
decomp_twsa_df.columns=['Values']
twas_deseas = twsa_df - decomp_twsa_df

########################################
#Modeled Q (Q = P - ET - dS) ????
#03-06-2022 does this work?

#P
p_df

#ET
et_df

#dS
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
dS = scaled_grace.salem.roi(shape=shapefile)
dS_df_values = dS.mean(['lat','lon']).to_dataframe('Values')

#os.chdir(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED')
#dS_df_values.to_csv('dS_grace.csv') #need to manually introduce time gap

dS_df = pd.read_csv(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\dS_grace.csv')
dS_df.interpolate(method='linear',inplace=True)
dS_df_values = dS_df.Values[1::] #remove september 2002
dS_df_dates = pd.date_range(start='10-01-2002',end='06-01-2021',freq='1M')
dS_df = pd.DataFrame({"Values": list(dS_df_values)},index=dS_df_dates)
dS_df = dS_df.Values[0:219]

#Q
Q = p_df - et_df - dS_df
Q.plot()
plt.plot(Q,r_df,'*')

##########################################################
#PLOTS

fig1, ax = plt.subplots(figsize=(11,5))
plt.rc('font', size = 8)
#ax.plot(sm_df,color='blue')
ax.plot(twas_deseas,color='green')
ax.plot(twsa_df,color='red')
ax.legend(['Observed','Observed without seasonality','Anomaly'])

twsa_df_anomaly = month_anomaly(twsa_df)

fig1, ax = plt.subplots(figsize=(11,5))
plt.rc('font', size = 8)
ax.plot(twsa_df,color='blue')
ax.plot(decomp_twsa_df,color='red')
ax.plot(twas_deseas,color='green')
ax.legend(['Observed','Seasonality','Observed without seasonality'])

#Water Balance Equation
print(len(p_df))
print(len(et_df))
print(len(r_df))
lhs = p_df - et_df - r_df
dates = pd.date_range(start='10-01-2002',end='01-01-2021',freq='1M')

#Without Lag
plt.rc('font', size = 40)
fig = plt.figure(figsize=(22,10))
ax = fig.add_subplot()
ax2 = ax.twinx()
ax.grid(True)
ax.plot(dates,lhs,linewidth=3,color='gray',label='P-ET-R')
ax2.plot(dates,twsa_df,linewidth=6,color='c',label='TWS Anomaly')
ax2.set_ylim(-15,20)
ax.set_ylim(-112.5,150)
ax.set_ylabel('P-ET-R')
ax2.set_ylabel('Total Water Storage Anomaly',rotation=-90, labelpad=45)
ax.legend(loc='upper right')
ax2.legend(loc='upper right',bbox_to_anchor=(1, 0.9))
#ax.set_xlabel('Year')
ax.set_xlim([datetime.date(2002, 10, 31), datetime.date(2020, 12, 31)])

#With Lag
plt.rc('font', size = 40)
fig = plt.figure(figsize=(22,10))
ax = fig.add_subplot()
ax2 = ax.twinx()
ax.grid(True)
ax.plot(dates[3::],lhs[:-3],linewidth=3,color='gray',label='P-ET-R')
ax2.plot(dates[3::],twsa_df[3::],linewidth=6,color='c',label='TWS Anomaly')
ax2.set_ylim(-15,20)
ax.set_ylim(-112.5,150)
ax.set_ylabel('P-ET-R')
ax2.set_ylabel('Total Water Storage Anomaly',rotation=-90, labelpad=45)
ax.legend(loc='upper right')
ax2.legend(loc='upper right',bbox_to_anchor=(1, 0.9))
#ax.set_xlabel('Year')
ax.set_xlim([datetime.date(2002, 10, 31), datetime.date(2020, 12, 31)])

plt.rc('font', size = 40)
fig = plt.figure(figsize=(22,10))
ax = fig.add_subplot()
ax2 = ax.twinx()
ax.grid(True)
plt.rc('font', size = 20)
ax.plot(dates,et_anomaly_df_index,'-.',linewidth=3,color='black',label='ET')
ax.plot(dates,sm_anomaly_df_index,linewidth=3,color='black',label='Soil Moisture')
ax.plot(dates,ndvi_anomaly_df_index,'--',linewidth=3,color='black',label='NDVI')
ax2.plot(dates,twsa_df,linewidth=6,color='c',label='TWS Anomaly')
ax2.set_ylim(-20,20)
ax.set_ylim(-2,2)
ax.set_ylabel('Anomaly Index')
ax2.set_ylabel('Total Water Storage Anomaly',rotation=-90, labelpad=45)
ax.legend(loc='upper right')
ax2.legend(loc='lower left')
#ax.set_xlabel('Year')
ax.set_xlim([datetime.date(2002, 10, 31), datetime.date(2020, 12, 31)])

def linear_plot(independent,dependent,ind_label,d_label,color_choice):
    y = np.array(dependent)
    x = np.array(independent)
    X = smapi.add_constant(x)
    est = smapi.OLS(y, X)
    model = est.fit()
    plt.rc('font', size = 30)
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot()
    ax.scatter(independent,dependent,color=color_choice)
    ax.plot([], [], ' ', label='R2 = {}'.format(round(model.rsquared, 3)))
    ax.legend(loc='upper right')
    ax.set_xlabel(ind_label)
    ax.set_ylabel(d_label)
    
linear_plot(lhs[:-3],twsa_df[3::],'P-ET-R','TWS Anomaly','gray')

plt.rc('font', size = 30)
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot()
ax.scatter(lhs[:-2],twsa_df[2::],color='gray',label='TWS Anomaly')
ax.set_xlabel('P-ET-R')
ax.set_ylabel('Total Water Storage Anomaly')


#P,ET,R,SM,NDVI
plt.rc('font', size = 25)
fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1,figsize=(25,20))

p_df.plot.bar(ax=ax1,color='black')
ax1.set_xticks(range(3,221,12))
ax1.set_xticklabels([i for i in range(2003,2022,1)], rotation=15, Fontsize=20)

et_df.plot.bar(ax=ax2,color='black')
ax2.set_xticks(range(3,221,12))
ax2.set_xticklabels([i for i in range(2003,2022,1)], rotation=15, Fontsize=20)

r_df.plot.bar(ax=ax3,color='black')
ax3.set_xticks(range(3,221,12))
ax3.set_xticklabels([i for i in range(2003,2022,1)], rotation=15, Fontsize=20)

sm_df.plot.bar(ax=ax4,color='black')
ax4.set_xticks(range(3,221,12))
ax4.set_xticklabels([i for i in range(2003,2022,1)], rotation=15, Fontsize=20)

ndvi_df.plot.bar(ax=ax5,color='black')
ax5.set_xticks(range(3,221,12))
ax5.set_xticklabels([i for i in range(2003,2022,1)], rotation=15, Fontsize=20)

ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
ax4.grid(True)
ax5.grid(True)

ax1.set_xlabel(None)
ax1.set_ylabel('P (mm)',fontsize='35')
ax2.set_xlabel(None)
ax2.set_ylabel('ET (kg/m{})'.format(get_super('2')), fontsize='35',labelpad=25)
ax3.set_xlabel(None)
ax3.set_ylabel('R (kg/m{})'.format(get_super('2')), fontsize='35',labelpad=25)
ax4.set_xlabel(None)
ax4.set_ylabel('SM (kg/m{})'.format(get_super('2')),fontsize='35')
ax5.set_xlabel(None)
ax5.set_ylabel('NDVI',fontsize='35')

'''
#SST for ENSO/IOD; LST for Limpopo
plt.rc('font', size = 10)
fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(11,10))

enso_skin_df.plot.bar(ax=ax1,color='orange')
ax1.set_xticks(range(3,221,12))
ax1.set_xticklabels([i for i in range(2003,2022,1)], rotation=0)
ax1.set_ylim(24,30)

iod_skin_df.plot.bar(ax=ax2,color='orange')
ax2.set_xticks(range(3,221,12))
ax2.set_xticklabels([i for i in range(2003,2022,1)], rotation=0)
ax2.set_ylim(24,30)

limpopo_skin_df.plot.bar(ax=ax3,color='orange')
ax3.set_xticks(range(3,221,12))
ax3.set_xticklabels([i for i in range(2003,2022,1)], rotation=0)
ax3.set_ylim(15,35)

ax1.grid(True)
ax2.grid(True)
ax3.grid(True)

ax1.set_xlabel(None)
ax1.set_ylabel('SST (dC)')
ax2.set_xlabel(None)
ax2.set_ylabel('SST (dC)')
ax3.set_xlabel(None)
ax3.set_ylabel('LST (dC)')
'''

#Anomalies of Timeseries
plt.rc('font', size = 30)
fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1,figsize=(22,20))

ax1.plot(dates,p_anomaly_df_index)
ax1.yaxis.tick_right()
ax1.fill_between(dates, p_anomaly_df_index, 0, alpha=0.30)
ax1.set_ylim(-2,2)
ax1.set_xlim([datetime.date(2002, 10, 31), datetime.date(2020, 12, 31)])

ax2.plot(dates,et_anomaly_df_index)
ax2.yaxis.tick_right()
ax2.fill_between(dates, et_anomaly_df_index, 0, alpha=0.30)
ax2.set_ylim(-2,2)
ax2.set_xlim([datetime.date(2002, 10, 31), datetime.date(2020, 12, 31)])

ax3.plot(dates,r_anomaly_df_index)
ax3.yaxis.tick_right()
ax3.fill_between(dates, r_anomaly_df_index, 0, alpha=0.30)
ax3.set_ylim(-2,2)
ax3.set_xlim([datetime.date(2002, 10, 31), datetime.date(2020, 12, 31)])

ax4.plot(dates,sm_anomaly_df_index)
ax4.yaxis.tick_right()
ax4.fill_between(dates, sm_anomaly_df_index, 0, alpha=0.30)
ax4.set_ylim(-2,2)
ax4.set_xlim([datetime.date(2002, 10, 31), datetime.date(2020, 12, 31)])

ax5.plot(dates,ndvi_anomaly_df_index)
ax5.yaxis.tick_right()
ax5.fill_between(dates, ndvi_anomaly_df_index, 0, alpha=0.30)
ax5.set_ylim(-2,2)
ax5.set_xlim([datetime.date(2002, 10, 31), datetime.date(2020, 12, 31)])

ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
ax4.grid(True)
ax5.grid(True)


#SST for ENSO/IOD; LST for Limpopo
plt.rc('font', size = 40)
fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(22,20))

ax1twin = ax1.twinx()
ax1.plot(dates,enso_skin_df,'--',label='Temperature')
ax1twin.plot(dates,enso_skin_anomaly_df_index,label='Anomaly Index')
ax1twin.fill_between(dates, enso_skin_anomaly_df_index, 0, alpha=0.30)
ax1twin.set_ylim(-3,3)
ax1.set_ylim((min(enso_skin_df) + max(enso_skin_df))/2-3,(min(enso_skin_df) + max(enso_skin_df))/2+3)
ax1.set_xlim([datetime.date(2002, 10, 31), datetime.date(2020, 12, 31)])
ax1twin.legend(loc='lower right')
ax1.legend(loc='upper left')
ax1.set_ylabel('ENSO SST (C)')

ax2twin = ax2.twinx()
ax2.plot(dates,iod_skin_df,'--',label='Temperature')
ax2twin.plot(dates,iod_skin_anomaly_df_index,label='Anomaly Index')
ax2twin.fill_between(dates, iod_skin_anomaly_df_index, 0, alpha=0.30)
ax2twin.set_ylim(-2,2)
ax2.set_ylim((min(iod_skin_df) + max(iod_skin_df))/2-2,(min(iod_skin_df) + max(iod_skin_df))/2+2)
ax2.set_xlim([datetime.date(2002, 10, 31), datetime.date(2020, 12, 31)])
ax2.set_ylabel('IOD SST (C)')

ax3twin = ax3.twinx()
ax3.plot(dates,limpopo_skin_df,'--',label='Temperature')
ax3twin.plot(dates,limpopo_skin_anomaly_df_index,label='Anomaly Index')
ax3twin.fill_between(dates, limpopo_skin_anomaly_df_index, 0, alpha=0.30)
ax3twin.set_ylim(-1,1)
ax3.set_ylim((min(limpopo_skin_df) + max(limpopo_skin_df))/2-9,(min(limpopo_skin_df) + max(limpopo_skin_df))/2+9)
ax3.set_xlim([datetime.date(2002, 10, 31), datetime.date(2020, 12, 31)])
ax3.set_ylabel('Limpopo LST (C)')

ax1.grid(True)
ax2.grid(True)
ax3.grid(True)


#AIR Temp for ENSO, IOD, Limpopo
plt.rc('font', size = 10)
fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(11,10))

ax1twin = ax1.twinx()
ax1.plot(dates,enso_air_df,'--',label='Temperature')
ax1twin.plot(dates,enso_air_anomaly_df_index,label='Anomaly Index')
ax1twin.fill_between(dates, enso_air_anomaly_df_index, 0, alpha=0.30)
ax1twin.set_ylim(-3,3)
ax1.set_ylim((min(enso_air_df) + max(enso_air_df))/2-3,(min(enso_air_df) + max(enso_air_df))/2+3)
ax1.set_xlim([datetime.date(2002, 10, 31), datetime.date(2020, 12, 31)])
ax1twin.legend(loc='upper right')
ax1.legend(loc='upper left')
ax1.set_ylabel('ENSO AT (Celsius)')

ax2twin = ax2.twinx()
ax2.plot(dates,iod_air_df,'--',label='Temperature')
ax2twin.plot(dates,iod_air_anomaly_df_index,label='Anomaly Index')
ax2twin.fill_between(dates, iod_air_anomaly_df_index, 0, alpha=0.30)
ax2twin.set_ylim(-2,2)
ax2.set_ylim((min(iod_air_df) + max(iod_air_df))/2-2,(min(iod_air_df) + max(iod_air_df))/2+2)
ax2.set_xlim([datetime.date(2002, 10, 31), datetime.date(2020, 12, 31)])
ax2.set_ylabel('IOD AT (Celsius)')

ax3twin = ax3.twinx()
ax3.plot(dates,limpopo_air_df,'--',label='Temperature')
ax3twin.plot(dates,limpopo_air_anomaly_df_index,label='Anomaly Index')
ax3twin.fill_between(dates, limpopo_air_anomaly_df_index, 0, alpha=0.30)
ax3twin.set_ylim(-1,1)
ax3.set_ylim((min(limpopo_air_df) + max(limpopo_air_df))/2-9,(min(limpopo_air_df) + max(limpopo_air_df))/2+9)
ax3.set_xlim([datetime.date(2002, 10, 31), datetime.date(2020, 12, 31)])
ax3.set_ylabel('Limpopo AT (Celsius)')

ax1.grid(True)
ax2.grid(True)
ax3.grid(True)


#Linear Regressions

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

'''
enso_skin_anomaly_df.SKIN
iod_skin_anomaly_df.SKIN
limpopo_skin_anomaly_df.SKIN
twsa_df
lhs
'''

p_anomaly_df_index
et_anomaly_df_index
r_anomaly_df_index
sm_anomaly_df_index
ndvi_anomaly_df_index
enso_skin_anomaly_df_index.rename('ENSO_SST')
iod_skin_anomaly_df_index.rename('IOD_SST')
limpopo_skin_anomaly_df_index.rename('LIMPOPO_LST')
limpopo_air_anomaly_df_index.rename('LIMPOPO_AT')

p_anomal_df_index_frame = p_anomaly_df_index.to_frame()
p_anomal_df_index_frame.insert(1,'ET',et_anomaly_df_index.to_frame())
p_anomal_df_index_frame.insert(2,'R',r_anomaly_df_index.to_frame())
p_anomal_df_index_frame.insert(3,'SM',sm_anomaly_df_index.to_frame())
p_anomal_df_index_frame.insert(4,'NDVI',ndvi_anomaly_df_index.to_frame())
p_anomal_df_index_frame.insert(5,'ENSO SST',enso_skin_anomaly_df_index.to_frame())
p_anomal_df_index_frame.insert(6,'IOD SST',iod_skin_anomaly_df_index.to_frame())
p_anomal_df_index_frame.insert(7,'LIMPOPO LST',limpopo_skin_anomaly_df_index.to_frame())
p_anomal_df_index_frame.insert(8,'TWSA',twsa_df)
p_anomal_df_index_frame.insert(9,'LIMPOPO AT',limpopo_air_anomaly_df_index.to_frame())
anomalies_index = p_anomal_df_index_frame

p_df_frame = p_df.to_frame()
p_df_frame.insert(1,'ET',et_df.to_frame())
p_df_frame.insert(2,'R',r_df.to_frame())
p_df_frame.insert(3,'SM',sm_df.to_frame())
p_df_frame.insert(4,'NDVI',ndvi_df.to_frame())
p_df_frame.insert(5,'ENSO SST',enso_skin_df.to_frame())
p_df_frame.insert(6,'IOD SST',iod_skin_df.to_frame())
p_df_frame.insert(7,'LIMPOPO LST',limpopo_skin_df.to_frame())
p_df_frame.insert(8,'TWSA',twsa_df)
time_series = p_df_frame

enso_air_df_frame = enso_air_df.to_frame()
enso_air_df_frame.insert(1,'IOD AIR',iod_air_df.to_frame())
enso_air_df_frame.insert(2,'LIMPOPO AIR',limpopo_air_df.to_frame())
enso_air_df_frame.insert(3,'ENSO TROP',enso_trop_df.to_frame())
enso_air_df_frame.insert(4,'IOD TROP',iod_trop_df.to_frame())
enso_air_df_frame.insert(5,'LIMPOPO TROP',limpopo_trop_df.to_frame())
air_series = enso_air_df_frame

def linear_plot(independent,dependent,ind_label,d_label,color_choice):
    y = np.array(dependent)
    x = np.array(independent)
    X = smapi.add_constant(x)
    est = smapi.OLS(y, X)
    model = est.fit()
    plt.rc('font', size = 30)
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot()
    ax.scatter(independent,dependent,color=color_choice)
    ax.plot([], [], ' ', label='R{} = {}'.format(get_super('2'),round(model.rsquared, 3)))
    ax.plot([], [], ' ', label='p-val = {}'.format(round(model.pvalues[-1], 3)))
    ax.legend(loc='upper right')
    ax.set_xlabel(ind_label)
    ax.set_ylabel(d_label)

for i in anomalies_index:
    linear_plot(anomalies_index.loc[:,'{}'.format(i)],twsa_df,'{} Anomaly Index'.format(i),'TWS Anomaly', 'black')

independent = enso_skin_anomaly_df_index
dependent = iod_skin_anomaly_df_index
ind_label = 'ENSO SST Anomaly'
d_label = 'IOD SST Anomaly'
color_choice = 'black'

linear_plot(independent,dependent,ind_label,d_label,color_choice)

#Mann Kendall Tests
import pymannkendall as mk

for i in anomalies_index:
    print(i)
    print(mk.seasonal_test(anomalies_index.loc[:,'{}'.format(i)],period=12))

for i in time_series:
    print(i)
    print(mk.seasonal_test(time_series.loc[:,'{}'.format(i)],period=12))

for i in air_series:
    print(i)
    print(mk.seasonal_test(air_series.loc[:,'{}'.format(i)],period=12))


#Lag Analysis
#calculate cross correlation

p_vals = []
r_sqs = []
for i in range(0,25):
    if i == 0:
        lag_SST = anomalies_index.loc[:,'ENSO SST']
        lag_TWSA = anomalies_index.loc[:,'IOD SST']
    else:
        lag_SST = anomalies_index.loc[:,'ENSO SST'][:-i]
        lag_TWSA = anomalies_index.loc[:,'IOD SST'][i::]
    y = np.array(lag_TWSA)
    x = np.array(lag_SST)
    X = smapi.add_constant(x)
    est = smapi.OLS(y, X)
    model = est.fit()
    r_sqs.append(round(model.rsquared, 3))
    p_vals.append(round(model.pvalues[-1], 3))

lag = pd.DataFrame({'R_sq':r_sqs,'P-Val':p_vals})

#correlation increases from months 3 [.106] to 9 [0.087]; peaks at month 7 [0.11]

#Cross Correlation
variable1 = enso_skin_anomaly_df.SKIN
variable2 = lhs
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

#ACF & PACF
smapi.graphics.tsa.plot_acf(variable1, lags=24)
plt.show()
smapi.graphics.tsa.plot_pacf(variable1, lags=24)
plt.show()

smapi.graphics.tsa.plot_acf(variable2, lags=24)
plt.show()
smapi.graphics.tsa.plot_pacf(variable2, lags=24)
plt.show()


#Stationarity

from statsmodels.tsa.stattools import adfuller
result = adfuller(variable1)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
if result[0] > result[4]["5%"]:
    print ("Failed to Reject Ho - Time Series is Non-Stationary")
else:
    print ("Reject Ho - Time Series is Stationary")

fig, ax = plt.subplots(figsize=(12,7))
ax.plot(variable1,linewidth=1.5)


#Tutorial
#https://timeseriesreasoning.com/contents/partial-auto-correlation/

df = enso_skin_anomaly_df
df['T_(i-1)'] = df['SKIN'].shift(1)
df['T_(i-2)'] = df['SKIN'].shift(2)

df = df.drop(df.index[[0,1]])

from sklearn import linear_model
lm = linear_model.LinearRegression()
 
df_X = df[['T_(i-1)']] #Note the double brackets! [[]]
df_y = df['SKIN'] #Note the single brackets! []
model = lm.fit(df_X,df_y)
#Predicted values based on regression of SKIN and T_(i-1)
df['Predicted_T_i|T_(i-1)'] = lm.predict(df_X) 
#Observed minus predicted
df['Residual_T_i|T_(i-1)'] = df['SKIN'] - df['Predicted_T_i|T_(i-1)']

#Let’s repeat the above procedure to calculate the second time series of residuals, 
#this time using the columns: T_(i-2) and T_(i-1).
lm = linear_model.LinearRegression()
 
df_X = df[['T_(i-1)']] #Note the double brackets! [[]]
df_y = df['T_(i-2)'] #Note the single brackets! []
model = lm.fit(df_X,df_y)
#Predicted values based on regression of T_(i-2) and T_(i-1)
df['Predicted_T_(i-2)|T_(i-1)'] = lm.predict(df_X)
#Residual = Observed - predicted
df['Residual_T_(i-2)|T_(i-1)'] = df['T_(i-2)'] - df['Predicted_T_(i-2)|T_(i-1)']

#Option 1
print(df.corr(method='pearson')['Residual_T_i|T_(i-1)']['Residual_T_(i-2)|T_(i-1)'])
#Option 2
from statsmodels.tsa.stattools import pacf
print(pacf(df['SKIN'], nlags=2)[2])


#Test ACF
df = limpopo_skin_df

independent = (df[24::])
dependent = df[:-24]

plt.rc('font', size = 20)
difference = np.array(independent) - np.array(dependent)
fig = plt.figure(figsize=(20,15))
ax = fig.add_subplot()
ax.plot(pd.date_range(start='10-01-2004',end='01-01-2021',freq='1M'),difference)

y = np.array(dependent)
x = np.array(independent)
X = smapi.add_constant(x)
est = smapi.OLS(y, X)
model = est.fit()
plt.rc('font', size = 30)
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot()
ax.scatter(independent,dependent)
ax.plot([], [], ' ', label='R{} = {}'.format(get_super('2'),round(model.rsquared, 3)))
ax.plot([], [], ' ', label='p-val = {}'.format(round(model.pvalues[-1], 3)))
ax.legend(loc='upper right')

smapi.graphics.tsa.plot_pacf(difference, lags=40)
plt.show()