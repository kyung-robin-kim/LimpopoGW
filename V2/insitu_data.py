import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from shapely.geometry import Point
import datetime
import glob

SA_gw_data = pd.read_csv(glob.glob(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\sophia\Limpopo In-Situ Data\Boreholes\*.csv')[-1])
insitu_points = [Point(xy) for xy in zip(SA_gw_data['Longitude'], SA_gw_data['Latitude'])]
insitu_gpd = gpd.GeoDataFrame(SA_gw_data, geometry=insitu_points).set_crs('EPSG:4326')

limpopo_watershed = gpd.read_file(r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp')
limpopo_gw_insitu = insitu_gpd[insitu_gpd.within(limpopo_watershed.loc[0,'geometry'])]


start = 40 #October 1999:0 .... Feb 2003:  January 2011: 135
years = 19
end = start + 12*years+1
#10 years = 132 months
limpopo_gw_insitu_filtered = limpopo_gw_insitu[limpopo_gw_insitu.filter(like='lev').iloc[:,start+1:end+1].T.count()>120]

limpopo_gw_insitu_filtered.plot()
for i, label in enumerate(limpopo_gw_insitu_filtered.index):
    plt.annotate(label, (limpopo_gw_insitu_filtered.Longitude.iloc[i], limpopo_gw_insitu_filtered.Latitude.iloc[i]), textcoords="offset points", xytext=(0,0), ha='center')



#Individual
limpopo_gw_insitu_filtered.iloc[:,585] #rocks
points = np.array([geom.xy for geom in limpopo_gw_insitu_filtered.geometry])
geology = limpopo_gw_insitu_filtered.iloc[:,584:586]
site_elevation= limpopo_gw_insitu_filtered.iloc[:,10].T
gw_masl = limpopo_gw_insitu_filtered.filter(like='masl').iloc[:,start:(end)].T
gw_level = limpopo_gw_insitu_filtered.filter(like='lev').iloc[:,start+1:(end+1)].T

dt_index = [pd.to_datetime(year_month[0:8],format='%Y-%b') for year_month in gw_level.index]
#dt_index = [pd.Period(pd.to_datetime(year_month[0:8],format='%Y-%b'),freq='M') for year_month in gw_level.index]
gw_level.index = dt_index
gw_level = gw_level.interpolate('linear')

gw_anomaly = (gw_level - gw_level.mean(axis=0))/ gw_level.std(axis=0,ddof=1)
gw_anomaly.mean(axis=1).plot()
(-2* gw_anomaly.std(axis=1,ddof=1) + gw_anomaly.mean(axis=1)).plot()
(2* gw_anomaly.std(axis=1,ddof=1) + gw_anomaly.mean(axis=1)).plot()


gw_anomaly_by_geology = [gw_anomaly.T[(geology.iloc[:,0]==index)] for index in np.unique(geology.iloc[:,0])]
gw_anomaly_by_geology = [gw_anomaly.T[(geology.iloc[:,0]==index)] for index in np.unique(geology.iloc[:,0])]

for i,ii in zip(range(0,len(np.unique(geology.iloc[:,0]))),np.unique(geology.iloc[:,0])):
    plt.figure()
    gw_anomaly_by_geology[i].T.plot(legend=False)
    plt.title(geology.iloc[:,1][geology.iloc[:,0]==ii].iloc[0])
    plt.ylim(-8,8)
    

#Clustered

# Flatten the array and transpose it to the correct shape for KMeans clustering
XY = points.reshape(-1, 2)
ELEV = np.array(site_elevation).reshape(-1,1)
GEO = np.array(geology.iloc[:,0]).reshape(-1,1)
XYZ = np.concatenate((XY,ELEV),axis=1)

ALL = np.concatenate((XYZ,GEO),axis=1)

#GEO = np.array(geology.iloc[:,0]).reshape(-1,1)

# Cluster the points using k-means
gw_clusters = gw_anomaly.T

cluster_no = 100
kmeans_xy = KMeans(n_clusters=cluster_no, random_state=0).fit(XY)
gw_clusters['cluster_xy'] = kmeans_xy.labels_
#limpopo_gw_insitu_xy_centroids = gw_clusters.dissolve(by='cluster_xy').centroid.reset_index()

cluster_no = 11
kmeans_elev = KMeans(n_clusters=cluster_no, random_state=0).fit(ELEV)
gw_clusters['cluster_elev'] = kmeans_elev.labels_
#limpopo_gw_insitu_elev_centroids = gw_clusters.dissolve(by='cluster_elev').centroid.reset_index()

cluster_no = 100
kmeans_xyz = KMeans(n_clusters=cluster_no, random_state=0).fit(XYZ)
gw_clusters['cluster_xyz'] = kmeans_xyz.labels_
#limpopo_gw_insitu_xyz_centroids = gw_clusters.dissolve(by='cluster_xyz').centroid.reset_index()

cluster_no = len(np.unique(GEO))
kmeans_geo = KMeans(n_clusters=cluster_no, random_state=0).fit(GEO)
gw_clusters['cluster_geo'] = kmeans_geo.labels_

cluster_no = 150
kmeans_all = KMeans(n_clusters=cluster_no, random_state=0).fit(ALL)
gw_clusters['cluster_all'] = kmeans_all.labels_

gw_clusters.groupby(by='cluster_elev').mean().iloc[:,0:-4].T.plot(legend=False) #.mean(axis=1)
gw_clusters.groupby(by='cluster_xy').mean().iloc[:,0:-4].T.plot(legend=False) #.mean(axis=1)
gw_clusters.groupby(by='cluster_geo').mean().iloc[:,0:-4].T.plot(legend=False) #.mean(axis=1)
gw_clusters.groupby(by='cluster_xyz').mean().iloc[:,0:-4].T.plot(legend=False) #.mean(axis=1)
gw_clusters.groupby(by='cluster_all').mean().iloc[:,0:-4].T.plot(legend=False) #.mean(axis=1)
gw_anomaly.plot(legend=False)


###########################################################################
#DISCHARGE DATA FROM GRDC
import xarray as xr
import pymannkendall as mk


#print(mk.seasonal_test(i.iloc[:,1],period=12))


q_data = glob.glob(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\GRDC\2023-05-03_00-21\*nc')[0]
insitu_q = xr.open_mfdataset(q_data)

runoff = insitu_q.runoff_mean[(insitu_q.time>pd.to_datetime('2003-02')) & (insitu_q.time<pd.to_datetime('2022-03'))].transpose('id','time')
runoff_filtered = runoff[runoff.count(dim='time')>6200].transpose('time','id')

station_id = insitu_q.station_name[runoff.count(dim='time')>6200]
geo_x = insitu_q.geo_x[runoff.count(dim='time')>6200]
geo_y = insitu_q.geo_y[runoff.count(dim='time')>6200]
geo_z = insitu_q.geo_z[runoff.count(dim='time')>6200]

insitu_points_q = [Point(xy) for xy in zip(geo_x, geo_y)]
insitu_gpd_q = gpd.GeoDataFrame(station_id, geometry=insitu_points_q).set_crs('EPSG:4326')

runoff_ln = np.log(runoff_filtered)
runoff_ln = runoff_ln.where(runoff_ln>-100, np.nan)
runoff_sqrt = np.sqrt(runoff)
runoff_sqrt = runoff_sqrt.where(runoff_sqrt>-100, np.nan)

transformed = runoff_ln
q_anomaly = (transformed - transformed.mean(dim='time'))/ transformed.std(dim='time',ddof=1)

'''
for i in range(0,len(runoff_filtered.id)):
    plt.figure()
    q_anomaly[:,i].plot()
'''
plt.rcParams["font.family"] = "Times New Roman"
q_anomaly.rolling(time=30, min_periods=1).mean().mean(dim='id').plot()
gw_anomaly.mean(axis=1).plot()
plt.xlabel('Date')
plt.ylabel('Standardized Unit')

#gw_anomaly.mean(axis=1).rolling(12).mean().plot(legend=False)





#####################################



#ANOMALIES (monthly) -- need to calculate ET/PET manually 
monthly_anom_files = sorted(glob.glob(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\*.csv'))
monthly_anoms = pd.concat([pd.read_csv(file).set_index('time') for file in monthly_anom_files],axis=1)
monthly_anoms.set_index(pd.to_datetime(monthly_anoms.index),inplace=True)
variables = ['LST','NDVI','GW','RZ','Surface SM','PPT','TCI','VCI','VHI']



[print(mk.seasonal_test(monthly_anoms.iloc[:,i],period=12)) for i in range(0,len(monthly_anoms.keys()))]



gw_anomaly.mean(axis=1).plot(legend=False,linewidth=2)
#gw_anomaly.std(ddof=1,axis=1).rolling(12).mean().plot(legend=False,linewidth=2)
gw_anomaly.mean(axis=1).rolling(12).mean().plot(legend=False)
#gw_anomaly.mean(axis=1).rolling(3).mean().plot(legend=False)
gw_anomaly.mean(axis=1).rolling(2).mean().plot(legend=False)






path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\PRECIPITATION'
files = sorted(glob.glob(path+'\*.nc'))
p_gpm = xr.open_mfdataset(files[1],parallel=True,chunks={"y": 100,"x":100}).P_mm
p_chirps = xr.open_mfdataset(files[0],parallel=True,chunks={"y": 100,"x":100}).P_mm

#Mean Precip
fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(p_chirps.time,p_gpm.mean(dim=['x','y']),color='C1')
ax.plot(p_chirps.time,p_chirps.mean(dim=['x','y']),color='C0')
ax.set_ylabel('Precipitation (mm)',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)

#Mean Precip (LOG-NORMAL - -INF removed)

log_p_gpm = xr.apply_ufunc(np.log, p_gpm, input_core_dims=[['time']], output_core_dims=[['time']], vectorize=True,dask='parallelized')
log_p_chirps = xr.apply_ufunc(np.log, p_chirps, input_core_dims=[['time']], output_core_dims=[['time']], vectorize=True,dask='parallelized')

log_p_gpm = log_p_gpm.where(log_p_gpm>-100, np.nan)
log_p_chirps = log_p_chirps.where(log_p_chirps>-100, np.nan)

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(p_chirps.time,np.log(p_gpm).where(np.log(p_gpm)>-100, np.nan).mean(dim=['x','y']),color='C1')
ax.plot(p_chirps.time,np.log(p_chirps).where(np.log(p_chirps)>-100, np.nan).mean(dim=['x','y']),color='C0')
ax.set_ylabel('Precipitation (mm)',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)


q_anomaly.rolling(time=30, min_periods=1).mean().mean(dim='id').plot()
gw_anomaly.mean(axis=1).plot()
monthly_anoms.iloc[:,1].plot()
