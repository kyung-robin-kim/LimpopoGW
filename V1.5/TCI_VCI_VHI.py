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


#MASK OUT WATER BODIES FIRST
#MODIS
file_paths = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\MODIS_LULC'
files = sorted(glob.glob(file_paths+'\*.nc'))

#LW MASK
lulc = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 100,"lon":100}).LW
land_mask = lulc.rio.write_crs('epsg:4328').rio.reproject_match(lst.rio.write_crs('epsg:4328'),resampling=Resampling.mode)

##############################
#TEMPERATURE CONDITION INDEX (TCI)
path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_LST'
files = sorted(glob.glob(path+'\*.nc'))
lst = xr.open_mfdataset(files[0],parallel=True,chunks={"y": 100,"x":100}).LST_K[7:]
lst = lst.where(land_mask[0]==2)
dates= pd.date_range('2003-02-01','2022-01-01' , freq='1M') #to match with CLSM data period

lst_max = lst.max(dim='time')
lst_min = lst.min(dim='time')

tci = [ (lst_max-month)/(lst_max-lst_min)*100 for month in lst  ]
tci = xr.concat(tci,dim='time')

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(tci.time,tci.mean(dim={'x','y'}),color='red')
ax.set_ylabel('TCI',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('TCI Time Series',weight='bold',fontsize=15)

##############################
#VEGETATION CONDITION INDEX (VCI)

path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_NDVI'
files = sorted(glob.glob(path+'\*.nc'))
ndvi = xr.open_mfdataset(files[0],parallel=True,chunks={"y": 100,"x":100}).NDVI[7:]
ndvi = ndvi.where(land_mask[0]==2)
dates= pd.date_range('2003-02-01','2022-01-01' , freq='1M') #to match with CLSM data period

ndvi_max = ndvi.max(dim='time')
ndvi_min = ndvi.min(dim='time')

vci = [ (month-ndvi_min)/(ndvi_max-ndvi_min)*100 for month in ndvi  ]
vci = xr.concat(vci,dim='time')

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(vci.time,vci.mean(dim={'x','y'}),color='green')
ax.plot(tci.time,tci.mean(dim={'x','y'}),color='red')
ax.set_ylabel('VCI',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('VCI Time Series',weight='bold',fontsize=15)


##############################
#VEGETATION HEALTH INDEX (VHI)

vhi = [ (vci_month*0.5) + ((1-0.5)*tci_month) for vci_month,tci_month in zip(vci,tci)]
vhi = xr.concat(vhi,dim='time')

vhi_a = [ (vci_month*0.2) + ((1-0.2)*tci_month) for vci_month,tci_month in zip(vci,tci)]
vhi_a = xr.concat(vhi_a,dim='time')

vhi_b = [ (vci_month*0.8) + ((1-0.8)*tci_month) for vci_month,tci_month in zip(vci,tci)]
vhi_b = xr.concat(vhi_b,dim='time')


fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(vhi.time,vhi.mean(dim={'x','y'}),color='orange')
ax.plot(vhi.time,vhi_a.mean(dim={'x','y'}),color='red')
ax.plot(vhi.time,vhi_b.mean(dim={'x','y'}),color='blue')
ax.set_ylabel('VHI',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('VHI Time Series',weight='bold',fontsize=15)



################################
#ALL INDICES
fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot()
ax.plot(vhi.time,vhi.mean(dim={'x','y'}),color='orange')
ax.plot(vhi.time,vci.mean(dim={'x','y'}),color='green')
ax.plot(vhi.time,tci.mean(dim={'x','y'}),color='red')
ax.set_ylabel('Index',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Time Series',weight='bold',fontsize=15)
ax.legend()


#SOIL MOISTURE

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
dates= pd.date_range('2015-01-01','2021-08-01' , freq='1M')
dates_sm= pd.date_range('2015-04-01','2021-01-01' , freq='1M')

fig = plt.figure(figsize=(18,6))
ax = fig.add_subplot()
ax2 = ax.twinx()
ax3 = ax.twinx()

ax.grid(color='black', linestyle='-', linewidth=0.05)
ax.plot(dates,monthly_avgs_vci,color='green',label='Vegetation Condition (VCI)')
ax.plot(dates,monthly_avgs_tci[:-5],color='red',label='Temperature Condition (TCI)')
ax.plot(dates,monthly_avgs_vhi,'--',color='orange',label='Vegetation Health (VHI)')
ax2.plot(dates_sm,monthly_avgs_smap[1::],color='blue',label='VWC (cm3/cm3) SMAP')
ax3.plot(dates_sm,monthly_avgs_gldas[1::],'--',color='blue',label='SM (kg/m^2) GLDAS')
ax.legend(loc='upper left')
ax2.legend(loc='lower right')
ax3.legend(loc='upper right')
ax.set_ylabel('Percent',weight='bold',fontsize=12)
ax.set_xlabel('Date',weight='bold',fontsize=12)
ax.set_title('Limpopo Basin Indices (2015-2021)',weight='bold',fontsize=15)
ax.set_ylim(0,100)
