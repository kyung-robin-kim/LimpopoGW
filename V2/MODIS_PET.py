import pandas as pd
import xarray as xr
import rioxarray
import geopandas as gpd
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

def plot_np(array,vmin,vmax,title):
    array = np.where(array==0,np.nan,array)
    fig1, ax1 = plt.subplots(figsize=(20,16))
    image = ax1.imshow(array,cmap = 'RdBu_r',vmin=vmin,vmax=vmax)
    cbar = fig1.colorbar(image,ax=ax1)
    ax1.set_title('{}'.format(title))


#TIFs to NETCDF
######################
#2022-10-23

path = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_PET'
dates = pd.date_range('2002-01','2022-11',  freq='1M') 

files = sorted(glob.glob(path+"/MODMYD/*.tif"))
da = [xr.open_rasterio(file)[0].rename(new_name_or_name_dict='PET_kg_m2').rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
            for file in files]
dataset = xr.concat(da, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'\MODMYD_PET_2002_2022.nc')


#AVERAGE MOD/MYD PET
######################
#2022-10-23

file_paths = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_PET'

#Use Terra only for 2016 - feb & march!!!
years=range(2002,2022)
for year in years:

    mod = sorted(list(glob.glob(file_paths+'/TERRA/*{}*.tif'.format(year))))
    myd = sorted(list(glob.glob(file_paths+'/AQUA/*{}*.tif'.format(year))))

    months = np.arange(0,12)
    for month in months:
        print('{}-'f"{month+1:02d}".format(year))

        month_mod = xr.open_rasterio(mod[month]).rename(new_name_or_name_dict='PET')
        month_myd = xr.open_rasterio(myd[month]).rename(new_name_or_name_dict='PET')
        y = month_mod.y
        x = month_mod.x

        month_modmyd = np.nanmean([month_mod,month_myd],axis=0)[0]
        del month_mod, month_myd

        month_modmyd_masked = np.where(month_modmyd>10000,np.nan,month_modmyd)

        savepath = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_PET\MODMYD'
        array_tif = xr.DataArray(month_modmyd_masked, dims=("y", "x"), coords={"y": y, "x": x}, name="PET")
        array_tif.rio.set_crs("epsg:4326")
        array_tif.rio.set_spatial_dims('x','y',inplace=True)
        os.chdir(savepath)
        array_tif.rio.to_raster('MODMYD_ET_{}_'f"{month+1:02d}.tif".format(year))
        del month_modmyd, month_modmyd_masked



#Partition MOD/MYD PET
######################
#08-12-21

path = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\ET_MODIS\Update'
files = sorted(glob.glob(path+"/*.nc"))

shpname = r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
shapefile = gpd.read_file(shpname)

mod = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 1000,"lon":1000}).PET_500m
myd = xr.open_mfdataset(files[1],parallel=True,chunks={"lat": 1000,"lon":1000}).PET_500m

lon = np.array(mod.lon)
lat = np.array(myd.lat)

#Export CSV of all datetimes in dataset
days = []
months = []
years = []
for week in mod.time:
    days.append(int(week.dt.day))
    months.append(int(week.dt.month))
    years.append(int(week.dt.year))
dates = {'month':months, 'day':days, 'year':years}
df = pd.DataFrame(data=dates)

#df.to_csv(r'C:\Users\robin\Box\SouthernAfrica\DATA\NDVI_MODIS\datetimes.csv')

#Weight each 8-day composite appropriately based on month & year
#11-15-21 -- edited from Prakrut Kansara's ArcPy code
#10-10-22 -- updated for MYD and MOD

#TERRA & AQUA
for year in range(2002,2022):

    mod = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 1000,"lon":1000}).PET_500m.sel(time='{}'.format(year))

    #group xarray data by selecting data from each year
    #TERRA
    jan = xr.zeros_like(mod[0])
    feb = xr.zeros_like(mod[0])
    mar = xr.zeros_like(mod[0])
    apr = xr.zeros_like(mod[0])
    may = xr.zeros_like(mod[0])
    jun = xr.zeros_like(mod[0])
    jul = xr.zeros_like(mod[0])
    aug = xr.zeros_like(mod[0])
    sep = xr.zeros_like(mod[0])
    nov = xr.zeros_like(mod[0])
    oct = xr.zeros_like(mod[0])
    dec = xr.zeros_like(mod[0])

    for k in range(len(mod.time)):
        
        if int(mod[k].time.dt.year)%4 == 0:  #Leap Year Calculations

            if int(mod[k].time.dt.month) == 1:
                if int(mod[k].time.dt.day) == 25:
                    jan = jan + (mod[k] * 0.875)
                    feb = feb + (mod[k] * 0.125)
                else:
                    jan = jan + mod[k] 
            if int(mod[k].time.dt.month) == 2:
                if int(mod[k].time.dt.day) == 26:
                    feb = feb + (mod[k] * 0.500)
                    mar = mar + (mod[k] * 0.500)
                else:
                    feb = feb + mod[k] 
            if int(mod[k].time.dt.month) == 3:
                if int(mod[k].time.dt.day) == 29:
                    mar = mar + (mod[k] * 0.375)
                    apr = apr + (mod[k] * 0.625)
                else:
                    mar = mar + mod[k] 
            if int(mod[k].time.dt.month) == 4:
                if int(mod[k].time.dt.day) == 30:
                    apr = apr + (mod[k] * 0.125)
                    may = may + (mod[k] * 0.875)
                else:
                    apr = apr + mod[k] 
            if int(mod[k].time.dt.month) == 5:
                may = may + mod[k] 
            if int(mod[k].time.dt.month) == 6:
                if int(mod[k].time.dt.day) == 25:
                    jun = jun + (mod[k] * 0.750)
                    jul = jul + (mod[k] * 0.250)
                else:
                    jun = jun + mod[k] 
            if int(mod[k].time.dt.month) == 7:
                if int(mod[k].time.dt.day) == 27:
                    jul = jul + (mod[k] * 0.625)
                    aug = aug + (mod[k] * 0.375)
                else:
                    jul = jul + mod[k] 
            if int(mod[k].time.dt.month) == 8:
                if int(mod[k].time.dt.day) == 28:
                    aug = aug + (mod[k] * 0.500)
                    sep = sep + (mod[k] * 0.500)
                else:
                    aug = aug + mod[k] 
            if int(mod[k].time.dt.month) == 9:
                if int(mod[k].time.dt.day) == 29:
                    sep = sep + (mod[k] * 0.250)
                    oct = oct + (mod[k] * 0.750)
                else:
                    sep = sep + mod[k] 
            if int(mod[k].time.dt.month) == 10:
                if int(mod[k].time.dt.day) == 31:
                    oct = oct + (mod[k] * 0.125)
                    nov = nov + (mod[k] * 0.875)
                else:
                    oct = oct + mod[k] 
            if int(mod[k].time.dt.month) == 11:
                if int(mod[k].time.dt.day) == 24:
                    nov = nov + (mod[k] * 0.875)
                    dec = dec + (mod[k] * 0.125)
                else:
                    nov = nov + mod[k] 
            if int(mod[k].time.dt.month) == 12:
                dec = dec + mod[k] 
            
        else:  #Non-Leap Year Calculations

            if int(mod[k].time.dt.month) == 1:
                if int(mod[k].time.dt.day) == 24:
                    jan = jan + (mod[k] * 0.875)
                    feb = feb + (mod[k] * 0.125)
                else:
                    jan = jan + mod[k] 
            if int(mod[k].time.dt.month) == 2:
                if int(mod[k].time.dt.day) == 26:
                    feb = feb + (mod[k] * 0.375)
                    mar = mar + (mod[k] * 0.625)
                else:
                    feb = feb + mod[k] 
            if int(mod[k].time.dt.month) == 3:
                if int(mod[k].time.dt.day) == 30:
                    mar = mar + (mod[k] * 0.250)
                    apr = apr + (mod[k] * 0.750)
                else:
                    mar = mar + mod[k] 
            if int(mod[k].time.dt.month) == 4:
                apr = apr + mod[k] 
            if int(mod[k].time.dt.month) == 5:
                if int(mod[k].time.dt.day) == 25:
                    may = may + (mod[k] * 0.875)
                    jun = jun + (mod[k] * 0.125)
                else:
                    may = may + mod[k] 
            if int(mod[k].time.dt.month) == 6:
                if int(mod[k].time.dt.day) == 26:
                    jun = jun + (mod[k] * 0.625)
                    jul = jul + (mod[k] * 0.375)
                else:
                    jun = jun + mod[k] 
            if int(mod[k].time.dt.month) == 7:
                if int(mod[k].time.dt.day) == 28:
                    jul = jul + (mod[k] * 0.500)
                    aug = aug + (mod[k] * 0.500)
                else:
                    jul = jul + mod[k] 
            if int(mod[k].time.dt.month) == 8:
                if int(mod[k].time.dt.day) == 29:
                    aug = aug + (mod[k] * 0.375)
                    sep = sep + (mod[k] * 0.625)
                else:
                    aug = aug + mod[k] 
            if int(mod[k].time.dt.month) == 9:
                if int(mod[k].time.dt.day) == 30:
                    sep = sep + (mod[k] * 0.125)
                    oct = oct + (mod[k] * 0.875)
                else:
                    sep = sep + mod[k] 
            if int(mod[k].time.dt.month) == 10:
                oct = oct + mod[k] 
            if int(mod[k].time.dt.month) == 11:
                if int(mod[k].time.dt.day) == 25:
                    nov = nov + (mod[k] * 0.750)
                    dec = dec + (mod[k] * 0.250)
                else:
                    nov = nov + mod[k] 
            if int(mod[k].time.dt.month) == 12:
                dec = dec + mod[k] 

    #Print TIFF for each month (looped for each year) 
    # confirm each indexed array prints correctly & labeling for month & year
    array_tifs = [xr.DataArray(month, dims=("lat", "lon"), coords={"lat":lat, "lon":lon}, name="PET_kg_m2")
                    for month in [jan,feb,mar,apr,may,jun,jul,aug,sep,nov,oct,dec]]
    [array_tif.rio.set_crs("epsg:4326").rio.set_spatial_dims('lon','lat',inplace=True) for array_tif in array_tifs]
    os.chdir(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_PET\TERRA')
    [array_tif.rio.to_raster('T_MODIS_PET_{}_{:02}.tif'.format(year,month)) for array_tif,month in zip(array_tifs,range(1,13))]

    mod.close()

    #AQUA
    myd = xr.open_mfdataset(files[1],parallel=True,chunks={"lat": 1000,"lon":1000}).PET_500m.sel(time='{}'.format(year))

    jan = xr.zeros_like(myd[0])
    feb = xr.zeros_like(myd[0])
    mar = xr.zeros_like(myd[0])
    apr = xr.zeros_like(myd[0])
    may = xr.zeros_like(myd[0])
    jun = xr.zeros_like(myd[0])
    jul = xr.zeros_like(myd[0])
    aug = xr.zeros_like(myd[0])
    sep = xr.zeros_like(myd[0])
    nov = xr.zeros_like(myd[0])
    oct = xr.zeros_like(myd[0])
    dec = xr.zeros_like(myd[0])

    for k in range(len(myd.time)):
        
        if int(myd[k].time.dt.year)%4 == 0:  #Leap Year Calculations

            if int(myd[k].time.dt.month) == 1:
                if int(myd[k].time.dt.day) == 25:
                    jan = jan + (myd[k] * 0.875)
                    feb = feb + (myd[k] * 0.125)
                else:
                    jan = jan + myd[k] 
            if int(myd[k].time.dt.month) == 2:
                if int(myd[k].time.dt.day) == 26:
                    feb = feb + (myd[k] * 0.500)
                    mar = mar + (myd[k] * 0.500)
                else:
                    feb = feb + myd[k] 
            if int(myd[k].time.dt.month) == 3:
                if int(myd[k].time.dt.day) == 29:
                    mar = mar + (myd[k] * 0.375)
                    apr = apr + (myd[k] * 0.625)
                else:
                    mar = mar + myd[k] 
            if int(myd[k].time.dt.month) == 4:
                if int(myd[k].time.dt.day) == 30:
                    apr = apr + (myd[k] * 0.125)
                    may = may + (myd[k] * 0.875)
                else:
                    apr = apr + myd[k] 
            if int(myd[k].time.dt.month) == 5:
                may = may + myd[k] 
            if int(myd[k].time.dt.month) == 6:
                if int(myd[k].time.dt.day) == 25:
                    jun = jun + (myd[k] * 0.750)
                    jul = jul + (myd[k] * 0.250)
                else:
                    jun = jun + myd[k] 
            if int(myd[k].time.dt.month) == 7:
                if int(myd[k].time.dt.day) == 27:
                    jul = jul + (myd[k] * 0.625)
                    aug = aug + (myd[k] * 0.375)
                else:
                    jul = jul + myd[k] 
            if int(myd[k].time.dt.month) == 8:
                if int(myd[k].time.dt.day) == 28:
                    aug = aug + (myd[k] * 0.500)
                    sep = sep + (myd[k] * 0.500)
                else:
                    aug = aug + myd[k] 
            if int(myd[k].time.dt.month) == 9:
                if int(myd[k].time.dt.day) == 29:
                    sep = sep + (myd[k] * 0.250)
                    oct = oct + (myd[k] * 0.750)
                else:
                    sep = sep + myd[k] 
            if int(myd[k].time.dt.month) == 10:
                if int(myd[k].time.dt.day) == 31:
                    oct = oct + (myd[k] * 0.125)
                    nov = nov + (myd[k] * 0.875)
                else:
                    oct = oct + myd[k] 
            if int(myd[k].time.dt.month) == 11:
                if int(myd[k].time.dt.day) == 24:
                    nov = nov + (myd[k] * 0.875)
                    dec = dec + (myd[k] * 0.125)
                else:
                    nov = nov + myd[k] 
            if int(myd[k].time.dt.month) == 12:
                dec = dec + myd[k] 
            
        else:  #Non-Leap Year Calculations

            if int(myd[k].time.dt.month) == 1:
                if int(myd[k].time.dt.day) == 24:
                    jan = jan + (myd[k] * 0.875)
                    feb = feb + (myd[k] * 0.125)
                else:
                    jan = jan + myd[k] 
            if int(myd[k].time.dt.month) == 2:
                if int(myd[k].time.dt.day) == 26:
                    feb = feb + (myd[k] * 0.375)
                    mar = mar + (myd[k] * 0.625)
                else:
                    feb = feb + myd[k] 
            if int(myd[k].time.dt.month) == 3:
                if int(myd[k].time.dt.day) == 30:
                    mar = mar + (myd[k] * 0.250)
                    apr = apr + (myd[k] * 0.750)
                else:
                    mar = mar + myd[k] 
            if int(myd[k].time.dt.month) == 4:
                apr = apr + myd[k] 
            if int(myd[k].time.dt.month) == 5:
                if int(myd[k].time.dt.day) == 25:
                    may = may + (myd[k] * 0.875)
                    jun = jun + (myd[k] * 0.125)
                else:
                    may = may + myd[k] 
            if int(myd[k].time.dt.month) == 6:
                if int(myd[k].time.dt.day) == 26:
                    jun = jun + (myd[k] * 0.625)
                    jul = jul + (myd[k] * 0.375)
                else:
                    jun = jun + myd[k] 
            if int(myd[k].time.dt.month) == 7:
                if int(myd[k].time.dt.day) == 28:
                    jul = jul + (myd[k] * 0.500)
                    aug = aug + (myd[k] * 0.500)
                else:
                    jul = jul + myd[k] 
            if int(myd[k].time.dt.month) == 8:
                if int(myd[k].time.dt.day) == 29:
                    aug = aug + (myd[k] * 0.375)
                    sep = sep + (myd[k] * 0.625)
                else:
                    aug = aug + myd[k] 
            if int(myd[k].time.dt.month) == 9:
                if int(myd[k].time.dt.day) == 30:
                    sep = sep + (myd[k] * 0.125)
                    oct = oct + (myd[k] * 0.875)
                else:
                    sep = sep + myd[k] 
            if int(myd[k].time.dt.month) == 10:
                oct = oct + myd[k] 
            if int(myd[k].time.dt.month) == 11:
                if int(myd[k].time.dt.day) == 25:
                    nov = nov + (myd[k] * 0.750)
                    dec = dec + (myd[k] * 0.250)
                else:
                    nov = nov + myd[k] 
            if int(myd[k].time.dt.month) == 12:
                dec = dec + myd[k] 
        
    #Print TIFF for each month (looped for each year) 
    # confirm each indexed array prints correctly & labeling for month & year
    array_tifs = [xr.DataArray(month, dims=("lat", "lon"), coords={"lat":lat, "lon":lon}, name="PET_kg_m2")
                    for month in [jan,feb,mar,apr,may,jun,jul,aug,sep,nov,oct,dec]]
    [array_tif.rio.set_crs("epsg:4326").rio.set_spatial_dims('lon','lat',inplace=True) for array_tif in array_tifs]
    os.chdir(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\MODIS_PET\AQUA')
    [array_tif.rio.to_raster('A_MODIS_PET_{}_{:02}.tif'.format(year,month)) for array_tif,month in zip(array_tifs,range(1,13))]

    myd.close()