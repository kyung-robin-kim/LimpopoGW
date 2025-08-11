import os
from pathlib import Path
from glob import glob
from collections import deque
import xarray as xr
import rioxarray
import rasterio as rio
#from netCDF4 import Dataset #isn't working for AIRS hdf...
#from pyhdf.SD import SD,SDC
import numpy as np
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd
from fiona.crs import from_epsg
from shapely.geometry import box
from rasterio.mask import mask
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import pandas as pd

def read_file(file):
    with rio.open(file) as src:
        return(src.read())

import earthpy as et
import earthpy.plot as ep
import earthpy.spatial as es
import earthpy.mask as em
import re
import glob
import datetime

#NearSurface Air Temp
#Relative Humidity at Surface? or higher elevation?
#Relative humidity at equilibrium or in liquid phasee?

def relhum_D(FILE_NAME,shapefile):
    with rio.open(FILE_NAME) as dataset:
        for name in dataset.subdatasets:
            if re.search(("RelHumSurf_TqJ_D$"), name):
                with rio.open(name) as subdataset:
                    #print("AIRS:", subdataset)
                    airs_meta = subdataset.profile
                    #print(subdataset.meta, subdataset.profile)
                    #print(subdataset.crs)
                    shapefile_sin = shapefile.to_crs(subdataset.crs)
                    #Crop based on HMA shapefile and same CRS as AIRS data
                    crop_band, crop_meta = es.crop_image(subdataset, shapefile_sin)
                    #print(crop_meta['crs']['Latitude'])
                    # Append the cropped band as 2D array to the list
    return crop_band,crop_meta

def relhum_A(FILE_NAME,shapefile):
    FILE_NAME = airs[0]
    with rio.open(FILE_NAME) as dataset:
        for name in dataset.subdatasets:
            if re.search(("RelHumSurf_TqJ_A$"), name):
                with rio.open(name) as subdataset:
                    #print("AIRS:", subdataset)
                    #airs_meta = subdataset.profile
                    #print(subdataset.meta, subdataset.profile)
                    #print(subdataset.crs)
                    shapefile_sin = shapefile.to_crs(subdataset.crs)
                    #Crop based on HMA shapefile and same CRS as AIRS data
                    crop_band, crop_meta = es.crop_image(subdataset, shapefile_sin)
                    #print(crop_meta['crs']['Latitude'])
                    # Append the cropped band as 2D array to the list
    return crop_band,crop_meta

savepath = r'F:\processed-data\AIRS\RelativeHumidity\Limpopo\MONTHLY'

p=Path(r'F:\raw-data\AIRS\V7_MONTHLY')
files = sorted(p.glob('*.hdf'))
dates = pd.date_range('2002-09-30','2022-09-30',  freq='1M') 

#Shapefiles
limpopo = r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
shp = gpd.read_file(limpopo)


#For corrupted files -- using AIRS+AMSU for 2010-01 & AIRS v6 2020 09 - 2020 12
#in F: drive (?) 2010-01 couldn't be replaced
def save_airs_tif(file_name,file):
    with rio.open(file_name,'w',**airs_meta) as da:
        da.write_band(1,file.astype(rio.int16))

years=range(2002,2023)
for year in years:
    airs = sorted(glob.glob(r'F:\raw-data\AIRS\V7_MONTHLY\*.{}.*.hdf'.format(year)))
    file_dates = [datetime.datetime.strptime(file[33:40],'%Y.%m') for file in airs]
    file_datestrings = [datetime.datetime.strftime(date,'%Y-%m') for date in file_dates]
    airs_meta = relhum_D(airs[0],shp)[1]

    airs_AM = [relhum_D(file,shp)[0][0] for file in airs]
    airs_PM = [relhum_A(file,shp)[0][0] for file in airs]

    [save_airs_tif(savepath+'\{}_Desc.tif'.format(month), air_am) for month,air_am in zip(file_datestrings,airs_AM)]
    [save_airs_tif(savepath+'\{}_Asen.tif'.format(month), air_pm) for month,air_pm in zip(file_datestrings,airs_PM)]
    
#############################################################
savepath = r'D:\processed-data\averages\2003-2015'  

def read_file(file):
    with rio.open(file) as src:
        return(src.read())

#Average
path=r'D:\processed-data\AIRS+AMSU'
p=Path(path)
files = sorted(list(p.glob('*.tif')))[0:13]
os.chdir(path)

#AVG
airs_avg = np.nanmean([read_file(i) for i in files], axis=0)
with rio.open(files[0]) as src:
    meta = src.meta
os.chdir(savepath)
with rio.open('AIRS_AMSU_averaged_2003_2015.tif', 'w', **meta) as dst:
    dst.write(airs_avg[0], 1)


#PNG
path = 'D:\\processed-data\\AIRS+AMSU'
file_path = path+'\\*.tif'
os.chdir(path)

savepath='D:\processed-data\AIRS+AMSU\GIF'

airsamsu = list(glob(file_path))[1:14]



years=range(2003,2017)

    for file,year in zip(airsamsu,years):
        os.chdir(path)
        fig, ax = plt.subplots()
        item = read_file(file)
        image = ax.imshow(item, cmap='RdBu_r',vmin=-9,vmax=9)
        
        pixel = 1
        dimensions = [63.5, 22.5, 107.5, 46.5]
        Width = 44
        Height = 24

        x = [0,44,4]
        y = [0,24,3]

        plt.grid(color='black', linestyle='-', linewidth=0.05)
        ax.set_xticks([i for i in range(x[0],x[1],x[2])]) 
        ax.set_xticklabels([round(i,1) for i in np.arange((dimensions[0]+(pixel*x[0])),(dimensions[0]+(pixel*x[1])),(pixel*x[2]))]) 
        ax.set_yticks([i for i in range(y[0],y[1],y[2])])
        ax.set_yticklabels([round(i,1) for i in reversed(np.arange((dimensions[1]+(pixel*y[0])),(dimensions[1]+(pixel*y[1])),(pixel*y[2])))])
        degree_sign = u"\N{DEGREE SIGN}"
        ax.set_xlabel("Longitude ({0} E)".format(degree_sign))
        ax.set_ylabel("Latitude ({0} N)".format(degree_sign))
        cbar = fig.colorbar(image,ax=ax)
        cbar.ax.set_ylabel("Degrees Celsius",rotation=270,labelpad=15)
        ax.set_title('AIRS+AMSU MAAT {}'.format(year), pad=10)
        os.chdir(savepath)
        plt.savefig('{}_MAAT_airs+amsu.png'.format(year),bbox_inches='tight',dpi = 300, transparent=False)

#GIF
from PIL import Image

os.chdir(savepath)
graphics_folder = os.path.join('*.png')

images=(glob(graphics_folder))
frames=[]
for i in images:
    new_frame=Image.open(i)
    frames.append(new_frame)

frames[0].save('AIRS+AMSU_MAAT.gif', format='GIF',append_images=frames[1:],save_all=True,duration=1000,loop=0)




#LIS PNG
"
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
import gdal
from rasterstats import zonal_stats
import csv

import os
from pathlib import Path
from glob import glob
"
#Save PNG loop (updated)


path = 'C:/Users/robin/Box/HMA_robin/02-data/raster/annual_tair/lis/clipped/years'
file_path = path+'/*.tif'
os.chdir(path)

lis = list(glob(file_path))[0:13]
count = range(2003,2016)

    for file,year in zip(lis,count):
        os.chdir(path)
        fig, ax = plt.subplots()
        item = read_file(file)[::-1]
        image = ax.imshow(item, cmap='RdBu_r',vmin=-9,vmax=9)

        pixel = 0.25
        dimensions = [66.750, 20.750, 101.0, 41.000]
        Width=137
        Height=81
        
        x = [0,137,15]
        y = [0,81,10]

        plt.grid(color='black', linestyle='-', linewidth=0.05)
        ax.set_xticks([i for i in range(x[0],x[1],x[2])]) 
        ax.set_xticklabels([round(i,1) for i in np.arange((dimensions[0]+(pixel*x[0])),(dimensions[0]+(pixel*x[1])),(pixel*x[2]))]) 
        ax.set_yticks([i for i in range(y[0],y[1],y[2])])
        ax.set_yticklabels([round(i,1) for i in reversed(np.arange((dimensions[1]+(pixel*y[0])),(dimensions[1]+(pixel*y[1])),(pixel*y[2])))])
        degree_sign = u"\N{DEGREE SIGN}"
        ax.set_xlabel("Longitude ({0} E)".format(degree_sign))
        ax.set_ylabel("Latitude ({0} N)".format(degree_sign))
        cbar = fig.colorbar(image,ax=ax)
        cbar.ax.set_ylabel("Degrees Celsius",rotation=270,labelpad=15)
        ax.set_title('HMA-LIS MAAT {}'.format(year), pad=10)
        os.chdir('C:/Users/robin/Box/HMA_robin/03-output-graphics/')
        plt.savefig('{}_MAAT_lis.png'.format(year),bbox_inches='tight',dpi = 300, transparent=False)

#############################
#Average AIRS+AMSU data by 3-year windows
#Last Used: 04-20-21

def read_file(file):
    with rio.open(file) as src:
        return(src.read(1))

path = r'D:\processed-data\averages\MAAT_2003-2016\AIRSAMSU_windows\annual'
file_path = path+'/*.tif'
os.chdir(path)

files = sorted(list(glob(file_path)))

start = 2003
end=2005
span=3

for i in range(0,len(files),3):
    print(i)
    avg_window = files[i:i+3]
    print(avg_window)
    arrays=[]
    arrays = [read_file(x) for x in avg_window]
    avg_array = np.nanmean(arrays,axis=0)
    
    # Get metadata from one of the input files
    with rio.open(files[0]) as src:
        meta = src.meta

    meta.update(dtype=rio.float32)

    # Write output file
    with rio.open('AIRS+AMSU_average_clipped_{}_{}.tif'.format(start,end), 'w', **meta) as dst:
        dst.write(avg_array.astype(rio.float32), 1)
        


#############################
#Average LIS data by 4-year windows
path = 'C:/Users/robin/Box/HMA_robin/02-data/raster/annual_tair/lis/clipped'
file_path = path+'/*.tif'
os.chdir(path)

files = list(glob(file_path))[0:16]

def read_file(file):
    with rasterio.open(file) as src:
        return(src.read(1))

start = 2003
end=2007
span=4

for i in range(0,len(files),4):
    avg_window = files[i:i+5]
    arrays=[]
    arrays = [read_file(x) for x in avg_window]
    avg_array = np.nanmean(arrays,axis=0)
    
    # Get metadata from one of the input files
    with rasterio.open(files[0]) as src:
        meta = src.meta

    meta.update(dtype=rasterio.float32)

    # Write output file
    with rasterio.open('LIS_average_clipped_{}_{}.tif'.format(start,end), 'w', **meta) as dst:
        dst.write(avg_array.astype(rasterio.float32), 1)
        
    start = start+span 
    end=end+span



#Histograms; 4-20-21
import seaborn as sns

main = 'D:/processed-data/averages/MAAT_2003-2016'
pathname = main+'/AIRSAMSU_windows'
title = 'AIRS+AMSU Three-Year Averages'
palette = 'YlOrBr_r'
savepath = pathname
p=Path(pathname)
files = sorted(list(p.glob('*tif')))
os.chdir(pathname)
    
    raster=gdal.Open(files[0].stem+'.tif')
    raster5 = raster.ReadAsArray()
    raster5=np.where(raster5==-999,np.nan,raster5)
    raster=gdal.Open(files[1].stem+'.tif')
    raster6 = raster.ReadAsArray()
    raster6=np.where(raster6==-999,np.nan,raster6)
    raster=gdal.Open(files[2].stem+'.tif')
    raster7 = raster.ReadAsArray()
    raster7=np.where(raster7==-999,np.nan,raster7)
    raster=gdal.Open(files[3].stem+'.tif')
    raster8 = raster.ReadAsArray()
    raster8=np.where(raster8==-999,np.nan,raster8)
    raster=gdal.Open(files[4].stem+'.tif')
    raster9 = raster.ReadAsArray()
    raster9=np.where(raster9==-999,np.nan,raster9)
    
    raster5.ravel()
    raster6.ravel()
    raster7.ravel()
    raster8.ravel()
    raster9.ravel()

    label5='2003-2005'
    label6='2006-2008'
    label7='2009-2011'
    label8='2012-2014'
    label9='2015'

    fig, ax = plt.subplots(figsize=(12,8))
    plt.grid()
    sns.distplot(raster5, bins=range(int(np.floor(np.nanmin(raster5))),int(np.ceil(np.nanmax(raster5))), 1), ax=ax, kde=False,norm_hist=True,color=sns.color_palette('{}'.format(palette))[0])
    sns.distplot(raster6, bins=range(int(np.floor(np.nanmin(raster6))),int(np.ceil(np.nanmax(raster6))), 1), ax=ax, kde=False,norm_hist=True,color=sns.color_palette('{}'.format(palette))[1])
    sns.distplot(raster7, bins=range(int(np.floor(np.nanmin(raster7))),int(np.ceil(np.nanmax(raster7))), 1), ax=ax, kde=False,norm_hist=True,color=sns.color_palette('{}'.format(palette))[2])
    sns.distplot(raster8, bins=range(int(np.floor(np.nanmin(raster8))),int(np.ceil(np.nanmax(raster8))), 1), ax=ax, kde=False,norm_hist=True,color=sns.color_palette('{}'.format(palette))[3])
    sns.distplot(raster9, bins=range(int(np.floor(np.nanmin(raster9))),int(np.ceil(np.nanmax(raster9))), 1), ax=ax, kde=False,norm_hist=True,color=sns.color_palette('{}'.format(palette))[4])
    fig.legend(labels=[label5,label6,label7,label8,label9], loc='upper right',borderaxespad=9)
    ax.set_title('{}'.format(title),weight='bold')
    ax.set_xlabel('Degrees Celsius', weight='bold')
    ax.set_ylabel('Density',weight='bold')
    os.chdir(savepath)
    plt.savefig('{}'.format(title)+'.png',bbox_inches='tight',dpi = 400)

#Raster Statistics for Clipped HMA region

import csv
def read_file(file):
    with rio.open(file) as src:
        return(src.read())

main=r'D:\processed-data\averages\MAAT_2003-2016\AIRSAMSU_windows\annual'
p=Path(main)
files = sorted(list(p.glob('*tif')))[0:13] #2003 -- 2015

years=['years']
stds = ['stdevs']
variances=['variances']
means=['means']
medians=['medians']

for file in files:
    dataset = read_file(file)
    dataset = np.where(dataset==-999, np.nan,dataset)
    years.append(file.stem[-9:-5])
    stds.append(np.nanstd(dataset))
    variances.append((np.nanstd(dataset))**2)
    means.append(np.nanmean(dataset))
    medians.append(np.nanmedian(dataset))

os.chdir(main)

with open('AIRSAMSU_MAAT_Averaged_Clipped.csv',"a", newline='') as fp: #Change Title
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(years)
    wr.writerow(stds)
    wr.writerow(variances)
    wr.writerow(means)
    wr.writerow(medians)

#Check stationarity
from statsmodels.tsa.stattools import adfuller

variables = [stds[1::], variances[1::], means[1::], medians[1::]]

for variable in variables:
    result = adfuller(variable)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    if result[0] > result[4]["5%"]:
        print ("Failed to Reject Ho - Time Series is Non-Stationary\n")
    else:
        print ("Reject Ho - Time Series is Stationary\n")