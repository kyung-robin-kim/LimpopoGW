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
import rasterstats as rs

def read_file(file):
    with rio.open(file) as src:
        return(src.read())
        
def plot_np(array,vmin,vmax,title):
    array = np.where(array==0,np.nan,array)
    fig1, ax1 = plt.subplots(figsize=(20,16))
    image = ax1.imshow(array,cmap = 'RdBu_r',vmin=vmin,vmax=vmax)
    cbar = fig1.colorbar(image,ax=ax1)
    ax1.set_title('{}'.format(title))


file_paths = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\MODIS_LULC'
files = sorted(glob.glob(file_paths+'\*.nc'))

#LC Type 1
lulc = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 100,"lon":100})

lulc_files = sorted(glob.glob(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\MODIS_LULC\LC_Type1_tifs\*.tif'))
shpname = r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
shapefile = gpd.read_file(shpname).to_crs('EPSG:6933')
lulcs = [xr.open_rasterio(file)[0].rio.write_crs('epsg:4326').rio.reproject('EPSG:6933') for file in lulc_files]


def calculate_lulc(lulc):
    EG_NL_F = lulc.where(lulc==1) 
    EG_BL_F = lulc.where(lulc==2) 
    DC_NL_F = lulc.where(lulc==3) 
    DC_BL_F = lulc.where(lulc==4) 
    MIXED_F = lulc.where(lulc==5)
    SHRUB_CLOSED = lulc.where(lulc==6) 
    SHRUB_OPEN = lulc.where(lulc==7) 
    WOOD_SAV = lulc.where(lulc==8) 
    SAVANNA = lulc.where(lulc==9) 
    GRASS = lulc.where(lulc==10) 
    WETLAND = lulc.where(lulc==11) 
    CROPLAND_A = lulc.where(lulc==12) 
    URBAN = lulc.where(lulc==13) 
    CROPLAND_B = lulc.where(lulc==14) 
    ICE = lulc.where(lulc==15) 
    BARREN = lulc.where(lulc==16) 
    WATER = lulc.where(lulc==17) 

    return EG_NL_F, EG_BL_F, DC_NL_F, DC_BL_F, MIXED_F,SHRUB_CLOSED,SHRUB_OPEN,WOOD_SAV,SAVANNA,GRASS,WETLAND,CROPLAND_A,URBAN,CROPLAND_B,ICE,BARREN,WATER

EG_NL_F, EG_BL_F, DC_NL_F, DC_BL_F, MIXED_F, SHRUB_CLOSED, SHRUB_OPEN,WOOD_SAV,SAVANNA,GRASS,WETLAND,CROPLAND_A,URBAN,CROPLAND_B,ICE,BARREN,WATER = map(list,zip(*[calculate_lulc(array) for array in lulcs]))

total_area = (shapefile.area)/10**6 #in km^2
all_pixel_sum = [rs.zonal_stats(shapefile,np.array(array),affine = lulcs[0].rio.transform(),stats='count sum',all_touched=True) for array in lulcs]
all_pixel_cumsum = [all_pixel_sum[year][0]['sum'] for year in range(0,len(all_pixel_sum))]

area2pixel_ratio = total_area/all_pixel_cumsum[0]

EG_NL_F = [rs.zonal_stats(shapefile,np.array(array),affine = lulcs[0].rio.transform(),stats='count sum',all_touched=True) for array in EG_NL_F]
EG_BL_F = [rs.zonal_stats(shapefile,np.array(array),affine = lulcs[0].rio.transform(),stats='count sum',all_touched=True) for array in EG_BL_F]
DC_NL_F = [rs.zonal_stats(shapefile,np.array(array),affine = lulcs[0].rio.transform(),stats='count sum',all_touched=True) for array in DC_NL_F]
DC_BL_F = [rs.zonal_stats(shapefile,np.array(array),affine = lulcs[0].rio.transform(),stats='count sum',all_touched=True) for array in DC_BL_F]
MIXED_F = [rs.zonal_stats(shapefile,np.array(array),affine = lulcs[0].rio.transform(),stats='count sum',all_touched=True) for array in MIXED_F]
SHRUB_CLOSED = [rs.zonal_stats(shapefile,np.array(array),affine = lulcs[0].rio.transform(),stats='count sum',all_touched=True) for array in SHRUB_CLOSED]
SHRUB_OPEN = [rs.zonal_stats(shapefile,np.array(array),affine = lulcs[0].rio.transform(),stats='count sum',all_touched=True) for array in SHRUB_OPEN]
WOOD_SAV = [rs.zonal_stats(shapefile,np.array(array),affine = lulcs[0].rio.transform(),stats='count sum',all_touched=True) for array in WOOD_SAV]
SAVANNA = [rs.zonal_stats(shapefile,np.array(array),affine = lulcs[0].rio.transform(),stats='count sum',all_touched=True) for array in SAVANNA]
GRASS = [rs.zonal_stats(shapefile,np.array(array),affine = lulcs[0].rio.transform(),stats='count sum',all_touched=True) for array in GRASS]
WETLAND = [rs.zonal_stats(shapefile,np.array(array),affine = lulcs[0].rio.transform(),stats='count sum',all_touched=True) for array in WETLAND]
CROPLAND_A = [rs.zonal_stats(shapefile,np.array(array),affine = lulcs[0].rio.transform(),stats='count sum',all_touched=True) for array in CROPLAND_A]
URBAN = [rs.zonal_stats(shapefile,np.array(array),affine = lulcs[0].rio.transform(),stats='count sum',all_touched=True) for array in URBAN]
CROPLAND_B = [rs.zonal_stats(shapefile,np.array(array),affine = lulcs[0].rio.transform(),stats='count sum',all_touched=True) for array in CROPLAND_B]
ICE = [rs.zonal_stats(shapefile,np.array(array),affine = lulcs[0].rio.transform(),stats='count sum',all_touched=True) for array in ICE]
BARREN = [rs.zonal_stats(shapefile,np.array(array),affine = lulcs[0].rio.transform(),stats='count sum',all_touched=True) for array in BARREN]
WATER = [rs.zonal_stats(shapefile,np.array(array),affine = lulcs[0].rio.transform(),stats='count sum',all_touched=True) for array in WATER]

lulc_types = [EG_NL_F, EG_BL_F, DC_NL_F, DC_BL_F, MIXED_F, SHRUB_CLOSED, SHRUB_OPEN,WOOD_SAV,SAVANNA,GRASS,WETLAND,CROPLAND_A,URBAN,CROPLAND_B,ICE,BARREN,WATER]
lulc_names = ['EG_NL_F', 'EG_BL_F', 'DC_NL_F', 'DC_BL_F', 'MIXED_F', 'SHRUB_CLOSED', 'SHRUB_OPEN','WOOD_SAV','SAVANNA','GRASS','WETLAND','CROPLAND_A','URBAN','CROPLAND_B','ICE','BARREN','WATER']
cumcounts = [[cover[i][0]['count'] for i in range(0,len(all_pixel_sum))] for cover in lulc_types]
pixel_area_df = pd.DataFrame(cumcounts).T.rename(columns={0:'EG_NL_F', 1:'EG_BL_F', 2:'DC_NL_F', 3:'DC_BL_F', 4:'MIXED_F', 5:'SHRUB_CLOSED', 6:'SHRUB_OPEN',7:'WOOD_SAV',8:'SAVANNA',9:'GRASS',10:'WETLAND',11:'CROPLAND_A',12:'URBAN',13:'CROPLAND_B',14:'ICE',15:'BARREN',16:'WATER'})
pixel_area_df.to_csv(r'D:\raw-data\MODIS_LC\Limpopo\LC_TYPE1.csv')



def calculate_lulc(lulc):
    forests = lulc.where( (lulc==1) | (lulc==2) | (lulc==3) | (lulc==4) | (lulc==5)) 
    shrubs_sav = lulc.where( (lulc==6) | (lulc==7) | (lulc==8) | (lulc==9) ) 
    grass = lulc.where(lulc==10) 
    crop = lulc.where((lulc==12) | (lulc==14)) 
    urban = lulc.where(lulc==13) 
    water = lulc.where(lulc==17) 

    return forests, shrubs_sav, grass, crop, urban,water

forests, shrubs_sav, grass, crop, urban,water = map(list,zip(*[calculate_lulc(array) for array in lulcs]))

for i in range(0,len(lulc.LC_Type1)):
   #lulc.LC_Type1[i].plot.hist()
    urban[i].rio.to_raster(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\MODIS_LULC\LC_Type1_tifs\URBAN\{}.tif'.format(i+2001))


#SPATIAL PLOTS
maindir = r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\MODIS_LULC\LC_Type1_tifs'

arrays = [read_file(i)[0] for i in sorted(glob.glob(maindir+r'/CROP/*.tif'))]
titles = [i for i in range(2001,2022)]

lat =lulc.lat
lon = lulc.lon
count_ticks = 1000

plt.rcParams.update({'font.size': 100})
plt.rcParams["font.family"] = "Times New Roman"
for i,title in zip(arrays,titles):
    fig,ax = plt.subplots(figsize=(100, 75.5))
    plt.grid(color='black', linestyle='-', linewidth=0.05)
    image=ax.imshow(i)
    ax.set_xticks([i for i in range(0,len(lon),count_ticks)]) 
    ax.set_xticklabels(np.int_(np.around(lon[0::count_ticks])))
    ax.set_yticks([i for i in range(0,len(lat),count_ticks)])
    ax.set_yticklabels(((np.int_(np.around(lat[0::count_ticks])))))

    degree_sign = u"\N{DEGREE SIGN}"
    ax.set_xlabel("Longitude ({0} E)".format(degree_sign))
    ax.set_ylabel("Latitude ({0} N)".format(degree_sign))

    cbar = fig.colorbar(image,ax=ax)
    #cbar.ax.set_ylabel("dTWS",rotation=270, labelpad=15)
    ax.set_title('{}'.format(title), pad=10)
    plt.savefig(maindir+'/CROP/{}.png'.format(title))



lulc_dynamics = xr.open_mfdataset(files[1],parallel=True,chunks={"lat": 100,"lon":100})


#LC Type 3 - vegetation
lulc = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 100,"lon":100}).LC_Type3


#LC Type 4 - LAI
lulc = xr.open_mfdataset(files[0],parallel=True,chunks={"lat": 100,"lon":100}).LC_Type4