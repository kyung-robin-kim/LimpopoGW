import pandas as pd
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import h5py
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
import netCDF4
import salem
import skimage

#####################
#08-12-21

path = r'C:\Users\robin\Box\Data\Precipitation\IMERG_V6'
files = sorted(glob.glob(path+"/*.hdf5"))

sample_text = '3B-MO.MS.MRG.3IMERG.20000601-S000000-E235959.06.V06B.HDF5'
# year --> sample_text[20:24] 
# month --> sample_text[24:26] 

days_count = {'month': ['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], 
            'month_no': ['01', '02', '03','04','05','06','07','08','09','10','11','12'],
            'non_leap': [31,28,31,30,31,30,31,31,30,31,30,31], 
            'leap': [31,29,31,30,31,30,31,31,30,31,30,31]}
days_df = pd.DataFrame(data=days_count)

#Use the dataframe to multiply no. of days & 24 hours to get accumulated monthly precipitation 

#days_df.loc[days_df['month_no']==file[24:26]]['non_leap'] #number of days in a given month for non-leap year
#days_df.loc[days_df['month_no']==file[24:26]]['leap'] #number of days in a given month for leap year

#####################
#10-09-2022


for year in range(2002,2022):

    gpm_files = sorted(glob.glob(path_gpm+"//*{}????-*.nc4".format(year)))
    chirps_files= sorted(glob.glob(path_chirps+"//*{}*.nc".format(year)))

    p_gpm = xr.open_mfdataset(gpm_files,parallel=True,chunks={"lat": 100,"lon":100}).precipitationCal
    p_chirps = xr.open_mfdataset(chirps_files,parallel=True,chunks={"latitude": 100,"longitude":100}).rename({'latitude':'lat','longitude':'lon'}).precip

    precips_shapefile = [p.rio.set_spatial_dims('lon','lat',inplace=True).rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True) 
    for p in [p_chirps,p_gpm]]
    
    precip_shapefile_sums = [ precip.sum(dim={'lon','lat'}) for precip in precips_shapefile]
    precip_shapefile_means = [precip.mean(dim={'lon','lat'}) for precip in precips_shapefile]

    ti_chirps = precip_shapefile_sums[0].time
    ti_gpm = precip_shapefile_means[1].indexes['time'].to_datetimeindex()

    times = [ti_chirps,ti_gpm]
    data_names = ['CHIRPS', 'GPM IMERG']



#12-02-21
path = r'D:\raw-data\GPM_IMERG'
files = sorted(glob.glob(path+"/*.NC4"))
shpname = r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile\limpopo.shp'
shapefile = gpd.read_file(shpname)

date_strings = []
for file in files:
    date_strings.append(str(file[-32:-26]))
unique_months = np.unique(date_strings)

for date in unique_months:
    monthly_sum = []
    for file in files:
        if file[-32:-26] == date:
            da = xr.open_mfdataset(file)
            da = da.salem.roi(shape=shapefile,all_touched=True).rename({'lat': 'y','lon': 'x'}).transpose('time','y','x','nv').precipitationCal      

            monthly_sum.append(da)

    dataset = xr.concat(monthly_sum,dim='time')
    p_accumulation = dataset.sum(dim='time')
    print(date)
    p_accumulation.rio.to_raster(r'C:\Users\robin\Box\SouthernAfrica\DATA\IMERG_{}.tif'.format(date))

    del dataset
    del da
    del p_accumulation
    del monthly_sum


#Compare calculated accumulations vs. tif accumulations (from another NASA source)
path = r'C:\Users\robin\Box\SouthernAfrica\DATA'
files = sorted(glob.glob(path+"/*.tif"))
dates = pd.date_range('2020-01-31','2020-12-31',  freq='1M') 
arrays = []
for file in files:
    da = xr.open_rasterio(file)[0].rename(new_name_or_name_dict='P')
    arrays.append(da)

dataset = xr.concat(arrays, dim=dates).rename({'concat_dim':'time'})
dataset.to_netcdf(path+'\GPM_P_2020_120321.nc')

p = xr.open_mfdataset(r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GPM_IMERG/*.nc')
p_1 = p.rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
p_1_df = p.P.mean(dim=['y','x']).to_dataframe('Values')[-9:]

p = xr.open_mfdataset(r'C:\Users\robin\Box\SouthernAfrica\DATA\*.nc')
p_2 = p.rio.write_crs('WGS84').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=False)
p_2_df = p_2.P.mean(dim=['y','x']).to_dataframe('Values')[-12:-3]


#################################################
#MONTHLY ACCUMULATION TIFS (10/2002 - 9/2020)

shape_path=r'C:\Users\robin\Box\SouthernAfrica\DATA\shapefile'
p_shape=Path(shape_path)
shapefiles = sorted(list(p_shape.glob('*.shp')))
shapefile = gpd.read_file(shapefiles[0])

fileslist = [name for name in os.listdir(r'D:\raw-data\GPM_IMERG\MONTHLY_ACCUMULATION') if name.endswith('.tif')]
savepath=r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\GPM_IMERG'

for i in range(0,len(fileslist)):
    os.chdir(r'D:\raw-data\GPM_IMERG\MONTHLY_ACCUMULATION')
    year = fileslist[i][24:28]
    month = fileslist[i][28:30]

    array = rxr.open_rasterio(fileslist[i],masked=True).squeeze()
    clipped = array.rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)
    clipped.rio.to_raster(savepath+'\IMERG_{}_{}.tif'.format(year,month))


#Example Code for HDF5
test = h5py.File(files[0],'r')

# View the available groups in the file and the variables in the 'Grid' group:
groups = [ x for x in test.keys() ]
print(groups)
gridMembers = [ x for x in f['Grid'] ]
print(gridMembers)

# Read the precipitation, latitude, and longitude data:

precip = f['Grid/precipitation'][0][:][:]
precip = np.transpose(precip)
theLats = f['Grid/lat'][:]
theLons = f['Grid/lon'][:]
x, y = np.float32(np.meshgrid(theLons, theLats))

# Set the figure size, projection, and extent
fig = plt.figure(figsize=(21,7))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-180,180,-60,60], crs=ccrs.PlateCarree())  

# Add coastlines and formatted gridlines
ax.coastlines(resolution="110m",linewidth=1)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='black', linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = True
gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
gl.ylocator = mticker.FixedLocator([-60, -50, -25, 0, 25, 50, 60])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size':16, 'color':'black'}
gl.ylabel_style = {'size':16, 'color':'black'}

# Set contour levels and draw the plot
clevs = np.arange(0,600,5)
plt.contourf(x, y, precip, clevs, cmap=plt.cm.rainbow)
plt.title('GPM IMERG Monthly Mean Rain Rate for June 2000', size=24)
cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
cb.set_label('mm',size=20)
cb.ax.tick_params(labelsize=16)
