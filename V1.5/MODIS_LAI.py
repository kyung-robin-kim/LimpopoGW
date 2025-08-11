import pandas as pd
import xarray as xr
import rioxarray
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
import skimage
import rasterstats as rs
from rasterio.enums import Resampling
from datetime import datetime
import geopandas as gpd

import rasterio.mask
from rasterio.plot import show
from rasterio.transform import Affine
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import box
from shapely.geometry import Polygon, Point
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

#Leaf Area Index & FPAR
path = r'D:\raw-data\MCD15'
files = sorted(glob.glob(path+'\*.nc'))
lai = xr.open_mfdataset(files[0],parallel=True,chunks={"y": 100,"x":100})
