import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy.ma as ma
import xarray as xr
import rioxarray as rxr
from shapely.geometry import mapping, box
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import glob
import rasterio as rio
from rasterio.enums import Resampling
from sklearn.linear_model import LinearRegression
import datetime
import gc
import seaborn as sns
from sklearn.decomposition import PCA

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

#ANOMALIES (monthly) -- need to calculate ET/PET manually 
monthly_anom_files = sorted(glob.glob(r'C:\Users\robin\Desktop\Limpopo_VegetationHydrology\Data\anomalies\monthly\*.csv'))
monthly_anoms = pd.concat([pd.read_csv(file).set_index('time') for file in monthly_anom_files],axis=1)
variables = ['LST','NDVI','GW','RZ','Surface SM','PPT','TCI','VCI','VHI']

# Step 2: Compute the covariance matrix
cov_matrix = np.cov(monthly_anoms.T)
sns.heatmap(cov_matrix, cmap='coolwarm',
            xticklabels=variables, 
            yticklabels=variables)
plt.show()

# Step 3: Find the eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 4: Sort the eigenvectors by decreasing eigenvalues
idx = eigenvalues.argsort()[::-1]
eigenvectors = eigenvectors[:, idx]

num=3
# Step 5: Select the number of principal components to keep (e.g., based on the explained variance)
pca = PCA(n_components=num)
principal_components = pca.fit_transform(monthly_anoms)

# Step 6: Project the data onto the principal components
projected_data = np.dot(monthly_anoms, eigenvectors[:, :num])

