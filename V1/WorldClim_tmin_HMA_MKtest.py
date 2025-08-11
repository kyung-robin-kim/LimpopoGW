#Ryan Haagenson 07-14-21
#edited by Robin Kim 07-15-21

import rasterio as rio
from rasterio.plot import show
import numpy as np
import matplotlib.pyplot as plt
import pymannkendall as pmk
import xarray as xr

###
# Setting the lat and lon arrays for plotting
initial_dataset = rio.open(r"C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\AIRS\SKIN\LIMPOPO\2002_09.tif")
left = initial_dataset.bounds[0]
bottom = initial_dataset.bounds[1]
right = initial_dataset.bounds[2]
top = initial_dataset.bounds[3]
nx = initial_dataset.width
nz = initial_dataset.height
lat = np.linspace(bottom,top,nz)
lon = np.linspace(left,right,nx)

###
# Setting up time range
months = np.arange(12)+1
years = np.arange(2003,2021)
nt = len(months)*len(years)

###
# Read in the data
data = np.zeros((nz,nx,nt))
it = 0

file = r'C:\Users\robin\Box\SouthernAfrica\DATA\PROCESSED\AIRS\SKIN\LIMPOPO\LIMPOPO_SKINTEMP_2002_2020_120321.nc'

for iy in years:
    for im in months:
        data[:,:,it] = np.array(xr.open_mfdataset(file).sel(time='{}-{:02d}'.format(iy,im)).SKIN[0])

        it += 1

###
# Perform MK test
trnd = np.zeros(np.shape(data)[0:2])
pvals = np.ones(np.shape(data)[0:2])

for ix in range(nx):
	for iz in range(nz):
		if data[iz,ix,1] - data[iz,ix,0] == 0:
			pass
		else:
			result = pmk.seasonal_test(data[iz,ix,:])

			if result.trend == "no trend":
				trnd[iz,ix] = 0.0
			else:
				trnd[iz,ix] = 1.0

			pvals[iz,ix] = result.p

# Plot trends
fig = plt.figure()
ax = fig.add_subplot(111)
plt.pcolormesh(lon,lat,np.flipud(trnd),vmin=0.0,vmax=1.0,cmap='Greys')
ax.set_aspect(aspect=1.0)
c = plt.colorbar(shrink=0.6)
plt.xlabel('Longitidue')
plt.ylabel('Latitude')
plt.show()
fig_name = 'Trend_map_1961-99.png'
plt.savefig(fig_name)

# Plot p values
fig = plt.figure()
ax = fig.add_subplot(111)
plt.pcolormesh(lon,lat,np.flipud(pvals),vmin=0.0,vmax=1.0,cmap='bone')
ax.set_aspect(aspect=1.0)
c = plt.colorbar(shrink=0.6)
c.ax.set_title("p-value")
plt.xlabel('Longitidue')
plt.ylabel('Latitude')
plt.show()
fig_name = 'pvals_map_1961-99.png'
plt.savefig(fig_name)

# Map test plot
fig = plt.figure()
ax = fig.add_subplot(111)
plt.pcolormesh(lon,lat,np.flipud(data[:,:,0]),cmap='coolwarm')
ax.set_aspect(aspect=1.0)
c = plt.colorbar(shrink=0.6)
c.ax.set_title("$^\circ$ C")
plt.xlabel('Longitidue')
plt.ylabel('Latitude')
plt.show()
fig_name = 'Temp_map_Jan_1961.png'
plt.savefig(fig_name)

# Time-series test plot
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(np.shape(data)[0]-1):
	plt.plot(np.arange(len(data[i,100,:])),data[i,100,:])
plt.xlabel('Month since 1960')
plt.ylabel('Air Temp')
plt.show()
fig_name = 'Time_series_along_transect.png'
plt.savefig(fig_name)
