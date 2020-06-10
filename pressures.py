# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:28:45 2019

@author: bickels
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmocean
import cartopy.crs as ccrs
import xarray as xa

#%%
def plot_Pl(da,cmap='cmo.dense'):
    plt.pcolormesh(da.lon,da.lat,da.PL.where(da.PL>0)*1e-3,vmin=0,vmax=4e2,transform=proj,cmap=cmap)
    plt.colorbar(label='$P_L\, (kPa)$',shrink=0.8)

#%%
plt.style.use(r'C:/Users/bickels/Documents/Publication/ncomms.mplstyle')
#plt.style.use(r'C:/Users/bickels/Documents/Publication/poster.mplstyle')

cwidth = 3.50394
height = cwidth/1.618

proj = ccrs.PlateCarree()

#%%
arit = xa.open_dataset(r'C:\Users\bickels\Documents\GitHub\geoworm\data\ERA\era_PL_Y_clim.nc')
harm = xa.open_dataset(r'C:\Users\bickels\Documents\GitHub\geoworm\data\ERA\era_PL_Y_harmonic.nc')
medi = xa.open_dataset(r'C:\Users\bickels\Documents\GitHub\geoworm\data\ERA\era_PL_Y_median.nc')
#clim = xa.open_dataset(r'C:\Users\bickels\Documents\GitHub\geoworm\data\ERA\era_PL_Y_cwc.nc').rename({'PL_cwc':'PL'})

#%%
fig = plt.figure()
fig.set_size_inches(2*cwidth, 6*height)
#gs = mpl.gridspec.GridSpec(4,1,wspace=0.05)

ax = plt.subplot(4, 1, 1, projection=proj)#fig.add_subplot(gs[0],projection=proj)
plt.title('Arithmetic average')
ax.coastlines(resolution='50m')
plot_Pl(arit)

ax = plt.subplot(4, 1, 2, projection=proj)#fig.add_subplot(gs[0],projection=proj)
plt.title('Harmonic average')
ax.coastlines(resolution='50m')
plot_Pl(harm)

ax = plt.subplot(4, 1, 3, projection=proj)#fig.add_subplot(gs[0],projection=proj)
plt.title('Median')
ax.coastlines(resolution='50m')
plot_Pl(medi)

ax = plt.subplot(4, 1, 4, projection=proj)#fig.add_subplot(gs[0],projection=proj)
plt.title('Number of estimates < 200 kPa')
ax.coastlines(resolution='50m')
b = (arit.PL<2e5).astype(int)+(harm.PL<2e5).astype(int)+(medi.PL<2e5).astype(int)
plt.pcolormesh(b.lon,b.lat,b.where(b>0),vmin=0.5,vmax=3.5,transform=proj,cmap=mpl.cm.get_cmap('cmo.thermal_r',3))
plt.colorbar(label='Number of times classified as habitat',shrink=0.8)
#plot_Pl(clim)
plt.savefig(r'./out/comparison_averaging.png',dpi=300)