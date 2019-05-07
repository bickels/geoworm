# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 ‏‎12:39:10 2019

@author: bickels
"""

import xarray as xa
import numpy as np
import cartopy.crs as ccrs
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

#%%
cdict = mpl.cm.get_cmap('Spectral')._segmentdata.copy()
cdict['red'][5] = (0.5,0.9,0.9) 
cdict['red'][6] = (0.6,0.7,0.7) 
cdict['red'][8] = (0.9,0.2,0.2)
cdict['red'][9] = (1.0,0.2,0.2)
cdict['green'][4] = (0.4,0.7,0.7)
cdict['green'][5] = (0.5,0.75,0.75)
cdict['green'][8] = (0.9,0.8,0.8)
cdict['green'][9] = (1.0,0.9,0.9)
cdict['blue'][3] = (0.3,0.3,0.3)
cdict['blue'][4] = (0.4,0.35,0.35)
cdict['blue'][5] = (0.5,0.4,0.4)
cdict['blue'][6] = (0.6,0.45,0.45)
cdict['blue'][7] = (0.7,0.6,0.6)
cdict['blue'][9] = (1.0,0.7,0.7)

inc = np.linspace(0.0,1.0,10)
am = 5
cdict['alpha'][:] = zip(inc,np.clip(inc*am,0.0,1.0),np.clip(inc*am,0.0,1.0))

cm0 = mpl.colors.LinearSegmentedColormap('Spec',cdict)

cmap = mpl.colors.LinearSegmentedColormap.from_list('trunc', cm0(np.linspace(0.0, 1.0, 2**9)))


#%%
var = xa.open_dataset(r'.\data\misc\VAR0.1deg_v1.nc')

ds015 = xa.open_dataset(r'.\out\days_above_0.15.nc')
ds02 = xa.open_dataset(r'.\out\days_above_0.20.nc')
ds025 = xa.open_dataset(r'.\out\days_above_0.25.nc')

ds015 = ds015.where(ds015.sm>0)
ds015['sm'] /= 30.44
ds02 = ds02.where(ds02.sm>0)
ds02['sm'] /= 30.44
ds025 = ds02.where(ds025.sm>0)
ds025['sm'] /= 30.44

bbNA = [slice(-135,-62),slice(56,22)]
bbAU = [slice(111,160),slice(-10,-42)]
bbEU = [slice(-14,37),slice(64,32)]

dmax = 4#np.max([ds02.sm,ds025.sm,ds015.sm])
dmin = 1#np.min([ds02.sm,ds025.sm,ds015.sm])

cbpars = dict(shrink=0.5,orientation='horizontal', extend='both',ticks=range(dmin,dmax+1))
impars = dict(cmap=cmap,vmin=dmin,vmax=dmax,interpolation='None')

with PdfPages(r'.\res\days_above_plot.pdf') as pdf:
#    ax = plt.axes(projection=proj)
#    ax.coastlines()
    plt.imshow(ds02.sm,**impars)
    cb = plt.colorbar(**cbpars)
    cb.set_label(r'Months of potential activity; $\theta_c>$0.2 [$y^{-1}$]')
    cb.ax.xaxis.set_label_position('top')
#    plt.xlabel('lon')
#    plt.ylabel('lat')
    plt.axis('off')
    plt.tight_layout()
    pdf.savefig(dpi=300)
    plt.close()

#    ax = plt.axes(projection=proj)
#    ax.coastlines()
    plt.imshow(ds025.sm,**impars)
    cb = plt.colorbar(**cbpars)
    cb.ax.xaxis.set_label_position('top')
    cb.set_label(r'Months of potential activity; $\theta_c>$0.25 [$y^{-1}$]')
#    plt.xlabel('lon')
#    plt.ylabel('lat')
    plt.axis('off')
    plt.tight_layout()
    pdf.savefig(dpi=300)
    plt.close()

#    ax = plt.axes(projection=proj)
#    ax.coastlines()
    plt.imshow(ds015.sm,**impars)
    cb = plt.colorbar(**cbpars)
    cb.ax.xaxis.set_label_position('top')
    cb.set_label(r'Months of potential activity; $\theta_c>$0.15 [$y^{-1}$]')
#    plt.xlabel('lon')
#    plt.ylabel('lat')
    plt.axis('off')
    plt.tight_layout()
    pdf.savefig(dpi=300)
    plt.close()

#    ax = plt.axes(projection=proj)
#    ax.coastlines()
    plt.imshow(ds02.sm-ds025.sm,cmap='seismic')
    cb = plt.colorbar(shrink=0.5)
    cb.set_label(r'difference Months(>0.2)-Months(>0.25) [$y^{-1}$]')
#    plt.xlabel('lon')
#    plt.ylabel('lat')
    plt.axis('off')
    plt.tight_layout()
    pdf.savefig(dpi=300)
    plt.close()

#    ax = plt.axes(projection=proj)
#    ax.coastlines()
    plt.imshow(ds02.sm-ds015.sm,cmap='seismic')
    cb = plt.colorbar(shrink=0.5)
    cb.set_label(r'difference Months(>0.2)-Months(>0.15) [$y^{-1}$]')
#    plt.xlabel('lon')
#    plt.ylabel('lat')
    plt.axis('off')
    plt.tight_layout()
    pdf.savefig(dpi=300)
    plt.close()
print 'done'

#%%
proj = ccrs.PlateCarree()
with PdfPages(r'.\res\days_above_plot_ROI_0.2.pdf') as pdf:
    ax = plt.axes(projection=proj)
    ax.coastlines(resolution='110m')
#    plt.title('Global')
    plt.contourf(ds02.lon,ds02.lat,ds02.sm,range(dmin,dmax+1),cmap=cmap,extend='both')
    cb = plt.colorbar(orientation='horizontal',shrink=0.5)
    cb.set_label(r'Months of potential activity; $\theta_c>$0.2 [$y^{-1}$]')
    cb.ax.xaxis.set_label_position('top')
    plt.xlabel('lon')
    plt.ylabel('lat')
    plt.tight_layout()
    pdf.savefig(dpi=300)
    plt.close()

    dat = ds02.sel(lon=bbNA[0],lat=bbNA[1])
    ax = plt.axes(projection=proj)
    ax.coastlines(resolution='110m')
    plt.title('North America')
    plt.contourf(dat.lon,dat.lat,dat.sm,range(dmin,dmax+1),cmap=cmap,extend='both')
    cb = plt.colorbar(orientation='horizontal',shrink=0.5)
    cb.set_label(r'Months of potential activity; $\theta_c>$0.2 [$y^{-1}$]')
    cb.ax.xaxis.set_label_position('top')
    plt.xlabel('lon')
    plt.ylabel('lat')
    plt.tight_layout()
    pdf.savefig(dpi=300)
    plt.close()

    dat = ds02.sel(lon=bbAU[0],lat=bbAU[1])
    ax = plt.axes(projection=proj)
    ax.coastlines(resolution='110m')
    plt.title('Australia')
    plt.contourf(dat.lon,dat.lat,dat.sm,range(dmin,dmax+1),cmap=cmap,extend='both')
    cb = plt.colorbar(orientation='horizontal',shrink=0.5)
    cb.set_label(r'Months of potential activity; $\theta_c>$0.2 [$y^{-1}$]')
    cb.ax.xaxis.set_label_position('top')
    plt.xlabel('lon')
    plt.ylabel('lat')
    plt.tight_layout()
    pdf.savefig(dpi=300)
    plt.close()

    dat = ds02.sel(lon=bbEU[0],lat=bbEU[1])
    ax = plt.axes(projection=proj)
    ax.coastlines(resolution='110m')
    plt.title('Europe')
    plt.contourf(dat.lon,dat.lat,dat.sm,range(1,7),cmap=cmap,extend='both')
    cb = plt.colorbar(orientation='horizontal',shrink=0.5)
    cb.set_label(r'Months of potential activity; $\theta_c>$0.2 [$y^{-1}$]')
    cb.ax.xaxis.set_label_position('top')
    plt.xlabel('lon')
    plt.ylabel('lat')
    plt.tight_layout()
    pdf.savefig(dpi=300)
    plt.close()
print 'done again!'

with PdfPages(r'.\res\additional_variables.pdf') as pdf:

    ax = plt.axes(projection=proj)
    ax.coastlines(resolution='110m')
#    plt.title('Global')
    plt.contourf(var.lon,var.lat,0.1*var.NPP.where(var.NPP<=60000),cmap='Greens')
    cb = plt.colorbar(shrink=0.5)
    cb.set_label(r'NPP [$gC y^{-1} m^{-2}$]')
    plt.xlabel('lon')
    plt.ylabel('lat')
    plt.tight_layout()
    pdf.savefig(dpi=300)
    plt.close()

    ax = plt.axes(projection=proj)
    ax.coastlines(resolution='110m')
#    plt.title('Global')
    plt.contourf(var.lon,var.lat,var.SND.where(var.SND<=100) ,[0,25,50,75,100],cmap='inferno')
    cb = plt.colorbar(shrink=0.5)
    cb.set_label(r'Sand content [%]')
    plt.xlabel('lon')
    plt.ylabel('lat')
    plt.tight_layout()
    pdf.savefig(dpi=300)
    plt.close()
    
    ax = plt.axes(projection=proj)
    ax.coastlines(resolution='110m')
#    plt.title('Global')
    plt.contourf(var.lon,var.lat,var.ORC.where(var.ORC>0) ,cmap='copper_r')
    cb = plt.colorbar(shrink=0.5)
    cb.set_label(r'Organic carbon content [$g kg^{-2}$]')
    plt.xlabel('lon')
    plt.ylabel('lat')
    plt.tight_layout()
    pdf.savefig(dpi=300)
    plt.close()
    
print 'done for real!'