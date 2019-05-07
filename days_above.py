# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 ‏‎12:39:10 2019

@author: bickels
"""

import xarray as xa
from dask.diagnostics import ProgressBar

#%%
with xa.open_mfdataset('*.nc',concat_dim='time',autoclose=True,lock=False) as dat:
    nyears = dat.sm.shape[0]/365.25

    b15 = dat.sm > 0.15
    b15 = b15.sum(dim='time')/nyears
    
    b20 = dat.sm > 0.2
    b20 = b20.sum(dim='time')/nyears
    
    b25 = dat.sm > 0.25
    b25 = b25.sum(dim='time')/nyears
    
    with ProgressBar():
        b15.to_netcdf(r'.\out\days_above_0.15.nc')
        
    with ProgressBar():
        b20.to_netcdf(r'.\out\days_above_0.20.nc')
        
    with ProgressBar():
        b25.to_netcdf(r'.\out\days_above_0.25.nc')