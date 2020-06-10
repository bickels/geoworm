# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:04:49 2019

@author: bickels
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xa

#%%
#met = pd.read_excel(r'C:/Users/bickels/Documents/GitHub/geoworm/data/1804_2_sWormModelData_meta-data_complete.csv')
df = pd.read_csv(r'C:/Users/bickels/Documents/GitHub/geoworm/data/1804_2_sWormModelData.csv')

#pd.merge([df,met])