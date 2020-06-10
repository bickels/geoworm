# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:44:49 2019

@author: bickels
"""

import pandas as pd
#import seaborn as sns

#%%
df = pd.read_excel(r'./data/desert.xlsx')
df['biome'] = 'desert'

dftmp = pd.read_excel(r'./data/grassland.xlsx')
dftmp['biome'] = 'grassland'

df = pd.concat([df,dftmp])

#%%
df['dayofyear'] = df.time.dt.dayofyear
dfs = df.groupby(['biome','dayofyear']).precipitation.describe()
dfs.reset_index().to_excel(r'./data/precipitation_climatic.xlsx',index=False)

#%%
dft = df.groupby(['biome','time']).precipitation.describe()
dft.reset_index().to_excel(r'./data/precipitation_timeseries.xlsx',index=False)


#%%
df = pd.read_excel(r'./data/desert_PL.xlsx')
df['biome'] = 'desert'

dftmp = pd.read_excel(r'./data/grassland_PL.xlsx')
dftmp['biome'] = 'grassland'

df = pd.concat([df,dftmp])

df['moy'] = df.time.dt.month
dfs = df.groupby(['biome','moy']).PL.describe()
dfs.reset_index().to_excel(r'./data/PL_climatic.xlsx',index=False)

dft = df.groupby(['biome','time']).PL.describe()
dft.reset_index().to_excel(r'./data/PL_timeseries.xlsx',index=False)