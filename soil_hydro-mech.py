# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:16:51 2019

@author: bickels
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmocean

from scipy.optimize import curve_fit
from scipy.cluster.vq import kmeans2
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

import statsmodels.api as sm
import xarray as xa
from dask.diagnostics import ProgressBar
import ternary as ter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader

#%%
plt.style.use(r'C:/Users/bickels/Documents/Publication/ncomms.mplstyle')
#plt.style.use(r'C:/Users/bickels/Documents/Publication/poster.mplstyle')

cwidth = 3.50394
height = cwidth/1.618

proj = ccrs.PlateCarree()

panel = lambda xy,xytext,s: ax.annotate(s, xy=xy, xytext=xytext, 
                                        xycoords='axes fraction',
                                        textcoords='offset points',
                                        ha='left',
                                        va='top',
                                        fontsize=8,
                                        weight='bold')

#%%
def closest_node(node, nodes):
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

def distances(node, nodes):
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return dist_2

def kdequantile(dat,Nint=2**8,quantiles=np.array([0.975,0.75,0.5,0.25,0.025])):
    kde = gaussian_kde(dat)
    
    vv,nn = np.meshgrid(np.linspace(dat[0].min(),dat[0].max(),Nint),
                        np.linspace(dat[1].min(),dat[1].max(),Nint))
    z = kde(np.vstack([vv.ravel(),nn.ravel()]))
    
    z = z.reshape((Nint,Nint))
    z /= z.sum()
    
    t = np.linspace(0., z.max(), Nint)
    integral = ((z >= t[:, None, None]) * z).sum(axis=(1,2))
    
    f = interp1d(integral, t)
    t_cont = f(quantiles)
    return vv, nn, z, t_cont

def smhist(x,Nint=2**8,color='k'):
    kde = gaussian_kde(x)
    v = np.linspace(x.min(),x.max(),Nint)
#    v50 = np.linspace(x.quantile(0.25),x.quantile(0.75),Nint)
    v95 = np.linspace(x.quantile(0.025),x.quantile(0.975),Nint)
    plt.plot(v,kde(v),color=color,lw=0.5)
#    plt.fill_between(v50,kde(v50),color=color,alpha=0.2,lw=0)
    plt.fill_between(v95,kde(v95),color=color,alpha=0.2,lw=0)
    plt.vlines(x.quantile(0.5),0,kde(x.quantile(0.5)),lw=1,colors=color)

def area(lat, delta, R=6371.):
    return (np.pi/180.)*R**2*np.abs(np.sin(np.deg2rad(lat+delta/2.))-np.sin(np.deg2rad(lat-delta/2.)))*np.abs(delta)

#%%
dat = xa.open_dataset(r'C:\Users\bickels\Documents\Publication\2016-05-19_microgoegraphy\data\final\tmp\VAR0.1deg_v1.nc')
cov = xa.open_dataset(r'C:\Users\bickels\Documents\Publication\2016-05-19_microgoegraphy\data\final\tmp\OUT0.1deg_v1.nc')
cov = cov.where(cov.lat> -60)

pfz = xa.open_dataset(r'C:/Users/bickels/Documents/data/permafrost/GlobalPermafrostZonationIndexMap_2018.nc').Band1
pfz = pfz.where(pfz<=1).fillna(0)
dat['bPR'] = pfz.reindex_like(dat,method='nearest')>0.1
dat = dat.where((dat.lat> -60)&~dat.bPR)

dat['LAN'] = dat.LAN.astype(bool)
dat['bPR'] = dat.bPR.astype(bool)
#%%
import glob
files = glob.glob(r'C:/Users/bickels/Documents/data/GBIF/earthworms/*.csv')
occ = []
for f in files:
    occ0 = pd.read_csv(f,sep='\t')
    occ0 = occ0.rename({'decimalLatitude':'lat','decimalLongitude':'lon'},axis=1)
    occ0['src']='GBIF'
    occ.append(occ0[['lat','lon','src','family']].dropna())
occ = pd.concat(occ,ignore_index=True)

oAU = pd.read_csv('C:/Users/bickels/Documents/GitHub/geoworm/data/abbott_map/abbott1994_earthworm_occurences_AU.csv')
oAU = oAU.rename({'Y':'lat','X':'lon'},axis=1)
oAU['src']='Abbott 1994'
oAU['family']='Various'
oAU = oAU[['lat','lon','src','family']].dropna()

occ = pd.concat([occ,oAU],ignore_index=True)

#%%
occlan = dat.LAN.sel_points(lon=occ.lon.values,lat=occ.lat.values,method='nearest').to_dataframe()
occlan.loc[occ.lat< -60,'LAN']=False
occ['LAN'] = occlan['LAN']
occ = occ.where(occ.LAN).dropna()

#%%
'''
names = ['Uetliberg','Agroscope','Milville','Fraternidad']
cly = np.array([11,30,16,52])
slt = np.array([50,49,55,31])
snd = np.array([39,20,29,18])

bld = np.array([1200,1500,1650,np.nan])
bld[-1]=np.nanmean(bld)

ay = np.array([3e6,3e6,2e6,2e6])
by = np.array([-26.2,-15.92,-25.63,-5.97])
aG = np.array([1e7,8e7,2e8,1e7])#aG = np.array([1e7,8e7,2e8,np.nan])
bG = np.array([-20.8,-26.5,-27.5,-8.8])#bG = np.array([-20.8,-26.5,-27.5,np.nan])

#%%
pwr = lambda x,a,b: a*x**b
hyp = lambda x,a,b: a*x/(b+x)
lgs = lambda x,a,b,c: a/(1+np.exp(-b*(x-c)))
nex = lambda x,a,b: a*(1-np.exp(b*x))
'''
#%%
N = 2**8
wc = np.linspace(0.0,0.6,N/2)
#bl = np.linspace(1e3,2e3,N)
#ag = np.linspace(aG[:-1].min(),aG[:-1].max(),N)
#ft = np.linspace(0,100,N)

#%%
'''
aG_pars,_ = curve_fit(pwr,bld[:-1],aG[:-1], p0=(1e-24,10))

plt.plot(bld,aG,'o')
plt.plot(bl,pwr(bl,*aG_pars))
plt.loglog()
plt.xlabel('Bulk density')
plt.ylabel(r'$a_G$')
plt.show()

#%%
bG_fpars,_ = curve_fit(hyp,aG,-bG)
#bG_fpars,_ = curve_fit(nex, aG[:-1],-bG[:-1], p0=(30,-8.5e8))

plt.plot(aG,-bG,'o')
plt.plot(ag,hyp(ag,*bG_fpars))
#plt.loglog()
plt.xlabel(r'$a_G$')
plt.ylabel(r'$-b_G$')
plt.show()

#%%
by_pars,_ = curve_fit(lgs,100-cly,-by,p0=(30,1e-1,60))

plt.plot(100-cly,-by,'o')
plt.plot(ft,lgs(ft,*by_pars))
plt.xlabel(r'100-clay (%)')
plt.ylabel(r'$-b_y$')
plt.show()

##%%
#fpars,_ = curve_fit(pwr,100-cly,-by)
#
#ft = np.linspace(0,100)
#
#plt.plot(100-cly,-by,'o')
#plt.plot(ft,pwr(ft,*fpars))
#plt.xlabel(r'100-clay(%)')
#plt.ylabel(r'$b_y$')
#plt.show()

#%%
aG_est = pwr(bl,*aG_pars)
bG_est = -hyp(aG_est,*bG_fpars)
ay_est = np.mean(ay)
by_est = -lgs(ft,*by_pars)

plt.plot(aG,bG,'o')
plt.plot(aG_est,bG_est)
plt.xscale('log')
plt.show()

G = aG_est*np.exp(bG_est*wc[:,None])
su = ay_est*np.exp(by_est*wc[:,None])/3**0.5

plt.contourf(wc,bl,G.T,norm=mpl.colors.LogNorm())
plt.hlines(bld[:-1].max(),0,1)
plt.hlines(bld[:-1].min(),0,1)
plt.ylabel(r'Bulk density (kg/m$^{3}$)')
plt.xlabel(r'Water content (g/g)')
plt.colorbar(label='G (Pa)')
plt.show()

plt.contourf(wc,ft,su.T,norm=mpl.colors.LogNorm())
plt.hlines((100-cly).max(),0,1)
plt.hlines((100-cly).min(),0,1)
plt.ylabel(r'100-clay (%)')
plt.xlabel(r'Water content (g/g)')
plt.colorbar(label='$s_u (Pa)$')
plt.show()

plt.contourf(wc,ft,(G/su).T,norm=mpl.colors.LogNorm())
plt.ylabel(r'100-clay (%)')
plt.xlabel(r'Water content (g/g)')
plt.colorbar(label='$G/s_u$')
plt.show()
'''
#%%

df = dat[['SND','CLY','SLT']].to_dataframe()
#df = dat[['SND','CLY','SLT','BLD']].to_dataframe()

df.loc[df.SLT==255.,'SLT']=np.nan
df.loc[df.CLY==255.,'CLY']=np.nan
df.loc[df.SND==255.,'SND']=np.nan
#df.loc[df.BLD<0,'BLD']=np.nan
df = df.drop(columns=['band','level']).dropna()

#bld[-1]=np.nanmean(bld)

#k = np.c_[snd,cly,slt]#,bld]

#centroid,labels = kmeans2(df.values,k)
#
#tex = np.c_[snd,slt,cly]
#
#labels = [closest_node(te,tex) for te in df[['SND','SLT','CLY']].values]
#
#parms = np.c_[ay,by,aG,bG]
#
#ays,bys,aGs,bGs = np.array([parms[i] for i in labels]).T
#
##%%
#dists = np.array([distances(te,tex) for te in df[['SND','SLT','CLY']].values])
#wavg = np.array([np.average(parms,axis=0,weights=1./dd) for dd in dists])
#ays,bys,aGs,bGs = wavg.T

#%%
prePL = lambda ff,a,b: a*np.exp(b*ff)
expPL = lambda ff,a,b: a+b*ff
explg = lambda p,a,b: a+b*np.log(p)

suPL = lambda wc,a,b: a*wc**-b

ff = np.linspace(0,100,N)
#ff = cly+slt

ayPL = prePL(ff,0.5027,0.1052)*1e5
byPL = expPL(ff,1.2457,0.0162)

aGPL = prePL(ff,37.941,0.2276)*1e6#aGs
bGPL = expPL(ff,0.2908,0.0689)
#su = ay*np.exp(by*wc[:,None]*(bld*1e-3))/3**0.5
sua = suPL(100*wc[:,None],ayPL,byPL)
Ga = suPL(100*wc[:,None],aGPL,bGPL)
PLa = sua*(1+np.log(Ga/sua))

#plt.plot(wc,su)
plt.plot(wc,sua)
plt.plot(wc,Ga)

plt.yscale('log')
plt.show()

fig, ax = plt.subplots()
CS = ax.contour(100*wc,ff,sua.T,[2e5],colors='r')
ax.clabel(CS, inline=1, fontsize=10, fmt='%.e')
plt.pcolormesh(100*wc,ff,sua.T,norm=mpl.colors.LogNorm(),vmax=1e8)
plt.ylabel(r'Silt+clay (%)')
plt.xlabel(r'Water content (%v/v)')
plt.colorbar(label='$s_u (Pa)$', extend='max')
plt.show()

fig, ax = plt.subplots()
CS = ax.contour(100*wc,ff,Ga.T,[2e5],colors='r')
ax.clabel(CS, inline=1, fontsize=10, fmt='%.e')
plt.pcolormesh(100*wc,ff,Ga.T,norm=mpl.colors.LogNorm(),vmax=1e8)
plt.ylabel(r'Silt+clay (%)')
plt.xlabel(r'Water content (%v/v)')
plt.colorbar(label='$G (Pa)$', extend='max')
plt.show()

fig, ax = plt.subplots()
CS = ax.contour(100*wc,ff,PLa.T,[2e5],colors='r')
ax.clabel(CS, inline=1, fontsize=10, fmt='%.e')
plt.pcolormesh(100*wc,ff,PLa.T,norm=mpl.colors.LogNorm(),vmax=1e8)
plt.ylabel(r'Silt+clay (%)')
plt.xlabel(r'Water content (%v/v)')
plt.colorbar(label='$P_L (Pa)$', extend='max')
plt.show()

#%%
df['ff'] = df.SLT+df.CLY
#df['labels']=labels

df['ay']=prePL(df.ff,0.5027,0.1052)*1e5#ays
df['by']=expPL(df.ff,1.2457,0.0162)#bys

df['aG']=prePL(df.ff,37.941,0.2276)*1e6#aGs
df['bG']=expPL(df.ff,0.2908,0.0689)

out = df.to_xarray()

#%%
wet = xa.open_dataset(r'C:/Users/bickels/Documents/data/ESACCI-SOILMOISTURE-WETLAND_FRACTION_V01.1.nc').wetland_fraction*1e-2
wet = wet.interp(lon=dat.lon,lat=dat.lat,method='linear')

#era = xa.open_dataset('Z:/bickels/data/ERA5/ERA5-land-swvl.nc').rename({'longitude':'lon','latitude':'lat'})
#era.lon = era.lon-180.
#era = era.swvl1.reindex(lon=dat.lon,lat=dat.lat,method='nearest')
#era = era.chunk({'time':1})

#%%
ts = dat['AWC'].where(dat['AWC']!=255)*1e-2

tfc = ts*0.5
alp = 1e-3*dat['PET']
cov['wcc'] = tfc*np.exp(-alp*dat['DRY']/tfc)#(1-wet.values)*tfc*np.exp(-alp*dat['DRY']/tfc)+wet.values*ts*np.exp(-alp*dat['DRY']/ts)
#cov['wcg'] = (cov.wc/(1e-3*dat.BLD.where(dat.BLD>0)))

out['G_cwc'] = suPL(100*cov.wc,out.aG,out.bG)#out.aG*np.exp(out.bG*cov.wcg)
out['su_cwc'] = suPL(100*cov.wc,out.ay,out.by)#out.ay*np.exp(out.by*cov.wcg)/3**0.5
out['PL_cwc'] = out.su_cwc*(1+np.log(out.G_cwc/out.su_cwc))

#%%
#era['G'] = suPL(100*era,out.aG,out.bG)
#era['su'] = suPL(100*era,out.ay,out.by)
#era['PL'] = out.su*(1+np.log(era.G/era.su))
era = xa.open_dataset(r'C:\Users\bickels\Documents\GitHub\geoworm\data\ERA\era_PL_Y_harmonic.nc')
#era = xa.open_dataset(r'C:\Users\bickels\Documents\GitHub\geoworm\data\ERA\era_PL_Y_clim.nc')

#eram = xa.open_dataset(r'C:\Users\bickels\Documents\GitHub\geoworm\data\ERA\era_PL_M_clim.nc')

erac = xa.open_dataset(r'C:\Users\bickels\Documents\GitHub\geoworm\data\ERA\era_bPL_M_consecutive.nc')

#eraf = xa.open_dataset(r'C:\Users\bickels\Documents\GitHub\geoworm\data\ERA\era_bPL_M_clim.nc')
#avg = 'aritmethic'
avg = 'harmonic'

#out['G'] = era.G
#out['su'] = era.su
out['PL'] = era.PL

nyr = 220/12.
out['aM'] = erac.PL.fillna(0).where(out.PL>0)#/nyr#(eram.PL<2e5).sum('month')

#%%
#plt.hexbin(cov.wc*100,out.ff,extent=(0,50,0,100),mincnt=1,gridsize=20)
#plt.colorbar(label='Counts')
#plt.xlabel(r'Water content (%v/v)')
#plt.ylabel(r'Silt+clay (%)')
#plt.show()

#%%
#from uncertainties import ufloat
#from uncertainties.umath import

#%%
#col0 = plt.cm.get_cmap('plasma',len(tex))(range(len(tex)))
#[plt.plot(snd[i],cly[i],'o',label=names[i],color=col0[i]) for i in range(len(snd))]
#plt.legend()
#plt.hexbin(df.SND,df.CLY,gridsize=15,mincnt=1,cmap='bone',norm=mpl.colors.LogNorm())
#plt.colorbar(label='Number of pixels in SoilGrids 10km')
#plt.xlabel('Sand (%)')
#plt.ylabel('Clay (%)')
#plt.xlim(0,100)
#plt.ylim(0,100)
#plt.show()

#%%
#plt.contourf(out.lon,out.lat,out.labels,cmap=plt.cm.get_cmap('plasma',len(tex)))
#plt.colorbar(ticks=range(len(tex)))
#plt.clim(-0.5,len(tex)-0.5)
#plt.show()

#%%
#plt.contourf(out.lon,out.lat,out.G,norm=mpl.colors.LogNorm(),vmin=1,vmax=1e12)
#plt.colorbar(label='G (Pa)')
#plt.show()
#
#plt.contourf(out.lon,out.lat,out.su,norm=mpl.colors.LogNorm(),vmin=1,vmax=1e12)
#plt.colorbar(label='$s_u (Pa)$')
#plt.show()

#%%
plt.pcolormesh(out.lon,out.lat,out.PL*1e-3,norm=mpl.colors.LogNorm(),vmin=10,vmax=1e3)
plt.colorbar(label='$P_L\, (kPa)$')
#plt.contourf(out.lon,out.lat,out.PL_cwc>2e5,[0.5,1],colors='r',alpha=0.1)
#plt.contour(out.lon,out.lat,out.PL>2e5,[0.5,1],colors='r',alpha=0.5,lw=0)
#plt.contourf(out.lon,out.lat,out.PL>2e5,[0.5,1],colors='r',hatches='//',alpha=0.5)
plt.show()

#%%
#(out.PL-out.PL_cwc).plot.imshow(vmin=-1e3,vmax=0.1)

#%%
#cols = np.array([np.average(col0,axis=0,weights=1./dd) for dd in dists])

#df = df.reset_index()
#plt.scatter(df.lon,df.lat,color=cols,s=1)

#%%
#scale=100
#offset=0.1
#fig, tax = ter.figure(scale=scale)
#tax.boundary(linewidth=2.0)
#tax.gridlines(color="black", multiple=10)
##tax.gridlines(color="blue", multiple=1, linewidth=0.5)
#
#data = {key:val for key, val in zip(zip(df.SND,df.SLT,df.CLY),df.labels)}
##tax.heatmap(data,style='hexagonal',cmap=plt.cm.get_cmap('plasma',len(tex)), colorbar=False)
##tax.scatter(np.c_[df.SND,df.SLT,df.CLY],c=df.labels)
#tax.scatter(np.c_[snd,slt,cly],color=col0,edgecolor='w',zorder=10)
#tax.scatter(zip(df.SND,df.SLT,df.CLY),color=cols)
#[plt.scatter([],[],label=names[i],color=col0[i]) for i in range(len(snd))]
#plt.legend()
#tax.ticks(axis='lbr', linewidth=1, multiple=10)
#
#tax.left_axis_label("Clay (%)", offset=offset)
#tax.right_axis_label("Silt (%)", offset=offset)
#tax.bottom_axis_label("Sand (%)", offset=offset)
#tax.get_axes().axis('off')
#tax.clear_matplotlib_ticks()
#ter.plt.show()

#%%
bdict = {1:'Tropical Forests', 2:'Tropical Forests', 3:'Tropical Forests', 4:'Temperate Forests', 5:'Temperate Forests', 6:'Boreal Forests', 7:'Tropical Grasslands',8:'Temperate Grasslands',9:'Others', 10:'Montane Grasslands', 11:'Tundra',12:'Mediterranean', 13:'Deserts', 14:'Others', 99:'Others'}

BIO = xa.open_rasterio(r'C:\Users\bickels\Documents\data\official\BIOME.tif').rename({'x':'lon','y':'lat'})
cov['BIO'] = BIO.reindex(lon=cov.lon,lat=cov.lat,method='nearest').astype('int8')
'''
smd = xa.open_dataset(r'C:/Users/bickels/Documents/data/rain-moist/ESA_all_SM_daily_climatic.nc')
smd['BIO'] = BIO.reindex(lon=smd.lon,lat=smd.lat,method='nearest').astype('int8')
smd = smd.assign_coords(bio=smd.BIO)
smd = smd.stack(space=('lat','lon'))
#with ProgressBar():
#    smd.to_netcdf('.\ESA_biom_mean_clim_bio.nc')
#ds = xa.concat([smd.sm.where(smd.BIO==i).mean(dim=('space')) for i in range(1,15)])

#df = pd.DataFrame(index=range(1,15),data=ds.squeeze().values)
#df['biome'] = [bdict[k] for k in range(1,15)]
#df.to_excel('.\esa_sm_biome_clim.xlsx')
df = pd.read_excel('.\esa_sm_biome_clim.xlsx')
cwc = cov.wc.values
cwcB = np.array([np.nanmean(cwc[cov.BIO[0].values==i]) for i in range(1,15)])
df0 = pd.DataFrame(index=range(1,15),data=cwcB)
df0['biome'] = [bdict[k] for k in range(1,15)]
df0 = df0.groupby('biome').mean().reset_index()

df = df.groupby('biome').mean().reset_index()
#df = df.melt(id_vars='biome')

#%%
doy = np.arange(366)

for name,ssm,cw in zip(df.biome,df.drop(columns='biome').values,df0[0].values):
    plt.plot(doy,ssm,color=mpl.cm.nipy_spectral_r(cw/0.4),label=name)
plt.legend()
plt.show()
#    plt.hlines(cw,0,366,color=mpl.cm.nipy_spectral_r(cw/0.4))
#smd.groupby('bio').mean(dim=('space')).sm.plot.line(x='dayofyear')
#smg = smd.where((smd.BIO<15)&(smd.BIO>0)).groupby('bio').chunk({'time':1})
#
#with ProgressBar():
#    smg.mean(dim=('space')).to_netcdf('.\ESA_biom_mean_clim.nc')
'''
#%%
"""
with xa.open_mfdataset('.\data\ESA-CCI-SM\*.nc',concat_dim='time',chunks={'time':1}) as ssm:
    ssm['BIO'] = BIO[0].reindex(lon=ssm.lon,lat=ssm.lat,method='nearest')
    
    out['wc'] = cov.wc
    out['BLD'] = dat.BLD
    ous = out.reindex(lon=ssm.lon,lat=ssm.lat,method='nearest')
    
    ssm['wcg'] = (ssm.sm/(1e-3*ous.BLD.where(ous.BLD>0)))
    
    ssm['G'] = ous.aG*np.exp(ous.bG*ssm.wcg)
    ssm['su'] = ous.ay*np.exp(ous.by*ssm.wcg)/3**0.5
    ssm['PL'] = ous.su*(1+np.log(ssm.G/ssm.su))
    
    ssma = (ssm.PL<2e5).sum(dim='time')
    with ProgressBar():
        ssma.to_netcdf('.\ESA_days_above.nc')
        
    ssmy = ssm.mean(dim='time')
    with ProgressBar():
        ssmy.to_netcdf('.\ESA_mean.nc')"""
'''
ndays = 13941
ssmy = xa.open_dataset('.\ESA_mean.nc')
ssmy['cwc'] = cov.wc.reindex(lon=ssmy.lon,lat=ssmy.lat,method='nearest')

ssma = xa.open_dataset('.\ESA_days_above.nc')
ssma['fd'] = ssma.PL/float(ndays)

plt.hexbin(ssmy.cwc.values,ssmy.sm.values,norm=mpl.colors.LogNorm(),extent=(0,0.5,0,0.5))
plt.plot([0,0.5],[0,0.5],'r')
plt.xlim(0,0.5)
plt.ylim(0,0.5)
plt.axis('equal')
plt.show()

#%%
plt.pcolormesh(ssmy.lon,ssmy.lat,ssmy.PL,norm=mpl.colors.LogNorm(),vmin=1e4)
plt.colorbar(label='$P_L (Pa)$')
plt.contourf(ssmy.lon,ssmy.lat,ssmy.PL>2e5,[0.5,1],colors='r',hatches='//',alpha=0.5)
plt.show()
'''

#%%
out['A'] = (('lat','lon'),np.repeat(area(out.lat,0.1).values[:,None],out.lon.shape,axis=1))

out['bPL'] = out.PL<=2e5

out['baM'] = out.aM >= 2

out['bMAT'] = dat.MAT >= 0
out['MAT'] = dat.MAT

out['bpH'] = dat.PHH.where(dat.PHH!=255)*0.1 >= 4.5#https://www.dpi.nsw.gov.au/agriculture/soils/biology/earthworms
out['pH'] = dat.PHH.where(dat.PHH!=255)*0.1

out['bSND'] = dat.SND.where(dat.SND!=255) <= 80.
out['SND'] = dat.SND.where(dat.SND!=255)

out['b'] = out.bPL&out.bMAT&out.bpH&out.bSND&out.baM&~dat.bPR


OCC = np.histogram2d(occ.lat,occ.lon,
   bins=[np.pad(out.lat,1,'constant',constant_values=out.lat[-1]+0.1)[1:],
         np.pad(out.lon,1,'constant',constant_values=out.lon[-1]+0.1)[1:]])[0]
out['occ'] = xa.DataArray(OCC,coords={'lat':out.lat.values,'lon':out.lon.values},dims=('lat','lon'))
out['bocc'] = out.occ>0

dff = out.sel_points(lon=occ.lon.values,lat=occ.lat.values,method='nearest')
dff = dff.to_dataframe()
dff = dff.loc[(dff.PL>=0)&(dff.PL<=1e9)]
#dff['bPL'] = dff.PL<=2e5
np.count_nonzero(dff.bPL)/float(len(dff))

occ00 = occ.groupby([occ.lon.round(1),occ.lat.round(1)]).count().rename(columns={'lon':'lon_native','lat':'lat_native'}).reset_index()

#dff = dat.sel_points(lon=occ00.lon.values,lat=occ00.lat.values,method='nearest')
#dff = dff.to_dataframe()
#dff['PL'] = out.PL.sel_points(lon=occ00.lon.values,lat=occ00.lat.values,method='nearest')
#dff['A'] = out.A.sel_points(lon=occ00.lon.values,lat=occ00.lat.values,method='nearest')
#dff['occ'] = occ00.src/dff.A
#dff = dff.loc[(dff.PL>=0)&(dff.PL<=1e9)]
#dff.dropna().to_excel('.\data\occ.xlsx')

#%%
rgb = np.stack([out.bMAT.values*255,out.baM.values*255,out.bpH.values*255],axis=-1)

out['rgb']=(['lat','lon','ch'],rgb)

#%%
fig = plt.figure()
fig.set_size_inches(2*cwidth, 2*height)

ax = plt.axes(projection=proj)
ax.coastlines(resolution='50m')
out.rgb.where(dat.lat>-60).plot.imshow(rgb='ch')
#plt.contour(out.lon,out.lat,out.bocc,color='b')
#plt.loglog()
#plt.xlim(1,1e7)
plt.show()

#%% Logit
#import seaborn as sns
#sns.lmplot('PL','bPL',data=dff,logistic=True)
#mod = sm.Logit(dff.bPL,sm.add_constant(dff.PL)).fit()
mod = sm.Logit(dff.b,sm.add_constant(dff.PL)).fit()

dfit = dff[['b','PL','MAT','pH','aM','SND']].dropna()
mod1 = sm.Logit(dfit.b,sm.add_constant(dfit[['PL','MAT','pH','aM','SND']])).fit()
ax = plt.subplot(211)
plt.hist([dff.PL[dff.b],dff.PL[~dff.b]],bins=np.logspace(4,7),stacked=True)
plt.ylabel('Occurrences')

ax = plt.subplot(212,sharex=ax)
plt.scatter(dff.PL[dff.b],dff.bPL[dff.b])
plt.scatter(dff.PL[~dff.b],dff.bPL[~dff.b])
plt.plot(np.logspace(4,7),mod.predict(sm.add_constant(np.logspace(4,7))),'k')
plt.ylabel('Probability')
plt.xlabel('$P_L (Pa)$')
plt.xscale('log')
plt.show()

out['Ppl']=(['lat','lon'],mod1.predict(sm.add_constant(zip(out.PL.values.ravel(),out.MAT.values.ravel(),out.pH.values.ravel(),out.aM.values.ravel(),out.SND.values.ravel()))).reshape(out.PL.shape))
#out['Ppl']=(['lat','lon'],mod.predict(sm.add_constant(out.PL.values.ravel())).reshape(out.PL.shape))

#%%
fig = plt.figure()
fig.set_size_inches(2*cwidth, 2*height)

ax = plt.axes(projection=proj)
ax.coastlines(resolution='50m')
plt.pcolormesh(out.lon,out.lat,out.Ppl,cmap='coolwarm_r',vmin=0,vmax=1)
plt.colorbar(label='Probability of true positive',shrink=0.5)

plt.contourf(out.lon,out.lat,1-out.Ppl.where(~out.bPL),[0.5,0.95,1.0],cmap='Reds',alpha=0.3,vmin=0.5,vmax=1.0)
plt.contourf(out.lon,out.lat,out.Ppl.where(out.bPL),[0.5,0.95,1.],cmap='Greens',alpha=0.3,vmin=0.5,vmax=1.0)
#plt.contourf(out.lon,out.lat,out.PL<=2e5,[0.5,1],colors='g',alpha=0.3)
#plt.contourf(out.lon,out.lat,out.PL>2e5,[0.5,1],colors='r',alpha=0.3)
#plt.contourf(out.lon,out.lat,out.b,[0.5,1],colors='g',alpha=0.3)
#plt.contourf(out.lon,out.lat,~out.b,[0.5,1],colors='r',alpha=0.3)
plt.scatter(occ.lon,occ.lat,color='b',marker=',',s=3)
#plt.pcolormesh(out.lon,out.lat,(out.occ>0).where(out.occ>0),color='b')
plt.show()

#%%
valid = (out.occ>0)&(out.PL>0)&~np.isinf(out.PL)
din = np.c_[1e-3*out.PL.values[valid],out.occ.values[valid]]
qs = np.array([0.95, 0.75 , 0.5 , 0.25])
pl,oc,de,co = kdequantile(np.log10(din.T),quantiles=qs)
plt.scatter(*din.T,c='0.75',s=3)
cs = plt.contour(10**pl,10**oc,de,co,linewidths=2,colors=['0.0','0.2','0.4','0.6'])
#plt.clabel(cs, cs.levels)
[cs.collections[i].set_label(int(q*100)) for i,q in enumerate(qs)]
plt.legend(title='Percentile (%)',frameon=False)
plt.vlines(2e2,8e-1,2.5e3,color='r', linestyles='--')
plt.text(2e2,1e3,'200 kPa', color='r',rotation=-90)
plt.xlim(1e0,6e2)
plt.yscale('log')
#plt.loglog()
plt.xlabel(r'Limiting pressure $P_L \,(kPa)$')
plt.ylabel(r'Earthworm occurence (#)')
plt.savefig(r'./out/distribution_{:}.png'.format(avg),dpi=300)
plt.show()

#%%
#lgs0 = lambda x,b,c: lgs(x,1,b,c)
#pa,co = curve_fit(lgs0,dff.PL,dff.bPL,p0=(2e-3,0.1))
#plt.scatter(dff.PL,dff.bPL)

#%%
#a = np.random.uniform(1,out.PL.max(),(2**7,2**7))
def crit_cmap(a,t,f=1e-2,log=True):
    if log:
        la=np.log10(a)
        cp = (np.log10(t)-la.min())/(la.max()-la.min())
    else:
        cp = (t-a.min())/(a.max()-a.min())
    
    sub = cp*(1-f)
#    cdict = {'red':   [[0.0, 1.0, 1.0],
#                       [sub, 0.1, 0.1],
#                       [ cp, 0.2, 0.2],
#                       [1.0, 1.0, 1.0]],
#
#             'green': [[0.0, 1.0, 1.0],
#                       [sub, 0.1, 0.1],
#                       [ cp, 0.0, 0.0],
#                       [1.0, 0.0, 0.0]],
#                       
#             'blue':  [[0.0, 1.0, 1.0],
#                       [sub, 0.1, 0.1],
#                       [ cp, 0.0, 0.0],
#                       [1.0, 0.0, 0.0]]}
             
    cdict = {'red':   [[0.0, 0.1, 0.1],
                       [sub, 0.8, 0.8],
                       [ cp, 0.8, 0.8],
                       [1.0, 0.3, 0.3]],
    
             'green': [[0.0, 0.6, 0.6],
                       [sub, 0.8, 0.8],
                       [ cp, 0.8, 0.8],
                       [1.0, 0.3, 0.3]],
                       
             'blue':  [[0.0, 0.1, 0.1],
                       [sub, 0.8, 0.8],
                       [ cp, 0.8, 0.8],
                       [1.0, 0.3, 0.3]]}
             
    return mpl.colors.LinearSegmentedColormap('GrRd', cdict)

cm0 = crit_cmap(np.array([0,4e2]),2e2,log=False)

#cm1 = mpl.colors.LinearSegmentedColormap.from_list('cm1',['limegreen'])

plt.imshow(1e-3*out.PL,cmap=cm0,vmin=0,vmax=4e2,origin='lower')
plt.colorbar()

#%%
cmat = '#5544ae'
csnd = '#b1c44e'
cpH = '#bc466e'
caM = '#56cba4'
alpha = 1.
#bMAT = dat.MAT < 0
#bfd = ssma.fd.interp_like(dat) > (2./12.)
fig = plt.figure()
fig.set_size_inches(cwidth, 2*height)

#ax = plt.axes(projection=proj)
ax = plt.subplot(2, 1, 1, projection=proj)
ax.text(-175,78,'a',weight='bold',fontsize=8)
#plt.title('Average ({:}) annual liminting pressure'.format(avg))

ax.set_ylim(-60,90)
ax.set_xlim(-180,179)
ax.coastlines(resolution='50m',lw=0.5)
#plt.title('North America')
#plt.pcolormesh(out.lon,out.lat,1e-3*out.PL,cmap=cm0,norm=mpl.colors.LogNorm(),vmin=1e1,vmax=1e3)
#plt.contour(out.lon,out.lat,1e-3*out.PL.where((out.lat>-60.)&(out.PL>0)),[200],colors='m',linewidths=0.25)

plt.pcolormesh(out.lon,out.lat,1e-3*out.PL.where((out.lat>-60.)&(out.PL>0)&~dat.bPR),cmap=cm0,vmin=0,vmax=4e2)

cb = plt.colorbar(orientation='horizontal', shrink=0.5, pad=0.05)#,extend='both')
cb.set_label(r'Limiting pressure $P_L \,(kPa)$')
#cb.ax.plot([0.5, 0.5], [0.0, 1.0], 'm',lw=1)
#cb.ax.xaxis.set_label_position('top')
#plt.clim(1e1,1e3)
#plt.contourf(out.lon,out.lat,1-out.bMAT.where(dat.LAN==1),[0.5,1],colors='b',alpha=0.2)
#plt.contourf(bfd.lon,bfd.lat,bfd.where((out.PL<2e5)&~bMAT),[0.5,1],colors='g',alpha=0.2)
#plt.xlabel('lon')
#plt.ylabel('lat')
#plt.tight_layout()
#plt.legend()
ax = plt.subplot(2, 1, 2, projection=proj)
ax.text(-175,78,'b',weight='bold',fontsize=8)
#plt.title('Other factors for exclusion')
ax.set_ylim(-60,90)
ax.set_xlim(-180,179)
#ax.background_img(name='ne_gray',resolution='high')
ax.coastlines(resolution='50m',lw=0.5)
#plt.contourf(out.lon,out.lat,out.aM.where((out.lat>-60.)&(out.PL>0)),[0,1,2,3,4], extend='max', cmap='gray',transform=proj,alpha=0.5)
#cb = plt.colorbar(orientation='horizontal', shrink=0.5, pad=0.05)#,extend='both')
#cb.set_label(r'Consecutive months with $P_L<200kPa$')
plt.contourf(out.lon,out.lat,(~out.baM).where(dat.LAN&(out.SND<100)),[0.49,1], colors=caM,transform=proj,alpha=alpha, linewidths=0)
plt.contourf(out.lon,out.lat,(~out.bMAT).where(dat.LAN&(out.SND<100)),[0.49,1], colors=cmat,transform=proj,alpha=alpha, linewidths=0)
plt.contourf(out.lon,out.lat,(~out.bSND).where(dat.LAN&(out.SND<100)),[0.49,1], colors=csnd,transform=proj,alpha=alpha, linewidths=0)
plt.contourf(out.lon,out.lat,(~out.bpH).where(dat.LAN&(out.SND<100)),[0.49,1], colors=cpH,transform=proj,alpha=alpha, linewidths=0)

#plt.contour(out.lon,out.lat,1e-3*out.PL.where((out.lat>-60.)&(out.PL>0)),[200],colors='m',linewidths=0.25)

#proxy = [plt.Rectangle((0,0),1,1,fc='b'),plt.Rectangle((0,0),1,1,fc='g'),plt.Rectangle((0,0),1,1,fc='r')]
#plt.legend(proxy, [u'MAT<0°C', r'SND>80%', r'pH<4'],bbox_to_anchor=(1.04,0.5), loc='center left')

proxy = [plt.Rectangle((0,0),1,1,fc=cmat,alpha=alpha),plt.Rectangle((0,0),1,1,fc=csnd,alpha=alpha),plt.Rectangle((0,0),1,1,fc=cpH,alpha=alpha),plt.Rectangle((0,0),1,1,fc=caM,alpha=alpha)]
plt.legend(proxy, [u'MAT < 0°C',r'Sand > 80%', r'Soil pH < 4.5',r'Activity < 2 mos.'],frameon=False,bbox_to_anchor=(0.5,0.05), bbox_transform=fig.transFigure,loc='lower center', ncol=2)

plt.savefig(r'./out/pressure_map_{:}.png'.format(avg),dpi=300)
plt.show()

#%%
fig, ax = plt.subplots()
CS = ax.contour(100*wc,ff,1e-3*PLa.T,[200],colors='r')
ax.clabel(CS, inline=1, fontsize=6, fmt='%.d kPa')
cs = plt.contour(100*wc,ff,1e-3*PLa.T,[50,100,400,800],linewidths=1,colors='k',norm=mpl.colors.LogNorm())
ax.clabel(cs, inline=1, fontsize=6, fmt='%.d kPa')
#plt.text(10,84,'$P_L \,(kPa)$',rotation=47,fontsize=7)
plt.ylabel(r'Silt+clay (%)')
plt.xlabel(r'% Water content $(m^3\, m^{-3}$)')
#plt.colorbar(label='$P_L (kPa)$', extend='max')
plt.plot([],[],label='Limiting pressure $P_W$',color='r')
plt.legend(frameon=False,loc=4)

plt.savefig(r'./out/relation.png',dpi=300)
plt.show()

#%%
'''
bbNA = [slice(-135,-62),slice(22,56)]
bbAU = [slice(111,160),slice(-45,-10)]
bbEU = [slice(-14,37),slice(32,64)]

NA = out.sel(lon=bbNA[0],lat=bbNA[1])
AU = out.sel(lon=bbAU[0],lat=bbAU[1])
EU = out.sel(lon=bbEU[0],lat=bbEU[1])

#proj = ccrs.LambertConformal()
res = '50m'
occol = 'b'

fig = plt.figure()
fig.set_size_inches(2*cwidth, 4*height)
#gs = mpl.gridspec.GridSpec(4,1,wspace=0.05)

ax = plt.subplot(4, 1, 1, projection=proj)#fig.add_subplot(gs[0],projection=proj)
ax.background_img(name='ne_gray',resolution='high')
ax.coastlines(resolution=res,zorder=2,lw=1)
ax.add_feature(cfeature.OCEAN.with_scale(res),zorder=1)
ax.add_feature(cfeature.LAKES.with_scale(res),zorder=1,edgecolor='None')
#ax.add_feature(cfeature.RIVERS.with_scale(res))
#plt.contourf(out.lon,out.lat,(out.PL<2e5).where(bfd&~bMAT),[0.5,1],colors='0.5')
#plt.pcolormesh(out.lon,out.lat,(out.PL<2e5).where(bfd&~bMAT),cmap='Greens')
#plt.pcolormesh(out.lon,out.lat,(out.PL<2e5).where(~bMAT),color='g',alpha=0.5)
plt.contourf(out.lon,out.lat,out.bPL,[0.49,1],colors='g',alpha=0.3)
#plt.contourf(out.lon,out.lat,out.occ.where(out.occ>0),[0.5,1],colors='b',alpha=1)
#plt.pcolormesh(out.lon,out.lat,(out.bocc).where(out.bocc),cmap=mpl.colors.ListedColormap([occol]),edgecolor=None,transform=proj,alpha=1)
#cs = plt.contour(dat.lon,dat.lat,dat.MAP,[400],colors='cyan',linewidths=0.5,zorder=2)
#plt.pcolormesh(out.lon,out.lat,(out.bocc).where(out.bPL&out.bocc),cmap=mpl.colors.ListedColormap([occol]),edgecolor=None,transform=proj,alpha=1)
#plt.pcolormesh(out.lon,out.lat,(out.bocc).where(~out.bPL&out.bocc),cmap=mpl.colors.ListedColormap(['red']),edgecolor=None,transform=proj,alpha=1)
plt.scatter(occ.lon,occ.lat,color=occol,marker='o',s=1,zorder=3)

fname = 'C:/Users/bickels/Documents/GitHub/geoworm/data/hendrix/hendrix.shp'
shape_feature = cfeature.ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree())
ax = plt.subplot(4, 1, 2, projection=proj)#fig.add_subplot(gs[1],projection=proj)
ax.background_img(name='ne_gray',resolution='high')
ax.coastlines(resolution=res,zorder=2,lw=1.0)
ax.add_feature(cfeature.OCEAN.with_scale(res),zorder=1)
ax.add_feature(cfeature.LAKES.with_scale(res),zorder=1,edgecolor='None')
ax.add_feature(cfeature.RIVERS.with_scale(res),lw=0.25,zorder=1)#,edgecolor='k')
ax.add_feature(shape_feature, edgecolor='tab:red',zorder=2,facecolor='None',linestyle='-',linewidth=0.5)
plt.text(-100,40,'Hendrix and Bohlen, 2002',fontdict=dict(color='tab:red'))

plt.contourf(NA.lon,NA.lat,NA.bPL,[0.49,1],colors='g',alpha=0.3,transform=proj)
#plt.pcolormesh(NA.lon,NA.lat,(NA.bocc).where(NA.bocc),cmap=mpl.colors.ListedColormap([occol]),edgecolor=None,transform=proj,alpha=1)
#plt.pcolormesh(NA.lon,NA.lat,(NA.bocc).where(NA.bPL&NA.bocc),cmap=mpl.colors.ListedColormap([occol]),edgecolor=None,transform=proj,alpha=1)
#plt.pcolormesh(NA.lon,NA.lat,(NA.bocc).where(~NA.bPL&NA.bocc),cmap=mpl.colors.ListedColormap(['red']),edgecolor=None,transform=proj,alpha=1)
plt.scatter(occ.lon,occ.lat,color=occol,marker='o',s=1,zorder=3)
#cs = plt.contour(dat.lon,dat.lat,dat.MAP,[400],colors='cyan',linewidths=0.5,zorder=2)
plt.xlim(NA.lon.min(), NA.lon.max())
plt.ylim(NA.lat.min(), NA.lat.max())

ax2 = plt.subplot(4, 1, 3, projection=proj)#fig.add_subplot(gs[2],projection=proj)
ax2.background_img(name='ne_gray',resolution='high')
ax2.coastlines(resolution=res,zorder=2,lw=1.0)
ax2.add_feature(cfeature.OCEAN.with_scale(res),zorder=1)
ax2.add_feature(cfeature.LAKES.with_scale(res),zorder=1,edgecolor='None')
ax2.add_feature(cfeature.RIVERS.with_scale(res),lw=0.25,zorder=1)#,edgecolor='k')
plt.contourf(AU.lon,AU.lat,AU.bPL,[0.49,1],colors='g',alpha=0.3,transform=proj,zorder=1)
#plt.pcolormesh(AU.lon,AU.lat,(AU.bocc).where(AU.bocc),cmap=mpl.colors.ListedColormap([occol]),edgecolor=None,transform=proj,alpha=1,zorder=1)
#plt.pcolormesh(AU.lon,AU.lat,(AU.bocc).where(AU.bPL&AU.bocc),cmap=mpl.colors.ListedColormap([occol]),edgecolor=None,transform=proj,alpha=1)
#plt.pcolormesh(AU.lon,AU.lat,(AU.bocc).where(~AU.bPL&AU.bocc),cmap=mpl.colors.ListedColormap(['red']),edgecolor=None,transform=proj,alpha=1)
plt.scatter(occ[occ.src=='Abbott 1994'].lon,occ[occ.src=='Abbott 1994'].lat,color='tab:orange',marker='o',s=1,facecolor='None',zorder=2)
plt.scatter(occ[occ.src!='Abbott 1994'].lon,occ[occ.src!='Abbott 1994'].lat,color=occol,marker='o',s=1,zorder=2)

cs = plt.contour(dat.lon,dat.lat,dat.MAP,[400],colors='cyan',linewidths=0.5,zorder=3)
plt.text(120,-25,'400 $mm\, yr^{-1}$\n Abbott, 1994',fontdict=dict(color='cyan'))
#plt.clabel(cs, fontsize=6, fmt=r'%.d $mm\, y^{-1}$')
#plt.legend(frameon=False)
plt.xlim(AU.lon.min(), AU.lon.max())
plt.ylim(AU.lat.min(), AU.lat.max())

ax3 = plt.subplot(4, 1, 4, projection=proj)#fig.add_subplot(gs[3],projection=proj)
ax3.background_img(name='ne_gray',resolution='high')
ax3.coastlines(resolution=res,zorder=2,lw=1.0)
ax3.add_feature(cfeature.OCEAN.with_scale(res),zorder=1)
ax3.add_feature(cfeature.LAKES.with_scale(res),zorder=1,edgecolor='None')
ax3.add_feature(cfeature.RIVERS.with_scale(res),lw=0.25,zorder=1)#,edgecolor='k')
plt.contourf(EU.lon,EU.lat,EU.bPL,[0.49,1],colors='g',alpha=0.3,transform=proj)
plt.pcolormesh(EU.lon,EU.lat,(EU.bocc).where(EU.bocc),cmap=mpl.colors.ListedColormap([occol]),edgecolor=None,transform=proj,alpha=1)
#plt.pcolormesh(EU.lon,EU.lat,(EU.bocc).where(EU.bPL&EU.bocc),cmap=mpl.colors.ListedColormap([occol]),edgecolor=None,transform=proj,alpha=1)
#plt.pcolormesh(EU.lon,EU.lat,(EU.bocc).where(~EU.bPL&EU.bocc),cmap=mpl.colors.ListedColormap(['red']),edgecolor=None,transform=proj,alpha=1)
#cs = plt.contour(dat.lon,dat.lat,dat.MAP,[400],colors='cyan',linewidths=0.5,zorder=2)
plt.scatter(occ.lon,occ.lat,color=occol,marker='o',s=1,zorder=3)
plt.xlim(EU.lon.min(), EU.lon.max())
plt.ylim(EU.lat.min(), EU.lat.max())
#plt.xlabel('lon')
#plt.ylabel('lat')
#plt.tight_layout()
plt.savefig(r'./out/only_pressure_occupancy_{:}.png'.format(avg),dpi=300)
plt.show()
'''
#%%
bbNA = [slice(-135,-62),slice(22,56)]
bbAU = [slice(111,160),slice(-44,-10)]
bbEU = [slice(-20,-20+73+35),slice(0,72)]

NA = out.sel(lon=bbNA[0],lat=bbNA[1])
AU = out.sel(lon=bbAU[0],lat=bbAU[1])
EU = out.sel(lon=bbEU[0],lat=bbEU[1])

#proj = ccrs.LambertConformal()
res = '50m'
occol = 'b'
habitat = 'g'
halpha = 0.33
litcol = 'tab:red'
litcol0 = 'tab:orange'

fig = plt.figure(constrained_layout=True)
fig.set_size_inches(2*cwidth, 2.6*height)
#fig.set_size_inches(2*2*cwidth, 2*2.6*height)
gs0 = mpl.gridspec.GridSpec(2,1,wspace=0, hspace=0, figure=fig)
gs1 = mpl.gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[1,73/49.], subplot_spec=gs0[0],wspace=0.02, hspace=0)
gs2 = mpl.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[1],wspace=0.02, hspace=0)

ax = fig.add_subplot(gs2[0,0],projection=proj)#ax = plt.subplot(4, 1, 1, projection=proj)#
panel((0,1),(5,-2),'c')
#ax.text(-176,76,'c',weight='bold',fontsize=8)
ax.background_img(name='ne_gray',resolution='high')
ax.coastlines(resolution=res,zorder=2,lw=1)
ax.add_feature(cfeature.OCEAN.with_scale(res),zorder=1)
ax.add_feature(cfeature.LAKES.with_scale(res),zorder=1,edgecolor='None')
ax.add_feature(cfeature.RIVERS.with_scale(res),lw=0.25,zorder=1)
#plt.contourf(out.lon,out.lat,(out.PL<2e5).where(bfd&~bMAT),[0.5,1],colors='0.5')
#plt.pcolormesh(out.lon,out.lat,(out.PL<2e5).where(bfd&~bMAT),cmap='Greens')
#plt.pcolormesh(out.lon,out.lat,(out.PL<2e5).where(~bMAT),color='g',alpha=0.5)
plt.contourf(EU.lon,EU.lat,EU.b,[0.49,1],colors=habitat,alpha=halpha,transform=proj,zorder=1)
#plt.contour(out.lon,out.lat,out.occ,np.logspace(0,12,12,base=2),cmap='cmo.dense',zorder=1)

#plt.pcolormesh(out.lon,out.lat,out.b.where(out.b),color=habitat,alpha=halpha)
#plt.contourf(out.lon,out.lat,out.occ.where(out.occ>0),[0.5,1],colors='b',alpha=1)
#plt.pcolormesh(out.lon,out.lat,(out.bocc).where(out.bocc),cmap=mpl.colors.ListedColormap([occol]),edgecolor=None,transform=proj,alpha=1)
#cs = plt.contour(dat.lon,dat.lat,dat.MAP,[400],colors='cyan',linewidths=0.5,zorder=2)
#plt.pcolormesh(out.lon,out.lat,(out.bocc).where(out.bPL&out.bocc),cmap=mpl.colors.ListedColormap([occol]),edgecolor=None,transform=proj,alpha=1)
#plt.pcolormesh(out.lon,out.lat,(out.bocc).where(~out.bPL&out.bocc),cmap=mpl.colors.ListedColormap(['red']),edgecolor=None,transform=proj,alpha=1)
#plt.contourf(out.lon,out.lat,out.occ,[1,2,4,8,16,32,64],extend='max',cmap='cmo.ice',edgecolor='k',alpha=1,transform=proj)
plt.text(14.7-1.75,9.6+1.75,'G',weight='bold',fontsize=6,va='top',ha='left')
#plt.text(133-1.8,-23+1.8,'D',weight='bold',fontsize=6,va='top',ha='left')
ax.add_patch(mpl.patches.Rectangle((14.7-2,9.6-2),4,4,linewidth=1,edgecolor='k',facecolor='none',zorder=3))

occ0 = occ.groupby([occ.lon.round(1),occ.lat.round(1)]).size().rename('counts').reset_index()
#va0 = dat.sel_points(lon=occ0.lon.values,lat=occ0.lat.values,method='nearest')
#ou0 = out.sel_points(lon=occ0.lon.values,lat=occ0.lat.values,method='nearest')
#dfit = va0.to_dataframe()
#dfit['PL'] = ou0.PL.to_dataframe().PL
#dfit['A'] = 1e-6*ou0.A.to_dataframe().A
#dfit['counts'] = occ0.counts/dfit.A
#dfit['counts'].loc[dfit.counts>1e7] = np.nan
#dfit['NPP'].loc[dfit.NPP==65535] = np.nan
#dfit['NPP'] *= 0.1
#dfit['SND'].loc[dfit.SND==255] = np.nan
#dfit['SLT'].loc[dfit.SLT==255] = np.nan
#dfit['CLY'].loc[dfit.CLY==255] = np.nan
#dfit['AWC'].loc[dfit.AWC==255] = np.nan
#dfit['PHH'].loc[dfit.PHH==255] = np.nan
#dfit['PHH'] *= 0.1
#dfit['BLD'].loc[dfit.BLD==-32768] = np.nan
#dfit['CEC'].loc[dfit.CEC==-32768] = np.nan
#dfit['ORC'].loc[dfit.ORC==-32768] = np.nan
#dfit['DRY'] = 1./dfit['DRY']
#dfit = dfit.drop(columns=['lon','lat','band','level','LAN','A']).dropna()
#modsp = sm.GLM(endog=dfit.counts,exog=sm.add_constant(dfit.drop(columns=['counts'])),family=sm.families.Poisson()).fit()

#import hdbscan as hdb
#
#hds = hdb.HDBSCAN(metric='haversine')
#clu = hds.fit(np.deg2rad(occ0[['lon','lat']]))
#occ0['label'] = clu.labels_
#occ0 = occ0.groupby('label').mean()

#occ0['b'] = out.b.sel_points(lon=occ0.lon.values,lat=occ0.lat.values,method='nearest')

#plt.scatter(occ0.lon,occ0.lat,c=gaussian_kde(occ0.counts)(occ0.counts),s=2,lw=0.25,marker='o',zorder=3,cmap=mpl.colors.ListedColormap([(0,0,1,0.5),(0.5,0.5,0.5,0.25),(1,1,1,0.0)]),edgecolor=(0,0,1,0.25))
plt.scatter(occ0.lon,occ0.lat,
            color=occol,
#            c=occ0.b,
#            cmap=mpl.colors.ListedColormap(['0.75',(0,1,0,halpha)]),
#            s=np.clip(occ0.counts,3,10),
            s=3,
            edgecolor='k',
            lw=0.25,
            marker='o',
            zorder=3)
plt.xlim(EU.lon.min(), EU.lon.max())
plt.ylim(EU.lat.min(), EU.lat.max())

rectbg = plt.Rectangle((-178, -60), 87, 53, facecolor=cfeature.OCEAN.kwargs['facecolor'],zorder=4)
rect = plt.Rectangle((-178, -60), 87, 53, facecolor='w',alpha=0.15,zorder=4)
ax.add_patch(rectbg)
ax.add_patch(rect)

out['Ater'] = out.A.where(dat.LAN&(dat.lat>-60))
out['Ahab'] = out.Ater.where(out.b)
out['Aihab'] = out.Ater.where(~out.b)
out['Anoc'] = out.Ater.where(out.bocc&~out.b)
out['Apoc'] = out.Ater.where(out.bocc&out.b)

fAhab = out.Ahab.sum()/out.Ater.sum()
fAihab = out.Aihab.sum()/out.Ater.sum()

fAnoc = out.Anoc.sum()/(out.Apoc.sum()+out.Anoc.sum())
fApoc = out.Apoc.sum()/(out.Apoc.sum()+out.Anoc.sum())
#idx = np.argwhere(dat.LAN.values)
#irnd = np.random.randint(0,len(idx),size=np.count_nonzero(out.bocc))

#out['nF'] = (~out.baM.astype(int)+~out.bMAT+~out.bpH+~out.bSND)
#
#out['Ap0'] = out.Ater.where(out.bocc&~out.bPL&~out.baM&~out.bMAT&~out.bpH&~out.bSND)
#out['Ap1'] = out.Ater.where(out.bocc&(out.nF==1))
#out['Ap2'] = out.Ater.where(out.bocc&(out.nF==2))
#out['Ap3'] = out.Ater.where(out.bocc&(out.nF==3))
#out['Ap4'] = out.Ater.where(out.bocc&(out.nF==4))
#out['Ap1'] = out.Ater.where(out.bocc&~out.bPL&out.baM&~out.bMAT&~out.bpH&~out.bSND)
#out['Ap2'] = out.Ater.where(out.bocc&~out.bPL&~out.baM&out.bMAT&~out.bpH&~out.bSND)
#out['Ap3'] = out.Ater.where(out.bocc&~out.bPL&~out.baM&~out.bMAT&out.bpH&~out.bSND)
#out['Ap4'] = out.Ater.where(out.bocc&~out.bPL&~out.baM&~out.bMAT&~out.bpH&out.bSND)
#out['An0'] = out.Ater.where(out.bocc&~out.bPL&~out.baM&~out.bMAT&~out.bpH&~out.bSND)

N = int(out.bocc.where(~np.isnan(out.Ater)).sum())#np.count_nonzero(out.bocc)

#gs3 = mpl.gridspec.GridSpecFromSubplotSpec(2, 3, width_ratios=[2,0.5,0.5], subplot_spec=gs2[0,1],hspace=0,wspace=0)
gs3 = mpl.gridspec.GridSpecFromSubplotSpec(3, 3, height_ratios=[4,0.25,0.5], width_ratios=[1,2,1], subplot_spec=gs2[0,1],hspace=0,wspace=0)

#plt.Rectangle((0.02, 0.05), 0.045*cwidth, 0.045*cwidth)
#axx = fig.add_subplot(gs3[1,0])#fig.add_axes([0.02, 0.05, 0.045*cwidth, 0.045*cwidth])
'''
ax = axx
panel((0,1),(5,-2),'e')
#gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[4, 1],wspace=0) 
#plt.subplot(gs[0])
#plt.subplot(121)
#plt.title('% Terrestrial area',fontsize=6)
plt.pie([fAihab,fAhab],
        colors=['0.75',(0,1,0,halpha)],
        autopct='%1.0f%%',
        startangle=90,
        wedgeprops={'edgecolor':'k'},
        textprops={'fontsize': 6})#,labels=['excluding','hospitable'])
plt.axis('equal')
plt.ylabel('% Terrestrial area',fontsize=6,labelpad=-9)
'''
#leg = plt.legend(fancybox=False)
#leg.get_frame().set_linewidth(0.0)
#leg.get_frame().set_alpha(0.15)

#plt.subplot(gs[1])
#plt.bar(['FN$_b$','FN','TP$_b$','TP'],[np.mean(FNb),FN,np.mean(TPb),TP],color=['0.5','b','0.5','b'],yerr=[np.std(FNb),0,np.std(TPb),0])
#plt.subplot(122)
#axx = fig.add_axes([0.19, 0.05, 0.01*cwidth, 0.045*cwidth])
#axx.set_facecolor('None')

axx = fig.add_subplot(gs3[2,1])
ax=axx
panel((0,1),(-10,-2),'e')
plt.barh(0,fApoc,edgecolor=occol, color=(0,1,0,halpha),lw=1)
plt.barh(0,fAnoc,edgecolor=occol, color='0.75', lw=1, left=fApoc)
#plt.bar(0,fApoc,edgecolor=occol, color=(0,1,0,halpha),width=0.5,lw=1)
#plt.bar(0,fAnoc,edgecolor=occol, color='0.75', width=0.5, lw=1, bottom=fApoc)
plt.text(fApoc/2.,0,'{:}% TP'.format(int(fApoc*100)),
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=6)
plt.text(fApoc+fAnoc/2.,0,'{:}% \n FN'.format(int(fAnoc*100)),
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=6)
#plt.text(0.3,fApoc/2.,'TP',
#         horizontalalignment='left',
#         verticalalignment='center',
#         fontsize=6)#,weight='bold')
#plt.text(0.3,fApoc+fAnoc/2.,'FN',
#         horizontalalignment='left',
#         verticalalignment='center',
#         fontsize=6)#,weight='bold')
#plt.hist([np.random.choice(H.ravel(),Nsmp).astype(int),poc.astype(int)],bins=2,colors=['0.5','w'])
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
#plt.axis('off')
#plt.xlabel('Occurence (n={:})'.format(N),fontsize=6)
#plt.show()

ax = fig.add_subplot(gs3[0,:],projection=proj)#fig.add_axes([0.02, 0.05, 0.045*cwidth, 0.045*cwidth])
panel((0,1),(5,5),'d')
ax.coastlines(resolution=res,zorder=-1,lw=0.5)
#plt.pcolormesh(dat.lon,dat.lat,dat.LAN)
plt.gca().outline_patch.set_visible(False)
plt.hexbin(occ.lon,occ.lat,mincnt=1,norm=mpl.colors.LogNorm(),cmap='cmo.ice')

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axins = inset_axes(ax,width='50%',height='5%',
                   loc='lower center',
                   bbox_to_anchor=(0., -0.05, 1, 1),
                   bbox_transform=ax.transAxes,
                   borderpad=0)
cb = plt.colorbar(cax=axins,pad=0,orientation='horizontal')
cb.set_label(label='Earthworm occurrence (n={:})'.format(N),fontsize=6)


'''
import scipy.ndimage as ndi
Hs = out.b.values.astype(bool)
ocs = out.bocc.values.astype(bool)
scales = np.logspace(0,-1.8,50)
res = []
for f in scales:
    Hi = ndi.zoom(Hs,f,mode='wrap',order=0)
    oci = ndi.zoom(ocs,f,mode='wrap',order=0)
    TPi = np.count_nonzero(Hi&oci)
    FNi = np.count_nonzero(~Hi&oci)
#    plt.imshow(Hi.astype(int)+oci)
#    plt.show()
    res.append(float(FNi)/(TPi+FNi))
    print f
plt.plot(scales,res)
plt.show()
'''

#tl = []
#for s in np.arange(1,100):
#    a = np.histogram2d(occ.lon,occ.lat,bins=[np.linspace(-180,180,36*s),np.linspace(-90,90,18*s)])[0]
#    tl.append([a.mean(),a.std()**2])
#    print s, a.max()
#
#mm,vv = np.array(tl).T
#k = mm**2/(vv-mm)
#
#TL = sm.OLS(np.log10(vv),sm.add_constant(np.log10(mm))).fit()
#
#kk = mm/(10**TL.params[0]*mm**(TL.params[1]-1)-1)
#
#plt.plot(mm,vv,marker='o')
#plt.plot(mm,10**TL.predict(sm.add_constant(np.log10(mm))))
#plt.loglog()

#axx.hist([1e-3*dff.PL[dff.b],1e-3*dff.PL[~dff.b]],np.linspace(1,4e2),color=[habitat,'0.5'],alpha=halpha,stacked=True)
#axx.hist([1e-3*out.PL.values[out.bocc&out.b],1e-3*out.PL.values[out.bocc&~out.b]],
#         bins=np.linspace(1,4e2),
#         weights=[out.Apoc.values[out.bocc&out.b]/out.Ater.sum().values,out.Anoc.values[out.bocc&~out.b]/out.Ater.sum().values],
#         stacked=True)
#axx.set_ylabel('Occurrences')
#axx.set_xlabel(r'Limiting pressure $P_L \,(kPa)$')
#axx.pie([out.Ap0.sum(),out.Ap1.sum(),out.Ap2.sum(),out.Ap3.sum(),out.Ap4.sum(),out.An0.sum()],colors=[habitat,caM,cmat,cpH,csnd,'0.5'])
#axx.pie([out.Ap0.sum(),out.Ap1.sum(),out.Ap2.sum(),out.Ap3.sum(),out.Ap4.sum()],colors=[habitat,caM,cmat,cpH,csnd])
#axx.axis('equal')

fname = 'C:/Users/bickels/Documents/GitHub/geoworm/data/hendrix/hendrix.shp'
shape_feature = cfeature.ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree())
ax = fig.add_subplot(gs1[0,1],projection=proj)
panel((0,0.1),(5,-2),'b')
#ax.text(-133,24,'b',weight='bold',fontsize=8)
ax.background_img(name='ne_gray',resolution='high')
ax.coastlines(resolution=res,zorder=2,lw=1.0)
ax.add_feature(cfeature.OCEAN.with_scale(res),zorder=1)
ax.add_feature(cfeature.LAKES.with_scale(res),zorder=1,edgecolor='None')
ax.add_feature(cfeature.RIVERS.with_scale(res),lw=0.25,zorder=2)#,edgecolor='k')
ax.add_feature(shape_feature, edgecolor=litcol,zorder=2,facecolor='k',alpha=0.1,linewidth=0)
ax.add_feature(shape_feature, edgecolor=litcol,zorder=2,facecolor='None',linestyle='-',linewidth=1)
#plt.text(-100,40,'Hendrix and Bohlen, 2002',fontdict=dict(color=litcol,fontsize=7))

plt.contourf(NA.lon,NA.lat,NA.b,[0.49,1],colors=habitat,alpha=halpha,transform=proj,zorder=1)
#plt.pcolormesh(NA.lon,NA.lat,(NA.bocc).where(NA.bocc),cmap=mpl.colors.ListedColormap([occol]),edgecolor=None,transform=proj,alpha=1)
#plt.pcolormesh(NA.lon,NA.lat,(NA.bocc).where(NA.bPL&NA.bocc),cmap=mpl.colors.ListedColormap([occol]),edgecolor=None,transform=proj,alpha=1)
#plt.pcolormesh(NA.lon,NA.lat,(NA.bocc).where(~NA.bPL&NA.bocc),cmap=mpl.colors.ListedColormap(['red']),edgecolor=None,transform=proj,alpha=1)
plt.scatter(occ.lon,occ.lat,color=occol,marker='o',s=3,zorder=3,label=None,edgecolor='k',lw=0.25)
#cs = plt.contour(dat.lon,dat.lat,dat.MAP,[400],colors='cyan',linewidths=0.5,zorder=2)
#plt.plot([],[],color=litcol,lw=1,label='Hendrix and\n Bohlen, 2002')
#leg = plt.legend(fancybox=False,bbox_to_anchor=(0.29,0), loc="lower right")
#leg.get_frame().set_linewidth(0.0)
#leg.get_frame().set_alpha(0.15)
plt.xlim(NA.lon.min(), NA.lon.max())
plt.ylim(NA.lat.min(), NA.lat.max())

ax2 = fig.add_subplot(gs1[0,0],projection=proj)
ax=ax2
panel((0,0.1),(5,-2),'a')
#ax2.text(113,-42,'a',weight='bold',fontsize=8)
ax2.background_img(name='ne_gray',resolution='high')
ax2.coastlines(resolution=res,zorder=2,lw=1.0)
ax2.add_feature(cfeature.OCEAN.with_scale(res),zorder=1)
ax2.add_feature(cfeature.LAKES.with_scale(res),zorder=1,edgecolor='None')
ax2.add_feature(cfeature.RIVERS.with_scale(res),lw=0.25,zorder=2)#,edgecolor='k')
plt.contourf(AU.lon,AU.lat,AU.b,[0.49,1],colors=habitat,alpha=halpha,transform=proj,zorder=1)
#plt.pcolormesh(AU.lon,AU.lat,(AU.bocc).where(AU.bocc),cmap=mpl.colors.ListedColormap([occol]),edgecolor=None,transform=proj,alpha=1,zorder=1)
#plt.pcolormesh(AU.lon,AU.lat,(AU.bocc).where(AU.bPL&AU.bocc),cmap=mpl.colors.ListedColormap([occol]),edgecolor=None,transform=proj,alpha=1)
#plt.pcolormesh(AU.lon,AU.lat,(AU.bocc).where(~AU.bPL&AU.bocc),cmap=mpl.colors.ListedColormap(['red']),edgecolor=None,transform=proj,alpha=1)
plt.scatter(occ[occ.src=='Abbott 1994'].lon,occ[occ.src=='Abbott 1994'].lat,color=litcol0,marker='o',s=3,edgecolor='k',lw=0.25,zorder=3,label='Abbott, 1994')
plt.scatter(occ[occ.src!='Abbott 1994'].lon,occ[occ.src!='Abbott 1994'].lat,color=occol,marker='o',s=3,edgecolor='k',lw=0.25,zorder=3,label='GBIF')
plt.plot([],[],color='cyan',lw=1,label='Mean annual precipitation')
plt.plot([],[],color=litcol,lw=1,label='Hendrix and Bohlen, 2002')
#leg = plt.legend(fancybox=False,bbox_to_anchor=(0.29,0), loc="lower right")
leg = plt.legend(fancybox=False,bbox_to_anchor=(0.5,-0.15), loc="lower left",ncol=4,handletextpad=0.05,markerscale=2)
leg.get_frame().set_linewidth(0.0)
leg.get_frame().set_alpha(0.15)

cs = plt.contour(dat.lon,dat.lat,dat.MAP,[400],colors='cyan',linewidths=0.5,zorder=2)
plt.text(120,-22,'400 $mm\, y^{-1}$',fontdict=dict(color='cyan',fontsize=6))

plt.text(133-1.75,-23+1.75,'D',weight='bold',fontsize=6,va='top',ha='left')
ax2.add_patch(mpl.patches.Rectangle((133-2,-23-2),4,4,linewidth=1,edgecolor='k',facecolor='none',zorder=3))

#plt.clabel(cs, fontsize=6, fmt=r'%.d $mm\, y^{-1}$')
#plt.legend(frameon=False)
#plt.text(133-7,-23+2,'D',weight='bold',fontsize=8)
#ax2.add_patch(mpl.patches.Rectangle((133-2,-23-2),4,4,linewidth=1,edgecolor='k',facecolor='none',zorder=3))

plt.xlim(AU.lon.min(), AU.lon.max())
plt.ylim(AU.lat.min(), AU.lat.max())
plt.tight_layout(h_pad=0.01,pad=0.05)
plt.savefig(r'./out/all_factors_occupancy_{:}.png'.format(avg),dpi=300)
plt.show()

#%%
'''
bbNA = [slice(-135,-62),slice(22,56)]
bbAU = [slice(111,160),slice(-44,-10)]
bbEU = [slice(-14,37),slice(32,64)]

NA = out.sel(lon=bbNA[0],lat=bbNA[1])
AU = out.sel(lon=bbAU[0],lat=bbAU[1])
EU = out.sel(lon=bbEU[0],lat=bbEU[1])

#proj = ccrs.LambertConformal()
res = '50m'
occol = 'b'
habitat = 'g'
halpha = 0.33
litcol = 'tab:red'
litcol0 = 'tab:orange'

fig = plt.figure(constrained_layout=True)
fig.set_size_inches(2*cwidth, 2.6*height)
#fig.set_size_inches(2*2*cwidth, 2*2.6*height)
gs = mpl.gridspec.GridSpec(2,2, width_ratios=[1,73/49.])

ax = fig.add_subplot(gs[1,:],projection=proj)#ax = plt.subplot(4, 1, 1, projection=proj)#
ax.text(-176,76,'c',weight='bold',fontsize=8)
ax.background_img(name='ne_gray',resolution='high')
ax.coastlines(resolution=res,zorder=2,lw=1)
ax.add_feature(cfeature.OCEAN.with_scale(res),zorder=1)
ax.add_feature(cfeature.LAKES.with_scale(res),zorder=1,edgecolor='None')
ax.add_feature(cfeature.RIVERS.with_scale(res),lw=0.25,zorder=1)
#plt.contourf(out.lon,out.lat,(out.PL<2e5).where(bfd&~bMAT),[0.5,1],colors='0.5')
#plt.pcolormesh(out.lon,out.lat,(out.PL<2e5).where(bfd&~bMAT),cmap='Greens')
#plt.pcolormesh(out.lon,out.lat,(out.PL<2e5).where(~bMAT),color='g',alpha=0.5)
plt.contourf(out.lon,out.lat,out.b,[0.49,1],colors=habitat,alpha=halpha,transform=proj,zorder=1)
#plt.contour(out.lon,out.lat,out.occ,np.logspace(0,12,12,base=2),cmap='cmo.dense',zorder=1)

#plt.pcolormesh(out.lon,out.lat,out.b.where(out.b),color=habitat,alpha=halpha)
#plt.contourf(out.lon,out.lat,out.occ.where(out.occ>0),[0.5,1],colors='b',alpha=1)
#plt.pcolormesh(out.lon,out.lat,(out.bocc).where(out.bocc),cmap=mpl.colors.ListedColormap([occol]),edgecolor=None,transform=proj,alpha=1)
#cs = plt.contour(dat.lon,dat.lat,dat.MAP,[400],colors='cyan',linewidths=0.5,zorder=2)
#plt.pcolormesh(out.lon,out.lat,(out.bocc).where(out.bPL&out.bocc),cmap=mpl.colors.ListedColormap([occol]),edgecolor=None,transform=proj,alpha=1)
#plt.pcolormesh(out.lon,out.lat,(out.bocc).where(~out.bPL&out.bocc),cmap=mpl.colors.ListedColormap(['red']),edgecolor=None,transform=proj,alpha=1)
#plt.contourf(out.lon,out.lat,out.occ,[1,2,4,8,16,32,64],extend='max',cmap='cmo.ice',edgecolor='k',alpha=1,transform=proj)
plt.text(14.7-7,9.6+2,'G',weight='bold',fontsize=8)
ax.add_patch(mpl.patches.Rectangle((14.7-2,9.6-2),4,4,linewidth=1,edgecolor='k',facecolor='none',zorder=3))
plt.text(133-7,-23+2,'D',weight='bold',fontsize=8)
ax.add_patch(mpl.patches.Rectangle((133-2,-23-2),4,4,linewidth=1,edgecolor='k',facecolor='none',zorder=3))

occ0 = occ.groupby([occ.lon.round(1),occ.lat.round(1)]).size().rename('counts').reset_index()
#va0 = dat.sel_points(lon=occ0.lon.values,lat=occ0.lat.values,method='nearest')
#ou0 = out.sel_points(lon=occ0.lon.values,lat=occ0.lat.values,method='nearest')
#dfit = va0.to_dataframe()
#dfit['PL'] = ou0.PL.to_dataframe().PL
#dfit['A'] = 1e-6*ou0.A.to_dataframe().A
#dfit['counts'] = occ0.counts/dfit.A
#dfit['counts'].loc[dfit.counts>1e7] = np.nan
#dfit['NPP'].loc[dfit.NPP==65535] = np.nan
#dfit['NPP'] *= 0.1
#dfit['SND'].loc[dfit.SND==255] = np.nan
#dfit['SLT'].loc[dfit.SLT==255] = np.nan
#dfit['CLY'].loc[dfit.CLY==255] = np.nan
#dfit['AWC'].loc[dfit.AWC==255] = np.nan
#dfit['PHH'].loc[dfit.PHH==255] = np.nan
#dfit['PHH'] *= 0.1
#dfit['BLD'].loc[dfit.BLD==-32768] = np.nan
#dfit['CEC'].loc[dfit.CEC==-32768] = np.nan
#dfit['ORC'].loc[dfit.ORC==-32768] = np.nan
#dfit['DRY'] = 1./dfit['DRY']
#dfit = dfit.drop(columns=['lon','lat','band','level','LAN','A']).dropna()
#modsp = sm.GLM(endog=dfit.counts,exog=sm.add_constant(dfit.drop(columns=['counts'])),family=sm.families.Poisson()).fit()

#import hdbscan as hdb
#
#hds = hdb.HDBSCAN(metric='haversine')
#clu = hds.fit(np.deg2rad(occ0[['lon','lat']]))
#occ0['label'] = clu.labels_
#occ0 = occ0.groupby('label').mean()

#occ0['b'] = out.b.sel_points(lon=occ0.lon.values,lat=occ0.lat.values,method='nearest')

#plt.scatter(occ0.lon,occ0.lat,c=gaussian_kde(occ0.counts)(occ0.counts),s=2,lw=0.25,marker='o',zorder=3,cmap=mpl.colors.ListedColormap([(0,0,1,0.5),(0.5,0.5,0.5,0.25),(1,1,1,0.0)]),edgecolor=(0,0,1,0.25))
plt.scatter(occ0.lon,occ0.lat,
            color=occol,
#            c=occ0.b,
#            cmap=mpl.colors.ListedColormap(['0.75',(0,1,0,halpha)]),
#            s=np.clip(occ0.counts,3,10),
            s=3,
            edgecolor='k',
            lw=0.25,
            marker='o',
            zorder=3)
rectbg = plt.Rectangle((-178, -60), 87, 53, facecolor=cfeature.OCEAN.kwargs['facecolor'],zorder=4)
rect = plt.Rectangle((-178, -60), 87, 53, facecolor='w',alpha=0.15,zorder=4)
ax.add_patch(rectbg)
ax.add_patch(rect)

out['Ater'] = out.A.where(dat.LAN&(dat.lat>-60))
out['Ahab'] = out.Ater.where(out.b)
out['Aihab'] = out.Ater.where(~out.b)
out['Anoc'] = out.Ater.where(out.bocc&~out.b)
out['Apoc'] = out.Ater.where(out.bocc&out.b)

fAhab = out.Ahab.sum()/out.Ater.sum()
fAihab = out.Aihab.sum()/out.Ater.sum()

fAnoc = out.Anoc.sum()/(out.Apoc.sum()+out.Anoc.sum())
fApoc = out.Apoc.sum()/(out.Apoc.sum()+out.Anoc.sum())
#idx = np.argwhere(dat.LAN.values)
#irnd = np.random.randint(0,len(idx),size=np.count_nonzero(out.bocc))

#out['nF'] = (~out.baM.astype(int)+~out.bMAT+~out.bpH+~out.bSND)
#
#out['Ap0'] = out.Ater.where(out.bocc&~out.bPL&~out.baM&~out.bMAT&~out.bpH&~out.bSND)
#out['Ap1'] = out.Ater.where(out.bocc&(out.nF==1))
#out['Ap2'] = out.Ater.where(out.bocc&(out.nF==2))
#out['Ap3'] = out.Ater.where(out.bocc&(out.nF==3))
#out['Ap4'] = out.Ater.where(out.bocc&(out.nF==4))
#out['Ap1'] = out.Ater.where(out.bocc&~out.bPL&out.baM&~out.bMAT&~out.bpH&~out.bSND)
#out['Ap2'] = out.Ater.where(out.bocc&~out.bPL&~out.baM&out.bMAT&~out.bpH&~out.bSND)
#out['Ap3'] = out.Ater.where(out.bocc&~out.bPL&~out.baM&~out.bMAT&out.bpH&~out.bSND)
#out['Ap4'] = out.Ater.where(out.bocc&~out.bPL&~out.baM&~out.bMAT&~out.bpH&out.bSND)
#out['An0'] = out.Ater.where(out.bocc&~out.bPL&~out.baM&~out.bMAT&~out.bpH&~out.bSND)

N = int(out.bocc.where(~np.isnan(out.Ater)).sum())#np.count_nonzero(out.bocc)


plt.Rectangle((0.02, 0.05), 0.045*cwidth, 0.045*cwidth)
axx = fig.add_axes([0.02, 0.05, 0.045*cwidth, 0.045*cwidth])
#gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[4, 1],wspace=0) 
#plt.subplot(gs[0])
#plt.subplot(121)
#plt.title('% Terrestrial area',fontsize=6)
plt.pie([fAihab,fAhab],
        colors=['0.75',(0,1,0,halpha)],
        autopct='%1.0f%%',
        startangle=90,
        wedgeprops={'edgecolor':'k'},
        textprops={'fontsize': 6})#,labels=['excluding','hospitable'])
plt.axis('equal')
plt.ylabel('% Terrestrial area',fontsize=6,labelpad=-9)
#leg = plt.legend(fancybox=False)
#leg.get_frame().set_linewidth(0.0)
#leg.get_frame().set_alpha(0.15)

#plt.subplot(gs[1])
#plt.bar(['FN$_b$','FN','TP$_b$','TP'],[np.mean(FNb),FN,np.mean(TPb),TP],color=['0.5','b','0.5','b'],yerr=[np.std(FNb),0,np.std(TPb),0])
#plt.subplot(122)
axx = fig.add_axes([0.19, 0.05, 0.01*cwidth, 0.045*cwidth])
axx.set_facecolor('None')
plt.bar(0,fApoc,edgecolor=occol, color=(0,1,0,halpha),width=0.5,lw=1)
plt.bar(0,fAnoc,edgecolor=occol, color='0.75', width=0.5, lw=1, bottom=fApoc)
plt.text(0,fApoc/2.,'{:}%'.format(int(fApoc*100)),
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=6)
plt.text(0,fApoc+fAnoc/2.,'{:}%'.format(int(fAnoc*100)),
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=6)
plt.text(0.3,fApoc/2.,'TP',
         horizontalalignment='left',
         verticalalignment='center',
         fontsize=6)#,weight='bold')
plt.text(0.3,fApoc+fAnoc/2.,'FN',
         horizontalalignment='left',
         verticalalignment='center',
         fontsize=6)#,weight='bold')
#plt.hist([np.random.choice(H.ravel(),Nsmp).astype(int),poc.astype(int)],bins=2,colors=['0.5','w'])
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
#plt.axis('off')
plt.ylabel('Occurence (n={:})'.format(N),fontsize=6)
#plt.show()

import scipy.ndimage as ndi
Hs = out.b.values.astype(bool)
ocs = out.bocc.values.astype(bool)
scales = np.logspace(0,-1.8,50)
res = []
for f in scales:
    Hi = ndi.zoom(Hs,f,mode='wrap',order=0)
    oci = ndi.zoom(ocs,f,mode='wrap',order=0)
    TPi = np.count_nonzero(Hi&oci)
    FNi = np.count_nonzero(~Hi&oci)
#    plt.imshow(Hi.astype(int)+oci)
#    plt.show()
    res.append(float(FNi)/(TPi+FNi))
    print f
plt.plot(scales,res)
plt.show()

#tl = []
#for s in np.arange(1,100):
#    a = np.histogram2d(occ.lon,occ.lat,bins=[np.linspace(-180,180,36*s),np.linspace(-90,90,18*s)])[0]
#    tl.append([a.mean(),a.std()**2])
#    print s, a.max()
#
#mm,vv = np.array(tl).T
#k = mm**2/(vv-mm)
#
#TL = sm.OLS(np.log10(vv),sm.add_constant(np.log10(mm))).fit()
#
#kk = mm/(10**TL.params[0]*mm**(TL.params[1]-1)-1)
#
#plt.plot(mm,vv,marker='o')
#plt.plot(mm,10**TL.predict(sm.add_constant(np.log10(mm))))
#plt.loglog()

#axx.hist([1e-3*dff.PL[dff.b],1e-3*dff.PL[~dff.b]],np.linspace(1,4e2),color=[habitat,'0.5'],alpha=halpha,stacked=True)
#axx.hist([1e-3*out.PL.values[out.bocc&out.b],1e-3*out.PL.values[out.bocc&~out.b]],
#         bins=np.linspace(1,4e2),
#         weights=[out.Apoc.values[out.bocc&out.b]/out.Ater.sum().values,out.Anoc.values[out.bocc&~out.b]/out.Ater.sum().values],
#         stacked=True)
#axx.set_ylabel('Occurrences')
#axx.set_xlabel(r'Limiting pressure $P_L \,(kPa)$')
#axx.pie([out.Ap0.sum(),out.Ap1.sum(),out.Ap2.sum(),out.Ap3.sum(),out.Ap4.sum(),out.An0.sum()],colors=[habitat,caM,cmat,cpH,csnd,'0.5'])
#axx.pie([out.Ap0.sum(),out.Ap1.sum(),out.Ap2.sum(),out.Ap3.sum(),out.Ap4.sum()],colors=[habitat,caM,cmat,cpH,csnd])
#axx.axis('equal')

fname = 'C:/Users/bickels/Documents/GitHub/geoworm/data/hendrix/hendrix.shp'
shape_feature = cfeature.ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree())
ax = fig.add_subplot(gs[0,1],projection=proj)
ax.text(-133,24,'b',weight='bold',fontsize=8)
ax.background_img(name='ne_gray',resolution='high')
ax.coastlines(resolution=res,zorder=2,lw=1.0)
ax.add_feature(cfeature.OCEAN.with_scale(res),zorder=1)
ax.add_feature(cfeature.LAKES.with_scale(res),zorder=1,edgecolor='None')
ax.add_feature(cfeature.RIVERS.with_scale(res),lw=0.25,zorder=2)#,edgecolor='k')
ax.add_feature(shape_feature, edgecolor=litcol,zorder=2,facecolor='k',alpha=0.1,linewidth=0)
ax.add_feature(shape_feature, edgecolor=litcol,zorder=2,facecolor='None',linestyle='-',linewidth=1)
#plt.text(-100,40,'Hendrix and Bohlen, 2002',fontdict=dict(color=litcol,fontsize=7))

plt.contourf(NA.lon,NA.lat,NA.b,[0.49,1],colors=habitat,alpha=halpha,transform=proj,zorder=1)
#plt.pcolormesh(NA.lon,NA.lat,(NA.bocc).where(NA.bocc),cmap=mpl.colors.ListedColormap([occol]),edgecolor=None,transform=proj,alpha=1)
#plt.pcolormesh(NA.lon,NA.lat,(NA.bocc).where(NA.bPL&NA.bocc),cmap=mpl.colors.ListedColormap([occol]),edgecolor=None,transform=proj,alpha=1)
#plt.pcolormesh(NA.lon,NA.lat,(NA.bocc).where(~NA.bPL&NA.bocc),cmap=mpl.colors.ListedColormap(['red']),edgecolor=None,transform=proj,alpha=1)
plt.scatter(occ.lon,occ.lat,color=occol,marker='o',s=3,zorder=3,label=None,edgecolor='k',lw=0.25)
#cs = plt.contour(dat.lon,dat.lat,dat.MAP,[400],colors='cyan',linewidths=0.5,zorder=2)
plt.plot([],[],color=litcol,lw=1,label='Hendrix and\n Bohlen, 2002')
leg = plt.legend(fancybox=False,bbox_to_anchor=(0.29,0), loc="lower right")
leg.get_frame().set_linewidth(0.0)
leg.get_frame().set_alpha(0.15)
plt.xlim(NA.lon.min(), NA.lon.max())
plt.ylim(NA.lat.min(), NA.lat.max())

ax2 = fig.add_subplot(gs[0,0],projection=proj)
ax2.text(113,-42,'a',weight='bold',fontsize=8)
ax2.background_img(name='ne_gray',resolution='high')
ax2.coastlines(resolution=res,zorder=2,lw=1.0)
ax2.add_feature(cfeature.OCEAN.with_scale(res),zorder=1)
ax2.add_feature(cfeature.LAKES.with_scale(res),zorder=1,edgecolor='None')
ax2.add_feature(cfeature.RIVERS.with_scale(res),lw=0.25,zorder=2)#,edgecolor='k')
plt.contourf(AU.lon,AU.lat,AU.b,[0.49,1],colors=habitat,alpha=halpha,transform=proj,zorder=1)
#plt.pcolormesh(AU.lon,AU.lat,(AU.bocc).where(AU.bocc),cmap=mpl.colors.ListedColormap([occol]),edgecolor=None,transform=proj,alpha=1,zorder=1)
#plt.pcolormesh(AU.lon,AU.lat,(AU.bocc).where(AU.bPL&AU.bocc),cmap=mpl.colors.ListedColormap([occol]),edgecolor=None,transform=proj,alpha=1)
#plt.pcolormesh(AU.lon,AU.lat,(AU.bocc).where(~AU.bPL&AU.bocc),cmap=mpl.colors.ListedColormap(['red']),edgecolor=None,transform=proj,alpha=1)
plt.scatter(occ[occ.src=='Abbott 1994'].lon,occ[occ.src=='Abbott 1994'].lat,color=litcol0,marker='o',s=3,edgecolor='k',lw=0.25,zorder=3,label='Abbott, 1994')
plt.scatter(occ[occ.src!='Abbott 1994'].lon,occ[occ.src!='Abbott 1994'].lat,color=occol,marker='o',s=3,edgecolor='k',lw=0.25,zorder=3,label='$Lumbricidae$ (GBIF)')
leg = plt.legend(fancybox=False,handletextpad=0.1,bbox_to_anchor=(0.5,0), loc="lower right")
leg.get_frame().set_linewidth(0.0)
leg.get_frame().set_alpha(0.15)

cs = plt.contour(dat.lon,dat.lat,dat.MAP,[400],colors='cyan',linewidths=0.5,zorder=2)
plt.text(120,-22,'400 $mm\, yr^{-1}$',fontdict=dict(color='cyan',fontsize=6))
#plt.clabel(cs, fontsize=6, fmt=r'%.d $mm\, y^{-1}$')
#plt.legend(frameon=False)
#plt.text(133-7,-23+2,'D',weight='bold',fontsize=8)
#ax2.add_patch(mpl.patches.Rectangle((133-2,-23-2),4,4,linewidth=1,edgecolor='k',facecolor='none',zorder=3))

plt.xlim(AU.lon.min(), AU.lon.max())
plt.ylim(AU.lat.min(), AU.lat.max())
plt.tight_layout(h_pad=0.01,pad=0.05)
plt.savefig(r'./out/all_factors_occupancy_{:}.png'.format(avg),dpi=300)
plt.show()
'''
#%% bootstrap
ha = out.b.where(dat.LAN&(dat.lat>-60)).values
H = ha[~np.isnan(ha)]


#w = w[~np.isnan(w)]
#poc = ha[out.bocc]
#poc = poc[~np.isnan(poc)]
#fsmp = 0.1
fsmps = np.logspace(-4,0,5,base=2)
nboot = 5000

bo = out.bocc.where(~np.isnan(out.Ater)).values>0
ibocc = np.argwhere(bo)

bo = (out.occ/out.A).values#.astype(int)
nn = bo+np.roll(bo,1,axis=0)+np.roll(bo,-1,axis=0)+np.roll(bo,1,axis=1)+np.roll(bo,-1,axis=1)

w = 1./nn
#w = out.occ.values#+1.#/(1./N+out.occ.values)
w = w[ibocc[:,0],ibocc[:,1]]
w /= np.sum(w)

poc = out.Apoc.values#(out.bocc&out.b).values
noc = out.Anoc.values#(out.bocc&~out.b).values
TP = np.nansum(poc)#np.count_nonzero(poc)
FN = np.nansum(noc)#np.count_nonzero(noc)

#TPb = []
#FNb = []
fTPbi = []
fTPbs = []
for fsmp in fsmps:
    fbi = []
#    FNbi = []
    
    fbs = []
#    FNbs = []
    #TPbw = []
    #FNbw = []
    for n in range(nboot):
    #    Hb = np.random.choice(H.ravel(),int(N*fsmp)).astype('bool')
    #    TPb.append(np.count_nonzero(Hb))
    #    FNb.append(np.count_nonzero(~Hb))
        
        iby,ibx = ibocc[np.random.choice(range(0,N),size=int(N*fsmp),replace=True)].T
        TPi = np.nansum(poc[iby,ibx])
#        FNi = np.count_nonzero(noc[iby,ibx])
        fbi.append(100*TPi/(np.nansum(poc[iby,ibx])+np.nansum(noc[iby,ibx])))#(N*fsmp))
#        FNbi.append(FNi)
        
        iby,ibx = ibocc[np.random.choice(range(0,N),size=int(N*fsmp),replace=True,p=w)].T
        TPi = np.nansum(poc[iby,ibx])
#        FNi = np.count_nonzero(noc[iby,ibx])
        fbs.append(100*TPi/(np.nansum(poc[iby,ibx])+np.nansum(noc[iby,ibx])))
    #    Hbw = np.random.choice(H.ravel(),N,p=w).astype('bool')
    #    TPbw.append(np.count_nonzero(Hbw))
    #    FNbw.append(np.count_nonzero(~Hbw))
    
    #plt.bar(['FN$_b$','FN','TP$_b$','TP'],
    #        [np.mean(FNbi),FN,np.mean(TPbi),TP],
    #        yerr=[np.std(FNbi),0,np.std(TPbi),0],
    #        color=['0.5','b','0.5','b'])
    #plt.show()
    fTPbi.append(fbi)
    fTPbs.append(fbs)
    
#%%
la = lambda l,f,**args: [f(x,**args) for x in l]

fig = plt.figure(constrained_layout=True)
fig.set_size_inches(cwidth, height)

#plt.hlines(100*fAhab,np.min(N*fsmps),np.max(N*fsmps),linestyles='--',lw=1)

plt.plot(N*fsmps,la(fTPbi,np.median),label='uniform')
plt.fill_between(N*fsmps,la(fTPbi,np.percentile,q=2.5),la(fTPbi,np.percentile,q=97.5),alpha=0.3)

plt.plot(N*fsmps,la(fTPbs,np.median),label='density')
plt.fill_between(N*fsmps,la(fTPbs,np.percentile,q=2.5),la(fTPbs,np.percentile,q=97.5),alpha=0.3)

plt.hlines(100*fApoc,np.min(N*fsmps),np.max(N*fsmps),linestyles=':',lw=1, zorder=10)

plt.legend()
plt.xticks(N*fsmps)
plt.xlabel('Number of sites')
plt.ylabel('True positive rate (%)')
plt.savefig(r'./out/SI_TP.png',dpi=300)

plt.show()

#%%
fig = plt.figure()
fig.set_size_inches(2*cwidth, 2*height)
#gs = mpl.gridspec.GridSpec(4,1,wspace=0.05)
#res = '10m'
ax = plt.axes(projection=proj)
ax.background_img(name='ne_gray',resolution='high')
ax.coastlines(resolution=res,zorder=2,lw=0.66)
ax.add_feature(cfeature.OCEAN.with_scale(res),zorder=1)
ax.add_feature(cfeature.LAKES.with_scale(res),zorder=1,edgecolor='None')
ax.add_feature(cfeature.RIVERS.with_scale(res),lw=0.25,zorder=1)
plt.contourf(out.lon,out.lat,out.b.where(out.lat>-60.),[0.49,1],colors=habitat,alpha=halpha,zorder=2)
#plt.contourf(out.lon,out.lat,(out.aM>=2).where(out.lat>-60.),[0.49,1],colors='g',alpha=0.3,zorder=2)
#plt.contourf(out.lon,out.lat,out.aM.where(out.lat>-60.),[2,4,8],cmap=mpl.colors.ListedColormap(['lightgreen','g']),extend='max',alpha=0.3,zorder=2)
#plt.scatter(occ.lon,occ.lat,color=occol,marker='o',s=1,lw=0,zorder=3)
for i,fam in enumerate(np.unique(occ.family)):
    occf = occ.loc[occ.family==fam]
    if fam=='Various':
        plt.scatter(occf.lon,occf.lat, color=mpl.cm.tab10(i), marker='^',s=1,lw=0.1,edgecolor='k',zorder=3,label=fam+' (Abbott, 1994)')
    else:
        plt.scatter(occf.lon,occf.lat, color=mpl.cm.tab10(i), marker='o',s=1,lw=0.1,edgecolor='k',zorder=3,label=fam+' (GBIF)')
plt.legend(fancybox=False,markerscale=2)
plt.savefig(r'./out/all_factors_global_occupancy_{:}.png'.format(avg),dpi=300)
plt.show()

#%%
fig = plt.figure()
fig.set_size_inches(2*cwidth, 6*height)
#gs = mpl.gridspec.GridSpec(4,1,wspace=0.05)

ax = plt.subplot(6, 1, 1, projection=proj)#fig.add_subplot(gs[0],projection=proj)
ax.coastlines(resolution='50m')
plt.pcolormesh(out.lon,out.lat,out.PL.where(out.PL>0)*1e-3,norm=mpl.colors.LogNorm(),vmin=10,vmax=1e3,transform=proj,cmap='cividis')
plt.colorbar(label='$P_L\, (kPa)$')
plt.contour(out.lon,out.lat,(out.bPL).where(out.PL>0),[0.49,1], colors='r',linewidths=0.25, transform=proj)

ax = plt.subplot(6, 1, 2, projection=proj)#fig.add_subplot(gs[0],projection=proj)
ax.coastlines(resolution='50m')
plt.pcolormesh(out.lon,out.lat,out.aM.where(out.PL>0),transform=proj,cmap='cividis')
plt.colorbar(label='Mean consecutive months of activity $P_L\,<200\,kPa$')
plt.contour(out.lon,out.lat,(out.baM).where(out.PL>0),[0.49,1], colors='r',linewidths=0.25, transform=proj)

ax = plt.subplot(6, 1, 3, projection=proj)#fig.add_subplot(gs[0],projection=proj)
ax.coastlines(resolution='50m')
plt.pcolormesh(out.lon,out.lat,out.MAT.where(out.PL>0),transform=proj,cmap='cividis')
plt.colorbar(label=u'MAT (°C)')
plt.contour(out.lon,out.lat,(out.bMAT).where(out.PL>0),[0.49,1], colors='r',linewidths=0.25, transform=proj)

ax = plt.subplot(6, 1, 4, projection=proj)#fig.add_subplot(gs[0],projection=proj)
ax.coastlines(resolution='50m')
plt.pcolormesh(out.lon,out.lat,out.SND.where(out.PL>0),transform=proj,cmap='cividis')
plt.colorbar(label=r'Sand Content (%)')
plt.contour(out.lon,out.lat,(out.bSND).where(out.PL>0),[0.49,1], colors='r',linewidths=0.25, transform=proj)

ax = plt.subplot(6, 1, 5, projection=proj)#fig.add_subplot(gs[0],projection=proj)
ax.coastlines(resolution='50m')
plt.pcolormesh(out.lon,out.lat,out.pH.where(out.PL>0),transform=proj,cmap='cividis')
plt.colorbar(label=r'pH (-)')
plt.contour(out.lon,out.lat,(out.bpH).where(out.PL>0),[0.49,1], colors='r',linewidths=0.25, transform=proj)

ax = plt.subplot(6, 1, 6, projection=proj)#fig.add_subplot(gs[0],projection=proj)
ax.coastlines(resolution='50m')

plt.contourf(out.lon,out.lat,(~out.bMAT).where(out.PL>0),[0.49,1], colors='b',transform=proj,alpha=0.33,label=u'MAT<0°C')
plt.contourf(out.lon,out.lat,(~out.bSND).where(out.PL>0),[0.49,1], colors='g',transform=proj,alpha=0.33,label=r'SND>80%')
plt.contourf(out.lon,out.lat,(~out.bpH).where(out.PL>0),[0.49,1], colors='r',transform=proj,alpha=0.33,label=r'pH<4')

proxy = [plt.Rectangle((0,0),1,1,fc='b'),plt.Rectangle((0,0),1,1,fc='g'),plt.Rectangle((0,0),1,1,fc='r')]

plt.legend(proxy, [u'MAT<0°C', r'SND>80%', r'pH<4.5'],bbox_to_anchor=(1.04,0.5), loc='center left')

plt.savefig(r'./out/cutoff.png',dpi=300)
plt.show()

#%%
fig = plt.figure()
fig.set_size_inches(2*cwidth, 2*height)

ax = plt.axes(projection=proj)
ax.coastlines(resolution='50m')
plt.contourf(out.lon,out.lat,out.aM.where((out.lat>-60.)&(out.PL>0)),[0,1,2,3,4,5,6,7,8,9,10,11,12],transform=proj,cmap='RdYlGn',extend='max')
plt.colorbar(label='Consecutive months of activity $P_L\,<200\,kPa$', orientation='horizontal', shrink=0.5, pad=0.05)

plt.savefig(r'./out/months_below.png',dpi=300)
plt.show()

#%%
dff['MAP'] = dat.MAP.sel_points(lon=dff.lon.values,lat=dff.lat.values,method='nearest').values
dff0 = dff.groupby(['lon','lat']).mean()

fig = plt.figure()
fig.set_size_inches(2*cwidth, 2*height)

sw = pd.read_csv('C:/Users/bickels/Documents/GitHub/geoworm/data/1804_2_sWormModelData.csv')
sw['occ'] = sw.Sites_Abundancem2>0

ax = plt.subplot(231)
panel((0,1),(5,-2),'a')
smhist(sw.loc[sw.occ].phFinal.dropna())
plt.xlabel('Soil pH')
plt.ylabel('Probability density')
#plt.show()

ax = plt.subplot(232)
panel((0,1),(5,-2),'b')
smhist(sw.loc[sw.occ].PHIHOX.dropna())
smhist(dff0.pH.dropna(),color=occol)
plt.xlabel('Soilgrids pH')
#plt.show()

ax = plt.subplot(233)
panel((0,1),(5,-2),'c')
smhist(sw.loc[sw.occ].SNDPPT.dropna())
smhist(dff0.SND.dropna(),color=occol)
plt.xlim(0,100)
plt.xlabel('Soilgrids sand (%)')
#plt.show()

ax = plt.subplot(234)
panel((0,1),(5,-2),'d')
smhist(sw.loc[sw.occ].bio10_1.dropna())
smhist(dff0.MAT.dropna(),color=occol)
plt.xlabel(u'MAT (°C)')
plt.ylabel('Probability density')
#plt.show()

ax = plt.subplot(235)
panel((0,1),(5,-2),'e')
smhist(sw.loc[sw.occ].bio10_12.dropna())
smhist(dff0.MAP.dropna(),color=occol)
plt.xlim(0,4000)
plt.xlabel(u'MAP ($mm\, y^{-1}$)')

ax = plt.subplot(236)
panel((0,1),(5,-2),'f')
#smhist(sw.loc[sw.occ].bio10_12.dropna())
smhist(dff0.PL.dropna()*1e-3,color=occol,Nint=2**12)
plt.xlim(0,400)
plt.xlabel(r'Limiting pressure $P_L$ ($kPa$)')

plt.savefig(r'./out/SI_dists.png',dpi=300)
plt.show()

#%%
tPL = pd.read_excel(r'./data/PL_timeseries.xlsx')
#tPL['moy'] = tPL.time.dt.month
#cPL = tPL.groupby(['biome','moy']).mean().reset_index()
cPL  = pd.read_excel(r'./data/PL_climatic.xlsx')

tPr = pd.read_excel('./data/precipitation_timeseries.xlsx')
cPr = pd.read_excel('./data/precipitation_climatic.xlsx')

#tPr['moy'] = tPr.time.dt.month
#cPr = tPr.groupby(['biome','moy']).describe().reset_index()

#%%
def plot_time():
    #plt.step(tPL0.time.dt.month,1e-3*tPL0['mean'],'k',lw=1)
#    plt.plot(cPL0.moy,1e-3*cPL0['50%'],'k')
    yerr=1e-3*np.array([cPL0['50%']-cPL0['25%'],cPL0['75%']-cPL0['50%']])
    plt.bar(cPL0.moy,1e-3*cPL0['50%'],yerr=yerr,color='0.75',edgecolor='k',capsize=2,error_kw=dict(lw=0.5))
#    plt.fill_between(cPL0.moy,1e-3*cPL0['25%'],1e-3*cPL0['75%'],color='k',alpha=0.3,lw=0)
    plt.hlines(200,1,12,lw=1,linestyles='--')
    plt.ylabel(r'Median limiting pressure $P_L$ ($kPa$)')
    ax = plt.twinx()
    #plt.step(tPr0.time.dt.dayofyear/31.,tPr0['mean'],'b',lw=1)
#    plt.plot(cPr0.dayofyear/31.+1,cPr0['mean'],'royalblue',alpha=0.5)
    rollm = cPr0['mean'].rolling(31,1,center=True).mean()
    rolls = cPr0['mean'].rolling(31,1,center=True).std()
    plt.plot(cPr0.dayofyear/31.+1,rollm,'royalblue')
    plt.fill_between(cPr0.dayofyear/31.+1,rollm-rolls,rollm+rolls,color='royalblue',alpha=0.3,lw=0)
    ax.tick_params(axis='y', labelcolor='royalblue',colors='royalblue')
    ax.spines["right"].set_edgecolor('royalblue')
    #plt.plot(cPr0.moy,cPr0['mean'],'cornflowerblue')
    plt.ylabel(r'Mean precipitation ($mm\, d^{-1}$)',color='royalblue')

fig = plt.figure()
fig.set_size_inches(cwidth, 2*height)

year = 2016
tPL0 = tPL.loc[(tPL.biome=='grassland')&(tPL.time.dt.year==year)]
cPL0 = cPL.loc[cPL.biome=='grassland']
tPr0 = tPr.loc[(tPr.biome=='grassland')&(tPr.time.dt.year==year)]
cPr0 = cPr.loc[cPr.biome=='grassland']

ax = plt.subplot(211)
plt.title('Grassland')
#ax.set_zorder(1)
#ax.patch.set_visible(False)
panel((0,1),(5,-2),'a')
plt.ylim(0,600)
plot_time()
plt.ylim(0,14)
#plt.xlim(1,12)
#plt.xticks(range(1,13))
#ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])

tPL0 = tPL.loc[(tPL.biome=='desert')&(tPL.time.dt.year==year)]
cPL0 = cPL.loc[cPL.biome=='desert']
tPr0 = tPr.loc[(tPr.biome=='desert')&(tPr.time.dt.year==year)]
cPr0 = cPr.loc[cPr.biome=='desert']

ax = plt.subplot(212,sharex=ax)
plt.title('Desert')
#ax.set_zorder(1)
#ax.patch.set_visible(False)
panel((0,1),(5,-2),'b')
plt.ylim(0,1400)
plot_time()
plt.ylim(0,7)

plt.xlim(1,12)
plt.xticks(range(1,13))
#ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax.set_xticklabels(['JAN','FEB','MAR','APR','MAI','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])

plt.savefig(r'./out/timewindows.png',dpi=300)
plt.show()