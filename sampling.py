# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] tags=[]
# ##  Used saved DL model to produce figures

# + tags=[]
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import sample
import xarray as xr
import itertools
#from numpy.random import seed
from random import seed 
#tf.__version__
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#tf.config.list_physical_devices()
import fit_lib 
# -

# %load_ext autoreload
# %autoreload 2

# # Load the data 

# + tags=[]
loc=0.0
file1='loc'+str(loc)+'.nc'
da=xr.open_dataset(file1)
file1
# -

test_results = {}


# # Organise the data to use in the DL model 

# + tags=[]
asol=da.asol    # % return value error
t_ret=da.t_ret  # true solution
i_ret=da.i_ret  # solution for each iteration
aerr=i_ret.std(axis=3)/i_ret.mean(axis=3) # std/mean from iterations

n=asol.coords['size'].values
ashp=asol.coords['shape'].values*(-1)  # GEV code uses the opposite for shape parameter
ascl=asol.coords['scale'].values
ari=asol.coords['ari'].values
rtmp=aerr*100*2
# redefine relative error based on the (95% - 5% / 50% ) values times 100 to get percent 
a50=i_ret.reduce(np.percentile,axis=3,q=50)
a5=i_ret.reduce(np.percentile,axis=3,q=5)
a95=i_ret.reduce(np.percentile,axis=3,q=95)
rtmp=100*((a95-a5)/a50) *.5  # to reflect +/-

# +

print(rtmp[0,10,0,:].values)
for rr in range(4):
    plt.hist(i_ret[0,10,4,:,rr],label='ARI ='+str(int((ari[rr]+.1))) )
    plt.plot([a5[0,10,4,rr],a50[0,10,4,rr],a95[0,10,4,rr]],[rr,rr,rr],'bx-')
    plt.title('Shape ='+str(ashp[10])+ ' Scale = ' + str(ascl[0]) )
    plt.xlabel('Return Value')
    plt.ylabel('Number')
    plt.legend()
#plt.hist(i_ret[0,10,4,:,1])
#plt.plot([a5[0,10,4,1],a50[0,10,4,1],a95[0,10,4,1]],[10,10,10],'gx-')
plt.savefig('figf0a.png')
# -

da

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ##  Unroll the 4d rtmp into a 1d array

# +
# turn data into form suitable for DL
seed(2)
xtmp,ytmp=fit_lib.xroll(rtmp)
print(xtmp.shape)
nt=ytmp.size
i=sample(range(0, nt), nt)

# resample data for training and validation
xt=np.copy(xtmp[:,0:4])
yt=np.copy(ytmp)
xt[:,0:4]=xtmp[i[:],0:4]
yt[:]=ytmp[i[:]]

print(i[0:10])
print(xt.shape,yt.shape)

# + [markdown] tags=[]
# #  Load Multiple models 

# +
loss='mean_absolute_error'
learn=0.004
#loss='mean_squared_error'

# load all models into memory
n_save=50 
members = fit_lib.load_all_models('casea1/amodel1',0,n_save)
print('Loaded %d models' % len(members))
# prepare an array of equal weights
members[0].summary()
# -


imember=0; a1=np.zeros(len(members))
for n in members :
    a1[imember]=n.evaluate(xt,yt,verbose=0)
    print(a1[imember])
    imember=imember+1


ibest=13
ibest=23
print(a1[ibest])

# + [markdown] tags=[]
# # Compute the sample size for a rainfall and temperature extreme examples
#
# for the calculation I will use the models but account for non-zero location and scale parameters by normalising to scale and correcting relative error using location 

# + tags=[]
from scipy.stats import genextreme as gev
# make new dataset with ns x 4G
# rainfall example shape=0 and 0.25. mu' = 3 scale=1
# temperature example shape=-0.2 mu' = 13.2 scale = 1
scl1=np.array([1]); shp1=np.array([0, .25])
loc=3

erel=np.array([10.])  # desired uncertainty o
eari=np.array([20,50,100,200]) # desired return period
#eari=np.arange(10,200,2) # desired return period

xp = fit_lib.rroll(scl1,shp1,erel,eari)

inv_ari = 1./ xp[:,3]
shpa= -1*xp[:,1]
scla= xp[:,0]

R0 = gev.isf(inv_ari, shpa,0,scla)
Rm = gev.isf(inv_ari, shpa,loc,scla)

E_Rm = xp[:,2]* (Rm/R0)
xp[:,2]=E_Rm

print (R0)
print (Rm)
print (E_Rm)
print (xp)

# -

ya = members[ibest].predict(xp).flatten()
print(ya)
print(ya*xp[:,3]) # convert into number of samples by multiplying by ARI
ya_rain=np.copy(ya)

# +
imember=0; a=np.zeros([len(ya),len(members)])
print(a.shape)

for n in members :
    a[:,imember]=n.predict(xp).flatten()
    imember=imember+1
# -

yall=np.copy(a[:,:])
y_rain1=yall[:,:].min(axis=1)
y_rain2=yall[:,:].max(axis=1)
y_rain=yall[:,ibest]

# +
# make new dataset with ns x 4G
# rainfall example shape=0 and 0.25. mu' = 3 scale=1
# temperature example shape=-0.2 mu' = 13.2 scale = 1
scl1=np.array([1]); shp1=np.array([-.2,-.10])
loc=13.2
erel=np.array([3.])  # desired uncertainty o
eari=np.array([20,50,100,200]) # desired return period

xp = fit_lib.rroll(scl1,shp1,erel,eari)

inv_ari = 1./ xp[:,3]
shpa= -1*xp[:,1]
scla= xp[:,0]

R0 = gev.isf(inv_ari, shpa,0,scla)
Rm = gev.isf(inv_ari, shpa,loc,scla)

E_Rm = xp[:,2]* (Rm/R0)
xp[:,2]=E_Rm

print (R0)
print (Rm)
print (E_Rm)
print (xp)

# -

ya = members[ibest].predict(xp).flatten()
print(ya)
print(ya*xp[:,3]) # convert into number of samples by multiplying by ARI

# +
imember=0; a=np.zeros([len(ya),len(members)])
print(a.shape)

for n in members :
    a[:,imember]=n.predict(xp).flatten()
    imember=imember+1
# -

yall=np.copy(a[:,:])
y_temp1=yall[:,:].min(axis=1)
y_temp2=yall[:,:].max(axis=1)
y_temp=np.median(yall,axis=1) 
y_temp=yall[:,ibest]

# +
plt.figure(figsize=(8.2,5))

plt.subplot(1,2,1)
plt.plot(eari,y_rain[0:np.size(eari)]*eari,label="Shape="+str(0))
plt.fill_between(eari,y_rain1[0:4]*eari, y_rain2[0:4]*eari,alpha=.4)
plt.plot(eari,y_rain[np.size(eari)-1:-1]*eari,label="Shape="+str(0.25))
plt.fill_between(eari,y_rain1[np.size(eari)-1:-1]*eari, y_rain2[np.size(eari)-1:-1]*eari,alpha=.4)
plt.xlabel("Annual Return Period")
plt.ylabel("Sample Size")
plt.title("Rainfall Extreme")
plt.legend()

plt.subplot(1,2,2)

#plt.plot(eari,np.rint(ya_temp[0:np.size(eari)]*eari),label="Shape="+str(-0.2))
plt.plot(eari,np.rint(y_temp[0:np.size(eari)]*eari),label="Shape="+str(shp1[0]))
plt.fill_between(eari,y_temp1[0:4]*eari, y_temp2[0:4]*eari,alpha=.4)
plt.plot(eari,np.rint(y_temp[np.size(eari)-1:-1]*eari),label="Shape="+str(shp1[1]))
plt.fill_between(eari,y_temp1[np.size(eari)-1:-1]*eari, y_temp2[np.size(eari)-1:-1]*eari,alpha=.4)
plt.xlabel("Annual Return Period")
plt.ylabel("Sample Size")
plt.title("Temperature Extreme")
#plt.ylim(0,110)
plt.legend()

plt.savefig('fig1.pdf',dpi=600)

