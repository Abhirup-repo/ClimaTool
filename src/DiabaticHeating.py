
# @ Author: Abhirup Banerjee
# @ Create Time: 2023-11-24 11:26:53
# @ Modified by: Abhirup Banerjee
# @ Modified time: 2023-11-27 15:19:12
# @ Description: This file is written to compute the diabatic heating
# source: Estimates of Tropical Diabatic Heating Profiles: Commonalities and Uncertainties https://journals.ametsoc.org/view/journals/clim/23/3/2009jcli3025.1.xml
# Sample exmaple: https://hannahlab.org/a-comparison-of-methods-for-estimating-diabatic-heating/
# We use ERA 5, August 2016, and calculated the heating estimates with both method and averaged them over the entire Tropics (30°S-30°N).


import numpy as np
import xarray as xr
import pylab as plt
from tqdm import tqdm 

def ddt(da,dt):
    nt,nlev,nlat,nlon=da.shape
    dqdt=np.ones(da.shape)*np.nan

    dqdt=np.zeros(da.shape)
    tq=da.values
    dqdt[1:nt-2,:,:,:]=(tq[2:nt-1,:,:,:]-tq[0:nt-3,:,:,:])/(2*dt)


    da_ddt = xr.DataArray(
    data=dqdt,
    dims=["time", "level", "lat","lon"],
    coords={"time":da.time.values,"level":da.level.values,"lat":da.lat.values,"lon":da.lon.values},   
    )
    return da_ddt



def ddx(da,lon_cyclic=False):
    rd = 6378140
    dim = da.shape
    lat =  da.lat.values
    lon =  (da.lon.values)
    nlat = dim[2]
    nlon = dim[3]
    df = np.zeros(da.shape)
    X=da.values
    if lon_cyclic :
        xvals = np.arange(0,nlon-1,1)      ;
        Xvals = xvals[0:nlon-1]        # East-West Center Values
        Evals = (Xvals+1)%nlon         # East values
        Wvals = (Xvals+nlon-1)%nlon         # West Values
    else:
        xvals = np.arange(0,nlon-1,1)      
        Evals = xvals[2:nlon-1]        # East values
        Xvals = xvals[1:nlon-2]        # East-West Center Values
        Wvals = xvals[0:nlon-3]        # West Values
    dx = lon
    dx[Xvals] = (lon[Evals]+lon[Xvals])/2 - (lon[Wvals]+lon[Xvals])/2
    deltax=np.broadcast_to(dx[None,None,None,:],(dim))
    deltax=deltax*111000
    latx=np.cos(lat)
    dellon=deltax*np.broadcast_to(latx[None,None,:,None],dim)
    df[:,:,:,Xvals] = ( (X[:,:,:,Evals]+X[:,:,:,Xvals])/2 - (X[:,:,:,Wvals]+X[:,:,:,Xvals])/2 ) /dellon[:,:,:,Xvals]

    return df

# def ddx(da):

#     rd = 6378140
#     dim = da.shape
#     lat =  da.lat
#     lon =  da.lon
#     NLATS = dim[2]
#     NLONS = dim[3]
#     df = np.zeros(da.shape)
#     x=da
#     for i in tqdm(range(NLATS-1)):
#         j = 0
#         dx = ((lon[j+1] - lon[j]) + 360 - lon[NLONS - 1])*rd*2*3.14159/360
#         dx = dx*np.cos(lat[i]/180.*3.14159)
#         df[:,:,i,j] = (x[:,:,i,j+1] - x[:,:,i,NLONS - 1]) / dx

#         for j in range(1,NLONS-2):
#             dx = (lon[j+1] - lon[j-1])*rd*2*3.14159/360
#             dx = dx*np.cos(lat[i]/180.*3.14159)
#             df[:,:,i,j] = (x[:,:,i,j+1] - x[:,:,i,j-1]) / dx
        
#         j = NLONS - 1
#         dx = (360 - lon[j-1])*rd*2*3.14159/360
#         dx = dx*np.cos(lat[i]/180.*3.14159)
#         df[:,:,i,j] = (x[:,:,i,0] - x[:,:,i,j-1]) / dx



#     return df


def ddy(da):
    rd = 6378140
    dim = da.shape
    lat =  da.lat
    lon =  da.lon
    NLATS = dim[2]
    NLONS = dim[3]
    df = np.zeros(da.shape)
    x=da

    i = 0
#   ;lattitude N-->S; dy ahead north is positive
    dx = (lat[i] - lat[i+1])*rd*2*3.14159/360
    df[:,:,i,:] = (x[:,:,i,:] - x[:,:,i+1,:]) / dx

    for i in range(1,NLATS-2):
        dx = (lat[i] - lat[i+1])*rd*2*3.14159/360
        df[:,:,i,:] = (x[:,:,i,:] - x[:,:,i+1,:]) / dx
    i = NLATS - 1
    dx = (lat[i-1] - lat[i])*rd*2*3.14159/360
    df[:,:,i,:] = (x[:,:,i-1,:] - x[:,:,i,:]) / dx
    return df


def ddp(da):
    rd = 6378140
    dim = da.shape
    lev=da.level
    llev=np.log10(lev)
    nlev=dim[1]
    df = np.zeros(da.shape)
    x=da

    i = 0
    dx = llev[i+1] - llev[i]
    df[:,i,:,:]  = (x[:,i+1,:,:] - x[:,i,:,:]) / dx / lev[i]

    for i in range(1,nlev-2):
        dx1 = llev[i] - llev[i-1]
        dx2 = llev[i+1] - llev[i-1]
        df[:,i,:,:] = ((x[:,i+1,:,:] - x[:,i,:,:])*dx1/dx2 + \
                     (x[:,i,:,:] - x[:,i-1,:,:])*dx2/dx1 ) / (dx1+dx2) / lev[i]


    i = nlev - 1
    dx = llev[i] - llev[i-1]
    df[:,i,:,:] = (x[:,i,:,:] - x[:,i-1,:,:]) / dx / lev[i]
    return df




ds=xr.open_dataset("../data/ERA5_1deg.nc")
#ds=ds.rename({"latitude":"lat","longitude":"lon"})
daT=ds.t
daU=ds.u
daV=ds.v 
daW=ds.w

nt,nlev,nlat,nlon=daT.shape
lev=daT.level.values
pt=daT.copy()

for i in range(nlev-1):
     zl=lev[i] ;print(zl)
     pt[:,i,:,:]=daT[:,i,:,:]*((1000/zl)**.286)



dt=6*3600
cal_dt=ddt(pt,dt)
cal_dx=ddx(pt)
cal_dy=ddy(pt)
cal_dp=ddp(pt)

R = 287.05
Cp = 1005

cons=Cp*(daT.data/pt.data)

Q=cons*(cal_dt+daU*cal_dx+daV*cal_dy+daW*cal_dp)

Q_zonal=Q.mean("lon")
Q_time=Q_zonal.mean("time")

ax=plt.figure().add_subplot()
Q_time.plot.contourf(ax=ax,levels=20)
ax.invert_yaxis()

#daPT=daT.copy()


# dapt = xr.DataArray(
#     data=pt.values,
#     dims=["time", "level", "lat","lon"],
#     coords={"time":daT.time.values,"level":daT.level.values,"lat":daT.lat.values,"lon":daT.lon.values},   
#     attrs=dict(
#         description="Ambient temperature.",
#         units="degC",
#     ),
# )

# dt=6*3600
# cal_dt=ddt(pt,dt)

# cal_hori=ddx(pt)

# 
# cal_ddt=ddt(daT,dt)
# cal_ddx=ddx(daT)
# cal_ddy=ddy(daT)



# lon=ds.longitude.values
# lat=ds.latitude.values
# lonsz=len(ds.longitude)
# #if lon_cyclic then
# # xvals = np.arange(0,lonsz-1,1)      ;
# # Xvals = xvals[0:lonsz-1]        # East-West Center Values
# # Evals = (Xvals+1)%lonsz         # East values
# # Wvals = (Xvals+lonsz-1)%lonsz         # West Values

# xvals = np.arange(0,lonsz-1,1)      
# Evals = xvals[2:lonsz-1]        # East values
# Xvals = xvals[1:lonsz-2]        # East-West Center Values
# Wvals = xvals[0:lonsz-3]        # West Values

# dx = lon
# dx[Xvals] = (lon[Evals]+lon[Xvals])/2 - (lon[Wvals]+lon[Xvals])/2

# dims=daT.shape
# deltax = np.broadcast_to(dx, dims[-1])
# deltax = deltax*111000.*np.broadcast_to(np.cos(lat*3.14159/180.), dims[-2])

# da=daT
# dqdx=np.zeros(da.shape)
# X=da.values
# dqdx[:,:,:,:]=( (X[:,:,:,Evals]+X[:,:,:,Xvals])/2 - (X[:,:,:,Wvals]+X[:,:,:,Xvals])/2 )/deltax





# function difft(x[*][*][*][*]:numeric,y[*][*][*]:numeric)
# local df,dim,lat,lon,NTIME,i,dy
# begin
#   dim = dimsizes(x)
#   NTIME = dim(0)
#   df = x
#   dy = 1
#   i = 0
#   df(i,:,:,:) = ( x(i,:,:,:) - y) /24/3600
#   do i = 1, NTIME - 1
#     df(i,:,:,:) = (x(i,:,:,:) - x(i-1,:,:,:)) /24/3600/dy
#   end do
#   return df
# end
