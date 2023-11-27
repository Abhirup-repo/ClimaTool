import numpy as np


ar=np.arange(1,10)
a,b=5,2
d=np.broadcast_to(ar[None,None,:],(a,b,len(ar)))
e=np.arange(11,16)
d=d*10
f=d*np.broadcast_to(e[:,None,None],d.shape)