from .common import *

def naive(hist,load=True):
    d=len(hist)
    if d==0:return np.full(144,5000/6 if load else 0.)
    return hist[d-7] if load and d>=7 else hist[-min(3,d):].mean(0)

def features(hist,d,load=True,holiday=True):
    assert len(hist)==d
    t=np.arange(144);wd=int(CAL['weekday'][d]);a=[]
    for f in [1,2,3,4]:a.extend([np.sin(2*np.pi*f*t/144),np.cos(2*np.pi*f*t/144)])
    a.extend([t/144,np.full(144,np.sin(2*np.pi*d/365)),np.full(144,np.cos(2*np.pi*d/365)),np.full(144,(d+1)/365)])
    if load:
        a.extend([np.full(144,wd>=5)]+[np.full(144,wd==i) for i in range(7)])
        if holiday:a.extend([np.full(144,CAL[x][d]) for x in ['holiday','adjust']]+[np.full(144,CAL['daytype'][d]==i) for i in range(4)])
    default=naive(hist,load)
    for lag in ([1,7,14] if load else [1,2,3,7]):a.extend([hist[-lag] if d>=lag else default,np.full(144,d>=lag)])
    for n in [3,7,14]:
        h=hist[max(0,d-n):d]
        a.extend([h.mean(0) if d else default,h.std(0) if d else np.zeros(144)])
    if d:
        totals=hist.sum(1);a.extend([np.full(144,totals[-1]),np.full(144,totals[-min(7,d):].mean()),np.full(144,totals[-min(7,d):].std()),hist[-1]-hist[-min(7,d):].mean(0)])
    else:a.extend([np.zeros(144)]*4)
    return np.stack(a,axis=1).astype(float)
