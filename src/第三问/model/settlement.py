from common import *
CAP=B

def dispatch(q,net,s0,capacity=1.):
    n=len(q);s=np.zeros(n+1);s[0]=s0;c=np.zeros(n);d=c.copy();e=c.copy();w=c.copy()
    for t in range(n):
        balance=q[t]-net[t]
        if balance>=0:
            c[t]=min(balance,CAP*capacity,max(0,(HIGH*capacity-s[t])/ETA));w[t]=max(0,balance-c[t])
        else:
            d[t]=min(-balance,CAP*capacity,max(0,(s[t]-LOW*capacity)*ETA));e[t]=max(0,-balance-d[t])
        s[t+1]=s[t]+ETA*c[t]-d[t]/ETA
    return dict(c=c,d=d,e=e,w=w,s=s)
