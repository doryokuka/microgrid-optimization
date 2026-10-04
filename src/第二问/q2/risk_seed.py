"""Feasible B-policy local seed only. Certification still uses global MILP/LP bounds."""
from .common import *
from .settlement import dispatch
from scipy.optimize import minimize

def costs_jac(q,nets,p,s0,cuts):
    K,T=nets.shape;values=[];grads=[]
    for n in nets:
        s=float(s0);ds=np.zeros(T);cost=0.;dcost=np.zeros(T)
        for t in range(T):
            unit=np.zeros(T);unit[t]=1;balance=q[t]-n[t]
            if balance>=0:
                vals=[balance,B,max(0.,(HIGH-s)/ETA)];j=int(np.argmin(vals));c=vals[j]
                dc=[unit,np.zeros(T),-ds/ETA][j];s+=ETA*c;ds+=ETA*dc
            else:
                vals=[-balance,B,max(0.,(s-LOW)*ETA)];j=int(np.argmin(vals));u=vals[j]
                du=[-unit,np.zeros(T),ds*ETA][j];e=-balance-u
                cost+=5*p[t]*e;dcost+=5*p[t]*(-unit-du);s-=u/ETA;ds-=du/ETA
        candidates=[(0.,0.)]+list(cuts);a,b=max(candidates,key=lambda z:z[0]*s+z[1])
        values.append(cost+a*s+b);grads.append(dcost+a*ds)
    return np.array(values),np.array(grads)

def improve(q,nets,p,s0,cuts,qub):
    T=len(q);cache={}
    def cj(x):
        key=x[:T].tobytes()
        if cache.get('key')!=key:cache.update(key=key,result=costs_jac(x[:T],nets,p,s0,cuts))
        return cache['result']
    values,_=costs_jac(q,nets,p,s0,cuts)
    x0=np.r_[q,values.max()]
    cons=dict(type='ineq',fun=lambda x:x[-1]-cj(x)[0],jac=lambda x:np.c_[-cj(x)[1],np.ones(len(nets))])
    result=minimize(lambda x:p@x[:T]+x[-1],x0,jac=lambda x:np.r_[p,1.],bounds=list(zip(np.zeros(T),qub))+[(0,None)],
                    constraints=cons,method='SLSQP',options=dict(maxiter=200,ftol=1e-7))
    return np.clip(result.x[:T],0,qub),dict(success=bool(result.success),iterations=int(result.nit),message=str(result.message))

