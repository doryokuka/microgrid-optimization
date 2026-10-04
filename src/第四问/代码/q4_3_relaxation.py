"""LP relaxation: same Q2 battery relaxation, plus exact convex adjustment cost."""
from q4_3_bridge import *
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def relaxed_seed(nets,p,s0,weights,cuts,reference):
    K,T=nets.shape;S=K*T;tail=T+4*S;ab=tail+K;nv=ab+T
    mask=np.isfinite(reference)
    cost=np.zeros(nv);cost[:T]=p;cost[tail:ab]=weights;cost[ab:]=.5*p*mask
    rr=[];cc=[];vv=[];rhs=[];ur=[];uc=[];uv=[];ubs=[]
    ix=lambda j,k,t:T+j*S+k*T+t
    def ub(dic,rhs):
        row=len(ubs)
        for j,v in dic.items():ur.append(row);uc.append(j);uv.append(v)
        ubs.append(rhs)
    for k in range(K):
        for t in range(T):
            c,u,s,e=[ix(j,k,t) for j in range(4)];row=len(rhs)
            dic={s:1,c:-ETA,u:1/ETA}
            if t:dic[ix(2,k,t-1)]=-1
            for j,v in dic.items():rr.append(row);cc.append(j);vv.append(v)
            rhs.append(s0 if not t else 0)
            ub({t:-1,c:1,u:-1,e:-1},-nets[k,t]);cost[e]=5*p[t]*weights[k]
        for slope,intercept in cuts:ub({tail+k:-1,ix(2,k,T-1):slope},-intercept)
    for t in np.flatnonzero(mask):
        ub({t:1,ab+t:-1},reference[t]);ub({t:-1,ab+t:-1},-reference[t])
    eq=coo_matrix((vv,(rr,cc)),shape=(len(rhs),nv)).tocsr()
    mat=coo_matrix((uv,(ur,uc)),shape=(len(ubs),nv)).tocsr()
    bounds=[(0,None)]*T+[(0,B)]*(2*S)+[(LOW,HIGH)]*S+[(0,None)]*S+[(0,None)]*K+[(0,None if m else 0) for m in mask]
    result=linprog(cost,A_eq=eq,b_eq=rhs,A_ub=mat,b_ub=ubs,bounds=bounds,method='highs')
    assert result.success,result.message
    return np.maximum(result.x[:T],0.),float(result.fun)
