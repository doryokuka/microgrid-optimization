"""Exact greedy policy MILP. All scenario recourse is policy-determined."""
from .common import *
from .settlement import dispatch
from scipy.sparse import coo_matrix
from scipy.optimize import linprog
import highspy,time

def relaxed_seed(nets,p,s0,weights,cuts):
    K,T=nets.shape;S=K*T;nv=T+4*S+K;cost=np.zeros(nv);cost[:T]=p;cost[-K:]=weights
    rr=[];cc=[];vv=[];rhs=[];ur=[];uc=[];uv=[];ubs=[]
    ix=lambda j,k,t:T+j*S+k*T+t
    for k in range(K):
        for t in range(T):
            c,u,s,e=[ix(j,k,t) for j in range(4)];row=len(rhs)
            dic={s:1,c:-ETA,u:1/ETA}
            if t:dic[ix(2,k,t-1)]=-1
            for j,v in dic.items():rr.append(row);cc.append(j);vv.append(v)
            rhs.append(s0 if not t else 0);row=len(ubs)
            for j,v in {t:-1,c:1,u:-1,e:-1}.items():ur.append(row);uc.append(j);uv.append(v)
            ubs.append(-nets[k,t]);cost[e]=5*p[t]*weights[k]
    for k in range(K):
        for slope,intercept in cuts:
            row=len(ubs)
            for j,v in {T+4*S+k:-1,ix(2,k,T-1):slope}.items():ur.append(row);uc.append(j);uv.append(v)
            ubs.append(-intercept)
    eq=coo_matrix((vv,(rr,cc)),shape=(len(rhs),nv)).tocsr();ub=coo_matrix((uv,(ur,uc)),shape=(len(ubs),nv)).tocsr()
    bounds=[(0,None)]*T+[(0,B)]*(2*S)+[(LOW,HIGH)]*S+[(0,None)]*S+[(0,None)]*K
    r=linprog(cost,A_eq=eq,b_eq=rhs,A_ub=ub,b_ub=ubs,bounds=bounds,method='highs')
    return (r.x[:T],float(r.fun)) if r.success else (np.maximum(0,nets.mean(0)),0.)

def terminal_optimize(nets,p,s0,cuts,weights=None,limit=10.,gap=.001,policy='B',seed=2026):
    nets=np.atleast_2d(nets);K,T=nets.shape
    assert np.shape(p)==(T,) and np.all(np.asarray(p)>0), 'This formulation requires known strictly positive prices.'
    weights=np.ones(K)/K if weights is None else np.asarray(weights)
    assert np.all(weights>=0) and abs(weights.sum()-1)<1e-10
    # q above max scenario net+B cannot increase charging; removing it saves money.
    qub=np.maximum(0,nets.max(0))+B
    S=K*T;tail=T+10*S;nv=tail+K
    ix=lambda typ,k,t:T+typ*S+k*T+t
    cost=np.zeros(nv);cost[:T]=p;cost[tail:]=weights
    lo=np.zeros(nv);hi=np.ones(nv)*np.inf;hi[:T]=qub
    integer=np.zeros(nv,int)
    rr=[];cc=[];vv=[];rl=[];ru=[]
    def row(dic,l=-np.inf,u=np.inf):
        z=len(rl)
        for col,val in dic.items():rr.append(z);cc.append(col);vv.append(val)
        rl.append(l);ru.append(u)
    for k in range(K):
        for t in range(T):
            c,u,e,w,s,b,cb,hb,ub,lb=[ix(j,k,t) for j in range(10)]
            em=max(0,nets[k,t])+(B if policy=='A' else 0);wm=max(0,qub[t]-nets[k,t])+(B if policy=='A' else 0)
            hi[c]=hi[u]=B;hi[e]=em;hi[w]=wm;lo[s]=LOW;hi[s]=HIGH
            for z in [b,cb,hb,ub,lb]:hi[z]=1;integer[z]=1
            cost[e]=weights[k]*5*p[t]
            row({t:1,e:1,u:1,c:-1,w:-1},nets[k,t],nets[k,t])
            dic={s:1,c:-ETA,u:1/ETA}
            if t:dic[ix(4,k,t-1)]=-1
            row(dic,s0 if t==0 else 0,s0 if t==0 else 0)
            row({c:1,b:-B},u=0);row({u:1,b:B},u=B)
            if policy=='B':
                row({w:1,b:-wm},u=0);row({e:1,b:em},u=em)
                row({c:-1,cb:B},u=0);row({s:-1,hb:HIGH-LOW},u=-LOW)
                row({w:1,cb:-wm,hb:-wm},u=0)
                row({u:-1,ub:B},u=0);row({s:1,lb:HIGH-LOW},u=HIGH)
                row({e:1,ub:-em,lb:-em},u=0)
            else:
                # A: one common day-ahead battery schedule, no greedy constraints.
                for z in [cb,hb,ub,lb]:hi[z]=0
                if k:
                    row({c:1,ix(0,0,t):-1},0,0);row({u:1,ix(1,0,t):-1},0,0)
    for k in range(K):
        for slope,intercept in cuts:
            row({tail+k:1,ix(4,k,T-1):-slope},l=intercept)
    a=coo_matrix((vv,(rr,cc)),shape=(len(rl),nv)).tocsr()
    h=highspy.Highs();h.setOptionValue('output_flag',False);h.setOptionValue('threads',1)
    h.setOptionValue('time_limit',limit);h.setOptionValue('mip_rel_gap',gap);h.setOptionValue('random_seed',seed)
    lp=highspy.HighsLp();lp.num_col_=nv;lp.num_row_=len(rl);lp.col_cost_=cost;lp.col_lower_=lo;lp.col_upper_=hi;lp.row_lower_=rl;lp.row_upper_=ru
    lp.integrality_=[highspy.HighsVarType.kInteger if z else highspy.HighsVarType.kContinuous for z in integer]
    lp.a_matrix_.format_=highspy.MatrixFormat.kRowwise;lp.a_matrix_.start_=a.indptr;lp.a_matrix_.index_=a.indices;lp.a_matrix_.value_=a.data
    h.passModel(lp)
    # Guaranteed feasible warm start. Independently replay full candidate, including branch bits.
    seed_q,relaxation_lower_bound=relaxed_seed(nets,p,s0,weights,cuts)
    seeds=[np.clip(np.quantile(nets,.8,axis=0),0,qub),np.clip(seed_q,0,qub)]
    # LP is only a candidate generator; every candidate is scored under the exact policy.
    value=lambda e:max(0.,max(slope*e+intercept for slope,intercept in cuts))
    q=min(seeds,key=lambda q:p@q+sum(weights[k]*(5*p@dispatch(q,nets[k],s0)['e']+value(dispatch(q,nets[k],s0)['s'][-1])) for k in range(K)))
    x=np.zeros(nv);x[:T]=q
    if policy=='B':
        for k in range(K):
            r=dispatch(q,nets[k],s0)
            x[tail+k]=value(r['s'][-1])
            for t in range(T):
                vals=[r['c'][t],r['d'][t],r['e'][t],r['w'][t],r['s'][t+1],int(q[t]>=nets[k,t]),int(r['c'][t]>=B-1e-7),int(r['s'][t+1]>=HIGH-1e-7),int(r['d'][t]>=B-1e-7),int(r['s'][t+1]<=LOW+1e-7)]
                for j,val in enumerate(vals):x[ix(j,k,t)]=val
        h.setSolution(nv,np.arange(nv,dtype=np.int32),x)
    start=time.time();h.run();elapsed=time.time()-start;info=h.getInfo();sol=h.getSolution()
    if not sol.value_valid:raise RuntimeError('MILP no incumbent '+str(h.getModelStatus()))
    x=np.asarray(sol.col_value);q=x[:T];q=np.maximum(q,0)
    trajectories={key:x[T+j*S:T+(j+1)*S].reshape(K,T) for j,key in enumerate(['c','d','e','w','s'])}
    err=0.;obj=p@q
    for k in range(K):
        if policy=='B':
            r=dispatch(q,nets[k],s0)
            for key in ['c','d','e','w','s']:
                z=r[key][1:] if key=='s' else r[key]
                err=max(err,float(np.max(abs(z-trajectories[key][k]))))
            obj+=weights[k]*(5*p@r['e']+value(r['s'][-1]))
        else:obj+=weights[k]*5*p@trajectories['e'][k]
    certified_bound=max(float(info.mip_dual_bound),relaxation_lower_bound)
    stats=dict(continuous=int(nv-integer.sum()),binary=int(integer.sum()),constraints=len(rl),seconds=elapsed,incumbent=float(obj),lower_bound=certified_bound,gap=max(0.,float((obj-certified_bound)/max(abs(obj),1e-12))),raw_mip_gap=float(info.mip_gap),relaxation_lower_bound=relaxation_lower_bound,replay_max_error=err,status=str(h.getModelStatus()),objective_error=abs(obj-info.objective_function_value))
    if policy=='B' and (err>2e-3 or stats['objective_error']>1e-2):raise AssertionError(stats)
    return q,stats,trajectories
