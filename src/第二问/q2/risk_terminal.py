"""Exact greedy policy MILP. All scenario recourse is policy-determined."""
from .common import *
from .settlement import dispatch
from scipy.sparse import coo_matrix
from scipy.optimize import linprog
import highspy,time

from .risk_value import relaxed_seed
from .risk_seed import improve

def terminal_optimize(nets,p,s0,cuts,weights=None,limit=10.,gap=.001,policy='B',seed=2026,risk='mean'):
    nets=np.atleast_2d(nets);K,T=nets.shape
    assert risk in ['mean','worst'] and policy=='B'
    agg=lambda v:float(np.max(v) if risk=='worst' else np.dot(weights,v))
    assert np.shape(p)==(T,) and np.all(np.asarray(p)>0), 'This formulation requires known strictly positive prices.'
    weights=np.ones(K)/K if weights is None else np.asarray(weights)
    assert np.all(weights>=0) and abs(weights.sum()-1)<1e-10
    # q above max scenario net+B cannot increase charging; removing it saves money.
    qub=np.maximum(0,nets.max(0))+B
    S=K*T;tail=T+10*S;nv=tail+K+(risk=='worst')
    ix=lambda typ,k,t:T+typ*S+k*T+t
    cost=np.zeros(nv);cost[:T]=p;cost[tail:tail+K]=weights if risk=='mean' else 0
    if risk=='worst':cost[-1]=1
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
            cost[e]=weights[k]*5*p[t] if risk=='mean' else 0
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
    if risk=='worst':
        for k in range(K):
            dic={nv-1:1,tail+k:-1}
            dic.update({ix(2,k,t):-5*p[t] for t in range(T)})
            row(dic,l=0)
    a=coo_matrix((vv,(rr,cc)),shape=(len(rl),nv)).tocsr()
    h=highspy.Highs();h.setOptionValue('output_flag',False);h.setOptionValue('threads',1)
    h.setOptionValue('time_limit',limit);h.setOptionValue('mip_rel_gap',gap);h.setOptionValue('random_seed',seed)
    lp=highspy.HighsLp();lp.num_col_=nv;lp.num_row_=len(rl);lp.col_cost_=cost;lp.col_lower_=lo;lp.col_upper_=hi;lp.row_lower_=rl;lp.row_upper_=ru
    lp.integrality_=[highspy.HighsVarType.kInteger if z else highspy.HighsVarType.kContinuous for z in integer]
    lp.a_matrix_.format_=highspy.MatrixFormat.kRowwise;lp.a_matrix_.start_=a.indptr;lp.a_matrix_.index_=a.indices;lp.a_matrix_.value_=a.data
    h.passModel(lp)
    # Guaranteed feasible warm start. Independently replay full candidate, including branch bits.
    seed_q,relaxation_lower_bound=relaxed_seed(nets,p,s0,weights,cuts,risk)
    seeds=[np.clip(np.quantile(nets,.8,axis=0),0,qub),np.clip(seed_q,0,qub)]
    # LP is only a candidate generator; every candidate is scored under the exact policy.
    value=lambda e:max(0.,max((slope*e+intercept for slope,intercept in cuts),default=0.))
    q=min(seeds,key=lambda q:p@q+agg([5*p@dispatch(q,nets[k],s0)['e']+value(dispatch(q,nets[k],s0)['s'][-1]) for k in range(K)]))
    seed_diagnostic=None
    if risk=='worst':
        initial=float(p@q+agg([5*p@dispatch(q,nets[k],s0)['e']+value(dispatch(q,nets[k],s0)['s'][-1]) for k in range(K)]))
        if (initial-relaxation_lower_bound)/max(abs(initial),1e-12)>gap:
            candidate,seed_diagnostic=improve(q,nets,p,s0,cuts,qub)
            cq=float(p@candidate+agg([5*p@dispatch(candidate,nets[k],s0)['e']+value(dispatch(candidate,nets[k],s0)['s'][-1]) for k in range(K)]))
            if cq<initial:q=candidate
    x=np.zeros(nv);x[:T]=q
    if policy=='B':
        for k in range(K):
            r=dispatch(q,nets[k],s0)
            x[tail+k]=value(r['s'][-1])
            for t in range(T):
                vals=[r['c'][t],r['d'][t],r['e'][t],r['w'][t],r['s'][t+1],int(q[t]>=nets[k,t]),int(r['c'][t]>=B-1e-7),int(r['s'][t+1]>=HIGH-1e-7),int(r['d'][t]>=B-1e-7),int(r['s'][t+1]<=LOW+1e-7)]
                for j,val in enumerate(vals):x[ix(j,k,t)]=val
        if risk=='worst':x[-1]=max(5*p@dispatch(q,nets[k],s0)['e']+x[tail+k] for k in range(K))
        h.setSolution(nv,np.arange(nv,dtype=np.int32),x)
    seed_objective=float(cost@x)
    seed_gap=max(0.,(seed_objective-relaxation_lower_bound)/max(abs(seed_objective),1e-12))
    if risk=='worst' and seed_gap<=gap+1e-8:
        trajectories={key:x[T+j*S:T+(j+1)*S].reshape(K,T) for j,key in enumerate(['c','d','e','w','s'])}
        stats=dict(continuous=int(nv-integer.sum()),binary=int(integer.sum()),constraints=len(rl),seconds=0.,incumbent=seed_objective,lower_bound=relaxation_lower_bound,gap=seed_gap,raw_mip_gap=None,relaxation_lower_bound=relaxation_lower_bound,replay_max_error=0.,status='Certified B replay against global LP lower bound',objective_error=0.,seed_diagnostic=seed_diagnostic)
        return q,stats,trajectories
    start=time.time();h.run();elapsed=time.time()-start;info=h.getInfo();sol=h.getSolution()
    if not sol.value_valid:raise RuntimeError('MILP no incumbent '+str(h.getModelStatus()))
    x=np.asarray(sol.col_value);q=x[:T];q=np.maximum(q,0)
    trajectories={key:x[T+j*S:T+(j+1)*S].reshape(K,T) for j,key in enumerate(['c','d','e','w','s'])}
    err=0.;obj=p@q;branch_costs=[]
    for k in range(K):
        if policy=='B':
            r=dispatch(q,nets[k],s0)
            for key in ['c','d','e','w','s']:
                z=r[key][1:] if key=='s' else r[key]
                err=max(err,float(np.max(abs(z-trajectories[key][k]))))
            branch_costs.append(5*p@r['e']+value(r['s'][-1]))
        else:obj+=weights[k]*5*p@trajectories['e'][k]
    obj+=agg(branch_costs)
    certified_bound=max(float(info.mip_dual_bound),relaxation_lower_bound)
    stats=dict(continuous=int(nv-integer.sum()),binary=int(integer.sum()),constraints=len(rl),seconds=elapsed,incumbent=float(obj),lower_bound=certified_bound,gap=max(0.,float((obj-certified_bound)/max(abs(obj),1e-12))),raw_mip_gap=float(info.mip_gap),relaxation_lower_bound=relaxation_lower_bound,replay_max_error=err,status=str(h.getModelStatus()),objective_error=abs(obj-info.objective_function_value))
    if policy=='B' and (err>2e-3 or stats['objective_error']>1e-2):raise AssertionError(stats)
    return q,stats,trajectories
