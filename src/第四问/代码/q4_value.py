"""LP lower bounds for the unchanged B-policy problem, with explicit risk aggregation."""
from q2.common import *
from q2.settlement import dispatch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def relaxation(nets,p,E,weights,cuts=(),risk='mean'):
    K,T=nets.shape;S=K*T;tail=T+4*S;nv=tail+K+(risk=='worst')
    cost=np.zeros(nv);cost[:T]=p
    if risk=='mean':cost[tail:tail+K]=weights
    else:cost[-1]=1
    rr=[];cc=[];vv=[];rhs=[];ur=[];uc=[];uv=[];ubs=[]
    ix=lambda j,k,t:T+j*S+k*T+t
    def ub(dic,b):
        row=len(ubs)
        for j,v in dic.items():ur.append(row);uc.append(j);uv.append(v)
        ubs.append(b)
    for k in range(K):
        for t in range(T):
            c,u,s,e=[ix(j,k,t) for j in range(4)];row=len(rhs)
            dic={s:1,c:-ETA,u:1/ETA}
            if t:dic[ix(2,k,t-1)]=-1
            for j,v in dic.items():rr.append(row);cc.append(j);vv.append(v)
            rhs.append(E if not t else 0)
            ub({t:-1,c:1,u:-1,e:-1},-nets[k,t])
            if risk=='mean':cost[e]=5*p[t]*weights[k]
        for a,b in cuts:ub({tail+k:-1,ix(2,k,T-1):a},-b)
        if risk=='worst':
            dic={tail+k:1,nv-1:-1}
            dic.update({ix(3,k,t):5*p[t] for t in range(T)})
            ub(dic,0)
    eq=coo_matrix((vv,(rr,cc)),shape=(len(rhs),nv)).tocsr()
    mat=coo_matrix((uv,(ur,uc)),shape=(len(ubs),nv)).tocsr()
    bounds=[(0,None)]*T+[(0,B)]*(2*S)+[(LOW,HIGH)]*S+[(0,None)]*(S+K+(risk=='worst'))
    r=linprog(cost,A_eq=eq,b_eq=rhs,A_ub=mat,b_ub=ubs,bounds=bounds,method='highs')
    assert r.success,r.message
    return r

def relaxed_seed(nets,p,s0,weights,cuts,risk):
    r=relaxation(nets,p,s0,weights,cuts,risk)
    return r.x[:nets.shape[1]],float(r.fun)

def value_point(nets,p,E,risk):
    from q2.risk_terminal import terminal_optimize
    K,T=nets.shape;weights=np.full(K,1/K)
    r=relaxation(nets,p,E,weights,risk=risk)
    q=np.maximum(r.x[:T],0)
    costs=[float(5*p@dispatch(q,n,E)['e']) for n in nets]
    true=float(p@q+(max(costs) if risk=='worst' else np.mean(costs)))
    if true-r.fun>1e-5:
        q2,stat,_=terminal_optimize(nets,p,E,[],limit=5,gap=.00001,risk=risk)
        if stat['incumbent']<true:q=q2;true=stat['incumbent']
    slope=float(r.eqlin.marginals[::T].sum())
    assert true>=r.fun-1e-5
    return dict(E=float(E),value=true,lower=float(r.fun),slope=slope,intercept=float(r.fun-slope*E),q=q,point_gap=max(0.,(true-r.fun)/max(abs(true),1e-12)))

def continuation(n0,n1,p,s0,limit=20,gap=.0001,max_iter=8,risk='mean',pnext=None):
    from q2.risk_terminal import terminal_optimize
    pnext=p if pnext is None else pnext
    start=time.time();points=[value_point(n1,pnext,float(e),risk) for e in np.arange(1200,10801,1200)]
    best=None;iterations=[];bound=-np.inf
    for it in range(max_iter):
        cuts=[(z['slope'],z['intercept']) for z in points]
        q,root,_=terminal_optimize(n0,p,s0,cuts,limit=limit,gap=gap,risk=risk)
        replay=[dispatch(q,n,s0) for n in n0];ends=np.array([z['s'][-1] for z in replay])
        nxt=[value_point(n1,pnext,e,risk) for e in ends]
        c0=np.array([5*p@z['e'] for z in replay]);branches=c0+np.array([z['value'] for z in nxt])
        worst=int(np.argmax(branches))
        true=float(p@q+(branches[worst] if risk=='worst' else branches.mean()))
        today=float(p@q+(c0[worst] if risk=='worst' else c0.mean()))
        bound=max(bound,root['lower_bound'])
        iterations.append(dict(iteration=it,root=root,true_incumbent=true,lower_bound=bound,point_gap_max=max(z['point_gap'] for z in nxt)))
        if best is None or true<best[1]['incumbent']:
            stats=dict(root,incumbent=true,root_approx_incumbent=root['incumbent'],continuation_iterations=it+1,
                       points=len(points),point_gap_max=max(z['point_gap'] for z in points+nxt))
            detail=dict(day1_terminal=ends,provisional_q=np.array([z['q'] for z in nxt]),objective_today=today,
                        objective_tomorrow=true-today,worst_today_branch=worst if risk=='worst' else None,branch_costs=branches)
            best=(q,stats,detail)
        best[1]['lower_bound']=bound;best[1]['gap']=max(0.,(best[1]['incumbent']-bound)/max(abs(best[1]['incumbent']),1e-12))
        if best[1]['gap']<=gap+1e-8:break
        new=[z for z in nxt if all(abs(z['E']-v['E'])>1e-5 for v in points)]
        if not new:break
        points+=new
    q,stats,detail=best
    stats.update(total_seconds=time.time()-start,all_iterations=iterations,value_points=[{k:v for k,v in z.items() if k!='q'} for z in points])
    return q,stats,detail
