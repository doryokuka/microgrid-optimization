from bridge import *
from accounting import planning_fee
from stage import stage_optimize
import time
BASE_GAP=.001
STRICT_GAP=.0001

def continuation48(n0,n1,price,s0,reference,limit=2.,gap=BASE_GAP,max_iter=3,seed_plan=None):
    assert n0.shape==n1.shape==(5,144)
    assert np.shape(price)==np.shape(reference)==(144,)
    start=time.time();points=[value_point(n1,price,float(e)) for e in np.arange(LOW,HIGH+1,1200)]
    iterations=[];best=None;global_lower=-np.inf
    for iteration in range(max_iter):
        cuts=[(p['slope'],p['intercept']) for p in points]
        q,root,_=stage_optimize(n0,price,s0,cuts,reference,limit=limit,gap=gap,seed_q_extra=seed_plan if best is None else best['q'])
        r=[dispatch(q,n,s0) for n in n0]
        nxt=[value_point(n1,price,float(x['s'][-1])) for x in r]
        immediate=planning_fee(q,price,reference)+np.mean([5*price@x['e'] for x in r])
        upper=float(immediate+np.mean([v['value'] for v in nxt]))
        global_lower=max(global_lower,root['lower_bound'])
        if best is None or upper<best['upper']:
            best=dict(q=q.copy(),upper=upper,root=root,provisional_q=np.array([v['q'] for v in nxt]),terminal=np.array([x['s'][-1] for x in r]),first24_objective=float(immediate))
        relative=max(0.,(best['upper']-global_lower)/max(abs(best['upper']),1e-12))
        iterations.append(dict(iteration=iteration,upper=upper,best_upper=best['upper'],lower=global_lower,gap=relative,root_gap=root['gap'],root_seconds=root['seconds'],root_replay_error=root['replay_max_error']))
        if relative<=gap+1e-9:break
        new=[v for v in nxt if all(abs(v['E']-p['E'])>1e-5 for p in points)]
        if not new:break
        points+=new
    stats=dict(incumbent=best['upper'],lower_bound=global_lower,gap=relative,seconds=time.time()-start,iterations=iterations,root=best['root'],grid_points=len(points),horizon_steps=288,first24_objective=best['first24_objective'],tail24_objective=best['upper']-best['first24_objective'])
    return best['q'],stats,dict(provisional_q=best['provisional_q'],terminal=best['terminal'])

def plan(n0,n1,price,s0,reference):
    q,stats,detail=continuation48(n0,n1,price,s0,reference)
    # Accuracy is a documented target. A finite feasible result is retained and
    # explicitly flagged if it does not close, instead of silently called optimal.
    stats['target_gap']=BASE_GAP
    stats['target_met']=bool(stats['gap']<=BASE_GAP+1e-9)
    stats['strict_target_gap']=STRICT_GAP
    stats['strict_target_met']=bool(stats['gap']<=STRICT_GAP+1e-9)
    stats['root_time_limit_seconds']=2.
    stats['max_continuation_iterations']=3
    assert np.isfinite(q).all() and np.isfinite(stats['gap'])
    return q,stats,detail
