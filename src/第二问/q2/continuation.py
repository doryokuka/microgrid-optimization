from .terminal import terminal_optimize
from .common import *
from .procurement import optimize
from .settlement import dispatch
from scipy.sparse import coo_matrix
from scipy.optimize import linprog

def value_point(nets,p,E):
 K,T=nets.shape;S=K*T;nv=T+4*S;cost=np.zeros(nv);cost[:T]=p
 rr=[];cc=[];vv=[];rhs=[];ur=[];uc=[];uv=[];ubs=[]
 ix=lambda j,k,t:T+j*S+k*T+t
 for k in range(K):
  for t in range(T):
   c,u,s,e=[ix(j,k,t) for j in range(4)];row=len(rhs);dic={s:1,c:-ETA,u:1/ETA}
   if t:dic[ix(2,k,t-1)]=-1
   for j,v in dic.items():rr.append(row);cc.append(j);vv.append(v)
   rhs.append(E if not t else 0);row=len(ubs)
   for j,v in {t:-1,c:1,u:-1,e:-1}.items():ur.append(row);uc.append(j);uv.append(v)
   ubs.append(-nets[k,t]);cost[e]=5*p[t]/K
 eq=coo_matrix((vv,(rr,cc)),shape=(len(rhs),nv)).tocsr();ub=coo_matrix((uv,(ur,uc)),shape=(len(ubs),nv)).tocsr()
 r=linprog(cost,A_eq=eq,b_eq=rhs,A_ub=ub,b_ub=ubs,bounds=[(0,None)]*T+[(0,B)]*(2*S)+[(LOW,HIGH)]*S+[(0,None)]*S,method='highs');assert r.success
 # Feasible greedy replay certifies the LP value whenever its objective matches.
 q=np.maximum(r.x[:T],0);true=float(p@q+sum(p@dispatch(q,nets[k],E)['e'] for k in range(K)));slope=float(r.eqlin.marginals[::T].sum())
 if true-r.fun>1e-5:
  q,stat,_=optimize(nets,p,E,limit=5,gap=.00001);true=stat['incumbent']
 return dict(E=float(E),value=true,lower=float(r.fun),slope=slope,intercept=float(r.fun-slope*E),q=q,point_gap=max(0.,(true-r.fun)/max(abs(true),1e-12)))

def continuation(n0,n1,p,s0,limit=10,gap=.001,max_iter=8,initial_grid=None):
 start=time.time();grid=np.arange(1200,10801,1200) if initial_grid is None else initial_grid
 points=[value_point(n1,p,float(e)) for e in grid];iterations=[];best=None
 for it in range(max_iter):
  cuts=[(z['slope'],z['intercept']) for z in points]
  q,stat,tr=terminal_optimize(n0,p,s0,cuts,limit=limit,gap=gap)
  ends=np.array([dispatch(q,n,s0)['s'][-1] for n in n0]);nxt=[value_point(n1,p,e) for e in ends]
  today=float(p@q+sum(p@dispatch(q,n,s0)['e'] for n in n0));true=today+np.mean([z['value'] for z in nxt]);lb=stat['lower_bound'];certgap=max(0.,(true-lb)/max(abs(true),1e-12))
  iterations.append(dict(iteration=it,root=stat,true_incumbent=true,lower_bound=lb,certified_gap=certgap,terminal_error=max(z['value']-max(a*z['E']+b for a,b in cuts) for z in nxt),point_gap_max=max(z['point_gap'] for z in nxt)))
  if best is None or true<best[1]['incumbent']:
   detail=dict(day1_terminal=ends,provisional_q=np.array([z['q'] for z in nxt]),objective_today=today,objective_tomorrow=true-today)
   stats=dict(stat,incumbent=true,lower_bound=lb,gap=certgap,root_approx_incumbent=stat['incumbent'],continuation_iterations=it+1,points=len(points),total_seconds=time.time()-start,point_gap_max=max(z['point_gap'] for z in points+nxt),iterations=iterations.copy())
   best=(q,stats,detail)
  if certgap<=gap+1e-8:break
  new=[z for z in nxt if all(abs(z['E']-v['E'])>1e-5 for v in points)]
  if not new:break
  points+=new
 q,stats,detail=best;stats['total_seconds']=time.time()-start;stats['all_iterations']=iterations
 stats['value_points']=[{k:v for k,v in z.items() if k!='q'} for z in points]
 return q,stats,detail

