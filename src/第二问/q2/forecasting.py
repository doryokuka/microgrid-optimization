from .common import *
from .features import features as legacy_features, naive
from lightgbm import LGBMRegressor
import argparse
PARAMS=dict(n_estimators=120,learning_rate=.04,num_leaves=15,max_depth=-1,min_child_samples=144,reg_lambda=20.,reg_alpha=.1,colsample_bytree=.9,subsample=1.,verbosity=-1,n_jobs=1,random_state=20260912)

def feature_pair(L,V,d,target):
    L=np.asarray(L[:d]);V=np.asarray(V[:d]);assert len(L)==len(V)==d
    own=L if target==0 else V;other=V if target==0 else L
    # All calendar attributes are deterministic, known at issuance; no weather or in-day actuals.
    x=legacy_features(own,d,target==0,True)
    z=[];net=L-V
    for h in [other,net]:
        fallback=h[-min(3,d):].mean(0) if d else np.zeros(144)
        for lag in [1,2,7,14,28]:z.append(h[-lag] if d>=lag else fallback)
        for win in [3,7,14,28]:
            a=h[max(0,d-win):d]
            z.extend([a.mean(0) if d else fallback,a.std(0) if d else np.zeros(144)])
        if d:
            totals=h.sum(1)
            z.extend([np.full(144,totals[-1]),np.full(144,totals[-min(7,d):].mean()),np.full(144,totals[-min(28,d):].std())])
        else:z.extend([np.zeros(144)]*3)
    return np.column_stack([x]+z)

def fit_issue(L,V,d,caches=None):
    L=np.asarray(L[:d]);V=np.asarray(V[:d])
    if d<21:
        today=np.array([naive(L,True),naive(V,False)])
        tomorrow=np.array([naive(np.vstack([L,today[0]]),True),naive(np.vstack([V,today[1]]),False)])
        return today,tomorrow,dict(train_first=None,train_last=None,fallback=True)
    ids=np.arange(max(7,d-180),d);regs=[];today=[]
    for j,y in enumerate([L,V]):
        X=np.concatenate([feature_pair(L[:i],V[:i],i,j) for i in ids]) if caches is None else caches[j][ids].reshape(-1,caches[j].shape[-1])
        reg=LGBMRegressor(**PARAMS).fit(X,y[ids].reshape(-1));regs.append(reg)
        today.append(np.maximum(reg.booster_.predict(feature_pair(L,V,d,j),num_threads=1),0))
    today=np.array(today)
    # Recursive causal proxy uses today's issued forecast, never actual day d.
    if d<364:
        lp=np.vstack([L,today[0]]);vp=np.vstack([V,today[1]])
        tomorrow=np.array([np.maximum(reg.booster_.predict(feature_pair(lp,vp,d+1,j),num_threads=1),0) for j,reg in enumerate(regs)])
    else:tomorrow=np.zeros((2,144))
    return today,tomorrow,dict(train_first=int(ids[0]),train_last=int(ids[-1]),training_days=len(ids),features=[feature_pair(L,V,d,j).shape[1] for j in range(2)],fallback=False)
