from .common import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
SEED=20260912
def centers(L,V,d,forecast,issued=None):
    L=np.asarray(L[:d]);V=np.asarray(V[:d])
    if forecast=='naive':
        today=L[d-7]-V[-3:].mean(0);tomorrow=L[d-6]-V[-3:].mean(0)
        residual=np.array([L[i]-V[i]-L[i-7]+V[i-3:i].mean(0) for i in range(7,d)])
    else:
        assert issued is not None
        # Current forecast is issued at cutoff d-1; residual archive uses only issued past dates.
        past=issued['today'][:d];today=issued['today'][d,0]-issued['today'][d,1]
        tomorrow=issued['tomorrow'][d,0]-issued['tomorrow'][d,1]
        residual=np.array([L[i]-V[i]-past[i,0]+past[i,1] for i in range(7,d)])
    return today,tomorrow,residual
def make(L,V,p,d,forecast,method,issued=None):
    today,tomorrow,residual=centers(L,V,d,forecast,issued)
    ids=np.arange(7,d);meta=dict(day=d,cutoff=str(DATES[d-1]),forecast=forecast,method=method,archive_first=7,archive_last=d-1,archive_size=len(ids),residual_hash=digest(residual),weights=[.2]*5)
    scores=np.maximum(residual,0)@p
    if method=='raw5':
        sel=np.linspace(0,len(ids)-1,5,dtype=int);paths=residual[sel]
        meta.update(selected_history_days=ids[sel].tolist(),pca_components=0,pca_variance=1.,historical_risk_scores=scores[sel].tolist())
    else:
        with threadpool_limits(limits=1):
            pc=PCA(n_components=min(12,len(residual)-1,144),svd_solver='full').fit(residual)
            m=min(len(pc.components_),int(np.searchsorted(np.cumsum(pc.explained_variance_ratio_),.95)+1))
            Z=pc.transform(residual)[:,:m];ncluster=5 if method=='pca5' else 4
            risk=int(np.argmax(scores));keep=np.arange(len(ids)) if ncluster==5 else np.delete(np.arange(len(ids)),risk)
            km=KMeans(n_clusters=ncluster,n_init=20,random_state=SEED,algorithm='lloyd').fit(Z[keep])
            reps=pc.mean_+km.cluster_centers_@pc.components_[:m]
            order=np.argsort(np.maximum(reps,0)@p,kind='stable');reps=reps[order]
            counts=np.bincount(km.labels_,minlength=ncluster)[order]
            paths=reps if ncluster==5 else np.vstack([reps,residual[risk]])
        reconstructed=pc.mean_+Z@pc.components_[:m]
        meta.update(pca_components=m,pca_variance=float(pc.explained_variance_ratio_[:m].sum()),reconstruction_rmse_kwh=float(np.sqrt(np.mean((residual-reconstructed)**2))),cluster_counts=counts.tolist(),extreme_history_day=int(ids[risk]) if ncluster==4 else None,extreme_score=float(scores[risk]) if ncluster==4 else None,extreme_empirical_mass=1/len(ids) if ncluster==4 else None,representative_risk_scores=(np.maximum(paths,0)@p).tolist())
    n0=today+paths;n1=tomorrow+paths
    meta.update(today_center_hash=digest(today),tomorrow_center_hash=digest(tomorrow),scenario_hash=digest(n0),tomorrow_scenario_hash=digest(n1))
    assert n0.shape==n1.shape==(5,144) and np.isfinite(n0).all() and np.isfinite(n1).all()
    return n0,n1,meta
