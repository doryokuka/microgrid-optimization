"""Issue-time information and historical-only intraday load correction."""
from q4_3_bridge import *

def base_window(base72, day, slot):
    offset = HOURS[slot]*6
    return base72[day].reshape(2, 432)[:, offset:offset+288].copy()

def official_pv(forecasts_kw, observed_pv_prefix, day, slot):
    """Hourly point forecasts interpolated at ten-minute right endpoints, as in Q2."""
    now = day*144+HOURS[slot]*6
    assert len(observed_pv_prefix) == now
    anchor = observed_pv_prefix[-1] if now else 0.
    knots = np.r_[anchor, forecasts_kw[day, slot]/6.]
    return np.interp(np.arange(1, 145), np.arange(0, 145, 6), knots)

def observed_features(load_prefix, issued_today, offset):
    assert len(load_prefix) == offset and offset in (36,72,108)
    residual = load_prefix-issued_today[:offset]
    return np.array([residual.mean(), residual[-6:].mean()])

def fit_correction(day, slot, load_completed_days, base72):
    """8 six-hour lead blocks; ridge penalty selected on earlier/later historical days.

    Both coefficient fitting and selection use only dates completed before midnight
    of the issue day. Previously issued historical corrections are never refitted.
    """
    assert len(load_completed_days) == day
    if slot == 0:
        return np.zeros((2,8)), dict(enabled=False, reason='no intraday observations at midnight')
    offset = HOURS[slot]*6
    ids = [j for j in range(max(7,day-180),day) if j*144+offset+288 <= day*144]
    meta = dict(enabled=True, train_days=ids, max_target_end=timestamp((ids[-1]*144+offset+288)) if ids else None)
    if len(ids) < 14:
        return np.zeros((2,8)), dict(meta, reason='fewer than 14 fully observed historical windows', alpha=None)
    flat = load_completed_days.ravel()
    X = np.array([observed_features(load_completed_days[j,:offset], base72[j,0,0],offset) for j in ids])
    Y = np.array([(flat[j*144+offset:j*144+offset+288]-base_window(base72,j,slot)[0]).reshape(8,36).mean(1) for j in ids])
    split = max(8, int(len(ids)*.7))
    def ridge(x,y,alpha):
        scale = np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-6)
        z = x/scale
        return np.linalg.solve(z.T@z+alpha*np.eye(2),z.T@y)/scale[:,None]
    scores = {str(a):float(np.mean((X[split:]@ridge(X[:split],Y[:split],a)-Y[split:])**2)) for a in (1.,10.,100.)}
    alpha = float(min(scores,key=scores.get))
    coef = ridge(X,Y,alpha)
    return coef, dict(meta, alpha=alpha, validation_mse=scores, coefficients=coef.tolist(), split=split)

def issue_centers(day, slot, raw, base72, with_correction, coefficients):
    offset = HOURS[slot]*6
    now = day*144+offset
    center = base_window(base72,day,slot)
    center[1,:144] = official_pv(raw['official_kw'],raw['V'].ravel()[:now],day,slot)
    delta = np.zeros(288)
    if with_correction and slot:
        feat = observed_features(raw['L'][day,:offset],base72[day,0,0],offset)
        delta = np.repeat(feat@coefficients,36)
        center[0] = np.maximum(center[0]+delta,0.)
    return center[0]-center[1], center, delta

def residual_library(centers, actual_net, day, slot, block):
    """Only issued forecasts whose target block has fully elapsed may enter."""
    offset = HOURS[slot]*6
    now = day*144+offset
    ids = [j for j in range(7,day) if j*144+offset+(block+1)*144 <= now]
    rows=[]
    for j in ids:
        start=j*144+offset+block*144
        rows.append(actual_net[start:start+144]-centers[j,slot,block*144:(block+1)*144])
    return np.array(rows), ids

def pca5(residual, price):
    """Same PCA cap, 95% target, K-means options and equal weights as frozen Q2."""
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    assert residual.ndim == 2 and residual.shape[1] == 144 and len(residual)>=5
    with threadpool_limits(limits=1):
        pc=PCA(n_components=min(12,len(residual)-1,144),svd_solver='full').fit(residual)
        m=min(len(pc.components_),int(np.searchsorted(np.cumsum(pc.explained_variance_ratio_),.95)+1))
        z=pc.transform(residual)[:,:m]
        km=KMeans(n_clusters=5,n_init=20,random_state=20260912,algorithm='lloyd').fit(z)
        paths=pc.mean_+km.cluster_centers_@pc.components_[:m]
        order=np.argsort(np.maximum(paths,0)@price,kind='stable')
    return paths[order], dict(components=m,variance=float(pc.explained_variance_ratio_[:m].sum()),counts=np.bincount(km.labels_,minlength=5)[order].tolist(),weights=[.2]*5,residual_hash=digest(residual))

def scenarios(centers, actual_net, day, slot, price):
    nets=[];logs=[]
    for block in (0,1):
        residual,ids=residual_library(centers,actual_net,day,slot,block)
        paths,log=pca5(residual,price)
        nets.append(centers[day,slot,block*144:(block+1)*144]+paths)
        logs.append(dict(log,history_days=ids,last_target_end=timestamp(ids[-1]*144+HOURS[slot]*6+(block+1)*144)))
    return nets[0],nets[1],logs
