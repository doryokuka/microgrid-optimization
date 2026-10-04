"""Independent audit helpers; no archived code is imported."""
from q2.final_pipeline import *

def replay_48(n0,n1,p,s0,q,de,risk):
    branches=[]
    for k,n in enumerate(n0):
        x=dispatch(q,n,s0);qp=de['provisional_q'][k]
        costs=np.array([5*p@dispatch(qp,m,x['s'][-1])['e'] for m in n1])
        branches.append(5*p@x['e']+p@qp+(costs.max() if risk=='worst' else costs.mean()))
    return float(p@q+(max(branches) if risk=='worst' else np.mean(branches)))

def audit(a,L,V,p):
    error=0.
    for d in range(len(a['q'])):
        soc=INITIAL_SOC if d==0 else a['s'][d-1,-1]
        expected=dispatch(a['q'][d],L[d]-V[d],soc)
        for k,v in expected.items():error=max(error,float(np.max(abs(v-a[k][d]))))
    assert error<1e-7 and np.min(a['q'])>=-1e-7
    assert np.min(a['s'])>=LOW-1e-7 and np.max(a['s'])<=HIGH+1e-7
    assert np.max(a['c'])<=B+1e-7 and np.max(a['d'])<=B+1e-7 and np.max(a['c']*a['d'])<1e-7
    return dict(status='PASS',days=len(a['q']),physical_replay_max_error=error,initial_soc_kwh=float(a['s'][0,0]))

def ledger():
    result={}
    for directory in ROOT.parent.iterdir():
        if directory.name.startswith(ROOT.name) or not(directory.name.startswith('第二问') or directory.name.startswith('Q2_Python') or directory.name=='附件'):continue
        for f in ([directory] if directory.is_file() else directory.rglob('*')):
            if f.is_file():
                st=f.stat();result[str(f.relative_to(ROOT.parent))]=dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(),size=st.st_size,mtime_ns=st.st_mtime_ns)
    return result
