"""Portable entry for the frozen-center Q3 baseline and Scheme-1 model."""
from pathlib import Path
import argparse
import os
import sys
import json
import hashlib

sys.dont_write_bytecode=True
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
PACKAGE=Path(__file__).resolve().parent
sys.path.insert(0,str(PACKAGE/'model'))

def prepare(output):
    from common import np,inputs,read_npz,savejson,filehash
    from branches import build_release
    from information import residual_library
    fingerprint={str(p.relative_to(PACKAGE)):filehash(p) for p in sorted((PACKAGE/'model').glob('*.py'))+sorted((PACKAGE/'data').glob('*.npz'))}
    fingerprint.update({f'../附件/{x}':filehash(PACKAGE.parent/'附件'/x) for x in ['附件1.xlsx','附件2.xlsx']})
    manifest=output/'prepare_manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text(encoding='utf-8'))
        assert old['source']==fingerprint,'Source changed: select a new output directory.'
        assert all(filehash(output/k)==v for k,v in old['cache'].items()),'Prepared cache changed.'
        return
    output.mkdir(parents=True,exist_ok=True)
    for relative in ['frozen/数据','数据','结果','验证']:(output/relative).mkdir(parents=True,exist_ok=True)
    raw=inputs();centers=read_npz(PACKAGE/'data/centers.npz')['net']
    assert centers.shape==(365,4,288)
    actual=(raw['L']-raw['V']).ravel()
    old0=np.full((365,4,5,144),np.nan);old1=old0.copy()
    n0s=np.empty((334,3,3,144));n1s=np.empty_like(n0s);weights=np.empty((334,3,3));reps=np.empty((334,3,3,108))
    provenance=[];max_baseline_difference=0.
    for day in range(31,365):
        for slot in range(4):
            now=day*144+slot*36
            paths=[]
            for block in (0,1):
                residual,ids=residual_library(centers[:day+1],actual[:now],day,slot,block)
                assert ids[-1]*144+slot*36+(block+1)*144<=now
                path=centers[day,slot,block*144:(block+1)*144]+np.quantile(residual,.8,axis=0,method='linear')
                paths.append(np.repeat(path[None,:],5,axis=0))
            old0[day,slot],old1[day,slot]=paths
            if slot<3:
                a,b,w,meta,bn0,bn1,r=build_release(day,slot,centers[day,slot].copy(),centers[:day].copy(),actual[:now].copy(),np.roll(raw['p'],-slot*36))
                max_baseline_difference=max(max_baseline_difference,float(np.max(abs(bn0-paths[0][0]))),float(np.max(abs(bn1-paths[1][0]))))
                assert max_baseline_difference<1e-8
                n0s[day-31,slot],n1s[day-31,slot],weights[day-31,slot],reps[day-31,slot]=a,b,w,r
                provenance.append(meta)
        if day in (31,180,364):print('Prepared forecast branches through',str(np.datetime64('2025-01-01')+day),flush=True)
    np.savez_compressed(output/'frozen/数据/inputs.npz',**raw)
    np.savez_compressed(output/'frozen/数据/centers_with_correction.npz',net=centers)
    np.savez_compressed(output/'frozen/数据/scenarios_q80.npz',n0=old0,n1=old1)
    np.savez_compressed(output/'frozen/数据/common_startup.npz',**read_npz(PACKAGE/'data/startup.npz'))
    np.savez_compressed(output/'数据/scenarios_lookahead.npz',dayids=np.arange(31,365),n0=n0s,n1=n1s,weights=weights,representative_revision=reps)
    savejson(output/'验证/scenario_provenance.json',provenance)
    files=[p for folder in ['frozen/数据','数据','验证'] for p in (output/folder).glob('*') if p.is_file()]
    savejson(manifest,dict(source=fingerprint,cache={str(p.relative_to(output)):filehash(p) for p in files},scenario_releases=1002,baseline_reconstruction_max_error=max_baseline_difference))

def reference_check(output):
    from common import np,read_npz,inputs,savejson
    from audit_core import replay,fees
    raw=inputs();reports={}
    for treatment,filename in [('scheme1','reference.npz'),('baseline','baseline_reference.npz')]:
        a=read_npz(PACKAGE/'data'/filename)
        error=0.
        assert np.array_equal(a['price'],raw['p'])
        assert np.array_equal(a['dates'],np.arange(np.datetime64('2025-01-01'),np.datetime64('2026-01-01')))
        assert abs(a['s'][0,0]-6000.)<1e-8 and np.max(abs(a['s'][1:,0]-a['s'][:-1,-1]))<1e-8
        for day in range(365):
            r=replay(a['q'][day],raw['L'][day]-raw['V'][day],a['s'][day,0])
            for k in ['c','d','e','w','s']:error=max(error,float(np.max(abs(r[k]-a[k][day]))))
        assert error<2e-3
        f=fees(a['q0'][31:],a['q'][31:],a['e'][31:],raw['p'])
        expected=13393736.619218748 if treatment=='scheme1' else 13455875.529465428
        total=float(f['total_fee'].sum());assert abs(total-expected)<1e-5
        reports[treatment]=dict(calendar_days=365,formal_days=334,independent_replay_max_error=error,formal_total_cost=total,source='Extracted completed reference trajectory; no annual optimization was rerun by this command.')
    savejson(output/'reference_check.json',dict(status='PASS',results=reports));print(json.dumps(reports,ensure_ascii=False),flush=True)

def run_pair(output,days):
    from common import read_npz,savejson
    import run_full
    import audit_core
    prepare(output)
    run_full.run('baseline',end_day=31+days)
    run_full.run('scheme1',end_day=31+days)
    raw=read_npz(output/'frozen/数据/inputs.npz')
    sc=read_npz(output/'frozen/数据/scenarios_q80.npz')
    branches=read_npz(output/'数据/scenarios_lookahead.npz')
    scene=audit_core.scenario_audit(raw,sc,branches)
    reports={}
    for name in ['baseline','scheme1']:
        frozen=read_npz(PACKAGE/'data/baseline_reference.npz')
        reports[name]=audit_core.results_audit(name,raw,frozen,sc,branches,allow_partial=days<334)
    savejson(output/'audit.json',dict(status='PASS',full_annual_rerun=days==334,formal_days_rerun=days,scenario_audit=scene,results=reports))
    print('INDEPENDENT_AUDIT_PASS',days,'formal day(s) for each treatment',flush=True)
    reference_check(output)

def export_json(output):
    from common import np,read_npz,savejson
    a=read_npz(PACKAGE/'data/reference.npz');p=a['price'];q0=a['q0'][31:];q=a['q'][31:];e=a['e'][31:]
    ordinary=(p*q+.5*p*np.abs(q-q0)).sum(1)
    result=dict(source='Existing Scheme-1 reference trajectory; not a new annual optimization',dates=[str(x) for x in a['dates'][31:]],**{k:a[k][31:] for k in ['q0','q','c','d','e','w','s']},price=p,plan_daily_cost=(p*q0).sum(1),ordinary_daily_cost=ordinary,emergency_daily_cost=(5*p*e).sum(1),total_daily_cost=ordinary+(5*p*e).sum(1))
    savejson(output/'result3_export.json',result)
    print('Exported',output/'result3_export.json',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','check-reference','export-json'])
    parser.add_argument('--days',type=int,default=334,help='Formal days from 2025-02-01, 1 to 334; run only.')
    parser.add_argument('--output',type=Path,default=PACKAGE/'outputs')
    args=parser.parse_args()
    if not 1<=args.days<=334:parser.error('--days must be 1 to 334')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    os.environ['Q3_OUTPUT_DIR']=str(output)
    if args.command=='run':run_pair(output,args.days)
    elif args.command=='check-reference':reference_check(output)
    else:export_json(output)

if __name__=='__main__':main()
