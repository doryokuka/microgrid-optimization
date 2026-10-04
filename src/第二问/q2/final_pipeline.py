"""Pure Q80 startup, fixed causal issuance, optional five-way formal controls.

Default formal model and Jan22 onward startup are Q80. No PCA method is called
on this path. Frozen outcomes never enter decisions. Compare all methods only
from the same Q80 January trajectory; the primary baseline is point.
"""
from .common import *
from .features import naive
from .forecasting import fit_issue,PARAMS
from .procurement import optimize
from .risk_value import continuation as scalar_continuation
from .risk_terminal import terminal_optimize
from .settlement import dispatch
import sys,argparse,platform,importlib.metadata

METHODS=['q80','point','raw5','pca5','robust5'];TARGET=.0001

def make(L,V,p,day,method,issued):
    if method in ['raw5','pca5','robust5']:
        # Optional comparison module. The final Q80 path does not import PCA.
        from .scenarios import make as old_make
        n0,n1,meta=old_make(L,V,p,day,'lightgbm','raw5' if method=='raw5' else 'pca5',issued)
    else:
        history=(L[7:day]-V[7:day])-issued['today'][7:day,0]+issued['today'][7:day,1]
        today=issued['today'][day,0]-issued['today'][day,1]
        tomorrow=issued['tomorrow'][day,0]-issued['tomorrow'][day,1]
        margin=np.quantile(history,.8,axis=0,method='linear') if method=='q80' else np.zeros(144)
        n0=(today+margin)[None];n1=(tomorrow+margin)[None]
        meta=dict(day=day,cutoff=str(DATES[day-1]),archive_first=7,archive_last=day-1,archive_size=day-7,
            residual_hash=digest(history),margin_hash=digest(margin),quantile=.8 if method=='q80' else None,
            scenario_hash=digest(n0),tomorrow_scenario_hash=digest(n1))
    meta.update(method=method,risk='worst' if method=='robust5' else 'mean',weights=[1/len(n0)]*len(n0))
    return n0,n1,meta

def plan(n0,n1,p,s0,day,method):
    start=time.time();attempts=[]
    risk='worst' if method=='robust5' else 'mean'
    if day==364:
        if method in ['raw5','pca5']:q,st,_=optimize(n0,p,s0,limit=20,gap=TARGET)
        else:q,st,_=terminal_optimize(n0,p,s0,[],limit=20,gap=TARGET,risk=risk)
        de=None
    else:
        if method in ['raw5','pca5']:
            from .continuation import continuation
            kw={}
        else:continuation=scalar_continuation;kw=dict(risk=risk)
        q,st,de=continuation(n0,n1,p,s0,limit=20,gap=TARGET,**kw)
        attempts.append(dict(limit=20,gap=st['gap'],incumbent=st['incumbent'],lower_bound=st['lower_bound']))
        if st['gap']>TARGET+1e-8:
            q2,s2,d2=continuation(n0,n1,p,s0,limit=60,gap=TARGET,max_iter=15,**kw)
            attempts.append(dict(limit=60,gap=s2['gap'],incumbent=s2['incumbent'],lower_bound=s2['lower_bound']))
            if method in ['raw5','pca5']:q,st,de=q2,s2,d2
            else:
                lower=max(st['lower_bound'],s2['lower_bound'])
                if s2['incumbent']<st['incumbent']:q,st,de=q2,s2,d2
                st['lower_bound']=lower;st['gap']=max(0,(st['incumbent']-lower)/abs(st['incumbent']))
    st['total_seconds']=time.time()-start;st['attempts']=attempts
    if st['gap']>TARGET+1e-8:raise RuntimeError(f'{method} day {day}: full objective gap {st["gap"]} > {TARGET}; no silent acceptance')
    return q,st,de

def empty():return {k:np.zeros((365,145 if k=='s' else 144)) for k in ['q','c','d','e','w','s']}

def prepare(out,retrain=False,days=365):
    verify_data();L,V,p=inputs();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use an empty output directory when preparing a new run')
    versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','highspy','scikit-learn','lightgbm','openpyxl','matplotlib']}
    savejson(out/'run_config.json',dict(final_model='q80',methods=METHODS,formal_days=max(0,days-31),simulated_days=days,
        startup='Jan1-Jan21 naive24h; Jan22-Jan31 Q80+48h; shared by all formal methods',initial_soc=INITIAL_SOC,
        primary_baseline='point',previous_model_reference='pca5',forecast='retrain' if retrain else 'fixed_causal_issued_cache',
        lightgbm_parameters=PARAMS,quantile=.8,quantile_interpolation='linear',residual_archive_first=7,residual_cutoff='day-1',
        gap=TARGET,continuation_hours=48,year_end='Dec31 original24h',python=sys.version,dependencies=versions,
        platform=platform.platform(),input_checksums=verify_data()))
    if retrain:
        fc=np.full((365,2,144),np.nan);tom=fc.copy();issuance=[]
        for day in range(days):
            fc[day],tom[day],meta=fit_issue(L,V,day);issuance.append(dict(day=day,**meta))
            if day%20==0:print('FORECAST',day,flush=True)
        issued=dict(today=fc,tomorrow=tom);savejson(out/'forecast_issuance.json',issuance)
    else:issued=dict(np.load(DATA/'issued_forecasts.npz'))
    save_npz(out/'issued_forecasts.npz',**issued)
    a=empty();logs=[];start=time.time()
    for day in range(31):
        soc=INITIAL_SOC if day==0 else a['s'][day-1,-1]
        if day<21:
            net=naive(L[:day],True)-naive(V[:day],False)
            q,st,_=optimize(net,p,soc,limit=2,gap=.005)
            meta=dict(method='naive_cold_start',cutoff=str(DATES[day-1]) if day else None,forecast_prior='5000kW load, zero PV' if day==0 else None)
        else:
            n0,n1,meta=make(L,V,p,day,'q80',issued);q,st,_=plan(n0,n1,p,soc,day,'q80')
        a['q'][day]=q
        for k,v in dispatch(q,L[day]-V[day],soc).items():a[k][day]=v
        logs.append(dict(day=day,date=str(DATES[day]),start_soc=float(soc),q_hash=digest(q),scenario=meta,solver=st))
    save_npz(out/'common_startup.npz',**a,records_json=np.array(json.dumps(logs)))
    savejson(out/'startup.json',dict(days=31,initial_soc=INITIAL_SOC,formal_initial_soc=float(a['s'][30,-1]),
        january_normal_cost=float((a['q'][:31]*p).sum()),january_emergency_cost=float(5*(a['e'][:31]*p).sum()),
        january_total_cost=float(((a['q'][:31]+5*a['e'][:31])*p).sum()),seconds=time.time()-start,logs=logs))
    config=json.loads((out/'run_config.json').read_text())
    config['issued_forecasts_sha256']=hashlib.sha256((out/'issued_forecasts.npz').read_bytes()).hexdigest()
    config['common_startup_sha256']=hashlib.sha256((out/'common_startup.npz').read_bytes()).hexdigest()
    savejson(out/'run_config.json',config)
    print('STARTUP COMPLETE',float(a['s'][30,-1]),flush=True)

def run_model(out,method,days=365):
    verify_data();L,V,p=inputs();issued=dict(np.load(out/'issued_forecasts.npz'));path=out/f'{method}.npz'
    K=1 if method in ['q80','point'] else 5
    if path.exists():
        old=np.load(path);a={k:old[k].copy() for k in empty()};logs=json.loads(str(old['records_json']))
        qs=old['provisional_q'].copy();ends=old['scenario_terminals'].copy()
        assert [r['day'] for r in logs]==list(range(31,31+len(logs))) and len(logs)<=days-31
        common=np.load(out/'common_startup.npz')
        for key in a:assert np.array_equal(a[key][:31],common[key][:31]),'Incompatible January checkpoint'
        for row in logs:
            assert row['q_hash']==digest(a['q'][row['day']]) and row['scenario']['method']==method,'Incompatible method checkpoint'
    else:
        initial=np.load(out/'common_startup.npz');a=empty()
        for k in a:a[k][:31]=initial[k][:31]
        logs=[];qs=np.zeros((365,K,144));ends=np.zeros((365,K))
    for day in range(31+len(logs),days):
        n0,n1,meta=make(L,V,p,day,method,issued);soc=a['s'][day-1,-1]
        try:q,st,de=plan(n0,n1,p,soc,day,method)
        except Exception as exc:
            savejson(out/f'failure_{method}.json',dict(day=day,message=str(exc),completed_days=len(logs)))
            raise
        a['q'][day]=q
        for k,v in dispatch(q,L[day]-V[day],soc).items():a[k][day]=v
        if de is not None:qs[day]=de['provisional_q'];ends[day]=de['day1_terminal']
        logs.append(dict(day=day,date=str(DATES[day]),cutoff=str(DATES[day-1]),start_soc=float(soc),scenario=meta,
            scenario_hash=digest(n0),tomorrow_scenario_hash=digest(n1),q_hash=digest(q),solver=st))
        save_npz(path,**a,provisional_q=qs,scenario_terminals=ends,records_json=np.array(json.dumps(logs)))
        savejson(out/f'progress_{method}.json',dict(completed_days=len(logs),last_date=str(DATES[day]),gap=st['gap']))
        if day%20==0 or day==days-1:print(method,str(DATES[day]),'formal cost',round(float(((a['q'][31:day+1]+5*a['e'][31:day+1])*p).sum()),2),'gap',st['gap'],flush=True)
    savejson(out/f'{method}_solver.json',logs)
    # Portable detailed CSV. Comprehensive comparison is generated by release analysis.
    rows=[]
    for day in range(31,31+len(logs)):
        rows.append(dict(method=method,date=str(DATES[day]),normal_cost=float(a['q'][day]@p),emergency_cost=float(5*a['e'][day]@p),
            total_cost=float((a['q'][day]+5*a['e'][day])@p),emergency_kwh=float(a['e'][day].sum()),spill_kwh=float(a['w'][day].sum()),
            initial_soc=float(a['s'][day,0]),end_soc=float(a['s'][day,-1]),gap=logs[day-31]['solver']['gap']))
    csvout(out/f'{method}_daily.csv',rows)
    print('COMPLETE',method,len(logs),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs');parser.add_argument('--method',choices=METHODS,default='q80')
    parser.add_argument('--all-models',action='store_true');parser.add_argument('--retrain',action='store_true')
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();out=args.output.resolve();days=33 if args.smoke else 365
    if out in [ROOT,DATA,ROOT/'q2',ROOT/'字体']:parser.error('Use a separate result directory')
    if not args.resume:prepare(out,args.retrain,days)
    else:
        config=json.loads((out/'run_config.json').read_text());assert config['simulated_days']==days,'Resume with matching smoke/full setting'
        assert not args.retrain,'Resume uses the same already issued forecasts'
        assert config['final_model']=='q80' and config['quantile']==.8 and config['gap']==TARGET
        assert config['lightgbm_parameters']==PARAMS and config['primary_baseline']=='point'
        assert config['input_checksums']==verify_data()
        for file,key in [('issued_forecasts.npz','issued_forecasts_sha256'),('common_startup.npz','common_startup_sha256')]:
            assert hashlib.sha256((out/file).read_bytes()).hexdigest()==config[key],f'Changed run input: {file}'
    if args.prepare_only:return
    chosen=METHODS if args.all_models else [args.method]
    for method in chosen:run_model(out,method,days)
    from .export import export_outputs
    export_outputs(out,chosen,days)
