"""Q4-3: user's final two-stage Q80 method + causal rolling LightGBM prices."""
import os,sys
from pathlib import Path
from run_q4_2 import load_inputs,PARAMS,WORK
from q4_3_bridge import *
from q2 import forecasting as forecast
from q2 import features as net_features
from q4_3_information import fit_correction,base_window,official_pv,observed_features
from q4_3_branches import build_release,mature_residuals
from q4_3_lookahead import solve_lookahead
from q4_3_planner import plan
from q4_3_accounting import components
from lightgbm import LGBMRegressor
from openpyxl import load_workbook
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

OUT=WORK/'结果4-3'
OUT.mkdir(exist_ok=True)

def protocol():
    paths=[WORK.parent/'附件'/f'附件{i}.xlsx' for i in (2,3,4)]
    paths += [SOURCE/'data/issued_forecasts.npz',SOURCE/'data/calendar.npz',WORK/'结果4-2/q4_2.npz']
    paths += sorted((SOURCE/'q2').glob('*.py'))
    paths += sorted(Path(__file__).parent.glob('q4_3_*.py'))
    paths += [Path(__file__),Path(__file__).parent/'run_q4_2.py',Path(__file__).parent/'q4_value.py']
    manifest={p.relative_to(WORK.parent).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    target=OUT/'输入与模型哈希.json'
    if target.exists():
        saved=json.loads(target.read_text(encoding='utf-8'))
        changed=[p for p in sorted(manifest.keys() | saved.keys()) if manifest.get(p)!=saved.get(p)]
        if changed:
            raise RuntimeError('输入或代码内容已变化，不能复用旧检查点。差异文件：\n'+'\n'.join(changed))
    else:savejson(target,manifest)

def price_features(prefix,now):
    """288 target rows; every measurement comes from prefix strictly before now."""
    assert len(prefix)==now and now>=144
    target=now+np.arange(288);phase=target%144;day=target//144
    cols=[phase/144,np.arange(288)/144,np.full(288,(now%144)/144),(day+2)%7,
          np.sin(2*np.pi*day/365),np.cos(2*np.pi*day/365)]
    for k in range(1,5):cols.extend([np.sin(2*np.pi*k*phase/144),np.cos(2*np.pi*k*phase/144)])
    # Most recent OBSERVED matching slot, including elapsed current-day observations.
    latest=now-1-((now-1-phase)%144)
    for lag in [1,2,3,7,14,28]:
        idx=target-lag*144;idx=np.where((idx>=0)&(idx<now),idx,latest)
        cols.append(prefix[idx])
    for win in [3,7,14,28]:
        idx=latest[:,None]-144*np.arange(win)[None,:]
        vals=np.where(idx>=0,prefix[np.maximum(idx,0)],np.nan)
        cols.extend([np.nanmean(vals,1),np.nanstd(vals,1)])
    for win in [1,6,36,144]:cols.append(np.full(288,prefix[-win:].mean()))
    return np.column_stack(cols)

def fit_price_day(day,flat,X):
    cutoff=day*144
    pairs=[(h,s) for h in range(max(7,day-180),day) for s in range(4) if h*144+s*36+288<=cutoff]
    train=np.concatenate([X[h,s] for h,s in pairs])
    labels=np.concatenate([flat[h*144+s*36:h*144+s*36+288] for h,s in pairs])
    reg=LGBMRegressor(**PARAMS,force_col_wise=True).fit(train,labels)
    prices=np.maximum(reg.booster_.predict(X[day].reshape(-1,32),num_threads=1),1e-6).reshape(4,288)
    return prices,dict(day=day,training_cutoff=timestamp(cutoff),latest_label_end=timestamp(max(h*144+s*36+288 for h,s in pairs)),training_windows=len(pairs))

def prepare(L,V,P,issued,end_day=365):
    assert 32<=end_day<=365
    scope='' if end_day==365 else f'_through_{end_day}'
    path=OUT/f'forecast_inputs{scope}.npz'
    if path.exists():
        with np.load(path) as z:return dict(z)
    # Extend known calendar only; no 2026 observations are loaded.
    for key,values in dict(weekday=[3,4],holiday=[1,0],adjust=[0,0],daytype=[2,0]).items():
        if len(net_features.CAL[key])==365:net_features.CAL[key]=np.r_[net_features.CAL[key],values]
    bp=OUT/f'base72{scope}.npz'
    if bp.exists():
        with np.load(bp) as z:base=z['base'].copy();done=int(z['done'])
    else:base=np.full((365,2,3,144),np.nan);done=0
    caches=[np.array([forecast.feature_pair(L,V,d,j) for d in range(end_day)]) for j in (0,1)]
    start=time.time();refit_errors=[]
    for day in range(done,end_day):
        l=L[:day].copy();v=V[:day].copy();regs=[]
        if day>=21:
            ids=np.arange(max(7,day-180),day)
            for j,y in enumerate((L,V)):
                regs.append(LGBMRegressor(**forecast.PARAMS).fit(caches[j][ids].reshape(-1,caches[j].shape[-1]),y[ids].ravel()))
        for lead in range(3):
            pred=np.array([forecast.naive(l,True),forecast.naive(v,False)]) if day<21 else np.array([
                np.maximum(reg.booster_.predict(forecast.feature_pair(l,v,day+lead,j),num_threads=1),0) for j,reg in enumerate(regs)])
            # Preserve frozen issued predictions across LightGBM platform versions.
            # Only the extra third-day proxy must be refitted locally.
            if lead==0 or (lead==1 and day<364):
                frozen=issued['today' if lead==0 else 'tomorrow'][day]
                refit_errors.append(dict(day=day,lead=lead,max_abs_error=float(np.max(abs(pred-frozen)))))
                pred=frozen.copy()
            base[day,:,lead]=pred;l=np.vstack([l,pred[0]]);v=np.vstack([v,pred[1]])
        assert np.max(abs(base[day,:,0]-issued['today'][day]))<1e-8
        if day<364:assert np.max(abs(base[day,:,1]-issued['tomorrow'][day]))<1e-8
        if day%10==0 or day==end_day-1:
            save_npz(bp,base=base,done=np.array(day+1))
            print('NET_FORECAST',day+1,'/365',round(time.time()-start,1),flush=True)
    if refit_errors:savejson(OUT/'冻结预测继承.json',dict(first48h='Original Q2 issued archive copied exactly through Dec30; Dec31 tomorrow and all third days are causal recursive proxies',runtime_refit_comparison=refit_errors))
    del caches
    wb=load_workbook(WORK.parent/'附件/附件3.xlsx',read_only=True,data_only=True)
    rows=list(wb.active.values)[1:];wb.close()
    assert len(rows)==1460
    for day in range(365):
        assert str(datetime.strptime(rows[day*4][0],'%Y-%m-%d').date())==str(DATES[day])
        assert [r[1] for r in rows[day*4:day*4+4]]==['0:00','6:00','12:00','18:00']
    official=np.array([r[2:] for r in rows],float).reshape(365,4,24)
    centers=np.empty((365,4,288));correction_logs=[]
    for day in range(end_day):
        for slot in range(4):
            now=day*144+slot*36
            coef,meta=fit_correction(day,slot,L[:day],base[:day+1])
            pair=base_window(base[:day+1],day,slot)
            pair[1,:144]=official_pv(official[:day+1,:slot+1],V.ravel()[:now],day,slot)
            if slot:pair[0]=np.maximum(pair[0]+np.repeat(observed_features(L[day,:slot*36],base[day,0,0],slot*36)@coef,36),0)
            centers[day,slot]=pair[0]-pair[1]
            correction_logs.append(dict(day=day,slot=slot,**meta))
    savejson(OUT/'负荷校正记录.json',correction_logs)
    # Historical feature rows themselves have a strict observation-prefix interface.
    flat=P.ravel();X=np.empty((365,4,288,32))
    for day in range(1,end_day):
        for slot in range(4):
            now=day*144+slot*36;X[day,slot]=price_features(flat[:now],now)
    pp=OUT/f'price_issued{scope}.npz'
    if pp.exists():
        with np.load(pp) as z:prices=z['prices'].copy();done=int(z['done'])
    else:prices=np.full((365,4,288),np.nan);done=31
    price_logs=[]
    # Independent past-only training windows; decision/SOC simulation stays sequential.
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=pool.map(lambda day:fit_price_day(day,flat,X),range(done,end_day))
        for day,(pred,meta) in zip(range(done,end_day),results):
            prices[day]=pred;price_logs.append(meta)
            if day%5==0 or day==end_day-1:
                save_npz(pp,prices=prices,done=np.array(day+1))
                print('PRICE_FORECAST',day+1,'/365',round(time.time()-start,1),flush=True)
    # Reconstruct identical training-cutoff metadata for already saved releases.
    for day in range(31,done):
        pairs=[(h,s) for h in range(max(7,day-180),day) for s in range(4) if h*144+s*36+288<=day*144]
        price_logs.append(dict(day=day,training_cutoff=timestamp(day*144),latest_label_end=timestamp(max(h*144+s*36+288 for h,s in pairs)),training_windows=len(pairs)))
    price_logs.sort(key=lambda r:r['day'])
    savejson(OUT/'电价训练记录.json',price_logs)
    save_npz(path,centers=centers,prices=prices)
    return dict(centers=centers,prices=prices)

def run(end_day=365):
    if not 32<=end_day<=365:raise ValueError('end-day must be 32..365 (32 means one formal day)')
    protocol()
    L,V,prior,P,issued=load_inputs()
    data=prepare(L,V,P,issued,end_day=end_day);centers=data['centers'];prices=data['prices'];actual=(L-V).ravel()
    path=OUT/'q4_3.npz'
    if path.exists():
        with np.load(path) as z:a=dict(z)
        records=json.loads(str(a.pop('records_json')))
    else:
        with np.load(WORK/'结果4-2/q4_2.npz') as z:warm=dict(z)
        a={k:np.full((365,145 if k=='s' else 144),np.nan) for k in ('q0','q','c','d','e','w','s')}
        for k in a:a[k][:31]=warm['q' if k=='q0' else k][:31]
        a['plans']=np.full((365,4,144),np.nan)
        a['branch_q']=np.full((365,3,3,144),np.nan)
        a['branch_soc6']=np.full((365,3,3),np.nan)
        records=[]
    assert len(records)%4==0
    start=time.time()
    for day in range(31+len(records)//4,end_day):
        a['s'][day,0]=a['s'][day-1,-1]
        for slot in range(4):
            off=slot*36;now=day*144+off;s0=float(a['s'][day,off]);pred=prices[day,slot]
            ref=np.full(144,np.nan)
            if slot:ref[:144-off]=a['q0'][day,off:]
            if slot<3:
                n0,n1,weights,provenance,*_=build_release(day,slot,centers[day,slot].copy(),centers[:day].copy(),actual[:now].copy(),pred[:144])
                q,stats,detail=solve_lookahead(n0,n1,pred[:144],s0,weights,reference=None if slot==0 else ref,pnext=pred[144:])
                a['branch_q'][day,slot]=detail['branch_q'];a['branch_soc6'][day,slot]=detail['soc6']
            else:
                nets=[]
                for block in (0,1):
                    residual,ids=mature_residuals(centers[:day],actual[:now],day,slot,slot,block)
                    nets.append(np.repeat((centers[day,slot,block*144:(block+1)*144]+np.quantile(residual,.8,axis=0,method='linear'))[None],5,axis=0))
                q,stats,detail=plan(*nets,pred[:144],s0,ref,pnext=pred[144:]);provenance=None
            assert np.isfinite(q).all() and q.min()>=-1e-7
            if slot==0:a['q0'][day]=q.copy()
            a['plans'][day,slot]=q
            z=dispatch(q[:36],actual[now:now+36],s0)
            a['q'][day,off:off+36]=q[:36]
            for k in ('c','d','e','w'):a[k][day,off:off+36]=z[k]
            a['s'][day,off:off+37]=z['s']
            fee=components(a['q0'][day,off:off+36],q[:36],z['e'],P[day,off:off+36])
            records.append(dict(day=day,hour=slot*6,initial_soc=s0,final_soc=float(z['s'][-1]),price_prefix_end=timestamp(now),
                solver=stats,scenario_provenance=provenance,total_fee=float(fee['total_fee'].sum())))
        save_npz(path,**a,records_json=np.array(json.dumps(records,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())))
        if day%5==0 or day in (31,364):
            progress=dict(completed=day-30,total=334,date=str(DATES[day]),cost=sum(x['total_fee'] for x in records),elapsed=round(time.time()-start,1))
            savejson(OUT/'运行进度.json',progress);print('MPC',json.dumps(progress),flush=True)
    if end_day==365:verify(a,records,L,V,P,centers,prices)
    protocol()

def verify(a,records,L,V,P,centers,prices):
    for key in ('q0','q','c','d','e','w','s'):assert np.isfinite(a[key]).all()
    replay=dispatch(a['q'].ravel(),(L-V).ravel(),6000.)
    errors={k:float(np.max(abs(replay[k]-a[k].ravel()))) for k in ('c','d','e','w')}
    errors['s']=float(np.max(abs(replay['s'][1:]-a['s'][:,1:].ravel())))
    assert max(errors.values())<1e-5 and np.max(abs(a['s'][1:,0]-a['s'][:-1,-1]))<1e-8
    assert np.max(abs(a['branch_soc6'][31:,:,1:]-a['branch_soc6'][31:,:,:1]))<1e-5
    checks=[]
    for day in [31,180,364]:
        for slot in range(4):
            now=day*144+slot*36;bad=P.ravel().copy();bad[now:]=np.nan
            assert np.array_equal(price_features(P.ravel()[:now],now),price_features(bad[:now],now))
            if slot<3:
                bc=centers.copy();bc[day,slot+1:]=np.nan;bc[day+1:]=np.nan
                net=(L-V).ravel();bn=net.copy();bn[now:]=np.nan
                x=build_release(day,slot,centers[day,slot],centers[:day],net[:now],prices[day,slot,:144])
                y=build_release(day,slot,bc[day,slot],bc[:day],bn[:now],prices[day,slot,:144])
                assert all(np.array_equal(x[k],y[k]) for k in [0,1,2,4,5,6])
            checks.append(dict(day=day,hour=slot*6,passed=True))
    ledger=components(a['q0'][31:],a['q'][31:],a['e'][31:],P[31:])
    assert abs(sum(r['total_fee'] for r in records)-ledger['total_fee'].sum())<1e-5
    summary={k:float(v.sum()) for k,v in ledger.items()}
    summary.update(formal_days=334,releases=len(records),emergency_kwh=float(a['e'][31:].sum()),
        initial_soc=float(a['s'][31,0]),final_soc=float(a['s'][-1,-1]),
        max_solver_gap=max(r['solver']['gap'] for r in records),unclosed_solves=sum(not r['solver']['target_met'] for r in records),
        physical_replay_errors=errors,future_masking_checks=checks,
        scheme='Q3 final rolling two-stage: 0/6/12 anticipate next release, 18 ordinary continuation; common predicted 48h prices across information branches',
        price_training='Daily LightGBM; 180-day historical issue windows whose 48h labels have fully elapsed; refresh features at 0/6/12/18',
        january='Inherited corrected Q4-2 observed-price warmup; attachment4 Jan1 prior; no attachment1',
        limitations='Future price revisions are not separately branched. Original Q3 next-release information branches and 0.1% solver target retained.')
    savejson(OUT/'计算结果与核验.json',summary)
    daily=[]
    for j,day in enumerate(range(31,365)):
        daily.append(dict(date=str(DATES[day]),q0=float(a['q0'][day].sum()),q=float(a['q'][day].sum()),emergency=float(a['e'][day].sum()),**{k:float(v[j].sum()) for k,v in ledger.items()}))
    csvout(OUT/'每日费用.csv',daily)
    savejson(OUT/'export.json',dict(dates=list(map(str,DATES[31:])),**{k:a[k][31:] for k in a},price=P[31:],daily=daily))
    print('COMPLETE',json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--end-day',type=int,default=365)
    run(parser.parse_args().end_day)
