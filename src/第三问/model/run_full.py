from common_full import *
HOURS=(0,6,12,18)

def blank():
    warm = read_npz(FROZEN/'数据/common_startup.npz')
    a={k:np.full((365,145 if k=='s' else 144),np.nan) for k in ('q0','q','c','d','e','w','s')}
    for k in a: a[k][:31]=warm['q' if k=='q0' else k][:31]
    a.update(plans=np.full((365,4,144),np.nan),tail_plans=np.full((365,4,5,144),np.nan),
             scenario_mid_soc=np.full((365,4,5),np.nan),
             branch_q=np.full((365,3,3,144),np.nan),branch_tail_q=np.full((365,3,3,144),np.nan),
             branch_soc6=np.full((365,3,3),np.nan),branch_soc24=np.full((365,3,3),np.nan))
    return a

def checkpoint(path,a,records):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**a,records_json=np.array(json.dumps(records,ensure_ascii=False)))
    tmp.replace(path)

def run(name,end_day=365):
    raw=read_npz(FROZEN/'数据/inputs.npz')
    old=read_npz(FROZEN/'数据/scenarios_q80.npz')
    if name=='scheme1':
        from lookahead_full import solve_lookahead
        sc=read_npz(EXP/'数据/scenarios_lookahead.npz')
        assert np.array_equal(sc['dayids'],np.arange(31,365))
    manifest=runtime_manifest(name)
    mp=EXP/f'验证/runtime_{name}.json'
    if mp.exists(): assert json.loads(mp.read_text())==manifest,'Runtime changed since checkpoint'
    else: savejson(mp,manifest)
    path=EXP/f'结果/{name}.npz'
    if path.exists():
        a=read_npz(path);records=json.loads(str(a.pop('records_json')))
    else: a=blank();records=[]
    assert len(records)%4==0
    begin=time.perf_counter();last_print=begin
    for day in range(31+len(records)//4,min(end_day,365)):
        a['s'][day,0]=a['s'][day-1,-1]
        for slot,hour in enumerate(HOURS):
            off=hour*6;s0=float(a['s'][day,off]);price=np.roll(raw['p'],-off)
            before={k:a[k][day,:off].copy() for k in ('q','c','d','e','w')}
            reference=np.full(144,np.nan)
            if slot: reference[:144-off]=a['q0'][day,off:]
            original=a['q0'][day].copy() if slot else None
            started=time.perf_counter()
            if name=='scheme1' and slot<3:
                q,stats,detail=solve_lookahead(sc['n0'][day-31,slot],sc['n1'][day-31,slot],
                    price,s0,sc['weights'][day-31,slot],reference=None if slot==0 else reference)
                a['branch_q'][day,slot]=detail['branch_q']
                a['branch_tail_q'][day,slot]=detail['tail_q']
                a['branch_soc6'][day,slot]=detail['soc6']
                a['branch_soc24'][day,slot]=detail['soc24']
            else:
                q,stats,detail=plan(old['n0'][day,slot],old['n1'][day,slot],price,s0,reference)
                a['tail_plans'][day,slot]=detail['provisional_q']
                a['scenario_mid_soc'][day,slot]=detail['terminal']
            call_seconds=time.perf_counter()-started
            assert np.isfinite(q).all() and np.min(q)>=-1e-7
            if slot==0: a['q0'][day]=q.copy();original=q.copy()
            a['plans'][day,slot]=q
            actual=dispatch(q[:36],(raw['L'][day]-raw['V'][day])[off:off+36],s0)
            a['q'][day,off:off+36]=q[:36]
            for k in ('c','d','e','w'): a[k][day,off:off+36]=actual[k]
            a['s'][day,off:off+37]=actual['s']
            assert all(np.array_equal(a[k][day,:off],v) for k,v in before.items())
            assert np.array_equal(a['q0'][day],original)
            ledger=components(original[off:off+36],q[:36],actual['e'],price[:36])
            records.append(dict(treatment=name,day=day,date=date(day),hour=hour,execute_steps=36,
                lookahead_next_hour=hour+6 if name=='scheme1' and slot<3 else None,
                absolute_issue=day*144+off,horizon_end=day*144+off+288,
                original_day_plan_hash=bridge.digest(original),reference_hash=bridge.digest(reference),
                initial_soc=s0,final_soc=float(actual['s'][-1]),call_seconds=call_seconds,solver=stats,
                realized_interval_cost={k:float(v.sum()) for k,v in ledger.items()}))
        checkpoint(path,a,records)
        total=float(components(a['q0'][31:day+1],a['q'][31:day+1],a['e'][31:day+1],raw['p'])['total_fee'].sum())
        progress=dict(treatment=name,completed_days=day-30,completed_releases=len(records),last_date=date(day),
            formal_total_cost=total,call_seconds=sum(r['call_seconds'] for r in records),
            unclosed=sum(not r['solver'].get('target_met',r['solver']['gap']<=.001+1e-9) for r in records),
            max_gap=max(r['solver']['gap'] for r in records),elapsed_seconds=time.perf_counter()-begin)
        savejson(EXP/f'验证/progress_{name}.json',progress)
        if time.perf_counter()-last_print>=30 or day in (31,364):
            print(json.dumps(progress,ensure_ascii=False),flush=True);last_print=time.perf_counter()
    savejson(EXP/f'验证/run_{name}.json',records)
    assert runtime_manifest(name)==manifest
    print('COMPLETE',name,len(records),flush=True)

def runtime_manifest(name):
    paths=sorted((PACKAGE/'model').glob('*.py'))+sorted((PACKAGE/'data').glob('*.npz'))
    paths += [FROZEN/'数据'/x for x in ('inputs.npz','scenarios_q80.npz','common_startup.npz')]
    if name=='scheme1':paths.append(EXP/'数据/scenarios_lookahead.npz')
    return {str(p.resolve()):filehash(p) for p in paths}
