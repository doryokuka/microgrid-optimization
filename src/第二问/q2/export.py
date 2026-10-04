"""Read completed independent runs, audit every objective, and write requested metrics."""
from release_common import *

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])

def export_outputs(out,methods=None,days=365):
    out=Path(out)
    methods=[m for m in METHODS if (out/f'{m}.npz').exists()] if methods is None else list(methods)
    if not methods:raise ValueError('No completed model output found')
    issued=dict(np.load(out/'issued_forecasts.npz'))
    L,V,p=inputs();startup=np.load(out/'common_startup.npz');summary=[];daily=[];quarters=[];audits=[];arrays={};costs={};logs_by={}
    for method in methods:
        z=np.load(out/f'{method}.npz');logs=json.loads(str(z['records_json']))
        assert len(logs)==days-31 and [r['day'] for r in logs]==list(range(31,days))
        a={k:z[k] for k in ['q','c','d','e','w','s']};arrays[method]=a;logs_by[method]=logs
        for k in a:assert np.array_equal(a[k][:31],startup[k][:31])
        physical=audit({k:v[:days] for k,v in a.items()},L,V,p)
        paths=[make(L,V,p,d,method,issued) for d in range(31,days)]
        sc={key:np.array([np.zeros_like(paths[0][i])]*31+[z[i] for z in paths]) for i,key in enumerate(['n0','n1'])}
        objerr=0.;enderr=0.;max_balance=0.;max_soc=0.
        for d,log in zip(range(31,days),logs):
            assert log['q_hash']==digest(a['q'][d]) and log['scenario_hash']==digest(sc['n0'][d])
            assert log['tomorrow_scenario_hash']==digest(sc['n1'][d]) and log['cutoff']==str(DATES[d-1])
            s0=a['s'][d,0];n0=sc['n0'][d];n1=sc['n1'][d];q=a['q'][d]
            assert abs(s0-a['s'][d-1,-1])<1e-8
            if d<364:
                de=dict(provisional_q=z['provisional_q'][d])
                objective=replay_48(n0,n1,p,s0,q,de,'worst' if method=='robust5' else 'mean')
                enderr=max(enderr,max(abs(dispatch(q,n,s0)['s'][-1]-z['scenario_terminals'][d,k]) for k,n in enumerate(n0)))
            else:
                em=[5*p@dispatch(q,n,s0)['e'] for n in n0]
                objective=float(p@q+(max(em) if method=='robust5' else np.mean(em)))
            st=log['solver'];objerr=max(objerr,abs(objective-st['incumbent']))
            assert st['gap']<=TARGET+1e-8 and st['lower_bound']<=objective+1e-5
            recgap=max(0.,(objective-st['lower_bound'])/max(abs(objective),1e-12))
            assert abs(recgap-st['gap'])<1e-9
            max_balance=max(max_balance,float(np.max(abs(q+a['d'][d]+a['e'][d]-(L[d]-V[d])-a['c'][d]-a['w'][d]))))
            max_soc=max(max_soc,float(np.max(abs(np.diff(a['s'][d])-ETA*a['c'][d]+a['d'][d]/ETA))))
        assert objerr<1e-5 and enderr<1e-7 and max_balance<1e-7 and max_soc<1e-7
        norm=a['q']@p;emer=5*a['e']@p;total=norm+emer;costs[method]=total[31:days]
        gaps=np.array([x['solver']['gap'] for x in logs]);worst=31+int(np.argmax(total[31:days]))
        r=dict(method=method,formal_days=days-31,normal_cost=float(norm[31:days].sum()),emergency_cost=float(emer[31:days].sum()),
            total_cost=float(total[31:days].sum()),emergency_kwh=float(a['e'][31:days].sum()),spill_kwh=float(a['w'][31:days].sum()),
            mean_soc_kwh=float(a['s'][31:days,1:].mean()),mean_day_end_soc_kwh=float(a['s'][31:days,-1].mean()),end_soc_kwh=float(a['s'][days-1,-1]),
            initial_formal_soc_kwh=float(a['s'][31,0]),january_cost=float(total[:31].sum()),
            max_daily_cost=float(total[worst]),max_daily_date=str(DATES[worst]),
            gap_mean=float(gaps.mean()),gap_max=float(gaps.max()),gap_p95=float(np.quantile(gaps,.95)),gap_target_met_days=int(np.sum(gaps<=TARGET+1e-8)),
            planning_seconds=float(sum(x['solver']['total_seconds'] for x in logs)),
            low_soc_slot_share=float(np.mean(a['s'][31:days,1:]<=LOW+1e-6)),high_soc_slot_share=float(np.mean(a['s'][31:days,1:]>=HIGH-1e-6)))
        r["all_simulated_days_cost"]=r["total_cost"]+r["january_cost"]
        summary.append(r)
        for d,log in zip(range(31,days),logs):
            month=int(str(DATES[d])[5:7]);daily.append(dict(method=method,date=str(DATES[d]),quarter=(month-1)//3+1,
                normal_cost=float(norm[d]),emergency_cost=float(emer[d]),total_cost=float(total[d]),emergency_kwh=float(a['e'][d].sum()),
                spill_kwh=float(a['w'][d].sum()),start_soc_kwh=float(a['s'][d,0]),mean_soc_kwh=float(a['s'][d,1:].mean()),end_soc_kwh=float(a['s'][d,-1]),gap=log['solver']['gap']))
        for quarter in range(1,5):
            ids=[d for d in range(31,days) if (int(str(DATES[d])[5:7])-1)//3+1==quarter]
            quarters.append(dict(method=method,quarter=f'Q{quarter}',days=len(ids),normal_cost=float(norm[ids].sum()),
                emergency_cost=float(emer[ids].sum()),total_cost=float(total[ids].sum()),emergency_kwh=float(a['e'][ids].sum()),spill_kwh=float(a['w'][ids].sum())))
        audits.append(dict(method=method,**physical,full48_objective_replay_error=objerr,scenario_terminal_error=enderr,
                           energy_balance_error=max_balance,soc_balance_error=max_soc,formal_gap_target_met=True,shared_january_exact=True))
    pairwise=[];pairdaily=[]
    for x in methods:
        for y in methods:
            delta=costs[x]-costs[y];wins=int(np.sum(delta<-.01));ties=int(np.sum(abs(delta)<=.01));loss=int(np.sum(delta>.01))
            pairwise.append(dict(method=x,baseline=y,win_days=wins,tie_days=ties,loss_days=loss,win_rate=wins/(days-31),
                total_saving=float(-delta.sum()),relative_saving=float(-delta.sum()/costs[y].sum()),max_daily_excess=float(delta.max()),max_daily_saving=float(-delta.min())))
            if x!=y:
                for i,d in enumerate(range(31,days)):pairdaily.append(dict(method=x,baseline=y,date=str(DATES[d]),cost_difference=float(delta[i])))
    for r in summary:
        for base in ['point','pca5']:
            v=next((z for z in pairwise if z['method']==r['method'] and z['baseline']==base),None)
            r[f'win_rate_vs_{base}']=v['win_rate'] if v else None;r[f'saving_vs_{base}']=v['total_saving'] if v else None
    # Fixed clock blocks describe procurement/stock coupling, with no extra optimization methods.
    blocks=[]
    for method,a in arrays.items():
        for name,l,r in [('00-08',0,48),('08-16',48,96),('16-24',96,144)]:
            blocks.append(dict(method=method,clock_block=name,normal_cost=float(np.sum(a['q'][31:days,l:r]*p[l:r])),
                emergency_cost=float(5*np.sum(a['e'][31:days,l:r]*p[l:r])),normal_kwh=float(a['q'][31:days,l:r].sum()),
                emergency_kwh=float(a['e'][31:days,l:r].sum()),spill_kwh=float(a['w'][31:days,l:r].sum()),
                mean_start_soc_kwh=float(a['s'][31:days,l].mean()),mean_end_soc_kwh=float(a['s'][31:days,r].mean())))
    for name,data in [('五组汇总',summary),('逐日明细',daily),('季度费用',quarters),('日胜率与配对比较',pairwise),('逐日配对费用差',pairdaily),('分时费用与SOC',blocks)]:
        csvout(out/f'{name if len(methods)==5 or name!="五组汇总" else "模型汇总"}.csv',data)
    savejson(out/'summary.json',summary);savejson(out/'audit.json',audits)
    for row in summary:
        row['all_simulated_days_cost']=row['total_cost']+row['january_cost']
        savejson(out/f"{row['method']}_summary.json",row)
    savejson(out/'summary.json',summary)
    print('VERIFIED',len(methods),'methods x',days-31,'formal days',flush=True)
    return summary,audits
