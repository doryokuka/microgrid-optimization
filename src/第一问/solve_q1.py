"""Solve problem C, question 1 with a continuous linear program.

Requirements: numpy, scipy, openpyxl. Run from any working directory.
The result XLSX is exported separately with the supplied template.
Validate charge/discharge exclusivity after solving this dataset's LP optimum.
"""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix
import openpyxl
import argparse

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'outputs',
                    help='Directory for calculation.json and the ten-minute CSV')
OUT = parser.parse_args().output.resolve()
OUT.mkdir(parents=True, exist_ok=True)
N, ETA, STEP, CAP, E0 = 144, 0.9, 1/6, 5000/6, 6000.0

def minute(value):
    if isinstance(value, dt.time):
        return value.hour*60+value.minute
    value = str(value)
    h, m = value.replace('+1', '').split(':')[:2]
    return int(h)*60+int(m)+(1440 if '+1' in value else 0)

def label(m):
    return f'{m//60}:{m%60:02d}'

source_path = ROOT/'附件/附件1.xlsx'
source = list(openpyxl.load_workbook(source_path, data_only=True).active.values)[1:]
assert len(source) == N
assert [minute(r[0]) for r in source] == list(range(10, 1441, 10))
# Source and result template use interval START times. Identify 24:00 with 0:00
# under the daily cyclic convention, solve in midnight-to-midnight order,
# then map the results back to the original template order.
data = [source[-1]] + source[:-1]
price, load, pv = np.array([r[1:] for r in data], dtype=float).T
assert np.all(np.isfinite([price, load, pv])) and np.all(price > 0)
assert np.all(load >= 0) and np.all(pv >= 0)
load_e, pv_e = load*STEP, pv*STEP
# x = [grid q, AC-side charge c, AC-side discharge d, curtailed PV w, end SOC e].
NV = 5*N
objective = np.r_[price, np.zeros(4*N)]
Aeq = lil_matrix((2*N, NV))
beq = np.zeros(2*N)
for k in range(N):
    Aeq[k,k], Aeq[k,N+k], Aeq[k,2*N+k], Aeq[k,3*N+k] = 1,-1,1,-1
    beq[k] = load_e[k]-pv_e[k]
    Aeq[N+k,4*N+k], Aeq[N+k,N+k], Aeq[N+k,2*N+k] = 1,-ETA,1/ETA
    if k:
        Aeq[N+k,4*N+k-1] = -1
    else:
        beq[N+k] = E0
Aeq = csr_matrix(Aeq)
lower = np.r_[np.zeros(4*N),np.full(N,1200.0)]
upper = np.r_[np.full(N,np.inf),np.full(2*N,CAP),pv_e,np.full(N,10800.0)]
lower[-1] = upper[-1] = E0
lp = linprog(objective,A_eq=Aeq,b_eq=beq,bounds=list(zip(lower,upper)),method='highs')
assert lp.success,lp.message
# All 720 variables are continuous. No complementarity or integer constraints
# enter the LP. Its solution is accepted as a physical schedule only after the
# explicit charge/discharge exclusivity check below.
x = lp.x.copy()
x[np.abs(x)<1e-8] = 0
q,c,d,w,e = x.reshape(5,N)
soc = np.r_[E0,e]
cost = float(price@q)
balance = q+pv_e+d-load_e-c-w
soc_error = np.diff(soc)-ETA*c+d/ETA
# A dual certificate verifies LP optimality without a second optimization.
y, alpha, beta = lp.eqlin.marginals, lp.lower.marginals, lp.upper.marginals
finite_upper = np.isfinite(upper)
dual_cost = float(beq@y + lower@alpha + upper[finite_upper]@beta[finite_upper])
dual_residual = objective-Aeq.T@y-alpha-beta
checks = {
    'interval_count': N,
    'model_type': 'LP',
    'continuous_variable_count': NV,
    'equality_constraint_count': Aeq.shape[0],
    'max_energy_balance_error_kwh':float(np.max(np.abs(balance))),
    'max_soc_transition_error_kwh':float(np.max(np.abs(soc_error))),
    'minimum_soc_kwh':float(soc.min()),
    'maximum_soc_kwh':float(soc.max()),
    'initial_soc_kwh':float(soc[0]),
    'final_soc_kwh':float(soc[-1]),
    'max_charge_power_kw':float(c.max()/STEP),
    'max_discharge_power_kw':float(d.max()/STEP),
    'simultaneous_charge_discharge_intervals':int(np.sum((c>1e-7)&(d>1e-7))),
    'lp_optimum_yuan':float(lp.fun),
    'lp_dual_objective_yuan':dual_cost,
    'lp_primal_dual_gap_yuan':float(abs(cost-dual_cost)),
    'lp_max_dual_stationarity_error':float(np.max(np.abs(dual_residual))),
}
assert checks['max_energy_balance_error_kwh'] < 1e-6
assert checks['max_soc_transition_error_kwh'] < 1e-6
if checks['simultaneous_charge_discharge_intervals']:
    intervals = [f'{label(k*10)}-{label((k+1)*10)}' for k in range(N)
                 if c[k] > 1e-7 and d[k] > 1e-7]
    raise RuntimeError('The LP solution simultaneously charges and discharges at '
                       + ', '.join(intervals)
                       + '; no physical schedule has been exported.')
assert checks['lp_primal_dual_gap_yuan'] < 1e-6
assert checks['lp_max_dual_stationarity_error'] < 1e-7
assert alpha.min() >= -1e-7 and beta.max() <= 1e-7
assert np.max(np.abs(beta[~finite_upper])) < 1e-7
assert soc.min() >= 1200-1e-6 and soc.max() <= 10800+1e-6
assert abs(soc[-1]-E0)<1e-6
assert max(c.max(),d.max()) <= CAP+1e-6
assert q.min() >= -1e-6 and w.min() >= -1e-6
assert np.max(w-pv_e)<1e-6
baseline_q = np.maximum(load_e-pv_e,0)
blocks = [{'interval':f'{h}:00-{h+4}:00','charge_kwh':float(c[h*6:(h+4)*6].sum()),
           'discharge_kwh':float(d[h*6:(h+4)*6].sum())} for h in range(0,24,4)]
table1 = [{'interval':f'{h}:00-{h}:10','grid_kwh':float(q[h*6])} for h in [10,12,14,16,18,20]]
template = openpyxl.load_workbook(ROOT/'附件/附件5/result1.xlsx',data_only=True)
template_labels = [r[0] for r in list(template['计划购电量'].values)[1:]]
assert [minute(s.split('-')[0]) for s in template_labels] == list(range(10,1441,10))
template_q = np.r_[q[1:],q[:1]]
totals = {'grid_kwh':float(q.sum()),'cost_yuan':cost,'load_kwh':float(load_e.sum()),
    'pv_kwh':float(pv_e.sum()),'charge_kwh':float(c.sum()),'discharge_kwh':float(d.sum()),
    'curtailment_kwh':float(w.sum()),'losses_kwh':float(c.sum()-d.sum()),
    'baseline_grid_kwh':float(baseline_q.sum()),'baseline_cost_yuan':float(price@baseline_q),
    'savings_yuan':float(price@baseline_q-cost),'savings_fraction':float(1-cost/(price@baseline_q))}
details = []
for k in range(N):
    details.append({'start':label(k*10),'end':label((k+1)*10),'source_excel_row':145 if k==0 else k+1,
        'price_yuan_per_kwh':float(price[k]),'load_kw':float(load[k]),'pv_kw':float(pv[k]),
        'grid_kwh':float(q[k]),'charge_kwh':float(c[k]),'discharge_kwh':float(d[k]),
        'curtailment_kwh':float(w[k]),'soc_start_kwh':float(soc[k]),'soc_end_kwh':float(soc[k+1]),
        'cost_yuan':float(price[k]*q[k])})
result={'assumptions':{'interval':'source timestamp is interval start; midnight is cyclic',
    'charging_efficiency':ETA,'discharging_efficiency':ETA,'initial_soc_kwh':E0,
    'energy_metering':'AC bus side for both charging and discharging'},
    'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
    'totals':totals,'table1':table1,'blocks':blocks,'checks':checks,
    'template_labels':template_labels,'template_q':template_q.tolist(),'details':details}
(OUT/'calculation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'10分钟调度明细.csv').open('w',encoding='utf-8-sig',newline='') as file:
    writer=csv.DictWriter(file,fieldnames=details[0].keys());writer.writeheader();writer.writerows(details)
print(json.dumps({k:result[k] for k in ['totals','table1','blocks','checks']},ensure_ascii=False,indent=2))
