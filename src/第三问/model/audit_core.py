import os,sys,json,hashlib
from pathlib import Path
import numpy as np
PACKAGE=Path(__file__).resolve().parents[1]
EXP=Path(os.environ.get("Q3_OUTPUT_DIR",str(PACKAGE/"outputs"))).resolve()
ROOT=PACKAGE
FROZEN=EXP/"frozen"
LOW,HIGH,ETA,CAP=1200.,10800.,.9,5000./6.
TOL=2e-3

def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def max_error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    return float(np.max(np.abs(a-b))) if a.size else 0.

def close(a, b, label, tol=TOL):
    error = max_error(a, b)
    assert error <= tol, (label, error, tol)
    return error

def replay(q, net, s0):
    q, net = np.asarray(q, float), np.asarray(net, float)
    out = {k: np.zeros(len(q)) for k in ('c', 'd', 'e', 'w')}
    out['s'] = np.r_[float(s0), np.zeros(len(q))]
    for t, (buy, demand) in enumerate(zip(q, net)):
        delta = buy-demand
        if delta >= 0:
            out['c'][t] = min(delta, CAP, max(0., (HIGH-out['s'][t])/ETA))
            out['w'][t] = max(0., delta-out['c'][t])
        else:
            out['d'][t] = min(-delta, CAP, max(0., (out['s'][t]-LOW)*ETA))
            out['e'][t] = max(0., -delta-out['d'][t])
        out['s'][t+1] = out['s'][t]+ETA*out['c'][t]-out['d'][t]/ETA
    return out

def fees(q0, q, e, p):
    up, down = np.maximum(q-q0, 0.), np.maximum(q0-q, 0.)
    base, adjustment, emergency = p*q, .5*p*(up+down), 5*p*e
    return dict(original_plan_fee=p*q0, cancellation_refund=p*down,
                cancellation_penalty=.5*p*down, additional_purchase_fee=1.5*p*up,
                emergency_fee=emergency, settled_ordinary_fee=base+adjustment,
                total_fee=base+adjustment+emergency, increase_kwh=up, decrease_kwh=down,
                ordinary_execution_base_fee=base, adjustment_surcharge=adjustment)

def scenario_audit(raw, frozen_sc, branch_sc):
    centers = load(FROZEN/'数据/centers_with_correction.npz')['net']
    provenance = json.loads((EXP/'验证/scenario_provenance.json').read_text())
    assert len(provenance) == 1002
    assert branch_sc['dayids'].tolist() == list(range(31, 365))
    assert branch_sc['n0'].shape == branch_sc['n1'].shape == (334, 3, 3, 144)
    assert branch_sc['weights'].shape == (334, 3, 3)
    actual = (raw['L']-raw['V']).ravel()
    largest = 0.
    history_rows = []
    for i, day0 in enumerate(branch_sc['dayids']):
        day = int(day0)
        past = centers[:day]
        for slot in range(3):
            off, next_off = slot*36, (slot+1)*36
            now = day*144+off
            current = centers[day, slot]
            log = provenance[i*3+slot]
            assert (log['day'], log['current_release_hour'], log['next_release_hour']) == (day, slot*6, (slot+1)*6)
            assert log['current_center_sha256'] == digest(current)
            assert log['actual_prefix_length'] == now
            assert not log['unavailable_future_forecast_passed_to_builder']
            assert not log['unavailable_actual_passed_to_builder']
            prefix = actual[:now]
            ids = np.arange(7, day)
            revisions = np.stack([past[h, slot+1, :108]-past[h, slot, 36:144] for h in ids])
            assert log['revision_history_day_ids'] == ids.tolist()
            assert log['revision_history_sha256'] == digest(revisions)
            price = np.roll(raw['p'], -off)
            order = sorted(range(len(ids)), key=lambda j: (float(revisions[j]@price[36:]), int(ids[j])))
            groups = np.array_split(order, 3)
            reps = np.stack([revisions[g].mean(0) for g in groups])
            weights = np.array([len(g)/len(ids) for g in groups])
            for k, group in enumerate(groups):
                assert log['branches'][k]['history_day_ids'] == ids[group].tolist()
                assert log['branches'][k]['count'] == len(group)
                assert log['branches'][k]['representative_sha256'] == digest(reps[k])
                close(log['branches'][k]['weight'], weights[k], 'provenance weights', 1e-12)
            risks, risk_ids = [], []
            for label, source_slot, block in [('current_first24', slot, 0), ('current_second24', slot, 1), ('next_first24_at_current_issue', slot+1, 0)]:
                ids_r = [h for h in range(7, day)
                         if h*144+source_slot*36+(block+1)*144 <= now]
                assert ids_r
                residual = np.stack([prefix[h*144+source_slot*36+block*144:h*144+source_slot*36+(block+1)*144]
                                     - past[h, source_slot, block*144:(block+1)*144] for h in ids_r])
                saved_risk = log['risk_histories'][label]
                assert saved_risk['history_day_ids'] == ids_r
                assert saved_risk['residual_sha256'] == digest(residual)
                assert saved_risk['historical_issue_hour'] == source_slot*6 and saved_risk['block'] == block
                assert np.datetime64(saved_risk['last_target_end']) <= np.datetime64(saved_risk['cutoff'])
                risks.append(np.quantile(residual, .8, axis=0, method='linear'))
                risk_ids.append(ids_r)
            n0 = np.repeat((current[:144]+risks[0])[None, :], 3, axis=0)
            n0[:, 36:] = current[36:144]+reps+risks[2][:108]
            n1 = np.repeat((current[144:]+risks[1])[None, :], 3, axis=0)
            assert log['n0_sha256'] == digest(branch_sc['n0'][i, slot])
            assert log['n1_sha256'] == digest(branch_sc['n1'][i, slot])
            errors = [close(branch_sc['n0'][i, slot], n0, 'causal branch main', 1e-8),
                      close(branch_sc['n1'][i, slot], n1, 'causal branch tail', 1e-8),
                      close(branch_sc['weights'][i, slot], weights, 'empirical weights', 1e-12),
                      close(frozen_sc['n0'][day, slot], np.tile(current[:144]+risks[0], (5, 1)), 'frozen main Q80', 1e-8),
                      close(frozen_sc['n1'][day, slot], np.tile(current[144:]+risks[1], (5, 1)), 'frozen tail Q80', 1e-8)]
            if 'representative_revision' in branch_sc:
                errors.append(close(branch_sc['representative_revision'][i, slot], reps, 'full-path revision means', 1e-8))
            largest = max(largest, *errors)
            assert max(ids)*144+next_off < now
            assert max(risk_ids[0]) == day-1
            assert max(risk_ids[1]) == max(risk_ids[2]) == day-2
            history_rows.append(dict(day=day, hour=slot*6, latest_revision_day=int(max(ids)),
                                     latest_current_main_residual_day=max(risk_ids[0]),
                                     latest_current_tail_residual_day=max(risk_ids[1]),
                                     latest_next_release_residual_day=max(risk_ids[2]),
                                     revision_sha256=digest(revisions)))
    return dict(status='PASS', rebuilt_releases=len(history_rows), max_abs_error=largest,
                constructor_today_future_forecast_required=False,
                all_current_main_residuals_latest_day='day-1',
                all_next_release_and_tail_residuals_latest_day='day-2',
                branch_tail_horizon='current issue+24h through current issue+48h',
                limitations=['All branches share deterministic current-Q80 first 6h and hence modeled next-release SOC.',
                             'Revision-group means plus unconditional next-release Q80 are not calibrated conditional distributions.'],
                reconstructed_history_records=history_rows)

def weighted_median(values, weights):
    out = np.empty(values.shape[1])
    for t in range(values.shape[1]):
        order = sorted(range(len(weights)), key=lambda k: (values[k, t], k))
        acc = 0.
        for k in order:
            acc += weights[k]
            if acc >= .5:
                out[t] = values[k, t]
                break
    return out

def objective_audit(name, a, record, sc, branch_sc, daily_price):
    day, slot = int(record['day']), int(record['hour'])//6
    off, price = slot*36, np.roll(daily_price, -slot*36)
    s0, q, solver = float(a['s'][day, off]), a['plans'][day, slot], record['solver']
    shared_error = median_error = 0.
    if name == 'scheme1' and slot < 3:
        weights = branch_sc['weights'][day-31, slot]
        qs = a['branch_q'][day, slot]
        nets, tails = branch_sc['n0'][day-31, slot], branch_sc['n1'][day-31, slot]
        shared_error = close(qs[:, :36], np.tile(q[:36], (3, 1)), 'nonanticipating common next-6h procurement')
        tr = [replay(qs[k], nets[k], s0) for k in range(3)]
        close([r['s'][36] for r in tr], a['branch_soc6'][day, slot], 'branch next-release SOC')
        close([r['s'][-1] for r in tr], a['branch_soc24'][day, slot], 'branch main-terminal SOC')
        close(a['branch_soc6'][day, slot], np.full(3, a['branch_soc6'][day, slot, 0]), 'shared modeled next-release SOC')
        if slot == 0:
            adjustment = .5*np.abs(qs[:, 36:]-a['q0'][day, None, 36:])@price[36:]
            assert solver['contract_mode'] == 'variable_midnight_original'
            assert solver['returned_postfix_role'] == 'original_contract'
            median_error = close(q[36:], weighted_median(qs[:, 36:], weights), 'original contract weighted-median property')
        else:
            count = 144-off
            adjustment = .5*np.abs(qs[:, :count]-a['q0'][day, None, off:])@price[:count]
            assert solver['contract_mode'] == 'fixed_original_reference'
            assert solver['returned_postfix_role'] == 'weighted_median_diagnostic'
            assert solver['committed_reference_steps'] == count
            median_error = close(q[36:], weighted_median(qs[:, 36:], weights), 'returned diagnostic median')
        first = float(weights@np.array([price@qs[k]+adjustment[k]+5*price@tr[k]['e'] for k in range(3)]))
        tq = a['branch_tail_q'][day, slot]
        tail = float(weights@np.array([price@tq[k]+5*price@replay(tq[k], tails[k], tr[k]['s'][-1])['e'] for k in range(3)]))
        assert solver['information_branches'] == 3 and solver['residual_branches_per_information'] == 1
    else:
        nets, tails = sc['n0'][day, slot], sc['n1'][day, slot]
        tr = [replay(q, n, s0) for n in nets]
        terminal = np.array([r['s'][-1] for r in tr])
        close(terminal, a['scenario_mid_soc'][day, slot], 'frozen-path main-terminal SOC')
        adjustment = .5*price[:144-off]@np.abs(q[:144-off]-a['q0'][day, off:]) if slot else 0.
        first = float(price@q+adjustment+np.mean([5*price@r['e'] for r in tr]))
        tq = a['tail_plans'][day, slot]
        tail = float(np.mean([price@tq[k]+np.mean([5*price@replay(tq[k], n, terminal[k])['e'] for n in tails]) for k in range(5)]))
    objective = first+tail
    error = max(close(first, solver['first24_objective'], 'first24 objective', .02),
                close(tail, solver['tail24_objective'], 'tail24 objective', .02),
                close(objective, solver['incumbent'], 'full48h feasible objective', .02))
    bound = solver['lower_bound']
    assert bound <= objective+1e-3
    gap = max(0., (objective-bound)/max(abs(objective), 1e-12))
    close(gap, solver['gap'], 'gap arithmetic', 1e-8)
    assert solver['target_met'] == (solver['gap'] <= .001+1e-9)
    assert solver['target_gap'] == .001 and solver['root_time_limit_seconds'] == 2.
    assert solver['max_continuation_iterations'] == 3 and solver['horizon_steps'] == 288
    return error, shared_error, median_error

def results_audit(name, raw, frozen, frozen_sc, branch_sc, allow_partial=False):
    a = load(EXP/f'结果/{name}.npz')
    records = json.loads(str(a['records_json']))
    assert len(records)%4 == 0
    done = len(records)//4
    assert (allow_partial and 0 < done <= 334) or done == 334
    stop = 31+done
    warm = load(FROZEN/'数据/common_startup.npz')
    for key in ('q0', 'q', 'c', 'd', 'e', 'w', 's'):
        close(a[key][:31], warm['q' if key == 'q0' else key][:31], 'identical common January', 1e-8)
    close(a['s'][0, 0], 6000., 'common January initial SOC', 1e-8)
    close(a['s'][1:stop, 0], a['s'][:stop-1, -1], 'continuous own SOC throughout year', 1e-8)
    replay_max = balance_max = ledger_max = objective_max = prefix_max = median_max = 0.
    for day in range(stop):
        net = raw['L'][day]-raw['V'][day]
        tr = replay(a['q'][day], net, a['s'][day, 0])
        for key in ('c', 'd', 'e', 'w', 's'):
            replay_max = max(replay_max, close(a[key][day], tr[key], f'{name}/{day}/{key} independent B replay'))
        assert a['q'][day].min() >= -1e-7 and a['q0'][day].min() >= -1e-7
        assert a['s'][day].min() >= LOW-1e-6 and a['s'][day].max() <= HIGH+1e-6
        assert a['c'][day].max() <= CAP+1e-6 and a['d'][day].max() <= CAP+1e-6
        balance_max = max(balance_max, close(a['q'][day]+a['e'][day]+a['d'][day]-a['c'][day]-a['w'][day], net, 'AC energy conservation'))
        if day < 31:
            continue
        close(a['q0'][day], a['plans'][day, 0], 'immutable midnight original contract', 1e-8)
        for slot in range(4):
            off = slot*36
            r = records[(day-31)*4+slot]
            assert (r['day'], r['hour'], r['execute_steps']) == (day, slot*6, 36)
            assert r['absolute_issue'] == day*144+off and r['horizon_end'] == day*144+off+288
            assert r['lookahead_next_hour'] == (slot*6+6 if name == 'scheme1' and slot < 3 else None)
            assert r['original_day_plan_hash'] == digest(a['q0'][day])
            reference = np.full(144, np.nan)
            if slot:
                reference[:144-off] = a['q0'][day, off:]
            assert r['reference_hash'] == digest(reference)
            close(a['q'][day, off:off+36], a['plans'][day, slot, :36], 'only common published next6h executed', 1e-8)
            close(r['initial_soc'], a['s'][day, off], 'record initial SOC', 1e-8)
            close(r['final_soc'], a['s'][day, off+36], 'record final SOC', 1e-8)
            ledger = fees(a['q0'][day, off:off+36], a['q'][day, off:off+36], a['e'][day, off:off+36], raw['p'][off:off+36])
            for key, value in r['realized_interval_cost'].items():
                ledger_max = max(ledger_max, close(ledger[key].sum(), value, 'once-only realized ledger', 1e-5))
            errors = objective_audit(name, a, r, frozen_sc, branch_sc, raw['p'])
            objective_max, prefix_max, median_max = max(objective_max, errors[0]), max(prefix_max, errors[1]), max(median_max, errors[2])
    manifest = json.loads((EXP/f'验证/runtime_{name}.json').read_text())
    changed = [p for p, h in manifest.items() if sha(ROOT/p) != h]
    assert not changed, ('Runtime manifest changed', changed)
    totals = {k: float(v.sum()) for k, v in fees(a['q0'][31:stop], a['q'][31:stop], a['e'][31:stop], raw['p']).items()}
    annual = {k: float(v.sum()) for k, v in fees(a['q0'][:stop], a['q'][:stop], a['e'][:stop], raw['p']).items()}
    result = dict(status='PASS', formal_completed_days=done, calendar_completed_days=stop,
                  audited_releases=len(records), replay_max_abs_error=replay_max,
                  balance_max_abs_error=balance_max, saved_ledger_max_abs_error=ledger_max,
                  full48h_objective_max_abs_error=objective_max,
                  shared_branch_prefix_max_abs_error=prefix_max, diagnostic_median_max_abs_error=median_max,
                  formal_totals=totals, annual_totals=annual, runtime_manifest_files=len(manifest),
                  max_reported_gap=max(r['solver']['gap'] for r in records),
                  unclosed_targets=sum(not r['solver']['target_met'] for r in records),
                  lower_bound_scope='Verified lower<=independent feasible incumbent and gap arithmetic; LP dual certificates were not regenerated.')
    if name == 'baseline':
        result['frozen_reproduction_max_errors'] = {k: close(a[k][:stop], frozen[k][:stop], f'frozen baseline/{k}') for k in ('q0', 'q', 'c', 'd', 'e', 'w', 's')}
        frozen_total = fees(frozen['q0'][31:stop], frozen['q'][31:stop], frozen['e'][31:stop], raw['p'])['total_fee'].sum()
        result['frozen_total_fee_difference'] = totals['total_fee']-float(frozen_total)
    return result
