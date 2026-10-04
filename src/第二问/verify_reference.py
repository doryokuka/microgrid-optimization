"""Replay the reference trajectory and all 334 full planning objectives; no solve."""
from pathlib import Path
import argparse
import json
import sys
sys.dont_write_bytecode = True
import numpy as np
from q2.common import ROOT, DATA, DATES, digest, inputs, savejson
from q2.final_pipeline import make, TARGET
from q2.settlement import dispatch
from release_common import audit, replay_48


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'reference/q80.npz')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/reference_audit.json')
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        z = {key: archive[key].copy() for key in archive.files}
    with np.load(DATA/'issued_forecasts.npz', allow_pickle=False) as archive:
        issued = dict(archive)
    a = {key: z[key] for key in ('q', 'c', 'd', 'e', 'w', 's')}
    logs = json.loads(str(z['records_json']))
    assert [r['day'] for r in logs] == list(range(31, 365))
    L, V, price = inputs()
    physical = audit(a, L, V, price)
    objective_error = terminal_error = 0.0
    for day, record in zip(range(31, 365), logs):
        n0, n1, _ = make(L, V, price, day, 'q80', issued)
        q = a['q'][day]
        assert record['q_hash'] == digest(q)
        assert record['scenario_hash'] == digest(n0)
        assert record['tomorrow_scenario_hash'] == digest(n1)
        assert record['cutoff'] == str(DATES[day-1])
        if day < 364:
            value = replay_48(n0, n1, price, a['s'][day, 0], q,
                              dict(provisional_q=z['provisional_q'][day]), 'mean')
            terminal_error = max(terminal_error, max(abs(dispatch(q, n, a['s'][day, 0])['s'][-1]
                                    - z['scenario_terminals'][day, k]) for k, n in enumerate(n0)))
        else:
            value = float(price @ q + np.mean([5*price @ dispatch(q, n, a['s'][day, 0])['e'] for n in n0]))
        stats = record['solver']
        objective_error = max(objective_error, abs(value-stats['incumbent']))
        assert stats['gap'] <= TARGET+1e-8 and stats['lower_bound'] <= value+1e-5
        replay_gap = max(0.0, (value-stats['lower_bound'])/max(abs(value), 1e-12))
        assert abs(replay_gap-stats['gap']) < 1e-9
    assert objective_error < 1e-5 and terminal_error < 1e-7
    normal = a['q'][31:] @ price
    emergency = 5*a['e'][31:] @ price
    report = dict(status='PASS', simulated_days=365, formal_days=334,
                  physical=physical, planning_objective_max_error_yuan=objective_error,
                  planning_terminal_soc_max_error_kwh=terminal_error,
                  all_formal_gaps_met=True, normal_cost_yuan=float(normal.sum()),
                  emergency_cost_yuan=float(emergency.sum()),
                  total_cost_yuan=float((normal+emergency).sum()),
                  emergency_kwh=float(a['e'][31:].sum()))
    savejson(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
