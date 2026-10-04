"""Export completed Q80 trajectories as JSON for the shared XLSX writer."""
from pathlib import Path
import argparse
import json
import sys
sys.dont_write_bytecode = True
import numpy as np
from q2.common import ROOT, DATES, inputs, savejson


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'reference/q80.npz')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/submission_data.json')
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        records = json.loads(str(archive['records_json']))
        if [r['day'] for r in records] != list(range(31, 365)):
            raise ValueError('Only a completed 365-day run with 334 formal days can be submitted')
        a = {key: archive[key].copy() for key in ('q', 'c', 'd', 'e', 'w', 's')}
    for key, values in a.items():
        if values.shape != (365, 145 if key == 's' else 144) or not np.isfinite(values).all():
            raise ValueError(f'Invalid trajectory: {key}')
    _, _, price = inputs()
    normal = a['q'][31:] @ price
    emergency = 5 * a['e'][31:] @ price
    output = dict(model='Q80', formal_days=334, dates=list(map(str, DATES[31:])),
                  **{key: values[31:] for key, values in a.items()},
                  daily_plan_cost=normal, daily_emergency_cost=emergency,
                  daily_total_cost=normal+emergency,
                  time_mapping='Preserve source-array ordinal order; the shared writer retains the official template headers.',
                  units='kWh for energy, yuan for costs')
    savejson(args.output, output)
    print(f'Exported 334 days to {args.output}')


if __name__ == '__main__':
    main()
