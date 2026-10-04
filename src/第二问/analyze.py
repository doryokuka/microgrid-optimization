"""Portable independent full 48h audit and result tables."""
import sys
sys.dont_write_bytecode=True
from q2.common import *
from q2.export import export_outputs
import argparse

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'结果')
    args=parser.parse_args();out=args.output.resolve()
    config=json.loads((out/'run_config.json').read_text())
    summary,audits=export_outputs(out,days=config['simulated_days'])
    if out==ROOT/'结果':savejson(ROOT/'验证/全年独立核验.json',audits)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
