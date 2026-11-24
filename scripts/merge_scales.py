import argparse
from rrq.scales import load,save
from rrq.config import MODELS, model_config
p=argparse.ArgumentParser();p.add_argument('parts',nargs='+');p.add_argument('--output',required=True);p.add_argument('--model',choices=tuple(MODELS),required=True);a=p.parse_args()
entries=[v for path in a.parts for v in load(path)]
depth=model_config(a.model)['layers']
if len(entries)!=3*depth or len({v[0] for v in entries})!=3*depth:raise ValueError(f'Expected {depth} layers x 3 distinct calibration sites')
if {int(v[0].split('.layers.')[1].split('.')[0]) for v in entries} != set(range(depth)):raise ValueError('Incomplete layer coverage')
from pathlib import Path
if Path(a.output).exists() or Path(a.output).with_suffix('.json').exists():raise FileExistsError(a.output)
entries.sort(key=lambda v:(int(v[0].split('.layers.')[1].split('.')[0]),v[0]))
save(a.output,entries)
