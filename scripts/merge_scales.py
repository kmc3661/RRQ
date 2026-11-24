import argparse
from rrq.scales import load,save
p=argparse.ArgumentParser();p.add_argument('parts',nargs='+');p.add_argument('--output',required=True);a=p.parse_args()
entries=[v for path in a.parts for v in load(path)]
if len(entries)!=108 or len({v[0] for v in entries})!=108:raise ValueError('Expected 36 layers x 3 distinct calibration sites')
entries.sort(key=lambda v:(int(v[0].split('.layers.')[1].split('.')[0]),v[0]))
save(a.output,entries)
