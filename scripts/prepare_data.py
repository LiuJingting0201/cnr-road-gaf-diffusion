import argparse
import json
from cnr.data import prepare

p=argparse.ArgumentParser()
p.add_argument('--source', default='data/raw/midimost_with_class2.csv')
p.add_argument('--output', default='data/processed/most64')
p.add_argument('--length', type=int, default=64)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--max-per-vehicle', type=int, default=16)
args=p.parse_args()
print(json.dumps(prepare(args.source,args.output,args.length,args.seed,args.max_per_vehicle),indent=2))
