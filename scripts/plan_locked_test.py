"""Plan final K-pools from a strict starts/classes-only request file."""
import argparse
import json
from pathlib import Path
import numpy as np
from cnr.data import CLASSES
from cnr.final_protocol import verify_lock, plan_conditions, sha256
from cnr.route_paths import RoutePaths

p = argparse.ArgumentParser(); p.add_argument('--lock', required=True)
p.add_argument('--data', default='data/processed/final-test-e7'); a = p.parse_args()
verify_lock(a.lock)
root = Path(a.data); out = root / 'planned'; out.mkdir(exist_ok=False)
requests = dict(np.load(root / 'requests.npz'))
matcher = RoutePaths('data/maps/most.net.xml', radius=2.)
lanes = {lane['id']: i for i, lane in enumerate(matcher.road.lanes)}
prior = json.loads(Path('data/processed/long-path-e3-512/route_prior.json').read_text())
counts = {(r['class'], lanes[r['from']], lanes[r['to']]): r['count'] for r in prior}
sample = lambda start, label, rng: matcher.sample(start, CLASSES[label], counts, rng, budget=4096.)
conditions, failures = plan_conditions(requests, sample)
(out / 'planning_failures.json').write_text(json.dumps(failures, indent=2))
coverage = {'requested_conditions': len(requests['labels']), 'served_conditions': len(conditions['labels']) // 8,
            'failed_conditions': len(failures), 'failed_candidate_slots': sum(len(r['failed_candidate_slots']) for r in failures),
            'K': 8, 'path_seed': 20261109, 'request_sha256': sha256(root / 'requests.npz')}
(out / 'coverage.json').write_text(json.dumps(coverage, indent=2))
if not len(conditions['labels']): raise RuntimeError('No complete planned K-pools; inspect retained coverage failure')
np.savez_compressed(out / 'conditions.npz', **conditions)
(out / 'complete.json').write_text(json.dumps({'complete': True, 'condition_sha256': sha256(out / 'conditions.npz')}))
print(json.dumps(coverage), flush=True)
