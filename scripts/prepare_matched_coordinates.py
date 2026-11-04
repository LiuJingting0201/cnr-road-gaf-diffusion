"""Same accepted windows/order as route512; raw XY target; no new split or test use."""
import hashlib,json
from pathlib import Path
import numpy as np
source=Path('data/processed/most64');route=Path('data/processed/route512');out=Path('data/processed/most64-route-cohort');out.mkdir(exist_ok=False)
meta=json.loads((source/'dataset.json').read_text());record={}
for split in ['train','val']:
 raw=np.load(source/f'{split}.npz');r=np.load(route/f'{split}.npz');ix=r['source_index'];xy=raw['xy'][ix];labels=raw['labels'][ix]
 assert np.array_equal(xy,r['xy']) and np.array_equal(labels,r['labels'])
 np.savez_compressed(out/f'{split}.npz',xy=xy,labels=labels,source_index=ix)
 record[split]={'windows':len(xy),'source_indices_sha256':hashlib.sha256(ix.tobytes()).hexdigest()}
# Retain the G5 start normalization (from parent training split) to match conditions.
meta.update({'cohort':record,'scope':'same accepted training/validation windows as path-g5, raw coordinate targets; parent training normalization retained for matched starts','parent':str(source)})
(out/'dataset.json').write_text(json.dumps(meta,indent=2));print(json.dumps(record))
