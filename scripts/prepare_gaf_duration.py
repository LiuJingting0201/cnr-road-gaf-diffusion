"""E2: one centered window of the longest continuous segment per ID and length.

No test rows encoded; frozen E1 training bounds; retain length-dependent coverage.
"""
import hashlib
import io
import json
import lzma
import subprocess
from pathlib import Path
import numpy as np
from cnr.data import continuous_ranges
from cnr.original_gaf import trajectory_sequence, encode_frozen

root = Path('data/processed/gaf-duration-e2')
root.mkdir(parents=True, exist_ok=False)
source = Path('data/canonical/most.npz.xz')
z = np.load(io.BytesIO(lzma.decompress(source.read_bytes())), allow_pickle=False)
raw_meta = json.loads(str(z['metadata']))
xy = np.cumsum(z['delta_xy'], axis=0, dtype=np.int64)/raw_meta['coordinate_scale']
t = np.cumsum(z['delta_t'], dtype=np.int64)/raw_meta['time_scale']
offsets = np.r_[0, np.cumsum(z['counts'])]
splits = json.loads(Path('data/processed/most64/dataset.json').read_text())['split_by_id']
base = json.loads(Path('data/processed/original-hilbert64grid224/dataset.json').read_text())
lo, hi = np.array(base['xy_min']), np.array(base['xy_max'])
windows = {length: {s: [] for s in ['train', 'val']} for length in [64,224,512]}
coverage = []
for j, ident in enumerate(z['ids']):
    split = splits[str(ident)]
    if split not in ['train','val']:
        continue
    start, stop = offsets[j:j+2]
    x, times = xy[start:stop], t[start:stop]
    left, right = max(continuous_ranges(times, x), key=lambda r:r[1]-r[0])
    coverage.append({'id':str(ident),'label':int(z['labels'][j]),'split':split,'longest_contiguous':int(right-left)})
    for length in windows:
        if right-left < length:
            continue
        begin = left+(right-left-length)//2
        windows[length][split].append((str(ident), int(z['labels'][j]), x[begin:begin+length], float(times[begin])))

diagnostic = {}
for length, parts in windows.items():
    out = root/f'length{length}'
    out.mkdir()
    summaries = {}
    rejected = []
    for split, rows in parts.items():
        sequences, ids, labels, starts, raw = [], [], [], [], []
        clipping = 0
        for ident, label, x, first_time in rows:
            try:
                seq, clipped = trajectory_sequence(x,lo,hi,eps=base['hilbert_eps'])
            except ValueError as ex:
                rejected.append({'id':ident,'split':split,'reason':str(ex)})
                continue
            sequences.append(seq);ids.append(ident);labels.append(label);starts.append(first_time);raw.append(x)
            clipping += clipped
        seq = np.array(sequences)
        images, channel_clipped = encode_frozen(seq,np.array(base['channel_min']),np.array(base['channel_max']))
        labels = np.array(labels,dtype=np.int64)
        unique = np.array([len(np.unique(h)) for h in seq])
        np.savez_compressed(out/f'{split}.npz',images=images,labels=labels,ids=np.array(ids),sequences=seq,xy=np.array(raw,dtype=np.float32),first_time=np.array(starts))
        summaries[split] = {'vehicles':len(ids),'constant_fraction':float(np.mean(unique==1)),
            'median_unique_hilbert':float(np.median(unique)),'coordinate_clipped_points':clipping,'channel_clipped_values':channel_clipped,
            'per_class':{name:{'vehicles':int(np.sum(labels==c)),'constant_fraction':float(np.mean(unique[labels==c]==1)) if np.any(labels==c) else None} for c,name in enumerate(raw_meta['classes'])}}
    meta = dict(base)
    meta.update({'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'scope':'E2 real continuous-window diagnostic; frozen full-training coordinate/channel bounds; no test encoding',
        'window_points':length,'span_seconds':(length-1)*.25,'selection':'One centered window in longest continuous segment per vehicle; dt=.25, rate<=100; different eligibility by duration',
        'summary':summaries,'source_package_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    (out/'dataset.json').write_text(json.dumps(meta,indent=2))
    (out/'rejections.json').write_text(json.dumps(rejected,indent=2))
    diagnostic[str(length)] = summaries
    print(json.dumps({'length':length,'summary':summaries}),flush=True)
(root/'coverage.json').write_text(json.dumps(coverage))
(root/'diagnostic.json').write_text(json.dumps(diagnostic,indent=2))
