import json
from pathlib import Path
import sys
p=Path(sys.argv[1])
lines=['# SUMO batch-size comparison','','Fixed training and sampling protocol; changed batch size and resulting update count. Scores are from the same 177 reserved vehicles.', '',
'| Seed | Batch | Density JS | OD JS | Length JS | Hotspot F1 | Generated median km | Length W1 km | Outside |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
hashes=set()
for seed in (42,43):
 for batch,folder in [(1024,p.parent/'sumo-original'),(32,p)]:
    d=json.loads((folder/str(seed)/'sumo-final.json').read_text())
    epochs=[json.loads(line) for line in (folder/str(seed)/'epochs.jsonl').read_text().splitlines()]
    assert len(epochs)==200 and all(e['global_batch']==batch and e['train_seed']==seed for e in epochs)
    assert d['epoch']==200 and d['requests']==177
    hashes.add(d['request_index_sha256'])
    lines.append(f"| {seed} | {batch} | {d['density_js']:.4f} | {d['trip_js']:.4f} | {d['length_js']:.4f} | {d['top20_cell_f1']:.2f} | {d['generated_median_m']/1000:.2f} | {d['length_w1_m']/1000:.2f} | {100*d['generated_outside_fraction']:.2f}% |")
assert len(hashes)==1
lines+=['',f"Observed median {d['reference_median_m']/1000:.2f} km. Real development versus real reserve: density, OD, length JS and hotspot F1 "+', '.join(f'{v:.4f}' for v in d['real_real'])+'.']
(p/'REPORT.md').write_text('\n'.join(lines)+'\n')
