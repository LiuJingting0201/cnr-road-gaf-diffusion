import json
import sys
from pathlib import Path
p=Path(sys.argv[1])
data=[json.loads((p/str(seed)/'turin-final.json').read_text()) for seed in [42,43]]
assert data[0]['request_index_sha256']==data[1]['request_index_sha256']
ctl=json.loads((p/'porto-equal-n.json').read_text())
assert ctl['n']==data[0]['requests']==data[1]['requests']==341
lines=['# Conditional DiffTraj on observed Turin bus journeys','','The single-day input contained 3,262 terminal-covered bus journeys with consistent fleet labels. Vehicle-disjoint partitions contain 2,555 training, 366 development and 341 reserved journeys. Both 200-epoch fits use the same conditional architecture and generation requests. No class input is used. The score is distributional and the trip header includes distance and OD cells.', '',
       '| Sample | Point density JS ↓ | OD JS ↓ | Length JS ↓ | Top-20 F1 ↑ |','|---|---:|---:|---:|---:|']
def row(label,v):
    lines.append('| '+label+' | '+' | '.join(f'{x:.4f}' for x in v)+' |')
for s,d in zip([42,43],data):
    row(f'Turin, seed {s}',[d[k] for k in ['density_js','trip_js','length_js','top20_cell_f1']])
row('Turin, real development vs real reserve',data[0]['real_real'])
for s in ['42','43']:
    row('Porto, long trips, equal-sized subsets, seed '+s,ctl['results'][s][1])
lines+=['','Porto rows are medians over 50 paired subsets of 341 old requests sampled from the 3,000 saved evaluations; the aggregate JSON reports the 10th and 90th percentiles. The real/real row uses 341 actual trips on each side. The absolute number of optimizer updates differs because the training sets differ in size. Turin reports one service day and a coarser observation clock.', '',
        '| Seed | Selected epoch | Real median (km) | Generated median (km) | Length W1 (km) | Points outside training bounds |','|---|---:|---:|---:|---:|---:|']
for s,d in zip([42,43],data):
    lines.append(f"| {s} | {d['epoch']} | {d['reference_median_m']/1000:.2f} | {d['generated_median_m']/1000:.2f} | {d['length_w1_m']/1000:.2f} | {d['generated_outside_fraction']:.2%} |")
(p/'REPORT.md').write_text('\n'.join(lines)+'\n')
