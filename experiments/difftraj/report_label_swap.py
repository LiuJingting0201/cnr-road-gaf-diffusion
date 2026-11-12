import json
from pathlib import Path
import sys
p=Path(sys.argv[1])
d=json.loads((p/'label-swap.json').read_text())
lines=['# Beijing vehicle-label switching experiment','','470 identical trip headers and common random numbers per intervention. Each of the four fits generated four labels per request. Values below are per-trip features. Speed is measured from model-clock positions.', '',
       '| Model | Median within-request range of mean speed (m/s) | Range of P90 speed (m/s) | Range of slow time (percentage points) | Range of absolute acceleration (m/s²) |','|---|---:|---:|---:|---:|']
for run,r in d['runs'].items():
    a,b,c,e=r['median_within_request_range']
    lines.append(f'| {run} | {a:.2f} | {b:.2f} | {c*100:.2f} | {e:.2f} |')
lines+=['','## Matched observational contrasts','','Signed difference of class means, computed on observed pairs and on label switches on the same request pool. A positive value means the first named class has the larger feature. Slow time is reported in percentage points. Counts of agreement are descriptive, not independent hypothesis tests.','']
for k,feature in enumerate(d['features']):
    scale=100 if k==2 else 1
    lines.extend([f'### {feature}','','| Contrast | Observed pairs | Observed difference | C-42 | C-43 | VC-42 | VC-43 |','|---|---:|---:|---:|---:|---:|---:|'])
    for name,v in d['matches'].items():
        nums=[v['real']]+[r['pairs'][name]['matched'] for r in d['runs'].values()]
        lines.append('| '+name+f" | {v['n']} | "+' | '.join('—' if x is None else f'{x[k]*scale:+.2f}' for x in nums)+' |')
    lines.append('')
lines+=['C adds class to the original header. VC also uses interval-speed supervision in training. Real trips are paired across labels within prespecified tolerances for header attributes. A real trip can appear in different class-pair comparisons. Exact support counts, unrestricted means, all paired effects and replay checks are in the aggregate JSON. See [protocol](PROTOCOL.md).']
(p/'REPORT.md').write_text('\n'.join(lines)+'\n')
