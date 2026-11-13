"""Write a readable matched-label scorecard from the aggregate result."""
import json
from pathlib import Path
import sys
p=Path(sys.argv[1])
d=json.loads((p/'summary.json').read_text())
lines=['# Does training on stop–go rhythm improve vehicle label control?', '',
       f"All 535 held-out requests were drawn with both labels. The comparison below uses the two headers from each of {d['pairs']} disjoint observed bus/bike pairs. Signed differences are bus minus bicycle. Both seeds use identical validation requests.", '',
       '| Model | Slow share (pp) | Slow entries per 10 min | Mean speed (m/s) | P90 speed (m/s) |',
       '|---|---:|---:|---:|---:|']
for k,v in [('Observed pairs',d['real'])]+[(k,r['delta']) for k,r in d['arms'].items()]:
    lines.append('| '+k+' | '+' | '.join(f'{v[i]:+.3f}' for i in (2,4,0,1))+' |')
lines+=['','C is the class-conditioned baseline. CR adds soft slow-time and entry-rate supervision. Hard thresholds are used for evaluation. Entries represent sustained low speed on the 200-point clock, not confirmed stops at bus stops.', '',
        '| Seed | Change in absolute slow-share error (pp) | Change in absolute entry-rate error (per 10 min) | Bike speed W1, C → CR (m/s) | Bus speed W1, C → CR (m/s) |',
        '|---|---:|---:|---:|---:|']
for seed in (42,43):
    a,b=[d['arms'][f'{k}-{seed}'] for k in ('C','CR')]
    vs=[f"{a['original_label_speed_w1'][m]:.2f} → {b['original_label_speed_w1'][m]:.2f}" for m in ('bike','bus')]
    lines.append(f"| {seed} | {b['abs_error'][2]-a['abs_error'][2]:+.3f} | {b['abs_error'][4]-a['abs_error'][4]:+.3f} | "+' | '.join(vs)+' |')
lines+=['','Negative error changes indicate a closer signed contrast. The original-label speed W1 uses all held-out requests for the indicated mode. See [protocol](PROTOCOL.md) and [full metrics](summary.json).']
(p/'REPORT.md').write_text('\n'.join(lines)+'\n')
