"""Render bus-minus-bike contrasts and actual-label quality on the common split."""
from pathlib import Path
import json
import sys
p=Path(sys.argv[1])
d=json.loads((p/'summary.json').read_text())
e=json.loads((p/'expansion.json').read_text())
lines=['# Does trip-average speed mask Beijing vehicle-class differences?', '',
       'Each signed effect is bus minus bicycle. For generated trajectories, the request and random draws are identical on both sides. Real trajectories use 94 disjoint observational pairs. Each pair contributes its two headers to the generated comparison. All variants share the same class-only architecture, data and fit budget. Changes in the no-mean and no-distance-cues rows include retraining from the corresponding seed.', '',
       '| Header and seed | Mean speed (m/s) | P90 speed (m/s) | Slow share (pp) | Absolute acceleration (m/s²) | Slow entries / 10 min |',
       '|---|---:|---:|---:|---:|---:|']
for k,vals in [('Observed',d['real'])]+[(k,v['delta']) for k,v in d['arms'].items()]:
    lines.append('| '+k+' | '+' | '.join(f'{v:+.3f}' for v in vals)+' |')
lines+=['','A slow entry starts after a non-slow interval and requires two consecutive intervals below 1 m/s. Initial slow periods do not count as entries. This metric depends on the observation clock; a low-speed interval does not establish a full stop.', '',
        '## Actual-label speed distribution error', '', 'Lower W1 is better. These values use each original bike or bus request with its observed class, independent of the switch contrast.', '',
        '| Variant | Bike W1 (m/s) | Bus W1 (m/s) |','|---|---:|---:|']
for k,row in d['arms'].items():
    v=row['original_label_speed_w1']
    lines.append(f"| {k} | {v['bike']:.2f} | {v['bus']:.2f} |")
lines+=['','## Additional observed segments','','| Class | Previous | Added | Expanded |','|---|---:|---:|---:|']
for k in e['original']['classes']:
    lines.append('| '+k+' | '+' | '.join(str(e[s]['classes'][k]) for s in ('original','added','expanded'))+' |')
lines+=['','The separate pack includes 4,729 training and 535 validation segments across 44 participants (originally 34). The added 782 spans contain 60–119 native observations; 169 recording files were previously unrepresented. The expanded pack was kept outside this ablation. [Protocol](PROTOCOL.md)']
(p/'REPORT.md').write_text('\n'.join(lines)+'\n')
