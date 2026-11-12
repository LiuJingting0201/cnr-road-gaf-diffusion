"""Report each stabilized cohort against the published Porto Table 5 row."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
metrics = ['density_js', 'trip_js', 'length_js', 'top20_cell_f1']
paper = [0.0087, 0.0132, 0.0242, 0.847]
lines = ['# Porto reproduction against DiffTraj Table 5', '',
         '| Population | Density ↓ | Trip ↓ | Length ↓ | Pattern ↑ | Length minus paper |',
         '|---|---:|---:|---:|---:|---:|',
         '| Published Porto | 0.0087 | 0.0132 | 0.0242 | 0.847 | — |']
for arm in ['long', 'mixed']:
    item = json.loads((root / f'{arm}-on-{arm}' / 'porto-final.json').read_text())
    assert item['requests'] == 3000
    values = [item[k] for k in metrics]
    lines.append('| '+arm+' | '+' | '.join(f'{v:.5f}' for v in values)+f' | {values[2]-paper[2]:+.5f} |')
lines.extend(['', 'The two rows describe separate population assumptions. Local length JS uses training-quantile bins and the pattern diagnostic uses 20 cells. The published Porto split, length bins and pattern settings remain unverified. Stabilized training uses clipping, cosine annealing and best development-loss selection. Cross-cohort scores and unstable runs serve as diagnostics.',''])
(root / 'PAPER_TABLE.md').write_text('\n'.join(lines))
