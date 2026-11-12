"""Render comparable whole-trip, interval, coverage, and per-mode scores."""
from pathlib import Path
import json,sys
out=Path(sys.argv[1])
records={p.stem:json.loads(p.read_text()) for p in out.glob('*.json') if p.stem.startswith(('Porto-','Beijing-'))}
coverage=json.loads((out/'coverage.json').read_text())
names=['Porto-real','Porto-gen-42','Porto-gen-43','Beijing-real']+[f'Beijing-{a}-{s}' for a in ('B','V','C','VC') for s in (42,43)]
pct=lambda x,n:f'{100*x/n:.1f}%'
lines=['# Directed-road and speed feasibility on Porto and Beijing', '',
       'Porto includes 3,000 previously scored mixed-population requests per seed. Beijing includes 470 held-out labelled segments per arm and seed. B is the original model, V adds interval-speed supervision, C adds vehicle class, and VC adds both. A trip passes only if every interval passes. Interval rates pool original intervals, counting matching failures in the denominator.', '',
       '| Cohort | Continuous trips | Continuous and speed-compatible trips | Connected intervals | Connected and speed-compatible intervals | Point projections within 50 m | Median / P95 offset of projected points (m) |',
       '|---|---:|---:|---:|---:|---:|---:|']
for name in names:
    d=records[name]['all']; n=d['trips']; q=d['matched_offset_p50_p95']
    lines.append(f"| {name} | {d['walk']}/{n} ({pct(d['walk'],n)}) | {d['passed']}/{n} ({pct(d['passed'],n)}) | {pct(d['linked'],d['intervals'])} | {pct(d['reasonable'],d['intervals'])} | {pct(d['projected_within50'],d['n'])} | {q[0]:.1f} / {q[1]:.1f} |")
lines+=['','## Beijing by vehicle class','','The same 200 points per request give 199 intervals per trip. Entries are the percentage of original intervals simultaneously connected and speed-compatible. The two values in each model column correspond to seeds 42 and 43.', '',
        '| Class | Trips | Real | B | V | C | VC |','|---|---:|---:|---:|---:|---:|---:|']
for m,d in records['Beijing-real']['classes'].items():
    vals=[]
    for a in ('B','V','C','VC'):
        pair=[records[f'Beijing-{a}-{s}']['classes'][m] for s in (42,43)]
        vals.append(' / '.join(pct(v['reasonable'],v['intervals']) for v in pair))
    lines.append(f"| {m} | {d['trips']} | {pct(d['reasonable'],d['intervals'])} | "+' | '.join(vals)+' |')
lines+=['','## Coverage and speed sensitivity','',
        '| Cohort | Trips with points beyond extract box | In-box trips passing / all in-box | Passing with 0.8× / 1× / 1.2× speed ceiling |',
        '|---|---:|---:|---:|']
for name in names:
    d=records[name]['all']; c=coverage[name]
    lines.append(f"| {name} | {c['trips_outside']} | {c['in_bounds_pass']} / {c['in_bounds']} | {d['pass08']} / {d['passed']} / {d['pass12']} |")
lines+=['','## Interpretation','',
        'Porto generated trips reproduce parts of the road network, but their whole-trip pass rate is 4–5% versus 74% for observations. Failing a strict trajectory test means at least one transition failed. On Beijing, requiring all 199 transitions to pass leaves zero passing generated trips, even when increasing speed ceilings by 20%. The real Beijing control retains 177 passing trips. This motivates reporting both transition and whole-trip scores.',
        '', 'Speed-only supervision reduces the Beijing passing-interval share for seed 42 and increases it for seed 43. Class alone does not consistently improve it. The joint extension reduces passing intervals at both seeds. These descriptive comparisons use the same requests; 470 segments come from ten held-out participants.',
        '', 'Fractions measure support under this matcher and snapshot. Alternative matching paths, historical changes, multiple modes, 200-point interpolation and observation errors can affect scores. Offsets summarize projected points; projection coverage is reported separately. Speed thresholds are coarse plausibility bounds. The results do not establish individual traffic violations.', '',
        'See [protocol](PROTOCOL.md), [coverage and paired counts](coverage.json), and [matcher controls](controls.json).']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
