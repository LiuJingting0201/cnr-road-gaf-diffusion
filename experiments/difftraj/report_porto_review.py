"""Write matched-cohort results from aggregated audit records."""
from pathlib import Path
import json
import sys

out = Path(sys.argv[1])
records = {p.stem:json.loads(p.read_text()) for p in sorted(out.glob('*-on-*.json'))}
assert len(records) >= 4
for cohort in ('long','mixed'):
    hashes = {d['ref_sha256'] for k,d in records.items() if k.endswith('-on-'+cohort)}
    assert len(hashes)==1
lines = ['# Porto matched reproduction results', '',
         'Each row uses 3,000 conditional requests. Both model arms use identical observed coordinates in each evaluation population, verified by content hashes. The mixed training cohort supplies one shared city-core grid and length binning. The appendix reports different, incompletely specified Porto settings (published density .0087, trip .0132, length .0242, pattern .847).', '',
         '| Seed / train → evaluation | Density JS ↓ | OD JS ↓ | Length JS ↓ | Top-20 F1 ↑ | Length W1 (km) ↓ | Median observed / generated (km) |',
         '|---|---:|---:|---:|---:|---:|---:|']
for name,d in records.items():
    s = d['grids']['core']['all']
    original = d['archived']
    lines.append(f"| {name} | {s['density']:.5f} | {s['trip']:.5f} | {s['length']:.5f} | {s['pattern']:.2f} | {s['length_w1_km']:.3f} | {original['reference_median_m']/1000:.2f} / {original['generated_median_m']/1000:.2f} |")
lines += ['', '## Benchmark scores under the historical per-arm grids', '',
          '| Source | Density JS ↓ | OD JS ↓ | Length JS ↓ | Pattern ↑ |',
          '|---|---:|---:|---:|---:|','| Published Table 5 | .0087 | .0132 | .0242 | .847 |']
for name,d in records.items():
    seed,arm,_,cohort=name.split('-')
    if arm==cohort:
        a=d['archived']
        lines.append(f"| {name} | {a['density_js']:.5f} | {a['trip_js']:.5f} | {a['length_js']:.5f} | {a['top20_cell_f1']:.2f} |")
lines += ['', 'The local historical grids and bins come from each training arm. The shared city-core table above holds their physical scale fixed. The exact published boundaries, bins, n, and Porto sample selection have not been identified.',
          '', '## Same-size controls', '', 'Twenty disjoint random half-splits of the 3,000 real trips, 1,500 per group. Entries are medians over splits. Real versus generated uses matched requests and thus shares conditioning information. Observed versus observed estimates within-cohort variation over trips, without separating vehicles.', '',
          '| Seed / train → evaluation | Real/real hotspot F1 | Real/generated hotspot F1 | Real/real length W1 (km) | Real/generated length W1 (km) |',
          '|---|---:|---:|---:|---:|']
for name,d in records.items():
    s=d['grids']['core']
    a,b=s['real_real'],s['real_generated']
    lines.append(f"| {name} | {a['pattern'][1]:.2f} | {b['pattern'][1]:.2f} | {a['length_w1_km'][1]:.3f} | {b['length_w1_km'][1]:.3f} |")
lines += ['', '## Interpretation', '',
          'The mixed reference occupied only 17 full-grid cells. Ranking 20 cells admits zero counts, with ties resolved by cell index. Its core-grid reference occupied 227 cells. At seed 42 the mixed model scores .45 on the historical full grid and .80 on the city core. On the same core, two independent halves of real trips overlap at .90. Physical length error persists although quantile-bin length JS is small.',
          '', 'On the same mixed requests, training with short trips reduces physical length W1 from 1.349 to 0.442 km at seed 42. On long requests the long-trained model has smaller W1 (2.279 versus 3.756 km) although mixed training gives lower JS. JS summarizes bin mass while W1 also measures the distance moved in kilometres.',
          '', 'The planar-distance sensitivity uses the same metre-valued bins. Across the four seed-42 pairs the largest absolute JS shift from great-circle length is '+f"{max(abs(d['projected_length_js_same_metre_bins']-d['grids']['core']['all']['length']) for k,d in records.items() if k.startswith('42-')):.5f}.",
          '', 'The heatmaps use the mixed-training core, longitude and latitude axes, and one colour scale across all panels. Length curves interpolate empirical percentiles, with the horizontal display limited to the central distribution. Full-distribution scores include the tails.', '',
          '![Porto density](porto-density.png)', '', '![Porto lengths](porto-length.png)', '',
          'See [measurement protocol and sources](PROTOCOL.md) and the per-arm aggregate JSON records for full-grid scores, overflow, and split quantiles.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
