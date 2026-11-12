"""Verify the paired score records and write a metric-by-metric comparison."""
from pathlib import Path
import json,sys
base=Path(sys.argv[1])
rows={}
for train in ['long','mixed']:
    for test in ['long','mixed']:
        d=json.loads((base/f'{train}-on-{test}'/'porto-final.json').read_text())
        assert d['requests']==3000 and d['epoch']==200
        rows[train,test]=d
for test in ['long','mixed']:
    assert rows['long',test]['request_index_sha256']==rows['mixed',test]['request_index_sha256']
    assert rows['long',test]['reference_median_m']==rows['mixed',test]['reference_median_m']
lines=['# Porto post-training comparison','','Length JS is dimensionless and lower is better. Both bin systems are frozen from training observations. Paper Table 5 reports 0.0242 under an incompletely specified Porto protocol.','',
'| Training pool | Evaluation pool | JS, long bins | JS, mixed bins | Length W1 (km) | Paired length ratio | Reference median (km) | Generated median (km) |',
'|---|---|---:|---:|---:|---:|---:|---:|']
for (tr,te),v in rows.items():
    lines.append(f"| {tr} | {te} | {v['length_js_long_bins']:.5f} | {v['length_js_mixed_bins']:.5f} | {v['length_w1_m']/1000:.2f} | {v['median_length_ratio_moving']:.3f} | {v['reference_median_m']/1000:.2f} | {v['generated_median_m']/1000:.2f} |")
for te in ['long','mixed']:
    a,b=rows['long',te],rows['mixed',te]
    lines+=['',f"On the same {te} evaluation trips, relaxing the training cutoff changes length JS by {b['length_js_long_bins']-a['length_js_long_bins']:+.5f} with long-trip bins and {b['length_js_mixed_bins']-a['length_js_mixed_bins']:+.5f} with mixed-trip bins. Length W1 changes by {(b['length_w1_m']-a['length_w1_m'])/1000:+.2f} km."]
(base/'RESULTS.md').write_text('\n'.join(lines)+'\n')
