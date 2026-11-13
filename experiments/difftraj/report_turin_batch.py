from pathlib import Path
import json
import sys
p=Path(sys.argv[1])
lines=['# Turin bus batch-size comparison','','Both fits use 200 epochs and the same 2,555 training journeys. The smaller batch makes 9,200 optimizer updates versus 600 for the original fit.', '']
for stage in ['development','reserved']:
    lines += [f'## {stage.capitalize()} requests','','| Seed | Global batch | Requests | Density JS | OD JS | Length JS | Pattern F1 | Observed / generated median km | Length W1 km | Outside |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    hashes=set()
    for seed in ['42','43']:
        for batch in [1024,56]:
            if stage=='development':
                f=p/(('old-' if batch==1024 else '')+seed)/'development.json'
            else:
                f=(p.parent/'turin-bus-only' if batch==1024 else p)/seed/'turin-final.json'
            d=json.loads(f.read_text());hashes.add(d['request_index_sha256'])
            lines.append(f"| {seed} | {batch} | {d['requests']} | {d['density_js']:.4f} | {d['trip_js']:.4f} | {d['length_js']:.4f} | {d['top20_cell_f1']:.2f} | {d['reference_median_m']/1000:.2f} / {d['generated_median_m']/1000:.2f} | {d['length_w1_m']/1000:.2f} | {100*d['generated_outside_fraction']:.2f}% |")
    assert len(hashes)==1
    lines+=['']
(p/'REPORT.md').write_text('\n'.join(lines))
