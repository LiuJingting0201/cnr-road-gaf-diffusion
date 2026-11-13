from pathlib import Path
import json,sys
p=Path(sys.argv[1]);control=json.loads((p/'porto-equal-n.json').read_text())
assert control['n']==1000
lines=['# Original DiffTraj on SUMO taxi trips and Porto long trips','','Each model receives the original eight trip attributes. Equal-sample Porto scores are medians across 50 paired subsets of 1,000 generated and observed trips. The local Porto full-population values use 3,000 requests. JS divergence is lower for closer distributions. Hotspot F1 increases with overlap.', '',
'| Data | Seed | Density JS ↓ | OD JS ↓ | Length JS ↓ | Hotspot F1 ↑ |','|---|---:|---:|---:|---:|---:|','| Porto, published Table 5 | — | .0087 | .0132 | .0242 | .847 |']
keys=['density_js','trip_js','length_js','top20_cell_f1']
def row(label,s,v):lines.append('| '+label+' | '+str(s)+' | '+' | '.join(f'{x:.4f}' for x in v)+' |')
phys=[]; hashes=set()
for seed in [42,43]:
    d=json.loads((p/str(seed)/'sumo-final.json').read_text())
    epochs=[json.loads(x) for x in (p/str(seed)/'epochs.jsonl').read_text().splitlines()]
    assert [e['epoch'] for e in epochs]==list(range(1,201)) and all(e['global_batch']==220 and e['train_seed']==seed for e in epochs)
    assert d['requests']==1000 and d['epoch']==200
    hashes.add(d['request_index_sha256'])
    porto=json.loads((p.parent/'porto-metric-review'/f'{seed}-long-on-long.json').read_text())['archived']
    row('Porto, all requests',seed,[porto[k] for k in keys])
    row('Porto, equal sample size',seed,control['results'][str(seed)][1])
    row('SUMO, taxi only',seed,[d[k] for k in keys])
    phys.append(f"| {seed} | {d['reference_median_m']/1000:.2f} | {d['generated_median_m']/1000:.2f} | {d['length_w1_m']/1000:.2f} | {100*d['generated_outside_fraction']:.2f}% |")
assert len(hashes)==1
row('SUMO observed vs observed','—',d['real_real'])
lines+=['','| SUMO seed | Observed median km | Generated median km | Length W1 km | Outside training bounds |','|---|---:|---:|---:|---:|',*phys,'','[Data and training protocol](PROTOCOL.md)']
(p/'REPORT.md').write_text('\n'.join(lines)+'\n')
