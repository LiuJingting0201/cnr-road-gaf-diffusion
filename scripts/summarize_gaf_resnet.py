"""Export small E1 metrics without weights, data, or fabricated pending results."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--runs', default='runs/gaf-resnet-e1')
p.add_argument('--output', default='docs/results/gaf-resnet-e1')
a = p.parse_args()
root, out = Path(a.runs), Path(a.output)
out.mkdir(parents=True, exist_ok=True)
results = {}
for grid in [32, 64]:
    for init in ['random', 'imagenet']:
        name = f'grid{grid}-{init}'
        rows = []
        for seed in [42, 43, 44]:
            d = root/f'{name}-seed{seed}'
            if not (d/'complete.json').exists():
                continue
            row = json.loads((d/'complete.json').read_text())
            assert row['complete']
            target = out/d.name
            target.mkdir(exist_ok=True)
            for file in ['complete.json', 'best_validation.json', 'metrics.jsonl', 'config.json']:
                (target/file).write_text((d/file).read_text())
            rows.append(row['final_validation'])
        results[name] = {'finished_seeds': len(rows)}
        if rows:
            results[name] |= {metric: {'mean': float(np.mean([r[metric] for r in rows])), 'std_population': float(np.std([r[metric] for r in rows]))} for metric in ['accuracy', 'macro_f1', 'ece_10bins', 'nll']}
complete = all(r['finished_seeds'] == 3 for r in results.values())
(out/'summary.json').write_text(json.dumps({'complete': complete, 'cells': results}, indent=2))
lines = ['# E1 inherited Hilbert GAF with a self-trained ResNet18', '',
         'Full-vehicle participant classification diagnostic. Fixed epoch 50 is primary. No test evaluation; not the supplied group CNN or a reproduction of unavailable journal details. No claim of short-trajectory guidance effectiveness.', '',
         f'Status: {"all 12 runs complete" if complete else "partial; pending results must not be interpreted"}.', '',
         '| Grid and initialization | Completed seeds | Accuracy % | Macro-F1 % |', '|---|---:|---:|---:|']
for name, r in results.items():
    def fmt(key):
        return f'{r[key]["mean"]*100:.2f} +/- {r[key]["std_population"]*100:.2f}' if key in r else 'pending'
    lines.append(f'| {name} | {r["finished_seeds"]}/3 | {fmt("accuracy")} | {fmt("macro_f1")} |')
lines += ['', 'The next gate is duration-matched validation and candidate-image distinguishability, before original-GAF selection of generated trajectories. Final synthesis evaluation must use independent models and downstream real held-out data.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'complete': complete, 'cells': results}, indent=2))
