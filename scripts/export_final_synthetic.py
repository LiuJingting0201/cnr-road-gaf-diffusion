"""Export saved synthetic trajectories in the user's five-column format."""
import csv
import gzip
import json
from pathlib import Path
import zipfile
import numpy as np
from cnr.data import CLASSES
from cnr.final_protocol import sha256

assert json.loads(Path('runs/final-test-e7/complete.json').read_text())['complete']
root = Path('runs/final-synthetic-export'); root.mkdir(parents=True, exist_ok=False)
catalog = {'format': ['time', 'id', 'x', 'y', 'type'], 'time': 'seconds relative to each synthetic trajectory start, 0..127.75 at dt=0.25',
           'coordinates': 'Local SUMO XY metres on MoST; not GPS longitude/latitude',
           'types': CLASSES, 'scope': 'Saved locked-test outputs, five moving participant classes, no pedestrians. No joint traffic timeline, collision or signal compliance guarantee. Do not train on these and reuse the same final test.', 'files': {}}
for method in ['all_candidates', 'random', 'gaf', 'tcn', 'shuffled_gaf']:
    path = root / f'diffusion_{method}.csv.gz'; count = 0
    with gzip.open(path, 'wt', newline='') as f:
        writer = csv.writer(f); writer.writerow(catalog['format'])
        for seed in [42, 43, 44]:
            pool = np.load(f'runs/final-diffusion-e7/seed{seed}/candidates.npz')
            indices = np.arange(len(pool['xy'])) if method == 'all_candidates' else np.load(f'runs/final-diffusion-e7/selection/seed{seed}/{method}.npz')['indices']
            for idx in indices:
                ident = f'syn_{method}_s{seed}_i{idx}'
                for j, (x, y) in enumerate(pool['xy'][idx]):
                    writer.writerow([f'{j*.25:.2f}', ident, f'{x:.6f}', f'{y:.6f}', CLASSES[int(pool['labels'][idx])]])
                count += 1
    # Read back schema, trajectory counts, time endpoints and finite coordinates.
    seen = {}; rows = 0
    with gzip.open(path, 'rt', newline='') as f:
        reader = csv.DictReader(f); assert reader.fieldnames == catalog['format']
        for row in reader:
            time = float(row['time']); coords = [float(row['x']), float(row['y'])]
            assert np.isfinite(coords).all() and row['type'] in CLASSES[1:]
            n = seen.get(row['id'], 0); assert time == n*.25
            seen[row['id']] = n+1; rows += 1
    assert len(seen) == count and set(seen.values()) == {512}
    catalog['files'][path.name] = {'trajectories': count, 'rows': rows, 'sha256': sha256(path)}
(root / 'catalog.json').write_text(json.dumps(catalog, indent=2))
(root / 'README.txt').write_text('Synthetic trajectories from frozen diffusion models. See catalog.json for coordinate, time, class and evaluation restrictions.\nGAF-selected outputs are an evaluated variant, not a proven improvement. All-candidate and selected files overlap; do not concatenate them as independent trajectories.\n')
archive = root / 'synthetic_trajectories.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as z:
    for path in sorted(root.iterdir()):
        if path != archive: z.write(path, path.name)
Path('docs/results/final-test-e7/synthetic_catalog.json').write_text(json.dumps(catalog | {'archive_sha256': sha256(archive), 'server_archive': str(archive)}, indent=2))
print(json.dumps({'complete': True, 'archive': str(archive), 'bytes': archive.stat().st_size, 'catalog': catalog['files']}), flush=True)
