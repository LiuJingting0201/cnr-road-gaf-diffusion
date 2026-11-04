"""Export only the saved synthetic extras used in E5; never export real rows."""
import csv
import gzip
import json
from pathlib import Path
import zipfile
import numpy as np
from cnr.data import CLASSES
from cnr.final_protocol import sha256

base = Path('data/processed/augmentation-e5')
meta = json.loads((base / 'dataset.json').read_text())
path = Path('runs/augmentation-e5/pool/candidates.npz')
assert sha256(path) == meta['candidate_sha256']
pool = np.load(path)
splits = json.loads(Path('data/processed/most64/dataset.json').read_text())['split_by_id']
assert all(splits[str(ident)] == 'train' for ident in pool['parent_ids'])
out = Path('runs/training-synthetic-export'); out.mkdir(parents=True, exist_ok=False)
catalog = {'format': ['time', 'id', 'x', 'y', 'type'], 'source_pool_sha256': sha256(path),
           'scope': 'Exact E5 synthetic extras from training IDs and known training paths; one generator seed42. No real, validation or test trajectories are exported. GAF benefit was not established. Variants overlap and are alternatives, not independent sets.',
           'time': 'Relative seconds per trajectory, 0..127.75, dt=0.25',
           'coordinates': 'MoST SUMO XY metres, not GPS; no joint traffic timeline', 'files': {}}
for method in ['synthetic_random', 'synthetic_gaf', 'synthetic_tcn']:
    indices = np.asarray(meta['selections'][method]['indices'])
    assert len(indices) == 425
    stored = np.load(base / method / 'train.npz')
    np.testing.assert_array_equal(pool['xy'][indices], stored['xy'][meta['real_train']:])
    np.testing.assert_array_equal(pool['labels'][indices], stored['labels'][meta['real_train']:])
    dest = out / f'training_{method}.csv.gz'
    with gzip.open(dest, 'wt', newline='') as f:
        writer = csv.writer(f); writer.writerow(catalog['format'])
        for idx in indices:
            for step, (x, y) in enumerate(pool['xy'][idx]):
                writer.writerow([f'{step*.25:.2f}', f'train_{method}_i{idx}', f'{x:.6f}', f'{y:.6f}', CLASSES[int(pool['labels'][idx])]])
    counts = {}; rows = 0
    with gzip.open(dest, 'rt', newline='') as f:
        reader = csv.DictReader(f); assert reader.fieldnames == catalog['format']
        for row in reader:
            n = counts.get(row['id'], 0); assert float(row['time']) == n*.25
            assert np.isfinite([float(row['x']), float(row['y'])]).all()
            counts[row['id']] = n+1; rows += 1
    assert len(counts) == 425 and set(counts.values()) == {512}
    catalog['files'][dest.name] = {'trajectories': 425, 'rows': rows, 'sha256': sha256(dest)}
(out / 'catalog.json').write_text(json.dumps(catalog, indent=2))
(out / 'README.txt').write_text(catalog['scope']+'\n'+catalog['time']+'\n'+catalog['coordinates']+'\nUse each variant separately. Current final test has already been inspected; do not claim new tuning is independently tested on it.\n')
archive = out / 'training_synthetic_trajectories.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as z:
    for file in sorted(out.iterdir()):
        if file != archive: z.write(file, file.name)
Path('docs/results/final-test-e7/training_synthetic_catalog.json').write_text(json.dumps(catalog | {'server_archive': str(archive), 'archive_sha256': sha256(archive)}, indent=2))
print(json.dumps({'complete': True, 'archive': str(archive), 'bytes': archive.stat().st_size}), flush=True)
