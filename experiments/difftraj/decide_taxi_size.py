"""Lock the training population from paired common-basis development results only."""
from pathlib import Path
import json
rows={}
for arm in ('10k','20k'):
    rows[arm]={}
    for seed in ('42','43'):
        d=json.loads(Path(arm,seed,'od-diagnostic.json').read_text())
        s=json.loads(Path(arm,seed,'development.json').read_text())
        rows[arm][seed]=dict(od=d['common']['full']['od_js'],w1=s['length_w1_m'],reference=d['reference_sha256'])
for seed in ('42','43'):
    assert rows['10k'][seed]['reference']==rows['20k'][seed]['reference']
winner='20k' if all(rows['20k'][s]['od']<rows['10k'][s]['od'] and rows['20k'][s]['w1']<=rows['10k'][s]['w1'] for s in ('42','43')) else '10k'
Path('selection.json').write_text(json.dumps(dict(rule='both seeds improve common-basis joint OD JS and neither worsens physical length W1',development=rows,selected=winner),indent=2))
Path('SELECTED').write_text(winner+'\n')
