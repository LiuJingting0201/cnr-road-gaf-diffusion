"""Paired inference and provenance checks for the fresh-road replication."""
from pathlib import Path
import json,numpy as np
root=Path('docs/results/g13-shared-road')
rows=[json.loads((root/f'switch{s}.json').read_text())['rows'] for s in (42,43,44)]
assert all([(r['id'],r['permitted']) for r in row]==[(r['id'],r['permitted']) for r in rows[0]] for row in rows)
eligible=[[r for r in row if all(r['permitted'])] for row in rows]
ep=np.array([r['episode'] for r in eligible[0]])-260401
counts=np.random.default_rng(530711).multinomial(8,[1/8]*8,2000)
w=counts[:,ep];assert np.all(w.sum(1)>0)
ex=json.loads((root/'example.json').read_text());assert ex['condition']==eligible[0][0]['index']
report=dict(attempts=len(rows[0]),routes=len(ep),by_episode=np.bincount(ep,minlength=8).tolist(),examples=2*8*len(ep)*3)
for key in ['travel','stop']:
    vals=np.array([[[np.mean(r[c][key]) for c in ['bus','private_car']] for r in row] for row in eligible])
    delta=vals[:,:,0]-vals[:,:,1];v=delta.mean(0)
    report[key]=dict(means=vals.mean((0,1)).tolist(),difference=float(v.mean()),seed_differences=delta.mean(1).tolist(),ci=np.percentile(w@v/w.sum(1),[2.5,97.5]).tolist(),positive_routes=int(sum(v>0)),negative_routes=int(sum(v<0)))
for k in ['valid','caps']:
    report[k]=sum(r[c][k] for row in eligible for r in row for c in ['bus','private_car'])
x=np.array(ex['xy']);v=np.linalg.norm(np.diff(x,axis=1),axis=-1)/.25
report['fixed_case']=dict(condition=ex['condition'],seed=42,slot=0,travel=(v.sum(1)*.25).tolist(),stop=(v<.1).mean(1).tolist())
np.testing.assert_allclose(report['fixed_case']['travel'],[eligible[0][0][c]['travel'][0] for c in ['bus','private_car']],atol=.001)
(root/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
