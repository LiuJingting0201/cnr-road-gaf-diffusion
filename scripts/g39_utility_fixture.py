"""End-to-end utility cache isolation test with manufactured tracks only."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from g33_train import digest,save
from g39_utility import prepare,ARMS,load


def check():
    with tempfile.TemporaryDirectory(prefix='g39-utility-fixture-') as tmp:
        root=Path(tmp)/'source';root.mkdir()
        save(root/'complete.json',{'complete':True,'route_duration_equal_across_arms':True})
        for e in range(390201,390209):
            b=root/str(e);(b/'requests').mkdir(parents=True)
            save(b/'requests/requests.json',{'requests':[{'id':f'{e}-{c}','label':c,'source_parent':f'cohort{(e-390201)//2}-c{c}'} for c in range(5)]})
            if e<=390204:
                for arm in ARMS:
                    for seed in (42,43):
                        f=b/'generated'/f'{arm}-{seed}';f.mkdir(parents=True)
                        rows=[{'id':f'{e}-{c}/draw-{k}','request_id':f'{e}-{c}','label':c,'index':c*25+k} for c in range(5) for k in range(25)]
                        save(f/'candidates.json',{'rows':rows})
                        txy=np.tile(np.array([[0.,0.,0.],[1.,2.,0.]]),(125,1))
                        np.savez_compressed(f/'samples.npz',offsets=np.arange(126)*2,txy=txy)
                        save(f/'complete.json',{'complete':True,'artifacts':{n:digest(f/n) for n in ('samples.npz','candidates.json')}})
        raw={'ids':np.array([f'real{i}' for i in range(1000)]),'parent_ids':np.array([f'realparent{i}' for i in range(1000)]),
             'labels':np.repeat(np.arange(5),200),'partition':np.array(['train']*1000),'offset':np.arange(1001)*2,
             'time':np.tile([0.,1.],1000),'xy':np.tile(np.array([[0.,0.],[2.,0.]]),(1000,1))}
        def pool(folder,name):
            e=int(folder.parent.name)
            return {'complete':True},{'ids':np.array([f'{e}-{c}' for c in range(5)]),'labels':np.arange(5),'offsets':np.arange(6)*2,
                'txy':np.tile(np.array([[0.,0.,0.],[1.,2.,0.]]),(5,1))}
        with patch('g39_utility.load_dataset',return_value=({},raw)),patch('g39_utility.read_pool',side_effect=pool):
            prepare(root,Path(tmp)/'unused-data',Path(tmp)/'cache')
        rec=json.loads((Path(tmp)/'cache/complete.json').read_text())
        assert rec['nonbus_condition_parent_disjoint'] and rec['normalizer_real_train_only']
        for name in ('real','empirical',*ARMS):
            d=load(Path(tmp)/'cache',name);assert d['x'].shape==(1000,3,257) and np.array_equal(np.bincount(d['y']),[200]*5)
        evaldata=load(Path(tmp)/'cache','evaluation');assert len(evaldata['y'])==20
        assert set(evaldata['episode'])==set(range(390205,390209))
        for a in ARMS:assert np.array_equal(load(Path(tmp)/'cache',a)['x'],load(Path(tmp)/'cache','B')['x'])
        print(json.dumps({'complete':True,'scope':'manufactured tracks only, no research outcomes',
            'checks':['1000real unique parents','equal200perclass','shared synthetic identity','heldout cohort split','normalizer train-only','cache hashes','matching all-arm tensors']}))


if __name__=='__main__':check()
