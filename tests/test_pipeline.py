import numpy as np
import pandas as pd
import torch
from cnr.data import split_ids,continuous_ranges,CLASSES,prepare
import json
from cnr.models import gaf_channels,TrajectoryDenoiser


def test_split_is_vehicle_grouped():
    rows=[{'vehicle_id':f'{c}_{i}','class2':c} for c in CLASSES for i in range(20) for _ in range(3)]
    df=pd.DataFrame(rows);splits=split_ids(df)
    assert len(splits)==120 and set(splits.values())=={'train','val','test'}
    assert splits==split_ids(df.sample(frac=1,random_state=7))


def test_gaps_and_jumps_are_boundaries():
    t=np.array([0,.25,.5,4,4.25,4.5]);xy=np.array([[0,0],[1,0],[2,0],[3,0],[1000,0],[1001,0]])
    assert continuous_ranges(t,xy)==[(0,3),(3,4),(4,6)]


def test_gaf_structure_and_constant_track():
    x=torch.rand(3,64,2);g=gaf_channels(x)
    torch.testing.assert_close(g[:,:2],g[:,:2].transpose(-1,-2))
    torch.testing.assert_close(g[:,2:],-g[:,2:].transpose(-1,-2))
    assert torch.isfinite(gaf_channels(torch.ones(1,64,2))).all()


def test_denoiser_backward_and_condition_shapes():
    m=TrajectoryDenoiser(width=32);x=torch.randn(2,2,63)
    y=m(x,torch.tensor([1,900]),torch.tensor([0,5]),torch.randn(2,2))
    assert y.shape==x.shape
    y.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_preparation_split_manifest_and_training_normalization(tmp_path):
    rows=[]
    for c,label in enumerate(CLASSES):
        for vehicle in range(20):
            for t in range(70):
                rows.append({'vehicle_id':f'{c}_{vehicle}', 'class2':label,
                             'timestep_time':t*.25,'vehicle_x':vehicle*100+t,
                             'vehicle_y':c*100+t*.2})
    source=tmp_path/'raw.csv';pd.DataFrame(rows).to_csv(source,index=False)
    dest=tmp_path/'prepared';prepare(source,dest)
    manifest=pd.read_csv(dest/'manifest.csv',dtype={'vehicle_id':str})
    assert manifest.groupby('vehicle_id').split.nunique().max()==1
    assert ((manifest.last_time-manifest.first_time)==15.75).all()
    meta=json.loads((dest/'dataset.json').read_text())
    train=np.load(dest/'train.npz')['xy']
    np.testing.assert_allclose(meta['mean'],train.mean(axis=(0,1)),rtol=1e-5)
    assert meta['normalization_source']=='training_windows_only'
