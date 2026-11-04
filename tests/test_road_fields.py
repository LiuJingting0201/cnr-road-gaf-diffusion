import numpy as np
import torch
from cnr.road_diffusion import RoadFields,RoadDenoiser


def test_field_geometry_gradients_and_boundary(tmp_path):
    # Vertical road at x=0, exact linear distance over this synthetic raster.
    field=np.broadcast_to(np.arange(8,dtype=np.float32)[None,:],(8,8)).copy()
    path=tmp_path/'map.npz'
    np.savez(path,distance=np.stack([field]*6),origin=np.array([0.,0.]),resolution=np.array(1.))
    roads=RoadFields(path)
    xy=torch.tensor([[[2.5,3.]],[[9.,3.]]],requires_grad=True)
    d=roads.distances(xy,torch.tensor([0,1]))
    torch.testing.assert_close(d,torch.tensor([[2.5],[9.]]))
    d.sum().backward();torch.testing.assert_close(xy.grad[...,0],torch.ones(2,1))
    patch=roads.patches(torch.tensor([[3.,3.]]),torch.tensor([0]),size=8,radius=2)
    assert patch.shape==(1,2,8,8) and torch.isfinite(patch).all()


def test_map_denoiser_has_finite_gradient():
    model=RoadDenoiser()
    y=model(torch.randn(2,2,63),torch.tensor([1,20]),torch.tensor([0,1]),torch.randn(2,2),torch.rand(2,2,32,32))
    y.square().mean().backward()
    assert torch.isfinite(y).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
