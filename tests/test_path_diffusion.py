import numpy as np
import torch
from cnr.path_diffusion import decode
from cnr.route_paths import RoutePaths,decode_progress_numpy
from test_topology import network

def test_explicit_connected_path_and_no_crossing_jump(tmp_path):
 network(tmp_path);m=RoutePaths(tmp_path/'net.xml',radius=.2)
 points=np.array([[7,0],[9,0],[11,0],[15,0],[15,3],[15,6]])
 trace=m.trace(points,'private_car');assert trace is not None
 record=m.reconstruct(trace,'private_car');assert record is not None
 route,length=m.extend(record['points'],record['last_lane'],record['last_position'],'private_car',{},np.random.default_rng(1))
 rebuilt=decode_progress_numpy(route,length,record['progress'],points[0])
 assert np.max(np.linalg.norm(points-rebuilt,axis=-1))<.03
 assert m.trace(np.array([[2,0],[5,0],[5,3]]),'private_car') is None

def test_decode_gradient_monotone_and_bounded():
 route=torch.tensor([[[0.,0.],[10.,0.],[10.,10.]]]);length=torch.tensor([20.]);start=torch.tensor([[0.,0.]])
 z=torch.zeros(1,1,4,requires_grad=True)
 xy,progress=decode(z,1.,1.,route,length,start)
 assert torch.all(progress[:,1:]>=progress[:,:-1])
 xy.sum().backward();assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
 xy,progress=decode(torch.ones(1,1,4)*10,1.,1.,route,length,start)
 assert torch.allclose(xy[0,-1],route[0,-1])
 assert torch.allclose(xy[0,0],start[0])
