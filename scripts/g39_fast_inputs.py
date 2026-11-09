"""Cache invariant native-clock inputs without changing samples, padding or RNG."""
import math
import numpy as np
import torch
from g33_model import DT,pack_motion


class MotionInputCache:
    def __init__(self,rows,clocks,targets,stats,device):
        self.device=torch.device(device)
        self.lengths=np.asarray([len(x) for x in targets],np.int64)
        offsets=np.r_[0,np.cumsum(self.lengths)]
        values=np.empty((int(offsets[-1]),5),np.float32)
        for i,(z,t) in enumerate(zip(targets,clocks)):
            a,b=offsets[i:i+2]
            if len(t)!=len(z)+1:raise ValueError('Motion and clock differ')
            delta=np.diff(t);midpoint=(t[:-1]+t[1:])/2
            values[a:b,0]=z
            values[a:b,1]=delta/t[-1]
            values[a:b,2:]=np.stack([midpoint/1000.,midpoint/t[-1],delta/DT],axis=1)
        paths=np.asarray(rows['paths']);start=paths[:,0]
        context=np.c_[(start-np.asarray(stats['start_mean']))/np.asarray(stats['start_sd']),
                      np.log(rows['length'])/10.,np.log(rows['duration'])/10.,
                      np.log(np.asarray(rows['length'])/rows['duration'])/5.].astype(np.float32)
        route=((paths-start[:,None])/1500.).astype(np.float32)
        convert=lambda x:torch.as_tensor(x,device=self.device)
        self.values=convert(values);self.offsets=convert(offsets[:-1]);self.sizes=convert(self.lengths)
        self.labels=convert(np.asarray(rows['labels'],np.int64));self.context=convert(context);self.route=convert(route)
        self.positions=torch.arange(int(math.ceil(self.lengths.max()/4)*4),device=self.device)
        self.bytes=sum(t.numel()*t.element_size() for t in
            [self.values,self.offsets,self.sizes,self.labels,self.context,self.route,self.positions])

    def pack(self,indices):
        indices=np.asarray(indices,np.int64)
        padded=int(math.ceil(self.lengths[indices].max()/4)*4)
        ix=torch.as_tensor(indices,device=self.device);position=self.positions[:padded][None,:]
        valid=position<self.sizes[ix,None]
        addresses=self.offsets[ix,None]+torch.minimum(position,self.sizes[ix,None]-1)
        packed=self.values[addresses].masked_fill(~valid[...,None],0.)
        # Original pack_motion returns contiguous motion/clock arrays, but a
        # dense transposed route. Keep those layouts as well as exact values.
        return {'x':packed[:,:,0].unsqueeze(1).contiguous(),
                'time_weights':packed[:,:,1].unsqueeze(1).contiguous(),
                'clock_features':packed[:,:,2:].transpose(1,2).contiguous(),
                'mask':valid[:,None,:].to(torch.float32),
                'labels':self.labels[ix],'context':self.context[ix],
                'route':self.route[ix].transpose(1,2)}


def checks(device='cpu'):
    from g33_model import clock
    times=[clock(t,'native_0_25s') for t in (.125,1.,5.125,31.7,128.25)]
    targets=[np.linspace(-1,2,len(t)-1,dtype=np.float32) for t in times]
    paths=np.arange(5*129*2,dtype=np.float32).reshape(5,129,2)/7
    rows={'paths':paths,'labels':np.arange(5),'length':np.arange(5,dtype=float)+100,
          'duration':np.asarray([t[-1] for t in times])}
    stats={'start_mean':[3.,6.],'start_sd':[7.,11.]}
    before=torch.get_rng_state().clone()
    cache=MotionInputCache(rows,times,targets,stats,device)
    assert torch.equal(before,torch.get_rng_state()),'Cache consumed CPU RNG'
    for indices in [[0],[2,0,2],[4,1,0,3],[3,4]]:
        ix=np.asarray(indices)
        expected=pack_motion([targets[i] for i in ix],[times[i] for i in ix],rows['labels'][ix],
            rows['paths'][ix],rows['length'][ix],rows['duration'][ix],stats,device)
        actual=cache.pack(ix)
        for key in expected:
            assert torch.equal(expected[key],actual[key]),key
            assert expected[key].stride()==actual[key].stride(),(key,expected[key].stride(),actual[key].stride())
    print('Exact cached input values/layouts: duplicates, padding, fractional endpoints, RNG')


if __name__=='__main__':checks()
