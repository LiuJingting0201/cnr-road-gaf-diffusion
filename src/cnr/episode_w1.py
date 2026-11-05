import numpy as np
from scipy.stats import wasserstein_distance

def episode_w1(a,b,e,stop,cnt):
    n=np.bincount(e,minlength=cnt.shape[1])
    denom=cnt@n
    if denom.min()==0: raise ValueError('Unsupported bootstrap group')
    if stop:
        diff=np.bincount(e,weights=a.mean(1)-b.mean(1),minlength=cnt.shape[1])
        return abs(cnt@diff/denom)
    grid=np.unique(np.r_[a.ravel(),b.ravel()]);width=np.diff(grid);cdf=np.zeros((cnt.shape[1],len(width)))
    for k in np.unique(e):
        aa=np.sort(a[e==k].ravel());bb=np.sort(b[e==k].ravel())
        cdf[k]=n[k]*(np.searchsorted(aa,grid[:-1],'right')/len(aa)-np.searchsorted(bb,grid[:-1],'right')/len(bb))
    w=cnt/denom[:,None]
    z=np.concatenate([np.abs(w[i:i+16]@cdf)@width for i in range(0,len(w),16)])
    assert abs(z[0]-wasserstein_distance(a.ravel(),b.ravel()))<.001
    return z
