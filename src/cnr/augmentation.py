"""Fixed small time-warp baseline; no fitting to validation or generated data."""
import numpy as np


def validate_training_parents(parent_ids, train_ids, validation_ids):
    parent,train,validation=map(set,(parent_ids,train_ids,validation_ids))
    if not parent.issubset(train) or parent & validation or train & validation:
        raise ValueError('Augmentation conditions must belong exclusively to training IDs')


def time_warp(xy, amplitude):
    if abs(amplitude) > .02:
        raise ValueError('Preset time warp limited to 2 percent')
    t=np.linspace(0,1,len(xy))
    u=t+amplitude*np.sin(2*np.pi*t)
    return np.stack([np.interp(u,t,xy[:,axis]) for axis in range(2)],axis=-1).astype(np.float32)
