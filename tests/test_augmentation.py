import numpy as np
import pytest
from cnr.augmentation import time_warp,validate_training_parents


def test_time_warp_preserves_endpoints_and_monotone_progress():
    x=np.stack([np.linspace(0,100,512),np.zeros(512)],-1)
    for amplitude in [-.02,.02]:
        y=time_warp(x,amplitude)
        assert np.allclose(y[[0,-1]],x[[0,-1]])
        assert (np.diff(y[:,0])>0).all()
        assert not np.allclose(y,x)
    with pytest.raises(ValueError):time_warp(x,.1)


def test_validation_or_unknown_conditions_cannot_enter_augmentation():
    validate_training_parents(['a','a'],['a','b'],['c'])
    with pytest.raises(ValueError):validate_training_parents(['c'],['a','b'],['c'])
    with pytest.raises(ValueError):validate_training_parents(['z'],['a','b'],['c'])
    with pytest.raises(ValueError):validate_training_parents(['a'],['a','b'],['b'])
