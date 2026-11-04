import numpy as np
from cnr.candidate_selection import select_candidates


def test_shared_validity_and_ties_and_explicit_fallback():
    groups=[np.array([0,1,2]),np.array([3,4])]
    valid=np.array([True,False,True,False,False])
    priority=np.array([.9,.1,.2,.7,.3])
    random,failed=select_candidates(groups,valid,None,priority)
    tied,_=select_candidates(groups,valid,np.ones(5),priority)
    assert random.tolist()==[2,4] and np.array_equal(random,tied)
    assert failed.tolist()==[False,True]
    scored,_=select_candidates(groups,valid,np.array([2,999,1,999,0]),priority)
    assert scored.tolist()==[0,4]  # invalid high scores cannot win
