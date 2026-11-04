import numpy as np
import torch
from pyts.image import GramianAngularField, MarkovTransitionField
from cnr.representations import hilbert_numpy, hilbert_torch, hilbert_rgb, training_stats, RepresentationClassifier, VARIANTS


def original_hilbert(x, y, eps):
    if eps > 1:
        return 0
    if x < .5:
        if y < .5:
            return original_hilbert(2*y, 2*x, 4*eps)/4
        return (1+original_hilbert(2*x, 2*y-1, 4*eps))/4
    if y >= .5:
        return (2+original_hilbert(2*x-1, 2*y-1, 4*eps))/4
    return (3+original_hilbert(1-2*y, 2-2*x, 4*eps))/4


def test_hilbert_matches_old_recursive_helper():
    x=np.r_[np.random.default_rng(42).random((300,2)), [[0,0],[1,1],[.5,.5],[0,1],[1,0]]]
    for eps in (.0037, 1e-7):
        expected=np.array([original_hilbert(a,b,eps) for a,b in x])
        np.testing.assert_array_equal(hilbert_numpy(x,eps),expected)
        np.testing.assert_array_equal(hilbert_torch(torch.tensor(x),eps).numpy(),expected)


def test_three_channels_against_pyts():
    h=np.r_[np.random.default_rng(5).random((4,64)),np.zeros((1,64)),
            np.linspace(0,1,64)[None,:], np.resize(np.linspace(0,1,9),64)[None,:]].astype(np.float32)
    gasf=GramianAngularField(sample_range=(0,1),method='summation').fit_transform(h)
    gadf=GramianAngularField(sample_range=(0,1),method='difference').fit_transform(h)
    mtf=MarkovTransitionField(strategy='uniform',n_bins=8).fit_transform(h)
    expected=np.stack([gasf,gadf,mtf*2-1],1)
    np.testing.assert_allclose(hilbert_rgb(torch.tensor(h),quantize=False).numpy(),expected,atol=2e-5)


def test_variants_finite_forward():
    xy=np.random.default_rng(5).normal(size=(4,64,2)).astype(np.float32)
    stats=training_stats(xy)
    for variant in VARIANTS:
        m=RepresentationClassifier(variant,stats)
        logits=m(torch.tensor(xy)); assert logits.shape==(4,6)
        assert torch.isfinite(logits).all()
        logits.square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
