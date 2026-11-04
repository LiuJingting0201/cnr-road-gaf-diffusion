import numpy as np
import torch
from cnr.representations import RepresentationClassifier,training_stats


def test_frozen_teacher_passes_finite_gradient_to_generated_coordinates():
    stats=training_stats(np.random.default_rng(1).normal(size=(8,64,2)).cumsum(1))
    for variant in ['delta_gaf','delta_tcn']:
        teacher=RepresentationClassifier(variant,stats).eval()
        for p in teacher.parameters():p.requires_grad_(False)
        xy=torch.randn(3,64,2,requires_grad=True)
        loss=torch.nn.functional.cross_entropy(teacher(xy),torch.tensor([0,1,2]))
        loss.backward()
        assert torch.isfinite(xy.grad).all() and xy.grad.abs().sum()>0
        assert all(p.grad is None for p in teacher.parameters())
