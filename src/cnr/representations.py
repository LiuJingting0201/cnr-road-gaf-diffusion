"""Historical Hilbert/GASF/GADF/MTF and explicit sequence controls.

The spatial quantizer follows the original repository's quadrant recursion.
Windows and train-only normalization follow this project's frozen protocol;
this is NOT a reproduction of the old full-trajectory classification accuracy.
"""
import numpy as np
import torch
from torch import nn
from cnr.models import gaf_channels

VARIANTS = ('hilbert_coarse_rgb', 'hilbert_fine_rgb', 'hilbert_fine_tcn',
            'delta_gaf', 'xy_motion_tcn', 'xy_gaf')


def hilbert_numpy(xy, eps=.0037):
    if eps <= 0 or not np.isfinite(eps):
        raise ValueError('eps must be finite and positive')
    x, y = xy[..., 0].copy(), xy[..., 1].copy()
    h = np.zeros_like(x); weight = 1.
    while eps <= 1:
        q = np.where(x < .5, np.where(y < .5, 0, 1), np.where(y < .5, 3, 2))
        weight /= 4; h += q*weight
        xx = np.where(q == 0, 2*y, np.where(q == 1, 2*x, np.where(q == 2, 2*x-1, 1-2*y)))
        yy = np.where(q == 0, 2*x, np.where(q == 1, 2*y-1, np.where(q == 2, 2*y-1, 2-2*x)))
        x, y = xx, yy; eps *= 4
    return h


def hilbert_torch(xy, eps=.0037):
    if eps <= 0 or not np.isfinite(eps):
        raise ValueError('eps must be finite and positive')
    x, y = xy[..., 0].double(), xy[..., 1].double()
    h = torch.zeros_like(x); weight = 1.
    while eps <= 1:
        q = torch.where(x < .5, torch.where(y < .5, 0, 1), torch.where(y < .5, 3, 2))
        weight /= 4; h = h+q*weight
        xx = torch.where(q == 0, 2*y, torch.where(q == 1, 2*x, torch.where(q == 2, 2*x-1, 1-2*y)))
        yy = torch.where(q == 0, 2*x, torch.where(q == 1, 2*y-1, torch.where(q == 2, 2*y-1, 2-2*x)))
        x, y = xx, yy; eps *= 4
    return h.float()


def per_sample_unit(h):
    return (h-h.amin(-1, keepdim=True))/(h.amax(-1, keepdim=True)-h.amin(-1, keepdim=True)).clamp_min(1e-12)


def hilbert_rgb(h, quantize=True, n_bins=8):
    """Matches pyts 0.13 GASF/GADF and uniform-bin MTF, tested below.

    Fixed theoretical channel scaling follows build_no_speed_paired_dataset.py;
    the old extracted helper's batch-dependent channel scaling is not used.
    """
    u = per_sample_unit(h).clamp(0, 1)
    v = (1-u.square()).clamp_min(0).sqrt()
    gasf = u[:, :, None]*u[:, None, :]-v[:, :, None]*v[:, None, :]
    gadf = v[:, :, None]*u[:, None, :]-u[:, :, None]*v[:, None, :]
    # pyts uses searchsorted(..., side='left') at uniform bin edges.
    edges = torch.arange(1, n_bins, device=h.device, dtype=h.dtype)/n_bins
    bins = (u[:, :, None] > edges).sum(-1)
    pairs = bins[:, :-1]*n_bins+bins[:, 1:]
    counts = torch.zeros(len(h), n_bins*n_bins, device=h.device, dtype=h.dtype)
    counts.scatter_add_(1, pairs, torch.ones_like(pairs, dtype=h.dtype))
    counts = counts.reshape(-1, n_bins, n_bins)
    prob = counts/counts.sum(-1, keepdim=True).clamp_min(1)
    batch = torch.arange(len(h), device=h.device)[:, None, None]
    mtf = prob[batch, bins[:, :, None], bins[:, None, :]]
    image = torch.stack(((gasf+1)/2, (gadf+1)/2, mtf), 1)
    if quantize:
        image = torch.round(image.clamp(0, 1)*255)/255
    return image*2-1


def training_stats(xy):
    x = xy.astype(np.float64)
    delta = np.diff(x, axis=1, prepend=x[:, :1])
    return {'xy_min': x.min(axis=(0, 1)).tolist(), 'xy_max': x.max(axis=(0, 1)).tolist(),
            'xy_mean': x.mean(axis=(0, 1)).tolist(), 'xy_std': x.std(axis=(0, 1)).clip(1e-6).tolist(),
            'delta_mean': delta.mean(axis=(0, 1)).tolist(),
            'delta_std': delta.std(axis=(0, 1)).clip(1e-6).tolist()}


class ImageClassifier(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
                                 nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
                                 nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 6))
    def forward(self, x):
        return self.net(x)


class TemporalBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8, 64), nn.SiLU(),
                                 nn.Conv1d(64, 64, 3, padding=dilation, dilation=dilation), nn.SiLU(),
                                 nn.Conv1d(64, 64, 1))
    def forward(self, x):
        return x+self.net(x)


class SequenceClassifier(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(channels, 64, 1),
                                 *[TemporalBlock(d) for d in (1, 2, 4, 8, 16, 32)],
                                 nn.GroupNorm(8, 64), nn.SiLU(), nn.AdaptiveAvgPool1d(1),
                                 nn.Flatten(), nn.Linear(64, 6))
    def forward(self, x):
        return self.net(x)


class RepresentationClassifier(nn.Module):
    def __init__(self, variant, stats):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        for k, value in stats.items():
            self.register_buffer(k, torch.tensor(value, dtype=torch.float64))
        channels = 1 if variant == 'hilbert_fine_tcn' else 4
        self.classifier = (SequenceClassifier(channels) if variant.endswith('tcn') else
                           ImageClassifier(3 if variant.startswith('hilbert') else 4))

    def encode(self, xy):
        if self.variant.startswith('hilbert'):
            unit = ((xy.double()-self.xy_min)/(self.xy_max-self.xy_min).clamp_min(1e-9)).clamp(0, 1)
            h = hilbert_torch(unit, .0037 if 'coarse' in self.variant else 1e-7)
            return per_sample_unit(h)[:, None] if self.variant.endswith('tcn') else hilbert_rgb(h)
        if self.variant == 'xy_gaf':
            return gaf_channels(xy)
        delta = torch.diff(xy, dim=1, prepend=xy[:, :1])
        zdelta = ((delta-self.delta_mean)/self.delta_std).float()
        if self.variant == 'xy_motion_tcn':
            zxy = ((xy-self.xy_mean)/self.xy_std).float()
            return torch.cat((zxy, zdelta), -1).transpose(1, 2)
        # Preserve the train-level displacement scale: do NOT re-minmax each window.
        u = torch.sigmoid(zdelta).transpose(1, 2)
        v = (1-u.square()).clamp_min(1e-7).sqrt()
        return torch.cat((u.unsqueeze(-1)*u.unsqueeze(-2)-v.unsqueeze(-1)*v.unsqueeze(-2),
                          v.unsqueeze(-1)*u.unsqueeze(-2)-u.unsqueeze(-1)*v.unsqueeze(-2)), 1)

    def forward(self, xy):
        return self.classifier(self.encode(xy))
