"""Matched physical-clock/257-bin motion diffusion.

Training pairs may contain observed route geometry. The sampling API accepts only
declared route/class requests and draws duration from the training-side prior.
The caller must enforce independent request provenance: numeric path arrays
cannot by themselves prove that a path was available before a reference run.
GAF is deliberately absent from this baseline, to permit frozen feedback later.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


CLASSES = ["bus", "motorcycle", "private_car", "bycicle", "taxi_or_uber"]
ARMS = ("relative_257", "native_0_25s")
DT = .25


def clock(duration, arm):
    if not np.isfinite(duration) or duration <= 0 or arm not in ARMS:
        raise ValueError("Positive finite duration and known arm required")
    return (np.linspace(0., duration, 257) if arm == "relative_257"
            else np.r_[np.arange(0., duration, DT), duration])


def encode(progress, times):
    progress, times = np.asarray(progress, float), np.asarray(times, float)
    if (len(progress) != len(times) or len(times) < 2 or not np.isfinite(progress).all()
            or not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0)
            or np.any(np.diff(progress) < -1e-10) or abs(progress[0]) > 1e-9
            or abs(progress[-1]-1.) > 1e-9):
        raise ValueError("Invalid normalized progress/explicit clock")
    return np.log1p(np.maximum(np.diff(progress), 0) / (np.diff(times)/times[-1])).astype(np.float32)


def decode(z, times, mean, sd):
    """Zero speed is allowed; report every clamp/fallback, preserve final time."""
    z, times = np.asarray(z, float), np.asarray(times, float)
    if len(z)+1 != len(times) or not np.isfinite(z).all():
        raise ValueError("Sample and clock lengths differ/nonfinite sample")
    raw = z*sd+mean
    clipped = np.clip(raw, 0., 8.)
    mass = np.expm1(clipped)*np.diff(times)/times[-1]
    flat = bool(mass.sum() < 1e-12)
    if flat:
        mass = np.diff(times)/times[-1]
    p = np.r_[0., np.cumsum(mass)]
    p /= p[-1]
    return p, {"negative_bins": int((raw < 0).sum()), "upper_clipped_bins": int((raw > 8).sum()),
               "constant_speed_fallback": flat}


def fit_duration(lengths, durations, labels):
    """Classwise log-length ridge with empirical residuals, train only.

    Same prior in both arms. This is not the historical joint-duration G22 model.
    Residual bootstrap preserves classwise non-Gaussian variation without using
    a held-out duration, and is reported as a semi-parametric prior.
    """
    result = {"kind": "classwise_log_length_ridge_empirical_residual",
              "ridge": .001, "minimum_duration_s": .25,
              "maximum_duration_s": None, "maximum_mean_speed_mps": 55.,
              "observed_training_duration_range_s": [float(np.min(durations)), float(np.max(durations))],
              "classes": []}
    for c in range(len(CLASSES)):
        selected = np.asarray(labels) == c
        x, y = np.log(np.asarray(lengths)[selected]), np.log(np.asarray(durations)[selected])
        if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Insufficient finite class training support for duration")
        center = float(x.mean())
        design = np.c_[np.ones(len(x)), x-center]
        beta = np.linalg.solve(design.T@design+np.diag([0., .001]), design.T@y)
        residual = y-design@beta
        result["classes"].append({"center_log_length": center, "intercept": float(beta[0]),
                                  "slope": float(beta[1]), "residuals": residual.tolist(),
                                  "train_n": int(selected.sum())})
    return result


def sample_duration(prior, lengths, labels, rng):
    result, flags = [], []
    for length, label in zip(lengths, labels):
        if not np.isfinite(length) or length <= 0:
            raise ValueError("Declared route length must be positive")
        row = prior["classes"][int(label)]
        log_t = row["intercept"]+row["slope"]*(math.log(length)-row["center_log_length"])+rng.choice(row["residuals"])
        raw = float(np.exp(np.clip(log_t, -30., 30.)))
        lower = max(prior["minimum_duration_s"], length/prior["maximum_mean_speed_mps"])
        value = max(raw, lower)
        result.append(value); flags.append(value != raw)
    return np.asarray(result), np.asarray(flags)


class ChannelNorm(nn.Module):
    """LayerNorm across channels at each physical position, never padded time."""
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, width, condition=128, dilation=1):
        super().__init__()
        self.norm = ChannelNorm(width)
        self.conv = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.condition = nn.Linear(condition, width)
        self.global_context = nn.Linear(width, width)
        self.out = nn.Conv1d(width, width, 1)

    def forward(self, x, condition, mask):
        x = x*mask
        pooled = x.sum(-1)/mask.sum(-1).clamp_min(1.)
        h = self.conv(F.silu(self.norm(x))*mask)
        h = h+(self.condition(condition)+self.global_context(pooled))[:, :, None]
        return (x+self.out(F.silu(h)))*mask


class MotionDenoiser(nn.Module):
    """Small masked temporal U-Net with route/class/duration conditioning.

    Pooling windows and upsampling ratios are fixed rather than stretched to the
    padded length. Combined with per-position normalization this makes arbitrary
    right padding observationally inert on valid output positions.
    """
    def __init__(self, base=64, classes=5):
        super().__init__()
        widths = [base, base*3//2, base*2]
        self.base, self.widths = base, widths
        self.time = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))
        self.label = nn.Embedding(classes, 128)
        self.context = nn.Sequential(nn.Linear(5, 128), nn.SiLU(), nn.Linear(128, 128))
        self.route = nn.Sequential(nn.Conv1d(2, 64, 5, padding=2), nn.SiLU(),
                                   nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.SiLU(),
                                   nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.input = nn.Conv1d(4, widths[0], 1)
        self.enc0 = nn.ModuleList([Block(widths[0]), Block(widths[0], dilation=2)])
        self.down1 = nn.Conv1d(widths[0], widths[1], 1)
        self.enc1 = nn.ModuleList([Block(widths[1]), Block(widths[1], dilation=2)])
        self.down2 = nn.Conv1d(widths[1], widths[2], 1)
        self.middle = nn.ModuleList([Block(widths[2], dilation=d) for d in (1, 2, 4, 8)])
        self.up1 = nn.Conv1d(widths[2]+widths[1], widths[1], 1)
        self.dec1 = nn.ModuleList([Block(widths[1]), Block(widths[1], dilation=2)])
        self.up0 = nn.Conv1d(widths[1]+widths[0], widths[0], 1)
        self.dec0 = nn.ModuleList([Block(widths[0]), Block(widths[0], dilation=2)])
        self.out = nn.Conv1d(widths[0], 1, 1)

    @staticmethod
    def blocks(blocks, h, condition, mask):
        for block in blocks:
            h = block(h, condition, mask)
        return h

    @staticmethod
    def pool(h, mask):
        count = F.avg_pool1d(mask, 2)*2
        pooled = F.avg_pool1d(h*mask, 2)*2/count.clamp_min(1.)
        return pooled, (count > 0).to(mask.dtype)

    def forward(self, x, t, labels, context, route, clock_features, mask):
        if x.shape[-1] % 4:
            raise ValueError("Pad sequence to a multiple of four")
        frequency = torch.exp(torch.arange(64, device=x.device)*(-math.log(10000)/63))
        angle = t[:, None]*frequency[None, :]
        condition = (self.time(torch.cat([angle.sin(), angle.cos()], 1))+
                     self.label(labels)+self.context(context)+self.route(route))
        h0 = self.blocks(self.enc0, self.input(torch.cat([x*mask, clock_features*mask], 1))*mask, condition, mask)
        h1, m1 = self.pool(h0, mask)
        h1 = self.blocks(self.enc1, self.down1(h1)*m1, condition, m1)
        h2, m2 = self.pool(h1, m1)
        h2 = self.blocks(self.middle, self.down2(h2)*m2, condition, m2)
        h1 = self.blocks(self.dec1, self.up1(torch.cat([F.interpolate(h2, scale_factor=2, mode="nearest"), h1], 1))*m1, condition, m1)
        h0 = self.blocks(self.dec0, self.up0(torch.cat([F.interpolate(h1, scale_factor=2, mode="nearest"), h0], 1))*mask, condition, mask)
        return self.out(h0)*mask


def pack_motion(z_values, clocks, labels, paths, lengths, durations, stats, device):
    lengths_bin = np.asarray([len(z) for z in z_values])
    padded = int(math.ceil(lengths_bin.max()/4)*4)
    z = np.zeros((len(z_values), 1, padded), np.float32)
    mask = np.zeros_like(z)
    weights = np.zeros_like(z)
    clock_feature = np.zeros((len(z_values), 3, padded), np.float32)
    for i, (values, times) in enumerate(zip(z_values, clocks)):
        n = len(values)
        if n+1 != len(times):
            raise ValueError("Motion target and clock disagree")
        z[i, 0, :n] = values; mask[i, 0, :n] = 1
        weights[i, 0, :n] = np.diff(times)/times[-1]
        midpoint = (times[:-1]+times[1:])/2
        clock_feature[i, :, :n] = np.stack([midpoint/1000., midpoint/times[-1], np.diff(times)/DT])
    paths = np.asarray(paths)
    start = paths[:, 0]
    context = np.c_[(start-np.asarray(stats["start_mean"]))/np.asarray(stats["start_sd"]),
                    np.log(lengths)/10., np.log(durations)/10., np.log(np.asarray(lengths)/durations)/5.].astype(np.float32)
    route = ((paths-start[:, None])/1500.).astype(np.float32).transpose(0, 2, 1)
    convert = lambda v: torch.as_tensor(v, device=device)
    return {"x": convert(z), "mask": convert(mask), "time_weights": convert(weights), "clock_features": convert(clock_feature),
            "labels": convert(np.asarray(labels, np.int64)), "context": convert(context), "route": convert(route)}


def predict(model, batch, x, t):
    return model(x, t, batch["labels"], batch["context"], batch["route"], batch["clock_features"], batch["mask"])


@torch.no_grad()
def sample(model, requests, stats, arm, seed, device="cuda", ddim_steps=100, token_budget=32768):
    """Only public/generated request paths/lengths/classes; never true duration.

    Stable per-request seeds preserve paired duration and initial motion noise
    across feedback strategies. Different model arms have different dimensions.
    """
    import hashlib
    rows = []
    schedule = torch.cumprod(1-torch.linspace(.0001, .02, 1000, device=device), 0)
    timesteps = torch.linspace(999, 0, ddim_steps).long().tolist()
    for request in requests:
        if any(key in request for key in ("duration", "true_duration", "progress", "target_xy")):
            raise ValueError("Generation request must not supply future duration/progress/coordinates")
        key = int.from_bytes(hashlib.sha256(f"G33:{request['id']}:{seed}".encode()).digest()[:8], "little")
        rng = np.random.default_rng(key)
        duration, bounded = sample_duration(stats["duration_prior"], [request["length"]], [request["label"]], rng)
        times = clock(float(duration[0]), arm)
        n = len(times)-1
        batch = pack_motion([np.zeros(n, np.float32)], [times], [request["label"]],
                            [request["path"]], [request["length"]], duration, stats, device)
        generator = torch.Generator(device=device).manual_seed(key)
        x = torch.randn(batch["x"].shape, generator=generator, device=device)*batch["mask"]
        for j, t in enumerate(timesteps):
            epsilon = predict(model, batch, x, torch.full((1,), t, dtype=torch.long, device=device))
            clean = (x-(1-schedule[t]).sqrt()*epsilon)/schedule[t].sqrt()
            previous = schedule[timesteps[j+1]] if j+1 < len(timesteps) else 1.
            x = (previous**.5*clean+(1-previous)**.5*epsilon)*batch["mask"]
        profile, flags = decode(x[0, 0, :n].cpu().numpy(), times, stats["motion_mean"], stats["motion_sd"])
        rows.append({"id": request["id"], "duration": float(duration[0]), "time": times,
                     "progress": profile, "duration_bounded": bool(bounded[0]), **flags})
    return rows


def self_test():
    torch.set_num_threads(1); torch.manual_seed(33001)
    model = MotionDenoiser(base=16).eval()
    stats = {"start_mean": [0, 0], "start_sd": [1, 1]}
    path = np.c_[np.linspace(0, 100, 129), np.zeros(129)]
    times = clock(2.125, "native_0_25s")
    assert times[-1] == 2.125 and times[-1]-times[-2] == .125
    p = np.linspace(0, 1, len(times)); p[2:5] = p[2]
    raw = encode(p, times)
    recovered, flags = decode(raw, times, 0., 1.)
    assert np.allclose(recovered, p, atol=1e-7) and np.all(np.diff(recovered) >= 0)
    assert recovered[0] == 0 and recovered[-1] == 1 and not flags["constant_speed_fallback"]
    b = pack_motion([raw], [times], [0], [path], [100.], [times[-1]], stats, "cpu")
    t = torch.tensor([42])
    out = predict(model, b, b["x"], t)
    expanded = dict(b)
    for name in ("x", "mask", "clock_features"):
        expanded[name] = F.pad(b[name], (0, 12))
    out_more = predict(model, expanded, expanded["x"], t)
    assert torch.allclose(out, out_more[..., :out.shape[-1]], atol=2e-6, rtol=2e-6)
    assert out_more[..., out.shape[-1]:].abs().max() == 0
    _, flags = decode(np.full(len(times)-1, -100.), times, 0., 1.)
    assert flags["constant_speed_fallback"]
    # Duration support and sample prior use only provided training records.
    labels = np.repeat(np.arange(5), 3)
    lengths = np.tile([100., 200., 400.], 5)
    duration = lengths/np.repeat(np.arange(1, 6), 3)
    prior = fit_duration(lengths, duration, labels)
    sampled, bounded = sample_duration(prior, lengths, labels, np.random.default_rng(1))
    assert np.all(np.isfinite(sampled)) and np.all(sampled > 0) and not bounded.any()
    print("g33_model_checks_passed: explicit_fractional_clock,zero_stops,endpoint,mask_padding_invariance,duration_prior")


if __name__ == "__main__":
    self_test()
