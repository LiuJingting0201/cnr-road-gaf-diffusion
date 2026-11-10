"""Shared one-second coordinate-derived speed for trajectory map colors."""
import numpy as np

def map_speed(txy):
    times = txy[:, 0] - txy[0, 0]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(txy[:, 1:], axis=0), axis=1))]
    count = int(np.floor(times[-1]))
    if count:
        speed = np.diff(np.interp(np.arange(count + 1), times, arc))
        return np.interp(times, np.arange(count) + .5, speed)
    return np.full(len(times), arc[-1] / max(times[-1], 1e-12))

def speed_limit(tracks):
    return max(1., float(np.ceil(max(map_speed(t).max() for t in tracks.values()))))
