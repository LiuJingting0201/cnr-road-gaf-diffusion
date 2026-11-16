"""Evaluate fixed-request vehicle-label interventions on supported pairs of modes."""
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment

from five_class_evaluate import FEATURES, MODES, features, string_digest

SIGNATURE = (*FEATURES, 'mean_abs_acceleration_mps2')


def make_pairs(headers, labels):
    """Use the fixed calipers and cost of the previous bus/bicycle diagnostic for all pairs."""
    labels = np.asarray(labels)
    if labels.dtype.kind not in 'iu':
        if not np.isin(labels, MODES).all():
            raise ValueError('Unexpected vehicle label')
        labels = np.searchsorted(MODES, labels)
    if not np.isin(labels, np.arange(len(MODES))).all():
        raise ValueError('Unexpected vehicle label')
    if headers.shape != (len(labels), 8) or not np.isfinite(headers).all():
        raise ValueError('Expected eight finite trip attributes per request')
    pairs = {}
    for first, second in combinations(range(len(MODES)), 2):
        left, right = [np.flatnonzero(labels == code) for code in (first, second)]
        if len(left) == 0 or len(right) == 0:
            pairs[first, second] = left[:0], right[:0]
            continue
        a, b = headers[left, None, :], headers[None, right, :]
        attr_delta = np.abs(a[:, :, 1:6] - b[:, :, 1:6])
        minutes_bin_delta = np.abs(a[:, :, 0] - b[:, :, 0])
        minutes_bin_delta = np.minimum(minutes_bin_delta, 288 - minutes_bin_delta)
        cell_delta = np.concatenate((np.abs(a[:, :, 6:] // 16 - b[:, :, 6:] // 16),
                                     np.abs(a[:, :, 6:] % 16 - b[:, :, 6:] % 16)), axis=2)
        allowed = (attr_delta.max(2) <= .75) & (minutes_bin_delta <= 36) & (cell_delta.max(2) <= 2)
        costs = attr_delta.sum(2) + minutes_bin_delta / 36 + cell_delta.sum(2) / 2
        i, j = linear_sum_assignment(np.where(allowed, costs, 1.e7))
        keep = allowed[i, j]
        pairs[first, second] = left[i[keep]], right[j[keep]]
    return pairs


def required_alternatives(pairs):
    """Return the distinct counterfactuals not included in the true-label samples."""
    requested = set()
    for (first, second), (left, right) in pairs.items():
        requested.update((int(i), second) for i in left)
        requested.update((int(j), first) for j in right)
    return sorted(requested)


def motion_signature(xy_m, durations):
    """Four established motion summaries plus mean absolute tangential acceleration."""
    durations = np.asarray(durations)
    if np.any(durations <= 0):
        raise ValueError('Positive physical duration required')
    summary, speed = features(xy_m, durations)
    interval_s = durations / (xy_m.shape[1] - 1)
    acceleration = np.abs(np.diff(speed, axis=1)).mean(1) / interval_s
    return np.column_stack((summary, acceleration))


def contrast_report(pairs, ids, observed_signatures, direct_signatures, alternative_signatures):
    """Second minus first; both generated label effects use the same header and noise."""
    observed = np.asarray(observed_signatures)
    direct = np.asarray(direct_signatures)
    if observed.shape != (len(ids), len(SIGNATURE)) or direct.shape != observed.shape:
        raise ValueError('Trip signatures are not aligned with the request list')
    report = {}
    for (first, second), (left, right) in pairs.items():
        a, b = MODES[first], MODES[second]
        n = len(left)
        rec = dict(first=a, second=b, contrast=f'{b} minus {a}', matched_pairs=n,
                   paired_request_sha256=string_digest([f'{ids[i]}|{ids[j]}' for i, j in zip(left, right)]))
        report[f'{b}_minus_{a}'] = rec
        if not n:
            rec['coverage_note'] = 'No eligible request pairs'
            continue
        obs_a, obs_b = observed[left], observed[right]
        alt_b = np.stack([alternative_signatures[int(i), second] for i in left])
        alt_a = np.stack([alternative_signatures[int(j), first] for j in right])
        # Fixed trip and identical latent noise on both sides of each subtraction.
        at_left = alt_b - direct[left]
        at_right = direct[right] - alt_a
        effect = ((at_left + at_right) / 2).mean(0)
        obs_delta = (obs_b - obs_a).mean(0)
        pooled_sd = np.sqrt((obs_a.var(0) + obs_b.var(0)) / 2)
        if not np.isfinite(np.stack((effect, obs_delta, pooled_sd))).all():
            raise ValueError('Nonfinite contrast')
        rec['features'] = {}
        for f, name in enumerate(SIGNATURE):
            delta, generated, sd = float(obs_delta[f]), float(effect[f]), float(pooled_sd[f])
            standardized = delta / sd if sd > 0 else None
            # Descriptive screen fixed before seeing outcomes; not a significance test.
            legible = n >= 20 and sd > 0 and abs(standardized) >= .2
            rec['features'][name] = dict(observed=delta, generated=generated,
                generated_at_first_headers=float(at_left[:, f].mean()),
                generated_at_second_headers=float(at_right[:, f].mean()),
                absolute_error=abs(generated - delta), observed_pooled_sd=sd,
                observed_standardized=standardized, direction_evaluable=legible,
                same_direction=bool(delta * generated > 0) if legible else None,
                generated_fraction_of_observed=generated/delta if legible else None)
    return report
