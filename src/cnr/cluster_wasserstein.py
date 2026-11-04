"""Exact 1D W1 after resampling whole, equally weighted clusters."""
import numpy as np


def cluster_w1(first, second, counts, batch_size=32):
    """Rows are clusters; columns are observations within each cluster.

    Both distributions receive identical cluster multiplicities. Within-cluster
    observation counts may differ (e.g. K generated candidates vs one real trace).
    Integration uses the full empirical support, with no histogram approximation.
    """
    first, second, counts = [np.asarray(x, dtype=np.float64) for x in (first, second, counts)]
    if (first.ndim != 2 or second.ndim != 2 or counts.ndim != 2
            or first.shape[0] != second.shape[0] or counts.shape[1] != len(first)
            or first.shape[1] == 0 or second.shape[1] == 0
            or not all(np.isfinite(x).all() for x in (first, second, counts))
            or np.any(counts < 0) or np.any(counts.sum(1) <= 0)):
        raise ValueError('Expected finite cluster arrays and nonnegative nonempty resamples')
    grid = np.unique(np.concatenate((first.ravel(), second.ravel())))
    if len(grid) < 2:
        return np.zeros(len(counts))
    cdf_difference = np.stack([
        np.searchsorted(np.sort(x), grid[:-1], side='right') / len(x)
        - np.searchsorted(np.sort(y), grid[:-1], side='right') / len(y)
        for x, y in zip(first, second)])
    widths = np.diff(grid)
    weights = counts / counts.sum(1, keepdims=True)
    result = []
    for start in range(0, len(weights), batch_size):
        cdf = weights[start:start + batch_size] @ cdf_difference
        result.extend(np.abs(cdf) @ widths)
    return np.array(result)


def nearest_motion(query, reference, query_labels, reference_labels):
    """Time-aligned, same-class mean absolute profile distance; no time warping."""
    query, reference = np.asarray(query), np.asarray(reference)
    distances, indices = [], []
    for row, label in zip(query, query_labels):
        eligible = np.flatnonzero(reference_labels == label)
        if not len(eligible):
            raise ValueError('Missing reference class')
        d = np.abs(reference[eligible] - row).mean(1)
        j = int(np.argmin(d))
        distances.append(float(d[j])); indices.append(int(eligible[j]))
    return np.array(distances), np.array(indices)
