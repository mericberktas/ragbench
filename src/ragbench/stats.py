"""Cluster bootstrap over articles (two questions of one article are not independent)."""
from __future__ import annotations

import numpy as np


def _cluster_means(values: np.ndarray, clusters: np.ndarray):
    uniq, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq))
    return sums, counts


def cluster_bootstrap_mean(values, clusters, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Mean and (1-alpha) percentile interval, resampling whole clusters."""
    values = np.asarray(values, dtype=float)
    sums, counts = _cluster_means(values, np.asarray(clusters))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boot = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), (float(lo), float(hi))


def paired_cluster_bootstrap(a, b, clusters, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Mean of a-b over the same questions and its interval; significant if the interval excludes 0."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mean, (lo, hi) = cluster_bootstrap_mean(diff, clusters, n_boot, seed, alpha)
    return mean, (lo, hi), not (lo <= 0.0 <= hi)
