"""Pure-Python rank correlations + small-sample stats for paper builders.

Avoids pulling in scipy. Spearman and Kendall implementations follow the
canonical definitions; ranks are averaged on ties so they reduce to the
no-tie formulas when none are present.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def average_ranks(values: Sequence[float]) -> list[float]:
    """Return rank of each element using midrank for ties (1-based)."""
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        midrank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k]] = midrank
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation; ties handled via midranks."""
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    return _pearson(average_ranks(xs), average_ranks(ys))


def kendall_tau(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Kendall's tau-b (handles ties by counting concordant - discordant over the
    geometric mean of total minus tied-x and total minus tied-y)."""
    n = len(xs)
    if n != len(ys) or n < 2:
        return float("nan")
    concordant = discordant = tied_x = tied_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 and dy == 0:
                continue                                 # mutual tie counts in neither
            if dx == 0:
                tied_x += 1
            elif dy == 0:
                tied_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    total = n * (n - 1) // 2
    denom = math.sqrt((total - tied_x) * (total - tied_y))
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom


def bootstrap_percentile_ci(
    xs: Sequence[float],
    ys: Sequence[float],
    statistic,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap (1-α) confidence interval for a paired-sample statistic.

    ``statistic(boot_xs, boot_ys)`` is called on each bootstrap resample of the
    paired (xs, ys) and must return a float. NaN-valued resamples are dropped.
    """
    import random

    n = len(xs)
    if n != len(ys) or n < 3:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boot_x = [xs[i] for i in idx]
        boot_y = [ys[i] for i in idx]
        v = statistic(boot_x, boot_y)
        if v == v:  # filter NaN
            estimates.append(v)
    if len(estimates) < 10:
        return (float("nan"), float("nan"))
    estimates.sort()
    lo = estimates[int(alpha / 2 * len(estimates))]
    hi = estimates[int((1 - alpha / 2) * len(estimates))]
    return (lo, hi)


def bootstrap_ci_1d(
    values: Sequence[float],
    statistic,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap (1-α) CI for a 1-D sample statistic (e.g. median)."""
    import random

    n = len(values)
    if n < 3:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_boot):
        boot = [values[rng.randrange(n)] for _ in range(n)]
        v = statistic(boot)
        if v == v:
            estimates.append(v)
    if len(estimates) < 10:
        return (float("nan"), float("nan"))
    estimates.sort()
    lo = estimates[int(alpha / 2 * len(estimates))]
    hi = estimates[int((1 - alpha / 2) * len(estimates))]
    return (lo, hi)


def _normal_cdf(z: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mann_whitney_u(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """Mann-Whitney U test (asymptotic normal with tie + continuity correction).

    Returns ``(U, p_value)`` where U is the U statistic for ``xs`` (count of
    pairs (x, y) with x > y, ties counted as ½). ``alternative`` is one of
    ``"two-sided"``, ``"greater"`` (xs stochastically larger than ys),
    ``"less"``. Asymptotic z-test; reasonable for n_x + n_y >= ~20.
    """
    from collections import Counter

    nx, ny = len(xs), len(ys)
    if nx == 0 or ny == 0:
        return (float("nan"), float("nan"))
    combined = list(xs) + list(ys)
    ranks = average_ranks(combined)
    rank_sum_x = sum(ranks[:nx])
    U = rank_sum_x - nx * (nx + 1) / 2

    mu = nx * ny / 2.0
    N = nx + ny
    counts = Counter(combined)
    tie_correction = sum(t ** 3 - t for t in counts.values() if t > 1)
    if tie_correction == 0:
        var = nx * ny * (N + 1) / 12.0
    else:
        var = nx * ny * ((N ** 3 - N) - tie_correction) / (12.0 * N * (N - 1))
    if var <= 0:
        return (U, float("nan"))
    sd = math.sqrt(var)

    if alternative == "greater":
        z = (U - mu - 0.5) / sd
        p = 1.0 - _normal_cdf(z)
    elif alternative == "less":
        z = (U - mu + 0.5) / sd
        p = _normal_cdf(z)
    elif alternative == "two-sided":
        z = (abs(U - mu) - 0.5) / sd
        p = 2.0 * (1.0 - _normal_cdf(max(z, 0.0)))
    else:
        raise ValueError(f"unknown alternative: {alternative!r}")
    return (U, max(0.0, min(1.0, p)))


def jonckheere_terpstra(
    groups: Sequence[Sequence[float]],
    *,
    alternative: str = "increasing",
) -> tuple[float, float]:
    """Jonckheere–Terpstra test for ordered alternative (asymptotic normal).

    ``groups`` is ``k`` ordered samples; under ``"increasing"`` the alternative
    is non-decreasing medians across groups (group 0 stochastically smallest).
    Returns ``(J, p_value)``. No tie correction (acceptable when ties are rare).
    """
    k = len(groups)
    if k < 2:
        return (float("nan"), float("nan"))
    sizes = [len(g) for g in groups]
    if any(n == 0 for n in sizes):
        return (float("nan"), float("nan"))
    N = sum(sizes)
    J = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            for a in groups[i]:
                for b in groups[j]:
                    if a < b:
                        J += 1.0
                    elif a == b:
                        J += 0.5

    mu = (N * N - sum(n * n for n in sizes)) / 4.0
    var = (N * N * (2 * N + 3)
           - sum(n * n * (2 * n + 3) for n in sizes)) / 72.0
    if var <= 0:
        return (J, float("nan"))
    sd = math.sqrt(var)

    if alternative == "increasing":
        z = (J - mu) / sd
        p = 1.0 - _normal_cdf(z)
    elif alternative == "decreasing":
        z = (J - mu) / sd
        p = _normal_cdf(z)
    elif alternative == "two-sided":
        z = abs(J - mu) / sd
        p = 2.0 * (1.0 - _normal_cdf(z))
    else:
        raise ValueError(f"unknown alternative: {alternative!r}")
    return (J, max(0.0, min(1.0, p)))


def fraction_at_least(values: Iterable[float], threshold: float) -> float:
    """Share of ``values`` at least ``threshold``; 0.0 on empty input."""
    total = 0
    hit = 0
    for v in values:
        total += 1
        if v >= threshold:
            hit += 1
    return hit / total if total else 0.0


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Return (mean, sample std). Sample std uses n-1; std=0 on n<2."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    mean = sum(values) / n
    if n < 2:
        return (mean, 0.0)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return (mean, math.sqrt(var))
