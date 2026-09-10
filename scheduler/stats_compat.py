"""Lightweight, standalone statistical routines compatible with SciPy.

Provides zero-dependency (pure Python/NumPy) implementations of:
- paired Student's t-test (ttest_rel)
- Student's t confidence intervals (t_interval / ci95)
- standard error of the mean (sem)
- Wilcoxon signed-rank test (wilcoxon)

This avoids issues in locked-down environments (such as Windows WDAC/AppLocker)
where native C/C++ .pyd DLLs in scipy.stats may be blocked.
"""
from __future__ import annotations

import math
from typing import NamedTuple
import numpy as np




def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 3e-15) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        del_h = d * c
        h *= del_h
        if abs(del_h - 1.0) < eps:
            break
    return h


def ibeta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    else:
        return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_critical(df: int, p: float = 0.975) -> float:
    """Two-tailed Student's t critical value such that P(|T| <= t) = 2*p - 1.
    
    For p=0.975, this corresponds to the 95% two-sided confidence critical multiplier.
    """
    if df < 1:
        return 1.95996
    target = 2.0 * (1.0 - p)
    low, high = 0.0, 100.0
    for _ in range(60):
        mid = (low + high) / 2.0
        val = ibeta(df / 2.0, 0.5, df / (df + mid * mid))
        if val > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def sem(arr) -> float:
    """Standard error of the mean."""
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n < 2:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(n))


def ci95(arr) -> list[float]:
    """Computes 95% two-sided Student's t confidence interval [lower, upper]."""
    arr = np.asarray(arr, dtype=float)
    m = float(np.mean(arr))
    n = len(arr)
    if n < 2:
        return [m, m]
    s = sem(arr)
    t_crit = t_critical(n - 1, 0.975)
    h = float(t_crit * s)
    return [m - h, m + h]


class TTestResult(NamedTuple):
    statistic: float
    pvalue: float


def ttest_rel(a, b) -> TTestResult:
    """Paired two-sample Student's t-test."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    diff = a_arr - b_arr
    n = len(diff)
    if n < 2:
        return TTestResult(0.0, 1.0)
    m = float(np.mean(diff))
    s = float(np.std(diff, ddof=1))
    if s <= 1e-15:
        return TTestResult(0.0, 1.0 if abs(m) < 1e-15 else 0.0)
    se = s / math.sqrt(n)
    t = m / se
    df = n - 1
    x = df / (df + t * t)
    p = float(ibeta(df / 2.0, 0.5, x))
    return TTestResult(t, p)


class WilcoxonResult(NamedTuple):
    statistic: float
    pvalue: float


def wilcoxon(diff, alternative: str = "two-sided") -> WilcoxonResult:
    """Wilcoxon signed-rank test for paired samples / differences."""
    diff_arr = np.asarray(diff, dtype=float)
    diff_arr = diff_arr[diff_arr != 0]
    n = len(diff_arr)
    if n == 0:
        return WilcoxonResult(0.0, 1.0)
    
    abs_d = np.abs(diff_arr)
    order = np.argsort(abs_d)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and abs_d[order[j]] == abs_d[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    w_pos = float(np.sum(ranks[diff_arr > 0]))
    w_neg = float(np.sum(ranks[diff_arr < 0]))
    w = min(w_pos, w_neg)
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w <= 0:
        return WilcoxonResult(w, 1.0)
    std_w = math.sqrt(var_w)
    z = (w - mean_w + 0.5) / std_w
    p = float(math.erfc(abs(z) / math.sqrt(2.0)))
    return WilcoxonResult(w, p)
