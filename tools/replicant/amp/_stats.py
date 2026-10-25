"""Student-t bands for finite-N runs over named seeds (amp/)."""
import math
from dataclasses import dataclass

import mpmath as mp

from ..core import Refusal


@dataclass(frozen=True)
class Band:
    """mean +- half_width: a two-sided Student-t band at confidence `conf`, df = n - 1."""
    mean: float
    stderr: float
    half_width: float
    df: int
    conf: float
    t: float


def t_quantile(prob, df):
    """Student-t quantile t_{prob, df} (regularized incomplete beta, bracketed root)."""
    with mp.workdps(20):
        cdf = lambda t: 1 - mp.betainc(mp.mpf(df) / 2, mp.mpf(1) / 2, 0, df / (df + t * t),
                                       regularized=True) / 2
        hi = mp.mpf(1)
        while cdf(hi) < prob:
            hi *= 2
        return float(mp.findroot(lambda t: cdf(t) - prob, (hi / 2 if hi > 1 else 0, hi),
                                 solver="illinois"))


def student_band(values, conf=0.99):
    """Band of the mean of independent runs: half-width t_{(1+conf)/2, n-1} * stderr."""
    xs = [float(v) for v in values]
    n = len(xs)
    if n < 2:
        raise Refusal("precondition:two-samples",
                      f"a Student-t band needs at least two independent runs, got {n}")
    mean = math.fsum(xs) / n
    sd = math.sqrt(math.fsum((x - mean)**2 for x in xs) / (n - 1))
    se = sd / math.sqrt(n)
    t = t_quantile((1 + conf) / 2, n - 1)
    return Band(mean, se, t * se, n - 1, conf, t)
