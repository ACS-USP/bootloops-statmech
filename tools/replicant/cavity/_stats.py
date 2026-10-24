"""Small statistics helpers for population dynamics (cavity/sparse.py)."""
import math

import mpmath as mp
import numpy as np


def t_quantile(prob, df):
    """Student-t quantile t_{prob, df} (mpmath: regularized incomplete beta, bracketed root)."""
    with mp.workdps(20):
        cdf = lambda t: 1 - mp.betainc(df / 2, mp.mpf(1) / 2, 0, df / (df + t * t),
                                       regularized=True) / 2
        hi = mp.mpf(1)
        while cdf(hi) < prob:
            hi *= 2
        return float(mp.findroot(lambda t: cdf(t) - prob, (hi / 2 if hi > 1 else 0, hi),
                                 solver="illinois"))


def drift(q_meas, q_traj, block):
    """Equilibration diagnostic from the dynamics only (no beta_c input).

    q_meas[r]: measured q per sweep of run r (measured window).
    q_traj[r]: the cavity population's own E tanh^2(beta h) at every sweep of run r
               (burn-in + measured), a cheap consistent trace.
    Two criteria; drifting if either fires:
      t-test   mean over runs of the relative change of q_meas between the halves of the
               measured window, two-sided Student-t at 99% with df = runs - 1;
      sign     the last three blocks of `block` sweeps of q_traj are monotone in the SAME
               direction in every run (under stationarity with independent blocks this has
               probability 2 / 6^runs: 0.15% at 4 runs) -- powerful with few runs.
    Returns (drifting, text)."""
    runs = len(q_meas)
    rel = []
    for qs in q_meas:
        a, c = np.mean(qs[:len(qs) // 2]), np.mean(qs[len(qs) - len(qs) // 2:])
        scale = (abs(a) + abs(c)) / 2
        rel.append((c - a) / scale if scale > 0 else 0.0)
    rel = np.array(rel)
    m, se = float(rel.mean()), float(rel.std(ddof=1) / math.sqrt(runs))
    tq = t_quantile(0.995, runs - 1)
    t_flag = abs(m) > tq * se if se > 0 else m != 0
    dirs = []
    for tr in q_traj:
        A, B, C = (np.mean(tr[len(tr) - (3 - i) * block:len(tr) - (2 - i) * block])
                   for i in range(3))
        dirs.append(-1 if A > B > C else (1 if A < B < C else 0))
    s_flag = len(set(dirs)) == 1 and dirs[0] != 0
    text = (f"relative q change {m:+.3g} between halves of the measured sweeps (99% "
            f"Student-t band {tq * se:.2g}, t = {tq:.3g}, {runs} runs); last three "
            f"{block}-sweep blocks monotone in {sum(d != 0 for d in dirs)}/{runs} runs"
            + (" all the same way" if s_flag else ""))
    return bool(t_flag or s_flag), text
