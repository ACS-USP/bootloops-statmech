"""E4: the Gardner transition, i.e. the stability of a 1RSB saddle toward further RSB.

  onersb_replicon(family, **params) -> mpf        innermost-block 1RSB replicon
  onersb_stability(family, **params) -> OneRSBStability (both block levels, the point)
  pspin_gardner_temperature(p, variables="spherical"|"ising") -> mpf (or a Refusal)
  replicon_formula(variables, p, beta, q, m)      the cited formula (registered rules)
  explicit_n_replicon(variables, p, beta, q, n, m) the independent integer-n rule
  ising_perceptron_1rsb_replicon(alpha)           a documented refusal

Families: "pspin" (spherical pure p-spin, families/pspin.py, its static 1RSB) and
"ising_pspin" (families/ising_pspin.py; static m, or a given m).  Couplings of variance
p!/(2 N^(p-1)), zero field, q0 = 0.

Formula (implemented as cited, not derived by the level-(a) machinery: neither p-spin is a
level-(a) family).  Replicon of the innermost 1RSB block, made dimensionless:
    Lambda_R = 1 - Sigma(q1) chi_R,   Sigma(q) = beta^2 p (p-1) q^(p-2) / 2,
chi_R = (1 - q1)^2 for spherical spins (Crisanti & Leuzzi, PRB 73, 014412 (2006),
arXiv:cond-mat/0505748, Eq. (31): Lambda_1 = -Sigma(q1) + 1/(1 - q1)^2, times (1 - q1)^2),
chi_R = E_m[sech^4(a z)] for Ising spins (Montanari & Ricci-Tersenghi, EPJB 33, 339 (2003),
arXiv:cond-mat/0301591, Eq. (2.6); Yeo & Moore, PRE 101, 032127 (2020), Eq. (17);
Gardner 1985 not fetched).  1RSB is stable toward a further step where Lambda_R > 0.  The
outer-block replicon is 1 - Sigma(q0) chi_0 with Sigma(0) = 0 for p >= 3 (CL Eq. (32)):
exactly 1 at q0 = 0.  The susceptibility rules sit in the registry CHI_R (a planted
error there is a test).

Independent rule (explicit_n_replicon).  At integer n with blocks of integer size m:
spherical, F_n(Q) = beta^2/4 sum_{a != b} Q_ab^p + (1/2) ln det Q; Ising, the Legendre form
H(Lambda) = sum_{a<b} [g(Q(Lambda_ab)) - Lambda_ab Q(Lambda_ab)] + ln sum_s exp(sum_{a<b}
Lambda_ab s_a s_b), g(Q) = beta^2 Q^p / 2, with the spin sum enumerated exactly (2^n).  Along
a replicon vector v of one block (zero row sums), the second derivatives of the coupling
part and of the entropic part are taken by mpmath finite differences, giving Sigma and
chi_R numerically, hence Lambda_R.  At q0 = 0 the blocks decouple, so the integer-m value is
the cosh^m-weighted formula at that m exactly; the formula's use at the solution's
non-integer m is the usual analytic continuation in m (not checked by this rule).

What is computed: the innermost-block replicon and the outer one (trivially 1) of the
1RSB saddle; for the spherical model also (families/pspin.py) the Hessian of f in
(q0, q, m) (a local maximum).  NOT computed: the longitudinal / anomalous eigenvalues of
the full replica Hessian, and, for the Ising p-spin, the Hessian in (q, m).  So `stable`
means "replicon positive", local stability in the replicon sector only.  Physics-level.
"""
from dataclasses import dataclass
import itertools

import mpmath as mp
import numpy as np
from scipy.optimize import brentq

from ..core import Refusal
from . import ising_pspin as I
from . import pspin as P

_DPS = 30
GARDNER_CITE = ("Crisanti & Leuzzi, PRB 73, 014412 (2006), Eq. (31) (spherical); "
                "Montanari & Ricci-Tersenghi, EPJB 33, 339 (2003), Eq. (2.6) (Ising)")


def ising_block_average(beta, p, q, m, g):
    """E_m[g(tanh(a z))] (cosh^m-weighted Gaussian average, a^2 = beta^2 p q^(p-1)/2)."""
    b, q = float(mp.mpf(beta)), float(mp.mpf(q))
    return mp.mpf(I.block_average(I.field(p, b, q), float(mp.mpf(m)),
                                  lambda t, x: g(t))[1])


CHI_R = {
    "spherical": lambda p, beta, q, m: (1 - mp.mpf(q))**2,
    "ising": lambda p, beta, q, m: ising_block_average(beta, p, q, m,
                                                       lambda t: (1 - t * t)**2),
}


def sigma(p, beta, q):
    """Sigma(q) = d^2/dq^2 of beta^2 q^p / 2 (the coupling part per replica pair)."""
    return mp.mpf(beta)**2 * p * (p - 1) * mp.mpf(q)**(p - 2) / 2


def replicon_formula(variables, p, beta, q, m):
    with mp.workdps(_DPS):
        if variables not in CHI_R:
            raise Refusal("precondition:known-variables",
                          f"variables = {variables!r}: rules exist for {sorted(CHI_R)}")
        return 1 - sigma(p, beta, q) * CHI_R[variables](p, beta, q, m)


def _replicon_vector(n, m):
    """A replicon vector inside the first block (needs m >= 4): v01 = v23 = 1,
    v02 = v13 = -1, so every row sum within the block vanishes."""
    if m < 4 or n % m:
        raise Refusal("precondition:replicon-block", f"n = {n}, m = {m}: need m >= 4, m | n")
    return {(0, 1): 1, (2, 3): 1, (0, 2): -1, (1, 3): -1}


def explicit_n_replicon(variables, p, beta, q, n, m):
    """Lambda_R from the explicit integer-n replica functional (module docstring)."""
    with mp.workdps(40):
        b, q = mp.mpf(beta), mp.mpf(q)
        v = _replicon_vector(n, m)
        norm = sum(x * x for x in v.values())
        pairs = [(a, c) for a in range(n) for c in range(a + 1, n)]
        inb = lambda a, c: a // m == c // m
        d2 = lambda F: mp.diff(F, 0, 2) / norm
        if variables == "spherical":
            def mat(t):
                M = mp.eye(n)
                for a, c in pairs:
                    x = (q if inb(a, c) else 0) + t * v.get((a, c), 0)
                    M[a, c] = M[c, a] = x
                return M
            coup = lambda t: b**2 / 2 * sum(mat(t)[a, c]**p for a, c in pairs)
            ent = lambda t: mp.log(mp.det(mat(t))) / 2
            sig, chi = d2(coup), -1 / d2(ent)
        elif variables == "ising":
            lam = b**2 * p * q**(p - 1) / 2
            Q = lambda L: (2 * L / (b**2 * p))**(mp.mpf(1) / (p - 1))
            Lam = lambda t: {(a, c): lam + t * v.get((a, c), 0) for a, c in pairs if inb(a, c)}
            coup = lambda t: sum(b**2 * Q(L)**p / 2 - L * Q(L) for L in Lam(t).values())
            confs = list(itertools.product((-1, 1), repeat=n))

            def ent(t):
                L = Lam(t)
                return mp.log(mp.fsum(mp.exp(mp.fsum(x * s[a] * s[c] for (a, c), x in L.items()))
                                      for s in confs))
            sig, chi = -1 / d2(coup), d2(ent)
        else:
            raise Refusal("precondition:known-variables", f"variables = {variables!r}")
        return 1 - sig * chi


@dataclass(frozen=True)
class OneRSBStability:
    family: str
    p: int
    beta: object
    q1: object
    m: object
    replicon: object          # innermost block
    replicon_outer: object    # between blocks (q0 = 0: exactly 1)
    note: str


_PARAMS = {"pspin": {"p", "beta"}, "ising_pspin": {"p", "beta", "m"}}


def onersb_stability(family, **params):
    if family not in _PARAMS:
        raise Refusal(f"missing-step:family:{family}", "1RSB replicon rules exist for "
                      f"{', '.join(sorted(_PARAMS))}")
    bad = set(params) - _PARAMS[family]
    if bad:
        raise Refusal("precondition:known-parameter", f"{family}: unknown {sorted(bad)}; "
                      f"accepted: {', '.join(sorted(_PARAMS[family]))}")
    if params.get("beta") is None:
        raise Refusal("precondition:state-point", f"{family}: the 1RSB replicon needs beta")
    p, beta = params.get("p", 3), params["beta"]
    with mp.workdps(max(_DPS, mp.mp.dps)):
        if family == "pspin":
            s, v = P.pspin_spherical_1rsb(p, beta), "spherical"
            how = "static 1RSB (families/pspin.py)"
        else:
            s, v = I.ising_pspin_1rsb(p, beta, params.get("m")), "ising"
            how = s.note
        lam = replicon_formula(v, p, beta, s.q1, s.m)
        # outer block: 1 - Sigma(q0) chi_0, and Sigma(0) = 0 for p >= 3 (q0 = 0 here)
        out = mp.mpf(1) if sigma(p, beta, s.q0) == 0 else None
        return OneRSBStability(family, p, mp.mpf(beta), s.q1, s.m, lam, out,
                               f"{how}; innermost replicon {mp.nstr(lam, 8)} ({GARDNER_CITE});"
                               " outer replicon 1 (Sigma(q0 = 0) = 0); longitudinal modes "
                               "not computed")


def onersb_replicon(family, **params):
    return onersb_stability(family, **params).replicon


def _static_replicon(p, T, seed=None):
    """(replicon, (q, m)) on the static line at T: Newton from `seed` when given (checked
    against the full solve at the end of a search), else the full solve."""
    b = 1 / float(T)
    pt = I.static_point(p, b, seed) if seed is not None else None
    if pt is None:
        s = I.ising_pspin_1rsb(p, b)
        pt = (float(s.q1), float(s.m))
    return float(replicon_formula("ising", p, b, pt[0], pt[1])), pt


_TG_CACHE = {}       # (p, rule object) -> T_G; the key keeps the rule alive (no id reuse)


def _ising_tg(p, rule):
    if (p, rule) not in _TG_CACHE:
        _TG_CACHE[(p, rule)] = _ising_tg_search(p)
    return _TG_CACHE[(p, rule)]


def _ising_tg_search(p):
    ts = float(I.ising_pspin_critical_temperature(p))
    grid = np.linspace(0.98 * ts, 0.15 * ts, 18)
    prev, seed = None, None
    for T in grid:
        v, seed = _static_replicon(p, T, seed)
        if prev is not None and (prev[1] > 0) != (v > 0):
            cell = {"seed": seed}

            def g(x):
                lam, cell["seed"] = _static_replicon(p, x, cell["seed"])
                return lam
            tg = brentq(g, T, prev[0], xtol=1e-12)
            full = _static_replicon(p, tg)[1]          # the full solve agrees with Newton
            if max(abs(a - c) for a, c in zip(full, cell["seed"])) > 1e-9:
                raise Refusal("no-solution:solver", f"T_G search: Newton {cell['seed']} and "
                              f"the full static solve {full} disagree at T = {tg}")
            return tg
        prev = (T, v)
    raise Refusal("no-transition:gardner", f"Ising p = {p}: the static 1RSB replicon does not "
                  f"change sign on the scan T in [{grid[-1]:.4g}, {grid[0]:.4g}] (18 points)")


_MARGIN = mp.mpf("1e-6")
_SPH_SCAN = ("0.02", "0.25", "0.5", "0.75", "0.99")     # fractions of T_s


def pspin_gardner_temperature(p, variables="spherical"):
    """T_G: the first zero below T_s of the static-line 1RSB replicon.  Ising: always
    searched.  Spherical: the answer follows its own scan (fractions 0.02..0.99 of T_s):
    every value > 1e-6 -> refused no-transition:gardner (the pure model's case); a sign
    change -> T_G by root-finding, audited by the signs at T_G +- 1e-6 T_s; otherwise
    (negative or marginal everywhere) refused unstable:1rsb.  Two sign changes between
    adjacent scan points would be missed (not searched)."""
    if variables == "ising":
        return mp.mpf(_ising_tg(p, CHI_R["ising"]))
    if variables != "spherical":
        raise Refusal("precondition:known-variables",
                      f"variables = {variables!r}: 'spherical' or 'ising'")
    with mp.workdps(_DPS):
        ts = P.pspin_critical_temperature(p)
        lam = lambda T: onersb_replicon("pspin", p=p, beta=1 / T)
        scan = [(mp.mpf(f) * ts, lam(mp.mpf(f) * ts)) for f in _SPH_SCAN]
        shown = ", ".join(f"T = {mp.nstr(T, 4)}: {mp.nstr(v, 8)}" for T, v in scan)
        if all(v > _MARGIN for _, v in scan):
            raise Refusal("no-transition:gardner",
                          f"no Gardner transition for the pure spherical p-spin (p = {p}); 1RSB "
                          f"replicon positive (> {mp.nstr(_MARGIN, 2)}) at every scanned T < T_s "
                          f"= {mp.nstr(ts, 8)}: computed scan {shown} (on the static line the "
                          "formula gives 1 - (p-1) y, the same for every T; Montanari & "
                          "Ricci-Tersenghi 2003, Sec. 1, state it for this model; Crisanti & "
                          "Leuzzi 2006, App. C, for the s+p model with s > 2, of which the pure "
                          "model is the mu_s -> 0 edge)")
        hot_first = scan[::-1]
        for (t_hi, v_hi), (t_lo, v_lo) in zip(hot_first, hot_first[1:]):
            if (v_hi > 0) != (v_lo > 0):
                tg = mp.findroot(lam, (t_lo, t_hi), solver="anderson")
                d = _MARGIN * ts
                if not (lam(tg + d) > 0) != (lam(tg - d) > 0) or not t_lo <= tg <= t_hi:
                    raise Refusal("no-solution:audit", f"spherical T_G candidate {tg}: the "
                                  "replicon does not change sign across it")
                return tg
        raise Refusal("unstable:1rsb", f"spherical p-spin (p = {p}): the 1RSB replicon is "
                      f"negative or marginal (<= {mp.nstr(_MARGIN, 2)}) at every scanned T < T_s "
                      f"without a sign change: {shown}")


def ising_perceptron_1rsb_replicon(alpha):
    """Not computed: refused with the reason (GUIDE, KNOWN HAZARDS)."""
    from ..refdata import load
    cap = mp.mpf(load("perceptron")["ising_1rsb_capacity"].value)
    a = mp.mpf(alpha)
    if not 0 < a <= cap:
        raise Refusal("precondition:alpha-range", f"alpha = {alpha}: the question is posed "
                      f"for 0 < alpha <= {mp.nstr(cap, 7)} (the frozen 1RSB capacity)")
    raise Refusal("not-implemented:frozen-1rsb-replicon",
                  f"alpha = {alpha}: the Ising-perceptron 1RSB that sets the capacity is FROZEN "
                  "(q1 = 1, Krauth-Mezard); its innermost-block replicon is a 0 x infinity limit "
                  "there (the site factor E_m sech^4 -> 0 while the Gaussian-integral energy "
                  "susceptibility diverges as q1 -> 1) that this package does not evaluate, and "
                  "the level-(a) 1RSB search finds only rs-degenerate full (q1 < 1) roots at "
                  "alpha = 0.8; the stability of the frozen 1RSB toward further RSB is not "
                  "computed.  (Below the RS entropy zero the RS saddle itself is the computed-"
                  "stable one: solve_derived_rs.)")
