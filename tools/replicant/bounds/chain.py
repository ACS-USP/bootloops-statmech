"""The bound chain on the package's own numbers (E10): a consistency check of its optimisers.

What is proved and what is checked.  Guerra (Comm. Math. Phys. 233, 1 (2003), arXiv:cond-mat/0205123,
Theorem 3): (1/N) E ln Z_N <= P_K(q, m) for every N, K and admissible (q, m), so every certified
P_K(params) below is a certified upper bound on the limit pressure, whatever the parameters.  The
infima over the K-step classes are ordered (a (K-1)-step order parameter is a K-step one with a
repeated level), and Talagrand (Ann. Math. 163, 221 (2006)) proves the K -> oo infimum is the limit.
The chain P_K(params_K) <= P_{K-1}(params_{K-1}) <= ... <= P_RS(q_RS) at the package's OWN optimiser
outputs is not a theorem: it holds when each solver reached its class infimum.  A violation is a
package defect.  So check_bound_chain reports
  per level   the certified bound at the package's parameters, and whether the package's reported
              pressure equals it within radius + the solver's stated tolerance ("functional": ok |
              mismatch; "not-computed" for levels not certified); it catches only functional
              errors larger than the solver's stated accuracy, smaller ones pass by design;
  per link    lower <= upper decided with enclosures: "ok" (certainly below), "violation"
              (certainly above), "undecided" (the enclosures overlap).
Inputs (package_values) are package outputs, not banked targets: families/sk.py sk_rs (q to ~1e-12,
dps 15; tol 1e-12), sk_1rsb (float64 Gauss-Hermite 80 x 80, validated against 160 x 160 to ~1e-11 in
f for beta <= 2 and ~2e-10 for beta <= 3, so tol beta * that), families/parisi.py's finite-T K-step
staircases (h = 0, 1 < beta <= 3, K <= 5; tol beta * (grid change + max |gradient|), the package's
own per-level error terms).  Levels up to certified_max are certified (nested acb.integral; K = 2 at
12 digits takes 37 s); higher levels enter with reported +- tol and are labelled numerical.

  package_values(beta, h=0, K=1) -> (PackageValue, ...)        RS, 1RSB, ..., K-RSB
  check_bound_chain(beta, h=0, K=1, *, digits=20, certified_max=2, package=None) -> ChainReport
      .ok, .violations ("ordering:<lower>><upper>", "functional:<name>"), .undecided, .levels, .links
"""
from dataclasses import dataclass, replace

import mpmath as mp

from ..core import Refusal
from ..lattice.ball import Ball, baller, rational
from .guerra import guerra_rs_bound
from .krsb import check_digits, exact, guerra_krsb_bound

_K_MAX = 5


def level_name(k):
    return "rs" if k == 0 else f"{k}rsb"


@dataclass(frozen=True)
class PackageValue:
    level: int              # K (0 = RS)
    qs: tuple               # (q_0 .. q_K)
    ms: tuple               # (m_1 .. m_K)
    pressure: object        # the package's reported -beta f (mpf)
    tol: object             # the solver's stated accuracy of `pressure` (mpf)
    source: str

    def replace(self, **kw):
        return replace(self, **kw)


@dataclass(frozen=True)
class LevelCheck:
    level: int
    name: str
    bound: Ball | None      # certified P_K(params) (None: not certified)
    certified: bool
    reported: object
    tol: object
    functional: str         # "ok" | "mismatch" | "not-computed"
    detail: str = ""


@dataclass(frozen=True)
class Link:
    upper: str
    lower: str
    verdict: str            # "ok" | "violation" | "undecided"
    numerical: bool         # True when either side is not certified (reported +- tol)
    detail: str = ""


@dataclass(frozen=True)
class ChainReport:
    beta: object
    h: object
    levels: tuple
    links: tuple
    violations: tuple
    undecided: tuple
    notes: tuple = ()

    @property
    def ok(self):
        return not self.violations and not self.undecided


def _check_K(K):
    if isinstance(K, bool) or not isinstance(K, int) or not 1 <= K <= _K_MAX:
        raise Refusal("precondition:K", f"K = {K!r}: need an int 1 .. {_K_MAX} (families/parisi.py "
                      f"validates finite-T staircases up to {_K_MAX} steps)")


def package_values(beta, h=0, K=1):
    """The package's RS, 1RSB and K-step optimiser outputs at (beta, h) (see module docstring)."""
    from ..families.sk import sk_1rsb, sk_rs
    _check_K(K)
    b, hq = rational(beta, "beta"), exact(h, "h")
    if K >= 2 and hq != 0:
        raise Refusal("unsupported:field", f"h = {h}: families/parisi.py's finite-T K-step "
                      "staircases are for h = 0 only")
    bm = mp.mpf(b.numerator) / b.denominator
    rs = sk_rs(str(bm), str(mp.mpf(hq.numerator) / hq.denominator))
    out = [PackageValue(0, (rs.q,), (), -bm * rs.f, mp.mpf("1e-12"), "families/sk.py sk_rs")]
    try:
        s = sk_1rsb(float(b), float(hq))
    except Refusal as r:
        if r.reason == "no-solution" and rs.stable:
            return tuple(out)                 # RS stable: 1RSB collapses to RS (sk_1rsb refuses)
        raise
    tol1 = bm * mp.mpf("1e-11" if b <= 2 else "2e-10")
    out.append(PackageValue(1, (s.q0, s.q1), (s.m,), -bm * s.f, tol1, "families/sk.py sk_1rsb"))
    if K >= 2:
        if not 1 < b <= 3:
            raise Refusal("precondition:temperature-range", f"beta = {beta}: families/parisi.py's "
                          "finite-T staircases are validated for 1 < beta <= 3")
        from ..families.parisi import _frsb_level
        for k in range(2, K + 1):
            v, f, gmax, _eig, grid, _e = _frsb_level(float(b), k)
            out.append(PackageValue(k, tuple(float(x) for x in v[:k + 1]),
                                    tuple(float(x) for x in v[k + 1:]), -bm * mp.mpf(f),
                                    bm * mp.mpf(grid + gmax), f"families/parisi.py level K = {k}"))
    return tuple(out)


def _arb_of(x, tol=0):
    """An arb enclosing [x - tol, x + tol] (x, tol mpf)."""
    from flint import arb
    with mp.workdps(40), baller().hygiene.ctx_guard(prec=200):
        return arb(mp.nstr(x, 35)) + arb(0, mp.nstr(abs(tol) + abs(x) * mp.mpf("1e-33"), 5))


def _level(beta, h, pv, digits, certified_max):
    name = level_name(pv.level)
    if pv.level <= certified_max:
        if pv.level == 0:
            bound = guerra_rs_bound(beta, h, pv.qs[0])
        else:
            bound = guerra_krsb_bound(beta, pv.qs, pv.ms, h=h, digits=digits)
        with mp.workdps(60):
            gap = abs(mp.mpf(pv.pressure) - bound.mid())
            ok = gap <= bound.rad() + mp.mpf(pv.tol)
        detail = (f"|reported - certified| = {mp.nstr(gap, 3)} vs radius {mp.nstr(bound.rad(), 3)} + "
                  f"tol {mp.nstr(pv.tol, 3)}")
        return LevelCheck(pv.level, name, bound, True, pv.pressure, pv.tol,
                          "ok" if ok else "mismatch", detail)
    return LevelCheck(pv.level, name, None, False, pv.pressure, pv.tol, "not-computed",
                      f"not certified (level > certified_max = {certified_max}): reported +- tol")


def _enclosure(lv):
    return lv.bound.arb if lv.certified else _arb_of(lv.reported, lv.tol)


def check_bound_chain(beta, h=0, K=1, *, digits=20, certified_max=2, package=None) -> ChainReport:
    _check_K(K)
    check_digits(digits)
    hq = exact(h, "h")
    if K >= 2 and hq != 0:
        raise Refusal("unsupported:field", f"h = {h}: the K >= 2 staircases are for h = 0 only")
    pkg = tuple(package) if package is not None else package_values(beta, h, K)
    notes = ()
    if len(pkg) == 1:
        notes = ("RS is stable here: sk_1rsb collapses to RS, the chain has the RS level only",)
    levels = tuple(_level(beta, h, pv, digits, certified_max) for pv in pkg)
    links, violations, undecided = [], [], []
    for lv in levels:
        if lv.functional == "mismatch":
            violations.append(f"functional:{lv.name}")
    for up, lo in zip(levels, levels[1:]):
        A, B = _enclosure(lo), _enclosure(up)
        if A < B:
            verdict = "ok"
        elif A > B:
            verdict = "violation"
            violations.append(f"ordering:{lo.name}>{up.name}")
        else:
            verdict = "undecided"
            undecided.append(f"{lo.name}~{up.name}")
        links.append(Link(up.name, lo.name, verdict, not (up.certified and lo.certified),
                          f"P_{lo.name} {A.str(12)} vs P_{up.name} {B.str(12)}"))
    return ChainReport(beta, h, levels, tuple(links), tuple(violations), tuple(undecided), notes)


__all__ = ["ChainReport", "LevelCheck", "Link", "PackageValue", "check_bound_chain", "level_name",
           "package_values"]
