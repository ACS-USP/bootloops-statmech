"""Catalog entries for the solvable lattice models (E9), certified in ball arithmetic (lattice/):
`ising2d` and `ising1d` (route `exact`), `bethe_ising` (route `cavity`: the BP fixed points).

Every physical value is a python-flint arb ball through tools/baller (lattice/ball.py); the entry
reports its midpoint (an mpf at 40 digits) and in `errors` an upper bound of |value - x| over the
ball (radius plus the rounding of the midpoint).  The routes are declared rigor "certified"
(CatalogEntry.rigor); compose refuses a certified run with a value lacking its radius.
Run parameters:
  ising2d (exact)     beta (rational, or "beta_c", the default), L (torus side, default 4; 2 .. 6),
                      Ly (default L)
  ising1d (exact)     beta (rational > 0, default 1), h (rational, default 0), N (ring, default 8)
  bethe_ising (cavity) k (coordination >= 2, default 3), beta (default 1), h (default 0), branch
"""
import mpmath as mp

from .catalog import Out, RefCheck, Stability, entry, fstep
from .core import Refusal
from .lattice import (BETA_C, bethe_critical_beta, bethe_ising_free_energy, critical_beta,
                      ising1d_free_energy, kaufman_free_energy, kaufman_polynomial,
                      ising_transfer_matrix_free_energy, onsager_free_energy,
                      torus_partition_polynomial, yang_magnetization)
from .lattice.onsager import _ordered, yang_loglog_slope
from .lattice.coupling import coupling
from .lattice.transfer import ising1d_ring_balls
from .spec import ModelSpec

DC = "Duminil-Copin, arXiv:2208.00864 (2022)"
ONSAGER = f"Onsager, Phys. Rev. 65, 117 (1944) (not fetched), as printed by {DC}, Eq. (3.6)"
YANG = f"Yang, Phys. Rev. 85, 808 (1952) (not fetched), as printed by {DC}, Eq. (3.7)"
KW = f"Kramers & Wannier, Phys. Rev. 60, 252 (1941) (not fetched), via {DC}, Sec. 3.1"
OEIS = "OEIS A242743 ((1/2) ln 2 + 2G/pi; Onsager (1944) Eq. (121) via OEIS)"
KAUF = ("Kaufman, Phys. Rev. 76, 1232 (1949) (not fetched), as printed by Izmailian & Hu, Phys. Rev. "
        "E 65 (2002), arXiv:cond-mat/0009024, Eqs. (10)-(16), and Kastner, arXiv:cond-mat/0605568, "
        "Eqs. (6)-(8)")
ONE_D = ("Wang, Diaz-Mendez & Capdevila, arXiv:1907.11701, Eqs. (1), (4)-(12) (read; journal "
         "reference not checked; their derivation is by induction on N, which they show equals the "
         "transfer-matrix trace Z_N = Tr P^N, Eq. (7)); Baxter, Exactly Solved Models (1982), Ch. 2 "
         "(not fetched)")
TM = "Duminil-Copin, arXiv:2208.00864 (2022), Frame 2 (Z of the torus = Tr V^N)"
DM = "Dembo & Montanari, Ann. Appl. Probab. 20, 565 (2010), arXiv:0804.4726"
ARB = "python-flint arb through tools/baller (baller.solve / baller.run; acb.integral)"
CAVEAT_2D = ("exact route, certified: Onsager's free energy, Yang's magnetization and Kaufman's "
             "finite-torus formula are the cited theorems, evaluated in ball arithmetic (" + ARB + "); "
             "radii in errors.  Kaufman's integer polynomial (recovered from balls, radius < 1/4) "
             "equals the exact transfer matrix's; the log-log slope of m* is a consistency check of "
             "the implemented formula, not a derivation of the exponent")
CAVEAT_1D = ("exact route, certified: -beta f = ln lambda_+ (cited closed form) in ball arithmetic; "
             "the ring identity Tr T^N = lambda_+^N + lambda_-^N holds within the balls at the run's "
             "parameters (and exactly in rationals in tests/test_lattice.py)")
CAVEAT_BETHE = ("cavity route, certified: every BP fixed point of the Bethe-lattice ferromagnet, "
                "enclosed by interval Newton in arb (a complete list: monotone intervals of the "
                "cavity equation); stability is the computed BP Jacobian (k - 1) xi'(x) < 1; the "
                "Bethe free energy is the random k-regular graph's by Dembo & Montanari Thm 2.4 "
                "(B != 0, and B -> 0)")
_BRANCHES = ("unique", "lower", "middle", "upper")
_OBS = ("cavity_field", "magnetization", "phi", "jacobian")


def _put(**balls):
    vals, errs = {}, {}
    for k, b in balls.items():
        vals[k], errs[k] = b.value_and_error(40)
    return vals, errs


def _int(p, name, default, lo, hi):
    v = p.get(name, default)
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise Refusal(f"precondition:{name}", f"{name} = {v!r}: need an int in {lo} .. {hi}")
    return v


# ---- ising2d -----------------------------------------------------------------------------------
def _beta2(p):
    return p.get("beta", BETA_C)


def _above(p):
    c = coupling(_beta2(p))
    return c.kind != "critical" and _ordered(c.arb)


def _crit(p):
    v, e = _put(beta_c=critical_beta())
    return Out({**v, "beta": str(_beta2(p))}, "beta_c = (1/2) ln(1 + sqrt 2), certified", errors=e)


def _onsager(p):
    v, e = _put(minus_beta_f=onsager_free_energy(_beta2(p)))
    how = ("closed form (1/2) ln 2 + 2G/pi" if _beta2(p) == BETA_C
           else "acb.integral of the single integral")
    return Out(v, f"-beta f by {how}, certified", errors=e)


def _yang(p):
    v, e = _put(magnetization=yang_magnetization(_beta2(p)))
    return Out(v, "m* = (1 - sinh(2 beta)^-4)^(1/8), certified", errors=e)


def _slope(p):
    v, e = _put(yang_exponent_loglog=yang_loglog_slope("1e-6", "1e-8"))
    return Out({**v, "slope_points": "beta - beta_c = 1e-6, 1e-8"},
               "d ln m* / d ln(beta - beta_c) from two certified values (consistency check)", errors=e)


def _torus(p):
    L = _int(p, "L", 4, 2, 6)
    Ly = _int(p, "Ly", L, 2, 6)
    tm, rec = torus_partition_polynomial(L, Ly), kaufman_polynomial(L, Ly)
    if rec.coefficients != tm:
        raise Refusal("no-solution:torus-disagreement", f"{L} x {Ly}: Kaufman's recovered polynomial "
                      f"differs from the transfer matrix's")
    v, e = _put(torus_minus_beta_f=kaufman_free_energy(L, Ly, _beta2(p)),
                torus_minus_beta_f_tm=ising_transfer_matrix_free_energy(Ly, _beta2(p), Lx=L))
    return Out({**v, "torus_size": f"{L} x {Ly}", "torus_coefficients": [str(c) for c in tm],
                "kaufman_max_radius": rec.max_radius, "kaufman_prec_bits": rec.prec},
               f"{L} x {Ly} torus: Kaufman's integer polynomial (radius < {mp.nstr(rec.max_radius, 2)}) "
               "equals the exact transfer matrix's", errors=e)


def ising2d():
    steps = [fstep("ising2d-critical-point", KW, _crit),
             fstep("ising2d-onsager-free-energy", f"{ONSAGER}; {OEIS}", _onsager),
             fstep("ising2d-yang-magnetization", YANG, _yang, when=_above,
                   skip="beta <= beta_c: m* = 0 (paramagnet)"),
             fstep("ising2d-yang-exponent", YANG, _slope),
             fstep("ising2d-kaufman-vs-transfer-matrix", KAUF, _torus)]
    refs = [RefCheck("lattice", "onsager_critical_minus_beta_f", lambda v: v["minus_beta_f"],
                     route="exact"),
            RefCheck("lattice", "beta_c_square", lambda v: v["beta_c"], route="exact"),
            RefCheck("lattice", "yang_exponent", lambda v: v["yang_exponent_loglog"], route="exact")]
    lit = lambda cite, note: Stability("exact-by-literature", "exact", cite, note)
    fin = {"beta_c": lit(KW, "self-dual point"),
           "minus_beta_f": lit(ONSAGER, "Onsager's free energy, certified ball"),
           "magnetization": lit(YANG, "Yang's formula, certified ball"),
           "yang_exponent_loglog": Stability("not-computed", "exact", YANG, "finite-difference "
                                             "log-log slope of the cited formula at beta - beta_c = "
                                             "1e-6, 1e-8: |slope - 1/8| < 2e-7 by the O(d) correction"),
           "torus_minus_beta_f": lit(KAUF, "Kaufman's finite-torus Z, certified ball"),
           "torus_minus_beta_f_tm": Stability("exact-by-literature", "exact", TM, "exact by "
                                              "construction: the transfer-matrix trace in exact "
                                              "integers (equal to Kaufman's polynomial)")}
    spec = ModelSpec("ising2d", "ising", "none", constraint="square lattice, exact solution")
    return entry("ising2d", spec, f"{ONSAGER}; {YANG}; {KAUF}", {"exact": steps}, refs,
                 {"exact": CAVEAT_2D}, "2D Ising model (square lattice, zero field): Onsager's free "
                 "energy, Yang's magnetization, Kaufman's torus versus the transfer matrix, certified",
                 default_route="exact", finality=fin,
                 meta=("beta", "slope_points", "torus_size", "torus_coefficients",
                       "kaufman_max_radius", "kaufman_prec_bits"),
                 parameters={"exact": ("beta", "L", "Ly")}, rigor={"exact": "certified"})


# ---- ising1d -----------------------------------------------------------------------------------
def _one_d(p):
    beta, h = p.get("beta", 1), p.get("h", 0)
    N = _int(p, "N", 8, 2, 1000)
    tr, cl = ising1d_ring_balls(N, beta, h)
    if not tr.overlaps(cl):
        raise Refusal("no-solution:ring-identity", f"Tr T^{N} {tr} and lambda_+^N + lambda_-^N {cl} "
                      "do not overlap")
    v, e = _put(minus_beta_f=ising1d_free_energy(beta, h), ring_Z=tr)
    return Out({**v, "ring_N": N}, f"-beta f = ln lambda_+; ring N = {N}: Tr T^N and lambda_+^N + "
               "lambda_-^N overlap", errors=e)


def ising1d():
    steps = [fstep("ising1d-transfer-matrix", ONE_D, _one_d)]
    refs = [RefCheck("lattice", "ising1d_minus_beta_f_beta1_h0", lambda v: v["minus_beta_f"],
                     route="exact")]
    fin = {"minus_beta_f": Stability("exact-by-literature", "exact", ONE_D, "ln lambda_+, certified"),
           "ring_Z": Stability("exact-by-literature", "exact", ONE_D, "Tr T^N of the N-spin ring")}
    spec = ModelSpec("ising1d", "ising", "none", constraint="ring, nearest neighbour, field")
    return entry("ising1d", spec, ONE_D, {"exact": steps}, refs, {"exact": CAVEAT_1D},
                 "1D Ising model in a field: transfer-matrix free energy, certified",
                 default_route="exact", finality=fin, meta=("ring_N",),
                 parameters={"exact": ("beta", "h", "N")}, rigor={"exact": "certified"})


# ---- bethe_ising -------------------------------------------------------------------------------
def _bk(p):
    return p.get("k", 3)


def _bcrit(p):
    v, e = _put(beta_c=bethe_critical_beta(_bk(p)))
    return Out(v, "(k - 1) tanh beta_c = 1, certified", errors=e)


def _bethe(p):
    res = bethe_ising_free_energy(_bk(p), p.get("beta", 1), p.get("h", 0), branch=p.get("branch"))
    balls, meta = {}, {"k": _bk(p), "fixed_point_branches": [f.branch for f in res.fixed_points]}
    for f in res.fixed_points:
        for o in _OBS:
            balls[f"{o}_{f.branch}"] = getattr(f, o)
        meta[f"stable_{f.branch}"] = f.stable
    balls["phi"] = res.phi
    beta = p.get("beta", 1)
    if res.selected is not None:
        balls["magnetization"] = res.selected.magnetization
    meta["selected_branch"] = res.selected.branch if res.selected else None
    meta["selected_stable"] = (res.selected or res.fixed_points[-1]).stable
    v, e = _put(**balls)
    return Out({**v, **meta}, f"{len(res.fixed_points)} BP fixed point(s) "
               f"({', '.join(meta['fixed_point_branches'])}) at beta = {beta}, h = {p.get('h', 0)}"
               + ("" if res.selected else "; h = 0 below T_c: magnetization not chosen (pass branch)"),
               errors=e)


def _fp_stability(stable):
    return Stability("stable" if stable else "unstable", "bethe", DM,
                     f"computed BP Jacobian (k - 1) xi'(x) {'<' if stable else '>'} 1")


def _bethe_finality():
    fin = {"beta_c": Stability("not-computed", "bethe", f"{DM}, Eq. (2.5)", "closed-form threshold "
                               "(k - 1) tanh beta_c = 1, stated by DM as the statistical-physics "
                               "prediction (their theorems do not locate it); not a computed stability")}
    for b in _BRANCHES:
        for o in _OBS:
            fin[f"{o}_{b}"] = lambda v, b=b: _fp_stability(v.get(f"stable_{b}"))
    sel = lambda v: Stability("stable" if v.get("selected_stable") else "unstable", "bethe",
                              f"{DM}, Lemma 2.3, Thm 2.4", "the thermodynamic fixed point (B > 0: the "
                              "unique one on [0, oo); B = 0: the limit B -> 0)")
    fin["phi"] = fin["magnetization"] = sel
    return fin


def bethe_ising():
    steps = [fstep("bethe-critical-point", f"{DM}, Eq. (2.5)", _bcrit),
             fstep("bethe-bp-fixed-points", f"{DM}, Eqs. (2.6)-(2.9)", _bethe)]
    refs = [RefCheck("lattice", "bethe_beta_c_k3", lambda v: v["beta_c"], route="cavity")]
    meta = ("k", "fixed_point_branches", "selected_branch", "selected_stable") + \
        tuple(f"stable_{b}" for b in _BRANCHES)
    spec = ModelSpec("bethe_ising", "ising", "none", constraint="random regular graph, ferromagnet")
    return entry("bethe_ising", spec, DM, {"cavity": steps}, refs, {"cavity": CAVEAT_BETHE},
                 "ferromagnetic Ising model on the Bethe lattice / random k-regular graph: all BP "
                 "fixed points, certified", default_route="cavity", finality=_bethe_finality(),
                 meta=meta, parameters={"cavity": ("k", "beta", "h", "branch")},
                 rigor={"cavity": "certified"})


def lattice_entries():
    return [ising2d(), ising1d(), bethe_ising()]


__all__ = ["bethe_ising", "ising1d", "ising2d", "lattice_entries"]
