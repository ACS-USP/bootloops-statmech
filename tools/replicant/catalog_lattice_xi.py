"""Catalog steps of `ising2d` (route `exact`) for the correlation length, the exponent consistency
checks and the critical correlations (I2), certified.  Wired into catalog_lattice.ising2d (STEPS,
REFS, FIN, META).  Every physical value is a python-flint arb ball (lattice/correlation2d.py,
lattice/critical2d.py); the direct strip transfer matrix (lattice/strip_tm.py, double precision)
and the strip -> bulk sequence are checks, reported in meta, never values.
  ising2d-correlation-length     (off beta_c) xi and 1/xi from the exact bulk form (the branch named
                                 in xi_branch); Onsager's strip gaps at n = 4, 6 against the direct
                                 transfer matrix (else no-solution:strip-disagreement); the strip
                                 -> bulk estimate within its band when resolved (else
                                 no-solution:strip-limit)
  ising2d-critical-exponents     nu from the log-slope of xi, eta from the strip gap at beta_c
                                 (n gap -> pi eta, Cardy) and from the diagonal correlation's
                                 log-slope, the amplitudes xi_0+-; outside a band
                                 no-solution:exponent-mismatch
  ising2d-critical-correlations  (at beta_c) nearest-neighbour sqrt 2 / 2, diagonal C(n, n) (run
                                 parameter n), its amplitude; Toeplitz = product, strip checks
                                 (else no-solution:critical-correlation)
"""
from functools import lru_cache

import mpmath as mp

from .catalog import UNCLAIMED, Out, RefCheck, Stability, fstep
from .core import Refusal
from .lattice import BETA_C
from .lattice import correlation2d as C
from .lattice import critical2d as D
from .lattice import strip_tm as S
from .lattice.ball import certify
from .lattice.coupling import coupling
from .lattice.onsager import _ordered

ONS44 = "Onsager, Phys. Rev. 65, 117 (1944) (read: journal scan, pp. 117, 137-138)"
LM = "Lyberg & McCoy, arXiv:math-ph/0612051 (read), Eqs. (5), (6), (13), (20), (21), (31), (32)"
DOHM = "Dohm, arXiv:1907.00859 (read), Eqs. (30), (41), (46)"
DIK = "Deift, Its & Krasovsky, arXiv:1207.4990 (read)"
PERK = "Perk & Au-Yang, arXiv:0901.1931 (read), Eqs. (31), (32)"
CARDY = "Cardy, arXiv:0807.3472 (read), Eq. (19) and Sec. 2 (<phi phi> = r^{-2x})"
XI_CITE = f"{ONS44}, Eqs. (17), (89a), (95)-(103); {LM}; {DOHM}"
LIT = lambda cite, note: Stability("exact-by-literature", "exact", cite, note)
CONS = lambda cite, note: Stability("not-computed", "exact", cite, note)
STRIP_TOL = 1e-9           # |ln ratio| agreement, Onsager's spectrum vs the double-precision matrix
NU_BAND, ETA_BAND, CARDY_BAND = mp.mpf("1e-5"), mp.mpf("1e-5"), mp.mpf("1e-8")
TOEPLITZ_MAX = 10

UNCLAIMED[("lattice", "ising2d_strip_critical_gap_amplitude")] = (
    "pi eta = 2 pi x_sigma: the n -> oo limit of n ln(lambda_0/lambda_1) at beta_c, a Richardson "
    "extrapolation of Onsager's certified strip gaps (n = 128, 256) compared within 1e-8 in the "
    "ising2d-critical-exponents step and tests/test_lattice_xi.py: not a certified pipeline value")


def _put(**balls):
    vals, errs = {}, {}
    for k, b in balls.items():
        vals[k], errs[k] = b.value_and_error(40)
    return vals, errs


def _b2(p):
    return p.get("beta", BETA_C)


def _critical(p):
    return _b2(p) == BETA_C


def _s(x, d=6):
    return mp.nstr(x, d)


# ---- the correlation length --------------------------------------------------------------------
def _xi(p):
    beta = _b2(p)
    below = _ordered(coupling(beta).arb)
    direct = {}
    for n in (4, 6):
        g, (do, de) = C.strip_gaps(n, beta), S.strip_spectrum(n, beta)
        diff = max(abs(g.odd.mid() - do), abs(g.even.mid() - de))
        direct[n] = _s(diff, 2)
        if not diff <= STRIP_TOL:
            raise Refusal("no-solution:strip-disagreement", f"n = {n}: Onsager's strip gaps "
                          f"({_s(g.odd.mid(), 12)}, {_s(g.even.mid(), 12)}) and the direct transfer "
                          f"matrix ({do:.12g}, {de:.12g}) differ by {_s(diff, 2)} > {STRIP_TOL}")
    lim = C.strip_limit(beta)
    xi_inv = C.inverse_correlation_length(beta)
    off = abs(lim.estimate - xi_inv.mid())
    if lim.resolved and not off <= lim.band:
        raise Refusal("no-solution:strip-limit", f"the strip estimate {_s(lim.estimate, 12)} of 1/xi "
                      f"({lim.pair}) is {_s(off, 2)} from the bulk form {_s(xi_inv.mid(), 12)}, "
                      f"outside its band {_s(lim.band, 2)}")
    v, e = _put(correlation_length=C.correlation_length(beta), inverse_correlation_length=xi_inv)
    branch = ("T < T_c: 1/xi = 4 (K - K*), the two-particle gap" if below else
              "T > T_c: 1/xi = 2 (K* - K), the one-particle gap")
    strip = {"pair": lim.pair, "widths": list(lim.widths),
             "gaps": [_s(x, 12) for x in lim.sequence], "estimate": _s(lim.estimate, 15),
             "band": _s(lim.band, 2), "offset": _s(off, 2),
             "resolved": lim.resolved, "kind": "finite-size consistency check, no proven bound"}
    return Out({**v, "xi_branch": branch, "strip_direct_check": direct, "strip_limit": strip},
               f"{branch}; exponential correlation length along an axis (K* dual, tanh K* = "
               f"e^-2K); Onsager's strip gaps = the direct transfer matrix at n = 4, 6 (to "
               f"{STRIP_TOL}); strip -> bulk " + ("within its band" if lim.resolved else
                                                  "not resolved here (band >= 1/xi / 10): printed, "
                                                  "not asserted"), errors=e)


# ---- exponents ---------------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _exponents(hooks):
    nu, eta = C._HOOKS["nu"], C._HOOKS["eta"]
    balls = {"nu_loglog_above": C.xi_log_slope("-1e-6", "-1e-8"),
             "nu_loglog_below": C.xi_log_slope("1e-6", "1e-8"),
             "correlation_length_amplitude_plus": C.xi_amplitude("-1e-12"),
             "correlation_length_amplitude_minus": C.xi_amplitude("1e-12"),
             "strip_critical_gap_n256": C.critical_gap_scaled(256),
             "eta_diagonal_loglog": D.diagonal_log_slope(64, 128)}
    g128 = C.critical_gap_scaled(128).mid()
    with mp.workdps(40):
        rich = (4 * balls["strip_critical_gap_n256"].mid() - g128) / 3
        checks = {"nu_above": abs(balls["nu_loglog_above"].mid() + nu),
                  "nu_below": abs(balls["nu_loglog_below"].mid() + nu),
                  "eta_cardy": abs(rich - mp.pi * mp.mpf(eta.numerator) / eta.denominator),
                  "eta_diagonal": abs(balls["eta_diagonal_loglog"].mid()
                                      + mp.mpf(eta.numerator) / eta.denominator)}
    plus, minus = balls["correlation_length_amplitude_plus"], balls["correlation_length_amplitude_minus"]
    ratio = certify(lambda: plus.arb / minus.arb, 30)
    return balls, rich, checks, ratio


def _exp(p):
    balls, rich, checks, ratio = _exponents(tuple(sorted((k, str(v)) for k, v in
                                                         {**C._HOOKS, **D._HOOKS}.items())))
    bands = {"nu_above": NU_BAND, "nu_below": NU_BAND, "eta_cardy": CARDY_BAND,
             "eta_diagonal": ETA_BAND}
    bad = [k for k in checks if not checks[k] <= bands[k]]
    if bad:
        raise Refusal("no-solution:exponent-mismatch", ", ".join(
            f"{k}: |data - target| = {_s(checks[k], 3)} > {_s(bands[k], 1)}" for k in bad) +
            f" (targets nu = {C._HOOKS['nu']}, eta = {C._HOOKS['eta']})")
    v, e = _put(**balls, correlation_length_amplitude_ratio=ratio)
    return Out({**v, "strip_critical_gap_extrapolated": _s(rich, 15),
                "exponent_checks": {k: f"{_s(checks[k], 3)} (band {_s(bands[k], 1)})" for k in checks},
                "exponent_points": "nu: beta - beta_c = -+1e-6, -+1e-8; amplitudes: +-1e-12; "
                                   "Cardy: n = 128, 256 (Richardson in 1/n^2); diagonal: n = 64, 128"},
               "consistency checks on exact data: d ln xi / d ln|K - K_c| = -nu (both sides), n "
               "ln(lambda_0/lambda_1) at beta_c -> pi eta (Cardy), ln C(n,n) slope -> -eta", errors=e)


# ---- critical correlations ---------------------------------------------------------------------
def _crit_corr(p):
    n = p.get("n", 1)
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= D.MAX_N:
        raise Refusal("precondition:distance", f"n = {n!r}: need an int in 1 .. {D.MAX_N}")
    cn = D.diagonal_correlation(n)
    m = min(n, TOEPLITZ_MAX)
    if not D.diagonal_correlation(m).overlaps(D.diagonal_toeplitz(m)):
        raise Refusal("no-solution:critical-correlation", f"n = {m}: Perk & Au-Yang's product and the "
                      "Toeplitz determinant of DIK's symbol do not overlap")
    nn, c1 = D.critical_nn_correlation(), D.diagonal_correlation(1)
    strip = {}
    for L in (4, 6):
        row, col, diag = S.strip_correlations(L, BETA_C)
        mean_off, diag_off = abs((row + col) / 2 - nn.mid()), abs(diag - c1.mid())
        strip[L] = {"nn_mean_minus_bulk": _s(mean_off, 2), "diag_minus_bulk": _s(diag_off, 3)}
        # observed (not proved): the strip's mean nn correlation equals sqrt 2 / 2 at every width
        if not mean_off <= 1e-12 or (L == 6 and not diag_off <= 1e-3):
            raise Refusal("no-solution:critical-correlation", f"strip L = {L}: mean nn {(row + col) / 2}"
                          f", diagonal {diag} against {nn.mid(15)}, {c1.mid(15)}")
    v, e = _put(critical_nn_correlation=nn, critical_diagonal_correlation=cn,
                critical_diagonal_amplitude=D.diagonal_amplitude())
    return Out({**v, "diagonal_n": n, "strip_critical_correlations": strip},
               f"at beta_c: <s s>_nn = -u_c/2; <s_00 s_nn> (n = {n}) by Perk & Au-Yang's product, "
               f"equal to the Toeplitz determinant at n = {m}; strip L = 4, 6 (direct transfer "
               "matrix, double precision): mean nn correlation = sqrt 2 / 2 within 1e-12, diagonal "
               "-> 2/pi (8.4e-4 at L = 6; finite-size, no bound)", errors=e)


STEPS_2D = [fstep("ising2d-correlation-length", XI_CITE, _xi, when=lambda p: not _critical(p),
                  skip="beta = beta_c: xi = oo (xi ~ xi_0 |t|^-1)"),
            fstep("ising2d-critical-exponents", f"{XI_CITE}; {CARDY}; {DIK}, Eq. (101)", _exp),
            fstep("ising2d-critical-correlations", f"{ONS44}, Eq. (121); {PERK}; {DIK}, Eqs. "
                  "(103)-(107)", _crit_corr, when=_critical, skip="beta != beta_c")]
_ref = lambda key, k, **params: RefCheck("lattice", key, (lambda k: lambda v: v[k])(k),
                                         params=params, route="exact")
REFS_2D = [_ref("ising2d_correlation_length_beta0_3", "correlation_length", beta="0.3"),
           _ref("ising2d_correlation_length_beta0_6", "correlation_length", beta="0.6"),
           _ref("ising2d_correlation_length_amplitude_plus", "correlation_length_amplitude_plus"),
           _ref("ising2d_correlation_length_amplitude_ratio", "correlation_length_amplitude_ratio"),
           _ref("ising2d_critical_nn_correlation", "critical_nn_correlation"),
           _ref("ising2d_critical_diagonal_correlation_n1", "critical_diagonal_correlation"),
           _ref("ising2d_critical_diagonal_amplitude", "critical_diagonal_amplitude")]
_XI_NOTE = ("exact bulk form, certified ball: 2(K* - K) above T_c (Onsager's one-particle gap, "
            "Eqs. (99)-(101)), 4(K - K*) below (the two-particle gap 2 gamma(0), Eq. (102); the "
            "identification below T_c taken here from Lyberg & McCoy's form-factor structure, "
            "consistent with Dohm's xi_0+ / xi_0- = 2)")
FIN_2D = {
    "correlation_length": LIT(XI_CITE, _XI_NOTE),
    "inverse_correlation_length": LIT(XI_CITE, _XI_NOTE),
    "nu_loglog_above": CONS(XI_CITE, "log-slope of the certified xi between beta - beta_c = -1e-6, "
                            "-1e-8: a consistency check of nu = 1, |slope + 1| < 1e-5"),
    "nu_loglog_below": CONS(XI_CITE, "the same at beta - beta_c = 1e-6, 1e-8"),
    "correlation_length_amplitude_plus": CONS(DOHM, "xi |t| at K - K_c = -1e-12 (t = K_c/K - 1): "
                                              "xi_0+ = [2 ln(1 + sqrt 2)]^-1 up to O(t)"),
    "correlation_length_amplitude_minus": CONS(DOHM, "xi |t| at K - K_c = +1e-12: xi_0- up to O(t)"),
    "correlation_length_amplitude_ratio": CONS(DOHM, "xi_0+ / xi_0- from the two values: 2 up to O(t)"),
    "strip_critical_gap_n256": LIT(f"{ONS44}, Eqs. (95), (99), (100)", "256 ln(lambda_0/lambda_1) of "
                                   "the width-256 strip at beta_c, certified ball (-> pi eta)"),
    "eta_diagonal_loglog": CONS(f"{PERK}; {DIK}, Eq. (106)", "ln[C(128,128)/C(64,64)]/ln 2 from "
                                "certified values: a consistency check of eta = 1/4, |slope + 1/4| "
                                "< 1e-5"),
    "critical_nn_correlation": LIT(f"{ONS44}, Eq. (121)", "-u_c/2 = sqrt 2 / 2 per bond (two bonds "
                                   "per site, isotropy), certified ball"),
    "critical_diagonal_correlation": LIT(f"{PERK}; {DIK}, Eqs. (103)-(105)", "Perk & Au-Yang's "
                                         "product, certified ball; equals the Toeplitz determinant"),
    "critical_diagonal_amplitude": LIT(f"{DIK}, Eqs. (106), (107)", "e^{1/4} 2^{1/12} A^{-3}, A "
                                       "Glaisher's constant, certified ball"),
}
META_2D = ("xi_branch", "strip_direct_check", "strip_limit", "strip_critical_gap_extrapolated",
           "exponent_checks", "exponent_points", "diagonal_n", "strip_critical_correlations")

__all__ = ["FIN_2D", "META_2D", "REFS_2D", "STEPS_2D"]
