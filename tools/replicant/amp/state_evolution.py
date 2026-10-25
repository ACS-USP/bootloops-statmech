"""State evolution (SE) of the SK AMP iteration of amp/sk.py.

Convention (amp/sk.py): m^t = f(x^t), f(x) = tanh(beta (h + x)); Q_t = <m^t, m^t>
(<a, b> = (1/N) sum_i a_i b_i), D_t = <m^t, m^{t-1}>.

Scalar SE (Bayati & Montanari, IEEE Trans. Inf. Theory 57, 764 (2011), arXiv:1001.3448,
Sec. 4, eq. (4.2), there tau_{t+1}^2 = E f(tau_t Z)^2):
    Q_{t+1} = F(Q_t),   F(q) = E tanh^2(beta (h + sqrt(q) Z)),   Q_0 = <m^0, m^0>.
F is increasing, so from Q_0 = 1 the recursion decreases monotonically to the largest
fixed point.  All fixed points in [0, 1] found by a grid scan are listed with their SE
slope F'(q) (q = 0 is one at h = 0: unstable for beta > 1).
Two-time SE (consecutive iterates; Bolthausen, Commun. Math. Phys. 325, 333-366
(2014), doi:10.1007/s00220-013-1862-3, arXiv:1201.2891, Lemma 2.2, his psi):
(X_{t+1}, X_t) centred Gaussian with covariance
[[Q_t, D_t], [D_t, Q_{t-1}]], Q_{t+1} = E f(X_{t+1})^2, D_{t+1} = E f(X_{t+1}) f(X_t),
D_1 = mean(m^0) E f(sqrt(Q_0) Z) (x^1 = J m^0 is asymptotically independent of m^0).
The predicted increment <m^t - m^{t-1}, m^t - m^{t-1}> = Q_t + Q_{t-1} - 2 D_t.
At the fixed point, D -> psi(D) with psi(c) = E_Z [E_Z' f(sqrt(c) Z + sqrt(q - c) Z')]^2;
the iterates converge iff the fixed point c = q attracts, i.e. psi'(q) < 1
(Bolthausen Thm 2.1, h > 0).  `corr_slope` is psi'(q) by finite differences of this
bivariate map (no closed form is used).

Status: SE is an N -> oo statement for each fixed t (a theorem for Gaussian J, the
references above); it is NOT a statement about the Gibbs measure.  Numbers here are
deterministic float64 quadrature: the trajectory, the fixed-point polish (secant) and
the grid scan with a 120-node Gauss-Hermite rule; the polished fixed point is audited with
an independent rule (adaptive mpmath quadrature, |F(q) - q| <= 1e-15, and within 1e-8 of
the trajectory end, else the result says not converged); the two-time map with an
80 x 80 Gauss-Hermite product rule.  q carries float64 accuracy (~1e-13).
"""
import math
from dataclasses import dataclass, field

import mpmath as mp
import numpy as np

from ..core import Refusal

_GH_Z, _GH_W = np.polynomial.hermite_e.hermegauss(120)
_GH_W = _GH_W / _GH_W.sum()
_Z2, _W2 = np.polynomial.hermite_e.hermegauss(80)
_W2 = _W2 / _W2.sum()
_DPS = 18
_MAX_ITERS = 2000


@dataclass(frozen=True)
class FixedPoint:
    q: object          # mpf
    slope: object      # F'(q), the SE (not AT) slope; < 1 attracts the scalar recursion


@dataclass
class SEResult:
    q: object                      # mpf: the limit of the trajectory (polished)
    trajectory: list               # Q_0, Q_1, ... (floats)
    converged: bool
    fixed_points: list = field(default_factory=list)
    corr_slope: object = None      # psi'(q) (float), None at q = 0
    amp_converges: object = None   # two-time SE prediction for AMP (True/False)
    beta: object = None
    h: object = None
    label: str = "deterministic: N -> oo state-evolution prediction (quadrature)"
    note: str = ""


@dataclass
class TwoTimeSE:
    Q: list
    D: list            # D[0] is None
    increment: list    # increment[0] is nan


def _num(x, name):
    try:
        v = mp.mpf(x)
    except (TypeError, ValueError):
        raise Refusal(f"precondition:numeric-{name}", f"{name}={x!r} is not a number")
    if not mp.isfinite(v):
        raise Refusal(f"precondition:finite-{name}", f"{name}={x!r} is not finite")
    return v


def _check_beta_h(beta, h):
    b, hh = _num(beta, "beta"), _num(h, "h")
    if b < 0:
        raise Refusal("precondition:beta-nonnegative", f"beta={beta} < 0 (beta = 1/T)")
    return b, hh


def _field(b, h, q, z):
    """beta (h + sqrt(q) z): the cavity field of the SE map (a hook the tests plant).
    Works on floats, numpy arrays and mpf alike."""
    return b * (h + q**0.5 * z)


def _F(b, h, q):
    """F(q) = E tanh^2(_field(z)), 120-node Gauss-Hermite in float64."""
    return float(_GH_W @ np.tanh(_field(b, h, q, _GH_Z))**2)


def _F_mp(b, h, q):
    """The same expectation by adaptive tanh-sinh quadrature on [-12, 12] (the omitted
    Gaussian tails are < 1e-31), split at z = 0 and at the zero of the field."""
    pts = [-12, 0, 12]
    if q > 0 and h != 0:
        c = max(-12, min(12, -h / mp.sqrt(q)))
        pts = [-12, min(0, c), max(0, c), 12]
    return mp.quad(lambda z: mp.tanh(_field(b, h, q, z))**2 * mp.exp(-z * z / 2),
                   pts) / mp.sqrt(2 * mp.pi)


def _slope(b, h, q, d=1e-6):
    """F'(q) by differences (forward at q = 0, where F is only defined for q >= 0)."""
    if q < 2 * d:
        return (-3 * _F(b, h, q) + 4 * _F(b, h, q + d) - _F(b, h, q + 2 * d)) / (2 * d)
    return (_F(b, h, q + d) - _F(b, h, q - d)) / (2 * d)


def _bisect(b, h, lo, hi):
    glo = _F(b, h, lo) - lo
    for _ in range(200):
        mid = (lo + hi) / 2
        gm = _F(b, h, mid) - mid
        if gm == 0 or hi - lo < 1e-15:
            return mid
        if (gm > 0) == (glo > 0):
            lo, glo = mid, gm
        else:
            hi = mid
    return (lo + hi) / 2


def _polish(b, h, q):
    """Secant on F(q) - q with the adaptive rule (dps 18) from the trajectory end;
    returns (root, |F(root) - root|).  q = 0 is exact at h = 0."""
    with mp.workdps(_DPS):
        if h == 0 and q < 1e-12:
            return mp.mpf(0), abs(_F_mp(b, h, mp.mpf(0)))
        x0, x1 = mp.mpf(q), mp.mpf(q) + mp.mpf("1e-9")
        g0, g1 = _F_mp(b, h, x0) - x0, _F_mp(b, h, x1) - x1
        for _ in range(8):          # the start is within ~1e-11 of the root
            if g1 == g0:
                break
            x0, x1, g0 = x1, x1 - g1 * (x1 - x0) / (g1 - g0), g1
            g1 = _F_mp(b, h, x1) - x1
            if abs(x1 - x0) < mp.mpf("1e-17"):
                break
        return x1, abs(g1)


def _fixed_points(b, h, q_star):
    """Grid scan of F(q) - q on (0, 1] (200 points) plus q = 0 at h = 0; refined by
    bisection, merged with the trajectory's limit, each with its SE slope F'(q)."""
    roots = [0.0] if h == 0 else []
    grid = np.linspace(1e-6, 1, 201)
    g = [_F(b, h, q) - q for q in grid]
    for q1, q2, g1, g2 in zip(grid, grid[1:], g, g[1:]):
        if g1 * g2 < 0:
            roots.append(_bisect(b, h, float(q1), float(q2)))
    if q_star is not None:
        roots.append(q_star)
    out = []
    for r in sorted(roots):
        if not out or abs(r - out[-1]) > 1e-8:
            out.append(r)
        elif r == q_star:
            out[-1] = r
    return [FixedPoint(mp.mpf(r), mp.mpf(_slope(b, h, float(r)))) for r in out]


def _psi(b, h, q, c):
    """psi(c) = E_Z [E_Z' tanh(beta(h + sqrt(c) Z + sqrt(q - c) Z'))]^2 (80 x 80 GH)."""
    inner = np.tanh(b * (h + math.sqrt(c) * _Z2[:, None] + math.sqrt(max(q - c, 0.0))
                         * _Z2[None, :])) @ _W2
    return float(_W2 @ inner**2)


def _corr_slope(b, h, q):
    """psi'(q), one-sided second-order difference (psi is defined on [0, q])."""
    d = 1e-4 * q
    return (3 * _psi(b, h, q, q) - 4 * _psi(b, h, q, q - d) + _psi(b, h, q, q - 2 * d)) / (2 * d)


def random_init_q(beta, h=0):
    """Q_0 of amp/sk.py's init="random": m^0 = tanh(beta (h + g)), g ~ N(0, 1) iid."""
    b, hh = (float(v) for v in _check_beta_h(beta, h))
    return float(_GH_W @ np.tanh(b * (hh + _GH_Z))**2)


def _random_init_mean(b, hh):
    return float(_GH_W @ np.tanh(b * (hh + _GH_Z)))


def sk_state_evolution(beta, h=0, *, q0=1, iters=None, tol=1e-13) -> SEResult:
    """Scalar SE from Q_0 = q0.  iters=None runs to |Q_{t+1} - Q_t| < tol (at most
    2000 steps); an integer runs exactly that many steps (trajectory has iters + 1
    entries).  `q` is the polished fixed point when the trajectory converged, else
    the last iterate (converged False: e.g. critical slowing down at h = 0, beta ~ 1)."""
    b, hh = _check_beta_h(beta, h)
    q0 = _num(q0, "q0")
    if not 0 <= q0 <= 1:
        raise Refusal("precondition:q0-range", f"q0={q0} outside [0, 1] (Q_0 = <m^0, m^0>)")
    if iters is not None and (not isinstance(iters, int) or iters < 1):
        raise Refusal("precondition:positive-iters", f"iters={iters!r} is not a positive int")
    bf, hf = float(b), float(hh)
    tr = [float(q0)]
    steps = iters if iters is not None else _MAX_ITERS
    for _ in range(steps):
        tr.append(_F(bf, hf, tr[-1]))
        if iters is None and abs(tr[-1] - tr[-2]) < tol:
            break
    converged = abs(tr[-1] - tr[-2]) < 1e-10
    note = []
    q = tr[-1]
    if converged:
        qp, audit = _polish(b, hh, tr[-1])
        if abs(qp - tr[-1]) > 1e-8 or audit > 1e-15:
            converged = False
            note.append(f"adaptive-quadrature root {mp.nstr(qp, 12)} (trajectory end "
                        f"{tr[-1]:.12g}, residual {mp.nstr(audit, 3)}) fails the audit: "
                        f"reported not converged")
        else:
            q = qp
            note.append(f"fixed point polished by adaptive mpmath quadrature (|F(q) - q| = "
                        f"{mp.nstr(audit, 3)}), within {abs(float(qp) - tr[-1]):.1g} of the "
                        f"Gauss-Hermite trajectory end")
    with mp.workdps(_DPS):
        fps = _fixed_points(bf, hf, q if converged else None)
        q = mp.mpf(q)
        if not converged:
            note.append(f"trajectory not converged after {len(tr) - 1} steps "
                        f"(last step {abs(tr[-1] - tr[-2]):.2g})")
        cs, conv = None, None
        if converged and q > 0:
            cs = _corr_slope(bf, hf, float(q))
            conv = bool(hh != 0 and cs < 1)
            note.append(f"two-time SE: psi'(q) = {cs:.6g}; " + (
                "AMP iterates predicted to converge (Bolthausen Thm 2.1, h > 0)" if conv else
                "AMP iterates predicted NOT to converge (h = 0: psi(0) = 0, a zero-mean start "
                "decorrelates successive iterates)" if hh == 0 else
                "AMP iterates predicted NOT to converge (psi'(q) >= 1: on the RS-unstable side of "
                "the AT line)"))
        elif converged:
            conv = True
            note.append("q = 0 (paramagnet): Q_t -> 0, so the increments vanish")
        note.append("SE is an N -> oo prediction for each fixed t; in the spin-glass phase "
                    "(h = 0, beta > 1) its fixed point need not describe the Gibbs measure")
    return SEResult(q, tr, converged, fps, cs, conv, b, hh, note="; ".join(note))


def sk_two_time_state_evolution(beta, h=0, *, iters, q0=None, m0_mean=None) -> TwoTimeSE:
    """Q_t, D_t and the predicted increments for t = 0..iters; the defaults are the
    moments of amp/sk.py's init="random" (Q_0 = E tanh^2(beta(h+Z)), mean E tanh)."""
    b, hh = (float(v) for v in _check_beta_h(beta, h))
    if not isinstance(iters, int) or iters < 1:
        raise Refusal("precondition:positive-iters", f"iters={iters!r} is not a positive int")
    Q0 = random_init_q(b, hh) if q0 is None else float(q0)
    mu = _random_init_mean(b, hh) if m0_mean is None else float(m0_mean)
    f = lambda x: np.tanh(b * (hh + x))
    Z1, Z2 = np.meshgrid(_Z2, _Z2, indexing="ij")
    W = np.outer(_W2, _W2)
    Q = [Q0, float(_W2 @ f(math.sqrt(Q0) * _Z2)**2)]
    D = [None, mu * float(_W2 @ f(math.sqrt(Q0) * _Z2))]
    for t in range(1, iters):
        a, c, d = Q[t], Q[t - 1], D[t]
        r = 0.0 if a * c == 0 else max(-1.0, min(1.0, d / math.sqrt(a * c)))
        X = math.sqrt(a) * Z1
        Y = math.sqrt(c) * (r * Z1 + math.sqrt(1 - r * r) * Z2)
        Q.append(float(np.sum(W * f(X)**2)))
        D.append(float(np.sum(W * f(X) * f(Y))))
    inc = [math.nan] + [Q[t] + Q[t - 1] - 2 * D[t] for t in range(1, iters + 1)]
    return TwoTimeSE(Q, D, inc)
