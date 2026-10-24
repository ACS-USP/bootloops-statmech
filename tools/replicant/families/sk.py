"""Family (b): Sherrington-Kirkpatrick, J_ij ~ N(0, 1/N), field h, beta = 1/T.

  sk_rs(beta, h=0, *, require_stable=False) -> RSSolution(q, f, e, stable, replicon)
  sk_replicon(beta, h=0, *, q=None)         -> lambda_AT (negative = RS unstable)
  sk_at_temperature(h)                      -> T on the AT line (h != 0)
  sk_1rsb(beta, h=0)                        -> RSBSolution(q0, q1, m, f, e)
  sk_ground_energy("rs" | "1rsb")           -> T = 0 energy per spin

Replica free energy (n -> 0 of (1/n) ln E Z^n, Parisi matrix q_ab):
  -beta f = ln 2 + beta^2/4 [1 - lim (1/n) sum_{a!=b} q_ab^2] + (1/n) ln Tr e^{...}
The quadratic term comes from ansatz.sum_offdiag_power(k=2) + n_to_zero
(RS: -q^2; 1RSB with ms=[m]: (m-1) q1^2 - m q0^2).  Hubbard-Stratonovich on
the trace (sum_{a!=b} q_ab s_a s_b = q0 (sum s)^2 + (q1-q0) sum_blocks (...)^2 - n q1):
  RS:   -beta f = ln 2 + beta^2 (1-q)^2/4 + E_z ln cosh Xi,  Xi = beta(h + sqrt(q) z)
        q = E tanh^2 Xi;  e = -beta(1 - q^2)/2 - h E tanh Xi
        replicon (de Almeida-Thouless 1978) lambda = 1 - beta^2 E sech^4 Xi.
  1RSB: -beta f = ln 2 + beta^2/4 [1 - 2 q1 + (1-m) q1^2 + m q0^2]
                  + (1/m) E_z ln E_y cosh^m Xi,  Xi = beta(h + sqrt(q0) z + sqrt(q1-q0) y)
        stationary in q0, q1, m:  q1 = E_z <tanh^2>,  q0 = E_z <tanh>^2,
        beta^2 (q0^2 - q1^2)/4 - E_z ln E_y cosh^m / m^2 + E_z <ln cosh> / m = 0,
        <g> = E_y[cosh^m Xi g] / E_y cosh^m Xi;  e = -d(-beta f)/d beta at fixed (q0,q1,m).
  1RSB at T = 0 (q1 -> 1, m -> 0, mu = beta m fixed, cosh^m -> 2^-m e^{mu |.|}):
        E(q0, mu) = mu (1-q0^2)/4 - (1/mu) E_z ln E_y exp(mu |sqrt(q0) z + sqrt(1-q0) y|),
        inner E_y in closed form (Gaussian times Phi).
At h = 0 the criterion is T_c = 1 (q > 0 iff beta > 1; the q = 0 branch has
replicon 1 - beta^2); the AT line h_AT(T) only exists for h != 0.

Numerics.  RS quantities use solve.gauss_expect/fixed_point at dps=15 (q to
~1e-12).  1RSB uses float64 Gauss-Hermite (80 x 80 nodes) and Newton on the
stationarity equations from several seeds (RS-derived and, at h = 0, the
q0 = 0 subspace); among converged stationary points the one with the LARGEST
f is returned, since for n -> 0 the replica free energy is maximised over the
order parameters (at h = 0 this is a q0 > 0 point, above the q0 = 0 one).
The grid is checked against 160 x 160 nodes: at beta = 2 everything agrees to
~1e-11, at beta = 3 f to ~2e-10 and (q0, q1, m) to ~1e-6; beta > 3 is refused
(the y-integrand gets too sharp).  The T = 0 1RSB uses 200 Hermite nodes.

Honest label: the n -> 0 continuation and the RS/1RSB ansatze are
physics-level (Parisi), not proven; the numbers carry quadrature error only.
"""
import math
from dataclasses import dataclass
from functools import lru_cache

import mpmath as mp
import numpy as np
import sympy as sp

from ..ansatz import n_to_zero, sum_offdiag_power
from ..core import Refusal
from ..solve import fixed_point, gauss_expect

CAVEAT = ("physics-level: replica n->0 continuation and RS/1RSB ansatz "
          "(Parisi), not proven; numbers carry quadrature error only")
_DPS = 15
_BETA_MAX_1RSB = 3.0


@dataclass(frozen=True)
class RSSolution:
    q: object
    f: object
    e: object
    stable: bool
    replicon: object = None
    caveat: str = CAVEAT


@dataclass(frozen=True)
class RSBSolution:
    q0: object
    q1: object
    m: object
    f: object
    e: object
    caveat: str = CAVEAT


def quadratic_term(kind):
    """1 - 2 q_EA - lim (1/n) sum_{a!=b} q_ab^2 via ansatz.py (sympy expr).

    The -2 q_EA is the diagonal left by Hubbard-Stratonovich (-n q_EA in the
    exponent); the n -> 0 of the overlap sum is ansatz.n_to_zero.
    """
    n, q, q0, q1, m = sp.symbols("n q q0 q1 m")
    if kind == "rs":
        return sp.expand(1 - 2 * q - n_to_zero(sum_offdiag_power("rs", [], [q], 2, n), n))
    return sp.expand(1 - 2 * q1 - n_to_zero(sum_offdiag_power("rsb", [m], [q0, q1], 2, n), n))


_QUAD_RSB = sp.lambdify(sp.symbols("q0 q1 m"), quadratic_term("rsb"), "math")
_QUAD_RS = sp.lambdify(sp.symbols("q"), quadratic_term("rs"), "mpmath")


def _ge(g):
    return gauss_expect(g, dps=_DPS)


@lru_cache(maxsize=256)
def _rs_q(beta: str, h: str):
    """RS overlap: the q > 0 branch when it exists (always at h != 0), else 0."""
    with mp.workdps(_DPS):
        b, hh = mp.mpf(beta), mp.mpf(h)
        F = lambda q: _ge(lambda z: mp.tanh(b * (hh + mp.sqrt(q) * z))**2)
        try:
            return fixed_point(F, mp.mpf("0.5"), dps=_DPS, tol="1e-12",
                               branch="nontrivial").value
        except Refusal:
            if hh != 0 or b > 1:   # below T_c a missing q > 0 branch is a failure
                raise
            return mp.mpf(0)    # h = 0, beta <= 1: paramagnet


def _key(x):   # cache key; a float goes through repr so 0.9 and "0.9" share it
    with mp.workdps(_DPS):   # independent of the caller's global precision
        return mp.nstr(mp.mpf(repr(x) if isinstance(x, float) else x), 30)


def sk_replicon(beta, h=0, *, q=None):
    """lambda_AT = 1 - beta^2 E sech^4(beta(h + sqrt(q) z)); q defaults to RS."""
    with mp.workdps(_DPS):
        b, hh = mp.mpf(beta), mp.mpf(h)
        q = _rs_q(_key(b), _key(hh)) if q is None else mp.mpf(q)
        return 1 - b**2 * _ge(lambda z: mp.sech(b * (hh + mp.sqrt(q) * z))**4)


def sk_rs(beta, h=0, *, require_stable=False) -> RSSolution:
    with mp.workdps(_DPS):
        b, hh = mp.mpf(beta), mp.mpf(h)
        q = _rs_q(_key(b), _key(hh))
        xi = lambda z: b * (hh + mp.sqrt(q) * z)
        phi = mp.log(2) + b**2 * _QUAD_RS(q) / 4 + _ge(lambda z: mp.log(mp.cosh(xi(z))))
        mag = _ge(lambda z: mp.tanh(xi(z))) if hh != 0 else mp.mpf(0)
        lam = sk_replicon(b, hh, q=q)
        sol = RSSolution(q, -phi / b, -b * (1 - q**2) / 2 - hh * mag, bool(lam >= 0), lam)
    if require_stable and not sol.stable:
        raise Refusal("unstable:rs", f"replicon = {mp.nstr(lam, 6)} < 0 at beta={beta}, "
                      f"h={h}; RS is invalid here (use sk_1rsb; no fallback)")
    return sol


def sk_at_temperature(h):
    """Temperature where the RS replicon vanishes at field h != 0."""
    with mp.workdps(_DPS):
        hh = mp.mpf(h)
        if hh == 0:
            raise Refusal("precondition:nonzero-field",
                          "at h = 0 the RS instability is T_c = 1, not an AT line")
        t0 = 1 - (3 * hh**2 / 4) ** (mp.mpf(1) / 3)       # leading-order seed
        b = mp.findroot(lambda b: sk_replicon(b, hh), (1 / t0, 1 / (t0 + mp.mpf("0.02"))),
                        solver="secant", tol=mp.mpf("1e-18"))
        return 1 / b


# ---- 1RSB: float64 Gauss-Hermite grids -------------------------------------
_X, _W = np.polynomial.hermite_e.hermegauss(80)
_W = _W / math.sqrt(2 * math.pi)


def _lncosh(u):   # ln cosh u, overflow-safe
    a = np.abs(u)
    return a + np.log1p(np.exp(-2 * a)) - math.log(2)


def _rsb_parts(b, h, q0, q1, m):
    y = h + math.sqrt(q0) * _X[:, None] + math.sqrt(q1 - q0) * _X[None, :]
    lc, t = _lncosh(b * y), np.tanh(b * y)
    L = m * lc
    Lmax = L.max(axis=1, keepdims=True)
    Wt = np.exp(L - Lmax) * _W[None, :]
    Z = Wt.sum(1)
    av = lambda g: (Wt * g).sum(1) / Z
    return np.log(Z) + Lmax[:, 0], av(t), av(t * t), av(lc), av(t * y)


def _rsb_eqs(b, h, q0, q1, m):
    lnZ, t1, t2, lc, _ = _rsb_parts(b, h, q0, q1, m)
    return np.array([q0 - _W @ (t1 * t1), q1 - _W @ t2,
                     b**2 * (q0**2 - q1**2) / 4 - (_W @ lnZ) / m**2 + (_W @ lc) / m])


def _rsb_phi_e(b, h, q0, q1, m):
    lnZ, _, _, _, ty = _rsb_parts(b, h, q0, q1, m)
    quad = _QUAD_RSB(q0, q1, m)
    return math.log(2) + b**2 * quad / 4 + (_W @ lnZ) / m, -(b * quad / 2 + _W @ ty)


def _newton(G, v, ok, tol=1e-12, maxit=60, eps=1e-7):
    """Newton with central-difference Jacobian; halves steps to stay in `ok`."""
    v = np.array(v, float)
    for _ in range(maxit):
        g = G(v)
        if not np.all(np.isfinite(g)):
            return None
        if np.abs(g).max() < tol:
            return v
        J = np.empty((len(v), len(v)))
        for k in range(len(v)):
            d = np.zeros(len(v))
            d[k] = eps
            J[:, k] = (G(v + d) - G(v - d)) / (2 * eps)
        try:
            step = np.linalg.solve(J, -g)
        except np.linalg.LinAlgError:
            return None
        lam = 1.0
        while not ok(v + lam * step):
            lam /= 2
            if lam < 1e-6:
                return None
        v = v + lam * step
    return None


@lru_cache(maxsize=64)
def _rsb(b: float, h: float):
    if b > _BETA_MAX_1RSB:
        raise Refusal("precondition:quadrature-range",
                      f"1RSB grid validated for beta <= {_BETA_MAX_1RSB}, got {b}")
    q = float(_rs_q(_key(b), _key(h)))
    eps = 1e-7
    ok3 = lambda v: eps < v[0] and v[0] + eps < v[1] < 1 - eps and eps < v[2] < 1
    cands = []
    for s in ((0.5, 1.15, 0.5), (0.3, 1.2, 0.3), (0.8, 1.05, 0.7)):
        v = _newton(lambda v: _rsb_eqs(b, h, *v), (s[0] * q, min(s[1] * q, 0.99), s[2]), ok3)
        if v is not None:
            cands.append(tuple(v))
    if h == 0:   # q0 = 0 solves the q0 equation exactly by symmetry
        ok2 = lambda v: eps < v[0] < 1 - eps and eps < v[1] < 1
        v = _newton(lambda v: _rsb_eqs(b, 0.0, 0.0, *v)[1:], (min(1.15 * q, 0.99), 0.5), ok2)
        if v is not None:
            cands.append((0.0, *v))
    cands = [c for c in cands if c[1] - c[0] > 1e-6]       # q1 = q0 is just RS
    if not cands:
        lam = sk_replicon(b, h)
        if lam >= 0:
            raise Refusal("no-solution", f"1RSB collapses to RS: RS is stable here "
                          f"(replicon = {mp.nstr(lam, 6)} >= 0 at beta={b}, h={h})")
        raise Refusal("no-solution", f"solver: no 1RSB stationary point with q1 > q0 "
                      f"at beta={b}, h={h} although RS is unstable")
    best = max(cands, key=lambda c: -_rsb_phi_e(b, h, *c)[0] / b)
    eigs = _rsb_hessian_eigs(b, h, *best)
    if max(eigs) >= 0:
        raise Refusal("no-solution", f"selected 1RSB point {best} is not a local maximum "
                      f"of f (Hessian eigenvalues {eigs})")
    phi, e = _rsb_phi_e(b, h, *best)
    return (*best, -phi / b, e)


def _rsb_hessian_eigs(b, h, q0, q1, m, d=1e-4):
    """Eigenvalues of the central-difference Hessian of f in (q0, q1, m).

    The selection rule (largest f) is audited by requiring a local maximum
    (all eigenvalues < 0); the chosen point has q0 > d in every case tested.
    """
    f = lambda v: -_rsb_phi_e(b, h, *v)[0] / b
    v0, H = np.array([q0, q1, m]), np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            ei, ej = np.eye(3)[i] * d, np.eye(3)[j] * d
            H[i, j] = (f(v0 + ei + ej) - f(v0 + ei - ej) - f(v0 - ei + ej)
                       + f(v0 - ei - ej)) / (4 * d * d)
    return tuple(np.linalg.eigvalsh((H + H.T) / 2))


def sk_1rsb(beta, h=0) -> RSBSolution:
    q0, q1, m, f, e = _rsb(float(beta), float(h))
    return RSBSolution(*(mp.mpf(v) for v in (q0, q1, m, f, e)))


# ---- T = 0 ------------------------------------------------------------------
_X0, _W0 = np.polynomial.hermite_e.hermegauss(200)
_W0 = _W0 / math.sqrt(2 * math.pi)
_ncdf = np.vectorize(lambda t: 0.5 * math.erfc(-t / math.sqrt(2)))


def _e0_1rsb(q0, mu):
    """E(q0, mu); E_y e^{mu|a + s y|} = e^{mu^2 s^2/2} [e^{mu a} Phi(a/s + mu s) + (a -> -a)]."""
    a, s = math.sqrt(q0) * _X0, math.sqrt(1 - q0)
    with np.errstate(divide="ignore"):    # log Phi = -inf where Phi underflows
        inner = np.logaddexp(mu * a + np.log(_ncdf(a / s + mu * s)),
                             -mu * a + np.log(_ncdf(-a / s + mu * s))) + (mu * s)**2 / 2
    return mu * (1 - q0**2) / 4 - (_W0 @ inner) / mu


@lru_cache(maxsize=None)
def _ground_1rsb_points():
    """T = 0 stationary points [(q0, mu, E, d2E/dq0^2)] from a q0 > 0 seed and
    from the exact q0 = 0 subspace (q0 = 0 solves the q0 equation at h = 0)."""
    d = 1e-5
    dE = lambda q0, mu, k: ((_e0_1rsb(q0 + d, mu) - _e0_1rsb(q0 - d, mu)) if k == 0 else
                            (_e0_1rsb(q0, mu + d) - _e0_1rsb(q0, mu - d))) / (2 * d)
    pts = []
    v = _newton(lambda v: np.array([dE(v[0], v[1], 0), dE(v[0], v[1], 1)]), (0.5, 1.0),
                lambda v: 0.01 < v[0] < 0.99 and 0.05 < v[1] < 20, tol=1e-9, eps=1e-4)
    if v is not None:
        pts.append((float(v[0]), float(v[1])))
    v = _newton(lambda v: np.array([dE(0.0, v[0], 1)]), (1.0,),
                lambda v: 0.05 < v[0] < 20, tol=1e-9, eps=1e-4)
    if v is not None:
        pts.append((0.0, float(v[0])))
    dq, out = 1e-3, []
    for q0, mu in pts:      # central 2nd difference; forward one at the q0 = 0 edge
        e = [_e0_1rsb(q0 + k * dq, mu) for k in ((0, 1, 2) if q0 == 0 else (-1, 0, 1))]
        out.append((q0, mu, e[1] if q0 else e[0], (e[0] - 2 * e[1] + e[2]) / dq**2))
    return tuple(out)


@lru_cache(maxsize=None)
def _ground_1rsb():
    pts = _ground_1rsb_points()
    if not pts:
        raise Refusal("no-solution", "T = 0 1RSB stationary point not found")
    best = max(pts, key=lambda p: p[2])       # replica rule: largest E (f at T = 0)
    if best[3] >= 0:
        raise Refusal("no-solution", f"T = 0 1RSB point {best[:2]} is not a maximum in q0 "
                      f"(d2E/dq0^2 = {best[3]})")
    return best[2]


def sk_ground_energy(ansatz: str):
    if ansatz == "rs":   # e0 = -lim beta(1-q) = -2 phi(0) = -E|z|, by quadrature
        with mp.workdps(30):
            return -gauss_expect(abs, dps=30)
    if ansatz == "1rsb":
        return mp.mpf(_ground_1rsb())
    if ansatz == "fullrsb":
        raise Refusal("not-implemented:fullrsb", "banked in reference/sk.json; Task 16 solves it")
    raise Refusal("precondition:known-ansatz", f"{ansatz!r}: use 'rs', '1rsb' or 'fullrsb'")
