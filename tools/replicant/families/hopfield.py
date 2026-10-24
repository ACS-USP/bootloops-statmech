"""Family (b): Hopfield model, replica symmetric, with branch control.

Model and conventions (Amit, Gutfreund & Sompolinsky, PRL 55, 1530 (1985);
Ann. Phys. 173, 30 (1987); Hertz-Krogh-Palmer ch. 10):
  H = -(1/2N) sum_{ij} sum_{mu=1}^p xi_i^mu xi_j^mu s_i s_j,  p = alpha N random
  +-1 patterns, beta = 1/T, one condensed pattern with overlap m, the other p - 1
  patterns acting as Gaussian noise of variance alpha r on the local field.

  hopfield_rs(alpha, beta, *, branch="retrieval") -> HopfieldSolution
  hopfield_alpha_c(beta=None)                     -> alpha_c (T = 0, or at T = 1/beta)
  hopfield_t0(alpha, branch)                      -> HopfieldT0 (T = 0 erf reduction)
  hopfield_free_energy(m, q, r, alpha, beta)      -> the AGS RS functional f
  hopfield_gradient(m, q, r, alpha, beta)         -> its analytic gradient

RS free energy per spin (AGS PRL Eq. (5), one condensed pattern, C = beta(1 - q)):
  f = alpha/2 + m^2/2 + alpha/(2 beta) [ln(1 - C) - beta q/(1 - C)]
      + alpha beta r (1 - q)/2 - (1/beta) E_z ln 2 cosh Xi,   Xi = beta(m + sqrt(alpha r) z).
  Its gradient (derived here; the tests check it against numerical differentiation):
    df/dm = m - E tanh Xi,   df/dr = (alpha beta/2)(E tanh^2 Xi - q)   (Stein's lemma),
    df/dq = (alpha beta/2)(q/(1 - C)^2 - r),
  (the tests check these against numerical differentiation of f, and check
  numerically at each returned point that the gradient of f vanishes),
  so the stationary points are m = E tanh Xi, q = E tanh^2 Xi, r = q/(1 - C)^2 (AGS
  Eqs. (6)-(8)).  The domain needs 1 - C > 0 (the log); every returned point has it.

Branches (all three are stationary points of the same f; never substituted):
  "retrieval"  m > 0: the larger-m solution, the one AGS call locally stable in m
               (m-mode; see Stability for the replicon).
               Unknowns (m, sigma = sqrt(alpha r)); equations m = E tanh Xi and
               sigma (1 - C) = sqrt(alpha q); Newton with the analytic Jacobian
               (Stein: d/dsigma E h(Xi) = beta^2 sigma E h''(Xi)), seeded from the T = 0
               solution and from the Mattis state m = tanh(beta m).  If Newton finds
               nothing, alpha_c(T) is computed; above it the call refuses and says so,
               below it the upper branch is bracketed by the m-parametrisation below.
  "spinglass"  m = 0, q > 0: one equation in q, G(q) = E tanh^2(beta sigma(q) z) - q,
               sigma(q) = sqrt(alpha q)/(1 - C), bracketed between the edge of the domain
               and q = 1.  Exists iff T < T_g = 1 + sqrt(alpha) (linearising,
               q = beta^2 alpha q/(1 - beta)^2 + O(q^2)).
  "paramagnet" m = q = r = 0: only inside the domain for T > 1 (1 - C = 1 - beta).

Stability.  Three labels on every HopfieldSolution:
  mode_eig / mode_stable: the order-parameter mode of the branch.  Retrieval and spin
    glass: the m-mode, d^2 f_red/dm^2 = -det J / J22 with q, r held stationary at each m
    (J the Jacobian of the two retrieval equations in (m, sigma)); at m = 0 it equals
    1 - C, so the spin glass is always m-stable (AGS: 'stable to the development of
    finite overlaps').  Paramagnet: the q-mode, the linearisation slope of the spin-glass
    equation at q = 0, 1 - alpha beta^2/(1 - beta)^2 (computed, not the banked T_g).
  replicon: the AT eigenvalue of the RS saddle, AGS PRL Eq. (13),
    [1 - beta(1 - q)]^2 - alpha beta^2 E sech^4 Xi > 0, reported normalised as
    lambda_AT = 1 - alpha beta^2 E sech^4 Xi / (1 - C)^2 (same sign).  At q = 0 it equals
    the paramagnet's q-mode, so it vanishes at T_g; AGS find it negative for the spin
    glass at all T < T_g and for retrieval only at very low T (Steffan & Kuehn: the AT
    line meets the retrieval boundary just above T(alpha_max), so the reentrant sliver
    is RS-unacceptable).
  stable = mode_stable and replicon > 0.  AT-unstable RS points are still returned
  (this is the RS family) but never labelled stable.

Retrieval boundary alpha_c(T) (hopfield_alpha_c).  T = 0 (AGS Eqs. (9)-(10)): with
C = lim beta(1 - q), y = m/sqrt(2 alpha r): m = erf(y), C = (2/sqrt(pi)) (y/m) e^{-y^2},
r = (1 - C)^-2, which combine into y (sqrt(2 alpha) + (2/sqrt(pi)) e^{-y^2}) = erf(y),
i.e. sqrt(2 alpha) = g(y) = erf(y)/y - (2/sqrt(pi)) e^{-y^2}.  The m != 0 roots appear
as a double root, where g'(y) = 0: alpha_c = g(y*)^2/2 (mpmath; nothing hard-coded).
T = 0 energies (beta -> infinity of f): E = alpha (1 - r)/2 - m^2/2 (AGS Eq. (11)); the
spin glass has C = a/(1 + a), r = (1 + a)^2, a = sqrt(2/(pi alpha)).
Finite T: parametrise the retrieval curve by m.  For fixed m, sigma solves
m = E tanh(beta(m + sigma z)) (monotone in sigma), then alpha(m) = sigma^2 (1 - C)^2/q
is explicit; alpha_c(T) = max over m, located by a scan down from the Mattis value
and a bracketed root of d alpha/dm (implicit differentiation, analytic).  This is the
fold connected to the alpha = 0 Mattis state.

Numerics: composite Gauss-Legendre quadrature (below) at 20 digits (or the caller's
precision if higher); returned points satisfy the three equations to
< 1e-14 (re-evaluated after the solve) and the analytic gradient < 1e-8, or the call
refuses.  Quadrature: composite 24-point Gauss-Legendre on |z| < 9 with panel cuts at
the kink -m/sigma and at the tanh step width 1/(beta sigma), all moments in one pass.

Honest label: the n -> 0 continuation and the RS ansatz are physics-level, not proven
(RS is exact for the Hopfield model only at small alpha, Talagrand); numbers carry
quadrature error only.
"""
from dataclasses import dataclass
from functools import lru_cache

import mpmath as mp

from ..core import Refusal
from .common import CAVEAT

_DPS = 20
_BRANCHES = ("retrieval", "spinglass", "paramagnet")
_EQ_TOL = "1e-14"
_GRAD_TOL = "1e-8"


@dataclass(frozen=True)
class HopfieldSolution:
    m: object
    q: object
    r: object
    branch: str
    f: object = None
    stable: bool = None            # mode_stable and replicon > 0
    residual: object = None        # max |equation| over (m, q, r)
    grad: tuple = ()               # analytic (df/dm, df/dq, df/dr) at the point
    mode_stable: bool = None       # m-mode (retrieval, spin glass) or q-mode (paramagnet)
    mode_eig: object = None
    replicon: object = None        # lambda_AT = 1 - alpha beta^2 E sech^4 / (1 - C)^2
    note: str = ""
    caveat: str = CAVEAT


@dataclass(frozen=True)
class HopfieldT0:
    m: object
    r: object
    C: object
    y: object
    E: object
    branch: str
    caveat: str = CAVEAT


def _prec(dps=_DPS):   # never lower the caller's precision
    return mp.workdps(max(dps, mp.mp.dps))


def _mpf(x):
    return mp.mpf(repr(x) if isinstance(x, float) else x)


def _key(x):
    with mp.workdps(_DPS):
        return mp.nstr(_mpf(x), 30)


def _check_branch(branch, allowed=_BRANCHES):
    if branch not in allowed:
        raise Refusal("precondition:known-branch",
                      f"branch {branch!r} is not one of {', '.join(allowed)} "
                      "(the RS Hopfield branches with one condensed pattern)")


def _check_ab(alpha, beta):
    if not alpha > 0:
        raise Refusal("precondition:alpha-positive", f"alpha = p/N must be > 0, got {alpha}")
    if beta is not None and not beta > 0:
        raise Refusal("precondition:beta-positive", f"beta = 1/T must be > 0, got {beta}")


def _root(f, lo, hi, *, xtol="1e-18", maxit=300):
    """Bracketed root of f on [lo, hi] (f(lo) f(hi) < 0): Illinois regula falsi with
    a bisection step whenever the bracket fails to halve.  Stops when the bracket is
    narrower than xtol (1 + |x|) or f vanishes."""
    xtol = mp.mpf(xtol)
    flo, fhi = f(lo), f(hi)
    if flo == 0:
        return lo
    if fhi == 0:
        return hi
    if flo * fhi > 0:
        raise Refusal("no-solution:bracket", f"f({lo}) and f({hi}) have the same sign")
    side = 0
    for _ in range(maxit):
        width = hi - lo
        x = (lo * fhi - hi * flo) / (fhi - flo)
        if not lo < x < hi:
            x = (lo + hi) / 2
        fx = f(x)
        if fx == 0:
            return x
        if fx * flo < 0:
            hi, fhi = x, fx
            if side == -1:
                flo /= 2
            side = -1
        else:
            lo, flo = x, fx
            if side == 1:
                fhi /= 2
            side = 1
        if hi - lo > width / 2:          # slow progress: force a bisection
            mid = (lo + hi) / 2
            fm = f(mid)
            if fm == 0:
                return mid
            if fm * flo < 0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
            side = 0
        if hi - lo < xtol * (1 + abs(lo)):
            return (lo + hi) / 2
    raise Refusal("no-solution:bracket", f"no convergence in {maxit} steps on [{lo}, {hi}]")


# ---- Gaussian averages --------------------------------------------------------
_GL_N = 24        # Gauss-Legendre points per panel
_Z_MAX = 9        # |z| cut: Gaussian tail mass 2 Phi(-9) ~ 2e-19, below every tolerance


@lru_cache(maxsize=None)
def _gl_nodes(n, dps):
    """Gauss-Legendre nodes/weights on [-1, 1] at `dps` digits (Newton on P_n)."""
    with mp.workdps(dps + 10):
        out = []
        for k in range(1, n + 1):
            x = mp.cos(mp.pi * (k - mp.mpf(1) / 4) / (n + mp.mpf(1) / 2))
            for _ in range(100):
                p0, p1 = mp.mpf(1), x
                for j in range(2, n + 1):
                    p0, p1 = p1, ((2 * j - 1) * x * p1 - (j - 1) * p0) / j
                dp = n * (x * p1 - p0) / (x * x - 1)
                dx = p1 / dp
                x -= dx
                if abs(dx) < mp.mpf(10)**(-dps - 8):
                    break
            p0, p1 = mp.mpf(1), x
            for j in range(2, n + 1):
                p0, p1 = p1, ((2 * j - 1) * x * p1 - (j - 1) * p0) / j
            dp = n * (x * p1 - p0) / (x * x - 1)
            out.append((x, 2 / ((1 - x * x) * dp * dp)))
    return tuple((+x, +w) for x, w in out)


def _nodes(b, m, s):
    """[(z, w phi(z))] for E_z over |z| < 9: composite Gauss-Legendre with panel
    cuts at the kink z0 = -m/s and at z0 +- k/(b s) (k = 1, 2, 4, 8, 16), the width
    of the tanh step, then panels of length <= 2.  Checked against mp.quad at 40
    digits for beta up to 100 (errors 1e-17 to 1e-21; see the tests)."""
    L = mp.mpf(_Z_MAX)
    cuts = {-L, L}
    z0, w = -m / s, 1 / (b * s)
    for k in (0, 1, 2, 4, 8, 16):
        for sg in (1, -1):
            c = z0 + sg * k * w
            if -L < c < L:
                cuts.add(c)
    cuts = sorted(cuts)
    rule, c0, out = _gl_nodes(_GL_N, mp.mp.dps), 1 / mp.sqrt(2 * mp.pi), []
    for a, c in zip(cuts, cuts[1:]):
        k = int(mp.ceil((c - a) / 2))
        for i in range(k):
            lo, hi = a + (c - a) * i / k, a + (c - a) * (i + 1) / k
            hm, hw = (hi + lo) / 2, (hi - lo) / 2
            for x, wt in rule:
                z = hm + hw * x
                out.append((z, wt * hw * c0 * mp.exp(-z * z / 2)))
    return out


def _E(h, b, m, s):
    """E_z h(b (m + s z)), z ~ N(0, 1)."""
    if s == 0:
        return h(b * m)
    return mp.fsum(w * h(b * (m + s * z)) for z, w in _nodes(b, m, s))


def _moments(b, m, s):
    """(E t, E t^2, E t^3, E t^4), t = tanh(b (m + s z)), in one pass."""
    if s == 0:
        t = mp.tanh(b * m)
        return t, t**2, t**3, t**4
    acc = [mp.mpf(0)] * 4
    for z, w in _nodes(b, m, s):
        t = mp.tanh(b * (m + s * z))
        t2 = t * t
        acc[0] += w * t
        acc[1] += w * t2
        acc[2] += w * t2 * t
        acc[3] += w * t2 * t2
    if m == 0:
        acc[0] = acc[2] = mp.mpf(0)    # odd moments vanish by symmetry
    return tuple(acc)


def _lncosh2(x):    # ln 2cosh x, overflow-safe
    a = abs(x)
    return a + mp.log1p(mp.exp(-2 * a))


def hopfield_free_energy(m, q, r, alpha, beta):
    """The AGS RS functional f(m, q, r) per spin (PRL 55, 1530, Eq. (5))."""
    with _prec():
        m, q, r, a, b = (_mpf(v) for v in (m, q, r, alpha, beta))
        D = 1 - b * (1 - q)
        if not D > 0:
            raise Refusal("precondition:rs-domain",
                          f"1 - beta(1 - q) = {mp.nstr(D, 6)} <= 0: ln(1 - C) undefined")
        if r < 0:
            raise Refusal("precondition:rs-domain", f"r = {r} < 0")
        return (a / 2 + m**2 / 2 + a / (2 * b) * (mp.log(D) - b * q / D)
                + a * b * r * (1 - q) / 2 - _E(_lncosh2, b, m, mp.sqrt(a * r)) / b)


def hopfield_gradient(m, q, r, alpha, beta):
    """Analytic (df/dm, df/dq, df/dr) of hopfield_free_energy (module docstring)."""
    with _prec():
        m, q, r, a, b = (_mpf(v) for v in (m, q, r, alpha, beta))
        t1, t2 = _moments(b, m, mp.sqrt(a * r))[:2]
        D = 1 - b * (1 - q)
        return (m - t1, a * b / 2 * (q / D**2 - r), a * b / 2 * (t2 - q))


# ---- T = 0 reduction (AGS Eqs. (9)-(11)) ----------------------------------------
def _g(y):
    return mp.erf(y) / y - 2 / mp.sqrt(mp.pi) * mp.exp(-y * y)


@lru_cache(maxsize=None)
def _fold_t0(dps):
    """y* with g'(y*) = 0 (the double root of the T = 0 equation) and alpha_c."""
    with mp.workdps(dps):
        dg = lambda y: (2 / mp.sqrt(mp.pi) * mp.exp(-y * y) * (1 / y + 2 * y)
                        - mp.erf(y) / y**2)
        y = mp.findroot(dg, mp.mpf("1.5"))
        if not (mp.diff(_g, y, 2) < 0 and y > 0):
            raise Refusal("no-solution", f"T = 0 fold y* = {y} is not a maximum of g")
        return y, _g(y)**2 / 2


def _t0_dps():
    return max(30, mp.mp.dps)


def hopfield_t0(alpha, branch="retrieval") -> HopfieldT0:
    """T = 0 limit of the RS equations: retrieval (upper root) or spin glass."""
    _check_branch(branch, ("retrieval", "spinglass"))
    with _prec(30):
        a = _mpf(alpha)
        _check_ab(a, None)
        if branch == "spinglass":
            u = mp.sqrt(2 / (mp.pi * a))
            C, r = u / (1 + u), (1 + u)**2
            return HopfieldT0(mp.mpf(0), r, C, mp.mpf(0), a * (1 - r) / 2, branch)
        ys, ac = _fold_t0(_t0_dps())
        if a > ac:
            raise Refusal("no-solution:retrieval",
                          f"no retrieval solution at T = 0: alpha = {mp.nstr(a, 10)} above "
                          f"alpha_c = {mp.nstr(ac, 12)} (y(sqrt(2 alpha) + 2/sqrt(pi) e^-y^2)"
                          " = erf(y) has no root y > 0)")
        h = lambda y: mp.erf(y) - y * (mp.sqrt(2 * a) + 2 / mp.sqrt(mp.pi) * mp.exp(-y * y))
        y = ys if a == ac else _root(h, ys, 2 / mp.sqrt(2 * a), xtol=mp.mpf(10)**(-mp.mp.dps + 3))
        m = mp.erf(y)
        C = 2 / mp.sqrt(mp.pi) * (y / m) * mp.exp(-y * y)
        r = 1 / (1 - C)**2
        return HopfieldT0(m, r, C, y, a * (1 - r) / 2 - m**2 / 2, branch)


# ---- finite-T retrieval: equations, Jacobian, Newton -----------------------------
def _retrieval_system(a, b, m, s):
    """(E1, E2), Jacobian J (2x2) and t2 at (m, sigma).
    E1 = E tanh - m;  E2 = sigma (1 - b(1 - q)) - sqrt(a q),  q = E tanh^2."""
    t1, t2, t3, t4 = _moments(b, m, s)
    D = 1 - b + b * t2
    sq = mp.sqrt(a * t2)
    dt1_m, dt1_s = b * (1 - t2), -2 * b**2 * s * (t1 - t3)
    dt2_m, dt2_s = 2 * b * (t1 - t3), 2 * b**2 * s * (1 - 4 * t2 + 3 * t4)
    k = s * b - a / (2 * sq)            # d E2 / d q (through D and sqrt(a q))
    J = ((dt1_m - 1, dt1_s), (k * dt2_m, D + k * dt2_s))
    return (t1 - m, s * D - sq), J, t2, D


def _newton_retrieval(a, b, m, s, maxit=40):
    tol = mp.mpf(_EQ_TOL) / 100
    for _ in range(maxit):
        try:
            (e1, e2), J, t2, D = _retrieval_system(a, b, m, s)
        except (ValueError, ZeroDivisionError):
            return None
        res = max(abs(e1), abs(e2))
        if res < tol and D > 0:
            return m, s
        det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
        if det == 0:
            return None
        dm = -(J[1][1] * e1 - J[0][1] * e2) / det
        ds = -(-J[1][0] * e1 + J[0][0] * e2) / det
        lam = mp.mpf(1)
        while not (0 < m + lam * dm <= 1 and s + lam * ds > 0):
            lam /= 2
            if lam < mp.mpf("1e-6"):
                return None
        m, s = m + lam * dm, s + lam * ds
    return None


def _mattis(b):
    m = mp.findroot(lambda m: m - mp.tanh(b * m), mp.mpf(1))
    if not m > 0:
        raise Refusal("no-solution", f"Mattis root m = {m} at beta = {b}")
    return m


def _m_stability(a, b, m, s):
    """d^2 f_red / dm^2 = -det J / J22 (q, r stationary at each m)."""
    _, J, _, _ = _retrieval_system(a, b, m, s)
    return -(J[0][0] * J[1][1] - J[0][1] * J[1][0]) / J[1][1]


def _replicon(a, b, m, q, r):
    """lambda_AT = 1 - alpha beta^2 E sech^4 Xi / (1 - C)^2 (AGS PRL Eq. (13) / (1-C)^2)."""
    _, t2, _, t4 = _moments(b, m, mp.sqrt(a * r))
    return 1 - a * b**2 * (1 - 2 * t2 + t4) / (1 - b * (1 - q))**2


def _labels(lam_mode, lam_at):
    return dict(mode_stable=bool(lam_mode > 0), mode_eig=lam_mode, replicon=lam_at,
                stable=bool(lam_mode > 0 and lam_at > 0))


def _at_note(lam):
    return (f"replicon lambda_AT = {mp.nstr(lam, 6)}: "
            + ("RS locally stable" if lam > 0 else "RS AT-unstable (RSB needed; RS kept as RS)"))


# ---- finite-T fold: the retrieval curve parametrised by m ---------------------------
def _sigma_at_m(b, m):
    """sigma > 0 with E tanh(b(m + sigma z)) = m (requires m < Mattis value)."""
    E1 = lambda s: _E(mp.tanh, b, m, s) - m
    hi = mp.mpf(1)
    while E1(hi) > 0:
        hi *= 2
        if hi > 1e6:
            raise Refusal("no-solution", f"no sigma for m = {m}")
    lo = mp.mpf("1e-15")
    return _root(E1, lo, hi)


def _alpha_of_m(b, m):
    """(alpha(m), d alpha/dm) along the retrieval curve, or None outside the domain."""
    s = _sigma_at_m(b, m)
    t1, t2, t3, t4 = _moments(b, m, s)
    D = 1 - b + b * t2
    if not D > 0:
        return None
    alpha = s**2 * D**2 / t2
    dt1_m, dt1_s = b * (1 - t2), -2 * b**2 * s * (t1 - t3)
    dt2_m, dt2_s = 2 * b * (t1 - t3), 2 * b**2 * s * (1 - 4 * t2 + 3 * t4)
    ds_dm = -(dt1_m - 1) / dt1_s
    da_dt2 = s**2 * D * (2 * b * t2 - D) / t2**2
    da_dm = da_dt2 * dt2_m + (2 * s * D**2 / t2 + da_dt2 * dt2_s) * ds_dm
    return alpha, da_dm


@lru_cache(maxsize=64)
def _fold_finite(beta_key):
    with mp.workdps(_DPS):
        b = mp.mpf(beta_key)
        mm = _mattis(b)
        prev = None
        for c in ("0.995", "0.98", "0.96", "0.93", "0.88", "0.8", "0.65", "0.5", "0.3"):
            m = mm * mp.mpf(c)
            v = _alpha_of_m(b, m)
            if v is None:
                break
            if v[1] > 0:                    # below the fold: alpha rises with m
                if prev is None:
                    raise Refusal("no-solution", f"fold above m = {m}: scan too coarse")
                mf = _root(lambda x: _alpha_of_m(b, x)[1], m, prev, xtol="1e-12")
                return _alpha_of_m(b, mf)[0], mf
            prev = m
        raise Refusal("no-solution:retrieval-fold",
                      f"no maximum of alpha(m) found on the retrieval curve at beta = {b}")


def hopfield_alpha_c(beta=None):
    """Retrieval boundary: alpha_c at T = 0 (beta None) or alpha_c(T = 1/beta)."""
    if beta is None:
        with _prec(30):
            return +_fold_t0(_t0_dps())[1]
    with _prec():
        b = _mpf(beta)
        _check_ab(mp.mpf(1), b)
        if b <= 1:
            raise Refusal("no-solution:retrieval", _no_retrieval_hot(b))
        return +_fold_finite(_key(b))[0]


def _no_retrieval_hot(b):
    return (f"no retrieval solution at T = {mp.nstr(1 / b, 8)} >= 1: even at alpha -> 0 "
            "the Mattis equation m = tanh(beta m) has no root m > 0")


# ---- audits --------------------------------------------------------------------------
def _audit(a, b, m, q, r):
    """(residual of the three equations, gradient of f, f) at a candidate point.

    The averages are recomputed here, independently of the solver's last iterate;
    the gradient is the analytic one (module docstring), whose derivation the tests
    check against numerical differentiation of hopfield_free_energy."""
    t1, t2 = _moments(b, m, mp.sqrt(a * r))[:2]
    res = max(abs(t1 - m), abs(t2 - q), abs(q / (1 - b * (1 - q))**2 - r))
    g = hopfield_gradient(m, q, r, a, b)
    if res > mp.mpf(_EQ_TOL) or max(abs(x) for x in g) > mp.mpf(_GRAD_TOL):
        raise Refusal("no-solution:audit",
                      f"point (m, q, r) = {(m, q, r)} fails the audit: residual {res}, "
                      f"gradient {g}")
    return res, g, hopfield_free_energy(m, q, r, a, b)


# ---- branches ---------------------------------------------------------------------------
def _retrieval(a, b):
    if b <= 1:
        raise Refusal("no-solution:retrieval", _no_retrieval_hot(b))
    seeds = []
    try:
        t0 = hopfield_t0(a, "retrieval")
        seeds.append((t0.m, mp.sqrt(a * t0.r)))
    except Refusal:
        pass
    mm = _mattis(b)
    seeds += [(mm, mp.sqrt(a)), (mm * mp.mpf("0.97"), mp.sqrt(a) * mp.mpf("1.2"))]
    found = []
    for m0, s0 in seeds:
        p = _newton_retrieval(a, b, mp.mpf(m0), mp.mpf(s0))
        if p is not None and p[0] > mp.mpf("1e-8"):
            found.append(p)
            if _m_stability(a, b, *p) > 0:
                break        # the m-stable (upper) solution: stop seeding
    stable = [p for p in found if _m_stability(a, b, *p) > 0]
    if not stable:
        ac, mf = _fold_finite(_key(b))
        T = mp.nstr(1 / b, 8)
        if a > ac:
            raise Refusal("no-solution:retrieval",
                          f"no retrieval solution: alpha = {mp.nstr(a, 10)} above alpha_c(T) "
                          f"= {mp.nstr(ac, 10)} at T = {T} (the m > 0 branch ends in a fold)")
        mx = _mattis(b)       # bracket the upper branch: alpha(m) falls from alpha_c to 0
        m = _root(lambda x: _alpha_of_m(b, x)[0] - a, mf, mx * (1 - mp.mpf("1e-12")),
                  xtol="1e-12")
        p = _newton_retrieval(a, b, m, _sigma_at_m(b, m))
        if p is None or not _m_stability(a, b, *p) > 0:
            raise Refusal("no-solution:solver",
                          f"alpha = {mp.nstr(a, 10)} < alpha_c(T) = {mp.nstr(ac, 10)} at "
                          f"T = {T} but no stable retrieval point converged "
                          f"(found {len(found)} unstable)")
        stable = [p]
    m, s = stable[0]
    q = _moments(b, m, s)[1]
    r = s**2 / a
    res, g, f = _audit(a, b, m, q, r)
    lm, lat = _m_stability(a, b, m, s), _replicon(a, b, m, q, r)
    return HopfieldSolution(m, q, r, "retrieval", f, residual=res, grad=g,
                            note=f"larger-m retrieval solution, m-stable (d2f/dm2 = "
                                 f"{mp.nstr(lm, 6)} > 0); {_at_note(lat)}",
                            **_labels(lm, lat))


def _spinglass(a, b):
    tg = 1 + mp.sqrt(a)
    T = 1 / b
    qlo = max(mp.mpf(0), 1 - 1 / b)
    sig = lambda q: mp.sqrt(a * q) / (1 - b * (1 - q))
    G = lambda q: _E(lambda x: mp.tanh(x)**2, b, mp.mpf(0), sig(q)) - q
    eps = mp.mpf("1e-14")
    lo, hi = qlo + (1 - qlo) * eps, 1 - eps
    roots = []
    if G(lo) > 0 and G(hi) < 0:
        roots.append(_root(G, lo, hi))
    elif T < tg:
        raise Refusal("no-solution:solver",
                      f"T = {mp.nstr(T, 8)} < T_g = {mp.nstr(tg, 8)} but G(q) is not bracketed")
    else:   # T >= T_g: look for any (first-order) root on a grid before refusing
        grid = [qlo + (1 - qlo) * mp.mpf(10)**(-k) for k in range(12, 0, -1)]
        grid += [qlo + (1 - qlo) * mp.mpf(u) for u in ("0.3", "0.5", "0.7", "0.9", "0.99")]
        v = [G(q) for q in grid]
        if any(x * y < 0 for x, y in zip(v, v[1:])):
            raise Refusal("no-solution:solver",
                          f"T = {mp.nstr(T, 8)} >= T_g but G(q) changes sign: "
                          "unexpected first-order spin-glass root, not handled")
    if roots and roots[0] < mp.mpf("1e-13"):
        raise Refusal("no-solution:solver",
                      f"T = {mp.nstr(T, 8)} < T_g = {mp.nstr(tg, 8)} but the q root "
                      f"{mp.nstr(roots[0], 6)} is indistinguishable from q = 0")
    if not roots:
        raise Refusal("no-solution:spinglass",
                      f"no spin-glass solution (m = 0, q > 0) at T = {mp.nstr(T, 8)} >= "
                      f"T_g = 1 + sqrt(alpha) = {mp.nstr(tg, 8)}: only the paramagnet")
    q = roots[0]
    r = q / (1 - b * (1 - q))**2
    res, g, f = _audit(a, b, mp.mpf(0), q, r)
    lm, lat = 1 - b * (1 - q), _replicon(a, b, mp.mpf(0), q, r)
    return HopfieldSolution(mp.mpf(0), q, r, "spinglass", f, residual=res, grad=g,
                            note=f"m-mode d2f/dm2 = 1 - beta(1 - q) = {mp.nstr(lm, 6)}; "
                                 f"{_at_note(lat)}", **_labels(lm, lat))


def _paramagnet(a, b):
    if b >= 1:
        raise Refusal("no-solution:paramagnet",
                      f"at T = {mp.nstr(1 / b, 8)} <= 1 the point m = q = r = 0 is outside "
                      "the RS functional's domain: 1 - beta(1 - q) = 1 - beta <= 0 makes "
                      "ln(1 - beta(1 - q)) undefined")
    z = mp.mpf(0)
    res, g, f = _audit(a, b, z, z, z)
    lq = 1 - a * b**2 / (1 - b)**2            # q-mode: slope of the SG equation at q = 0
    lat = _replicon(a, b, z, z, z)
    return HopfieldSolution(z, z, z, "paramagnet", f, residual=res, grad=g,
                            note=f"q-mode 1 - alpha beta^2/(1 - beta)^2 = {mp.nstr(lq, 6)}: "
                                 + ("stable" if lq > 0 else "unstable, a spin-glass branch "
                                    "bifurcates (T below T_g)") + f"; {_at_note(lat)}",
                            **_labels(lq, lat))


@lru_cache(maxsize=128)
def _solve(a_key, b_key, branch):
    with mp.workdps(_DPS):
        a, b = mp.mpf(a_key), mp.mpf(b_key)
        return {"retrieval": _retrieval, "spinglass": _spinglass,
                "paramagnet": _paramagnet}[branch](a, b)


def hopfield_rs(alpha, beta, *, branch: str = "retrieval") -> HopfieldSolution:
    _check_branch(branch)
    with mp.workdps(_DPS):
        a, b = _mpf(alpha), _mpf(beta)
        _check_ab(a, b)
        return _solve(_key(a), _key(b), branch)
