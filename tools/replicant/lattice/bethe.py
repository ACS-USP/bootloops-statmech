"""The ferromagnetic Ising model on the Bethe lattice / random k-regular graph: every BP fixed point,
certified, with its free energy and its computed stability.

Conventions (Dembo & Montanari, Ann. Appl. Probab. 20 (2010) 565, arXiv:0804.4726, Eqs. (1.1),
(2.5)-(2.9)): mu(x) ~ exp(beta sum_<ij> x_i x_j + B sum_i x_i), J = 1, coordination k (every vertex
has k neighbours; branching k - 1), field h in units of J, so B = beta h.  NOTE: viana_bray's k
(cavity/sparse.py) is the excess degree, k_vb = k - 1.
  cavity field (dimensionless):  x = B + (k - 1) xi(x),  xi(x) = atanh(t tanh x),  t = tanh beta;
  full field X = B + k xi(x),  magnetization m = tanh X;
  -beta f (Bethe; the link/site form of cavity/sparse.py, MP 2001):
      phi = (k/2) ln[e^beta 2 cosh 2x + 2 e^-beta] - (k - 1) ln 2 cosh X;
  phi_dm (Dembo-Montanari Eq. (2.9), regular case): (k/2) ln cosh beta - (k/2) ln(1 + t tanh^2 x)
      + ln[e^B (1 + t tanh x)^k + e^-B (1 - t tanh x)^k]  (the two forms must agree: an audit);
  BP stability of the fixed point (homogeneous linearisation; on a random k-regular graph the
  non-backtracking operator has spectral radius k - 1): (k - 1) xi'(x) < 1, certified in arb.
  critical point (DM Eq. (2.5)): (k - 1) tanh beta_c = 1.
All fixed points.  F(x) = B + (k - 1) xi(x) - x has F' = (k - 1) xi' - 1 with xi' even and decreasing
in |x|, so F' < 0 everywhere when (k - 1) t <= 1 (one root), and otherwise F' = 0 exactly at +- x*,
tanh^2 x* = ((k - 1) t - 1) / (t (k - 1 - t)): F decreases, increases on (-x*, x*), decreases.  The
signs of F(-x*), F(x*) (certified) say which of the three monotone intervals hold a root (|root| <=
|B| + (k - 1) atanh t); each root is bracketed by bisection and certified by interval Newton
(Moore's test N(X) = m - F(m)/F'(X) inside int X: a unique zero in X), which with monotonicity on
its interval makes the list complete.  baller.certify does not apply: its Krawczyk engines are the
model-specific pipe_vac and the block-arrow block_krawczyk (needs a symmetric system with a border).
Selection (bethe_ising_free_energy): branch given -> that fixed point; h > 0 -> "upper", h < 0 ->
"lower" (DM Lemma 2.3 / Thm 2.4: the unique fixed point on [0, oo) for B > 0, and phi(beta, B) =
phi(beta, -B)); a single fixed point -> it; h = 0 with three: phi is DM's phi(beta, 0) = lim_{B -> 0}
phi(beta, B), the common value of "lower" and "upper", and the magnetization is refused
(ambiguous:branch) until a branch is given.

  bethe_critical_beta(k) -> Ball
  bethe_ising_fixed_points(k, beta, h) -> (BetheFixedPoint, ...)   branches "unique" or
      "lower" < "middle" < "upper"
  bethe_ising_free_energy(k, beta, h, branch=None) -> BetheResult
_HOOKS["link_weight"] (2, the 2 cosh 2x of the link term): planted-error hook; a wrong value makes
the phi / phi_dm audit refuse no-solution:audit.
"""
from dataclasses import dataclass

from ..core import Refusal
from .ball import Ball, arb_q, ball_of, baller, certify, rational

DIGITS = 40
_HOOKS = {"link_weight": 2}        # planted-error hook of the link term (tests)
_PREC0, _MAX_PREC = 192, 1 << 12


def _k(k):
    if isinstance(k, bool) or not isinstance(k, int) or k < 2:
        raise Refusal("precondition:coordination", f"k = {k!r}: need an int coordination >= 2")
    return k


def bethe_critical_beta(k) -> Ball:
    k = _k(k)
    return certify(lambda: (1 / arb_q(k - 1)).atanh())


@dataclass(frozen=True)
class BetheFixedPoint:
    branch: str
    cavity_field: Ball        # x (dimensionless)
    magnetization: Ball
    phi: Ball                 # -beta f, link/site form
    phi_dm: Ball              # -beta f, Dembo-Montanari Eq. (2.9)
    jacobian: Ball            # (k - 1) xi'(x)
    stable: bool              # jacobian < 1, certified


@dataclass(frozen=True)
class BetheResult:
    k: int
    beta: object
    h: object
    fixed_points: tuple
    selected: BetheFixedPoint | None
    phi: Ball

    @property
    def magnetization(self):
        if self.selected is None:
            raise Refusal("ambiguous:branch", f"h = 0 below T_c: fixed points "
                          f"{', '.join(f.branch for f in self.fixed_points)}; the magnetization "
                          "depends on the branch (+-m), pass branch='lower' or 'upper'")
        return self.selected.magnetization


class _Model:
    def __init__(self, k, beta, B):
        from flint import arb
        self.k, self.b, self.B = k, arb_q(beta), arb_q(B)
        self.t = self.b.tanh()

    def xi(self, x):
        return (self.t * x.tanh()).atanh()

    def dxi(self, x):
        th = x.tanh()
        return self.t * (1 - th * th) / (1 - self.t * self.t * th * th)

    def F(self, x):
        return self.B + (self.k - 1) * self.xi(x) - x

    def dF(self, x):
        return (self.k - 1) * self.dxi(x) - 1


def _bisect(m, lo, hi, sign_lo):
    """Root of m.F on [lo, hi] (F(lo) has sign sign_lo, F(hi) the other), to the precision."""
    from flint import arb
    for _ in range(_PREC0 + 64):                # the bracket halves; it starts below 2^64
        mid = (lo + hi) / 2
        f = m.F(mid)
        if f.contains(0) or (hi - lo).upper() < arb(2) ** (-_PREC0):
            return mid.mid()
        if (f > 0) == (sign_lo > 0):
            lo = mid
        else:
            hi = mid
    return ((lo + hi) / 2).mid()


def _newton(m, x0):
    """Interval-Newton certificate around the point x0: an arb X containing the unique zero."""
    from flint import arb
    x = arb(x0)
    for _ in range(8):                          # polish the point at this precision
        x = (x - m.F(x) / m.dF(x)).mid()
    r = arb(2) ** (-(_ctx_prec() // 2))
    for _ in range(40):
        X = arb(x, r)
        N = x - m.F(arb(x)) / m.dF(X)
        if X.contains_interior(N):
            return N
        r *= 4
    raise Refusal("no-solution:certification", f"interval Newton did not contract around {x0}")


def _ctx_prec():
    from flint import ctx
    return ctx.prec


def _fixed_points_at(k, beta, B):
    """[(branch, X)] at the current precision; None when a sign is undecided."""
    from flint import arb
    m = _Model(k, beta, B)
    R = abs(m.B) + (k - 1) * m.t.atanh() + 1
    q = (k - 1) * m.t - 1
    if q <= 0:
        x0 = _bisect(m, -R, R, 1)
        return [("unique", _newton(m, x0))], m
    if not q > 0:
        return None, m
    xs = (q / (m.t * (k - 1 - m.t))).sqrt().atanh()
    fl, fu = m.F(-xs), m.F(xs)
    if fl.contains(0) or fu.contains(0):
        return None, m
    roots = []
    if fl < 0:
        roots.append(("lower", _newton(m, _bisect(m, -R, -xs, 1))))
        if fu > 0:
            roots.append(("middle", _newton(m, _bisect(m, -xs, xs, -1))))
    if fu > 0:
        roots.append(("upper", _newton(m, _bisect(m, xs, R, 1))))
    if len(roots) == 1:
        roots = [("unique", roots[0][1])]
    return roots, m


def _observables(m, x):
    from flint import arb
    k, b, t = m.k, m.b, m.t
    X = m.B + k * m.xi(x)
    zl = b.exp() * _HOOKS["link_weight"] * (2 * x).cosh() + 2 * (-b).exp()
    phi = arb(k) / 2 * zl.log() - (k - 1) * (2 * X.cosh()).log()
    th = x.tanh()
    dm = (arb(k) / 2 * b.cosh().log() - arb(k) / 2 * (1 + t * th * th).log()
          + (m.B.exp() * (1 + t * th) ** k + (-m.B).exp() * (1 - t * th) ** k).log())
    return X.tanh(), phi, dm, (k - 1) * m.dxi(x)


def bethe_ising_fixed_points(k, beta, h):
    k = _k(k)
    bq, hq = rational(beta, "beta"), rational(h, "h", positive=False)
    hyg = baller().hygiene
    prec = _PREC0
    while prec <= _MAX_PREC:
        with hyg.ctx_guard(prec=prec):
            roots, m = _fixed_points_at(k, bq, bq * hq)
            if roots is not None:
                out = []
                for name, x in roots:
                    if hq == 0 and name in ("middle", "unique") and x.contains(0):
                        from flint import arb
                        x = arb(0)                  # F(0) = 0 exactly at B = 0 (odd F)
                    mag, phi, dm, jac = _observables(m, x)
                    balls = [ball_of(v) for v in (x, mag, phi, dm, jac)]
                    if jac < 1:
                        stable = True
                    elif jac > 1:
                        stable = False
                    else:
                        balls = None
                    if balls is None or min(bb.digits for bb in balls) < DIGITS:
                        break
                    if not balls[2].overlaps(balls[3]):
                        raise Refusal("no-solution:audit", f"Bethe phi {balls[2]} and Dembo-Montanari "
                                      f"(2.9) {balls[3]} disagree at the {name} fixed point")
                    out.append(BetheFixedPoint(name, *balls, stable))
                else:
                    return tuple(out)
        prec *= 2
    raise Refusal("undetermined:spinodal", f"k = {k}, beta = {beta}, h = {h}: the fixed-point count or "
                  f"a stability sign is undecided at {_MAX_PREC} bits (at a spinodal or beta_c?)")


def bethe_ising_free_energy(k, beta, h, branch=None) -> BetheResult:
    fps = bethe_ising_fixed_points(k, beta, h)
    names = [f.branch for f in fps]
    hq = rational(h, "h", positive=False)
    if branch is not None:
        if branch not in names:
            raise Refusal("precondition:branch", f"branch = {branch!r}: the fixed points here are "
                          f"{', '.join(names)}")
        sel = fps[names.index(branch)]
    elif len(fps) == 1:
        sel = fps[0]
    elif hq != 0:
        sel = fps[names.index("upper" if hq > 0 else "lower")]
    else:
        sel = None
    phi = (sel or fps[names.index("upper")]).phi
    return BetheResult(k, beta, h, fps, sel, phi)


__all__ = ["BetheFixedPoint", "BetheResult", "bethe_critical_beta", "bethe_ising_fixed_points",
           "bethe_ising_free_energy"]
