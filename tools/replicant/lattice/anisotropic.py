"""The anisotropic square-lattice Ising model in zero field (couplings K1 = beta J1 horizontal, K2 =
beta J2 vertical, J1, J2 > 0): its critical line, its free energy and Kramers-Wannier duality,
certified in ball arithmetic (python-flint arb through tools/baller; lattice/ball.py).

Sources read.
  Athanasopoulos & Ueltschi, Ann. Henri Poincare (2024), arXiv:2407.05303v2 (read, HTML), Theorem
  2.1(b) (the Kac-Ward solution; Houtappel 1950 for the triangular lattice, not fetched): for the triangular lattice with J1 (horizontal), J2 (vertical), J3 (north-east oblique),
  any signs, k3 = k1 + k2,
    -beta f = ln 2 + (1/(8 pi^2)) int_{[-pi, pi]^2} ln[C1 C2 C3 + S1 S2 S3 - sum_i S_i cos k_i] dk,
  C_i = cosh 2K_i, S_i = sinh 2K_i (the paper's f is per site, couplings absorbed in J); J3 = 0 is
  the anisotropic square lattice (their Eq. (2.7) is its isotropic case, Onsager's formula).
  Baxter, Exactly Solved Models (1982) (the ANU PDF, page images read): Eq. (6.2.14a) the dual
  couplings tanh K* = e^{-2L}, tanh L* = e^{-2K}; (6.2.11)-(6.2.15) the duality of the free energy;
  (6.2.17) the critical line sinh 2K sinh 2L = 1 (proved in Ch. 7: k = (sinh 2K sinh 2L)^-1 < 1, = 1,
  > 1 are T < T_c, T = T_c, T > T_c, p. 89); (11.7.15)-(11.7.16) the square-lattice free energy as
  a single integral of the diagonal-to-diagonal form, l^-1 = sinh 2K sinh 2K',
    -beta f_SQ(K, K') = (2 pi)^-1 int_0^pi ln{2[cosh 2K' cosh 2K + (1 + l^-2 - 2 l^-1 cos 2t)^(1/2)]} dt.

Derived here (and checked: tests/test_lattice_geom.py).  The inner angle of the Kac-Ward integral
in closed form (int_0^2pi ln(a - b cos t - c sin t) dt / 2pi = ln[(a + sqrt(a^2 - b^2 - c^2))/2],
a >= sqrt(b^2 + c^2)): with u = cos k1 and s = sin(k1/2),
  -beta f = ln 2 + (1/(2 pi)) int_0^pi ln[(a + sqrt D)/2] dk1,
  a = (alpha - S1) + 2 S1 s^2,  alpha = C1 C2 C3 + S1 S2 S3,
  D = a^2 - S2^2 - S3^2 - 2 S2 S3 u = D1 + 4 s^2 [S1 (alpha - S1) + S2 S3] + 4 S1^2 s^4,
  D1 = (alpha - S1 - S2 - S3)(alpha - S1 + S2 + S3),
a form with no difference of nearly equal terms inside the integrator's complex boxes (D1 is formed
once, on the real line; cf. lattice/onsager.py).  D1 = 0 is the critical surface: for J3 = 0,
(C1 C2)^2 - (S1 + S2)^2 = (S1 S2 - 1)^2, so D1 = 0 iff sinh 2K1 sinh 2K2 = 1, Baxter (6.2.17).
Duality in the -beta f convention (from Baxter (6.2.11), (6.2.12), (6.2.14a)):
  -beta f(K1*, K2*) = -beta f(K1, K2) + ln(2 cosh K1* cosh K2*) - K1 - K2,
  tanh K1* = e^{-2 K2}, tanh K2* = e^{-2 K1}.

  free_energy(beta, J1, J2) -> Ball          the Kac-Ward single integral (J3 = 0)
  free_energy_baxter(beta, J1, J2) -> Ball   Baxter (11.7.15), an independently derived integral
  criticality(beta, J1, J2) -> Ball          sinh 2K1 sinh 2K2 (= k^-1: > 1 below T_c, < 1 above)
  critical_coupling(K1) -> Ball              K2 on the critical line: (1/2) asinh(1 / sinh 2K1)
  critical_beta(J1, J2) -> Ball              the root of sinh(2 beta J1) sinh(2 beta J2) = 1 (a
                                             certified bisection; J1 = J2: (1/2) ln(1 + sqrt 2)/J)
  dual_couplings(beta, J1, J2) -> (Ball, Ball)  (K1*, K2*)
  duality_relation(beta, J1, J2) -> (Ball, Ball)  both sides of the free-energy duality
  integral3(K1, K2, K3), dual_coupling(K)     arb-level helpers shared with lattice/tri_honey.py
  certified_root(g, lo, hi) -> Ball           bisection of a monotone g (arb -> arb) on [lo, hi]
_HOOKS: "dual_mobius" (-1, 1, 1, 1): the duality map on x = e^{-2K}, x* = (p y + q)/(r y + s),
i.e. e^{-2K*} = (1 - e^{-2K})/(1 + e^{-2K}) = tanh K (planted: a wrong duality map, also seen by
lattice/tri_honey.py and lattice/geom_tori.py); "triple_sign" (+1): the sign of S1 S2 S3 in the
Kac-Ward integrand (planted 0: the triple product dropped, a wrong triangular lattice weight); "cross_coupling" (False:
planted True puts S1 where S2 belongs in D, invisible at J1 = J2); "dual_swapped" (False: planted
True pairs K1* with K1 instead of K2, Baxter (6.2.14a) read wrongly).
Memoisation (`memo`): Ball-returning functions of the I3 modules are cached on their arguments and
on the three I3 `_HOOKS` dicts.  Plant errors only through those dicts, never by monkeypatching a
function (a patched helper would not change the key); `clear_memo()` empties the cache.
"""
from fractions import Fraction

from ..core import Refusal
from .ball import Ball, arb_q, ball_of, baller, certify, rational

_HOOKS = {"dual_mobius": (-1, 1, 1, 1), "triple_sign": 1, "cross_coupling": False,
          "dual_swapped": False}
_MAX_PREC = 1 << 13
_MEMO = {}


def clear_memo():
    _MEMO.clear()


def memo(fn):
    """Memoise a function returning immutable Balls, keyed on its arguments and on every planted-
    error hook of the I3 modules (a plant is a different key; restoring it finds the old value)."""
    def hooks():
        from . import geom_tori, tri_honey
        return tuple(tuple(sorted(m._HOOKS.items())) for m in (geom_tori, tri_honey)) + \
            (tuple(sorted(_HOOKS.items())),)

    def wrapped(*args, **kw):
        key = (fn.__module__, fn.__name__, tuple(repr(a) for a in args),
               tuple(sorted((k, repr(v)) for k, v in kw.items())), hooks())
        if key not in _MEMO:
            _MEMO[key] = fn(*args, **kw)
        return _MEMO[key]
    wrapped.__name__, wrapped.__doc__, wrapped.uncached = fn.__name__, fn.__doc__, fn
    return wrapped


def couplings(beta, Js, positive=True):
    """[K_j = beta J_j] as Fractions; beta > 0; J_j > 0 (positive) or nonzero."""
    b = rational(beta, "beta")
    out = []
    for i, J in enumerate(Js, 1):
        q = rational(J, f"J{i}", positive=False)
        if positive and not q > 0:
            raise Refusal(f"precondition:J{i}", f"J{i} = {J!r}: need J{i} > 0 (the ferromagnet; "
                          "a bipartite lattice's antiferromagnet maps onto it by flipping one "
                          "sublattice)")
        if not positive and q == 0:
            raise Refusal(f"precondition:J{i}", f"J{i} = 0: need a nonzero coupling")
        out.append(b * q)
    return out


# ---- the free energy -------------------------------------------------------------------------
def integral3(K1, K2, K3):
    """-beta f of the Kac-Ward integrand with the inner angle in closed form (arb couplings)."""
    from flint import acb, arb
    C = (2 * K1).cosh() * (2 * K2).cosh() * (2 * K3).cosh()
    S1, S2, S3 = (2 * K1).sinh(), (2 * K2).sinh(), (2 * K3).sinh()
    if _HOOKS["cross_coupling"]:                       # planted: S1 where S2 belongs
        S2 = S1
    alpha = C + _HOOKS["triple_sign"] * S1 * S2 * S3
    a0 = alpha - S1
    D1 = (a0 - S2 - S3) * (a0 + S2 + S3)
    mid = 4 * (S1 * a0 + S2 * S3)
    q = 4 * S1 * S1

    def f(t, analytic):
        s = (t / 2).sin()
        s2 = s * s
        a = a0 + 2 * S1 * s2
        root = (D1 + s2 * (mid + q * s2)).sqrt(analytic=analytic)
        return ((a + root) / 2).log(analytic=analytic)

    return arb(2).log() + acb.integral(f, 0, arb.pi()).real / (2 * arb.pi())


def baxter_square(K, Kp):
    """Baxter (11.7.15) at arb couplings: -beta f_SQ(K, K'), the root written (1 - l^-1)^2 + 4 l^-1
    sin^2 t (= 1 + l^-2 - 2 l^-1 cos 2t)."""
    from flint import acb, arb
    li = (2 * K).sinh() * (2 * Kp).sinh()
    cc = (2 * K).cosh() * (2 * Kp).cosh()
    d = (1 - li) * (1 - li)

    def f(t, analytic):
        s = t.sin()
        return (2 * (cc + (d + 4 * li * s * s).sqrt(analytic=analytic))).log(analytic=analytic)

    return acb.integral(f, 0, arb.pi()).real / (2 * arb.pi())


@memo
def free_energy(beta, J1=1, J2=1) -> Ball:
    K1, K2 = couplings(beta, (J1, J2))
    return certify(lambda: integral3(arb_q(K1), arb_q(K2), arb_q(0)))


@memo
def free_energy_baxter(beta, J1=1, J2=1) -> Ball:
    K1, K2 = couplings(beta, (J1, J2))
    return certify(lambda: baxter_square(arb_q(K1), arb_q(K2)))


# ---- the critical line ---------------------------------------------------------------------
@memo
def criticality(beta, J1=1, J2=1) -> Ball:
    K1, K2 = couplings(beta, (J1, J2))
    return certify(lambda: (2 * arb_q(K1)).sinh() * (2 * arb_q(K2)).sinh())


@memo
def critical_coupling(K1) -> Ball:
    k = rational(K1, "K1")
    return certify(lambda: (1 / (2 * arb_q(k)).sinh()).asinh() / 2)


def _sign(g, x, prec=256):
    """Sign of g(x) at the rational x (escalating precision until decided)."""
    hyg = baller().hygiene
    while prec <= _MAX_PREC:
        with hyg.ctx_guard(prec=prec):
            v = g(arb_q(x))
            if v > 0:
                return 1
            if v < 0:
                return -1
        prec *= 2
    raise Refusal("no-solution:certification", f"sign of the critical condition undecided at {x}")


def certified_root(g, lo, hi, digits=40) -> Ball:
    """The root of g on [lo, hi] (exact rationals, g(lo), g(hi) of opposite signs) by bisection on
    certified signs: a ball of radius (hi - lo)/2 < 10^-(digits + 5) around the last midpoint.  A
    true enclosure of the root (unique when g is monotone, which the callers state)."""
    lo, hi = Fraction(lo), Fraction(hi)
    slo, shi = _sign(g, lo), _sign(g, hi)
    if slo == shi:
        raise Refusal("no-solution:bracket", f"the critical condition has the same sign at "
                      f"{float(lo):.6g} and {float(hi):.6g}")
    eps = Fraction(1, 10 ** (digits + 5))
    while hi - lo > eps:
        m = (lo + hi) / 2
        if _sign(g, m) == slo:
            lo = m
        else:
            hi = m
    with baller().hygiene.ctx_guard(prec=4 * digits + 128):
        return ball_of(arb_q(lo).union(arb_q(hi)))       # contains [lo, hi], hence the root


@memo
def critical_beta(J1=1, J2=1) -> Ball:
    j1, j2 = couplings(1, (J1, J2))
    if j1 == j2:
        return certify(lambda: (1 + arb_q(2).sqrt()).log() / 2 / arb_q(j1))
    g = lambda b: (2 * b * arb_q(j1)).sinh() * (2 * b * arb_q(j2)).sinh() - 1
    # monotone increasing in beta; sinh(2 b J1) sinh(2 b J2) >= 4 b^2 J1 J2 brackets the root
    hi = Fraction(1)
    while _sign(g, hi) < 0:
        hi *= 2
    return certified_root(g, Fraction(0), hi)


# ---- duality ---------------------------------------------------------------------------------
def dual_coupling(K):
    """K* (arb) with e^{-2K*} = M(e^{-2K}), M the hooked Moebius map (correct: tanh K)."""
    p, q, r, s = _HOOKS["dual_mobius"]
    y = (-2 * K).exp()
    return -((p * y + q) / (r * y + s)).log() / 2


def dual_pair(K1, K2):
    """(K1*, K2*) of Baxter (6.2.14a): tanh K1* = e^{-2 K2}, tanh K2* = e^{-2 K1} (arbs)."""
    if _HOOKS["dual_swapped"]:                          # planted: K1* from K1
        return dual_coupling(K1), dual_coupling(K2)
    return dual_coupling(K2), dual_coupling(K1)


@memo
def dual_couplings(beta, J1=1, J2=1):
    K1, K2 = couplings(beta, (J1, J2))
    return tuple(certify(lambda i=i: dual_pair(arb_q(K1), arb_q(K2))[i]) for i in (0, 1))


@memo
def duality_relation(beta, J1=1, J2=1):
    """(-beta f(K1*, K2*), -beta f(K1, K2) + ln(2 cosh K1* cosh K2*) - K1 - K2), certified."""
    K1, K2 = couplings(beta, (J1, J2))

    def lhs():
        d1, d2 = dual_pair(arb_q(K1), arb_q(K2))
        return integral3(d1, d2, arb_q(0))

    def rhs():
        k1, k2 = arb_q(K1), arb_q(K2)
        d1, d2 = dual_pair(k1, k2)
        return integral3(k1, k2, arb_q(0)) + (2 * d1.cosh() * d2.cosh()).log() - k1 - k2
    return certify(lhs), certify(rhs)


__all__ = ["baxter_square", "certified_root", "clear_memo", "couplings", "critical_beta", "critical_coupling",
           "criticality", "dual_coupling", "dual_couplings", "dual_pair", "duality_relation", "free_energy",
           "free_energy_baxter", "integral3"]
