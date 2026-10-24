"""Family (b): teacher-student learning curves of the spherical perceptron.

Convention.  Teacher T and student J on the sphere (T.T = J.J = N); P = alpha N
patterns with i.i.d. N(0, 1) components, noiseless labels sign(T.xi/sqrt N).
Gibbs learning: J uniform on the version space (zero training error).
Bayes: the version-space centre of mass, the Bayes-optimal predictor.
Generalization error eps = arccos(R)/pi, R = J.T/N (Buhot, Torres Moreno &
Gordon, arXiv:cond-mat/9703183, Eq. (6)).  H(x) = int_x^inf Dz.

RS with the teacher as an extra replica (overlaps q between students, R with
the teacher), n -> 0:
  s(q, R) = 1/2 ln(1-q) + (q - R^2)/(2(1-q))
            + 2 alpha E_t[ H(-R t / sqrt(q - R^2)) ln H(-sqrt(q) t / sqrt(1-q)) ]
(student fields sqrt(q) t + sqrt(1-q) z_a, teacher field R t/sqrt q +
sqrt(1 - R^2/q) w; the factor 2 is the two label signs).  Gibbs sampling is
Bayes-consistent with the teacher prior, so q = R (Nishimori); on that line
the two H arguments coincide and stationarity reduces to
  R / sqrt(1-R) = (alpha/pi) E_t[ exp(-R t^2/2) / H(-sqrt(R) t) ]
(Gyorgyi 1990; Seung, Sompolinsky & Tishby 1992; it is Buhot et al.'s
Eq. (15a) with R_Gibbs = R_Bayes^2).  tests/test_perceptron.py audits that
q = R is a stationary point of the two-variable s(q, R).
Bayes: R_Bayes = sqrt(R_Gibbs) (Opper & Haussler 1991; Buhot et al. (15a)).

Large alpha.  With R = 1 - d the right side tends to (alpha/pi) I,
I = E_t[exp(-t^2/2)/H(-t)], so d ~ pi^2/(alpha I)^2 and
  alpha eps_Gibbs -> sqrt(2)/I,   alpha eps_Bayes -> 1/I
(arccos(1-d) ~ sqrt(2d), arccos(sqrt(1-d)) ~ sqrt(d)); their ratio is sqrt 2.

Honest label: replica n -> 0 and RS (stable here: convex version space) are
physics-level, not proven in replicant.  Numerics: float64 bracket, mpmath
secant polish and residual audit at 15 digits (or the caller's, if higher).
"""
import math
from functools import lru_cache

import mpmath as mp

from ..core import Refusal
from . import _perceptron_float as F
from .perceptron import _bracket_root, _polish

_DPS = 15
_PTS = [-mp.inf, -6, 0, 6, mp.inf]


def _prec():
    return mp.workdps(max(_DPS, mp.mp.dps))


def _phi(x):
    return mp.exp(-x * x / 2) / mp.sqrt(2 * mp.pi)


def _H(x):
    return mp.erfc(x / mp.sqrt(2)) / 2


def _gauss(f, pts=_PTS):
    return mp.quad(lambda t: f(t) * _phi(t), pts)


def _alpha(a):
    a = mp.mpf(a)
    if not a >= 0:
        raise Refusal("precondition:nonnegative-alpha", f"alpha = {a}")
    return a


def _rhs_integral(R):
    """E_t[exp(-R t^2/2)/H(-sqrt(R) t)] = sqrt(2 pi) E_t[E(-sqrt(R) t)], E = phi/H."""
    s = mp.sqrt(R)
    return mp.sqrt(2 * mp.pi) * _gauss(lambda t: _phi(s * t) / _H(-s * t))


@lru_cache(maxsize=None)
def _gibbs_delta_cached(alpha_str, dps):
    with mp.workdps(dps):
        alpha = mp.mpf(alpha_str)

        def G(u):
            d = mp.exp(u)
            return mp.log((1 - d) / mp.sqrt(d)) - mp.log(alpha / mp.pi * _rhs_integral(1 - d))
        what = f"Gibbs saddle at alpha = {alpha_str}"
        u0 = _bracket_root(lambda u: F.gibbs_residual(u, float(alpha)), -60.0,
                           math.log(1 - 1e-12), what)
        u = _polish(G, u0, what)
        if abs(G(u)) > mp.mpf(10)**(-mp.mp.dps + 3):
            raise Refusal("no-solution", f"{what}: audit residual {mp.nstr(G(u), 3)}")
        return mp.exp(u)


def _gibbs_delta(alpha):
    """1 - R_Gibbs, solved in u = ln(1 - R) so that 1 - R keeps its digits."""
    return _gibbs_delta_cached(mp.nstr(alpha, mp.mp.dps), mp.mp.dps)


def gibbs_overlap(alpha):
    """Teacher-student overlap R of Gibbs learning at load alpha."""
    with _prec():
        a = _alpha(alpha)
        return mp.mpf(0) if a == 0 else 1 - _gibbs_delta(a)


def gibbs_functional(q, R, alpha):
    """s(q, R) above (needs R^2 < q < 1); used to audit the q = R saddle."""
    with _prec():
        q, R, a = mp.mpf(q), mp.mpf(R), _alpha(alpha)
        if not (R * R < q < 1):
            raise Refusal("precondition:order-in-range", f"need R^2 < q < 1, got q = {q}, R = {R}")
        u, v = R / mp.sqrt(q - R * R), mp.sqrt(q / (1 - q))
        en = 2 * a * _gauss(lambda t: _H(-u * t) * mp.log(_H(-v * t)))
        return mp.log(1 - q) / 2 + (q - R * R) / (2 * (1 - q)) + en


def _arccos_one_minus(d):
    """arccos(1 - d) without the cancellation at small d."""
    return 2 * mp.asin(mp.sqrt(d / 2))


def teacher_student_error(alpha, learning):
    if learning not in ("gibbs", "bayes"):
        raise Refusal("precondition:known-learning",
                      f"learning = {learning!r}: use 'gibbs' (version-space sample) or 'bayes'")
    with _prec():
        a = _alpha(alpha)
        if a == 0:
            return mp.mpf(1) / 2
        d = _gibbs_delta(a)
        if learning == "bayes":                       # R_B = sqrt(1 - d)
            d = d / (1 + mp.sqrt(1 - d))
        return _arccos_one_minus(d) / mp.pi


def asymptotic_constant(learning):
    """lim alpha * eps(alpha): sqrt(2)/I (Gibbs), 1/I (Bayes)."""
    if learning not in ("gibbs", "bayes"):
        raise Refusal("precondition:known-learning", f"learning = {learning!r}")
    with _prec():
        I = _rhs_integral(mp.mpf(1))
        return (mp.sqrt(2) if learning == "gibbs" else mp.mpf(1)) / I
