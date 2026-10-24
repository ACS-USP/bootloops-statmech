"""Level (a) hooks for the perceptron (spherical or Ising couplings) and Hopfield.

Registered into derive.FAMILIES on import (derive.py imports this module last).
Each hook is split into what is HAND-SUPPLIED (the model: its constraint or
energy per pattern, the condensed-pattern term) and the pattern average, which
the hook ASSERTS and writes down as a formal LnGaussProd term (the Gaussian-law
statement is not mechanised: no step proves it, the hook builds the term); the
RS ansatz, n -> 0 and stationarity are then the generic steps of derive.py.
`_perceptron_formal` / `_hopfield_formal` only build the displayed formal Z^n
(the trace's `Zn`); nothing downstream reads them.  Only symbolic n.

Perceptron, spec.params alpha, kappa; Gaussian patterns; J stores pattern mu if
J.xi^mu/sqrt(N) >= kappa (families/perceptron.py conventions).  spec.variables:
  "spherical"  J.J = N (Gardner 1988): site derive.SITES["spherical"], the
               hand-supplied volume (1/2) ln det Q, no conjugate;
  "ising"      J in {-1, +1}^N (Gardner-Derrida 1988, Krauth-Mezard 1989): site
               derive.SITES["ising"], the same trace SK uses,
               ln Tr_J exp(sum_{a<b} qhat_ab J^a J^b) with -sum_{a<b} qhat_ab q_ab
               (q_aa = 1 because J_i^2 = 1); rs_ansatz reduces it by
               Hubbard-Stratonovich to -qhat/2 ... -> -qhat(1-q)/2 + E ln 2cosh.
  The hook below does not look at spec.variables: both use the same pattern term.
  NOT HELD OUT: the Ising perceptron is derived and checked against Task 8 here;
  Task 16 must check a composed Ising perceptron by a route independent of Task 8
  (S = 0 at alpha ~ 0.833 from Krauth-Mezard, the RS endpoint 4/pi, or small-N
  enumeration).
  hand-supplied: the storage constraint g(lambda) = Theta(lambda - kappa) (and,
    spherical only, the volume).
  hook asserts (rule not mechanised): lambda_a = J^a.xi/sqrt(N) is Gaussian with
    covariance Q_ab = J^a.J^b/N (Q_aa = 1 for both sites),
    exactly, for Gaussian xi at fixed J; P = alpha N independent patterns give
    (1/N) ln E V^n = site + alpha ln E_lambda prod_a g(lambda_a)
    (the formal LnGaussProd(g, q_ab, n)).
Hopfield (Amit-Gutfreund-Sompolinsky 1987), spec.params alpha, beta; +-1 patterns:
  H = -(1/2N) sum_mu (sum_i xi_i^mu s_i)^2 (i = j terms kept: a constant -p/2).
  hand-supplied: the condensed pattern, gauged to xi^1 = +1 (exact for +-1) and
    decoupled by exp(N beta m^2/2) = int dm exp(N(-beta m^2/2 + beta m m_true)) at
    the saddle: -(beta/2) sum_a m_a^2 and the site field beta m_a; and the energy of
    one uncondensed pattern, g(y) = exp(beta y^2 / 2) per replica.
  computed: the pattern law's cumulant series (derive.LAWS["pm_one"]) is checked
    to have mean 0 and variance 1/N per entry, and its dropped fourth cumulant and
    order are stated.  hook asserts (rule not mechanised): with the series kept to
    second order, y_a = sum_i xi_i s_i^a / sqrt(N) is Gaussian with covariance
    Q_ab, and p - 1 ~ alpha N patterns give alpha LnGaussProd(g, q_ab, n).
"""
from dataclasses import dataclass

import sympy as sp

from . import derive as D
from .core import Refusal
from .spec import ModelSpec

ALPHA = sp.Symbol("alpha", positive=True)
KAPPA = sp.Symbol("kappa", real=True)
_LAM = sp.Symbol("lambda", real=True)
_i, _mu = sp.symbols("i mu", integer=True, positive=True)


@dataclass(frozen=True)
class PatternAverage:
    """alpha * LnGaussProd(weight, q_ab, n) per spin/coupling, plus what was dropped."""
    spec: ModelSpec
    n: object
    alpha: sp.Expr
    weight: sp.Lambda
    note: str
    dropped: sp.Expr = sp.S(0)


def _symbolic_n(zn):
    if not isinstance(zn.n, sp.Symbol):
        raise Refusal("precondition:symbolic-replica-number",
                      f"{zn.spec.family}: derived at symbolic n only (formal Gaussian "
                      "integrals over replicas; no explicit-n expansion)")


def _no_annealed(spec, N):
    raise Refusal("not-implemented:annealed",
                  f"{spec.family}: exact ln E Z at finite N is not implemented")


# ---- perceptron ---------------------------------------------------------------------

def _perceptron_formal(n, beta, spec=None):
    """Display only: the formal V^n stored as the trace's Zn."""
    kappa = spec.params.get("kappa", KAPPA) if spec else KAPPA
    J, xi = sp.IndexedBase("J"), sp.IndexedBase("xi")
    field = sp.Sum(J[_i, D._a] * xi[_mu, _i], (_i, 1, D.N_SYM)) / sp.sqrt(D.N_SYM)
    return sp.Function("Tr_J")(sp.Product(sp.Heaviside(field - kappa),
                                          (D._a, 1, n), (_mu, 1, sp.Symbol("P"))))


def _perceptron_average(zn) -> PatternAverage:
    _symbolic_n(zn)
    p = zn.spec.params
    kappa, alpha = p.get("kappa", KAPPA), p.get("alpha", ALPHA)
    g = sp.Lambda(_LAM, sp.Heaviside(_LAM - kappa))       # hand-supplied constraint
    note = ("Gaussian patterns (hook asserts, not mechanised): lambda_a = J^a.xi/sqrt(N) "
            "jointly Gaussian, covariance Q_ab = J^a.J^b/N (exact at fixed J); "
            "P = alpha N iid patterns -> "
            f"alpha ln E_lambda prod_a g(lambda_a), g = {g.expr} (hand-supplied)")
    return PatternAverage(zn.spec, zn.n, alpha, g, note)


def _gauss_prod_reduction(avg, n, extra_energy=0, field=0, extra=None, missing=None):
    q, _, _ = D._overlap_symbols(n)
    energy = extra_energy + avg.alpha * D.LnGaussProd(avg.weight, q[D._a, D._b], n)
    return D.Reduction(energy, sp.S(0), missing, field=field, extra=extra)


def _perceptron_order_parameters(avg, n, sum_pairs):
    return _gauss_prod_reduction(avg, n, missing="O(ln N / N) (saddle-point prefactor)")


# ---- Hopfield -----------------------------------------------------------------------

def _hopfield_formal(n, beta, spec=None):
    """Display only: the formal Z^n stored as the trace's Zn."""
    s, xi = sp.IndexedBase("s"), sp.IndexedBase("xi")
    y = sp.Sum(xi[_mu, _i] * s[_i, D._a], (_i, 1, D.N_SYM))
    return D.Tr(sp.exp(beta / (2 * D.N_SYM)
                       * sp.Sum(y**2, (_mu, 1, sp.Symbol("p")), (D._a, 1, n))))


def _hopfield_average(zn) -> PatternAverage:
    _symbolic_n(zn)
    law = D._law(zn.spec)
    beta = D._beta(zn.spec)
    alpha = zn.spec.params.get("alpha", ALPHA)
    ser = sp.expand(sp.series(law.cgf(D.T, D.N_SYM), D.T, 0, 5).removeO())
    if ser.coeff(D.T, 1) != 0 or sp.simplify(ser.coeff(D.T, 2) * D.N_SYM - sp.Rational(1, 2)) != 0:
        raise Refusal("precondition:unit-variance-patterns",
                      f"{law.name}: pattern entries need mean 0, variance 1 (K = {ser})")
    dropped = ser.coeff(D.T, 4) * D.T**4
    g = sp.Lambda(_LAM, sp.exp(beta * _LAM**2 / 2))      # hand-supplied energy per pattern
    note = ("condensed pattern gauged to +1 and decoupled by m_a (hand-supplied); "
            "uncondensed: y_a = sum_i xi_i s_i^a/sqrt(N), pattern cumulant kept to "
            f"t^2/(2N), dropped {dropped} per site and pattern -> O(1/N) per spin; "
            "then (hook asserts, not mechanised) y ~ N(0, Q), alpha N patterns -> "
            f"alpha ln E_y prod_a g(y_a), g = {g.expr}")
    return PatternAverage(zn.spec, zn.n, alpha, g, note, dropped)


def _hopfield_order_parameters(avg, n, sum_pairs):
    beta = D._beta(avg.spec)
    m = sp.IndexedBase("m", real=True)
    condensed = -beta / 2 * sp.Sum(m[D._a]**2, (D._a, 1, n))
    return _gauss_prod_reduction(avg, n, extra_energy=condensed, field=beta * m[D._a],
                                 extra={"m": m}, missing="O(1/N) per spin (dropped cumulant)")


D.FAMILIES["perceptron"] = D.Family(
    _perceptron_formal, _no_annealed, lambda d: d == "gaussian",
    _perceptron_average, _perceptron_order_parameters, variables=("spherical", "ising"),
    constraints=("margin kappa",))
D.FAMILIES["hopfield"] = D.Family(
    _hopfield_formal, _no_annealed, lambda d: d == "pm_one",
    _hopfield_average, _hopfield_order_parameters, variables=("ising",))
