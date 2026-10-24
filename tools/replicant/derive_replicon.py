"""Level (a): the replicon (de Almeida-Thouless) eigenvalue, where it is derivable.

  replicon(state) -> Replicon(expr | None, note)

`state` is the State of agree.derive_rs.  The n-replica functional is
  G = energy(q_ab) - sum_{a<b} qhat_ab q_ab + site(qhat_ab)
(derive.Functional).  Around the RS saddle, in the replicon sector
(delta q_ab with sum_b delta q_ab = 0) the second variation is
  E''_R dq^2 - 2 dq dqhat + chi_R dqhat^2,
with E''_R the replicon eigenvalue of the energy Hessian and chi_R that of the
site Hessian.  Eliminating dqhat (its own stationarity, dqhat = dq/chi_R) leaves
(E''_R - 1/chi_R) dq^2, so the RS saddle is replicon-stable iff
  lambda = 1 - E''_R chi_R > 0.
Two rules, each applied only where its precondition is checked:
  energy  pair-separable: energy = (terms free of q) + sum_{a<b} f(q_ab); the
          Hessian is diagonal, E''_R = f''(q) at the RS q.  (SK: f = beta^2 q^2/2,
          E''_R = beta^2.)  An energy holding q in any other form (a LnGaussProd:
          perceptron, Hopfield) is refused: its transverse Hessian needs a rule
          not implemented here.
  site    one Ising site trace (derive.IsingSiteRS, recorded by rs_ansatz):
          chi_R = P - 2Q + R of the connected 4-point function of s^a s^b in the
          replica measure, P = 1 - q^2, Q = q - q^2, R = r - q^2 with r = <s_a s_b
          s_c s_d>; at RS with Hubbard-Stratonovich (q = E tanh^2 u, r = E tanh^4 u)
          this is E_z[(1 - tanh^2 u)^2] = E_z[((ln single)''(u))^2], computed here
          by differentiating the recorded single-replica factor.  (SK:
          lambda = 1 - beta^2 E sech^4(beta(h + sqrt(q) z)), de Almeida-Thouless 1978.)
None means UNKNOWN (not derived), never "stable".
Honest label: physics-level (replica n -> 0, RS); the sign convention of lambda
is checked against families.sk.sk_replicon in tests/test_solve_derived.py.
"""
from dataclasses import dataclass

import sympy as sp

from . import derive as D
from .gauss import GaussE


@dataclass(frozen=True)
class Replicon:
    expr: object          # sympy expression in the RS unknowns and parameters, or None
    note: str


def _pair_f(energy, F):
    """f with energy = (q-free terms) + sum_{a<b} f(q_ab), else None + reason."""
    Q = sp.Dummy("Q", positive=True)
    f = sp.S(0)
    if energy.has(D.LnGaussProd) or energy.has(D.LogDet):
        return None, Q, "energy holds a formal Gaussian integral (LnGaussProd/LogDet)"
    for term in sp.Add.make_args(sp.expand(energy)):
        if not term.has(F.q):
            continue
        c, rest = term.as_independent(sp.Sum, as_Add=False)
        if not (isinstance(rest, sp.Sum)
                and rest.limits == ((D._b, D._a + 1, F.n), (D._a, 1, F.n))):
            return None, Q, f"energy term {term} is not a pair sum over a<b"
        body = (c * rest.function).xreplace({F.q[D._a, D._b]: Q})
        if body.has(F.q) or body.has(D._a) or body.has(D._b) or body.has(sp.Indexed):
            return None, Q, f"pair-sum summand {rest.function} is not a function of q_ab alone"
        f += body
    return f, Q, ""


def replicon(state) -> Replicon:
    F, P = state["functional"], state["rs_free"]
    if not F.conjugate or len(P.sites) != 1:
        return Replicon(None, "replicon not derived for this family: needs exactly one "
                              "Ising site trace with its conjugate qhat")
    f, Q, why = _pair_f(F.energy, F)
    if f is None:
        return Replicon(None, f"replicon not derived for this family: {why} "
                              "(only pair-separable energies have a replicon rule)")
    site = P.sites[0]
    U = sp.Dummy("U", real=True)
    d2 = sp.diff(sp.log(site.single(U)), U, 2).xreplace({U: site.u})
    chi = GaussE(d2**2, site.z)
    e2 = sp.diff(f, Q, 2).xreplace({Q: D.Q_RS})
    note = (f"replicon lambda = 1 - E''_R chi_R: pair-separable energy, E''_R = f''(q) = {e2}; "
            f"Ising site chi_R = E_z[((ln single)''(u))^2], u = {site.u}")
    return Replicon(1 - e2 * chi, note)
