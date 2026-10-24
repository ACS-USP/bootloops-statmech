"""Formal rules for the one-step RSB reduction (derive_rsb.py), each tested on its own in
tests/test_gauss_rsb.py.

  rsb_split(q0, q1, t, u, z)    lambda_a = sqrt(q0) t + sqrt(q1-q0) u_k + sqrt(1-q1) z_a for
                                replica a in block k (t shared by all replicas, u_k by the
                                block, z_a own): E l_a l_b = q1 in a block, q0 between blocks,
                                E l_a^2 = 1 (0 <= q0 <= q1 <= 1)
  block_quadratic(A0, A1, S, Sk, n)
                                sum_{a<b} A_ab s_a s_b = A0/2 S^2 + (A1-A0)/2 sum_k S_k^2
                                - n A1/2 for Ising s (s^2 = 1), A_ab = A1 in a block, A0
                                between, S = sum_a s_a, S_k the block sums
  rsb_logdet(d, c0, c1, m, n)   ln det of the n x n 1RSB matrix (d on the diagonal, c1 in a
                                block, c0 between): eigenvalues d - c1 (n - n/m times),
                                d - c1 + m (c1 - c0) (n/m - 1 times), d - c1 + m (c1 - c0) + n c0
  indicator_power(expr)         Theta(y)^p = Theta(y) for p > 0 (a 0/1 weight)
  abs_split(expr)               exp(c |y| + r) = exp(c y + r) Theta(y) + exp(-c y + r) Theta(-y)
  zero_temperature_power(expr, beta)
                                (e^{beta y} + e^{-beta y})^{mu/beta} -> e^{mu |y|} as
                                beta -> oo at fixed mu (pointwise; the power is bounded by
                                2^{mu/beta} e^{mu|y|}, so the limit passes under E and ln)
The exp-linear times step expectation E_z exp(B z + C) Theta(a z + b) is a case of
gauss.expect_closed.  Honest label: exact identities where their conditions hold; the
1RSB ansatz and the n -> 0 continuation that use them are physics-level.
"""
import sympy as sp

from .core import Refusal


def rsb_split(q0, q1, t, u, z):
    return sp.sqrt(q0) * t + sp.sqrt(q1 - q0) * u + sp.sqrt(1 - q1) * z


def block_quadratic(A0, A1, S, blocks, n):
    return A0 * S**2 / 2 + (A1 - A0) * sum(Sk**2 for Sk in blocks) / 2 - n * A1 / 2


def rsb_logdet(d, c0, c1, m, n):
    lam1, lam2 = d - c1, d - c1 + m * (c1 - c0)
    return (n - n / m) * sp.log(lam1) + (n / m - 1) * sp.log(lam2) + sp.log(lam2 + n * c0)


def indicator_power(expr):
    def one(p):
        if p.exp.is_positive is not True:
            raise Refusal("precondition:positive-exponent",
                          f"{p}: Theta^p = Theta needs p > 0 (got {p.exp})")
        return p.base
    return sp.sympify(expr).replace(
        lambda e: e.is_Pow and isinstance(e.base, sp.Heaviside), one)


def abs_split(expr):
    def one(e):
        arg = sp.expand(e.args[0])
        absv = [x for x in arg.atoms(sp.Abs)]
        if len(absv) != 1 or not arg.is_polynomial(absv[0]) or sp.degree(arg, absv[0]) != 1:
            raise Refusal("precondition:abs-linear-exponent", str(e))
        A = absv[0]
        c, r = arg.coeff(A, 1), arg.coeff(A, 0)
        y = A.args[0]
        return sp.exp(c * y + r) * sp.Heaviside(y) + sp.exp(-c * y + r) * sp.Heaviside(-y)
    return sp.sympify(expr).replace(lambda e: isinstance(e, sp.exp) and e.args[0].has(sp.Abs),
                                    one)


def _cosh_form(base):
    """A with base = exp(A) + exp(-A), or None."""
    if not base.is_Add or len(base.args) != 2 or \
            not all(isinstance(x, sp.exp) for x in base.args):
        return None
    A0, A1 = (x.args[0] for x in base.args)
    return A0 if sp.expand(A0 + A1) == 0 else None


def zero_temperature_power(expr, beta):
    def is_target(e):
        return e.is_Pow and _cosh_form(e.base) is not None and e.has(beta)

    def one(p):
        A = _cosh_form(p.base)
        mu = sp.simplify(p.exp * beta)
        y = sp.expand(A / beta)
        if mu.has(beta) or y.has(beta):
            raise Refusal("precondition:tempered-power",
                          f"{p}: the rule needs (e^(beta y) + e^(-beta y))^(mu/beta) with "
                          f"y, mu free of beta (got y = {y}, mu = {mu})")
        return sp.exp(mu * sp.Abs(sp.factor_terms(y)))
    return sp.sympify(expr).replace(is_target, one)
