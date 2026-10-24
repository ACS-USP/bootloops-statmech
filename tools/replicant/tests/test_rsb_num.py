"""The nested-expectation evaluator of the 1RSB solver (_rsb_num.py): E_t F(t, E_u f(t, u))
by product quadrature, three rules (main Gauss-Hermite, independent composite
Gauss-Legendre, coarse Gauss-Hermite for the search), checked against closed forms.
"""
import math
import os
import sys

import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant._rsb_num import NestedNum
from replicant.core import Refusal
from replicant.gauss import GaussE

t, u, w = sp.symbols("t u w", real=True)
a, b = sp.symbols("a b", positive=True)


def test_nested_expectation_matches_closed_form():
    # E_t ln E_u exp(a t + b u) = E_t (a t + b^2/2) = b^2/2; E_t (E_u exp(a t u))^2 =
    # E_t exp(a^2 t^2) = 1/sqrt(1 - 2 a^2)
    e1 = GaussE(sp.log(GaussE(sp.exp(a * t + b * u), u)), t)
    e2 = GaussE(GaussE(sp.exp(a * t * u), u)**2, t) + 3 * GaussE(sp.cos(a * w), w)
    for rule in ("main", "audit", "fast"):
        n1, n2 = NestedNum(e1, [a, b]), NestedNum(e2, [a, b])
        tol = 1e-12 if rule != "fast" else 1e-9
        assert abs(n1.value([0.3, 0.7], rule) - 0.7**2 / 2) < tol, rule
        exact = 1 / math.sqrt(1 - 2 * 0.3**2) + 3 * math.exp(-0.3**2 / 2)
        assert abs(n2.value([0.3, 0.7], rule) - exact) < tol, rule
    # the two rules are different: their difference on a sharp integrand is not 0
    sharp = NestedNum(GaussE(sp.tanh(40 * t)**2, t), [])
    assert sharp.value([], "main") != sharp.value([], "audit")


def test_nested_depth_and_points_are_checked():
    deep = GaussE(GaussE(GaussE(sp.exp(t * u * w), w), u), t)
    try:
        NestedNum(deep, [])
    except Refusal as r:
        assert r.reason == "precondition:nesting-depth"
    else:
        raise AssertionError("expected Refusal")
    n = NestedNum(GaussE(sp.log(GaussE(sp.exp(a * u), u) - 2), t), [a])
    try:                                     # log of a negative number: not a real value
        n.value([0.1], "main")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
