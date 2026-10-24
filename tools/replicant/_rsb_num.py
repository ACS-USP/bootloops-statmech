"""Float64 evaluation of derived 1RSB expressions with nested expectations (helper of
solve_derived_rsb.py).

  NestedNum(expr, args).value(vals, rule)   expr holding GaussE atoms nested at most two
                                            deep, E_t F(t, E_u f(t, u), ...), at vals
  rules (product quadrature, the same rule in t and u, each times the Gaussian density):
    "main"   composite Gauss-Legendre, MAIN_PANELS panels of MAIN_ORDER nodes on
             [-MAIN_CUT, MAIN_CUT]
    "audit"  trapezoid with step AUDIT_STEP on [-AUDIT_CUT, AUDIT_CUT]: a different rule with
             different nodes (spectrally accurate for analytic integrands of Gaussian
             decay), so main - audit measures quadrature error
    "fast"   probabilists' Gauss-Hermite with FAST_NODES nodes, for the root search only.
  Gauss-Hermite is NOT used for values: the 1RSB integrands (2 cosh U)^x have complex
  singularities close to the real axis (U = i pi/2 at |Im t| ~ pi/(2 sqrt(qhat))), and a
  200-node Gauss-Hermite rule was measured 1e-8 off on Hopfield equations (alpha = 0.05,
  beta = 3) where both composite rules agree to 1e-13.
A non-finite integrand value (float underflow such as ln 0 far in a tail) counts as 0
where the product weight is below 1e-25; elsewhere, or for a complex or non-finite
result, ValueError.  erfc is kept as written (_derived_num.safe).  Nothing here is
validated for integrands with kinks (the T = 0 kinks are removed by the closed forms of
gauss_rsb before evaluation) or features narrower than the node spacing; the
main-vs-audit difference is the check, reported by the solver as quad_error.
"""
import math

import numpy as np
import sympy as sp
from scipy.special import erfc as _erfc

from ._derived_num import safe
from .core import Refusal
from .gauss import GaussE

FAST_NODES = 64
MAIN_PANELS, MAIN_ORDER, MAIN_CUT = 32, 8, 12.0
AUDIT_STEP, AUDIT_CUT = 0.05, 13.0
_TINY = 1e-25
_NP = [{"Erfc": _erfc}, "numpy"]


def _gh(n):
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / math.sqrt(2 * math.pi)


def _trap():
    x = np.arange(-AUDIT_CUT, AUDIT_CUT + AUDIT_STEP / 2, AUDIT_STEP)
    return x, AUDIT_STEP * np.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def _gl():
    y, v = np.polynomial.legendre.leggauss(MAIN_ORDER)
    edges = np.linspace(-MAIN_CUT, MAIN_CUT, MAIN_PANELS + 1)
    x = np.concatenate([(lo + hi) / 2 + (hi - lo) / 2 * y for lo, hi in zip(edges, edges[1:])])
    wt = np.concatenate([(hi - lo) / 2 * v for lo, hi in zip(edges, edges[1:])])
    return x, wt * np.exp(-x * x / 2) / math.sqrt(2 * math.pi)


RULES = {"main": _gl(), "audit": _trap(), "fast": _gh(FAST_NODES)}


def top_atoms(expr):
    """The GaussE atoms of expr not inside another GaussE."""
    out = []

    def walk(e):
        if isinstance(e, GaussE):
            if e not in out:
                out.append(e)
            return
        for a in e.args:
            walk(a)
    walk(sp.sympify(expr))
    return sorted(out, key=str)


def _clean(y, wt):
    y = np.asarray(y)
    if np.iscomplexobj(y):
        if np.any(y.imag != 0):
            raise ValueError("complex integrand")
        y = y.real
    bad = ~np.isfinite(y)
    if bad.any():
        if np.any(np.broadcast_to(wt, y.shape)[bad] > _TINY):
            raise ValueError("non-finite integrand where the weight matters")
        y = np.where(bad, 0.0, y)
    return y


class NestedNum:
    def __init__(self, expr, args):
        expr = safe(sp.sympify(expr))
        self.args = list(args)
        tops = top_atoms(expr)
        dum = [sp.Dummy(f"A{k}") for k in range(len(tops))]
        self.outer = sp.lambdify(dum + self.args, expr.xreplace(dict(zip(tops, dum))), _NP)
        self.atoms = []
        for A in tops:
            inner = top_atoms(A.f)
            if any(top_atoms(B.f) for B in inner):
                raise Refusal("precondition:nesting-depth",
                              f"{A}: expectations nested more than two deep")
            idum = [sp.Dummy(f"B{k}") for k in range(len(inner))]
            F = sp.lambdify([A.z] + idum + self.args, A.f.xreplace(dict(zip(inner, idum))),
                            _NP, cse=True)
            fs = []
            for B in inner:         # the inner bound variable gets its own name (no capture)
                w = sp.Dummy("w", real=True)
                fs.append(sp.lambdify([A.z, w] + self.args, B.f.xreplace({B.z: w}), _NP,
                                      cse=True))
            self.atoms.append((F, fs))

    def value(self, vals, rule="main"):
        x, wt = RULES[rule]
        vals = [float(v) for v in vals]
        T, U = x[:, None], x[None, :]
        W2 = wt[:, None] * wt[None, :]
        got = []
        with np.errstate(all="ignore"):
            for F, fs in self.atoms:
                inner = [_clean(np.broadcast_to(f(T, U, *vals), W2.shape), W2) @ wt for f in fs]
                y = _clean(np.broadcast_to(F(x, *inner, *vals), x.shape), wt)
                got.append(float(wt @ y))
            v = self.outer(*got, *vals)
        v = complex(v) if np.iscomplexobj(v) else float(v)
        if isinstance(v, complex):
            if v.imag != 0:
                raise ValueError(f"complex value {v}")
            v = v.real
        if not math.isfinite(v):
            raise ValueError(f"non-finite value {v}")
        return v
