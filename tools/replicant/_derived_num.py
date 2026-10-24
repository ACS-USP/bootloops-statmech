"""Compiled numeric evaluation of derived expressions (helper of solve_derived).

  Num(expr, args, dps, gh_nodes)   expr holding GaussE atoms, as three evaluators:
    .mp(vals)      main: each expectation by solve.gauss_expect (tanh-sinh on
                   [-inf, -6, 0, 6, inf]) at dps + 3 digits
    .audit(vals)   independent: mpmath Gauss-Legendre on the split
                   [-inf, -7, -2, 2, 7, inf] at dps + 6 digits (a different rule,
                   different nodes, different precision), so a disagreement with
                   .mp measures quadrature error, which .mp alone cannot see
    .fast(vals)    float64: probabilists' Gauss-Hermite with gh_nodes nodes
                   (numpy), for the root search only; a complex value raises
                   ValueError, a non-finite integrand at nodes of total weight
                   < 1e-25 (float underflow such as 0/0 far in the tail) counts as
                   0 there, elsewhere it raises ValueError
  safe(expr)       erfc kept as written (Erfc): sympy rewrites erfc(-x) as
                   2 - erfc(x) (e.g. once kappa = 0 is substituted), which loses
                   every digit where erfc(x) -> 0 and divides by 0 in phi/H.

Each expectation is lambdified once (cse).  Linear combinations of
expectations are merged into one expectation first (gauss._merge_linear), which
also cancels the sqrt(qhat)/sqrt(qhat) that integration by parts leaves.
"""
import math

import mpmath as mp
import numpy as np
import sympy as sp
from scipy.special import erfc as _np_erfc

from .core import Refusal
from .gauss import GaussE, _merge_linear
from .solve import gauss_expect

AUDIT_SPLIT = (-mp.inf, -7, -2, 2, 7, mp.inf)
_TAIL_W = 1e-25     # float path: non-finite nodes allowed only where the weight sums below


class Erfc(sp.Function):
    """erfc, never auto-rewritten; same derivative; evaluated by mpmath / scipy."""
    def fdiff(self, argindex=1):
        u = self.args[0]
        return -2 * sp.exp(-u**2) / sp.sqrt(sp.pi)


def safe(expr):
    return sp.sympify(expr).replace(lambda e: isinstance(e, sp.erfc), lambda e: Erfc(e.args[0]))


_MP = [{"Erfc": mp.erfc}, "mpmath"]
_NP = [{"Erfc": _np_erfc}, "numpy"]
_MATH = [{"Erfc": math.erfc}, "math"]


def _gh(n):
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / math.sqrt(2 * math.pi)


class Num:
    def __init__(self, expr, args, dps, gh_nodes=80):
        self.dps, self.gh = dps, gh_nodes
        expr = sp.sympify(expr)
        atoms = sorted(expr.atoms(GaussE), key=str)
        if len(atoms) > 1:
            expr = _merge_linear(expr, atoms)
            atoms = sorted(expr.atoms(GaussE), key=str)
        if any(a.f.atoms(GaussE) for a in atoms):
            raise Refusal("precondition:no-nested-expectation", str(expr))
        dum = [sp.Dummy(f"G{k}") for k in range(len(atoms))]
        self._atoms, self._args = atoms, list(args)
        self._outer_expr = expr.xreplace(dict(zip(atoms, dum)))
        self._dum = dum
        self.inner = [sp.lambdify([a.z] + self._args, a.f, _MP, cse=True) for a in atoms]
        self.outer = sp.lambdify(dum + self._args, self._outer_expr, _MP)
        self._fast = None

    @staticmethod
    def _clean(v):
        if isinstance(v, mp.mpc) and v.imag == 0:
            v = v.real
        return v if isinstance(v, (mp.mpf, mp.mpc)) else mp.mpf(v)

    def mp(self, vals):
        g = [gauss_expect(lambda z, f=f: f(z, *vals), dps=self.dps + 3) for f in self.inner]
        return self._clean(self.outer(*g, *vals))

    def audit(self, vals):
        with mp.workdps(self.dps + 6):
            c = 1 / mp.sqrt(2 * mp.pi)
            g = [mp.quad(lambda z, f=f: f(z, *vals) * c * mp.exp(-z * z / 2), AUDIT_SPLIT,
                         method="gauss-legendre") for f in self.inner]
            return self._clean(self.outer(*g, *vals))

    def fast(self, vals):
        if self._fast is None:
            x, w = _gh(self.gh)
            inner = [sp.lambdify([a.z] + self._args, a.f, _NP, cse=True) for a in self._atoms]
            outer = sp.lambdify(self._dum + self._args, self._outer_expr, _MATH)
            self._fast = (x, w, inner, outer)
        x, w, inner, outer = self._fast
        vals = [float(v) for v in vals]
        g = []
        with np.errstate(all="ignore"):
            for f in inner:
                y = np.broadcast_to(np.asarray(f(x, *vals)), x.shape)
                if np.iscomplexobj(y):
                    if np.any(y.imag != 0):
                        raise ValueError("complex integrand")
                    y = y.real
                bad = ~np.isfinite(y)
                if bad.any():       # float under/overflow (0/0) far in the tails
                    if w[bad].sum() > _TAIL_W:
                        raise ValueError("non-finite integrand where the weight matters")
                    y = np.where(bad, 0.0, y)
                g.append(float(np.dot(w, y)))
        v = outer(*g, *vals)
        if isinstance(v, complex) or not math.isfinite(v):
            raise ValueError(f"non-finite or complex value {v}")
        return v
