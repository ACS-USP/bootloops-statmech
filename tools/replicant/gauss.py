"""Formal Gaussian integrals: the GaussE placeholder and a small, tested rule set.

  GaussE(f, z)                  E_{z ~ N(0,1)} f(z), unevaluated; z is bound
  GaussE(...).doit_numeric(pt)  its value at a parameter point (solve.gauss_expect)
  numeric(expr, pt)             an expression containing GaussE atoms, evaluated
  linear(expr)                  E[c f + g] -> c E[f] + E[g]  (c free of z)
  stein(expr)                   one round of E[z^k g] = E[z^(k-1) g'] + (k-1) E[z^(k-2) g]
  rename_bound(expr, z0)        alpha-renaming of every bound variable to z0

Rules used by the RS derivation (each tested on its own in tests/test_gauss.py):
  expect_closed(f, z)           E_z f in closed form for f = Theta(a z + b), a > 0
                                (= H(-b/a), H(x) = erfc(x/sqrt 2)/2) and for
                                f = exp(A z^2 + B z + C), 1 - 2A > 0
                                (= exp(C + B^2/(2(1-2A)))/sqrt(1-2A)); a condition
                                known false refuses, one sympy cannot decide (e.g.
                                sqrt(1-q) > 0) is assumed and reported; else a Refusal
  hubbard_stratonovich(A, S, z) exp(A S^2/2) = E_z exp(sqrt(A) z S),  A >= 0
  rs_split(q, t, z)             lambda_a = sqrt(q) t + sqrt(1-q) z_a: E l_a l_b = q,
                                E l_a^2 = 1, the RS covariance (0 <= q <= 1)
  rs_logdet(d, c, n)            ln det of the n x n matrix d on, c off the diagonal:
                                (n-1) ln(d-c) + ln(d+(n-1)c)  (RS spectrum)
  replica_limit(T, n)           lim_{n->0} T(n)/n for T(0) = 0: polynomial part by
                                ansatz.n_to_zero, the rest by T'(0), so that
                                (1/n) ln E[f^n] -> E ln f (f > 0)

A GaussE of a polynomial in z evaluates itself (E z^k = (k-1)!! for even k).
Honest label: the rules are exact identities where their stated conditions hold;
using them inside a replica calculation (n -> 0 of E[f^n], qhat on the real
axis at the saddle) is physics-level, not proven.
"""
import mpmath as mp
import sympy as sp

from .ansatz import n_to_zero
from .core import Refusal
from .solve import gauss_expect


class GaussE(sp.Expr):
    """E_z f(z), z ~ N(0, 1).  `z` is bound: it is not a free symbol, `subs` of
    z leaves the expectation alone, `subs` of a value containing a free z
    renames the bound one first (no capture), and d/dx passes under the
    integral.  (`xreplace` is raw structural replacement: no capture guard.)"""
    is_commutative = True

    def __new__(cls, f, z):
        f = sp.sympify(f)
        if z not in f.free_symbols:
            return f
        if f.is_polynomial(z):
            p = sp.Poly(sp.expand(f), z)
            return sp.Add(*(c * sp.factorial2(k - 1) for (k,), c in p.terms() if k % 2 == 0))
        return sp.Expr.__new__(cls, f, z)

    @property
    def f(self):
        return self.args[0]

    @property
    def z(self):
        return self.args[1]

    @property
    def free_symbols(self):
        return self.f.free_symbols - {self.z}

    def _eval_derivative(self, s):
        return sp.S.Zero if s == self.z else GaussE(sp.diff(self.f, s), self.z)

    def _eval_subs(self, old, new):
        if old == self.z:
            return self
        if self.z in sp.sympify(new).free_symbols:      # avoid capture: rename first
            w = sp.Dummy(self.z.name, real=True)
            return GaussE(self.f.xreplace({self.z: w}).subs(old, new), w)
        return None

    def _sympystr(self, printer):
        return f"E_{printer._print(self.z)}[{printer._print(self.f)}]"

    def _latex(self, printer):
        return (r"\mathbb{E}_{%s}\left[%s\right]"
                % (printer._print(self.z), printer._print(self.f)))

    def doit_numeric(self, point, dps=20):
        """Value at `point` (symbol -> number), by solve.gauss_expect at `dps`."""
        with mp.workdps(dps + 5):
            f = self.f.subs(_sympy_point(point, dps + 5))
            if f.atoms(GaussE):
                raise Refusal("precondition:no-nested-expectation", str(self))
            left = f.free_symbols - {self.z}
            if left:
                raise Refusal("precondition:complete-point",
                              f"no value for {sorted(map(str, left))}")
            fn = sp.lambdify(self.z, f, "mpmath", cse=True)
            return gauss_expect(fn, dps=dps)


def _sympy_point(point, dps):
    def num(v):
        if isinstance(v, mp.mpf):
            return sp.Float(mp.nstr(v, dps, min_fixed=-mp.inf, max_fixed=mp.inf), dps)
        if isinstance(v, str):
            return sp.Float(v, dps)
        if isinstance(v, float):
            return sp.Float(repr(v), dps)
        return sp.sympify(v)
    return {k: num(v) for k, v in point.items()}


def numeric(expr, point, dps=20):
    """expr at `point`: GaussE atoms by quadrature, then plain evaluation.  When
    expr is linear in its expectations the atoms are merged first (one
    quadrature per bound variable).  Returns an mpf (mpc if not real)."""
    with mp.workdps(dps + 5):
        expr = sp.sympify(expr)
        atoms = sorted(expr.atoms(GaussE), key=str)
        if len(atoms) > 1:
            expr = _merge_linear(expr, atoms)
        vals = {g: sp.Float(mp.nstr(g.doit_numeric(point, dps), dps + 5), dps + 5)
                for g in expr.atoms(GaussE)}
        v = sp.N(expr.xreplace(vals).subs(_sympy_point(point, dps + 5)), dps + 5)
        if v.free_symbols:
            raise Refusal("precondition:complete-point",
                          f"no value for {sorted(map(str, v.free_symbols))}")
        re, im = (mp.mpf(str(sp.N(x, dps + 5))) for x in v.as_real_imag())
        return re if im == 0 else mp.mpc(re, im)


def _merge_linear(expr, atoms):
    """c0 + sum_i c_i E_z[f_i] -> c0 + E_w[sum_i c_i f_i] when the c_i hold no
    expectation (bound variables renamed to one); otherwise expr unchanged."""
    dum = [sp.Dummy() for _ in atoms]
    m = sp.expand(expr.xreplace(dict(zip(atoms, dum))))
    try:
        p = sp.Poly(m, *dum)
    except sp.PolynomialError:
        return expr
    if p.total_degree() > 1:
        return expr
    w = sp.Dummy("w", real=True)
    body = sp.Add(*(p.coeff_monomial(tuple(int(j == k) for j in range(len(atoms))))
                    * a.f.xreplace({a.z: w}) for k, a in enumerate(atoms)))
    return p.coeff_monomial(1) + GaussE(body, w)


def _map_gauss(expr, fn):
    return sp.sympify(expr).replace(lambda e: isinstance(e, GaussE), fn)


def _terms(f, z):
    """[(c, k, h)]: f = sum c z^k h, c free of z, h without a bare z^k factor."""
    out = []
    for term in sp.Add.make_args(sp.expand(f, deep=False)):
        c, rest = term.as_independent(z, as_Add=False)
        k, others = 0, []
        for fac in sp.Mul.make_args(rest):
            if fac == z:
                k += 1
            elif fac.is_Pow and fac.base == z and fac.exp.is_Integer and fac.exp > 0:
                k += int(fac.exp)
            else:
                others.append(fac)
        out.append((c, k, sp.Mul(*others)))
    return out


def linear(expr):
    """Pull factors free of z out of each expectation and split sums."""
    def one(g):
        return sp.Add(*(c * GaussE(g.z**k * h, g.z) for c, k, h in _terms(g.f, g.z)))
    return _map_gauss(expr, one)


def stein(expr):
    """One round of Gaussian integration by parts on every bare z^k factor:
    E[z^k h] = E[z^(k-1) h'] + (k-1) E[z^(k-2) h]  (h phi -> 0 at +-infinity)."""
    def one(g):
        z, out = g.z, sp.S.Zero
        for c, k, h in _terms(g.f, z):
            if k == 0:
                out += c * GaussE(h, z)
            else:
                out += c * (GaussE(z**(k - 1) * sp.diff(h, z), z)
                            + (k - 1) * GaussE(z**(k - 2) * h, z))
        return out
    return linear(_map_gauss(linear(expr), one))


def rename_bound(expr, z0):
    return _map_gauss(expr, lambda g: GaussE(g.f.xreplace({g.z: z0}), z0))


# ---- rules -------------------------------------------------------------------------

def H(x):
    """H(x) = int_x^inf Dz = erfc(x/sqrt 2)/2."""
    return sp.erfc(x / sp.sqrt(2)) / 2


def expect_closed(f, z, notes=None):
    """E_z f in closed form for the two shapes the RS reductions meet.
    Preconditions: Theta slope a > 0; exp-quadratic 1 - 2A > 0 (integrable).
    One known false refuses; one sympy cannot decide is ASSUMED and the
    assumption is appended to `notes` (a list) for the trace."""
    c, g = sp.sympify(f).as_independent(z, as_Add=False)

    def need(cond, text, reason):
        if cond.is_positive is False:
            raise Refusal(reason, f"E_{z}[{f}]: {text} = {cond} is not > 0")
        if cond.is_positive is None and notes is not None:
            notes.append(f"assumed {text} = {cond} > 0")
    if isinstance(g, sp.Heaviside):
        arg = sp.expand(g.args[0])
        if arg.is_polynomial(z) and sp.degree(arg, z) == 1:
            a, b = arg.coeff(z, 1), arg.coeff(z, 0)
            need(a, "slope a", "precondition:known-gaussian-integral")
            return c * H(-b / a)
    if isinstance(g, sp.exp):
        arg = sp.expand(g.args[0])
        if arg.is_polynomial(z) and sp.degree(arg, z) == 2:
            A, B, C = (arg.coeff(z, k) for k in (2, 1, 0))
            need(sp.factor(1 - 2 * A), "1 - 2A", "precondition:gaussian-integrable")
            return c * sp.exp(C + B**2 / (2 * (1 - 2 * A))) / sp.sqrt(1 - 2 * A)
    raise Refusal("precondition:known-gaussian-integral",
                  f"E_{z}[{f}]: only Theta(a z + b), a > 0, and exp(quadratic in z)")


def hubbard_stratonovich(A, S, z):
    """exp(A S^2 / 2) = E_z exp(sqrt(A) z S), valid for A >= 0."""
    return GaussE(sp.exp(sp.sqrt(A) * z * S), z)


def rs_split(q, t, z):
    """A Gaussian vector with E l_a^2 = 1, E l_a l_b = q: l_a = sqrt(q) t + sqrt(1-q) z_a."""
    return sp.sqrt(q) * t + sp.sqrt(1 - q) * z


def rs_logdet(d, c, n):
    """ln det(d on the diagonal, c off it): eigenvalues d - c (n - 1 times), d + (n-1) c."""
    return (n - 1) * sp.log(d - c) + sp.log(d + (n - 1) * c)


def _log_clean(f):
    """ln of a positive product: expand the log, ln exp(x) -> x (x real on the
    physical domain of the RS parameters)."""
    f = sp.expand_log(f, force=True)
    return f.replace(lambda e: isinstance(e, sp.log) and isinstance(e.args[0], sp.exp),
                     lambda e: e.args[0].args[0])


def replica_limit(T, n):
    """lim_{n -> 0} T(n)/n for T(0) = 0 (otherwise a Refusal)."""
    T = sp.sympify(T)
    if sp.simplify(T.subs(n, 0)) != 0:
        raise Refusal("precondition:vanishes-at-n-zero",
                      f"T(0) = {sp.simplify(T.subs(n, 0))}: (1/n) T(n) diverges at n -> 0")
    poly = rest = sp.S.Zero
    for term in sp.Add.make_args(sp.expand(T, deep=False)):
        if term.is_polynomial(n):
            poly += term
        else:
            rest += term
    p0 = poly.subs(n, 0)
    lim = n_to_zero(poly - p0, n) + sp.diff(rest, n).subs(n, 0)
    return _map_gauss(lim, lambda g: GaussE(_log_clean(g.f), g.z))
