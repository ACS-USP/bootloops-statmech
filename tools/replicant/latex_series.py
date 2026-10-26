"""Displayed equations of the `series_ising2d` catalog entry (series route, E8).  Same rule as
latex_entries.py: each sympy block carries a tex.Check at three points (tests/test_latex.py).  The
truncated series is built from the code's exact coefficients and checked against the Onsager integral
by quadrature (series/ising2d.onsager_free_energy_numeric, which shares no code with the expansion);
the elliptic form of theta g is typed here and checked against a numerical derivative of the same
quadrature; the leading coefficient is typed in factored form and checked against the operator the
annihilator returns.  Theorems (Onsager, the hypergeometric equation, Frobenius) are cited strings.
"""
import mpmath as mp
import sympy as sp

from .tex import Check, cited, sym, text

V, U = sp.symbols("v u", positive=True)
_ONS = "Duminil-Copin, arXiv:2208.00864 (2022), Eq. (3.6) (Onsager, Phys. Rev. 65, 117 (1944))"
_DLMF = "DLMF 15.10.1 and 19.5.1"
_LOG = "Duminil-Copin, arXiv:2208.00864 (2022), Sec. 3.2; Baxter, arXiv:1010.0710 (2010), Sec. 5.6"
_ORDER = 24                                  # the displayed truncation (v^24)


def _op():
    from .catalog_series import _rec
    return _rec({}).operator              # the default run's recognition (memoised)


def _theta_g(u):
    from .series.ising2d import onsager_free_energy_numeric
    with mp.workdps(30):
        u, h = mp.mpf(u), mp.mpf("1e-9")             # central difference, error O(h^2)
        f = [onsager_free_energy_numeric(mp.sqrt(u + s * h), dps=28) for s in (1, -1)]
        return u * (f[0] - f[1]) / (2 * h)


def series_ising2d(route):
    from .series.ising2d import ising2d_free_energy_series, onsager_free_energy_numeric
    a = ising2d_free_energy_series(_ORDER)
    g = sum(sp.Rational(c.numerator, c.denominator) * V**n for n, c in enumerate(a))
    W = U * (1 - U)**2 / (1 + U)**4
    F = sp.hyper([sp.Rational(1, 2), sp.Rational(1, 2)], [1], 16 * W)
    tg = (1 - 3 * U) / (4 * (1 - U)) - (U**2 - 6 * U + 1) / (4 * (1 - U**2)) * F
    lead = (U - 1)**2 * (U + 1)**2 * (U**2 - 6 * U + 1)**2

    def lead_code(u):
        with mp.workdps(30):
            return sum(mp.mpf(c.numerator) / c.denominator * mp.mpf(u)**i
                       for i, c in enumerate(_op().coefficient(3)))
    return [
        text("Series route", "Square-lattice Ising model, isotropic coupling, zero field, v = tanh K, "
             "u = v^2; g(v) = ln Z/N - ln 2 - 2 ln cosh K. The Onsager integral is expanded in exact "
             "rational arithmetic; the L x L torus (parity transfer matrix) is an independent exact "
             "route that equals the bulk for every v^k with k < L and, for odd L, for every even k "
             "through v^(2L - 2) (L = 7: a4 .. a12)", source=_ONS),
        cited("Onsager's free energy (zero field)",
              r"-\beta f = \ln 2 + \frac{1}{8\pi^2}\int_0^{2\pi}\!\!\int_0^{2\pi}\ln\left[\cosh^2 2K - "
              r"\sinh 2K\,(\cos\theta_1 + \cos\theta_2)\right]d\theta_1\, d\theta_2", _ONS),
        cited("Exact expansion (W = u (1 - u)^2/(1 + u)^4; moments of cos t1 + cos t2 integrated exactly)",
              r"g = \ln(1+u) - \sum_{k\ge 1}\frac{4^k M_{2k}}{4k}\,W^k,\quad M_{2k} = \left\langle"
              r"(\cos\theta_1+\cos\theta_2)^{2k}\right\rangle = \binom{2k}{k}^2 4^{-k}", _ONS),
        sym(f"The series through v^{_ORDER} (exact coefficients of the code)", sp.Eq(sp.Symbol("g"), g),
            source=_ONS, check=Check((V,), lambda v: onsager_free_energy_numeric(v),
                                     (("0.05",), ("0.08",), ("0.1",)), "rhs")),
        sym("Elliptic form of theta g, F = 2F1(1/2, 1/2; 1; 16 W) = (2/pi) K(k), k = 4v(1 - u)/(1 + u)^2",
            sp.Eq(sp.Symbol(r"\theta g"), tg), source=_DLMF,
            check=Check((U,), _theta_g, (("0.01",), ("0.02",), ("0.04",)), "rhs")),
        sym("Leading coefficient of the recovered operator (theta-order 3, degree 8)",
            sp.Eq(sp.Symbol("P_3"), lead), source="tools/annihilator",
            check=Check((U,), lead_code, (("0.3",), ("2",), ("-0.7",)), "rhs")),
        cited("Operator identity (proved exactly by series/elliptic.py)",
              r"L = M\,\theta,\qquad M\,(s F) = 0,\qquad L\, g = M(R) = P(u),\quad "
              r"z(1-z)F'' + (1-2z)F' - \tfrac14 F = 0", _DLMF),
        cited("Local exponents at u_c = 3 - 2 sqrt 2 (Frobenius, exact in Q(sqrt 2))",
              r"\rho \in \{0, 2, 2\}:\ g \ni (u-u_c)^2\ln|u-u_c|,\quad 2-\alpha = 2,\ \alpha = 0",
              _LOG),
    ]


__all__ = ["series_ising2d"]
