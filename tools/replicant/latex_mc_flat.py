"""Displayed equations of `wang_landau_mc` (catalog_mc_wl.py) and `histogram_reweighting_mc` (catalog_mc_fs.py).
The sympy blocks are checked against the code: the walk's acceptance probability (montecarlo/flat_walk.py), the
1/t refinement (montecarlo/wang_landau.py) and the single-histogram factor (montecarlo/reweight.py)."""
import sympy as sp

from .tex import Check, cited, sym, text

S1, S2, T, B, B0, E = sp.symbols("S_1 S_2 t beta beta_0 E", real=True)
_WL = "Wang & Landau, arXiv:cond-mat/0011174, Eq. (1) and p. 2"
_BP = "Belardinelli & Pereyra, arXiv:cond-mat/0701672, Eqs. (1), (2), Sec. III"
_BERG = "Berg, arXiv:cond-mat/9909236, Eqs. (1.5)-(1.8), (2.15), (2.16)"
_BN = "Berg & Neuhaus, arXiv:hep-lat/9202004, Eqs. (7), (10)"
_KUMAR = "Kumar et al., J. Comput. Chem. 13, 1011 (1992), Eqs. (17)-(20), (22) (the Ferrenberg-Swendsen equations)"


def _f(x):
    from fractions import Fraction
    return float(Fraction(str(x)))


def _acc(s1, s2):
    from .montecarlo.flat_walk import acceptance_probability
    return acceptance_probability(_f(s1), _f(s2))


def _ref(t):
    from .montecarlo.wang_landau import refinement
    return float(refinement(_f(t)))


def _fac(b0, b, e):
    import math
    from .montecarlo.reweight import log_factor
    return math.exp(float(log_factor(_f(b0), _f(b), _f(e))))


def wang_landau_mc(route):
    return [
        text("Sample route", "Wang-Landau walkers estimate S(E) = ln g(E) of the 2D Ising torus over named seeds "
             "(the 1/t refinement or the original schedule); a multicanonical run with the weights exp(-S) gives "
             "canonical averages at any beta; statistical, never final", source=_BP),
        sym("Acceptance of a flip from E_1 to E_2 (S = ln g)",
            sp.Eq(sp.Symbol("p"), sp.Min(1, sp.exp(S1 - S2))), source=_WL,
            check=Check((S1, S2), _acc, (("0", "1"), ("2", "1"), ("1/2", "3/2")), "rhs")),
        cited("Visit update and the original schedule",
              r"S(E) \to S(E) + F,\quad H(E) \to H(E) + 1;\qquad H\ \text{flat}: \min_E H \ge 0.8\,\langle H\rangle"
              r"\ \Rightarrow\ F \to F/2,\ H \to 0", _WL),
        sym("The 1/t refinement (once F <= 1/t), t = trial moves / number of levels",
            sp.Eq(sp.Symbol("F"), 1 / T), source=_BP, check=Check((T,), _ref, ((10,), (250,), (4000,)), "rhs")),
        cited("Error against the exact density of states (ln g(ground) = ln 2)",
              r"\varepsilon(E) = \left|\frac{\ln g_{\mathrm{est}}(E) - \ln g_{\mathrm{ex}}(E)}{\ln g_{\mathrm{ex}}(E)}"
              r"\right|,\qquad \langle\varepsilon(t)\rangle \sim t^{-1/2}\ \ (1/t)", _BP),
        cited("Multicanonical weight", r"w_{\mathrm{mu}}(E) \approx \frac{1}{g(E)} = e^{-S(E)},\qquad "
              r"w_{\mathrm{mu}} \propto e^{\alpha_k + \beta_k S}\ \ (S_k < S \le S_{k+1})", f"{_BERG}; {_BN}"),
        cited("Reweighting a multicanonical run to beta",
              r"\langle O\rangle_\beta = \frac{\sum_i O_i\, w_{\mathrm{mu}}^{-1}(E_i)\, e^{-\beta E_i}}"
              r"{\sum_i w_{\mathrm{mu}}^{-1}(E_i)\, e^{-\beta E_i}},\qquad C = \beta^2\,\mathrm{Var}(E)/N", _BERG)]


def histogram_reweighting_mc(route):
    return [
        text("Sample route", "canonical chains at beta_0 (and at several beta for the multiple histogram) "
             "reweighted to beta within the overlap window; statistical, never final", source=_KUMAR),
        sym("Single-histogram factor", sp.Eq(sp.Symbol("w"), sp.exp(-(B - B0) * E)), source=_KUMAR,
            check=Check((B0, B, E), _fac, (("2/5", "9/20", -20), ("1/2", "2/5", -30), ("1/3", "1/3", 5)), "rhs")),
        cited("Multiple histogram (Ferrenberg-Swendsen, as printed by Kumar et al.)",
              r"P_\beta(E) = \frac{\sum_k g_k^{-1} N_k(E)\, e^{-\beta E}}{\sum_m n_m g_m^{-1}\, e^{f_m - \beta_m E}},"
              r"\qquad e^{-f_j} = \sum_E P_{\beta_j}(E),\qquad g_k = 2\tau_{\mathrm{int},k}", _KUMAR),
        cited("Reliable window (this package's rule)",
              r"O(\beta) = \sum_E \min\!\big(p_0(E), p_\beta(E)\big) \ge \tfrac12\quad"
              r"\big(\text{Gaussian: } |\beta - \beta_0|\,\sigma_E \le 1.349\big)", _KUMAR)]


__all__ = ["histogram_reweighting_mc", "wang_landau_mc"]
