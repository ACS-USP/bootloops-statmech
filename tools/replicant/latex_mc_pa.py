"""Displayed equations of `population_annealing_mc` (catalog_mc_pa.py).  The sympy block is checked against
montecarlo/population.py: the reweighting factor's logarithm at three points (a planted wrong factor fails it)."""
import sympy as sp

from .tex import Check, cited, sym, text

B, B1, E = sp.symbols("beta beta' E", real=True)
_M = "Machta, arXiv:1006.0252, Sec. II, Eqs. (1)-(4), Sec. III, Eq. (7)"
_W = "Wang, Machta & Katzgraber, arXiv:1508.05647, Sec. II A, Eqs. (1), (2), (4)"
_WB = "Wang, Machta & Katzgraber, arXiv:1508.05647, Sec. IV A, Eqs. (12)-(16), (20), (23)"
_WF = "Wang, Machta & Katzgraber, arXiv:1508.05647, Sec. IV B, Eqs. (28), (30), (31), Secs. V B, VI B"


def _lw(b, b1, e):
    from fractions import Fraction
    from .montecarlo.population import log_weight
    return float(log_weight(float(Fraction(str(b))), float(Fraction(str(b1))), float(Fraction(str(e)))))


def population_annealing_mc(route):
    return [
        text("Sample route", "a population of R replicas annealed from beta = 0 in evenly spaced steps, resampled by "
             "the Boltzmann reweighting factor and equilibrated by Metropolis sweeps at each temperature; the free "
             "energy from the normalisations; independent runs over named seeds; statistical, never final",
             source=_W),
        sym("Logarithm of the reweighting factor of replica j from beta to beta'",
            sp.Eq(sp.Symbol("ln w_j"), -(B1 - B) * E), source=_W,
            check=Check((B, B1, E), _lw, (("0", "1/10", -20), ("2/5", "9/20", -30), ("1", "3/2", 4)), "rhs")),
        cited("Normalisation and resampling weights",
              r"Q(\beta,\beta') = \frac{1}{\tilde R_\beta}\sum_{j=1}^{\tilde R_\beta} e^{-(\beta'-\beta)E_j},\qquad"
              r"\tau_j = \frac{R}{\tilde R_\beta}\,\frac{e^{-(\beta'-\beta)E_j}}{Q(\beta,\beta')},\qquad"
              r"n_j \in \{\lfloor\tau_j\rfloor, \lceil\tau_j\rceil\},\ \ \mathbb{E}\,n_j = \tau_j", _W),
        cited("Free-energy estimator", r"-\beta_k\tilde F(\beta_k) = \sum_{\ell} \ln Q(\beta_\ell, \beta_{\ell+1})"
              r" + \ln\Omega,\qquad \Omega = 2^N", f"{_M}; {_W}"),
        cited("Weighted average over M independent runs",
              r"-\beta F = \ln\Big[\frac{1}{M}\sum_{r=1}^{M} e^{-\beta\tilde F_r}\Big]", _M),
        cited("Bias of one run at finite R",
              r"\langle -\beta\tilde F\rangle = -\beta F - \tfrac12\,\mathrm{var}(\beta\tilde F) + \dots,\qquad"
              r"\mathrm{var}(\beta\tilde F) \simeq \rho_f / R", _WB),
        cited("Family diagnostics",
              r"\rho_t = \tilde R\sum_i n_i^2,\qquad S_f = -\sum_i n_i\ln n_i,\qquad \rho_s = \tilde R/e^{S_f};"
              r"\qquad \text{equilibrated here: } \tilde R/\rho_t \ge 100,\ \ \mathrm{var}(\beta\tilde F) \le \tfrac12",
              f"{_WF}; {_M}")]


__all__ = ["population_annealing_mc"]
