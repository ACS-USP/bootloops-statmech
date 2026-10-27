"""Displayed equations of `cluster_mc` (catalog_mc_cluster.py).  The sympy blocks are checked against
montecarlo/fk.py: the Ising and Potts bond probabilities at three couplings each and the critical points
log(1 + sqrt q) against the code's two-route value."""
import sympy as sp

from .tex import Check, cited, sym, text

B, Q = sp.symbols("beta q", positive=True)
_SS = "Salas & Sokal, arXiv:hep-lat/9605018, Sec. 2.1, Eqs. (2.1)-(2.5)"
_SSW = "Salas & Sokal, arXiv:hep-lat/9605018, Eqs. (4.11), (4.13), p. 13"
_LPS = "Liu, Polkovnikov & Sandvik, arXiv:1310.6327, Secs. III B, C, Eq. (27)"
_BDC = "Beffara & Duminil-Copin, arXiv:1006.5073, Theorems 1, 3"


def _p(q, b):
    from .montecarlo.fk import bond_probability
    return bond_probability(q, b)


def _bc(q):
    from .montecarlo.fk import critical_beta
    return critical_beta(int(q))[0]


def cluster_mc(route):
    return [
        text("Sample route", "Swendsen-Wang and Wolff cluster updates on the Fortuin-Kasteleyn bonds, and the "
             "checkerboard Metropolis baseline, run over named seeds on the L x L torus; statistical, never final",
             source=_SS),
        cited("Edwards-Sokal joint weight and its Fortuin-Kasteleyn marginal",
              r"W(\sigma, n) \propto \prod_b \left[(1-p)\,\delta_{n_b,0} + p\,\delta_{\sigma_b}\,\delta_{n_b,1}\right],"
              r"\qquad W_{RC}(n) \propto p^{|n|}(1-p)^{|E|-|n|}\, q^{C(n)}", _SS),
        sym("Ising bond probability (H = -sum s_i s_j)", sp.Eq(sp.Symbol("p"), 1 - sp.exp(-2 * B)), source=_LPS,
            check=Check((B,), lambda b: _p(2, b), (("1/5",), ("2/5",), ("1",)), "rhs")),
        sym("Potts bond probability (H = -sum delta)", sp.Eq(sp.Symbol("p"), 1 - sp.exp(-B)), source=_SS,
            check=Check((B,), lambda b: _p(3, b), (("1/2",), ("1",), ("2",)), "rhs")),
        cited("Swendsen-Wang: bonds given spins, spins given bonds",
              r"P(n_b = 1 \mid \sigma) = p\,\delta_{\sigma_i\sigma_j},\qquad \sigma|_C \sim \mathrm{Uniform}\{1..q\}"
              r"\ \text{independently per cluster } C", _SS),
        cited("Wolff: one cluster grown from a uniform seed by the same bond rule, recoloured with probability one",
              r"\sigma_i \to b \ne a\ \ \forall i \in C(i_0),\qquad t_{\mathrm{sweep}} = t_{\mathrm{flip}}"
              r"\,\langle|C|\rangle / L^d", _LPS),
        sym("Potts critical point (q >= 2; at q = 2 halved for the Ising convention)",
            sp.Eq(sp.Symbol("beta_c"), sp.log(1 + sp.sqrt(Q))), source=_BDC,
            check=Check((Q,), _bc, ((3,), (4,), (7,)), "rhs")),
        cited("Integrated autocorrelation time, self-consistent window",
              r"\hat\tau_{\mathrm{int}}(M) = \tfrac12 + \sum_{t=1}^{M}\hat\rho(t),\quad M = \min\{M : M \ge 6\,"
              r"\hat\tau(M)\},\quad \mathrm{var}\,\hat\tau \approx \frac{2(2M+1)}{n}\,\tau^2", _SSW),
        cited("Dynamic scaling at beta_c", r"\tau_{\mathrm{int}} \sim A\,L^{z}", _LPS)]


__all__ = ["cluster_mc"]
