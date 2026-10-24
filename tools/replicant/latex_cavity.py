"""Displayed equations of the cavity routes: the dense SK / Hopfield / perceptron cavity
arguments (cavity/dense*.py, the four Steps remove_site, cavity_field_law, add_site,
self_consistency) and the sparse Viana-Bray / Bethe-lattice cavity (cavity/sparse.py,
rsb_sparse*.py).  Same rules as latex_entries.py: sympy forms carry a Check against the
cavity code; the rest are cited strings mirroring the module docstrings' formulas."""
import mpmath as mp
import sympy as sp

from .gauss import GaussE
from .latex_entries import A, B, HF, KA, LAM, M, MINUS_BF, Q, R, Z, _H, _phi
from .tex import Check, cited, sym, text

_MPV = "Mezard, Parisi & Virasoro, Spin Glass Theory and Beyond (1987), ch. V"
_TAP = "Thouless, Anderson & Palmer, Phil. Mag. 35, 593 (1977)"
_AGS = "Amit, Gutfreund & Sompolinsky, Ann. Phys. 173, 30 (1987)"
_MEZ = "Mezard, J. Phys. A 22, 2181 (1989)"
_MP01 = "Mezard & Parisi, Eur. Phys. J. B 20, 217 (2001)"
_MP03 = "Mezard & Parisi, J. Stat. Phys. 111, 1 (2003)"
_VB = "Viana & Bray, J. Phys. C 18, 3037 (1985)"


def _cav(name):
    return f"cavity step {name} (cavity/dense.py)"


def sk_cavity():
    from .cavity.dense import cavity_sk_rs
    u = HF + sp.sqrt(Q) * Z
    m0 = sp.tanh(B * u)
    e_j = -(GaussE(m0 * sp.sqrt(Q) * Z, Z) + B * (1 - Q)) / 2
    sol = lambda: tuple((b, h, cavity_sk_rs(b, h).q) for b, h in
                        (("1.5", "0.3"), ("0.9", "0.3"), ("2", "0")))
    at = lambda attr: (lambda b, h, q: getattr(cavity_sk_rs(b, h), attr))
    return [
        text("remove_site", "Remove one spin s_0; its couplings J_0i ~ N(0, 1/N) are "
             "independent of the cavity system (N spins without s_0). " + _cav("1")),
        cited("cavity_field_law: the cavity field in the system without s_0",
              r"h_c = h + \sum_i J_{0i} s_i \sim \mathcal{N}(u,\, 1 - q)\ \text{(thermal)},"
              r"\quad u = h + \sum_i J_{0i} m_i \sim \mathcal{N}(h,\, q)\ \text{(disorder)}",
              _MPV),
        cited("add_site: marginal, Onsager reaction, response",
              r"m_0 = \tanh\beta u,\quad \langle h_c \mid s_0\rangle = u + \beta(1-q)\, s_0,"
              r"\quad \frac{\partial m_0}{\partial u} = \beta(1 - m_0^2)", f"{_MPV}; {_TAP}"),
        sym("self_consistency: the removed spin is typical", sp.Eq(Q, GaussE(m0**2, Z)),
            Check((B, HF, Q), lambda b, h, q: q, sol, "rhs")),
        sym("Energy per spin (E<s_0(h_c - h)> by quadrature, no Stein identity)",
            sp.Eq(sp.Symbol("e"), e_j - HF * GaussE(m0, Z)),
            Check((B, HF, Q), at("e"), sol, "rhs")),
        sym("Free energy: ln Z_{N+1} - ln Z_N, with the coupling rescaling 1/N -> 1/(N+1)",
            sp.Eq(MINUS_BF, GaussE(sp.log(2 * sp.cosh(B * u)), Z) + B**2 * (1 - Q) / 2
                  + B * e_j / 2),
            Check((B, HF, Q), lambda b, h, q: -mp.mpf(b) * cavity_sk_rs(b, h).f, sol, "rhs")),
        sym("Stability: chi_SG = 1 + beta^2 E(1 - m_0^2)^2 chi_SG is finite iff lambda > 0",
            sp.Eq(LAM, 1 - B**2 * GaussE((1 - m0**2)**2, Z)),
            Check((B, HF, Q), at("replicon"), sol, "rhs")),
    ]


def hopfield_cavity():
    from .cavity import dense_hopfield as C
    xi = B * (M + sp.sqrt(A * R) * Z)
    D_ = 1 - B * (1 - Q)
    arb = (("0.9", "0.8", "0.5", "0.05", "3"), ("0.5", "0.6", "1.2", "0.1", "2"),
           ("0.95", "0.9", "0.3", "0.12", "5"))
    mom = lambda k: (lambda m, q, r, a, b: C._moments(float(b), float(m),
                                                      float(mp.sqrt(mp.mpf(a) * mp.mpf(r))))[k])
    return [
        cited("remove_site / cavity_field_law: one uncondensed pattern taken out",
              r"x = \sqrt N\, m_\mu \sim \mathcal{N}(a,\, 1 - q)\ \text{(thermal)},\quad "
              r"a \sim \mathcal{N}(0, q),\qquad r = \mathbb{E}\langle x\rangle^2 = "
              r"\frac{q}{(1 - C)^2},\ C = \beta(1-q)", f"{_MPV}; {_AGS}"),
        cited("add_site: the site's local field (signal plus crosstalk)",
              r"h_0 = m + \sum_{\mu > 1}\xi_0^\mu \frac{\langle x_\mu\rangle}{\sqrt N} \sim "
              r"\mathcal{N}(m,\, \alpha r),\quad m_0 = \tanh\beta(m + \sqrt{\alpha r}\, z)", _MPV),
        sym("self_consistency in m", sp.Eq(M, GaussE(sp.tanh(xi), Z)),
            Check((M, Q, R, A, B), mom(0), arb, "rhs")),
        sym("self_consistency in q", sp.Eq(Q, GaussE(sp.tanh(xi)**2, Z)),
            Check((M, Q, R, A, B), mom(1), arb, "rhs")),
        sym("Energy per spin (E<x^2> = r + (1 - q)/(1 - C) from the reweighted Gaussian)",
            sp.Eq(sp.Symbol("e"), A / 2 - M**2 / 2 - A / 2 * (R + (1 - Q) / D_)),
            Check((M, Q, R, A, B), lambda m, q, r, a, b: C.energy(*map(float, (m, q, r, a, b))),
                  arb, "rhs")),
        cited("Cavity susceptibility (the AT quantity)",
              r"\lambda = 1 - \frac{\alpha\beta^2\,\mathbb{E}(1 - m_0^2)^2}{(1 - C)^2}", _MPV),
    ]


def perceptron_cavity():
    from .cavity.dense import cavity_gardner_capacity
    return [
        cited("remove_site: one pattern (constraint) taken out",
              r"\lambda_0 = \frac{J\cdot\xi^0}{\sqrt N} \sim \mathcal{N}(\sqrt q\, t,\, 1 - q)",
              _MEZ),
        cited("add_site: the constraint Theta(lambda_0 - kappa) put back",
              r"\text{volume fraction } H(x),\quad \langle\lambda_0\rangle = \sqrt q\, t + "
              r"\sqrt{1-q}\, E(x),\quad x = \frac{\kappa - \sqrt q\, t}{\sqrt{1-q}}", _MEZ),
        cited("self_consistency: the cavity forces of the P patterns add up",
              r"\frac{q}{(1-q)^2} = \alpha\,\mathbb{E}_t\, g^2,\quad g = "
              r"\frac{\langle\lambda_0\rangle - \sqrt q\, t}{1 - q}", _MEZ),
        cited("Entropy by thermodynamic integration",
              r"\frac{dS}{d\alpha} = \mathbb{E}_t \ln H(x),\quad S(0) = 0", _MEZ),
        sym("Capacity (q -> 1; the cavity integrates 1/E_t[(kappa - t)_+^2] numerically)",
            sp.Eq(sp.Symbol("alpha_c"), 1 / ((1 + KA**2) * _H(-KA) + KA * _phi(KA))),
            Check((KA,), lambda k: cavity_gardner_capacity("spherical", k).alpha_c,
                  (("0",), ("0.5",), ("1",)), "rhs")),
        cited("Stability (variance reduction of the pattern's field)",
              r"\lambda = 1 - \alpha\,\mathbb{E}_t\left[\big(E(x)(E(x) - x)\big)^2\right]", _MEZ),
    ]


def viana_bray():
    from .cavity.sparse import viana_bray_beta_c
    Bb, J = sp.Symbol("B", positive=True), sp.Symbol("J", positive=True)
    return [
        cited("Cavity recursion (belief propagation)",
              r"u(J, h) = \frac{1}{\beta}\operatorname{atanh}\!\big(\tanh(\beta J)\tanh(\beta h)"
              r"\big),\qquad h_{i\to j} = H + \sum_{l\in\partial i\setminus j} "
              r"u(J_{il}, h_{l\to i})",
              f"{_MP01}, Eqs. (3)-(6)"),
        cited("Observables per spin (cbar = mean degree, t = tanh beta J, t_a = tanh beta h_a)",
              r"q = \mathbb{E}\tanh^2(\beta H_i),\quad e = \frac{\bar c}{2}\,\mathbb{E}\!\left[-J"
              r"\frac{t + t_1 t_2}{1 + t\, t_1 t_2}\right] - H\,\mathbb{E}\tanh(\beta H_i)",
              f"{_MP01}, Eqs. (12), (13)"),
        cited("Free energy per spin",
              r"-\beta f = \frac{\bar c}{2}\,\mathbb{E}\ln Z_{\mathrm{link}}(J, h_1, h_2) - "
              r"\mathbb{E}\big[(d - 1)\ln 2\cosh(\beta H_d)\big]", f"{_MP01}, Eq. (16)"),
        sym("Transition (+-J): B tanh^2(beta_c J) = 1, B = k (regular) or c (Poisson)",
            sp.Eq(sp.Symbol("beta_c"), sp.atanh(1 / sp.sqrt(Bb)) / J),
            Check((Bb, J), lambda b, j: viana_bray_beta_c(("regular", int(b)), ("pm1", float(j))),
                  ((2, "1"), (3, "1"), (5, "0.5")), "rhs")),
        cited("T = 0 limit of the recursion (integer cavity fields for +-1)",
              r"u(J, h) \to \frac{|J + h| - |J - h|}{2}", f"{_MP03}, Eq. (10)"),
        cited("RS population dynamics (one population of cavity fields)",
              r"h^{\mathrm{new}} = H + \sum_{l=1}^{d} u(J_l, h_{r_l}),\quad d \sim "
              r"\text{excess-degree law},\ r_l \text{ uniform in the population}",
              f"{_MP01}, Secs. III-IV"),
        text("Statistical estimates", "Population-dynamics observables are averaged "
                     "over sweeps after burn-in in each of several independent runs (numpy "
                     "SeedSequence(seed).spawn); the value is the mean over runs and the error "
                     "bar its standard error: statistical, not certified.", f"{_MP01}, Sec. IV"),
        text("1RSB", "1RSB: population of populations at Parisi parameter m "
                     "(reweighting by the free-energy shift, MP 2001 Secs. V-VI); T = 0 "
                     "surveys at y = beta m and the factorized 1RSB ansatz (MP 2003 Secs. "
                     "IV-V, V.F.1). These outputs carry the 1RSB caveat (one breakpoint; "
                     "full RSB may be the truth on sparse graphs).", f"{_MP01}; {_MP03}"),
        text("Paramagnetic instability (Poisson graph)", "c E tanh^2(beta_c J) = 1",
                     _VB),
    ]

