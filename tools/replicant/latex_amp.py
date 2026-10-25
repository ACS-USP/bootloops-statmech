"""Displayed equations of the `amp` routes (catalog_amp.py): SK AMP and its state
evolution (amp/sk.py, amp/state_evolution.py), Bayes-optimal GAMP for the teacher-student
perceptron and its state evolution (amp/gamp.py, amp/gamp_se.py).  Same rules as
latex_entries.py: a sympy form carries a Check against the amp/ code at three points;
the iterations themselves (vectors, matrices) are cited strings mirroring the module
docstrings."""
import sympy as sp

from .gauss import GaussE
from .latex_entries import A, B, HF, Q, Z, _H, _phi
from .tex import Check, cited, sym, text

_BM = "Bayati & Montanari, IEEE Trans. Inf. Theory 57, 764 (2011), arXiv:1001.3448"
_BOL = "Bolthausen, Commun. Math. Phys. 325, 333 (2014), arXiv:1201.2891"
_BK = ("Barbier, Krzakala, Macris, Miolane & Zdeborova, PNAS 116, 5451 (2019), "
       "arXiv:1708.03395")
_RAN = "Rangan, IEEE ISIT 2011, 2168, arXiv:1010.5141"
RR, XI = sp.Symbol("r", positive=True), sp.Symbol("xi", real=True)
OM, VV, YY = sp.Symbol("omega", real=True), sp.Symbol("V", positive=True), sp.Symbol("y")


def sk_amp():
    from .amp import state_evolution as SE
    import mpmath as mp
    pts = (("1.5", "0", "0.4"), ("1.5", "0.8", "0.6"), ("0.7", "0.3", "0.2"))
    return [
        cited("AMP iteration (Onsager term), m^{-1} = 0, Q_t = <m^t, m^t>",
              r"x^{t+1} = J m^t - \beta(1 - Q_t)\, m^{t-1},\qquad m^{t+1} = "
              r"\tanh\beta\left(h + x^{t+1}\right)", f"{_BM}, eq. (4.1); {_BOL}"),
        sym("State evolution: the N -> oo law of Q_t at each fixed t",
            sp.Eq(sp.Symbol("Q_{t+1}"), GaussE(sp.tanh(B * (HF + sp.sqrt(Q) * Z))**2, Z)),
            Check((B, HF, Q), lambda b, h, q: SE._F_mp(*(mp.mpf(x) for x in (b, h, q))), pts,
                  "rhs"), source=f"{_BM}, Thm 4"),
        text("What is proved", "For Gaussian (GOE-type) J and the Lipschitz tanh, lim_N Q_t "
             "of AMP equals the SE Q_t at each fixed t (Bayati-Montanari Thm 4, stated with "
             "diagonal variance 2/N; amp/sk.py uses a zero diagonal). q_se is the t -> oo "
             "limit of SE: a statement about the algorithm, not about the Gibbs measure. The "
             "iterates converge only on the RS-stable side of the AT line with h > 0 "
             "(Bolthausen Thm 2.1). Finite-N values are statistical: mean +- stderr over "
             "named seeds with a 99% Student-t half-width.", _BM),
    ]


def ts_amp():
    from .amp import gamp as G
    from .amp import gamp_se as GS
    s = sp.sqrt(Q / (1 - Q))
    u = YY * OM / sp.sqrt(VV)
    qr = (("0.2", "1"), ("0.6", "2"), ("0.85", "0.5"))
    rs = (("0.3",), ("2",), ("7",))
    return [
        cited("GAMP iteration (scalar variances), x^0 = 0, v^0 = 1, g^0 = 0",
              r"\omega^t = \frac{\Phi \hat x^{t-1}}{\sqrt N} - V^t g^{t-1},\quad "
              r"g^t = g_{\rm out}(y, \omega^t, V^t),\quad \lambda^t = \alpha\,\overline{(g^t)^2},"
              r"\quad R^t = \hat x^{t-1} + \frac{\Phi^{\top} g^t}{\sqrt N\,\lambda^t},\quad "
              r"\hat x^t = g_{P_0}(R^t, \lambda^t)",
              f"{_RAN}; {_BK}, Sec. 2.4 (their displayed g_Pout is sqrt(V) times the g_out "
              "below, so their omega and lambda lines differ by powers of sqrt(V))"),
        sym("Output function of the noiseless sign channel (d/d omega ln H(-y omega/sqrt V))",
            sp.Eq(sp.Symbol("g_out"), YY * _phi(u) / (sp.sqrt(VV) * _H(-u))),
            Check((YY, OM, VV), lambda y, w, v: G._gout(float(y), float(w), float(v))[0],
                  (("1", "0.3", "0.5"), ("-1", "0.8", "0.2"), ("1", "-2", "0.9")), "rhs")),
        sym("State evolution, channel: r = 2 alpha Psi_out'(q) (BKMMZ eq. (10))",
            sp.Eq(RR, 2 * A / (1 - Q) * GaussE(_phi(s * XI)**2 / _H(-s * XI), XI)),
            Check((Q, A), lambda q, a: GS._rhat_quadpack(float(q), float(a)), qr, "rhs"),
            source=f"{_BK}, eqs. (6), (10)"),
        sym("State evolution, gaussian prior: q = 2 psi'(r)", sp.Eq(Q, RR / (1 + RR)),
            Check((RR,), lambda r: GS._prior_map("gaussian", float(r)), rs, "rhs")),
        sym("State evolution, ising prior (+-1): q = 2 psi'(r)",
            sp.Eq(Q, GaussE(sp.tanh(RR + sp.sqrt(RR) * Z), Z)),
            Check((RR,), lambda r: GS._prior_map_quadpack("ising", float(r)), rs, "rhs")),
        cited("Errors at the SE overlap q (rho = 1; Nishimori: a posterior sample has overlap q)",
              r"\mathrm{mse} = 1 - q,\quad \epsilon_{\rm Bayes} = \frac{\arccos\sqrt q}{\pi},"
              r"\quad \epsilon_{\rm Gibbs} = \frac{\arccos q}{\pi}", f"{_BK}, Sec. 2.3"),
        text("What is (not) proved", "GAMP's state evolution for generalized linear models "
             "is stated by Barbier et al. as Claim 1 (from Rangan 2011, steps of the proof "
             "missing), so no theorem is applied here and the SE values are not-computed. "
             "The RS formula for the Bayes-optimal errors is their Thm 2 (the replica "
             "route). With the gaussian prior the SE fixed point solves the spherical Gibbs "
             "equation (r = q/(1 - q)): the agreement of the two routes checks an "
             "identity. Bayes-optimal only on the "
             "Nishimori line.", _BK),
    ]
