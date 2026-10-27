"""Displayed equations of the replica-route catalog entries (REM, SK, p-spin, perceptron,
teacher-student, Hopfield) and the model definitions; latex_cavity.py has the cavity
routes, latex_amp.py the amp routes, latex_pspin.py the spherical and Ising p-spin.  derivation(entry, route) -> [TexBlock] is what CatalogEntry.derivation returns.

Each equation is either built in sympy from what the family code holds (its
rs_equations_symbolic(), quadratic_term, pspin_functional, rem_1rsb_functional) or
written to mirror the formula the family function evaluates, and then carries a
tex.Check that evaluates it against that function at three parameter points
(tests/test_latex.py); an equation the code cannot evaluate here (nested expectations,
T = 0 reductions) is a cited LaTeX string.  Nothing is shown that the code does not
implement; no intermediate step is shown that the family does not perform.
"""
from functools import lru_cache

import mpmath as mp
import sympy as sp

from .gauss import GaussE
from .tex import Check, cited, sym, text

B, HF = sp.Symbol("beta", positive=True), sp.Symbol("h", real=True)
Q, Z, T = sp.Symbol("q", positive=True), sp.Symbol("z", real=True), sp.Symbol("t", real=True)
M, R = sp.Symbol("m", real=True), sp.Symbol("r", positive=True)
A, KA = sp.Symbol("alpha", positive=True), sp.Symbol("kappa", real=True)
MINUS_BF = -B * sp.Symbol("f")
LAM = sp.Symbol("lambda_AT")


def _H(x):
    """H(x) = int_x^inf Dz, kept as erfc(x/sqrt 2)/2 (unevaluated: sympy would rewrite
    erfc(-y) as 2 - erfc(y), which cancels catastrophically in the tails)."""
    return sp.erfc(x / sp.sqrt(2), evaluate=False) / 2


def _phi(x):
    return sp.exp(-x**2 / 2) / sp.sqrt(2 * sp.pi)


def _mpf(*xs):
    return [mp.mpf(str(x)) if not isinstance(x, mp.mpf) else x for x in xs]


# ---- model definitions (Section "Model") ---------------------------------------------
_SK = "Sherrington & Kirkpatrick, PRL 35, 1792 (1975)"
_AGS = "Amit, Gutfreund & Sompolinsky, PRL 55, 1530 (1985); Ann. Phys. 173, 30 (1987)"
_GARDNER = "Gardner, J. Phys. A 21, 257 (1988)"
_REM = "Derrida, PRL 45, 79 (1980); Phys. Rev. B 24, 2613 (1981)"
_CS = "Crisanti & Sommers, Z. Phys. B 87, 341 (1992)"
_TS = "Gyorgyi, Phys. Rev. A 41, 7097 (1990); Seung, Sompolinsky & Tishby, Phys. Rev. A 45, " \
      "6056 (1992)"
_MP01 = "Mezard & Parisi, Eur. Phys. J. B 20, 217 (2001)"

MODELS = {
    "sk": (r"H(s) = -\sum_{i<j} J_{ij} s_i s_j - h \sum_i s_i,\qquad s_i = \pm 1,\quad "
           r"\mathbb{E} J_{ij} = 0,\ \mathbb{E} J_{ij}^2 = 1/N,\quad \beta = 1/T", _SK),
    "rem": (r"2^N \text{ i.i.d. energies } E_i \sim \mathcal{N}(0, N_s N),\quad N_s = 1/2 "
            r"\text{ (Derrida's } J = 1\text{; default)}", _REM),
    "pspin": (r"H(s) = -\sum_{i_1<\dots<i_p} J_{i_1\dots i_p} s_{i_1}\cdots s_{i_p},\quad "
              r"\textstyle\sum_i s_i^2 = N,\quad J \sim \mathcal{N}\!\left(0, "
              r"\frac{p!}{2 N^{p-1}}\right),\ p = 3", _CS),
    ("pspin", "ising"): (r"H(s) = -\sum_{i_1<\dots<i_p} J_{i_1\dots i_p} s_{i_1}\cdots s_{i_p},"
                         r"\quad s_i = \pm 1,\quad J \sim \mathcal{N}\!\left(0, "
                         r"\frac{p!}{2 N^{p-1}}\right),\ p = 3",
                         "Gardner, Nucl. Phys. B 257, 747 (1985); Montanari & "
                         "Ricci-Tersenghi, EPJB 33, 339 (2003), Eq. (2.1)"),
    "perceptron": (r"V(\xi) = \int_{J\cdot J = N} dJ \prod_{\mu=1}^{\alpha N} "
                   r"\Theta\!\left(\frac{J\cdot\xi^\mu}{\sqrt N} - \kappa\right),\quad "
                   r"\xi_i^\mu \sim \mathcal{N}(0,1),\quad S = \frac{1}{N}\ln V", _GARDNER),
    "teacher_student": (r"\sigma(\xi) = \operatorname{sign}\!\left(\frac{T\cdot\xi}{\sqrt N}"
                        r"\right),\quad T\cdot T = J\cdot J = N,\quad P = \alpha N,\quad "
                        r"\epsilon = \frac{\arccos R}{\pi},\ R = \frac{J\cdot T}{N}", _TS),
    "hopfield": (r"H(s) = -\frac{1}{2N}\sum_{i,j}\sum_{\mu=1}^{p}\xi_i^\mu\xi_j^\mu s_i s_j,"
                 r"\quad p = \alpha N,\ \xi_i^\mu = \pm 1,\ \beta = 1/T", _AGS),
    "viana_bray": (r"E(s) = -\sum_{\langle ij\rangle} J_{ij} s_i s_j - H\sum_i s_i,\quad "
                   r"J_{ij} = \pm J, \\ \text{random regular (excess degree } k\text{) or "
                   r"Poisson (mean degree } c\text{) graph}", _MP01),
    "xorsat": (r"x_{i_1^a} \oplus \dots \oplus x_{i_K^a} = y_a,\quad a = 1, \dots, M = \alpha N,"
               r"\quad \{i^a\} \text{ uniform K-subsets},\ y_a \text{ uniform},\ K = 3",
               "Mezard, Ricci-Tersenghi & Zecchina, J. Stat. Phys. 111, 505 (2003), Eq. (1)"),
    "ksat": (r"\bigwedge_{a=1}^{M} \big(\ell_{a,1}\vee\dots\vee\ell_{a,K}\big),\quad M = "
             r"\alpha N,\ \ell = x_i \text{ or } \bar x_i \text{ (prob. 1/2), K distinct "
             r"uniform variables},\ K = 3", "Mertens, Mezard & Zecchina, Random Struct. "
             "Algorithms 28, 340 (2006), Sec. 2.1"),
}


def model_blocks(spec):
    """The model's definition (cited) and, where derive.LAWS has it, the coupling law's
    exact cumulant generating function K(t) = ln E exp(t J) (sympy, from the code)."""
    from . import derive as D
    out = []
    key = (spec.family, spec.variables) if (spec.family, spec.variables) in MODELS else spec.family
    if key in MODELS:
        body, src = MODELS[key]
        out.append(cited("Hamiltonian and conventions", body, src))
    else:
        out.append(text("Model", f"family {spec.family!r}: no displayed definition"))
    if spec.disorder in D.LAWS and spec.family in ("sk", "hopfield"):
        K, t = sp.Function("K"), sp.Symbol("t", real=True)
        out.append(sym(f"Coupling law {spec.disorder} (derive.LAWS)",
                       sp.Eq(K(t), D.LAWS[spec.disorder].cgf(t, D.N_SYM))))
    return out


# ---- SK --------------------------------------------------------------------------------
def _sk_1rsb_display(b, h, q0, q1, m):
    """The displayed 1RSB -beta f, evaluated as written: nested tanh-sinh quadrature over
    z and y at 12 digits (independent of sk_1rsb's 80 x 80 Gauss-Hermite grid)."""
    with mp.workdps(12):
        b, h, q0, q1, m = (mp.mpf(str(x)) for x in (b, h, q0, q1, m))
        D = lambda f: mp.quad(lambda x: f(x) * mp.exp(-x * x / 2), [-12, 0, 12]) / \
            mp.sqrt(2 * mp.pi)
        inner = lambda z: mp.log(D(lambda y: mp.cosh(b * (h + mp.sqrt(q0) * z
                                                          + mp.sqrt(q1 - q0) * y))**m))
        return (mp.log(2) + b**2 / 4 * (1 - 2 * q1 + (1 - m) * q1**2 + m * q0**2)
                + D(inner) / m)



_PARISI = "Parisi, PRL 43, 1754 (1979); J. Phys. A 13, L115 (1980)"
_AT = "de Almeida & Thouless, J. Phys. A 11, 983 (1978)"


def sk(route):
    if route in ("cavity", "amp"):
        from .latex_amp import sk_amp
        from .latex_cavity import sk_cavity
        return sk_cavity() if route == "cavity" else sk_amp()
    from .families import sk as F
    xi = B * (HF + sp.sqrt(Q) * Z)
    quad = F.quadratic_term("rs").subs(sp.Symbol("q"), Q)
    (q_eq,) = F.rs_equations_symbolic()
    arb = (("1.5", "0", "0.4"), ("0.9", "0.3", "0.5"), ("2", "0.1", "0.8"))
    rsb_pts = tuple((lambda b=b, h=h: (b, h, *(getattr(F.sk_1rsb(b, h), k)
                                               for k in ("q0", "q1", "m"))))
                    for b, h in (("1.5", "0"), ("2", "0"), ("2.5", "0")))
    at_sol = tuple((lambda b=b, h=h: (b, h, F.sk_rs(b, h).q)) for b, h in
                   (("1.5", "0"), ("0.9", "0.3"), ("2", "0.1")))
    return [
        text("RS ansatz", "q_ab = q (a != b); the quadratic term 1 - 2q - lim (1/n) "
             "sum_{a != b} q_ab^2 comes from ansatz.sum_offdiag_power + n_to_zero "
             "(families/sk.py quadratic_term)"),
        sym("RS free energy (n -> 0)", sp.Eq(MINUS_BF, sp.log(2) + B**2 * quad / 4
                                              + GaussE(sp.log(sp.cosh(xi)), Z)),
            Check((B, HF, Q), lambda b, h, q: -mp.mpf(b) * F.sk_rs(b, h).f, at_sol, "rhs")),
        sym("RS saddle-point equation (the map the solver iterates)", q_eq,
            Check((B, HF, Q), lambda b, h, q: F._rs_map(*_mpf(b, h, q)), arb, "rhs")),
        sym("Energy per spin", sp.Eq(sp.Symbol("e"), -B * (1 - Q**2) / 2
                                     - HF * GaussE(sp.tanh(xi), Z)),
            Check((B, HF, Q), lambda b, h, q: F.sk_rs(b, h).e, at_sol, "rhs", slow=True)),
        sym("Replicon (de Almeida-Thouless)", sp.Eq(LAM, 1 - B**2 * GaussE(sp.sech(xi)**4, Z)),
            Check((B, HF, Q), lambda b, h, q: F.sk_replicon(b, h, q=q), arb, "rhs")),
        sym("Paramagnet replicon (q = 0, h = 0): T_c = 1 where it vanishes",
            sp.Eq(sp.Symbol("lambda_0"), (1 - B**2 * GaussE(sp.sech(xi)**4, Z)).subs(
                {Q: 0, HF: 0})),
            Check((B,), lambda b: F.sk_replicon(b, 0, q=0), (("0.5",), ("1",), ("2",)), "rhs")),
        sym("RS ground energy (T = 0: e_0 = -lim beta(1 - q) = -E|z|)",
            sp.Eq(sp.Symbol("e_0^RS"), -GaussE(sp.Abs(Z), Z)),
            Check((), lambda: F.sk_ground_energy("rs"), ((),), "rhs")),
        cited("1RSB free energy (q0 between blocks, q1 inside blocks of size m)",
              r"-\beta f = \ln 2 + \frac{\beta^2}{4}\left[1 - 2 q_1 + (1-m) q_1^2 + m q_0^2"
              r"\right] + \frac{1}{m}\,\mathbb{E}_z \ln \mathbb{E}_y \cosh^m \Xi,\quad "
              r"\Xi = \beta\left(h + \sqrt{q_0}\, z + \sqrt{q_1 - q_0}\, y\right)", _PARISI,
              check=Check(("beta", "h", "q0", "q1", "m"), lambda b, h, *_: -mp.mpf(b) *
                          F.sk_1rsb(b, h).f, rsb_pts, slow=True, display=_sk_1rsb_display,
                          tol="1e-9")),
        cited("1RSB stationarity",
              r"q_1 = \mathbb{E}_z\langle\tanh^2\Xi\rangle,\quad "
              r"q_0 = \mathbb{E}_z\langle\tanh\Xi\rangle^2,\quad "
              r"\langle g\rangle = \frac{\mathbb{E}_y[\cosh^m\Xi\, g]}{\mathbb{E}_y\cosh^m\Xi}",
              _PARISI),
        cited("1RSB at T = 0 (q1 -> 1, m -> 0, mu = beta m fixed)",
              r"E(q_0, \mu) = \frac{\mu(1 - q_0^2)}{4} - \frac{1}{\mu}\,\mathbb{E}_z \ln "
              r"\mathbb{E}_y\, e^{\mu\left|\sqrt{q_0}\, z + \sqrt{1 - q_0}\, y\right|}", _PARISI),
        cited("Full RSB at T = 0: K-step Parisi staircase (families/parisi.py; maximised "
              "over q_0 < ... < q_{K-1} < q_K = 1 and 0 < mu_1 < ... < mu_K, mu_0 = 0)",
              r"E_K = \frac{1}{4}\sum_{k=0}^{K-1}(\mu_{k+1}-\mu_k)(1-q_k^2) - \mathbb{E}_z\,"
              r"\Phi_0\!\left(\sqrt{q_0}\,z\right),\quad \Phi_K(h) = |h|,\quad \Phi_{k-1}(h)"
              r" = \frac{1}{\mu_k}\ln \mathbb{E}_z\, e^{\mu_k \Phi_k\left(h + \sqrt{q_k - "
              r"q_{k-1}}\, z\right)}", _PARISI),
        text("Full RSB: numerical method", "K = 1 is the 1RSB T = 0 functional above. Each "
             "level is maximised (Newton, adjoint gradient, audited Hessian) on a uniform "
             "h-grid; E_K rises with K to the Parisi value, a theorem for SK. E_inf is fitted "
             "as E_K + C K^-s to K = 6, 7, 8, with error estimate = the whole extrapolated "
             "tail plus grid and optimiser terms", _FRSB),
    ]


_FRSB = ("Parisi, J. Phys. A 13, L115 (1980); Talagrand, Ann. Math. 163, 221 (2006); "
         "Panchenko, The Sherrington-Kirkpatrick Model (2013)")


# ---- REM -------------------------------------------------------------------------------
def rem(route):
    from .families import rem as F
    E_, S_ = sp.Symbol("e", real=True), sp.Symbol("s", positive=True)
    fun = F.rem_1rsb_functional()
    bc = lambda s: F.rem_critical_beta(N_scale=s)
    frozen = (("3", "0.5"), ("4", "1"), ("5", "2"))
    return [
        text("Conventions", "e = E/N, N_s = s (energy variance per spin); entropy, beta_c and "
             "the free energy as in families/rem.py"),
        sym("Microcanonical entropy", sp.Eq(sp.Function("s")(E_), sp.log(2) - E_**2 / (2 * S_)),
            source=_REM, check=Check((E_, S_), lambda e, s: F.rem_entropy(e, N_scale=s),
                  (("-0.5", "0.5"), ("0.3", "1"), ("-0.8", "0.5")), "rhs")),
        sym("Critical inverse temperature (s(e_0) = 0, beta_c = s'(e_0))",
            sp.Eq(sp.Symbol("beta_c"), sp.sqrt(2 * sp.log(2) / S_)),
            Check((S_,), lambda s: bc(s), (("0.5",), ("1",), ("2",)), "rhs")),
        sym("Paramagnet, beta <= beta_c (annealed = quenched)",
            sp.Eq(MINUS_BF, sp.log(2) + S_ * B**2 / 2),
            Check((B, S_), lambda b, s: -mp.mpf(b) * F.rem_free_energy(b, N_scale=s),
                  (("1", "0.5"), ("0.5", "1"), ("0.3", "2")), "rhs")),
        sym("1RSB functional (blocks of m replicas share a configuration; n -> 0)",
            sp.Eq(MINUS_BF, fun),
            Check(tuple(sorted(fun.free_symbols, key=str)),
                  lambda b, m, s: -mp.mpf(b) * F.rem_1rsb(b, N_scale=s).f,
                  lambda: tuple((b, bc(s) / mp.mpf(b), s) for b, s in frozen), "rhs")),
        sym("Frozen phase, beta > beta_c (m = beta_c / beta)",
            sp.Eq(sp.Symbol("f"), -sp.sqrt(2 * S_ * sp.log(2))),
            Check((B, S_), lambda b, s: F.rem_free_energy(b, N_scale=s), frozen, "rhs")),
    ]


# ---- perceptron (Gardner) --------------------------------------------------------------
def perceptron(route):
    if route == "cavity":
        from .latex_cavity import perceptron_cavity
        return perceptron_cavity()
    from .families import perceptron as F
    x = (KA - sp.sqrt(Q) * T) / sp.sqrt(1 - Q)
    E = _phi(x) / _H(x)
    (q_eq,) = F.rs_equations_symbolic("spherical")
    pt = lru_cache(maxsize=None)(lambda q, k: F.gardner_rs_point("spherical", q, k))
    pts = tuple((lambda q=q, k=k: (q, k, pt(q, k).alpha)) for q, k in
                (("0.3", "0"), ("0.5", "0.5"), ("0.7", "1")))
    return [
        text("Conventions", "x(t) = (kappa - sqrt(q) t)/sqrt(1 - q), H(x) = int_x^inf Dz, "
             "E(x) = phi(x)/H(x); the RS saddle is parametrised by q (alpha(q) explicit)"),
        sym("RS entropy (spherical couplings)",
            sp.Eq(sp.Symbol("S"), sp.log(1 - Q) / 2 + Q / (2 * (1 - Q))
                  + A * GaussE(sp.log(_H(x)), T)), source=_GARDNER,
            check=Check((Q, KA, A), lambda q, k, a: pt(q, k).S, pts, "rhs")),
        sym("RS saddle-point equation", q_eq, Check((Q, KA, A), lambda q, k, a: 0, pts,
                                                    "residual")),
        sym("Replicon", sp.Eq(LAM, 1 - A * GaussE((E * (E - x))**2, T)),
            Check((Q, KA, A), lambda q, k, a: pt(q, k).replicon, pts, "rhs")),
        sym("Capacity (q -> 1): alpha_c = 1/D(kappa)",
            sp.Eq(sp.Symbol("alpha_c"), 1 / ((1 + KA**2) * _H(-KA) + KA * _phi(KA))),
            Check((KA,), lambda k: F.gardner_capacity("spherical", k).alpha_c,
                  (("0",), ("0.5",), ("1",)), "rhs")),
    ]


# ---- teacher-student -------------------------------------------------------------------
_BAYES = {}       # str(eps_Gibbs) -> eps_Bayes at the same alpha (filled with the points)


def _gibbs_point(F, a):
    g = F.teacher_student_error(a, "gibbs")
    _BAYES[str(g)] = F.teacher_student_error(a, "bayes")
    return (g,)


def teacher_student(route):
    if route == "amp":
        from .latex_amp import ts_amp
        return ts_amp()
    from .families import teacher_student as F
    Rs = sp.Symbol("R", positive=True)
    EG = sp.Symbol("epsilon_G", positive=True)
    bayes_of = _BAYES
    I = GaussE(sp.exp(-T**2 / 2) / _H(-T), T)
    curve = lambda: tuple((mp.cos(mp.pi * F.teacher_student_error(a, "gibbs")), a)
                          for a in ("1", "2", "5"))
    return [
        cited("RS entropy with the teacher as an extra replica",
              r"s(q, R) = \tfrac12\ln(1-q) + \frac{q - R^2}{2(1-q)} + 2\alpha\,\mathbb{E}_t\!"
              r"\left[H\!\left(-\frac{R t}{\sqrt{q - R^2}}\right)\ln H\!\left(-\frac{\sqrt q\, t}"
              r"{\sqrt{1-q}}\right)\right]", _TS),
        sym("Gibbs learning on the Nishimori line q = R (stationarity)",
            sp.Eq(Rs / sp.sqrt(1 - Rs), A / sp.pi * GaussE(sp.exp(-Rs * T**2 / 2)
                                                           / _H(-sp.sqrt(Rs) * T), T)),
            Check((Rs, A), lambda r, a: 0, curve, "residual", slow=True)),
        cited("Generalization error", r"\epsilon = \frac{\arccos R}{\pi}", _TS),
        sym("Bayes from Gibbs: R_Bayes = sqrt(R_Gibbs), with R = cos(pi epsilon)",
            sp.Eq(sp.Symbol("epsilon_B"), sp.acos(sp.sqrt(sp.cos(sp.pi * EG))) / sp.pi),
            source="Opper & Haussler, PRL 66, 2677 (1991)",
            check=Check((EG,), lambda e: bayes_of[str(e)],
                        tuple((lambda a=a: _gibbs_point(F, a)) for a in ("1", "2", "5")),
                        "rhs", slow=True)),
        sym("Large alpha, Gibbs", sp.Eq(sp.Symbol("alpha epsilon_G"), sp.sqrt(2) / I),
            Check((), lambda: F.asymptotic_constant("gibbs"), ((),), "rhs")),
        sym("Large alpha, Bayes", sp.Eq(sp.Symbol("alpha epsilon_B"), 1 / I),
            Check((), lambda: F.asymptotic_constant("bayes"), ((),), "rhs")),
    ]


# ---- Hopfield --------------------------------------------------------------------------
def hopfield(route):
    if route == "cavity":
        from .latex_cavity import hopfield_cavity
        return hopfield_cavity()
    from .families import hopfield as F
    xi = B * (M + sp.sqrt(A * R) * Z)
    D_ = 1 - B * (1 - Q)
    eq_m, eq_q, eq_r = F.rs_equations_symbolic()
    arb = (("0.9", "0.8", "0.5", "0.05", "3"), ("0.5", "0.6", "1.2", "0.1", "2"),
           ("0.95", "0.9", "0.3", "0.12", "5"))
    g = lambda *p: F.hopfield_gradient(*p)
    args = (M, Q, R, A, B)
    y = sp.Symbol("y", positive=True)
    return [
        sym("AGS RS free energy per spin (one condensed pattern, C = beta(1 - q))",
            sp.Eq(sp.Symbol("f"), A / 2 + M**2 / 2 + A / (2 * B) * (sp.log(D_) - B * Q / D_)
                  + A * B * R * (1 - Q) / 2 - GaussE(sp.log(2 * sp.cosh(xi)), Z) / B),
            Check(args, F.hopfield_free_energy, arb, "rhs")),
        sym("Stationarity in m", eq_m, Check(args, lambda m, q, r, a, b:
                                             mp.mpf(m) - g(m, q, r, a, b)[0], arb, "rhs")),
        sym("Stationarity in r", eq_q, Check(args, lambda m, q, r, a, b: mp.mpf(q) + 2 *
                                             g(m, q, r, a, b)[2] / (mp.mpf(a) * mp.mpf(b)),
                                             arb, "rhs")),
        sym("Stationarity in q", eq_r, Check(args, lambda m, q, r, a, b: mp.mpf(r) + 2 *
                                             g(m, q, r, a, b)[1] / (mp.mpf(a) * mp.mpf(b)),
                                             arb, "rhs")),
        sym("Replicon (AGS PRL Eq. (13), normalised by (1 - C)^2)",
            sp.Eq(LAM, 1 - A * B**2 * GaussE(sp.sech(xi)**4, Z) / D_**2),
            Check(args, lambda m, q, r, a, b: F._replicon(*_mpf(a, b, m, q, r)), arb, "rhs")),
        cited("T = 0 reduction (AGS Eqs. (9)-(10)), y = m / sqrt(2 alpha r)",
              r"m = \operatorname{erf}(y),\quad C = \frac{2}{\sqrt\pi}\frac{y}{m}e^{-y^2},\quad "
              r"r = (1 - C)^{-2}\ \Longrightarrow\ \sqrt{2\alpha} = g(y)", _AGS),
        sym("The function g; alpha_c = g(y*)^2/2 at its maximum g'(y*) = 0",
            sp.Eq(sp.Function("g")(y), sp.erf(y) / y - 2 / sp.sqrt(sp.pi) * sp.exp(-y**2)),
            Check((y,), lambda yy: F._g(mp.mpf(yy)), (("0.5",), ("1",), ("1.5",)), "rhs")),
        sym("T = 0 energy (AGS Eq. (11)), at the T = 0 retrieval solution (m, r)",
            sp.Eq(sp.Symbol("E"), A * (1 - R) / 2 - M**2 / 2), source=_AGS,
            check=Check((A, M, R), lambda a, m, r: F.hopfield_t0(a).E,
                        tuple((lambda a=a: (a, F.hopfield_t0(a).m, F.hopfield_t0(a).r))
                              for a in ("0.05", "0.1", "0.13")), "rhs")),
    ]


def _pspin(name):
    def blocks(route):
        from . import latex_pspin
        return getattr(latex_pspin, name)(route)
    return blocks


_ENTRIES = {"rem": rem, "sk": sk, "pspin": _pspin("pspin"), "ising_pspin": _pspin("ising_pspin"),
            "perceptron": perceptron, "teacher_student": teacher_student, "hopfield": hopfield}


@lru_cache(maxsize=None)
def derivation(name, route):
    """[TexBlock] for catalog entry `name` on `route` (a Refusal if none is written);
    built once per process (the blocks are immutable; callers must not mutate the list)."""
    from .core import Refusal
    if name == "viana_bray":
        from .latex_cavity import viana_bray
        return viana_bray()
    if name in ("xorsat", "ksat"):
        from . import latex_csp
        return getattr(latex_csp, name)(route)
    mod = {"ldpc_bec": "inference", "series_ising2d": "series", "ridge": "learning",
           "spiked_wigner": "spiked_sbm", "sbm": "spiked_sbm", "ising2d": "lattice", "ising1d": "lattice",
           "bethe_ising": "lattice", "ising2d_anisotropic": "lattice_geom", "ising_triangular":
           "lattice_geom", "ising_honeycomb": "lattice_geom", "sk_bounds": "bounds", "online_perceptron": "online",
           "online_linear": "online_linear", "online_drift": "online_drift", "online_tcm": "online_tcm",
           "online_scm": "online_scm", "percolation_bethe": "percolation",
           "percolation_graph": "percolation", "percolation_lattice": "percolation_lattice"}.get(name)
    if mod is not None:
        from importlib import import_module
        return getattr(import_module(f".latex_{mod}", __package__), name)(route)
    if name not in _ENTRIES:
        raise Refusal(f"missing-step:derivation:{name}", "no displayed derivation for "
                      f"catalog entry {name!r} (latex_entries.py)")
    return _ENTRIES[name](route)
