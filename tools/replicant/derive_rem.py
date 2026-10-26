"""Level (a) for the Random Energy Model: the replica derivation of -beta f, step by step.

  rem_steps(beta=None, N_scale="0.5", pair_sum=None) -> [core.Step]   seven steps
  rem_level_a(params) -> Trace          the steps run at a catalog run's params (cached)
  compare_derrida(phi, mstar, beta, N_scale) -> (f_derived, f_derrida, branch, m)

Every displayed equation is a sympy object a step computed from the previous step's
State (tex.sym renders it; nothing is retyped), and every step checks what it computed
by an independent route, refusing (`check:<name>`) on a mismatch:

  1 rem-replicate        E ln Z = lim (1/n) ln E Z^n (sympy.limit of (Z^n - 1)/n);
                         Z^n as a sum over n-tuples (checked by expansion at n = 2, 3)
  2 rem-average-disorder the Gaussian MGF by sympy.integrate; for a fixed tuple
                         sum_a E_{i_a} = sum_j c_j E_j, so ln E exp(-beta sum_a E_{i_a})
                         = K(-beta) sum_j c_j^2 = K(-beta) sum_{a,b} q_ab, q_ab =
                         delta_{i_a i_b} (the count identity checked over all tuples)
  3 rem-rs-ansatz        q_ab = delta_ab (all replicas distinct) -> ln 2 + s beta^2/2;
                         q_ab = 1 (one configuration) -> ln 2/n + s beta^2 n/2, which
                         ansatz.n_to_zero refuses and sympy.limit sends to oo: shown, unphysical
  4 rem-1rsb-blocks      n/m blocks of m replicas on one configuration: (2^N)^(n/m)
                         tuples (falling-factorial correction O(2^-N), sympy series) and
                         sum_ab q_ab = n + ansatz.sum_offdiag_power (= n m), checked against
                         ansatz.rsb_matrix at integer n
  5 rem-replica-limit    Phi(m) = ansatz.n_to_zero(...) = ln 2/m + s beta^2 m/2, checked
                         equal to families/rem.py rem_1rsb_functional
  6 rem-extremize        dPhi/dm = 0 -> m* = beta_c/beta (convex: a minimum, the n < 1 rule);
                         m* >= 1 -> m = 1 (paramagnet), else frozen: Phi(m*) = beta
                         sqrt(2 s ln 2), f = e0, e = -dPhi/dbeta = e0, entropy 0
  7 rem-derrida-check    Derrida's microcanonical s(e) = ln 2 - e^2/(2 s) (sympy): e0 and
                         beta_c = s'(e0) equal the replica ones symbolically; at the requested
                         beta the derived f equals families/rem.py rem_free_energy

`pair_sum(n, m)` is the hook for sum_{a,b} q_ab of the 1RSB blocks (default: n plus
ansatz.sum_offdiag_power); a planted wrong count is caught by step 4 (tests).  Physics
level: the n -> 0 continuation and the block ansatz are not proved; the free energy is
Derrida's theorem, which is what step 7 compares with.
"""
import itertools
from functools import lru_cache

import mpmath as mp
import sympy as sp

from .ansatz import n_to_zero, rsb_matrix, sum_offdiag_power
from .core import Refusal, State, Step, run_pipeline
from .tex import sym, text

MPV = "Mezard, Parisi & Virasoro, Spin Glass Theory and Beyond (1987)"
DERRIDA = "Derrida, PRL 45, 79 (1980); Phys. Rev. B 24, 2613 (1981)"
DM = "Derrida & Mottishaw, arXiv:1410.1432 (m = T/T_c)"
LEVEL_A = "replicant level (a) derivation (this package)"

# the symbols of families/rem.py rem_1rsb_functional (same assumptions: equal objects)
n, m, beta, s, p = sp.symbols("n m beta s p", positive=True)
N = sp.Symbol("N", positive=True, integer=True)
Z = sp.Symbol("Z", positive=True)
t, z, x, e = (sp.Symbol(k, real=True) for k in ("t", "z", "x", "e"))
c = sp.Symbol("c_j", nonnegative=True)                 # occupation count of configuration j
a, b, j = sp.symbols("a b j", integer=True, positive=True)
En, i, q = sp.IndexedBase("E"), sp.IndexedBase("i"), sp.IndexedBase("q")
EX = sp.Function(r"\mathbb{E}")                       # the disorder average (display head)
PHI_N, PHI = sp.Symbol("Phi_n"), sp.Function("Phi")
BETA_C, MSTAR, E0 = sp.Symbol("beta_c"), sp.Symbol("m^*"), sp.Symbol("e_0")
F_, E_, S_ = sp.Symbol("f"), sp.Symbol("e"), sp.Symbol("S")
K2N = 2**N


def _check(ok, name, detail):
    if not ok:
        raise Refusal(f"check:{name}", detail)


def _zero(expr):
    return sp.simplify(expr) == 0


def _qsum():
    return sp.Sum(q[a, b], (b, 1, n), (a, 1, n))


def _tuple_sum(body):
    return sp.Sum(body, (i[n], 1, K2N), (i[1], 1, K2N))


# ---- 1: replicate ------------------------------------------------------------------------
def _replicate(st):
    lim = sp.limit((Z**n - 1) / n, n, 0)
    _check(lim == sp.log(Z), "replica-identity", f"lim (Z^n - 1)/n = {lim}, not ln Z")
    # Z^n = sum over n-tuples of configurations of exp(-beta sum_a E_{i_a}): expansion check
    for nn, K in ((2, 3), (3, 3)):
        w = sp.symbols(f"w0:{K}")
        tuples = sum(sp.Mul(*(w[k] for k in tup))
                     for tup in itertools.product(range(K), repeat=nn))
        _check(sp.expand(sum(w)**nn - tuples) == 0, "tuple-expansion",
               f"(sum_j w_j)^{nn} != the sum over {nn}-tuples")
    zdef = sp.Eq(Z, sp.Sum(sp.exp(-beta * En[j]), (j, 1, K2N)))
    zn = sp.Eq(Z**n, _tuple_sum(sp.exp(-beta * sp.Sum(En[i[a]], (a, 1, n)))))
    return st.with_(rem_identity=(sp.Eq(sp.Limit((Z**n - 1) / n, n, 0), lim),
                                  sp.Eq(EX(sp.log(Z)), sp.Limit(sp.log(EX(Z**n)) / n, n, 0))),
                    rem_tuples=(zdef, zn))


def _replicate_tex(old, new):
    lim_eq, ident = new["rem_identity"]
    zdef, zn = new["rem_tuples"]
    return [text("Replicas", "2^N configurations j with i.i.d. energies E_j ~ N(0, s N) "
                 "(s = N_scale); n replicas a = 1..n each pick a configuration i_a.  The "
                 "identity below is (Z^n - 1)/n -> ln Z, evaluated by sympy.limit; the "
                 "tuple sum is checked by expanding (sum_j w_j)^n at n = 2, 3"),
            sym("Replica limit (sympy.limit)", lim_eq),
            sym("Replica identity (n -> 0 continuation, physics level)", ident),
            sym("Partition function", zdef),
            sym("Replicated partition function: a sum over n-tuples (i_1, ..., i_n)", zn)]


# ---- 2: disorder average -----------------------------------------------------------------
def _average(st):
    v = s * N                                            # energy variance
    mgf = sp.simplify(sp.integrate(sp.exp(t * sp.sqrt(v) * z - z**2 / 2), (z, -sp.oo, sp.oo))
                      / sp.sqrt(2 * sp.pi))
    cgf = sp.expand(sp.expand_log(sp.log(mgf), force=True))
    _check(_zero(cgf - v * t**2 / 2), "gaussian-cgf", f"ln E exp(t E) = {cgf}")
    # a configuration chosen by c_j replicas enters with t = -beta c_j: quadratic in c_j
    per_j = sp.expand(cgf.subs(t, -beta * c))
    weight = sp.expand(cgf.subs(t, -beta))               # the weight of one unit of sum q_ab
    _check(_zero(per_j - c**2 * weight), "quadratic-in-count", f"K(-beta c_j) = {per_j}")
    # sum_j c_j^2 = sum_{a,b} delta_{i_a i_b} for every tuple (n = 3, 4 replicas, 3 configs)
    for nn in (3, 4):
        for tup in itertools.product(range(3), repeat=nn):
            cnt = sum(tup.count(k)**2 for k in range(3))
            pairs = sum(1 for aa in tup for bb in tup if aa == bb)
            _check(cnt == pairs, "count-identity", f"tuple {tup}: {cnt} != {pairs}")
    fixed = sp.Eq(EX(sp.exp(-beta * sp.Sum(En[i[a]], (a, 1, n)))), sp.exp(weight * _qsum()))
    avg = sp.Eq(EX(Z**n), _tuple_sum(sp.exp(weight * _qsum())))
    return st.with_(rem_avg=(sp.Eq(EX(sp.exp(t * En[j])), mgf),
                             sp.Eq(sp.log(EX(sp.exp(-beta * c * En[j]))), per_j), fixed, avg),
                    rem_weight=weight)


def _average_tex(old, new):
    mgf, per_j, fixed, avg = new["rem_avg"]
    return [text("Disorder average", "for a fixed tuple the energy sum over the replicas is "
                 "sum_j c_j E_j with c_j = #{a: i_a = j}; the E_j are independent Gaussians, so the log-average is "
                 "sum_j K(-beta c_j), quadratic in c_j, and sum_j c_j^2 = sum_{a,b} q_ab with "
                 "q_ab = 1 if i_a = i_b and 0 otherwise (checked over every tuple of 3 and 4 "
                 "replicas on 3 configurations)"),
            sym("Gaussian moment generating function (sympy.integrate)", mgf),
            sym("One configuration chosen by c_j replicas", per_j),
            sym("Fixed tuple: the overlaps q_ab in {0, 1}", fixed),
            sym("Averaged replicated partition function", avg)]


# ---- 3: RS ansaetze ----------------------------------------------------------------------
def _per_n(entropy, pairs, st):
    """(1/N) ln E Z^n for a tuple class: ln(#tuples)/N + (weight/N) sum_ab q_ab."""
    return sp.expand(entropy + st["rem_weight"] / N * pairs)


def _rs(st):
    log2N = lambda count: sp.expand(sp.expand_log(sp.log(count), force=True) / N)
    distinct = n + sum_offdiag_power("rs", [], [sp.Integer(0)], p, n)       # q_ab = delta_ab
    same = n + sum_offdiag_power("rs", [], [sp.Integer(1)], p, n)           # q_ab = 1
    para_n = _per_n(log2N(K2N**n), distinct, st)     # ff(2^N, n) ~ (2^N)^n (step 4: O(2^-N))
    frozen_n = _per_n(log2N(K2N), same, st)
    para = n_to_zero(para_n, n)
    try:
        n_to_zero(frozen_n, n)
        why = None
    except Refusal as r:
        why = f"{r.reason}: {r.detail}"
    _check(why is not None, "rs-one-configuration", "q_ab = 1 should not continue to n -> 0")
    per_rep = sp.expand(frozen_n / n)
    div = sp.limit(per_rep, n, 0, "+")
    _check(div == sp.oo, "rs-one-configuration", f"lim = {div}")
    return st.with_(rem_rsym=dict(distinct=distinct, same=same, para_n=para_n, para=para,
                                frozen_n=frozen_n, per_rep=per_rep, div=div, why=why))


def _rs_tex(old, new):
    R = new["rem_rsym"]
    return [text("Replica-symmetric ansaetze", "q_ab = delta_ab: every replica on its own "
                 "configuration (2^N (2^N - 1) ... ~ (2^N)^n tuples); q_ab = 1: all replicas on "
                 "one configuration (2^N tuples).  sum_{a,b} q_ab = n + ansatz.sum_offdiag_power"),
            sym("q_ab = delta_ab: pair sum", sp.Eq(_qsum(), R["distinct"])),
            sym("q_ab = delta_ab: (1/N) ln E Z^n", sp.Eq(PHI_N, R["para_n"])),
            sym("q_ab = delta_ab, n -> 0 (ansatz.n_to_zero): the paramagnet",
                sp.Eq(-beta * F_, R["para"])),
            sym("q_ab = 1: pair sum", sp.Eq(_qsum(), R["same"])),
            sym("q_ab = 1: (1/N) ln E Z^n", sp.Eq(PHI_N, R["frozen_n"])),
            sym("q_ab = 1, per replica (sympy.limit n -> 0+ diverges)",
                sp.Eq(sp.Limit(PHI_N / n, n, 0, "+"), sp.Limit(R["per_rep"], n, 0, "+").doit())),
            text("q_ab = 1 refused", "ansatz.n_to_zero refuses it (" + R["why"] + "): ln 2 / n "
                 "diverges as n -> 0, so the single-configuration RS saddle is unphysical; the "
                 "frozen phase needs the block ansatz below")]


# ---- 4: 1RSB blocks ----------------------------------------------------------------------
def default_pair_sum(n_, m_):
    """sum_{a,b} q_ab for n/m blocks of m replicas sharing a configuration: the diagonal n
    plus the off-diagonal 1RSB sum of ansatz (q1 = 1 in a block, q0 = 0 between)."""
    return n_ + sum_offdiag_power("rsb", [m_], [sp.Integer(0), sp.Integer(1)], p, n_)


def _rsb(pair_sum):
    def apply(st):
        pairs = sp.expand(pair_sum(n, m))
        for nn, mm in ((4, 2), (6, 2), (6, 3), (8, 4)):          # brute force at integer n
            brute = sum(rsb_matrix([mm], [0, 1], nn))
            got = pairs.subs({n: nn, m: mm})
            _check(got == brute, "pair-count", f"n = {nn}, m = {mm}: pair_sum gives {got}, "
                   f"ansatz.rsb_matrix sums to {brute}")
        # tuple count: n/m distinct configurations; the falling factorial vs (2^N)^(n/m)
        X = sp.Symbol("x", positive=True)                         # x = 2^-N
        corr = {}
        for g in (2, 3, 4, 5):
            ser = sp.series(sp.log(sp.Mul(*(1 - k * X for k in range(g)))), X, 0, 2).removeO()
            _check(_zero(ser + sp.Rational(g * (g - 1), 2) * X), "falling-factorial",
                   f"g = {g}: {ser}")
            corr[g] = ser
        ent = sp.expand(sp.expand_log(sp.log(K2N**(n / m)), force=True) / N)
        phin = _per_n(ent, pairs, st)
        return st.with_(rem_blocks=dict(pairs=pairs, ent=ent, phin=phin, corr3=corr[3], X=X))
    return apply


def _rsb_tex(old, new):
    R = new["rem_blocks"]
    K, X = sp.Symbol("K", positive=True), R["X"]
    return [text("One-step replica symmetry breaking", "n/m blocks of m replicas; the replicas "
                 "of a block share one configuration (q1 = 1), different blocks different "
                 "ones (q0 = 0).  The n/m configurations give 2^N (2^N - 1) ... ~ (2^N)^(n/m) "
                 "tuples; the falling-factorial correction is O(2^-N) (shown for 3 blocks, "
                 "K = 2^N; checked for 2 to 5 blocks) and drops out of (1/N) ln.  The pair "
                 "sum is checked against ansatz.rsb_matrix at integer n"),
            sym("Falling-factorial correction, 3 blocks (sympy series in 1/K)",
                sp.Eq(sp.log(sp.ff(K, 3) / K**3), R["corr3"].subs(X, 1 / K) +
                      sp.Order(1 / K**2, (K, sp.oo)))),
            sym("1RSB pair sum (n + ansatz.sum_offdiag_power)", sp.Eq(_qsum(), R["pairs"])),
            sym("1RSB: (1/N) ln E Z^n", sp.Eq(PHI_N, R["phin"]))]


# ---- 5: n -> 0 ---------------------------------------------------------------------------
def _limit(st):
    from .families.rem import rem_1rsb_functional
    phi = n_to_zero(st["rem_blocks"]["phin"], n)
    fam = rem_1rsb_functional()
    _check(_zero(phi - fam), "family-functional",
           f"derived Phi(m) = {phi}, families/rem.py rem_1rsb_functional = {fam}")
    _check(_zero(phi.subs(m, 1) - st["rem_rsym"]["para"]), "m-equals-1-paramagnet",
           f"Phi(1) = {phi.subs(m, 1)}")
    return st.with_(rem_Phim=phi)


def _limit_tex(old, new):
    phi = new["rem_Phim"]
    return [text("Continuation n -> 0", "Phi(m) = lim_{n->0} Phi_n / n by ansatz.n_to_zero; "
                 "equal (sympy) to families/rem.py rem_1rsb_functional, and Phi(1) is the "
                 "q_ab = delta_ab paramagnet of step 3"),
            sym("1RSB functional per replica and spin", sp.Eq(PHI(m), phi))]


# ---- 6: extremize ------------------------------------------------------------------------
def _extremize(beta_val, s_val):
    def apply(st):
        phi = st["rem_Phim"]
        d1 = sp.diff(phi, m)
        roots = sp.solve(sp.Eq(d1, 0), m)
        _check(len(roots) == 1, "stationary-m", f"roots {roots}")
        mstar = roots[0]
        bc = sp.sqrt(2 * sp.log(2) / s)
        _check(_zero(mstar - bc / beta), "m-star", f"m* = {mstar}, not beta_c/beta")
        d2 = sp.diff(phi, m, 2)
        _check(d2.is_positive is True, "convexity", f"d2 Phi/dm2 = {d2}")
        # frozen branch (m* <= 1)
        phis = sp.simplify(phi.subs(m, mstar))
        _check(_zero(phis - beta * sp.sqrt(2 * s * sp.log(2))), "frozen-phi", str(phis))
        e_m = sp.expand(-sp.diff(phi, beta))                    # e at fixed m (dPhi/dm = 0)
        e_fr = sp.simplify(e_m.subs(m, mstar))
        f_fr = sp.simplify(-phis / beta)
        s_fr = sp.simplify(phis + beta * e_fr)
        _check(_zero(s_fr) and _zero(e_fr - f_fr), "frozen-energy",
               f"e = {e_fr}, f = {f_fr}, entropy = {s_fr}")
        # paramagnet branch (m* >= 1: the minimum over 0 < m <= 1 sits at m = 1)
        phi1 = phi.subs(m, 1)
        e_pa = sp.expand(-sp.diff(phi1, beta))
        s_pa = sp.expand(phi1 + beta * e_pa)
        branch = mval = None
        if beta_val is not None:
            with mp.workdps(30):
                mv = sp.N(mstar.subs({beta: sp.Rational(str(beta_val)),
                                      s: sp.Rational(str(s_val))}), 30)
                branch = "paramagnet" if mv >= 1 else "frozen"
                mval = sp.Integer(1) if branch == "paramagnet" else mv
        return st.with_(rem_ext=dict(d1=d1, mstar=mstar, bc=bc, d2=d2, phis=phis, e_m=e_m,
                                     e_fr=e_fr, f_fr=f_fr, s_fr=s_fr, phi1=phi1, e_pa=e_pa,
                                     s_pa=s_pa, branch=branch, m=mval))
    return apply


def _extremize_tex(old, new):
    X = new["rem_ext"]
    out = [text("Extremization over 0 < m <= 1", "for n < 1 the 1RSB -beta f is the MINIMUM "
                "of Phi(m) over 0 < m <= 1 (the n -> 0 inversion of the saddle); Phi is convex "
                "in m, so the minimum is the stationary point m* when m* <= 1 (frozen phase) "
                "and m = 1 when m* >= 1 (paramagnet).  e = -dPhi/dbeta at fixed m "
                "(stationarity in m), entropy = Phi + beta e"),
           sym("Stationarity", sp.Eq(sp.Derivative(PHI(m), m), X["d1"])),
           sym("Stationary block size (beta_c = sqrt(2 ln 2 / s))",
               sp.Eq(MSTAR, sp.Eq(X["mstar"], BETA_C / beta))),
           sym("Convexity", sp.Eq(sp.Derivative(PHI(m), (m, 2)), X["d2"])),
           sym("Frozen phase (m* < 1): -beta f = Phi(m*)", sp.Eq(-beta * F_, X["phis"])),
           sym("Frozen phase: energy, free energy and entropy per spin",
               [sp.Eq(E_, X["e_fr"]), sp.Eq(F_, X["f_fr"]), sp.Eq(S_, X["s_fr"])]),
           sym("Paramagnet (m* >= 1, m = 1): -beta f, energy, entropy",
               [sp.Eq(-beta * F_, X["phi1"]), sp.Eq(E_, X["e_pa"]), sp.Eq(S_, X["s_pa"])])]
    if X["branch"] is not None:
        out.append(text("At the requested beta", f"branch: {X['branch']}, m = "
                        f"{sp.N(X['m'], 15)}"))
    return out


# ---- 7: Derrida --------------------------------------------------------------------------
def compare_derrida(phi, mstar, beta_val, s_val):
    """(f derived, f Derrida, branch, m): the derived Phi minimised over 0 < m <= 1 at
    (beta, N_scale), against families/rem.py rem_free_energy (Derrida's direct result)."""
    from .families.rem import rem_free_energy
    with mp.workdps(30):
        sub = {beta: sp.Rational(str(beta_val)), s: sp.Rational(str(s_val))}
        mv = sp.N(mstar.subs(sub), 30)
        mm = 1 if mv >= 1 else mv
        f = -sp.N(phi.subs(sub).subs(m, mm), 30) / sub[beta]
        fd = rem_free_energy(mp.mpf(str(beta_val)), N_scale=mp.mpf(str(s_val)))
        return (mp.mpf(str(sp.N(f, 30))), fd, "paramagnet" if mv >= 1 else "frozen",
                mp.mpf(str(sp.N(mm, 30))))


def _derrida(beta_val, s_val):
    def apply(st):
        from .families.rem import rem_entropy
        X = st["rem_ext"]
        sent = sp.log(2) - e**2 / (2 * s)
        with mp.workdps(30):
            got = sp.N(sent.subs({e: sp.Rational(-3, 10), s: sp.Rational(1, 2)}), 30)
            want = rem_entropy(mp.mpf("-0.3"), N_scale=mp.mpf("0.5"))
            _check(abs(mp.mpf(str(got)) - want) < mp.mpf("1e-25"), "entropy-family",
                   f"{got} vs {want}")
        roots = [r for r in sp.solve(sp.Eq(sent, 0), e) if r.is_negative]
        _check(len(roots) == 1, "ground-energy", str(roots))
        e0 = roots[0]
        bcm = sp.simplify(sp.diff(sent, e).subs(e, e0))         # beta_c = s'(e0)
        _check(_zero(bcm - X["bc"]), "beta-c", f"microcanonical beta_c = {bcm}")
        _check(_zero(e0 - X["f_fr"]), "frozen-f-is-e0", f"e0 = {e0}, f = {X['f_fr']}")
        num = None
        if beta_val is not None:
            f, fd, branch, mm = compare_derrida(st["rem_Phim"], X["mstar"], beta_val, s_val)
            diff = abs(f - fd)
            _check(diff <= mp.mpf("1e-25") * max(1, abs(fd)), "derrida-free-energy",
                   f"derived f = {f}, rem_free_energy = {fd}")
            num = dict(beta=beta_val, s=s_val, f=f, fd=fd, diff=diff, branch=branch, m=mm)
        return st.with_(rem_derrida=dict(sent=sent, e0=e0, bc=bcm, num=num))
    return apply


def _derrida_tex(old, new):
    D = new["rem_derrida"]
    out = [text("Derrida's direct solution", "microcanonical entropy (families/rem.py "
                "rem_entropy): its negative root e0 is the ground-state energy and beta_c = "
                "s'(e0) (the slope where the entropy vanishes).  sympy checks both equal the "
                "replica beta_c and the frozen f of step 6"),
           sym("Microcanonical entropy", sp.Eq(sp.Function("s")(e), D["sent"])),
           sym("Ground-state energy and critical inverse temperature",
               [sp.Eq(E0, D["e0"]), sp.Eq(BETA_C, D["bc"])])]
    u = D["num"]
    if u is not None:
        out.append(text("Agreement at the requested parameters", f"beta = {u['beta']}, "
                        f"N_scale = {u['s']}: branch {u['branch']}, m = {mp.nstr(u['m'], 15)}; "
                        f"replica f = {mp.nstr(u['f'], 20)}, Derrida rem_free_energy = "
                        f"{mp.nstr(u['fd'], 20)}, |difference| = {mp.nstr(u['diff'], 3)} "
                        "(30-digit evaluation, tolerance 1e-25)"))
    return out


# ---- pipeline ----------------------------------------------------------------------------
def rem_steps(beta=None, N_scale="0.5", pair_sum=None):
    """The seven level-(a) Steps; beta (a decimal string or number) selects the branch and
    turns on the numerical comparison of step 7."""
    if beta is not None and not mp.mpf(str(beta)) > 0:
        raise Refusal("precondition:positive-beta", f"beta = {beta}")
    if not mp.mpf(str(N_scale)) > 0:
        raise Refusal("precondition:positive-N_scale", f"N_scale = {N_scale}")
    pair_sum = pair_sum or default_pair_sum
    has = lambda *ks: (lambda st: [k for k in ks if k not in st])
    note = lambda k: (lambda st: k)
    return [
        Step("rem-replicate", has(), _replicate, f"{LEVEL_A}; {MPV}",
             note("E ln Z = lim (1/n) ln E Z^n; Z^n over n-tuples"), _replicate_tex),
        Step("rem-average-disorder", has("rem_tuples"), _average, f"{LEVEL_A}; {MPV}",
             note("Gaussian average of a fixed tuple: exp(s beta^2 N/2 sum_ab q_ab)"),
             _average_tex),
        Step("rem-rs-ansatz", has("rem_weight"), _rs, f"{LEVEL_A}; {MPV}",
             note("q_ab = delta_ab: paramagnet; q_ab = 1: refused (diverges as n -> 0)"),
             _rs_tex),
        Step("rem-1rsb-blocks", has("rem_weight"), _rsb(pair_sum), f"{LEVEL_A}; {MPV}",
             note("n/m blocks of m: (2^N)^(n/m) tuples, sum_ab q_ab = n m"), _rsb_tex),
        Step("rem-replica-limit", has("rem_blocks", "rem_rsym"), _limit, f"{LEVEL_A}; {MPV}",
             note("Phi(m) = ln 2/m + s beta^2 m/2 (= rem_1rsb_functional)"), _limit_tex),
        Step("rem-extremize", has("rem_Phim"), _extremize(beta, N_scale), f"{LEVEL_A}; {DM}",
             note("m* = beta_c/beta; m = 1 above T_c, frozen below"), _extremize_tex),
        Step("rem-derrida-check", has("rem_ext"), _derrida(beta, N_scale), f"{LEVEL_A}; {DERRIDA}",
             note("agrees with Derrida's microcanonical solution"), _derrida_tex)]


@lru_cache(maxsize=None)
def _cached(beta, N_scale):
    return run_pipeline(rem_steps(beta, N_scale), State())


def rem_level_a(params):
    """(State, Trace) of the seven steps at a catalog run's params (beta optional;
    N_scale defaults to the catalog's 0.5)."""
    b = params.get("beta")
    return _cached(None if b is None else str(b), str(params.get("N_scale", "0.5")))


__all__ = ["compare_derrida", "default_pair_sum", "rem_level_a", "rem_steps"]
