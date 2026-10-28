"""Level-(a)-style derivation of the REM noise of the dense associative memory: the replica calculation over
pattern LABELS (Lucibello & Mezard, Phys. Rev. Lett. 132, 077301 (2024), arXiv:2304.14964v4, App. A 4), step
by step, each step checked by an independent route and refusing (`check:<name>`) on a mismatch.

  label_replica_steps() -> [core.Step]    seven steps over the State keys dm_*

  1 dm-noise-split       E(x) = -(1/lam) log(e^{lam x.xi^1} + e^{lam N Phi}) + |x|^2/2 -> -max(x.xi^1, N phi)
                         (Eqs. (3)-(6)); (1/N) log(e^{Na} + e^{Nb}) -> max(a, b) by sympy.limit
  2 dm-replicate         E log Z = lim (1/n) log E Z^n; Z^n = sum over n-tuples of labels (checked by
                         expansion at n = 2, 3), the REM step of derive_rem.py
  3 dm-average-patterns  for a fixed label tuple with occupations m_mu the pattern average factorises
                         over labels, E exp(lam N sum_mu m_mu eps_mu) = exp(N sum_mu zeta(lam m_mu))
                         (A22), zeta computed by sympy for the ensemble (Gaussian: sympy.integrate;
                         spherical: the Legendre transform of s; binary: cosh) and checked against
                         dense_memory.ensembles.zeta
  4 dm-1rsb-labels       n/m groups of m replicas on one label (the 1RSB ansatz of (A24)): occupations
                         m on n/m labels, (e^{alpha N})^{n/m} tuples (falling-factorial correction
                         O(e^{-alpha N}), sympy series), sum_mu zeta(lam m_mu) = (n/m) zeta(lam m),
                         brute-forced over the tuples of (n, m) = (4,2), (6,2), (6,3)
  5 dm-replica-limit     Phi(m) = lim Phi_n / n by ansatz.n_to_zero = (alpha + zeta(lam m))/m (A24);
                         for the Gaussian ensemble EQUAL (sympy) to families/rem.py rem_1rsb_functional
                         with log 2 -> alpha and s beta^2 -> lam^2 rho^2: the REM of derive_rem.py is the
                         special case P = 2^N, Gaussian levels
  6 dm-extremize         dPhi/dm = 0 is (A25), lam m zeta'(lam m) - zeta(lam m) = alpha, and at it
                         d2Phi/dm2 = lam^2 zeta''(lam m)/m > 0 (sympy, for a generic zeta): a MINIMUM
                         (the n < 1 inversion), so m* = min(1, lam*/lam), (A26)-(A27); at the requested
                         (alpha, lam) phi from the minimised functional equals the closed form
  7 dm-rem-reduction     at alpha = log 2 and Gaussian levels phi equals -rem_free_energy(lam, N_scale =
                         rho^2) of families/rem.py (Derrida's theorem): the extension from the REM

Physics level: the n -> 0 continuation and the 1RSB ansatz are not proved; for i.i.d. levels the answer is
Derrida's REM theorem (Gaussian) and a large-deviation theorem for general zeta (Fedrigo, Flandoli &
Morandin 2007, as cited by the paper's Eq. (8); not read here).
"""
import itertools
from functools import lru_cache

import mpmath as mp
import sympy as sp

from .ansatz import n_to_zero
from .core import Refusal, State, Step
from .dense_memory import ensembles as EN
from .dense_memory import rem_noise as RN

LEVEL_A = "replicant level (a) derivation (this package)"
LM = ("Lucibello & Mezard, Phys. Rev. Lett. 132, 077301 (2024), arXiv:2304.14964v4")
DERRIDA = "Derrida, PRL 45, 79 (1980); Phys. Rev. B 24, 2613 (1981)"

n, m = sp.symbols("n m", positive=True)
lam, alpha, rho, s_ = sp.symbols("lambda alpha rho s", positive=True)
t = sp.Symbol("t", real=True)
N = sp.Symbol("N", positive=True)
ZETA = sp.Function("zeta")

# planted-error hook for the tests: {"blocks": k} counts n/(m+k) label groups instead of n/m
_PLANT = {}


def _check(ok, name, detail):
    if not ok:
        raise Refusal(f"check:{name}", detail)


def _zero(expr):
    return sp.simplify(expr) == 0


# ---- 1: signal and noise -------------------------------------------------------------------
@lru_cache(maxsize=None)
def _signal_vs_noise():
    for a, b in ((sp.Integer(1), sp.Rational(1, 2)), (sp.Rational(1, 3), sp.Rational(3, 4))):
        lim = sp.limit(sp.log(sp.exp(N * a) + sp.exp(N * b)) / N, N, sp.oo)
        _check(lim == max(a, b), "signal-vs-noise", f"(1/N) log(e^(N {a}) + e^(N {b})) -> {lim}")
    return True


def _split(fixed):
    def apply(st):
        ens = fixed or st["params"].get("ensemble", "spherical")
        EN.check_ensemble(ens, st["params"].get("rho", 1))
        _signal_vs_noise()
        return st.with_(dm_ens=ens, dm_split="E(x) = -(1/lam) log(e^(lam x.xi^1) + e^(lam N Phi)) + "
                        "|x|^2/2 ~ -max(x.xi^1, N phi) + |x|^2/2")
    return apply


# ---- 2: replicate --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _replica_identity():
    Zs = sp.Symbol("Z", positive=True)
    lim = sp.limit((Zs**n - 1) / n, n, 0)
    _check(lim == sp.log(Zs), "replica-identity", f"lim (Z^n - 1)/n = {lim}, not ln Z")
    for nn, K in ((2, 3), (3, 3)):
        w = sp.symbols(f"w0:{K}")
        tuples = sum(sp.Mul(*(w[k] for k in tup)) for tup in itertools.product(range(K), repeat=nn))
        _check(sp.expand(sum(w)**nn - tuples) == 0, "tuple-expansion",
               f"(sum_mu w_mu)^{nn} != the sum over {nn}-tuples of labels")
    return True


def _replicate(st):
    _replica_identity()
    return st.with_(dm_replicate="Z^n = sum over n-tuples (mu_1..mu_n) of exp(lam N sum_a eps_{mu_a})")


# ---- 3: pattern average --------------------------------------------------------------------
@lru_cache(maxsize=None)
def _zeta_symbolic(ens):
    """(zeta(t), how) for the ensemble, computed by sympy (rho symbolic; binary has rho = 1)."""
    z = sp.Symbol("z", real=True)
    if ens == "gaussian":
        # eps = rho z/sqrt(N): E exp(t N eps) = E exp(t rho sqrt(N) z) = exp(N t^2 rho^2/2)
        mgf = sp.simplify(sp.integrate(sp.exp(t * rho * sp.sqrt(N) * z - z**2 / 2), (z, -sp.oo, sp.oo))
                          / sp.sqrt(2 * sp.pi))
        zeta_g = sp.simplify(sp.expand_log(sp.log(mgf), force=True) / N)
        return zeta_g, "sympy.integrate (Gaussian moment generating function)"
    if ens == "spherical":
        e = sp.Symbol("e", real=True)
        s = -sp.log(1 - e**2 / rho**2) / 2                                       # (C4)
        roots = sp.solve(sp.Eq(t, sp.diff(s, e)), e)
        good = [r for r in roots if sp.simplify(r.subs({t: 1, rho: 1}) ** 2) < 1 and r.subs({t: 1, rho: 1}) > 0]
        _check(len(good) == 1, "spherical-stationary-level", f"roots {roots}")
        return sp.simplify(t * good[0] - s.subs(e, good[0])), "Legendre transform of s (C4): sup_eps t eps - s(eps)"
    return sp.log((sp.exp(t) + sp.exp(-t)) / 2), "E exp(t x_i xi_i) = cosh t per site, N sites"


def _average(st):
    ens = st["dm_ens"]
    EN.check_ensemble(ens, st["params"].get("rho", 1))
    zs, how = _zeta_symbolic(ens)
    for tv, rv in (("0.3", "1"), ("2", "1"), ("1.4", "1" if ens == "binary" else "0.75")):
        with mp.workdps(30):
            want = EN.zeta(ens, tv, rv)
            got = mp.mpf(str(sp.N(zs.subs({t: sp.Rational(tv), rho: sp.Rational(rv)}), 30)))
            _check(abs(got - want) < mp.mpf("1e-25"), "zeta-code",
                   f"{ens}: sympy zeta({tv}) = {got}, dense_memory.ensembles.zeta = {want}")
    return st.with_(dm_zeta=zs, dm_zeta_how=how,
                    dm_average="E exp(lam N sum_mu m_mu eps_mu) = exp(N sum_mu zeta(lam m_mu))  (A22)")


# ---- 4: 1RSB over labels -------------------------------------------------------------------
@lru_cache(maxsize=None)
def _falling_factorial():
    X = sp.Symbol("x", positive=True)                        # x = e^{-alpha N}
    for g in (2, 3, 4, 5):
        ser = sp.series(sp.log(sp.Mul(*(1 - j * X for j in range(g)))), X, 0, 2).removeO()
        _check(_zero(ser + sp.Rational(g * (g - 1), 2) * X), "falling-factorial", f"g = {g}: {ser}")
    return True


def _blocks(st):
    k = _PLANT.get("blocks", 0)
    groups = n / (m + k)
    for nn, mm in ((4, 2), (6, 2), (6, 3)):                  # brute force over label tuples
        labels = range(nn // mm)
        tup = [g for g in labels for _ in range(mm)]         # replicas in blocks of mm on one label
        occ = {g: tup.count(g) for g in set(tup)}
        _check(all(v == mm for v in occ.values()) and len(occ) == nn // mm, "block-occupations",
               f"n = {nn}, m = {mm}: occupations {occ}")
        got = groups.subs({n: nn, m: mm})
        _check(got == sp.Rational(nn, mm), "block-count",
               f"n = {nn}, m = {mm}: the ansatz has {got} groups, the tuple has {nn // mm}")
    _falling_factorial()
    phi_n = groups * (alpha + ZETA(lam * m))                 # per N: (n/m) alpha + sum_mu zeta
    return st.with_(dm_phin=phi_n)


# ---- 5: n -> 0 -----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _gaussian_is_rem(conc):
    """The Gaussian Phi(m) at alpha = log 2 equals families/rem.py rem_1rsb_functional (beta -> lam, s -> rho^2)."""
    from .families.rem import rem_1rsb_functional
    b, ss = sp.symbols("beta s", positive=True)
    fam = rem_1rsb_functional().subs({b: lam, ss: rho**2})
    _check(_zero(conc.subs(alpha, sp.log(2)) - fam), "rem-functional",
           f"Gaussian Phi(m) at alpha = log 2: {sp.simplify(conc.subs(alpha, sp.log(2)))}; "
           f"rem_1rsb_functional = {fam}")
    return True


def _limit(st):
    ens = st["dm_ens"]
    phi_m = n_to_zero(st["dm_phin"], n)
    _check(_zero(phi_m - (alpha + ZETA(lam * m)) / m), "functional", f"Phi(m) = {phi_m}")
    zs = st["dm_zeta"]
    conc = phi_m.replace(ZETA, sp.Lambda(t, zs.subs(rho, rho)))
    reused = None
    if ens == "gaussian":
        _gaussian_is_rem(conc)
        reused = ("families/rem.py rem_1rsb_functional at alpha = log 2 (beta -> lam, N_scale -> rho^2; the "
                  "alpha-dependence is the label entropy, linear)")
    return st.with_(dm_Phim=phi_m, dm_Phim_ens=sp.simplify(conc), dm_reused=reused)


# ---- 6: extremise --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _convexity_identity():
    """For a generic zeta: stationarity of Phi(m) = (alpha + zeta(lam m))/m is (A25), and at it
    d2Phi/dm2 = lam^2 zeta''(lam m)/m."""
    u = sp.Symbol("u", positive=True)                      # u = lam m
    g = sp.Function("g")                                   # a generic zeta
    Phi_u = lam * (alpha + g(u)) / u                       # Phi(m) = (alpha + zeta(lam m))/m
    stat = u * sp.diff(g(u), u) - g(u) - alpha             # (A25) in u: lam m zeta'(lam m) - zeta - alpha
    _check(_zero(sp.diff(Phi_u, u) * u**2 / lam - stat), "stationarity",
           f"u^2 dPhi/du / lam = {sp.simplify(sp.diff(Phi_u, u) * u**2 / lam)}")
    d2 = sp.simplify(sp.diff(Phi_u, u, 2).subs(alpha, u * sp.diff(g(u), u) - g(u)).doit())
    _check(_zero(d2 - lam * sp.diff(g(u), (u, 2)) / u), "convexity",
           f"d2Phi/du2 at (A25) = {d2}, expected lam zeta''(u)/u (so d2Phi/dm2 = lam^2 zeta''(lam m)/m > 0)")
    return d2


@lru_cache(maxsize=None)
def _zeta_convex(ens):
    zpp_ens = sp.simplify(sp.diff(_zeta_symbolic(ens)[0], t, 2))
    for tv in (sp.Rational(1, 10), sp.Integer(1), sp.Integer(5)):
        v = zpp_ens.subs({t: tv, rho: sp.Rational(3, 4) if ens != "binary" else 1})
        _check(v.is_positive is True or sp.N(v) > 0, "zeta-convex", f"zeta''({tv}) = {v}")
    return True


def _extremize(st):
    ens = st["dm_ens"]
    _convexity_identity()
    _zeta_convex(ens)
    p = st["params"]
    num = None
    if p.get("alpha") is not None and p.get("lam") is not None:
        r = p.get("rho", 1)
        got, mstar = RN.phi_1rsb(ens, p["alpha"], p["lam"], r)
        want_phi = RN.phi(ens, p["alpha"], p["lam"], r, "closed")
        _check(abs(got - want_phi) < mp.mpf("1e-25"), "minimised-functional",
               f"min_m Phi = {mp.nstr(got, 20)}, closed form {mp.nstr(want_phi, 20)}")
        num = dict(phi=got, m=mstar, closed=want_phi)
    return st.with_(dm_ext=dict(d2=_convexity_identity(), num=num))


# ---- 7: reduction to the REM ---------------------------------------------------------------
def _reduction(st):
    from .families.rem import rem_free_energy
    ens = st["dm_ens"]
    out = None
    if ens == "gaussian":
        with mp.workdps(30):
            for lv, rv in ((mp.mpf("0.9"), mp.mpf(1)), (mp.mpf("1.3"), mp.mpf("0.7")), (mp.mpf(4), mp.mpf(1))):
                got = RN.phi("gaussian", mp.log(2), lv, rv, "closed")
                want = -rem_free_energy(lv, N_scale=rv**2)
                _check(abs(got - want) < mp.mpf("1e-25"), "rem-reduction",
                       f"Gaussian phi(log 2, {lv}, {rv}) = {mp.nstr(got, 15)}, "
                       f"-rem_free_energy = {mp.nstr(want, 15)}")
        out = "phi_{log 2, rho}(lam) = -f_REM(beta = lam, N_scale = rho^2), both phases (3 points)"
    return st.with_(dm_reduction=out)


def label_replica_steps(ensemble=None):
    """The seven Steps (module docstring).  `ensemble` fixes the pattern ensemble (the Ising entry passes
    "binary"); None reads State["params"]["ensemble"] (default spherical).  They also read, when given,
    alpha, lam and rho."""
    has = lambda *ks: (lambda st: [k for k in ks if k not in st])
    return [
        Step("dm-noise-split", has("params"), _split(ensemble), f"{LEVEL_A}; {LM}, Eqs. (3)-(6)",
             lambda st: "E(x) ~ -max(x.xi^1, N phi): signal against the REM noise"),
        Step("dm-replicate", has("params"), _replicate, f"{LEVEL_A}; {LM}, App. A 4, Eq. (A19)",
             lambda st: "Z^n over n-tuples of pattern labels"),
        Step("dm-average-patterns", has("params", "dm_replicate"), _average,
             f"{LEVEL_A}; {LM}, App. A 4, Eqs. (A22)-(A23)",
             lambda st: f"zeta by {st['dm_zeta_how']}"),
        Step("dm-1rsb-labels", has("dm_zeta"), _blocks, f"{LEVEL_A}; {LM}, App. A 4 (1RSB ansatz)",
             lambda st: "n/m groups of m replicas on one label: (n/m)(alpha + zeta(lam m))"),
        Step("dm-replica-limit", has("params", "dm_phin"), _limit, f"{LEVEL_A}; {LM}, Eq. (A24)",
             lambda st: "Phi(m) = (alpha + zeta(lam m))/m"
             + (f"; reused {st['dm_reused']}" if st["dm_reused"] else "")),
        Step("dm-extremize", has("params", "dm_Phim"), _extremize, f"{LEVEL_A}; {LM}, Eqs. (A25)-(A27)",
             lambda st: "a minimum over 0 < m <= 1: m* = min(1, lam*/lam)" + (
                 f"; at the state point phi = {mp.nstr(st['dm_ext']['num']['phi'], 12)}, m* = "
                 f"{mp.nstr(st['dm_ext']['num']['m'], 12)}" if st["dm_ext"]["num"] else "")),
        Step("dm-rem-reduction", has("params", "dm_Phim_ens"), _reduction, f"{LEVEL_A}; {DERRIDA}",
             lambda st: st["dm_reduction"] or "skipped: the REM reduction is the Gaussian ensemble"),
    ]


__all__ = ["label_replica_steps"]
