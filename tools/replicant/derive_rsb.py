"""Level (a), one-step RSB: the 1RSB order-parameter functional derived by rules.

  rsb_ansatz(frozen=False)      -> State key `rsb`       (1RSB functional, symbolic n)
  rsb_replica_limit()           -> State key `rsb_free`  (lim_{n->0} of it / n)
  rsb_stationarity()            -> State key `rsb_equations` (d/d(order parameter) = 0)
  zero_temperature_limit()      -> State key `rsb_t0`    (lim_{beta->oo} phi/beta)
  derive_1rsb(spec, frozen=False, zero_temperature=False) -> (equations, State, Trace)
  derived_1rsb_functional(spec, frozen=False)              -> RSBFunctional (phi set)

The steps consume derive.introduce_order_parameters' formal Functional (the same family
hooks as the RS derivation: replicate, average-disorder and introduce-order-parameters are
reused unchanged) and reduce it under the Parisi 1RSB ansatz: blocks of x replicas (x is
Parisi's m; named x so that it does not collide with the Hopfield magnetization m),
q_ab = q1, qhat_ab = qhat1 in a block, q0, qhat0 between blocks; extras (m_a) replica
symmetric.  Rules (gauss.py, gauss_rsb.py, ansatz.py; each tested on its own):
  pair sum      sum_{a<b} f(q_ab) = (1/2) ansatz.sum_offdiag_power('rsb', [x], [f0, f1], 1, n)
                = (1/2) n [(x-1) f1 + (n-x) f0]
  Ising site    gauss_rsb.block_quadratic, then Hubbard-Stratonovich twice (t for all
                replicas, u_k per block): ln Tr = -n A1/2 + ln E_t[(E_u single(U)^x)^(n/x)],
                U = sqrt(A0) t + sqrt(A1-A0) u + B (A1 >= A0 >= 0 assumed, noted)
  LogDet        gauss_rsb.rsb_logdet (spherical site)
  LnGaussProd   gauss_rsb.rsb_split, E_z by gauss.expect_closed, product over a block ->
                (E_z g)^x, over blocks -> (E_u (E_z g)^x)^(n/x); E_u closed by expect_closed
                when it can be (exp-quadratic: Hopfield), else left as a nested expectation
  n -> 0        gauss.replica_limit: (1/n) ln E_t[F^(n/x)] -> (1/x) E_t ln F
Frozen 1RSB (frozen=True; discrete variables and a counting measure only): q1 = 1, the
replicas of a block are one configuration (s_a = sigma_k), imposed on the trace, so qhat1 is
not introduced.  Rules: the site trace runs over n/x block spins with coupling x^2 A0 and
field x B (block_quadratic at s_a = sigma_k); the pattern term has lambda_a = lambda_k =
sqrt(q0) t + sqrt(1-q0) u_k (rsb_split at q1 = 1: no z) and g^x = g for an indicator g
(gauss_rsb.indicator_power).  Then phi = S_RS(q0, x^2 qhat0)/x: d phi/d x = 0 forces S_RS = 0
(Krauth-Mezard); that is what solve_derived_rsb uses, nothing is typed in.  A weight that is
not an indicator, or an energy that is not a sum of LnGaussProd terms, is refused.
T = 0 (zero_temperature_limit): Parisi's T = 0 1RSB ansatz q1 -> 1, x = mu/beta (mu fixed):
conjugates eliminated through their linear stationarity equations, q1 = 1 and x = mu/beta
substituted, phi/beta: the polynomial part by sympy's limit, the expectations by
gauss_rsb.zero_temperature_power ((2 cosh beta y)^(mu/beta) -> e^(mu|y|)), then abs_split and
the exp-step rule close E_u.  Only for an energy polynomial in the overlaps (SK): with a
Gaussian-integral energy the limit needs q1 = 1 - C/beta (refused, not implemented).
Honest label: physics-level (Parisi ansatz, n -> 0); the rules are exact identities.
"""
from dataclasses import dataclass, replace

import sympy as sp

from . import derive as D
from . import gauss
from . import gauss_rsb as GR
from .ansatz import sum_offdiag_power
from .core import Refusal, State, Step, run_pipeline
from .families.common import CAVEAT

Q0, Q1 = sp.Symbol("q0", positive=True), sp.Symbol("q1", positive=True)
QH0, QH1 = sp.Symbol("qhat0", positive=True), sp.Symbol("qhat1", positive=True)
X = sp.Symbol("x", positive=True)           # Parisi block size m
MU = sp.Symbol("mu", positive=True)         # T = 0: mu = beta x
_T, _U, _Z = D._T, sp.Symbol("u", real=True), D._Z
_a, _b, _s = D._a, D._b, D._s


@dataclass(frozen=True)
class RSBFunctional:
    """The 1RSB functional: `expr` at symbolic n, `phi` = lim (1/n) expr (None before the
    replica limit).  `fixed` maps the order parameters the ansatz fixes (frozen: q1 = 1,
    no qhat1) to their values; `rules` lists the reductions applied."""
    spec: object
    n: object
    expr: sp.Expr
    params: tuple
    rules: tuple
    frozen: bool = False
    fixed: dict = None
    phi: sp.Expr = None
    caveat: str = CAVEAT
    conditions: tuple = ()      # expressions the reduction ASSUMED > 0 (a solver enforces them)


def _counting_measure(F):
    """Frozen 1RSB precondition: discrete variables, energy = sum of c LnGaussProd(g) with
    g an indicator."""
    if F.spec.variables not in ("ising",):
        raise Refusal("precondition:discrete-variables-for-frozen",
                      f"variables {F.spec.variables!r}: q1 = 1 (identical replicas in a block) "
                      "is a frozen state of discrete variables only")
    terms = F.energy.atoms(D.LnGaussProd)
    rest = F.energy.xreplace({t: 0 for t in terms})
    bad = [t for t in terms if not isinstance(t.args[0].expr, sp.Heaviside)]
    if rest != 0 or not terms or bad:
        raise Refusal("precondition:counting-measure-for-frozen",
                      f"energy {F.energy}: the frozen rule g^x = g needs every energy term to "
                      "be a pattern average of an indicator weight (a counting measure)")


def _tidy(e):
    """Rational exponents and bases put over a common denominator and factored."""
    e = e.replace(lambda y: isinstance(y, sp.exp),
                  lambda y: sp.exp(sp.factor(sp.cancel(y.args[0]))))
    return e.replace(lambda y: y.is_Pow and not y.base.is_number and not y.base.is_Symbol,
                     lambda y: sp.factor(sp.cancel(y.base))**y.exp)


def _reduce(F, frozen):
    n, rules, conds = F.n, [], []
    lv0 = {F.q: Q0, **({F.qhat: QH0} if F.conjugate else {})}
    lv1 = {F.q: sp.S(1) if frozen else Q1,
           **({F.qhat: sp.S(0) if frozen else QH1} if F.conjugate else {})}
    extra = {name: sp.Symbol(name, real=True) for name in (F.extra or {})}
    for name, base in (F.extra or {}).items():
        lv0[base] = lv1[base] = extra[name]
    if frozen:
        _counting_measure(F)

    def at(e, lv):
        e = sp.sympify(e).replace(lambda y: isinstance(y, sp.Indexed) and y.base in lv,
                                  lambda y: lv[y.base])
        if any(y.base != _s for y in e.atoms(sp.Indexed)):
            raise Refusal("precondition:rsb-reducible", f"unknown replica-indexed object in {e}")
        if e.has(_a) or e.has(_b) or e.has(_s):
            raise Refusal("precondition:rsb-reducible", f"{e} still depends on replicas")
        return e

    def site(tr):
        (arg,) = tr.args
        if not isinstance(arg, sp.exp):
            raise Refusal("precondition:rsb-site-trace-form", str(tr))
        pair = field = sp.S(0)
        for term in sp.Add.make_args(arg.args[0]):
            if not isinstance(term, sp.Sum):
                raise Refusal("precondition:rsb-site-trace-form", str(term))
            if term.limits == ((_b, _a + 1, n), (_a, 1, n)):
                pair += sp.cancel(term.function / (_s[_a] * _s[_b]))
            elif term.limits == ((_a, 1, n),):
                field += sp.cancel(term.function / _s[_a])
            else:
                raise Refusal("precondition:rsb-site-trace-form", str(term.limits))
        A0, A1, B = at(pair, lv0), at(pair, lv1), at(field, lv0)
        S, Sk = sp.Symbol("S", real=True), sp.Symbol("S_k", real=True)
        single = lambda U: sp.exp(U) + sp.exp(-U)
        if frozen:
            # s_a = sigma_k in block k: block_quadratic at S = x S' (S' = sum_k sigma_k) and
            # sum_k S_k^2 = sum_k (x sigma_k)^2 = n x; the S'^2 part is decoupled by HS in t
            quad = sp.expand(GR.block_quadratic(A0, A1, X * S, [], n) + (A1 - A0) * n * X / 2)
            const = sp.expand(quad - quad.coeff(S, 2) * S**2)
            hs = gauss.hubbard_stratonovich(2 * quad.coeff(S, 2), S, _T)
            U = sp.cancel(sp.diff(hs.f, S) / hs.f) + X * B
            rules.append("frozen site: s_a = sigma_k in a block, gauss_rsb.block_quadratic -> "
                         "coupling x^2 A0 between n/x block spins, field x B; "
                         "Hubbard-Stratonovich in t")
            return const + sp.log(gauss.GaussE(single(U)**(n / X), _T))
        hs0 = gauss.hubbard_stratonovich(A0, S, _T)
        hs1 = gauss.hubbard_stratonovich(A1 - A0, Sk, _U)
        conds.extend(c for c in (A0, A1 - A0) if c.is_positive is None)
        U = sp.cancel(sp.diff(hs0.f, S) / hs0.f) + sp.cancel(sp.diff(hs1.f, Sk) / hs1.f) + B
        rules.append("site: gauss_rsb.block_quadratic (Ising s^2 = 1), Hubbard-Stratonovich in "
                     "t (all replicas) and u (per block, A1 >= A0 >= 0 assumed), replicas of a "
                     "block decouple -> [E_u single^x]^(n/x)")
        return -n * A1 / 2 + sp.log(gauss.GaussE(gauss.GaussE(single(U)**X, _U)**(n / X), _T))

    def logdet(e):
        d, c, nn = e.args
        if frozen:
            raise Refusal("precondition:discrete-variables-for-frozen", "LogDet (spherical site)")
        rules.append("LogDet: gauss_rsb.rsb_logdet (1RSB spectrum)")
        return GR.rsb_logdet(at(d, lv0), at(c, lv0), at(c, lv1), X, nn)

    def gprod(e):
        g, c, nn = e.args
        c0, c1 = at(c, lv0), at(c, lv1)
        assumed = []
        if frozen:
            lam = GR.rsb_split(c0, c1, _T, _U, _Z)        # q1 = 1: no z left
            inner = gauss.GaussE(g(lam), _Z)              # E_z of a z-free function: itself
            block = GR.indicator_power(inner**X)
            Eu = gauss.expect_closed(block, _U, assumed, conds)
            rules.append(f"LnGaussProd (frozen): lambda_k = {lam}, g^x = g (indicator), "
                         f"E_u g = {Eu}" + "".join(f" ({y})" for y in assumed))
            return sp.log(gauss.GaussE(Eu**(nn / X), _T))
        lam = GR.rsb_split(c0, c1, _T, _U, _Z)
        inner = gauss.expect_closed(g(lam), _Z, assumed, conds)
        block = sp.powdenest(sp.expand_power_base(inner**X, force=True), force=True)
        try:
            Eu = _tidy(gauss.expect_closed(block, _U, assumed, conds))
            how = f"E_u closed: {Eu}"
        except Refusal:
            Eu, how = gauss.GaussE(block, _U), "E_u left as a nested expectation"
        rules.append(f"LnGaussProd: gauss_rsb.rsb_split lambda_a = {lam}, E_z g = {inner}, "
                     f"block product -> (E_z g)^x, {how}" + "".join(f" ({y})" for y in assumed))
        return sp.log(gauss.GaussE(Eu**(nn / X), _T))

    def rsum(S):
        if S.limits == ((_b, _a + 1, n), (_a, 1, n)):
            rules.append("pair sum: (1/2) ansatz.sum_offdiag_power('rsb', [x], [f0, f1], k=1)")
            return sum_offdiag_power("rsb", [X], [at(S.function, lv0), at(S.function, lv1)],
                                     1, n) / 2
        if S.limits == ((_a, 1, n),):
            rules.append("replica sum: n identical terms (extras replica symmetric)")
            return n * at(S.function, lv0)
        raise Refusal("precondition:rsb-reducible", f"sum limits {S.limits}")

    e = F.expr
    e = e.replace(lambda y: isinstance(y, sp.log) and y.args[0].func == D.Tr,
                  lambda y: site(y.args[0]))
    e = e.replace(lambda y: y.func == D.LogDet, logdet)
    e = e.replace(lambda y: y.func == D.LnGaussProd, gprod)
    e = e.replace(lambda y: isinstance(y, sp.Sum), rsum)
    if e.atoms(sp.Indexed) or e.has(D.Tr):
        raise Refusal("precondition:rsb-reducible", f"left over after 1RSB: {e}")
    if frozen:
        params = (Q0, X) + ((QH0,) if F.conjugate else ())
        fixed = {Q1: sp.S(1), **({QH1: None} if F.conjugate else {})}
    else:
        params = (Q0, Q1, X) + ((QH0, QH1) if F.conjugate else ())
        fixed = {}
    conds = tuple(dict.fromkeys(sp.factor(sp.cancel(c)) for c in conds))
    return e, params + tuple(extra.values()), tuple(dict.fromkeys(rules)), fixed, conds


def rsb_ansatz(frozen=False) -> Step:
    def requires(state):
        if "functional" not in state:
            return ["functional"]
        return [] if isinstance(state["functional"].n, sp.Symbol) else ["symbolic-replica-number"]

    def apply(state):
        F = state["functional"]
        expr, params, rules, fixed, conds = _reduce(F, frozen)
        return state.with_(rsb=RSBFunctional(F.spec, F.n, expr, params, rules, frozen, fixed,
                                             conditions=conds))

    return Step("rsb-ansatz", requires, apply,
                note=lambda s: ("frozen " if s["rsb"].frozen else "") + "1RSB; rules: "
                + "; ".join(s["rsb"].rules), tex=D._tex("rsb-ansatz"))


def rsb_replica_limit() -> Step:
    def apply(state):
        R = state["rsb"]
        return state.with_(rsb_free=replace(R, phi=gauss.replica_limit(R.expr, R.n)))

    return Step("rsb-replica-limit", lambda s: [] if "rsb" in s else ["rsb"], apply,
                note=lambda s: "n -> 0: polynomial part by ansatz.n_to_zero, the rest by d/dn "
                "at n = 0: (1/n) ln E_t[F^(n/x)] -> (1/x) E_t ln F; " + s["rsb_free"].caveat,
                tex=D._tex("rsb-replica-limit"))


def rsb_stationarity(key="rsb_free") -> Step:
    def apply(state):
        P = state[key]
        return state.with_(rsb_equations=[sp.Eq(sp.diff(P.phi, p), 0) for p in P.params])

    return Step("rsb-stationarity", lambda s: [] if key in s else [key], apply,
                note=lambda s: "d/d" + ", d/d".join(str(p) for p in s[key].params),
                tex=D._tex("rsb-stationarity"))


from .derive_rsb_t0 import zero_temperature_limit  # noqa: E402  (uses the names above)


def derive_1rsb(spec, frozen=False, zero_temperature=False, n=None):
    n = sp.Symbol("n", positive=True) if n is None else n
    steps = [D.replicate(spec, n), D.average_disorder(), D.introduce_order_parameters(),
             rsb_ansatz(frozen), rsb_replica_limit()]
    steps += [zero_temperature_limit(), rsb_stationarity("rsb_t0")] if zero_temperature \
        else [rsb_stationarity()]
    state, trace = run_pipeline(steps, State())
    return state["rsb_equations"], state, trace


def derived_1rsb_functional(spec, frozen=False) -> RSBFunctional:
    return derive_1rsb(spec, frozen)[1]["rsb_free"]
