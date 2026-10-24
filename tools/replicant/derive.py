"""Level (a): symbolic derivation steps on sympy expressions.

  replicate(spec, n)           -> State key `Zn`         (formal Z^n)
  average_disorder()           -> State key `avg`        (per-coupling cumulant)
  introduce_order_parameters() -> State key `functional` (q_ab, q_hat_ab)
  saddle_equations()           -> State key `equations`  (sympy.Eq list)
  rs_ansatz()                  -> State key `rs`         (RS functional, symbolic n)
  replica_limit()              -> State key `rs_free`    (lim_{n->0} of it / n)
  rs_stationarity()            -> State key `rs_equations` (d/d(order parameter) = 0)
  annealed_log_Z(spec, N)      -> ln E Z, exact at finite N (the n = 1 check)

SK takes an optional uniform field h (spec.params["h"], H -> H - h sum_i s_i).
The RS steps work at symbolic n only; they reduce the formal objects the
families produce (pair and single-replica Sums, log Tr_s exp(...), LogDet,
LnGaussProd) with the rules of gauss.py, and record which rule they used.
Perceptron and Hopfield register their hooks in derive_families.py.

Conventions: N Ising spins, couplings J_ij (i<j) with variance 1/N,
H = -sum_{i<j} J_ij s_i s_j, Z = sum_s exp(-beta H).  With
X_ij = sum_a s_i^a s_j^a the replicated weight is exp(beta sum_{i<j} J_ij X_ij),
so the disorder enters only through the cumulant generating function
K(t) = ln E exp(t J) at t = beta X.

Extension points (public registries; anything outside them is a named
Refusal):
  FAMILIES  name -> Family: the family owns its disorder average and its
            reduction to overlaps.  The steps only route: `average_disorder`
            calls `Family.average(zn)` and stores the result opaquely;
            `introduce_order_parameters` hands it back to
            `Family.order_parameters`, which returns a Reduction (energy,
            O(1) correction, order of what it neglects, a per-replica site
            field, extra replica-indexed order parameters such as m_a) or the
            plain 3-tuple.  `Family.variables` lists the spec.variables it
            accepts.  SK here; perceptron and Hopfield in derive_families.py.
  SITES     variable type -> Site(trace, conjugate): "ising" is the trace
            ln Tr exp(sum qhat s^a s^b + sum field s^a) with conjugates qhat;
            "spherical" the hand-supplied volume (1/2) ln det Q, no conjugate.
            The site is chosen by spec.variables alone (introduce_order_parameters),
            so a family listing several variables (the perceptron: spherical and
            ising) gets a different entropic term and nothing else changes.
  LAWS      coupling/pattern laws by their exact K; the iid-pair, variance-1/N,
            K(beta X_ij) structure is SK's (`_sk_average`), not the steps'.
A family whose energy carries conjugate fields beyond q_ab, qhat_ab (and the
m_a of a Reduction) still needs the Functional and the RS step to grow.
"""
import inspect
import itertools
from dataclasses import dataclass
from typing import Callable

import sympy as sp

from . import gauss
from .ansatz import sum_offdiag_power
from .core import Refusal, Step
from .families.common import CAVEAT
from .spec import ModelSpec

BETA = sp.Symbol("beta", positive=True)
N_SYM = sp.Symbol("N", positive=True, integer=True)
T = sp.Symbol("t", real=True)       # argument of the cumulant generating function
X = sp.Symbol("X", real=True)       # the coupling field X_ij = sum_a s_i^a s_j^a
Tr = sp.Function("Tr_s")            # formal trace over one site's n replicas
H_FIELD = sp.Symbol("h", real=True)
# LogDet(d, c, n): ln det of the n x n matrix with d on the diagonal and c (an
# expression in q[a, b]) off it.  LnGaussProd(g, c, n): ln E_lambda prod_a g(lambda_a),
# lambda ~ N(0, C) with C_aa = 1, C_ab = c.  Both stay formal until rs_ansatz.
LogDet = sp.Function("LogDet")
LnGaussProd = sp.Function("LnGaussProd")
# RS order parameters (q in [0, 1]; qhat >= 0 is the Hubbard-Stratonovich condition)
Q_RS = sp.Symbol("q", positive=True)
QHAT_RS = sp.Symbol("qhat", positive=True)


def _beta(spec: ModelSpec):
    return spec.params.get("beta", BETA)


def _field(spec: ModelSpec):
    return spec.params.get("h", 0)


# --- coupling laws ----------------------------------------------------------

@dataclass(frozen=True)
class DisorderLaw:
    name: str
    cgf: Callable[[sp.Expr, sp.Expr], sp.Expr]  # (t, N) -> exact ln E exp(t J)


LAWS = {
    "gaussian": DisorderLaw("gaussian", lambda t, N: t**2 / (2*N)),   # N(0, 1/N)
    "pm_one": DisorderLaw("pm_one",                                   # +-1/sqrt(N)
                          lambda t, N: sp.log(sp.cosh(t / sp.sqrt(N)))),
}


def _law(spec: ModelSpec) -> DisorderLaw:
    if spec.disorder not in LAWS:
        raise Refusal("precondition:supported-disorder-law",
                      f"no exact cumulant for disorder {spec.disorder!r}; "
                      f"known: {', '.join(sorted(LAWS))}")
    return LAWS[spec.disorder]


# --- State values -------------------------------------------------------------

@dataclass(frozen=True)
class Replicated:
    """Z^n before the average.  `n` is a positive int (explicit sums) or a
    sympy Symbol (formal Sums)."""
    spec: ModelSpec
    n: object
    beta: sp.Expr
    expr: sp.Expr


@dataclass(frozen=True)
class Averaged:
    """SK's averaged object (other families define their own).  Per-pair
    averaged exponent: ln E exp(beta J X) ~ `leading` (in X);
    `dropped` is the first neglected cumulant term (in t), 0 if exact."""
    spec: ModelSpec
    n: object
    law: DisorderLaw
    X: sp.Symbol
    t: sp.Symbol
    N: sp.Symbol
    leading: sp.Expr
    dropped: sp.Expr
    note: str


@dataclass(frozen=True)
class Functional:
    """ln E Z^n = N * extr_{q, q_hat} expr + correction + (terms below O(1)
    from the saddle-point prefactor), with
    expr = energy - sum_{a<b} q_hat_ab q_ab + entropic.
    `q`, `qhat` map (a, b), a<b, to symbols for explicit n; for symbolic n
    they are IndexedBases.  `q_zero` sets every overlap to 0 (use
    `xreplace(q_zero).doit()`: `subs` does not reach inside a Sum).
    `correction` is exact only when the average dropped nothing; otherwise
    `missing` names the order of the neglected term in ln E Z^n (pm_one:
    the quartic cumulant is "O(1)", the same order as `correction`; at n=1
    the exact minus (N energy + correction) is -beta^4/24 + O(1/N)).
    `missing` is None when nothing was neglected."""
    spec: ModelSpec
    n: object
    N: sp.Symbol
    q: object
    qhat: object
    energy: sp.Expr
    correction: sp.Expr
    entropic: sp.Expr
    expr: sp.Expr
    q_zero: dict
    missing: str | None = None
    extra: dict = None          # name -> IndexedBase of further order parameters (m_a)
    conjugate: bool = True      # the -sum qhat q term is present (False: spherical)


@dataclass(frozen=True)
class Reduction:
    """What a family's order_parameters hook returns (a plain 3-tuple
    (energy, correction, missing) still means: no field, no extras).
    `field` is the per-replica field coupling to s^a (may use the index
    `_a`); `extra` names replica-indexed order parameters other than q_ab."""
    energy: sp.Expr
    correction: sp.Expr
    missing: str | None
    field: sp.Expr = sp.S(0)
    extra: dict = None


# --- replica bookkeeping ----------------------------------------------------

_a, _b = sp.symbols("a b", integer=True, positive=True)


def _pairs(n):
    return list(itertools.combinations(range(1, n + 1), 2))


def _overlap_symbols(n):
    """(q, qhat, sum over a<b of f(q_ab, qhat_ab)) for explicit or symbolic n."""
    if isinstance(n, int):
        q = {p: sp.Symbol(f"q_{p[0]}_{p[1]}", real=True) for p in _pairs(n)}
        qh = {p: sp.Symbol(f"qhat_{p[0]}_{p[1]}", real=True) for p in _pairs(n)}
        return q, qh, lambda f: sp.Add(*(f(q[p], qh[p]) for p in _pairs(n)))
    q, qh = sp.IndexedBase("q", real=True), sp.IndexedBase("qhat", real=True)
    return q, qh, lambda f: sp.Sum(f(q[_a, _b], qh[_a, _b]),
                                   (_b, _a + 1, n), (_a, 1, n))


# --- families: how H couples to the disorder ---------------------------------

def _sk_formal(n, beta, h=0):
    i, j = sp.symbols("i j", integer=True, positive=True)
    s, J = sp.IndexedBase("s"), sp.IndexedBase("J")
    field = sp.Sum(s[i, _a] * s[j, _a], (_a, 1, n))
    ext = beta * h * sp.Sum(s[i, _a], (_a, 1, n), (i, 1, N_SYM))
    return Tr(sp.exp(beta * sp.Sum(J[i, j] * field, (j, i + 1, N_SYM),
                                   (i, 1, N_SYM)) + ext))


def _sk_annealed(spec, N, law):
    # s_i s_j = +-1 and the law is symmetric, so E exp(beta J s_i s_j) =
    # exp(K(beta)) for every state: ln E Z = N ln 2 + C(N, 2) K(beta).
    # With a field the spin sum gives (2 cosh beta h)^N instead of 2^N.
    if sp.simplify(law.cgf(T, N) - law.cgf(-T, N)) != 0:
        raise Refusal("precondition:symmetric-coupling-law", law.name)
    h = _field(spec)
    site = sp.log(2 * sp.cosh(_beta(spec) * h)) if h != 0 else sp.log(2)
    return N * site + (N * (N - 1) / 2) * law.cgf(_beta(spec), N)


def _sk_average(zn: Replicated) -> Averaged:
    """ln E exp(beta J_ij X_ij) per pair from the law's exact K: kept to
    second order in t = beta X, the first dropped term stated."""
    law = _law(zn.spec)
    K = law.cgf(T, N_SYM)
    ser = sp.expand(sp.series(K, T, 0, 5).removeO())
    if ser.coeff(T, 1) != 0:
        raise Refusal("precondition:zero-mean-couplings", law.name)
    second = ser.coeff(T, 2) * T**2
    if sp.simplify(K - second) == 0:
        dropped, note = sp.S(0), (f"{law.name}: exact, cumulant series "
                                  "ends at second order")
    else:
        k = next((k for k in (3, 4) if ser.coeff(T, k) != 0), None)
        if k is None:
            raise Refusal("precondition:dropped-cumulant-identified",
                          f"{law.name}: K is not quadratic but its t^3, t^4 "
                          "terms vanish; the dropped order is not established")
        dropped = ser.coeff(T, k) * T**k
        name = {3: "third", 4: "fourth"}[k]
        p = _order_in_N(dropped, N_SYM)
        # C(N,2) pairs, each O(N^-p), against an O(N) exponent.
        if p - 1 < 1:
            raise Refusal("precondition:higher-cumulants-subleading",
                          f"{dropped} is not small per spin")
        per_spin = "O(1/N)" if p == 2 else f"O(N^-{p - 1})"
        note = (f"{law.name}: kept second cumulant {second}; dropped "
                f"{name} cumulant {dropped} per pair, {per_spin} per spin")
    leading = second.subs(T, zn.beta * X)
    return Averaged(zn.spec, zn.n, law, X, T, N_SYM, leading, dropped, note)


def _sk_order_parameters(avg: Averaged, n, sum_pairs):
    """c sum_{i<j} X_ij^2 with c = leading / X^2.  Since
    X_ij^2 = sum_{a,b} (s_i^a s_i^b)(s_j^a s_j^b) and q_ab = (1/N) sum_i s_i^a s_i^b
    (q_aa = 1),  sum_{i<j} = (1/2)(sum_{i,j} - sum_{i=j}) gives
    c/2 sum_{a,b} (N^2 q_ab^2 - N)  =  N (cN/2) (n + 2 sum_{a<b} q_ab^2) - cN n^2/2.
    Returns (per-spin energy, O(1) correction, order of what was dropped)."""
    c = sp.simplify(avg.leading / avg.X**2)
    if c.has(avg.X):
        raise Refusal("precondition:quadratic-in-coupling-field",
                      f"averaged pair exponent {avg.leading} is not c X^2")
    cN = sp.simplify(c * avg.N)
    if cN.has(avg.N):
        raise Refusal("precondition:extensive-scaling",
                      f"coefficient {c} is not proportional to 1/N")
    energy = cN / 2 * (n + 2 * sum_pairs(lambda q, qh: q**2))
    missing = None
    if avg.dropped != 0:
        # Per pair O(N^-p), C(N,2) pairs: O(N^(2-p)) in ln E Z^n.
        e = 2 - _order_in_N(avg.dropped, avg.N)
        missing = "O(1)" if e == 0 else f"O(N^{e})"
    beta = _beta(avg.spec)
    return Reduction(energy, -cN * n**2 / 2, missing, field=beta * _field(avg.spec))


@dataclass(frozen=True)
class Family:
    formal: Callable            # (n, beta) -> formal Z^n
    annealed: Callable          # (spec, N) -> exact ln E Z
    supports_disorder: Callable  # disorder name -> bool
    average: Callable           # Replicated -> opaque averaged object with
                                #   a `.note` str for the trace
    order_parameters: Callable  # (averaged, n, sum_pairs) -> Reduction, or
                                #   (energy, correction, missing order | None)
    variables: tuple = ("ising",)   # spec.variables this family accepts


FAMILIES = {"sk": Family(lambda n, b, spec=None: _sk_formal(n, b, _field(spec) if spec else 0),
                         lambda spec, N: _sk_annealed(spec, N, _law(spec)),
                         lambda d: d in LAWS, _sk_average,
                         _sk_order_parameters)}


# --- single-site traces -------------------------------------------------------

def _ising_site(n, q, qh, field=0):
    """ln Tr_{s^1..s^n in {+-1}} exp(sum_{a<b} qhat_ab s^a s^b + sum_a field_a s^a).
    For explicit n the field must not depend on the replica index."""
    if isinstance(n, int):
        if sp.sympify(field).has(_a):
            raise Refusal("precondition:replica-independent-field", str(field))
        return sp.log(sp.Add(*(
            sp.exp(sp.Add(*(qh[(a, b)] * s[a - 1] * s[b - 1] for a, b in _pairs(n)))
                   + field * sum(s))
            for s in itertools.product((1, -1), repeat=n))))
    s = sp.IndexedBase("s")
    expo = sp.Sum(qh[_a, _b] * s[_a] * s[_b], (_b, _a + 1, n), (_a, 1, n))
    if field != 0:
        expo += sp.Sum(field * s[_a], (_a, 1, n))
    return sp.log(Tr(sp.exp(expo)))


def _spherical_site(n, q, qh, field=0):
    """(1/N) ln of the fraction of (sqrt(N) S^{N-1})^n with J^a.J^b = N q_ab:
    (1/2) ln det Q + O(ln N / N), the conjugate matrix already eliminated
    (the standard result of Gardner 1988, written from memory of the textbook
    form, not re-derived or checked against the paper here).  HAND-SUPPLIED."""
    if field != 0:
        raise Refusal("precondition:no-field-on-spherical", str(field))
    if isinstance(n, int):
        M = sp.Matrix(n, n, lambda i, j: 1 if i == j else q[tuple(sorted((i + 1, j + 1)))])
        return sp.log(M.det()) / 2
    return LogDet(1, q[_a, _b], n) / 2


@dataclass(frozen=True)
class Site:
    trace: Callable           # (n, q, qhat, field) -> per-site entropic term
    conjugate: bool           # uses qhat (adds -sum_{a<b} qhat q to the functional)


SITES = {"ising": Site(_ising_site, True), "spherical": Site(_spherical_site, False)}


# --- steps --------------------------------------------------------------------

def _valid_n(n):
    if isinstance(n, sp.Integer):
        n = int(n)
    if isinstance(n, int) and not isinstance(n, bool):
        return n if n >= 1 else None
    return n if isinstance(n, sp.Symbol) else None


def replicate(spec: ModelSpec, n) -> Step:
    def requires(state):
        missing = []
        if spec.family not in FAMILIES:
            missing.append("supported-family")
        elif spec.variables not in SITES or \
                spec.variables not in FAMILIES[spec.family].variables:
            accepted = [v for v in FAMILIES[spec.family].variables if v in SITES]
            raise Refusal("precondition:supported-variables",
                          f"{spec.family} is derived for variables {', '.join(accepted)} "
                          f"(a registered SITES entry each); got {spec.variables!r}")
        if _valid_n(n) is None:
            missing.append("replica-number")
        return missing

    def apply(state):
        m = _valid_n(n)
        beta = _beta(spec)
        formal = FAMILIES[spec.family].formal
        kw = {"spec": spec} if "spec" in inspect.signature(formal).parameters else {}
        return state.with_(Zn=Replicated(spec, m, beta, formal(m, beta, **kw)))

    return Step("replicate", requires, apply)


def _order_in_N(expr, N):
    """p such that expr ~ N^-p as N -> oo."""
    num, den = sp.fraction(sp.together(expr))
    return sp.degree(den, N) - sp.degree(num, N)


def average_disorder() -> Step:
    """Routes to the family's own average (`Family.average`)."""
    def requires(state):
        if "Zn" not in state:
            return ["Zn"]
        spec = state["Zn"].spec
        return [] if FAMILIES[spec.family].supports_disorder(spec.disorder) \
            else ["supported-disorder-law"]

    def apply(state):
        zn = state["Zn"]
        return state.with_(avg=FAMILIES[zn.spec.family].average(zn))

    return Step("average-disorder", requires, apply,
                note=lambda s: s["avg"].note)


def introduce_order_parameters() -> Step:
    """q_ab = (1/N) sum_i s_i^a s_i^b imposed by
    1 = int dq_ab dqhat_ab exp(-N qhat_ab q_ab + qhat_ab sum_i s_i^a s_i^b)
    (qhat on the imaginary axis); the spins then factorize over sites."""
    def requires(state):
        return [k for k in ("Zn", "avg") if k not in state]

    def apply(state):
        zn, avg = state["Zn"], state["avg"]
        n, spec = zn.n, zn.spec
        q, qh, sum_pairs = _overlap_symbols(n)
        red = FAMILIES[spec.family].order_parameters(avg, n, sum_pairs)
        if not isinstance(red, Reduction):
            red = Reduction(*red)
        site = SITES[spec.variables]
        entropic = site.trace(n, q, qh, red.field)
        expr = red.energy + entropic
        if site.conjugate:
            expr -= sum_pairs(lambda x, y: y * x)
        zero = ({v: 0 for v in q.values()} if isinstance(q, dict)
                else {q[_a, _b]: 0})
        return state.with_(functional=Functional(
            spec, n, N_SYM, q, qh, red.energy, red.correction, entropic, expr,
            zero, red.missing, dict(red.extra or {}), site.conjugate))

    return Step("introduce-order-parameters", requires, apply)


def saddle_equations() -> Step:
    """d expr / d q_ab = 0 and d expr / d qhat_ab = 0, by differentiation.
    Needs an explicit n: sympy does not collapse derivatives of a Sum over
    symbolic replica indices (impose an ansatz first, then continue n).
    The equations are returned unsimplified (e.g. exp ratios, not tanh):
    compare them as sp.simplify((a - b).rewrite(sp.exp)) == 0, or by a
    numerical residual at a few random points."""
    def requires(state):
        if "functional" not in state:
            return ["functional"]
        return [] if isinstance(state["functional"].n, int) else \
            ["explicit-replica-number"]

    def apply(state):
        F = state["functional"]
        eqs = [sp.Eq(sp.diff(F.expr, v), 0)
               for v in list(F.q.values()) + list(F.qhat.values())]
        return state.with_(equations=eqs)

    return Step("saddle-equations", requires, apply)


# --- replica symmetry, n -> 0, stationarity ------------------------------------

@dataclass(frozen=True)
class RSFunctional:
    """(1/N) ln E Z^n under the RS ansatz, as a function of symbolic n and the
    RS order parameters `params`; `rules` lists the reductions applied."""
    spec: ModelSpec
    n: sp.Symbol
    expr: sp.Expr
    params: tuple
    rules: tuple
    caveat: str = CAVEAT
    sites: tuple = ()           # IsingSiteRS records of the reduced site traces


@dataclass(frozen=True)
class RSFree:
    """phi(params) = lim_{n->0} (1/(N n)) ln E Z^n (RS), the per-replica functional."""
    spec: ModelSpec
    expr: sp.Expr
    params: tuple
    caveat: str = CAVEAT
    sites: tuple = ()


@dataclass(frozen=True)
class IsingSiteRS:
    """What rs_ansatz learned reducing one Ising site trace: the RS pair
    coupling A (the conjugate qhat), the field B, the Hubbard-Stratonovich
    variable z and the per-replica local field u = sqrt(A) z + B, with the
    single-replica factor single(U) = sum_{s=+-1} exp(U s).  Read by
    derive_replicon (the site's replicon susceptibility)."""
    A: sp.Expr
    B: sp.Expr
    z: sp.Symbol
    u: sp.Expr
    single: sp.Lambda


_Z = sp.Symbol("z", real=True)      # Hubbard-Stratonovich / replica-split variables
_T = sp.Symbol("t", real=True)
_s = sp.IndexedBase("s")


def _rs_reduce(F):
    """Apply the RS ansatz to the formal symbolic-n functional F.expr."""
    n, rules, sites = F.n, [], []
    bases = {F.q: Q_RS}
    if F.conjugate:
        bases[F.qhat] = QHAT_RS
    extra = {name: sp.Symbol(name, real=True) for name in (F.extra or {})}
    for name, base in (F.extra or {}).items():
        bases[base] = extra[name]

    def rs(e):
        e = sp.sympify(e).replace(lambda x: isinstance(x, sp.Indexed) and x.base in bases,
                                  lambda x: bases[x.base])
        if any(x.base != _s for x in e.atoms(sp.Indexed)):
            raise Refusal("precondition:rs-reducible", f"unknown replica-indexed object in {e}")
        return e

    def free_of_indices(e, what):
        if e.has(_a) or e.has(_b) or e.has(_s):
            raise Refusal("precondition:rs-reducible", f"{what} {e} still depends on replicas")
        return e

    def site(tr):
        # log Tr_s exp(sum_{a<b} A s_a s_b + sum_a B s_a), Ising spins
        (arg,) = tr.args
        if not isinstance(arg, sp.exp):
            raise Refusal("precondition:rs-site-trace-form", str(tr))
        A = B = sp.S(0)
        for term in sp.Add.make_args(arg.args[0]):
            if not isinstance(term, sp.Sum):
                raise Refusal("precondition:rs-site-trace-form", str(term))
            body = rs(term.function)
            if term.limits == ((_b, _a + 1, n), (_a, 1, n)):
                A += free_of_indices(sp.cancel(body / (_s[_a] * _s[_b])), "pair coupling")
            elif term.limits == ((_a, 1, n),):
                B += free_of_indices(sp.cancel(body / _s[_a]), "field")
            else:
                raise Refusal("precondition:rs-site-trace-form", str(term.limits))
        # Ising: sum_{a<b} s_a s_b = (S^2 - n)/2, S = sum_a s_a, then
        # exp(A S^2/2) = E_z exp(sqrt(A) z S): replicas decouple at fixed z.
        S = sp.Symbol("S", real=True)
        hs = gauss.hubbard_stratonovich(A, S, _Z) if A != 0 else sp.exp(0 * S)
        u = (sp.cancel(sp.diff(hs.f, S) / hs.f) if isinstance(hs, gauss.GaussE) else 0) + B
        single = sp.Add(*(sp.exp(u * v) for v in (1, -1)))      # Tr over one replica
        U = sp.Dummy("U", real=True)
        sites.append(IsingSiteRS(A, B, _Z, u, sp.Lambda(U, sp.Add(*(sp.exp(U * v)
                                                                   for v in (1, -1))))))
        rules.append("site: Ising s^2 = 1, Hubbard-Stratonovich on (sum_a s_a)^2, "
                     "product over replicas of identical factors -> factor^n")
        return -n * A / 2 + sp.log(gauss.GaussE(single**n, _Z))

    def logdet(e):
        d, c, nn = e.args
        rules.append("LogDet: RS spectrum (n-1) ln(d-c) + ln(d+(n-1)c)")
        return gauss.rs_logdet(free_of_indices(rs(d), "diagonal"),
                               free_of_indices(rs(c), "off-diagonal"), nn)

    def gprod(e):
        g, c, nn = e.args
        q = free_of_indices(rs(c), "covariance")
        lam = gauss.rs_split(q, _T, _Z)
        assumed = []
        inner = gauss.expect_closed(g(lam), _Z, assumed)
        rules.append(f"LnGaussProd: RS split lambda_a = {lam} (t, z_a iid N(0,1)), "
                     f"product over replicas -> [E_z g]^n, E_z g = {inner}"
                     + "".join(f" ({x})" for x in assumed))
        return sp.log(gauss.GaussE(inner**nn, _T))

    def rsum(S):
        body = free_of_indices(rs(S.function), "summand")
        if S.limits == ((_b, _a + 1, n), (_a, 1, n)):
            rules.append("pair sum: (1/2) ansatz.sum_offdiag_power('rs', k=1) = n(n-1)/2 terms")
            return sum_offdiag_power("rs", [], [body], 1, n) / 2
        if S.limits == ((_a, 1, n),):
            rules.append("replica sum: n identical terms")
            return n * body
        raise Refusal("precondition:rs-reducible", f"sum limits {S.limits}")

    e = F.expr
    e = e.replace(lambda x: isinstance(x, sp.log) and x.args[0].func == Tr,
                  lambda x: site(x.args[0]))
    e = e.replace(lambda x: x.func == LogDet, logdet)
    e = e.replace(lambda x: x.func == LnGaussProd, gprod)
    e = e.replace(lambda x: isinstance(x, sp.Sum), rsum)
    if e.atoms(sp.Indexed) or e.has(Tr):
        raise Refusal("precondition:rs-reducible", f"left over after RS: {e}")
    params = (Q_RS,) + ((QHAT_RS,) if F.conjugate else ()) + tuple(extra.values())
    return e, params, tuple(dict.fromkeys(rules)), tuple(sites)


def rs_ansatz() -> Step:
    """q_ab = q, qhat_ab = qhat, m_a = m for all a != b (RS) in the symbolic-n
    functional; formal traces and Gaussian integrals reduced by gauss.py rules."""
    def requires(state):
        if "functional" not in state:
            return ["functional"]
        return [] if isinstance(state["functional"].n, sp.Symbol) else \
            ["symbolic-replica-number"]

    def apply(state):
        F = state["functional"]
        expr, params, rules, sites = _rs_reduce(F)
        return state.with_(rs=RSFunctional(F.spec, F.n, expr, params, rules, sites=sites))

    return Step("rs-ansatz", requires, apply,
                note=lambda s: "RS; rules: " + "; ".join(s["rs"].rules))


def replica_limit() -> Step:
    """phi = lim_{n->0} (1/n) RS functional (gauss.replica_limit)."""
    def requires(state):
        return [] if "rs" in state else ["rs"]

    def apply(state):
        R = state["rs"]
        phi = gauss.replica_limit(R.expr, R.n)
        return state.with_(rs_free=RSFree(R.spec, phi, R.params, sites=R.sites))

    return Step("replica-limit", requires, apply,
                note=lambda s: ("n -> 0: polynomial part by ansatz.n_to_zero, the rest by "
                                "d/dn at n = 0, i.e. (1/n) ln E[f^n] -> E ln f; "
                                + s["rs_free"].caveat))


def rs_stationarity() -> Step:
    """d phi / d p = 0 for every RS order parameter p (unsimplified: exp ratios,
    explicit z factors; compare with agree.equations_agree)."""
    def requires(state):
        return [] if "rs_free" in state else ["rs_free"]

    def apply(state):
        P = state["rs_free"]
        return state.with_(rs_equations=[sp.Eq(sp.diff(P.expr, p), 0) for p in P.params])

    return Step("rs-stationarity", requires, apply,
                note=lambda s: "d/d" + ", d/d".join(str(p) for p in s["rs_free"].params))


def annealed_log_Z(spec: ModelSpec, N) -> sp.Expr:
    """ln E Z exactly at finite N (the n = 1 average)."""
    if spec.family not in FAMILIES:
        raise Refusal("precondition:supported-family", spec.family)
    return FAMILIES[spec.family].annealed(spec, N)


from . import derive_families  # noqa: E402,F401  (registers perceptron, hopfield)
