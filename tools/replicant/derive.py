"""Level (a): symbolic derivation steps on sympy expressions.

  replicate(spec, n)           -> State key `Zn`         (formal Z^n)
  average_disorder()           -> State key `avg`        (per-coupling cumulant)
  introduce_order_parameters() -> State key `functional` (q_ab, q_hat_ab)
  saddle_equations()           -> State key `equations`  (sympy.Eq list)
  annealed_log_Z(spec, N)      -> ln E Z, exact at finite N (the n = 1 check)

Conventions: N Ising spins, couplings J_ij (i<j) with variance 1/N,
H = -sum_{i<j} J_ij s_i s_j, Z = sum_s exp(-beta H).  With
X_ij = sum_a s_i^a s_j^a the replicated weight is exp(beta sum_{i<j} J_ij X_ij),
so the disorder enters only through the cumulant generating function
K(t) = ln E exp(t J) at t = beta X.

Extension points (R1: perceptron, Hopfield are added here, not rewritten):
`_LAWS` (coupling laws, by their exact K), `_FAMILIES` (how a Hamiltonian
couples to the disorder and turns the averaged exponent into overlaps) and
`_SITES` (the single-site trace for a variable type).  Anything outside
those registries is a named Refusal.
"""
import itertools
from dataclasses import dataclass
from typing import Callable

import sympy as sp

from .core import Refusal, Step
from .spec import ModelSpec

BETA = sp.Symbol("beta", positive=True)
N_SYM = sp.Symbol("N", positive=True, integer=True)
T = sp.Symbol("t", real=True)       # argument of the cumulant generating function
X = sp.Symbol("X", real=True)       # the coupling field X_ij = sum_a s_i^a s_j^a
Tr = sp.Function("Tr_s")            # formal trace over one site's n replicas


def _beta(spec: ModelSpec):
    return spec.params.get("beta", BETA)


# --- coupling laws ----------------------------------------------------------

@dataclass(frozen=True)
class DisorderLaw:
    name: str
    cgf: Callable[[sp.Expr, sp.Expr], sp.Expr]  # (t, N) -> exact ln E exp(t J)


_LAWS = {
    "gaussian": DisorderLaw("gaussian", lambda t, N: t**2 / (2*N)),   # N(0, 1/N)
    "pm_one": DisorderLaw("pm_one",                                   # +-1/sqrt(N)
                          lambda t, N: sp.log(sp.cosh(t / sp.sqrt(N)))),
}


def _law(spec: ModelSpec) -> DisorderLaw:
    if spec.disorder not in _LAWS:
        raise Refusal("precondition:supported-disorder-law",
                      f"no exact cumulant for disorder {spec.disorder!r}; "
                      f"known: {', '.join(sorted(_LAWS))}")
    return _LAWS[spec.disorder]


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
    """Per-pair averaged exponent: ln E exp(beta J X) ~ `leading` (in X);
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
    `xreplace(q_zero).doit()`: `subs` does not reach inside a Sum)."""
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

def _sk_formal(n, beta):
    i, j = sp.symbols("i j", integer=True, positive=True)
    s, J = sp.IndexedBase("s"), sp.IndexedBase("J")
    field = sp.Sum(s[i, _a] * s[j, _a], (_a, 1, n))
    return Tr(sp.exp(beta * sp.Sum(J[i, j] * field, (j, i + 1, N_SYM),
                                   (i, 1, N_SYM))))


def _sk_annealed(spec, N, law):
    # s_i s_j = +-1 and the law is symmetric, so E exp(beta J s_i s_j) =
    # exp(K(beta)) for every state: ln E Z = N ln 2 + C(N, 2) K(beta).
    if sp.simplify(law.cgf(T, N) - law.cgf(-T, N)) != 0:
        raise Refusal("precondition:symmetric-coupling-law", law.name)
    return N * sp.log(2) + (N * (N - 1) / 2) * law.cgf(_beta(spec), N)


def _sk_energy(avg: Averaged, sum_pairs):
    """c sum_{i<j} X_ij^2 with c = leading / X^2.  Since
    X_ij^2 = sum_{a,b} (s_i^a s_i^b)(s_j^a s_j^b) and q_ab = (1/N) sum_i s_i^a s_i^b
    (q_aa = 1),  sum_{i<j} = (1/2)(sum_{i,j} - sum_{i=j}) gives
    c/2 sum_{a,b} (N^2 q_ab^2 - N)  =  N (cN/2) (n + 2 sum_{a<b} q_ab^2) - cN n^2/2.
    Returns (per-spin energy, O(1) correction)."""
    c = sp.simplify(avg.leading / avg.X**2)
    if c.has(avg.X):
        raise Refusal("precondition:quadratic-in-coupling-field",
                      f"averaged pair exponent {avg.leading} is not c X^2")
    cN = sp.simplify(c * avg.N)
    if cN.has(avg.N):
        raise Refusal("precondition:extensive-scaling",
                      f"coefficient {c} is not proportional to 1/N")
    n = avg.n
    energy = cN / 2 * (n + 2 * sum_pairs(lambda q, qh: q**2))
    return energy, -cN * n**2 / 2


@dataclass(frozen=True)
class Family:
    formal: Callable        # (n, beta) -> formal Z^n
    annealed: Callable      # (spec, N, law) -> exact ln E Z
    energy: Callable        # (Averaged, sum_pairs) -> (energy, correction)


_FAMILIES = {"sk": Family(_sk_formal, _sk_annealed, _sk_energy)}


# --- single-site traces -------------------------------------------------------

def _ising_site(n, qh):
    """ln Tr_{s^1..s^n in {+-1}} exp(sum_{a<b} qhat_ab s^a s^b)."""
    if isinstance(n, int):
        return sp.log(sp.Add(*(
            sp.exp(sp.Add(*(qh[(a, b)] * s[a - 1] * s[b - 1] for a, b in _pairs(n))))
            for s in itertools.product((1, -1), repeat=n))))
    s = sp.IndexedBase("s")
    return sp.log(Tr(sp.exp(sp.Sum(qh[_a, _b] * s[_a] * s[_b],
                                   (_b, _a + 1, n), (_a, 1, n)))))


_SITES = {"ising": _ising_site}


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
        if spec.family not in _FAMILIES:
            missing.append("supported-family")
        if spec.variables not in _SITES:
            missing.append("supported-variables")
        if _valid_n(n) is None:
            missing.append("replica-number")
        return missing

    def apply(state):
        m = _valid_n(n)
        beta = _beta(spec)
        return state.with_(Zn=Replicated(spec, m, beta,
                                         _FAMILIES[spec.family].formal(m, beta)))

    return Step("replicate", requires, apply)


def _order_in_N(expr, N):
    """p such that expr ~ N^-p as N -> oo."""
    num, den = sp.fraction(sp.together(expr))
    return sp.degree(den, N) - sp.degree(num, N)


def average_disorder() -> Step:
    def requires(state):
        if "Zn" not in state:
            return ["Zn"]
        return [] if state["Zn"].spec.disorder in _LAWS else ["supported-disorder-law"]

    def apply(state):
        zn = state["Zn"]
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
            k = next(k for k in (3, 4) if ser.coeff(T, k) != 0)
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
        return state.with_(avg=Averaged(zn.spec, zn.n, law, X, T, N_SYM,
                                        leading, dropped, note))

    return Step("average-disorder", requires, apply,
                note=lambda s: s["avg"].note)


def introduce_order_parameters() -> Step:
    """q_ab = (1/N) sum_i s_i^a s_i^b imposed by
    1 = int dq_ab dqhat_ab exp(-N qhat_ab q_ab + qhat_ab sum_i s_i^a s_i^b)
    (qhat on the imaginary axis); the spins then factorize over sites."""
    def requires(state):
        return [] if "avg" in state else ["avg"]

    def apply(state):
        avg = state["avg"]
        q, qh, sum_pairs = _overlap_symbols(avg.n)
        energy, correction = _FAMILIES[avg.spec.family].energy(avg, sum_pairs)
        entropic = _SITES[avg.spec.variables](avg.n, qh)
        expr = energy - sum_pairs(lambda x, y: y * x) + entropic
        zero = ({v: 0 for v in q.values()} if isinstance(q, dict)
                else {q[_a, _b]: 0})
        return state.with_(functional=Functional(
            avg.spec, avg.n, avg.N, q, qh, energy, correction, entropic,
            expr, zero))

    return Step("introduce-order-parameters", requires, apply)


def saddle_equations() -> Step:
    """d expr / d q_ab = 0 and d expr / d qhat_ab = 0, by differentiation.
    Needs an explicit n: sympy does not collapse derivatives of a Sum over
    symbolic replica indices (impose an ansatz first, then continue n)."""
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


def annealed_log_Z(spec: ModelSpec, N) -> sp.Expr:
    """ln E Z exactly at finite N (the n = 1 average)."""
    if spec.family not in _FAMILIES:
        raise Refusal("precondition:supported-family", spec.family)
    return _FAMILIES[spec.family].annealed(spec, N, _law(spec))
