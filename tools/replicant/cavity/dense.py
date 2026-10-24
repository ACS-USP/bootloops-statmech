"""Cavity route, replica symmetric, on fully connected models.

A second, independent route to the RS quantities of families/sk.py,
families/hopfield.py and families/perceptron.py.  The cavity argument is
written as four typed Steps (core.Step: preconditions, trace, citation):

  remove_site       remove one site (spin s_0, or the pattern / constraint 0);
                    State keys: route = "cavity", cavity_system
  cavity_field_law  the law of the field the rest of the system exerts on it,
                    in the system without it; State key: field_law
  add_site          put it back: its marginal, the Onsager reaction (the shift
                    of the cavity field caused by the site itself) and its
                    response; State key: site
  self_consistency  the removed site is typical: close the equations, solve,
                    audit, label stability; State key: solution

The model-specific content of each step lives in a model object (SKCavity
here; HopfieldCavity in dense_hopfield.py; GardnerCavity in
dense_perceptron.py).  None of this imports the replica families: the
self-consistency maps come from the cavity argument, and the numerics are
the cavity route's own (cavity/_quad.py, float64).  The replica functions
appear only in the tests, as the comparison target.

  run_cavity(model, **params)        -> (State, Trace), route "cavity"
  cavity_sk_rs(beta, h=0, *, require_stable=False)   -> RSSolution
  cavity_hopfield_rs(alpha, beta, branch="retrieval") -> HopfieldSolution
  cavity_hopfield_energy(alpha, beta, branch="retrieval") -> e per spin
  cavity_gardner_entropy(alpha, variables, kappa=0)  -> S per coupling
  cavity_gardner_capacity(variables, kappa=0)        -> Capacity

SK cavity argument (Mezard, Parisi & Virasoro, "Spin Glass Theory and
Beyond" (1987), ch. V; J_ij ~ N(0, 1/N), field h, beta = 1/T):
  remove s_0.  Cavity field h_c = h + sum_i J_0i s_i in the system without s_0:
    the J_0i are independent of that system, so h_c is Gaussian with thermal
    mean u = h + sum_i J_0i m_i (disorder law N(h, q), q = (1/N) sum m_i^2)
    and thermal variance sum_i J_0i^2 (1 - m_i^2) = 1 - q.
  add s_0: weight exp(beta s_0 h_c) on the Gaussian h_c gives
    m_0 = tanh(beta u) and, given s_0, h_c ~ N(u + beta (1 - q) s_0, 1 - q):
    the Onsager reaction beta (1 - q) s_0.  Response dm_0/du = beta (1 - m_0^2).
  self-consistency q = E_z m_0^2 (u = h + sqrt(q) z).  At h = 0 the q-map is
    q -> beta^2 q + O(q^2): q > 0 exists iff beta > 1; for beta <= 1 the
    cavity field has zero variance and q = 0 (paramagnet).  Below T_c the
    q > 0 branch is the one returned (as on the replica side).
  energy per spin: e = -(1/2) E <s_0 (h_c - h)> - h E m_0, with
    E <s_0 (h_c - h)> = E[m_0 sqrt(q) z] + beta (1 - q) (reaction term),
    computed by quadrature (no Stein identity).
  free energy (a separate cavity formula, ln Z_{N+1} - ln Z_N): adding s_0
    gives E ln 2 cosh(beta u) + beta^2 (1 - q)/2 (the thermal Gaussian of h_c),
    and the coupling rescaling 1/N -> 1/(N + 1) of the old couplings changes
    ln Z_N by -beta^2 d(-beta f)/d(beta_J^2) = beta e_J / 2, with e_J the
    coupling part of e above;  -beta f = E ln 2cosh + beta^2 (1-q)/2 + beta e_J/2.
  stability: the spin-glass susceptibility chi_SG = (1/N) sum_ij <s_i s_j>_c^2
    obeys, by the cavity, chi_SG = 1 + beta^2 E(1 - m_0^2)^2 chi_SG: it is
    finite iff lambda = 1 - beta^2 E(1 - m_0^2)^2 > 0.  This is the same
    quantity as the de Almeida-Thouless replicon, derived here from the
    cavity susceptibility; stable = lambda >= 0 (replica convention).

Honest label: the cavity method (clustering of the Gibbs measure, Gaussian
cavity fields, the RS single-state assumption) is physics-level, not proven
here; outputs carry CAVITY_CAVEAT (families/common.py), not the replica CAVEAT.
Numbers carry quadrature error only (float64).
"""
import math
from dataclasses import dataclass
from functools import lru_cache

import mpmath as mp
import numpy as np

from ..core import Refusal, State, Step, run_pipeline
from ..families.common import CAVITY_CAVEAT, RSSolution
from . import _quad as Q
from .dense_hopfield import HopfieldCavity, energy as _hop_energy
from .dense_perceptron import GardnerCavity, capacity as _capacity

STEP_NAMES = ("remove_site", "cavity_field_law", "add_site", "self_consistency")
_GENERIC_CITE = "Mezard, Parisi & Virasoro, Spin Glass Theory and Beyond (1987), ch. V"


def _need(*keys):
    return lambda s: [k for k in keys if k not in s]


def _route(s, key):
    desc = s[key]["desc"] if isinstance(s[key], dict) else getattr(s[key], "desc", "")
    return f"route=cavity; {desc}"


def cavity_steps(model=None):
    """The four cavity Steps; citations from `model` (else the MPV book)."""
    cite = (lambda k: model.cites[k]) if model is not None else (lambda k: _GENERIC_CITE)

    def remove(s):
        m = s["model"]
        p = m.check(dict(s["params"]))
        return s.with_(route="cavity", params=p, cavity_system=m.remove(p))

    def law(s):
        return s.with_(field_law=s["model"].field_law(s["params"], s["cavity_system"]))

    def add(s):
        return s.with_(site=s["model"].add(s["params"], s["cavity_system"], s["field_law"]))

    def close(s):
        m = s["model"]
        return s.with_(solution=m.close(s["params"], s["cavity_system"], s["field_law"],
                                        s["site"]))

    return [
        Step("remove_site", _need("model", "params"), remove, cite("remove_site"),
             lambda s: _route(s, "cavity_system")),
        Step("cavity_field_law", _need("model", "cavity_system"), law, cite("cavity_field_law"),
             lambda s: _route(s, "field_law")),
        Step("add_site", _need("model", "field_law"), add, cite("add_site"),
             lambda s: _route(s, "site")),
        Step("self_consistency", _need("model", "site"), close, cite("self_consistency"),
             lambda s: "route=cavity; " + s["model"].summary(s["solution"])),
    ]


def run_cavity(model, **params):
    """Run the cavity pipeline for `model` (a name in MODELS or a model object)."""
    if isinstance(model, str):
        if model not in MODELS:
            raise Refusal("precondition:known-model",
                          f"cavity model {model!r}: known {', '.join(sorted(MODELS))}")
        model = MODELS[model]
    return run_pipeline(cavity_steps(model), State(model=model, params=params))


def _mpf(x):
    return mp.mpf(repr(x) if isinstance(x, float) else x)


# ---- SK -----------------------------------------------------------------------------
@dataclass(frozen=True)
class _Law:
    desc: str
    mean: float          # disorder mean of the cavity field's thermal mean u
    var_of: object       # q -> disorder variance of u
    thermal_of: object   # q -> thermal variance of h_c


@dataclass(frozen=True)
class _Site:
    desc: str
    m0: object           # u -> <s_0>
    reaction_of: object  # q -> shift of <h_c | s_0> per unit s_0
    response: object     # u -> d<s_0>/du


class SKCavity:
    name = "sk"
    cites = {"remove_site": _GENERIC_CITE + " (remove one spin of the SK model)",
             "cavity_field_law": _GENERIC_CITE + " (Gaussian cavity field)",
             "add_site": _GENERIC_CITE + "; Onsager reaction term: Thouless, Anderson & "
                                         "Palmer, Phil. Mag. 35, 593 (1977)",
             "self_consistency": _GENERIC_CITE + " (RS closure q = E tanh^2)"}

    def __init__(self, reaction=True):
        self.reaction = reaction        # False: the planted error (no Onsager term)

    def check(self, p):
        b, h = _mpf(p.get("beta")), _mpf(p.get("h", 0))
        if not b > 0:
            raise Refusal("precondition:beta-positive", f"beta = 1/T must be > 0, got {b}")
        return {"beta": b, "h": h, "require_stable": bool(p.get("require_stable", False))}

    def remove(self, p):
        return {"desc": "removed spin s_0; couplings J_0i ~ N(0, 1/N) independent of the "
                        "cavity system"}

    def field_law(self, p, sys_):
        h = float(p["h"])
        return _Law("h_c ~ N(u, 1 - q) thermally, u ~ N(h, q) over J_0i", h,
                    lambda q: q, lambda q: 1 - q)

    def add(self, p, sys_, law):
        b = float(p["beta"])
        react = (lambda q: b * law.thermal_of(q)) if self.reaction else (lambda q: 0.0)
        return _Site("m_0 = tanh(beta u); reaction <h_c|s_0> - u = "
                     + ("beta (1 - q) s_0" if self.reaction else "0 (PLANTED ERROR)"),
                     lambda u: np.tanh(b * u), react,
                     lambda u: b * (1 - np.tanh(b * u)**2))

    def close(self, p, sys_, law, site):
        return _sk_close(p["beta"], p["h"], p["require_stable"], law, site)

    def summary(self, sol):
        return (f"q = {mp.nstr(sol.q, 12)}, stable = {sol.stable} "
                f"(cavity susceptibility lambda = {mp.nstr(sol.replicon, 6)})")


def _sk_close(beta, h, require_stable, law, site):
    b, hh = float(beta), float(h)

    def avg(fn, q):
        s = math.sqrt(law.var_of(q)) if q > 0 else 0.0
        if s == 0:
            return float(fn(np.array([law.mean]), np.array([0.0]))[0])
        z, w = Q.nodes(Q.kink_cuts(-law.mean / s, 1 / (b * s)))
        return float(w @ fn(law.mean + s * z, z))

    Qmap = lambda q: avg(lambda u, z: site.m0(u)**2, q)
    G = lambda q: Qmap(q) - q
    if hh == 0 and b <= 1:
        q = 0.0                       # Q'(0) = beta^2 <= 1: only q = 0
    else:
        lo = 0.0
        if hh == 0:                    # q = 0 is a root too; bracket the q > 0 one
            lo = next((10.0**-k for k in range(2, 13) if G(10.0**-k) > 0), None)
            if lo is None:
                raise Refusal("no-solution", f"beta = {b} > 1 but the q > 0 cavity branch "
                              "was not bracketed")
        q = Q.illinois(G, lo, 1.0 - 1e-15)
    res = abs(G(q))
    if res > 1e-13:
        raise Refusal("no-solution:audit", f"cavity closure residual {res} at q = {q}")
    s = math.sqrt(q)
    react = site.reaction_of(q)
    corr = avg(lambda u, z: site.m0(u) * s * z, q) + react          # E <s_0 (h_c - h)>
    e_j = -corr / 2
    mag = avg(lambda u, z: site.m0(u), q) if hh != 0 else 0.0
    lncosh = avg(lambda u, z: np.abs(b * u) + np.log1p(np.exp(-2 * np.abs(b * u))), q)
    mbf = lncosh + b * b * law.thermal_of(q) / 2 + b * e_j / 2
    lam = 1 - avg(lambda u, z: site.response(u)**2, q)
    sol = RSSolution(mp.mpf(q), mp.mpf(-mbf / b), mp.mpf(e_j - hh * mag), bool(lam >= 0),
                     mp.mpf(lam), CAVITY_CAVEAT)
    if require_stable and not sol.stable:
        raise Refusal("unstable:rs", f"cavity stability lambda = 1 - beta^2 E(1 - m^2)^2 = "
                      f"{mp.nstr(lam, 6)} < 0 at beta={beta}, h={h} (chi_SG diverges; the "
                      "same quantity as the replicon): the RS cavity solution is invalid")
    return sol


@lru_cache(maxsize=128)
def _sk(beta, h, require_stable):
    return run_cavity("sk", beta=beta, h=h, require_stable=require_stable)[0]["solution"]


def _key(x):
    with mp.workdps(30):
        return mp.nstr(_mpf(x), 25)


def cavity_sk_rs(beta, h=0, *, require_stable=False) -> RSSolution:
    return _sk(_key(beta), _key(h), require_stable)


# ---- Hopfield and perceptron (models in dense_hopfield.py, dense_perceptron.py) -------
@lru_cache(maxsize=128)
def _hop(alpha, beta, branch):
    return run_cavity("hopfield", alpha=alpha, beta=beta, branch=branch)[0]["solution"]


def cavity_hopfield_rs(alpha, beta, branch="retrieval"):
    return _hop(_key(alpha), _key(beta), branch)


def cavity_hopfield_energy(alpha, beta, branch="retrieval"):
    """Cavity energy per spin of the cavity_hopfield_rs point (dense_hopfield.energy)."""
    sol = cavity_hopfield_rs(alpha, beta, branch)
    return mp.mpf(_hop_energy(sol.m, sol.q, sol.r, float(_mpf(alpha)), float(_mpf(beta))))


def cavity_gardner_entropy(alpha, variables, kappa=0):
    return run_cavity("perceptron", alpha=alpha, variables=variables,
                      kappa=kappa)[0]["solution"].S


def cavity_gardner_capacity(variables, kappa=0):
    GardnerCavity().check({"alpha": 0, "variables": variables, "kappa": kappa})
    return _capacity(_key(kappa))


MODELS = {"sk": SKCavity(), "hopfield": HopfieldCavity(), "perceptron": GardnerCavity()}
