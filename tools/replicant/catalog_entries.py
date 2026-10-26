"""Catalog entries for the fully connected families (level (b)) and their dense cavity
routes: REM, SK, spherical perceptron, Hopfield, and (catalog_ts.py) teacher-student,
(catalog_pspin.py) the spherical and the Ising p-spin.

Conventions are the families' own (stated in each families/*.py docstring):
  REM       2^N i.i.d. energies of variance N_scale N; default N_scale = 1/2, Derrida's
            J = 1 normalisation (beta_c = 2 sqrt(ln 2)), in which the bank is written.
  SK        J_ij ~ N(0, 1/N), field h, beta = 1/T.
  p-spin    spherical or Ising, J = 1, couplings of variance p!/(2 N^(p-1)); p = 3.
  perceptron spherical J.J = N, P = alpha N Gaussian patterns, margin kappa (default 0).
  teacher-student  spherical teacher and student, noiseless labels; learning gibbs|bayes.
  Hopfield  AGS: p = alpha N +-1 patterns, one condensed pattern, beta = 1/T.
Run parameters: with no state point (no beta / alpha) an entry computes its headline
quantities (transitions, capacities, ground energies); with one it solves the saddle
there.  Every number is the family function's own output; nothing is retyped.
"""
from functools import lru_cache

import mpmath as mp

from .agree import default_spec
from .catalog import Out, RefCheck, Stability, entry, fstep, from_flag, has, lacks, saddle
from .catalog_amp import sk_amp_route
from .catalog_pspin import ising_pspin, pspin
from .catalog_ts import _saddle, teacher_student
from .cavity import dense
from .cavity.dense_hopfield import energy as hopfield_cavity_energy
from .core import Refusal, Step
from .families.common import CAVEAT, CAVITY_CAVEAT
from .families.hopfield import hopfield_alpha_c, hopfield_rs, hopfield_t0
from .families.perceptron import gardner_capacity, gardner_entropy_point
from .families.rem import rem_1rsb, rem_critical_beta, rem_free_energy
from .families.parisi import PARISI_CITE, krsb_ground_energy, parisi_ground_energy
from .families.sk import FULLRSB_GRID, sk_1rsb, sk_ground_energy, sk_replicon, sk_rs
from .spec import ModelSpec

SK_CITE = ("Sherrington & Kirkpatrick, PRL 35, 1792 (1975); de Almeida & Thouless, "
           "J. Phys. A 11, 983 (1978); Parisi, PRL 43, 1754 (1979)")
REM_CITE = "Derrida, PRL 45, 79 (1980); Phys. Rev. B 24, 2613 (1981)"
GARDNER_CITE = "Gardner, J. Phys. A 21, 257 (1988); Gardner & Derrida, J. Phys. A 21, 271 (1988)"
MEZARD_CITE = "Mezard, J. Phys. A 22, 2181 (1989)"
AGS_CITE = ("Amit, Gutfreund & Sompolinsky, PRL 55, 1530 (1985); Ann. Phys. 173, 30 (1987)")
MPV = "Mezard, Parisi & Virasoro, Spin Glass Theory and Beyond (1987), ch. V"
AT_CITE = "de Almeida & Thouless, J. Phys. A 11, 983 (1978) (replicon)"


def cavity_route(model, required, collect, cite):
    """[select the dense cavity model] + the four cavity Steps of cavity/dense.py +
    [read the observables off the solution]."""
    def select(s):
        missing = [k for k in required if s["params"].get(k) is None]
        if missing:
            raise Refusal("precondition:state-point",
                          f"the {model} cavity route needs {', '.join(missing)}")
        return s.with_(model=dense.MODELS[model], step_note=f"cavity model {model!r}")

    def read(s):
        out = collect(s["solution"], s["params"])
        return s.with_(values={**s["values"], **out}, step_note="read " + ", ".join(out))

    note = lambda s: s["step_note"]
    return ([Step("cavity-model", lambda s: [], select, MPV, note)]
            + dense.cavity_steps(dense.MODELS[model])
            + [Step("cavity-observables", lambda s: [] if "solution" in s else ["solution"],
                    read, cite, note)])


# ---- REM -----------------------------------------------------------------------------
def _rem_scale(p):
    return mp.mpf(p.get("N_scale", "0.5"))


def _rem_tc(p):
    with mp.workdps(30):
        bc = rem_critical_beta(N_scale=_rem_scale(p))
        return Out({"N_scale": _rem_scale(p), "beta_c": bc, "T_c": 1 / bc})


def _rem_saddle(p):
    with mp.workdps(30):
        s, b = _rem_scale(p), mp.mpf(p["beta"])
        bc = rem_critical_beta(N_scale=s)
        if b <= bc:
            f = rem_free_energy(b, N_scale=s)
            return Out(_saddle("rs", None, f=f, q=mp.mpf(0)),
                       "paramagnet (beta <= beta_c): annealed = quenched, -beta f = ln 2 "
                       "+ N_scale beta^2/2")
        sol = rem_1rsb(b, N_scale=s)
        return Out(_saddle("1rsb", None, q0=sol.q0, q1=sol.q1, m=sol.m, f=sol.f, e=sol.e,
                           beta=b), "frozen 1RSB, m = beta_c/beta")


def rem():
    steps = [fstep("rem-critical-temperature", REM_CITE + " (s(e) = ln 2 - e^2/(2 N_scale))",
                   _rem_tc),
             fstep("rem-saddle", REM_CITE + "; 1RSB form: Mezard, Parisi & Virasoro (1987)",
                   _rem_saddle, when=has("beta"), skip="no beta given")]
    refs = [RefCheck("rem", "tc_derrida", lambda v: v["T_c"]),
            RefCheck("rem", "ground_energy_derrida", lambda v: v["f"], params={"beta": 3}),
            RefCheck("rem", "m_times_beta_over_beta_c", lambda v: v["m"] * v["beta"] / v["beta_c"],
                     params={"beta": 3})]
    exact = lambda v: Stability(
        "exact-by-literature", v.get("ansatz", "1rsb"),
        "Derrida, Phys. Rep. 67, 29 (1980); Phys. Rev. B 24, 2613 (1981) (exact solution: "
        "microcanonical entropy s(e)); Derrida & Mottishaw, arXiv:1410.1432 (m = T/T_c)")
    fin = {k: exact for k in ("beta_c", "T_c", "f", "q", "q0", "q1", "m", "e")}
    return entry("rem", ModelSpec("rem", "ising", "gaussian"), REM_CITE, {"replica": steps},
                 refs, {"replica": CAVEAT}, "random energy model: T_c, frozen 1RSB phase",
                 finality=fin, meta=("N_scale", "beta"),
                 parameters={"replica": ("beta", "N_scale")})


# ---- SK ------------------------------------------------------------------------------
def _ansatz_is(name):
    return lambda p: p.get("beta") is not None and p.get("ansatz", "rs") == name


def _sk_tc(p):
    b = mp.findroot(lambda x: sk_replicon(x, 0, q=0), mp.mpf("0.9"))
    return Out({"tc": 1 / b}, "T_c: root of the q = 0 replicon 1 - beta^2")


@lru_cache(maxsize=None)
def _sk_replicon_t0p5():
    return sk_replicon("2.0")


def _sk_e0(p):
    lam = _sk_replicon_t0p5()
    return Out({"rs_ground_energy": sk_ground_energy("rs"),
                "rsb1_ground_energy": sk_ground_energy("1rsb"), "rs_replicon_T0p5": lam},
               f"T = 0 energies; RS replicon at T = 0.5 (beta = 2) = {mp.nstr(lam, 6)} "
               "(the h = 0 AT line is T = 1: RS is unstable at all T < 1)")


def _sk_frsb(p):
    r = parisi_ground_energy(grid=FULLRSB_GRID)
    return Out({"fullrsb_ground_energy": r.value,
                "rsb2_ground_energy": mp.mpf(krsb_ground_energy(2).E),
                "rsb3_ground_energy": mp.mpf(krsb_ground_energy(3).E)},
               f"T = 0 full RSB: {r.note}; error estimate {mp.nstr(r.error_estimate, 3)}",
               errors={"fullrsb_ground_energy": r.error_estimate})


def _sk_rs(p):
    s = sk_rs(p["beta"], p.get("h", 0))
    return Out(_saddle("rs", s.stable, q=s.q, f=s.f, e=s.e, replicon=s.replicon),
               f"RS saddle, replicon {mp.nstr(s.replicon, 6)}")


def _sk_1rsb(p):
    s = sk_1rsb(p["beta"], p.get("h", 0))
    return Out(_saddle("1rsb", None, q0=s.q0, q1=s.q1, m=s.m, f=s.f, e=s.e),
               "1RSB saddle (its own stability is not computed)")


def sk():
    steps = [fstep("sk-critical-temperature", SK_CITE.split(";")[0], _sk_tc,
                   when=lacks("beta"), skip="beta given (state point)"),
             fstep("sk-ground-energy", SK_CITE.split(";")[0] + " (RS); Parisi, PRL 43, 1754 "
                   "(1979) (1RSB)", _sk_e0, when=lacks("beta"), skip="beta given (state point)"),
             fstep("sk-fullrsb-ground-energy", PARISI_CITE, _sk_frsb,
                   when=lambda p: p.get("fullrsb") and p.get("beta") is None,
                   skip="full RSB is behind the fullrsb=True flag (slow), at T = 0 only"),
             fstep("sk-rs-saddle", "; ".join(SK_CITE.split("; ")[:2]), _sk_rs,
                   when=_ansatz_is("rs"), skip="no beta, or ansatz != rs"),
             fstep("sk-1rsb-saddle", "Parisi, PRL 43, 1754 (1979)", _sk_1rsb,
                   when=_ansatz_is("1rsb"), skip="no beta, or ansatz != 1rsb")]
    cav = cavity_route("sk", ("beta",), lambda s, p: _saddle(
        "rs", s.stable, q=s.q, f=s.f, e=s.e, replicon=s.replicon), MPV)
    refs = [RefCheck("sk", "tc", lambda v: v["tc"]),
            RefCheck("sk", "rs_ground_energy", lambda v: v["rs_ground_energy"]),
            RefCheck("sk", "rsb1_ground_energy", lambda v: v["rsb1_ground_energy"])] + [
            RefCheck("sk", k, lambda v, k=k: v[k], params={"fullrsb": True}, slow=True)
            for k in ("rsb2_ground_energy", "rsb3_ground_energy", "fullrsb_ground_energy")]
    fin = {"tc": Stability("stable", "rs", AT_CITE, "paramagnet replicon 1 - beta^2 > 0 above "
                           "T_c (computed); T_c is where it vanishes"),
           "rs_ground_energy": lambda v: from_flag(
               bool(v["rs_replicon_T0p5"] > 0), "rs", AT_CITE,
               f"computed RS replicon {mp.nstr(v['rs_replicon_T0p5'], 6)} at T = 0.5"),
           "rsb1_ground_energy": Stability(
               "not-computed", "1rsb", "Parisi, PRL 43, 1754 (1979); J. Phys. A 13, L115 (1980)",
               "1RSB is not the exact SK solution (full RSB is); its stability is not computed"),
           "fullrsb_ground_energy": lambda v: Stability(
               "exact-by-literature", "fullrsb", PARISI_CITE,
               "the Parisi formula is a theorem for SK; this number is its K-step staircase "
               f"extrapolated in K (levels {FULLRSB_GRID - 2}..{FULLRSB_GRID}), numerical "
               "error estimate +- " + mp.nstr(parisi_ground_energy(
                   grid=FULLRSB_GRID).error_estimate, 3) + " (also in the run's errors)"),
           **{k: Stability("not-computed", k[:4], "Parisi, J. Phys. A 13, L115 (1980)",
                           "a finite-K staircase is not the exact SK solution; its value is "
                           "a variational bound below the full-RSB one")
              for k in ("rsb2_ground_energy", "rsb3_ground_energy")},
           **{k: saddle(AT_CITE) for k in ("q", "f", "e", "replicon", "q0", "q1", "m")}}
    amp = sk_amp_route()
    return entry("sk", default_spec("sk"), SK_CITE, {"replica": steps, "cavity": cav,
                                                     "amp": amp.steps}, refs,
                 {"replica": CAVEAT, "cavity": CAVITY_CAVEAT, "amp": amp.caveat},
                 "Sherrington-Kirkpatrick: RS, AT stability, 1RSB, full RSB (flag); AMP / SE",
                 finality={**fin, **amp.finality}, meta=("rs_replicon_T0p5", *amp.meta),
                 parameters={"replica": ("beta", "h", "ansatz", "fullrsb"),
                             "cavity": ("beta", "h", "require_stable"), "amp": amp.params})


# ---- spherical perceptron ----------------------------------------------------------------
def _gardner_cap(fn):
    def go(p):
        c = fn("spherical", p.get("kappa", 0))
        return Out({"alpha_c": c.alpha_c, "capacity_replicon": c.replicon,
                    "capacity_stable": c.stable}, c.note)
    return go


def _gardner_s(p):
    S, pt = gardner_entropy_point(p["alpha"], "spherical", p.get("kappa", 0))
    if pt is None:              # alpha >= alpha_c: S = -inf, no saddle to test
        return Out(_saddle("rs", None, S=S), "alpha >= alpha_c: empty version space, "
                   "S = -inf; no RS saddle, so no replicon")
    return Out(_saddle("rs", bool(pt.replicon > 0), S=S, replicon=pt.replicon),
               f"RS entropy, replicon {mp.nstr(pt.replicon, 6)} (gardner_entropy refuses, "
               "unstable:rs, where the replicon < 0)")


def perceptron():
    steps = [fstep("gardner-capacity", GARDNER_CITE.split(";")[0], _gardner_cap(gardner_capacity)),
             fstep("gardner-entropy", GARDNER_CITE.split(";")[0], _gardner_s, when=has("alpha"),
                   skip="no alpha given")]
    cav = ([fstep("cavity-capacity", MEZARD_CITE, _gardner_cap(dense.cavity_gardner_capacity))]
           + cavity_route("perceptron", (), lambda s, p: _saddle(
               "rs", None if s.stability is None else bool(s.stability > 0), q=s.q, S=s.S,
               stability=s.stability), MEZARD_CITE))
    refs = [RefCheck("perceptron", "spherical_capacity_kappa0", lambda v: v["alpha_c"]),
            RefCheck("perceptron", "spherical_capacity_kappa0", lambda v: v["alpha_c"],
                     route="cavity")]
    return entry("perceptron", default_spec("perceptron"), GARDNER_CITE,
                 {"replica": steps, "cavity": cav}, refs,
                 {"replica": CAVEAT, "cavity": CAVITY_CAVEAT},
                 "spherical perceptron (Gardner volume): capacity alpha_c(kappa), entropy",
                 finality={"alpha_c": lambda v: from_flag(
                               v["capacity_stable"], "rs", GARDNER_CITE.split(";")[0],
                               f"computed replicon at capacity {v['capacity_replicon']}"),
                           "S": lambda v: from_flag(
                               v.get("stable"), "rs", GARDNER_CITE.split(";")[0],
                               f"computed RS replicon {mp.nstr(v['replicon'], 6)}"
                               if "replicon" in v else ""),
                           "replicon": saddle(GARDNER_CITE.split(";")[0]),
                           "q": saddle(MEZARD_CITE)},
                 meta=("capacity_replicon", "capacity_stable", "stability"),
                 parameters={"replica": ("alpha", "kappa"),
                             "cavity": ("alpha", "kappa", "variables")})


# ---- Hopfield -----------------------------------------------------------------------------
def _hop_t0(p):
    ac = hopfield_alpha_c()
    t0 = hopfield_t0(ac, "retrieval")
    return Out({"alpha_c_T0": ac, "m_at_alpha_c_T0": t0.m, "energy_at_alpha_c_T0": t0.E},
               "T = 0 erf reduction: double root of the retrieval equation")


def _hop_global(p):
    with mp.workdps(30):
        d = lambda a: hopfield_t0(a, "retrieval").E - hopfield_t0(a, "spinglass").E
        a = mp.findroot(d, (mp.mpf("0.045"), mp.mpf("0.055")), solver="secant")
    return Out({"alpha_global_T0": a},
               "T = 0: root of E_retrieval - E_spinglass (RS vs RS; the SG state is AT-unstable)")


def _hop_rs(p):
    s = hopfield_rs(p["alpha"], p["beta"], branch=p.get("branch", "retrieval"))
    return Out(_saddle("rs", s.stable, m=s.m, q=s.q, r=s.r, f=s.f, replicon=s.replicon,
                       branch=s.branch), s.note)


def _hop_cavity(s, p):
    e = hopfield_cavity_energy(s.m, s.q, s.r, float(p["alpha"]), float(p["beta"]))
    return _saddle("rs", s.stable, m=s.m, q=s.q, r=s.r, e=mp.mpf(e), replicon=s.replicon,
                   branch=s.branch)


def hopfield():
    t0 = "Amit, Gutfreund & Sompolinsky, PRL 55, 1530 (1985), Eqs. (9)-(11)"
    steps = [fstep("hopfield-t0-capacity", t0, _hop_t0, when=lacks("alpha"),
                   skip="alpha given (state point)"),
             fstep("hopfield-t0-global-minimum", t0, _hop_global, when=lacks("alpha"),
                   skip="alpha given (state point)"),
             fstep("hopfield-rs", AGS_CITE, _hop_rs, when=has("alpha", "beta"),
                   skip="needs alpha and beta")]
    cav = cavity_route("hopfield", ("alpha", "beta"), _hop_cavity, AGS_CITE)
    t0rs = Stability("not-computed", "rs", "Steffan & Kuehn, arXiv:cond-mat/9404036 (1994)",
                     "RS at T = 0: the replicon is not evaluated at T = 0 here; Steffan & Kuehn "
                     "find RSB corrections to the T = 0 retrieval state")
    fin = {"alpha_c_T0": t0rs, "m_at_alpha_c_T0": t0rs, "energy_at_alpha_c_T0": t0rs,
           "alpha_global_T0": Stability("not-computed", "rs", AGS_CITE.split(";")[0],
                                        "compares the RS energies of the retrieval and the "
                                        "(AT-unstable) spin-glass states at T = 0"),
           **{k: saddle(AGS_CITE) for k in ("m", "q", "r", "f", "e", "replicon")}}
    refs = [RefCheck("hopfield", k, (lambda key: lambda v: v[key])(k))
            for k in ("alpha_c_T0", "m_at_alpha_c_T0", "energy_at_alpha_c_T0",
                      "alpha_global_T0")]
    return entry("hopfield", default_spec("hopfield"), AGS_CITE,
                 {"replica": steps, "cavity": cav}, refs,
                 {"replica": CAVEAT, "cavity": CAVITY_CAVEAT},
                 "Hopfield (AGS), RS with branch control: retrieval capacity, T = 0 energies",
                 finality=fin, meta=("branch",),
                 parameters={"replica": ("alpha", "beta", "branch"),
                             "cavity": ("alpha", "beta", "branch")})


def dense_entries():
    return [rem(), sk(), pspin(), ising_pspin(), perceptron(), teacher_student(), hopfield()]


__all__ = ["dense_entries"]
