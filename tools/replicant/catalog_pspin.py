"""Catalog entries `pspin` (spherical pure p-spin, Crisanti-Sommers) and `ising_pspin`
(Ising p-spin, Gardner 1985), both 1RSB at zero field, couplings of variance
p!/(2 N^(p-1)) (J = 1), default p = 3.  Split out of catalog_entries.py.

Finality (E4): every 1RSB value carries the COMPUTED innermost-block 1RSB replicon of
families/gardner.py: `stable` where it is positive, `unstable` where it is negative.  The
spherical model has no Gardner transition (replicon 1 - (p-1) y > 0 on the whole static
line), so its 1RSB values are `stable` for all T < T_s; the Ising p-spin is `stable` for
T_G < T < T_s and `unstable` below T_G.  Computed in the replicon sector only (gardner.py
lists what is not computed).
"""
import mpmath as mp

from .catalog import Out, RefCheck, Stability, entry, fstep, from_flag, has, lacks
from .families import gardner as G
from .families import ising_pspin as I
from .families import pspin as P
from .families.common import CAVEAT
from .spec import ModelSpec

PSPIN_CITE = "Crisanti & Sommers, Z. Phys. B 87, 341 (1992)"
ISING_CITE = ("Gardner, Nucl. Phys. B 257, 747 (1985); Montanari & Ricci-Tersenghi, EPJB 33, "
              "339 (2003); Rizzo, PRE 88, 032135 (2013)")
REPLICON = G.GARDNER_CITE


def _saddle(stable, **values):
    return {"ansatz": "1rsb", "stable": stable, **values}


def _computed(key, where):
    def label(v):
        lam = v[key]
        return from_flag(bool(lam > 0), "1rsb", REPLICON,
                         f"computed 1RSB replicon {mp.nstr(lam, 8)} {where} (innermost block; "
                         "outer block 1; longitudinal modes not computed)")
    return label


# ---- spherical p-spin ----------------------------------------------------------------
def _pspin_t(p):
    k = p.get("p", 3)
    with mp.workdps(30):
        ts, qs = P.pspin_critical_temperature(k), 1 - P._y(k)
        lam = G.replicon_formula("spherical", k, 1 / ts, qs, 1)
        return Out({"p": k, "T_s": ts, "T_d": P.pspin_critical_temperature(k, kind="dynamic"),
                    "q_s": qs, "replicon_1rsb_Ts": lam},
                   f"T_s and q_s = 1 - y at m = 1; T_d closed form; 1RSB replicon at T_s "
                   f"{mp.nstr(lam, 8)} (= 1 - (p-1) y on the whole static line: no Gardner "
                   "transition)")


def _pspin_1rsb(p):
    k = p.get("p", 3)
    s = P.pspin_spherical_1rsb(k, p["beta"])
    lam = G.replicon_formula("spherical", k, p["beta"], s.q1, s.m)
    return Out(_saddle(bool(lam > 0), q0=s.q0, q1=s.q1, m=s.m, f=s.f, e=s.e,
                       replicon_1rsb=lam),
               f"1RSB, audited as a local maximum of f in (q0, q, m); 1RSB replicon "
               f"{mp.nstr(lam, 8)}")


def pspin():
    steps = [fstep("pspin-transitions", PSPIN_CITE + "; Castellani & Cavagna, J. Stat. Mech. "
                   "(2005) P05012 (T_d); " + REPLICON + " (replicon)", _pspin_t),
             fstep("pspin-1rsb", PSPIN_CITE + "; " + REPLICON + " (replicon)", _pspin_1rsb,
                   when=has("beta"), skip="no beta given")]
    refs = [RefCheck("pspin", "ts_p3", lambda v: v["T_s"]),
            RefCheck("pspin", "td_p3", lambda v: v["T_d"]),
            RefCheck("pspin", "qs_p3", lambda v: v["q_s"]),
            RefCheck("pspin", "e0_p3", lambda v: v["e"], params={"beta": 10000}, slow=True)]
    head = _computed("replicon_1rsb_Ts", "at T_s (no Gardner transition: the same value on "
                     "the whole static line, gardner.pspin_gardner_temperature refuses)")
    point = _computed("replicon_1rsb", "at this beta")
    fin = {"T_s": head, "q_s": head,
           **{k: point for k in ("q0", "q1", "m", "f", "e", "replicon_1rsb")},
           "T_d": Stability("exact-by-literature", "1rsb",
                            "Castellani & Cavagna, J. Stat. Mech. (2005) P05012, Eq. (147)",
                            "dynamic (mode-coupling) transition, closed form")}
    return entry("pspin", ModelSpec("pspin", "spherical", "gaussian", params={"p": 3}),
                 PSPIN_CITE, {"replica": steps}, refs, {"replica": CAVEAT},
                 "spherical p-spin: static and dynamic transitions, 1RSB with its replicon",
                 finality=fin, meta=("p", "replicon_1rsb_Ts"),
                 parameters={"replica": ("p", "beta")})


# ---- Ising p-spin --------------------------------------------------------------------
def _ising_t(p):
    k = p.get("p", 3)
    ts, td = I.ising_pspin_critical_temperature(k), I.ising_pspin_critical_temperature(k, "dynamic")
    st = G.onersb_stability("ising_pspin", p=k, beta=1 / ts, m=1)
    tg = G.pspin_gardner_temperature(k, variables="ising")
    return Out({"p": k, "T_s": ts, "T_d": td, "T_G": tg, "q_s": st.q1,
                "replicon_1rsb_Ts": st.replicon},
               f"T_s (m_s = 1), T_d (fold of the m = 1 q equation), T_G (zero of the static-"
               f"line 1RSB replicon); replicon at T_s {mp.nstr(st.replicon, 8)}")


def _ising_1rsb(p):
    k = p.get("p", 3)
    s = I.ising_pspin_1rsb(k, p["beta"])
    lam = G.replicon_formula("ising", k, p["beta"], s.q1, s.m)
    return Out(_saddle(bool(lam > 0), q0=s.q0, q1=s.q1, m=s.m, f=s.f, e=s.e,
                       replicon_1rsb=lam), f"{s.note}; 1RSB replicon {mp.nstr(lam, 8)}")


def ising_pspin():
    steps = [fstep("ising-pspin-transitions", ISING_CITE + "; " + REPLICON + " (replicon)",
                   _ising_t, when=lacks("beta"), skip="beta given (state point)"),
             fstep("ising-pspin-1rsb", ISING_CITE + "; " + REPLICON + " (replicon)",
                   _ising_1rsb, when=has("beta"), skip="no beta given")]
    refs = [RefCheck("ising_pspin", k, (lambda key: lambda v: v[key])(key))
            for k, key in (("ts_p3", "T_s"), ("td_p3", "T_d"), ("tg_p3", "T_G"), ("qs_p3", "q_s"),
                           ("replicon_ts_p3", "replicon_1rsb_Ts"))]
    head = _computed("replicon_1rsb_Ts", "at T_s, m = 1")
    point = _computed("replicon_1rsb", "at this beta (negative below T_G: FRSB, not computed)")
    fin = {"T_s": head, "q_s": head,
           "T_G": lambda v: Stability(
               "stable", "1rsb", REPLICON,
               "the static-line 1RSB replicon is computed positive for T_G < T < T_s (it is "
               f"{mp.nstr(v['replicon_1rsb_Ts'], 6)} at T_s) and T_G is its zero, approached "
               "from the stable side; below T_G the 1RSB is unstable (FRSB, not computed)"),
           "T_d": Stability("not-computed", "1rsb", ISING_CITE.split("; ")[2],
                            "dynamic transition of the 1RSB scheme: the m = 1 states are "
                            "marginal there (replicon zero), and threshold states lie in the "
                            "1RSB-unstable region (Montanari & Ricci-Tersenghi 2003, Sec. 2): "
                            "the 1RSB T_d is NOT final"),
           **{k: point for k in ("q0", "q1", "m", "f", "e", "replicon_1rsb")}}
    return entry("ising_pspin", ModelSpec("pspin", "ising", "gaussian", params={"p": 3}),
                 ISING_CITE, {"replica": steps}, refs, {"replica": CAVEAT},
                 "Ising p-spin: static, dynamic and Gardner transitions, 1RSB with its "
                 "replicon", finality=fin, meta=("p", "replicon_1rsb_Ts"),
                 parameters={"replica": ("p", "beta")})


__all__ = ["ising_pspin", "pspin"]
