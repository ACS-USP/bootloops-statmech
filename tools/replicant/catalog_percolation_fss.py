"""The finite-size-scaling step of `percolation_lattice` (route sample, run parameter Ls): p_c from the
crossings of R_L and R_2L and 1/nu from a collapse (percolation/fss2d.py), statistical.

Before a lattice without a closed-form threshold (square or honeycomb site) is analysed, the same
analysis (same Ls, seeds, sweeps) is run on the solved lattice of the same dilution (triangular site
for site, square bond for bond), and whether it recovers the exact p_c and nu = 4/3 is reported in
meta["fss_validation"] (a printed diagnostic: a correct sampler is never refused; the tests assert it
on fixed named seeds).  Values: p_c_crossing (crossings), p_c_collapse and inv_nu_collapse (the joint
collapse), with errors = sqrt((99% half-width / 2.576)^2 + systematic^2) (the systematic an assumption,
not a bound); inv_nu_width_slope (a second estimator, errors from its bootstrap band).
meta["fss_contrasts"]: whether the bands separate nu = 4/3 from nu = 1 and, on solved lattices, contain
the exact p_c (printed, not asserted).
"""
import math

from .catalog import Out, Stability, fstep
from .percolation import exact2d as X
from .percolation.fss2d import contrast, fss

FSS_CITE = ("percolation/fss2d.py (crossings of R_L(p), a collapse in L^(1/nu), bootstrap over named seeds); "
            "Newman & Ziff, arXiv:cond-mat/0005264v2, Eq. (1)")
REFERENCE = {"site": ("triangular", "site"), "bond": ("square", "bond")}
FSS_META = ("fss_crossings", "fss_bands", "fss_validation", "fss_contrasts")
_STAT = Stability("not-computed", "sample", FSS_CITE, "finite-size scaling over named seeds: a consistency check "
                  "with a stated band and an assumed systematic; NOT final")
FSS_FINALITY = {k: _STAT for k in ("p_c_crossing", "p_c_collapse", "inv_nu_collapse", "inv_nu_width_slope")}


def _exact_pc(lattice, dilution):
    got = X.threshold(lattice, dilution)
    return None if got is None else float(got[0].mid())


def _eff(t):
    est, lo, hi = t[:3]
    s = t[3] if len(t) > 3 else 0.0
    return math.hypot((hi - lo) / 2 / 2.576, s)


def _summary(r, exact_pc):
    nu43, nu1 = 0.75, 1.0
    out = {"nu 4/3 vs nu 1": contrast(r.inv_nu, nu43, nu1)}
    if exact_pc is not None:
        out["exact p_c inside the crossing band (+ systematic)"] = contrast(r.p_c, exact_pc, exact_pc)["a_inside"]
        out["exact p_c inside the collapse band (+ systematic)"] = contrast(r.p_c_collapse, exact_pc,
                                                                              exact_pc)["a_inside"]
        out["exact p_c: collapse at (p_c, 3/4)"] = r.collapse_test(exact_pc, nu43)["verdict"]
    return out


def fss_step(when, lat_of):
    def go(p):
        lattice, dilution = lat_of(p)
        Ls, seeds, sweeps = p["Ls"], p.get("seeds", tuple(range(8))), p.get("sweeps", 16)
        validation = "not needed: closed-form threshold"
        exact_pc = _exact_pc(lattice, dilution)
        if exact_pc is None:
            ref = REFERENCE[dilution]
            rv = fss(*ref, Ls, seeds, sweeps)
            validation = {"reference": f"{ref[0]} {ref[1]}", "p_c": rv.p_c, "inv_nu": rv.inv_nu,
                          **_summary(rv, _exact_pc(*ref))}
        r = fss(lattice, dilution, Ls, seeds, sweeps)
        vals = {"p_c_crossing": r.p_c[0], "p_c_collapse": r.p_c_collapse[0], "inv_nu_collapse": r.inv_nu[0],
                "inv_nu_width_slope": r.width_slope[0]}
        errs = {"p_c_crossing": _eff(r.p_c), "p_c_collapse": _eff(r.p_c_collapse),
                "inv_nu_collapse": _eff(r.inv_nu), "inv_nu_width_slope": _eff(r.width_slope)}
        meta = {"fss_crossings": {f"{a}-{b}": c for (a, b), c in r.crossings.items()},
                "fss_bands": {"p_c (est, lo, hi, systematic)": r.p_c, "inv_nu (est, lo, hi, systematic)": r.inv_nu,
                              "width_slope (est, lo, hi)": r.width_slope,
                              "p_c of the joint collapse (est, lo, hi, systematic)": r.p_c_collapse},
                "fss_validation": validation, "fss_contrasts": _summary(r, exact_pc)}
        return Out({**vals, **meta}, f"FSS Ls = {Ls}: crossing p_c {r.p_c[0]:.5f} [{r.p_c[1]:.5f}, "
                   f"{r.p_c[2]:.5f}] "
                   f"+- {r.p_c[3]:.2g} (syst.); collapse p_c {r.p_c_collapse[0]:.5f}, 1/nu {r.inv_nu[0]:.3f} "
                   f"[{r.inv_nu[1]:.3f}, {r.inv_nu[2]:.3f}] +- {r.inv_nu[3]:.2g}", errors=errs,
                   statistical=tuple(vals),
                   notes=("finite-size scaling: 99% bootstrap bands over the named seeds plus an assumed "
                          "systematic (crossings: the drift between the last two; collapse: the shift without "
                          "the smallest L or the spread to the other estimator); a consistency check, no "
                          "proven bound",))
    return fstep("finite-size-scaling", FSS_CITE, go, when=when, skip="no Ls given")


__all__ = ["FSS_FINALITY", "FSS_META", "fss_step"]
