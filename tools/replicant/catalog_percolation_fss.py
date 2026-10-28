"""The finite-size-scaling step of `percolation_lattice` (route sample, run parameter Ls): p_c from the
crossings of R_L and R_2L and 1/nu from a collapse (percolation/fss2d.py), statistical.

Before a lattice without a closed-form threshold (square or honeycomb site) is analysed, the same
analysis (same Ls, seeds, sweeps) is run on the solved lattice of the same dilution (triangular site
for site, square bond for bond), and whether it recovers the exact p_c and nu = 4/3 is reported in
meta["fss_validation"] (a printed diagnostic: a correct sampler is never refused; the tests assert it
on fixed named seeds).  Values: p_c_crossing (crossings), p_c_collapse and inv_nu_collapse (the joint
collapse), with errors = the bootstrap standard deviation only; the assumed systematic (not a bound) is
in meta["fss_bands"] and is added outside any t-band (the statistical RefChecks); inv_nu_width_slope (a
second estimator).
meta["fss_contrasts"]: whether the bands separate nu = 4/3 from nu = 1 and, on solved lattices, contain
the exact p_c (printed, not asserted).
meta["fss_status"]: "determined" or why a value is undetermined (undetermined:crossing when the largest
pair's crossing is not resolved, undetermined:collapse when the fit is not finite, undetermined:width
when too few resamples have a nonzero width); an undetermined value
is omitted (never a NaN with a zero stderr) and a note says why.
"""
from .catalog import Out, Stability, fstep
from .percolation import exact2d as X
from .percolation.fss2d import contrast, fss

FSS_CITE = ("percolation/fss2d.py (crossings of R_L(p), a collapse in L^(1/nu), bootstrap over named seeds); "
            "Newman & Ziff, arXiv:cond-mat/0005264v2, Eq. (1)")
REFERENCE = {"site": ("triangular", "site"), "bond": ("square", "bond")}
BAND_P, BAND_Y = "p_c (est, lo, hi, systematic, stderr)", "inv_nu (est, lo, hi, systematic, stderr)"
BAND_PJ = "p_c of the joint collapse (est, lo, hi, systematic, stderr)"
FSS_META = ("fss_crossings", "fss_bands", "fss_validation", "fss_contrasts", "fss_status")
_STAT = Stability("not-computed", "sample", FSS_CITE, "finite-size scaling over named seeds: a consistency check "
                  "with a stated band and an assumed systematic; NOT final")
FSS_FINALITY = {k: _STAT for k in ("p_c_crossing", "p_c_collapse", "inv_nu_collapse", "inv_nu_width_slope")}


def _exact_pc(lattice, dilution):
    got = X.threshold(lattice, dilution)
    return None if got is None else float(got[0].mid())


def _se(t):
    """The bootstrap standard deviation of the estimator (index 4), else the 99% band / (2 x 2.576)."""
    return t[4] if len(t) > 4 else (t[2] - t[1]) / 2 / 2.576


def _summary(r, exact_pc):
    """Contrasts against the bands; a band whose value is undetermined gives "undetermined", never a
    comparison against NaN."""
    nu43, nu1 = 0.75, 1.0
    und = lambda *ks: next((r.undetermined[k] for k in ks if k in r.undetermined), None)
    half = lambda t: (t[2] - t[1]) / 2 + t[3]
    col = und("p_c_collapse", "inv_nu_collapse")
    out = {"nu 4/3 vs nu 1": und("inv_nu_collapse") or contrast(r.inv_nu, nu43, nu1)}
    if exact_pc is not None:
        out["exact p_c inside the crossing band (+ systematic)"] = (
            und("p_c_crossing") or contrast(r.p_c, exact_pc, exact_pc)["a_inside"])
        out["exact p_c inside the collapse band (+ systematic)"] = (
            und("p_c_collapse") or contrast(r.p_c_collapse, exact_pc, exact_pc)["a_inside"])
        out["exact p_c: collapse at (p_c, 3/4)"] = col or r.collapse_test(exact_pc, nu43)["verdict"]
        out["exact p_c: crossing"] = und("p_c_crossing") or r.crossing_test(exact_pc)
    out["resolution: p_c crossing box half-width"] = und("p_c_crossing") or half(r.p_c)
    out["resolution: p_c collapse box half-width"] = und("p_c_collapse") or half(r.p_c_collapse)
    out["resolution: 1/nu box half-width"] = und("inv_nu_collapse") or half(r.inv_nu)
    return out


def _band_text(name, t, why, digits):
    return f"{name} undetermined" if why else f"{name} {t[0]:.{digits}f} [{t[1]:.{digits}f}, {t[2]:.{digits}f}]"


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
                          **_summary(rv, _exact_pc(*ref)), "undetermined": dict(rv.undetermined)}
        r = fss(lattice, dilution, Ls, seeds, sweeps)
        vals = {"p_c_crossing": r.p_c[0], "p_c_collapse": r.p_c_collapse[0], "inv_nu_collapse": r.inv_nu[0],
                "inv_nu_width_slope": r.width_slope[0]}
        errs = {"p_c_crossing": _se(r.p_c), "p_c_collapse": _se(r.p_c_collapse),
                "inv_nu_collapse": _se(r.inv_nu), "inv_nu_width_slope": _se(r.width_slope)}
        for k in r.undetermined:             # labelled, never a NaN with a zero stderr
            vals.pop(k), errs.pop(k)
        meta = {"fss_crossings": {f"{a}-{b}": c for (a, b), c in r.crossings.items()},
                "fss_bands": {BAND_P: r.p_c, BAND_Y: r.inv_nu, "width_slope (est, lo, hi)": r.width_slope,
                              BAND_PJ: r.p_c_collapse, "critical window (std of n*/M at the largest L)": r.window},
                "fss_validation": validation, "fss_contrasts": _summary(r, exact_pc),
                "fss_status": {k: r.undetermined.get(k, "determined") for k in FSS_FINALITY}}
        und = "".join(f"; {k} {why}" for k, why in r.undetermined.items())
        u = r.undetermined
        cross = _band_text("crossing p_c", r.p_c, u.get("p_c_crossing"), 5)
        cross += "" if "p_c_crossing" in u else f" +- {r.p_c[3]:.2g} (syst.)"
        coll = "collapse p_c undetermined" if "p_c_collapse" in u else f"collapse p_c {r.p_c_collapse[0]:.5f}"
        nu = _band_text("1/nu", r.inv_nu, u.get("inv_nu_collapse"), 3)
        nu += "" if "inv_nu_collapse" in u else f" +- {r.inv_nu[3]:.2g}"
        return Out({**vals, **meta}, f"FSS Ls = {Ls}: {cross}; {coll}, {nu}" + und, errors=errs,
                   statistical=tuple(vals),
                   notes=tuple(f"{k} undetermined (value omitted): {why}" for k, why in r.undetermined.items())
                   + ("finite-size scaling: 99% bootstrap percentile bands (1000 replicates over the named "
                          "seeds; errors = the bootstrap standard deviation) plus an assumed systematic: for p_c "
                          "the larger of a tenth of the critical window at the largest L and the resolved drift "
                          "(to the previous crossing; to the fit without the smallest L), for 1/nu the resolved "
                          "drift; a consistency check, no proven bound",))
    return fstep("finite-size-scaling", FSS_CITE, go, when=when, skip="no Ls given")


__all__ = ["BAND_P", "BAND_PJ", "BAND_Y", "FSS_FINALITY", "FSS_META", "fss_step"]
