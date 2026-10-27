"""P2, slow tier: the finite-size scaling of the Newman-Ziff crossing probability must first recover the
exact p_c and 1/nu = 3/4 on the solved lattices (square bond, triangular site, triangular bond,
honeycomb bond) before it is applied to square-site percolation; Cardy's formula against the sampled
crossing probability at three aspect ratios; the exact self-dual rectangle at a large size.  Every
band is stated; planted errors (a wrong nu, a wrong p_c, a wrong neighbourhood, a wrong boundary
condition, an inverted aspect ratio) must leave it.  Contrasts the data cannot resolve are printed."""
import json
import os
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.catalog import catalog
from replicant.catalog_percolation_fss import BAND_P, BAND_PJ, BAND_Y
from replicant.compose import compose
from replicant.percolation import cardy as CY
from replicant.percolation import exact2d as X
from replicant.percolation import lattice2d as LT
from replicant.percolation.fss2d import contrast, fss
from replicant.testing import slow

LS, SEEDS, SWEEPS = (16, 32, 64, 128), tuple(range(48)), 25
SEED_SETS = (tuple(range(48)), tuple(range(100, 148)))
BANK = json.loads((Path(__file__).resolve().parent.parent / "reference" / "percolation_lattice.json").read_text())


def _planted(hooks, key, value, fn):
    saved = hooks[key]
    hooks[key] = value
    try:
        return fn()
    finally:
        hooks[key] = saved
        assert hooks[key] is saved


def _recover_one(lattice, dilution, pc, seeds):
    r = fss(lattice, dilution, LS, seeds, SWEEPS)
    res_x = (r.p_c[2] - r.p_c[1]) / 2 + r.p_c[3]
    res_c = (r.p_c_collapse[2] - r.p_c_collapse[1]) / 2 + r.p_c_collapse[3]
    print(f"{lattice} {dilution} seeds {seeds[0]}..{seeds[-1]}: crossing p_c {r.p_c}, collapse p_c "
          f"{r.p_c_collapse} (exact {pc:.6f}); 1/nu {r.inv_nu}; width slope {r.width_slope}; window {r.window:.4g}; "
          f"achieved resolution: crossing +-{res_x:.4f}, collapse +-{res_c:.4f}")
    # the collapse: the truth passes; planted p_c +- 0.005 and nu = 1 fail
    assert r.collapse_test(pc, 0.75)["verdict"] == "collapses", r.collapse_test(pc, 0.75)
    for d in (0.005, -0.005):
        assert r.collapse_test(pc + d, 0.75)["verdict"] == "fails", (d, r.collapse_test(pc + d, 0.75))
    assert r.collapse_test(pc, 1.0)["verdict"] == "fails"
    # the crossing estimator at the resolution it achieves (+- 0.01; +- 0.005 is printed, not asserted)
    assert r.crossing_test(pc) == "inside" and res_x < 0.01, (r.p_c, res_x)
    assert r.crossing_test(pc + 0.01) == r.crossing_test(pc - 0.01) == "outside"
    print("  unasserted: crossing at p_c +- 0.005:", r.crossing_test(pc + 0.005), r.crossing_test(pc - 0.005),
          "; collapse at 1/nu 0.6, 0.9:", r.collapse_test(pc, 0.6)["verdict"], r.collapse_test(pc, 0.9)["verdict"])
    assert contrast(r.inv_nu, 0.75, 1.0)["resolves"]                         # nu = 1 is excluded
    return r


def _recover(lattice, dilution):
    """Two independent named seed sets, each must pass on its own."""
    pc = float(X.threshold(lattice, dilution)[0].mid())
    for seeds in SEED_SETS:
        _recover_one(lattice, dilution, pc, seeds)


@slow
def test_fss_recovers_square_bond():
    _recover("square", "bond")


@slow
def test_fss_recovers_triangular_site():
    _recover("triangular", "site")


@slow
def test_fss_recovers_triangular_bond():
    _recover("triangular", "bond")


@slow
def test_fss_recovers_honeycomb_bond():
    _recover("honeycomb", "bond")


@slow
def test_square_site_after_validation_with_a_contrast_it_cannot_resolve():
    e = catalog()["percolation_lattice"]
    res = compose(e.spec, "sample").run(lattice="square", dilution="site", Ls=LS, seeds=SEEDS, sweeps=SWEEPS, Lx=4)
    v = res.values
    val = v["fss_validation"]
    print("validation on", val["reference"], val)
    assert val["exact p_c inside the crossing band (+ systematic)"] and val["exact p_c inside the collapse band "
                                                                            "(+ systematic)"]
    assert val["exact p_c: collapse at (p_c, 3/4)"] == "collapses"
    assert contrast(val["inv_nu"], 0.75, 0.75)["a_inside"]
    band = v["fss_bands"][BAND_P]
    nz, jac = float(BANK["square_site_pc_nz2000"]["value"]), float(BANK["square_site_pc_jacobsen2015"]["value"])
    c = contrast(band, nz, jac)
    print(f"square site p_c {band}; NZ 2000 {nz} vs Jacobsen 2015 {jac}: {c} (printed, not asserted)")
    assert c["a_inside"] and c["b_inside"]
    cc = contrast(v["fss_bands"][BAND_PJ], nz, jac)
    print(f"collapse p_c: NZ 2000 vs Jacobsen 2015: {cc} (printed, not asserted)")
    assert cc["a_inside"] and cc["b_inside"]
    assert contrast(v["fss_bands"][BAND_Y], 0.75, 0.75)["a_inside"]
    # planted wrong neighbourhood: diagonals turn the square into a triangulation, p_c moves to ~1/2
    bad = _planted(LT._HOOKS, "neighbourhood", "diagonal", lambda: fss("square", "site", LS, SEEDS, SWEEPS))
    print("planted diagonal neighbourhood:", bad.p_c)
    assert not contrast(bad.p_c, nz, nz)["a_inside"] and contrast(bad.p_c, 0.5, 0.5)["a_inside"]


def _mc_crossing(lattice, dilution, W, H, seeds, sweeps, p=0.5):
    r = LT.sample_lattice(lattice, dilution, p, W, H, seeds, sweeps)
    return r.mean["R"], r.halfwidth99["R"]


def _tri_shape(aspect, H):
    """W x H with lattice2d.physical_aspect (the package's one convention) closest to `aspect`."""
    W = round(float(aspect) * H * 3 ** 0.5 / 2)
    return W, H, Fraction(LT.physical_aspect("triangular", "site", W, H)).limit_denominator(10 ** 6)


@slow
def test_cardy_formula_against_sampled_crossings():
    """Triangular site at p = 1/2 (Smirnov's theorem) at width/height 1/2, 1, 2 with H = 48; square
    bond (physics-level) at 2.  Band: the 99% Student-t half-width over 32 seeds plus a systematic, the
    change of Cardy's value when width and height move by one lattice spacing (an assumption)."""
    seeds = tuple(range(32))
    rows = []
    for aspect in ("1/2", "1", "2"):
        W, H, a = _tri_shape(Fraction(aspect), 48)
        mc, hw = _mc_crossing("triangular", "site", W, H, seeds, 25)
        want = float(CY.cardy_ball(a.limit_denominator(10 ** 6)).mid())
        sysm = max(abs(float(CY.cardy_ball((a * (W + s) / W * H / (H - s)).limit_denominator(10 ** 6)).mid())
                       - want) for s in (-1, 1))
        rows.append((aspect, mc, hw, want, sysm))
        assert abs(mc - want) <= hw + sysm, rows[-1]
        if aspect != "1":
            wrong = _planted(CY._HOOKS, "invert_aspect", True,
                             lambda: float(CY.cardy_ball(a.limit_denominator(10 ** 6)).mid()))
            assert abs(mc - wrong) > hw + sysm, (aspect, mc, wrong)
    W, H, a = _tri_shape(2, 48)
    bad, hwb = _planted(LT._HOOKS, "wrap_crossing", True, lambda: _mc_crossing("triangular", "site", W, H, seeds, 25))
    assert abs(bad - rows[-1][3]) > hwb + rows[-1][4], (bad, rows[-1])          # planted periodic crossing
    # square bond (universality, physics-level): width/height counted in bonds (W - 1)/H, as in the
    # self-dual (n + 1) x n rectangle of Bollobas & Riordan's Corollary 5
    W, H = 97, 48
    mc, hw = _mc_crossing("square", "bond", W, H, seeds, 25)
    a = Fraction(LT.physical_aspect("square", "bond", W, H)).limit_denominator(10 ** 6)
    want = float(CY.cardy_ball(a).mid())
    sysm = abs(float(CY.cardy_ball(Fraction(W, H - 1)).mid()) - want)
    print("Cardy rows (aspect, MC, half-width, Cardy, systematic):", rows, ("square bond 2", mc, hw, want, sysm))
    assert abs(mc - want) <= hw + sysm


@slow
def test_self_dual_square_rectangle_is_crossed_with_probability_one_half():
    """Bollobas & Riordan Corollary 5 at a large size: (n + 2) x (n + 1) vertices, n = 63, p = 1/2:
    exactly 1/2 (no finite-size systematic); the planted periodic crossing leaves the band."""
    seeds = tuple(range(32))
    mc, hw = _mc_crossing("square", "bond", 65, 64, seeds, 25)
    assert abs(mc - 0.5) <= hw and hw < 0.05, (mc, hw)
    bad, hwb = _planted(LT._HOOKS, "wrap_crossing", True, lambda: _mc_crossing("square", "bond", 65, 64, seeds, 25))
    assert abs(bad - 0.5) > hwb, (bad, hwb)

