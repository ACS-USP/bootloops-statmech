"""SK energies per spin from finite N by parallel tempering, extrapolated in N^-omega (E11).

  gaussian_sk_couplings(N, S, seed)      S Gaussian SK matrices J_ij ~ N(0, 1/N) (named seed)
  pm_one_sk_couplings(N, S, seed)        S +-J SK matrices J_ij = +-1/sqrt(N)
  sk_ground_energy_finite_size(Ns, samples, seed, *, sweeps=None, betas=None, conf=0.99)
      -> FSResult(e_inf, band, ...)      ground-state energy, Gaussian J
  sk_energy_finite_size(beta, Ns, samples, seed, *, law="pm_one", sweeps=None, nbetas=None,
                        conf=0.99) -> FSResult    thermal energy at beta (equilibrium PT)

Ground states.  Per size N, S disorder samples; each is run twice by parallel tempering
(pt.pt_sk) with independent seeds, and the estimate is the lower of the two minima.  The two
runs must reach the same minimum (to 1e-9 relative) in every sample, otherwise the call is
refused `undetermined:ground-states` (lesson 7).  Agreement is evidence, not proof (both runs
could miss the same state); tests compare with exact enumeration where N <= 26.

Thermal energy.  One PT run per sample, betas geometric from 0.5 to beta; the time average of
the energy at beta over the second half of the run per sample; equilibration (pt.equilibration,
windows [T/4, T/2) vs [T/2, T)) must pass at every size, otherwise the call is refused
`undetermined:equilibration`.

Extrapolation (both): weighted least squares of the sample means e_N (weights 1/stderr_N^2) on
e_N = e_inf + a N^-omega with omega = 2/3.  Sources for omega = 2/3: Boettcher, Eur. Phys. J. B
46, 501 (2005), arXiv:cond-mat/0407130 (PDF read; +-J ground states, N = 15..1023, fit omega =
0.672(5), "2/3 within 1%", amplitude ~0.70) and Palassini, arXiv:cond-mat/0307713 (PDF read;
Gaussian ground states: with e0 fixed, omega = 0.673 +- 0.002 while omega = 2/3 alone is
rejected, so subleading corrections are visible; he quotes omega = 2/3 for the internal energy
near T_c and the AT line from his ref. 19, "no result is available deep in the ordered phase").
Band = stat + syst:
    stat  t_{(1+conf)/2, df} sigma(e_inf), df = (smallest sample count) - 1;
    syst  the largest shift of e_inf over the variant fits: ground states omega = 3/4 (an
          exponent stress: the fluctuation exponent rho, not a literature omega) and omega =
          2/3 without the smallest N; thermal energy omega = 3/4 and omega = 1 (no exponent is
          known deep in the ordered phase, so the stress is wider), and without the smallest N
          when there are four or more sizes.
Statistical and finite-size: never certified.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from ..amp._stats import student_band, t_quantile
from ..core import Refusal
from .pt import equilibration, geometric_betas, pt_sk

OMEGA = 2 / 3


@dataclass
class FSResult:
    e_inf: float
    band: float
    stat_half_width: float
    systematic: float
    variants: dict            # name -> e_inf of that fit
    sizes: tuple
    means: dict               # N -> sample mean of the per-sample estimate
    stderrs: dict
    per_sample: dict          # N -> array of per-sample estimates
    diagnostics: dict         # N -> text (run agreement or equilibration)
    chi2: float               # of the base fit, len(sizes) - 2 dof
    conf: float
    notes: list = field(default_factory=list)

    def excludes(self, value):
        return abs(float(value) - self.e_inf) > self.band

    def size_band(self, N):
        """The 99% Student-t band of the per-sample mean at size N (no extrapolation)."""
        return student_band(self.per_sample[N], conf=self.conf)


def gaussian_sk_couplings(N, S, seed):
    rng = np.random.default_rng(seed)
    G = np.triu(rng.standard_normal((S, N, N)), 1)
    return (G + G.transpose(0, 2, 1)) / math.sqrt(N)


def pm_one_sk_couplings(N, S, seed):
    rng = np.random.default_rng(seed)
    G = np.triu(rng.choice((-1.0, 1.0), size=(S, N, N)), 1)
    return (G + G.transpose(0, 2, 1)) / math.sqrt(N)


_LAWS = {"gaussian": gaussian_sk_couplings, "pm_one": pm_one_sk_couplings}


def _fit(Ns, means, ses, omega):
    x = np.array([N**-omega for N in Ns])
    w = 1 / np.array([ses[N] for N in Ns])**2
    y = np.array([means[N] for N in Ns])
    A = np.vstack([np.ones_like(x), x]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    coef = cov @ (A.T @ (w * y))
    return float(coef[0]), float(math.sqrt(cov[0, 0])), float(((y - A @ coef)**2 * w).sum())


def _extrapolate(Ns, per_sample, diagnostics, stress, conf, what):
    means = {N: float(v.mean()) for N, v in per_sample.items()}
    ses = {N: float(v.std(ddof=1) / math.sqrt(len(v))) for N, v in per_sample.items()}
    e0, s0, chi2 = _fit(Ns, means, ses, OMEGA)
    variants = {f"omega={name}": _fit(Ns, means, ses, w)[0] for name, w in stress}
    if len(Ns) >= 4:
        variants[f"omega=2/3 without N={Ns[0]}"] = _fit(Ns[1:], means, ses, OMEGA)[0]
    stat = t_quantile((1 + conf) / 2, min(len(v) for v in per_sample.values()) - 1) * s0
    syst = max(abs(v - e0) for v in variants.values())
    notes = [f"{what}: fit e_N = e_inf + a N^-2/3 over N = {Ns}: chi2 = {chi2:.3g} on "
             f"{len(Ns) - 2} dof; band = {conf:.0%} t-band {stat:.2g} + systematic {syst:.2g} "
             "(variants " + ", ".join(f"{k}: {v:.5f}" for k, v in variants.items()) + ")",
             "statistical and finite-size: not certified"]
    return FSResult(e0, stat + syst, stat, syst, variants, Ns, means, ses, per_sample,
                    diagnostics, chi2, conf, notes)


def _check_sizes(Ns, samples, need):
    if not (isinstance(Ns, (tuple, list)) and len(set(Ns)) >= need and
            all(isinstance(N, int) and not isinstance(N, bool) and N >= 4 for N in Ns)):
        raise Refusal("precondition:sizes",
                      f"Ns = {Ns!r}: need at least {need} distinct integer sizes N >= 4")
    Ns = tuple(sorted(set(Ns)))
    per = samples if isinstance(samples, dict) else {N: samples for N in Ns}
    if not all(isinstance(per.get(N), int) and per[N] >= 2 for N in Ns):
        raise Refusal("precondition:samples", f"samples = {samples!r}: need an int >= 2 per size")
    return Ns, per


def _pick(spec, N, default):
    return spec(N) if callable(spec) else (spec or default)


def sk_ground_energy_finite_size(Ns, samples, seed, *, sweeps=None, betas=None, conf=0.99):
    Ns, per = _check_sizes(Ns, samples, 3)
    est, diag = {}, {}
    for N in Ns:
        J = gaussian_sk_couplings(N, per[N], [seed, N, 0])
        sw = _pick(sweeps, N, max(200, 12 * N))
        bs = _pick(betas, N, geometric_betas(0.4, 4.0, max(8, round(2 * math.sqrt(N)))))
        a = pt_sk(J, bs, sw, [seed, N, 1]).min_energy_per_spin
        b = pt_sk(J, bs, sw, [seed, N, 2]).min_energy_per_spin
        same = np.abs(a - b) <= 1e-9 * np.abs(a)
        if not same.all():
            raise Refusal("undetermined:ground-states",
                          f"N = {N}: the two PT runs disagree on {int((~same).sum())} of "
                          f"{per[N]} samples ({sw} sweeps, {len(bs)} betas); raise sweeps")
        est[N] = np.minimum(a, b)
        diag[N] = f"two independent PT runs agree on all {per[N]} samples ({sw} sweeps)"
    return _extrapolate(Ns, est, diag, (("3/4", 3 / 4),), conf, "ground-state energy")


def sk_energy_finite_size(beta, Ns, samples, seed, *, law="pm_one", sweeps=None, nbetas=None,
                          conf=0.99):
    Ns, per = _check_sizes(Ns, samples, 3)
    if law not in _LAWS:
        raise Refusal("precondition:supported-disorder-law", f"law {law!r}: one of {sorted(_LAWS)}")
    beta = float(beta)
    if not (math.isfinite(beta) and beta > 0.5):
        raise Refusal("precondition:betas", f"beta = {beta!r}: need beta > 0.5")
    est, diag = {}, {}
    for N in Ns:
        J = _LAWS[law](N, per[N], [seed, N, 0])
        sw = _pick(sweeps, N, max(1000, 8 * N))
        R = _pick(nbetas, N, max(8, round(1.25 * math.sqrt(N))))
        res = pt_sk(J, geometric_betas(0.5, beta, R), sw, [seed, N, 1])
        eq = equilibration(res, R - 1)
        if not eq.equilibrated:
            raise Refusal("undetermined:equilibration", f"N = {N}: {eq.text}; raise sweeps")
        est[N] = res.energy_per_spin_trace[sw // 2:, :, R - 1].mean(axis=0)
        diag[N] = eq.text
    return _extrapolate(Ns, est, diag, (("3/4", 3 / 4), ("1", 1.0)), conf,
                        f"energy at beta = {beta:g} ({law})")
