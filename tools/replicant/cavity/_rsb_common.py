"""Shared pieces of the sparse 1RSB cavity (cavity/rsb_sparse.py, rsb_sparse_t0.py):
result type, caveat, size checks, run summary with the drift diagnostic, typed steps."""
import math
from dataclasses import dataclass

import numpy as np

from ..core import Refusal, State, Step, run_pipeline
from ._stats import drift
from .sparse import PDResult

STEP_NAMES = ("remove_site", "cavity_field_law", "add_site", "self_consistency")
MP01 = "Mezard & Parisi, Eur. Phys. J. B 20, 217 (2001), arXiv:cond-mat/0009418"
MP03 = "Mezard & Parisi, J. Stat. Phys. 111, 1 (2003), arXiv:cond-mat/0207121"
RSB_CAVEAT = ("physics-level: 1RSB cavity results assume clustering into many pure states "
              "with a single 1RSB structure (one breakpoint m; full RSB may be the truth on "
              "sparse graphs), tree-like local structure and decorrelated states across "
              "cavity edges; not proven; population outputs are statistical (value +- stderr "
              "from independent runs, finite populations, no extrapolation), not certified")
FACTORIZED_CAVEAT = ("physics-level: factorized 1RSB Ansatz (one symmetric survey for every "
                     "edge), known NOT to be the full 1RSB solution for +-J (MP 2003 Fig. 5); "
                     "deterministic, float precision (~1e-12), not certified")


@dataclass(frozen=True)
class RSBResult(PDResult):
    """value / stderr per observable (stderr over `runs` independent runs).  phase:
    "1rsb" (spread inner populations, stationary), "rs" (inner populations are deltas:
    the trivial solution), "factorized" (T = 0, every survey identical and symmetric),
    "paramagnet" (all fields 0) or "undetermined" (drift detected: not equilibrated).
    caveat: RSB_CAVEAT (not the RS cavity caveat: the states are not a single RS state)."""
    caveat: str = RSB_CAVEAT


def check_sizes(pop, sweeps, runs, burn_in, *, min_pop=50):
    if not (isinstance(pop, int) and pop >= min_pop):
        raise Refusal("precondition:population-size",
                      f"pop must be an int >= {min_pop}, got {pop!r}")
    if not (isinstance(runs, int) and runs >= 2):
        raise Refusal("precondition:runs", f"runs must be an int >= 2 for a stderr, got {runs!r}")
    if not (isinstance(sweeps, int) and sweeps >= 4):
        raise Refusal("precondition:sweeps", f"sweeps must be an int >= 4 (the drift test "
                      f"compares halves of the measured sweeps), got {sweeps!r}")
    if not (isinstance(burn_in, int) and burn_in >= max(2, sweeps // 2)):
        raise Refusal("precondition:burn-in", f"burn_in must be an int >= max(2, sweeps // 2) "
                      f"(the drift sign test uses one burn-in block), got {burn_in!r}")


def summarize(per_run, keys, meas_series, trajs, sweeps, require_equilibrated, phase):
    """Mean +- stderr over runs; drift diagnostic (cavity/_stats.drift) on each series.
    meas_series / trajs: lists (one per diagnosed series) of per-run lists."""
    a = np.array(per_run, float)
    mean, err = a.mean(0), a.std(0, ddof=1) / math.sqrt(len(a))
    if not np.all(np.isfinite(mean)):
        raise Refusal("no-solution:audit", f"non-finite 1RSB observables {mean}")
    block = max(2, sweeps // 2)
    flags, diags = zip(*(drift(q, t, block) for q, t in zip(meas_series, trajs)))
    diag = "; ".join(diags)
    if any(flags):
        note = (f"not equilibrated: {diag}; values carry relaxation bias not in stderr; "
                "increase burn_in (and runs)")
        if require_equilibrated:
            raise Refusal("no-convergence:equilibration", note)
        phase = "undetermined"
    else:
        few = (f"; only {len(a)} runs (< 8): drift tests are weak" if len(a) < 8 else "")
        note = f"no drift detected: {diag}{few}"
    return (dict(zip(keys, map(float, mean))), dict(zip(keys, map(float, err))), phase, note)


def _requires(key):
    return lambda s: [k for k in (key,) if k not in s]


def _put(key, text):
    return lambda s: s.with_(**{key: {"desc": text}})


def _note(key):
    return lambda s: "route=cavity-1rsb; " + s[key]["desc"]


def pipeline(params, descs, solve, cites):
    """Four typed steps (remove site, cavity law, add site, self-consistency) -> state."""
    keys = ("cavity_system", "field_law", "site")
    steps = [Step(name, _requires(need), _put(key, desc), cite, _note(key))
             for name, need, key, desc, cite in zip(STEP_NAMES, ("params",) + keys[:2], keys,
                                                     descs, cites)]
    steps.append(Step(STEP_NAMES[3], _requires("site"),
                      lambda s: s.with_(solution=solve(s["params"])), cites[3],
                      lambda s: f"route=cavity-1rsb; phase {s['solution'][2]}"))
    return run_pipeline(steps, State(params=params))
