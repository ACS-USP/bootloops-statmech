"""The `sample` route of the P1 percolation entries: finite random graphs over named seeds, percolated
and labelled by union-find (percolation/sampling.py, which shares no code with the generating-function
route).  Statistical: every value is a mean over the seeds with its stderr in errors, labelled
not-computed (never final, never certified).  Run parameters (besides the law's): N (default 2000),
seeds (default 0 .. 7), s_max (default 3).  The comparison with the certified cavity values is the
tests' (tests/test_percolation_sample.py) and the slow statistical bank checks'.
"""
from .catalog import Out, Stability, fstep
from .percolation.sampling import sample_percolation

SEEDS = tuple(range(8))
SAMPLE_CAVEAT = ("sample route, statistical: finite random graphs (configuration model or G(N, p)) "
                 "over named seeds, percolated and labelled by union-find; means +- stderr (99% "
                 "Student-t half-widths in the notes); never certified, never final; the giant's "
                 "estimate is the largest cluster, biased at O(1/N) and meaningless inside the "
                 "critical window |t G1'(1) - 1| <~ N^(-1/3)")
STAT = Stability("not-computed", "sample", "percolation/sampling.py (union-find on sampled graphs)",
                 "finite-N statistical estimate over named seeds: NOT final")
KEYS = ("P_inf", "mean_finite", "mean_degree")


def sample_step(name, law_of, cite):
    def go(p):
        law, dil = law_of(p)
        s_max = p.get("s_max", 3)
        r = sample_percolation(law, dil, p.get("N", 2000), p.get("seeds", SEEDS), s_max)
        keys = [k for k in r.mean]
        ren = {"P_inf": "P_inf_sampled", "mean_finite": "mean_finite_cluster_size_sampled",
               "mean_degree": "mean_degree_sampled"}
        ren.update({f"P_s{s}": f"P_s{s}_sampled" for s in range(1, s_max + 1)})
        vals = {ren[k]: r.mean[k] for k in keys}
        errs = {ren[k]: r.stderr[k] for k in keys}
        hw = ", ".join(f"{ren[k]} +- {r.halfwidth99[k]:.3g}" for k in keys)
        meta = {"N": r.N, "n_seeds": len(r.seeds), "sampler_checks": r.checks}
        return Out({**vals, **meta}, f"{len(r.seeds)} sampled graphs (seeds {r.seeds[0]}..{r.seeds[-1]}), "
                   f"N = {r.N}: largest-cluster fraction {r.mean['P_inf']:.4g}", errors=errs,
                   statistical=tuple(vals),
                   notes=(f"statistical: means over named seeds, stderr in errors; 99% Student-t "
                          f"half-widths: {hw}",))
    return fstep(name, cite, go)


def sample_finality(s_max_max=60):
    fin = {k: STAT for k in ("P_inf_sampled", "mean_finite_cluster_size_sampled", "mean_degree_sampled")}
    fin.update({f"P_s{s}_sampled": STAT for s in range(1, s_max_max + 1)})
    return fin


SAMPLE_META = ("N", "n_seeds", "sampler_checks")
SAMPLE_PARAMS = ("N", "seeds", "s_max")

__all__ = ["SAMPLE_CAVEAT", "SAMPLE_META", "SAMPLE_PARAMS", "sample_finality", "sample_step"]
