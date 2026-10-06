# Changelog

All notable changes to BootLoops are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) (1.0.x = fixes, 1.x.0 = new tools).

## 1.0.1 (unreleased)

### Fixed
- `run_selftests.py`: a crash in a data-gated package was reported as
  "REFUSED (by design)" and left the run green. Data-gated batteries now
  refuse with exit code 4 (`DATA_GATED_REFUSAL_RC`) and the printed reason
  is kept in `selftest_results.json`; any other nonzero exit of a data-gated
  package is a FAIL. `ffcapital`, `frobenius-boundary` (was exit 3) and
  `galois` updated to the contract; documented in `tools/README.md`
  (Verification classes) and `INSTALL.md`. Reported by @Aspect5
  (bootloops#3). An exit 4 without a printed reason, or whose last line is
  an uncaught exception, is a FAIL as well. `galois` resolves its reference
  data before any gate runs and its bank lookups raise a dedicated
  `ReferenceDataMissing` (a `RuntimeError` subclass, so existing callers
  are unaffected); only that class is the exit-4 refusal. Anything a gate
  raises, including a plain `RuntimeError` or another subclass such as
  `RecursionError` or `NotImplementedError`, is a crash.
- `run_selftests.py`: a timeout in the Julia `Pkg.instantiate()` pre-step
  aborted the whole run and `selftest_results.json` was never written. The
  timeout now fails that one package (status FAIL, cause in `tail`, remedy in
  `fix`) and the run continues; the exit code stays 1. Fix by @hzshen88
  (PR #2), reported in feedback#1.
- `nestor`: the farm's orphan guard compared each worker's parent pid with
  the pool parent's pid, which is wrong under the `forkserver` start method
  (Python 3.14's default on Linux): every worker exited at its first chunk
  and the self-test hung at P5. The guard now records the parent seen from
  inside the worker at initialization and also probes the pool parent's pid,
  so it works under `fork`, `forkserver` and `spawn` with the same behavior
  under `fork`. Reported by @drdavidglew-afk (feedback#2).
