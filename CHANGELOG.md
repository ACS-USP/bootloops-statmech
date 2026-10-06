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
  (bootloops#3).
- `run_selftests.py`: a timeout in the Julia `Pkg.instantiate()` pre-step
  aborted the whole run and `selftest_results.json` was never written. The
  timeout now fails that one package (status FAIL, cause in `tail`, remedy in
  `fix`) and the run continues; the exit code stays 1. Fix by @hzshen88
  (PR #2), reported in feedback#1.
