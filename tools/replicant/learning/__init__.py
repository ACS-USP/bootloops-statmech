"""High-dimensional learning (E7).

  ridge.py      ridge / ridgeless least squares on isotropic Gaussian design: the limiting excess
                risk from the Marchenko-Pastur Stieltjes transform (HMRT 2022 Cor. 5, Eq. (8)),
                the optimal penalty lam* = sigma^2 gamma / r^2, a numerical argmin; planted-error
                hook _SCALE
  ridge_rs.py   the replica-symmetric saddle point of the linear student with square loss
                (Loureiro et al. 2022, Thm 1 / Eq. (A.38), isotropic), solved without the
                Stieltjes transform
  ridge_sim.py  finite-n simulation by SVD over named seeds (99% Student-t bands), independent of
                both
"""
from .ridge import (argmin_lambda, optimal_lambda, optimal_test_error, ridge_bias_variance,
                    ridge_test_error, ridgeless_bias_variance, ridgeless_test_error)
from .ridge_rs import RidgeRS, ridge_replica_rs
from .ridge_sim import RidgeFit, extrapolate_in_n, ridge_ensemble, ridge_fit, ridge_simulate

__all__ = ["RidgeFit", "RidgeRS", "argmin_lambda", "extrapolate_in_n", "optimal_lambda", "optimal_test_error",
           "ridge_bias_variance", "ridge_ensemble", "ridge_fit", "ridge_replica_rs",
           "ridge_simulate", "ridge_test_error", "ridgeless_bias_variance", "ridgeless_test_error"]
