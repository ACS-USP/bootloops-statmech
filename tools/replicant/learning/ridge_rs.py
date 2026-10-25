"""The replica-symmetric saddle point of the linear student with square loss (ridge), isotropic.

Source: Loureiro, Gerbelot, Cui, Goldt, Krzakala, Mezard & Zdeborova, "Learning curves of generic
features maps for realistic datasets with a teacher-student model", J. Stat. Mech. (2022) 114001,
arXiv:2102.08127 (Loureiro et al.): the RS free energy of the Gibbs measure
exp(-beta [sum_mu g(y_mu, w.v_mu / sqrt d) + (lam_L / 2)|w|^2]) by the replica method (App. A,
heuristic) and its rigorous counterpart Theorem 1 (Gordon's comparison inequality, Gaussian
covariates); for ridge (square loss) App. A.5, Eqs. (A.37)-(A.38).  Specialised here to
u = v (teacher and student on the same features), Psi = Phi = Omega = I_d, p = d (their gamma = 1),
alpha = n / d = 1 / gamma (our gamma = p / n), rho = |theta_0|^2 / p = r^2:
    V_hat = alpha / (1 + V),   m_hat = alpha / (1 + V),
    q_hat = alpha (rho + sigma^2 + q - 2 m) / (1 + V)^2,
    V = 1 / (lam_L + V_hat),   m = m_hat rho / (lam_L + V_hat),
    q = (q_hat + m_hat^2 rho) / (lam_L + V_hat)^2.
The sigma^2 in q_hat is the label noise: Thm 1 Eq. (2.8) with f0(x) = x + sigma xi and the
square-loss proximal operator prox(x) = (x + V y) / (1 + V) gives
q_hat = alpha E (x - y)^2 / (1 + V)^2 = alpha (q - 2 m + rho + sigma^2) / (1 + V)^2 (Eq. (A.38) is
printed for the noiseless f0(x) = x; this specialisation is replicant's, stated here).
Penalty map: Loureiro's objective (1/2) sum (y - w.v / sqrt d)^2 + (lam_L / 2)|w|^2 with
b = w / sqrt d is (1/2)|y - X b|^2 + (lam_L d / 2)|b|^2, HMRT's (n lam / 2)|b|^2: lam_L = lam / gamma.
Overlaps (Eq. (2.9)): q = |b|^2, m = beta.b, rho = |beta|^2, so the excess risk is
|b - beta|^2 = rho + q - 2 m (the identity Loureiro print as E_gen in Eq. (A.37)).
Solution here: V from its scalar equation V (lam_L + alpha / (1 + V)) = 1 (bracketed root on
(0, 1 / lam_L], unique: the left side increases in V), then m explicitly and q from the equation
linear in q.  No Stieltjes transform and no closed form of ridge.py is used.
"""
from dataclasses import dataclass

import mpmath as mp

from ..core import Refusal
from .ridge import _num

_DPS = 30


@dataclass(frozen=True)
class RidgeRS:
    """RS overlaps (V, q, m) and conjugates of the isotropic ridge saddle point; risk = rho + q - 2m."""
    gamma: object
    lam: object
    V: object
    V_hat: object
    q: object
    q_hat: object
    m: object
    m_hat: object
    risk: object


def ridge_replica_rs(gamma, lam, sigma2=0, signal=1):
    """RS saddle point of ridge (lam > 0, HMRT's penalty convention) at gamma = p/n."""
    g = _num(gamma, "gamma", positive=True)
    lm = _num(lam, "lambda", positive=True)
    s2, rho = _num(sigma2, "sigma2"), _num(signal, "signal")
    with mp.workdps(_DPS):
        a, lL = 1 / g, lm / g
        f = lambda V: V * (lL + a / (1 + V)) - 1
        hi = 1 / lL                        # f(1 / lL) = a / (lL (1 + V)) > 0, f(0) = -1
        V = mp.findroot(f, (mp.mpf(0), hi), solver="anderson")
        if not (0 < V <= hi) or abs(f(V)) > mp.mpf(10) ** (8 - _DPS):
            raise Refusal("no-solution:bracket", f"V root of the RS saddle not found (gamma={g}, "
                          f"lam={lm}): residual {mp.nstr(f(V), 5)}")
        Vh = mh = a / (1 + V)
        m = mh * rho / (lL + Vh)
        k = a / ((1 + V) ** 2 * (lL + Vh) ** 2)          # coefficient of q in q_hat / (lL + Vh)^2
        q = (a * (rho + s2 - 2 * m) / (1 + V) ** 2 + mh * mh * rho) / (lL + Vh) ** 2 / (1 - k)
        qh = a * (rho + s2 + q - 2 * m) / (1 + V) ** 2
        return RidgeRS(g, lm, +V, +Vh, +q, +qh, +m, +mh, +(rho + q - 2 * m))


__all__ = ["RidgeRS", "ridge_replica_rs"]
