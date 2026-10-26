"""Exact enumeration (E11): numerics/enumeration.py.

Conventions: +-J SK J_ij = sigma_ij / sqrt(N), sigma_ij = +-1 (i < j), H = -sum_{i<j} J_ij s_i s_j,
h = 0; the spectrum is the integer histogram of K(s) = sum_{i<j} sigma_ij s_i s_j (E = -K / sqrt N)
over all 2^N states.  Ising perceptron: J in {-1,+1}^N, Gaussian patterns xi^mu (rows of
default_rng(seed).standard_normal((P, N))), a pattern is stored if J.xi^mu > 0 (kappa = 0).
Independent routes here: brute force over itertools.product (no Gray code, no blocks, no
matrix products), and the identity sum_K g_K = 2^N (ln Z = N ln 2 at beta = 0).
"""
import itertools
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from replicant.core import Refusal
from replicant.numerics import enumeration as E
from replicant.testing import slow


def _refusal(fn):
    try:
        fn()
    except Refusal as r:
        return r
    raise AssertionError("expected a Refusal")


def _pm_one_J(N, seed):
    rng = np.random.default_rng([seed, N])
    sig = np.triu(rng.choice((-1.0, 1.0), size=(N, N)), 1)
    return (sig + sig.T) / math.sqrt(N)


def _brute_K(J):
    """K(s) = sum_{i<j} sigma_ij s_i s_j for every state, by plain loops over itertools."""
    N = len(J)
    sig = np.rint(J * math.sqrt(N)).astype(int)
    out = []
    for s in itertools.product((1, -1), repeat=N):
        out.append(sum(sig[i, j] * s[i] * s[j] for i in range(N) for j in range(i + 1, N)))
    return out


def test_gray_states_are_a_gray_code():
    for m in range(0, 7):
        S = E.gray_states(m)
        assert S.shape == (2**m, m)
        assert len({tuple(r) for r in S}) == 2**m               # every state once
        if m:
            assert (np.abs(np.diff(S, axis=0)).sum(axis=1) == 2).all()   # one flip per step


def test_pm_one_spectrum_matches_brute_force():
    for N in range(2, 10):
        J = _pm_one_J(N, 7)
        sp = E.enumerate_pm_one_sk(J)
        hist = np.bincount(np.array(_brute_K(J)) + sp.M, minlength=2 * sp.M + 1)
        assert (sp.counts == hist).all(), N
        assert int(sp.counts.sum()) == 2**N
        assert sp.ground_energy_total == -max(_brute_K(J)) / math.sqrt(N)
        brute_lnZ = math.log(sum(math.exp(1.3 * k / math.sqrt(N)) for k in _brute_K(J)))
        assert abs(sp.log_Z(1.3) - brute_lnZ) < 1e-12 * abs(brute_lnZ)
        assert abs(sp.log_Z(0) - N * math.log(2)) < 1e-12


def test_pm_one_spectrum_at_sixteen_spins():
    """Default-tier size (ruling 4: N <= 16 here, up to 26 under @slow)."""
    J = _pm_one_J(16, 3)
    sp = E.enumerate_pm_one_sk(J)
    assert int(sp.counts.sum()) == 2**16
    assert (sp.counts[1::2] == 0).all()     # K is a sum of M terms +-1, so K = M (mod 2)
    assert abs(sp.log_Z(0) - 16 * math.log(2)) < 1e-12
    # the energy is -d lnZ / d beta: the spectrum's mean against a central difference
    d = 1e-5
    fd = -(sp.log_Z(0.7 + d) - sp.log_Z(0.7 - d)) / (2 * d) / 16
    assert abs(sp.mean_energy_per_spin(0.7) - fd) < 1e-8


def test_exact_ground_energy_gaussian_matches_brute_force():
    rng = np.random.default_rng(11)
    for N in (3, 6, 9):
        G = np.triu(rng.standard_normal((N, N)), 1)
        J = (G + G.T) / math.sqrt(N)
        best = min(-sum(J[i, j] * s[i] * s[j] for i in range(N) for j in range(i + 1, N))
                   for s in itertools.product((1, -1), repeat=N))
        assert abs(E.exact_ground_energy(J) - best / N) < 1e-12, N


def test_enumeration_refusals():
    J = _pm_one_J(5, 1)
    bad = J.copy()
    bad[0, 1] = bad[1, 0] = 0.3
    assert _refusal(lambda: E.enumerate_pm_one_sk(bad)).reason == "precondition:pm-one-couplings"
    asym = J.copy()
    asym[0, 1] = -asym[0, 1]
    assert _refusal(lambda: E.enumerate_pm_one_sk(asym)).reason == "precondition:coupling-matrix"
    assert _refusal(lambda: E.enumerate_pm_one_sk(_pm_one_J(27, 0))).reason == \
        "precondition:enumeration-size"
    assert _refusal(lambda: E.exact_ground_energy(np.zeros((2, 3)))).reason == \
        "precondition:coupling-matrix"


def _brute_counts(N, P, seed):
    xi = np.random.default_rng(seed).standard_normal((P, N))
    out = [0] * (P + 1)
    for w in itertools.product((1.0, -1.0), repeat=N):
        f = xi @ np.array(w)
        p = 0
        while p < P and f[p] > 0:
            p += 1
        for k in range(p + 1):
            out[k] += 1
    return out


def test_ising_perceptron_counts_match_brute_force():
    for N, P, seed in ((6, 5, 1), (9, 8, 2), (10, 12, 3)):
        got = E.ising_perceptron_counts(N, P, seed)
        assert list(got) == _brute_counts(N, P, seed), (N, P)
    c = E.ising_perceptron_counts(10, 5, 3)
    assert E.ising_perceptron_log_count("0.5", 10, 3) == math.log(c[5])
    assert E.ising_perceptron_log_count(0.5, 10, 3) == math.log(c[5])
    # nested pattern sets: the first P rows do not depend on how many are drawn
    assert list(E.ising_perceptron_counts(10, 12, 3)[:6]) == list(c)
    # ln 0 is returned as -inf (the caller states how it handles empty solution sets)
    assert E.ising_perceptron_log_count(3, 4, 0) == -math.inf
    assert E.ising_perceptron_capacity(E.ising_perceptron_counts(10, 30, 3)) == \
        max(p for p, n in enumerate(_brute_counts(10, 30, 3)) if n > 0)


def test_ising_perceptron_refusals():
    assert _refusal(lambda: E.ising_perceptron_log_count("0.35", 10, 0)).reason == \
        "precondition:integer-patterns"
    assert _refusal(lambda: E.ising_perceptron_log_count("0.5", 31, 0)).reason == \
        "precondition:enumeration-size"
    assert _refusal(lambda: E.ising_perceptron_log_count("-1", 10, 0)).reason == \
        "precondition:integer-patterns"
    assert _refusal(lambda: E.ising_perceptron_counts(10, 3, "a")).reason == "precondition:seed"


def test_annealed_moments_match_brute_force_over_couplings():
    """E Z^k (k = 1, 2, 3) of +-J SK at N = 4 by brute force over all 2^6 coupling sign
    patterns and all states, against numerics/moments.py's gauge-reduced sums; Gaussian
    ln E Z against N ln 2 + M beta^2 / (2N)."""
    from replicant.numerics import moments as Mo
    N, beta = 4, 0.8
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    S = np.array(list(itertools.product((1.0, -1.0), repeat=N)))
    X = np.stack([S[:, i] * S[:, j] for i, j in pairs], axis=1)
    Zs = np.array([np.exp(beta / math.sqrt(N) * (X @ np.array(g))).sum()
                   for g in itertools.product((1.0, -1.0), repeat=len(pairs))])
    L1, L2, L3 = Mo.log_moments(N, beta)
    lnEZ = math.log(Zs.mean())
    assert abs(float(L1) - lnEZ) < 1e-13
    assert abs(float(L2) - (math.log((Zs**2).mean()) - 2 * lnEZ)) < 1e-13
    assert abs(float(L3) - (math.log((Zs**3).mean()) - 3 * lnEZ)) < 1e-13
    G1 = Mo.log_moments(7, beta, law="gaussian")[0]
    assert abs(float(G1) - (7 * math.log(2) + 21 * beta**2 / 14)) < 1e-14
    # the closures: order 2 is -L2/2; order 3 solves the three cumulant equations
    assert Mo.quenched_gap(N, beta, 2) == float(-L2 / 2)
    k1 = Mo.quenched_gap(N, beta, 3)
    assert k1 < 0 and abs(k1 - float(-L2 / 2)) < abs(float(L2))
    assert _refusal(lambda: Mo.quenched_gap(N, beta, 4)).reason == "precondition:order"
    # +-J asymptote: the order-2 closure approaches -1/4 [-ln(1-b^2) - b^2 - b^4/2] as O(1/N)
    lim = Mo.pm_one_gap_limit(0.7)
    d400, d1600 = (abs(Mo.quenched_gap(n, 0.7, 2) - lim) for n in (400, 1600))
    assert d1600 < 1e-4 and 3 < d400 / d1600 < 5, (d400, d1600)


@slow
def test_pm_one_spectrum_at_twenty_six_spins():
    """The largest size (ruling 4: N <= 26 under @slow): sum g_K = 2^26, the parity rule,
    the spectrum's ground energy against the independent float route exact_ground_energy, and
    the thermal energy against -d lnZ / d beta."""
    J = _pm_one_J(26, 4)
    sp = E.enumerate_pm_one_sk(J)
    assert int(sp.counts.sum()) == 2**26 and (sp.counts[1::2] == 0).all()
    assert abs(sp.ground_energy_total / 26 - E.exact_ground_energy(J)) < 1e-12
    d = 1e-5
    fd = -(sp.log_Z(0.9 + d) - sp.log_Z(0.9 - d)) / (2 * d) / 26
    assert abs(sp.mean_energy_per_spin(0.9) - fd) < 1e-8
    assert abs(sp.log_Z(0) - 26 * math.log(2)) < 1e-12
