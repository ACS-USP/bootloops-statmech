import os
import sys

import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.core import Refusal
from replicant.ansatz import n_to_zero, rs_matrix, rsb_matrix, sum_offdiag_power

n = sp.Symbol("n")
m, m1, m2 = sp.symbols("m m1 m2")
q, q0, q1, q2 = sp.symbols("q q0 q1 q2")


def _brute(M, k, ordered=True):
    """Explicit double sum of M[a,b]**k over a != b (or a < b)."""
    N = M.shape[0]
    return sum(M[a, b]**k for a in range(N) for b in range(N)
               if (a != b if ordered else a < b))


def _closed(kind, ms, qs, k, n_val, ms_val=()):
    expr = sum_offdiag_power(kind, ms, qs, k, n)
    return sp.expand(expr.subs(dict(zip(ms, ms_val))).subs(n, n_val))


def test_rs_offdiag_sum_matches_bruteforce():
    for n_ in range(2, 7):
        M = rs_matrix(n_, q)
        for k in range(1, 5):
            assert sp.expand(_brute(M, k) - _closed("rs", [], [q], k, n_)) == 0


def test_1rsb_offdiag_sum_matches_bruteforce():
    cases = [(nn, mm) for nn in range(2, 9) for mm in range(1, nn + 1)
             if nn % mm == 0]
    assert len(cases) >= 10
    for n_, m_ in cases:
        M = rsb_matrix([m_], [q0, q1], n_)
        for k in range(1, 5):
            got = _closed("rsb", [m], [q0, q1], k, n_, [m_])
            assert sp.expand(_brute(M, k) - got) == 0, (n_, m_, k)


def test_2level_rsb_offdiag_sum_matches_bruteforce():
    cases = [(8, 4, 2), (8, 2, 1), (8, 8, 2), (6, 3, 1), (6, 6, 3), (4, 4, 2),
             (12, 6, 3), (12, 4, 2), (8, 4, 4)]
    for n_, a, b in cases:
        M = rsb_matrix([a, b], [q0, q1, q2], n_)
        for k in range(1, 4):
            got = _closed("rsb", [m1, m2], [q0, q1, q2], k, n_, [a, b])
            assert sp.expand(_brute(M, k) - got) == 0, (n_, a, b, k)


def test_matrices_symmetric_unit_diagonal_and_block_structure():
    for M in (rs_matrix(5, q), rsb_matrix([2], [q0, q1], 6),
              rsb_matrix([4, 2], [q0, q1, q2], 8)):
        assert M == M.T
        assert all(M[a, a] == 1 for a in range(M.shape[0]))
        assert all(M[a, b] != 1 for a in range(M.shape[0])
                   for b in range(M.shape[0]) if a != b)
    M = rsb_matrix([2], [q0, q1], 6)       # blocks {0,1} {2,3} {4,5}
    assert M[0, 1] == q1 and M[2, 3] == q1 and M[0, 2] == q0 and M[1, 5] == q0
    M = rsb_matrix([4, 2], [q0, q1, q2], 8)  # nested: pairs in quads
    assert M[0, 1] == q2 and M[0, 2] == q1 and M[0, 4] == q0 and M[5, 7] == q1


def test_n_to_zero_rs():
    for k in range(1, 6):
        s = sum_offdiag_power("rs", [], [q], k, n)
        assert sp.simplify(n_to_zero(s, n) + q**k) == 0


def test_n_to_zero_1rsb_k2():
    # The closed form is a polynomial in (n, m); check it against brute force
    # at integer points, then check its n -> 0 value against (m-1)q1^2 - m q0^2.
    s = sum_offdiag_power("rsb", [m], [q0, q1], 2, n)
    assert sp.Poly(sp.expand(s), n, m).total_degree() <= 2
    pts = [(2, 1), (2, 2), (4, 2), (6, 3), (8, 4), (6, 2), (6, 1)]
    assert len(pts) >= 5
    for n_, m_ in pts:
        brute = _brute(rsb_matrix([m_], [q0, q1], n_), 2)
        assert sp.expand(s.subs({n: n_, m: m_}) - brute) == 0
    assert sp.expand(n_to_zero(s, n) - ((m - 1)*q1**2 - m*q0**2)) == 0


def test_n_to_zero_2level():
    s = sum_offdiag_power("rsb", [m1, m2], [q0, q1, q2], 3, n)
    want = (m2 - 1)*q2**3 + (m1 - m2)*q1**3 - m1*q0**3
    assert sp.expand(n_to_zero(s, n) - want) == 0


def test_ordered_vs_unordered_pair_sum_factor_two():
    # sum_{a != b} counts each pair twice: it is 2 * sum_{a < b}.
    for n_ in (3, 4, 6):
        M = rs_matrix(n_, q)
        for k in (1, 2, 3):
            ordered, unordered = _brute(M, k), _brute(M, k, ordered=False)
            assert sp.expand(ordered - 2*unordered) == 0
            assert sp.expand(unordered - n_*(n_ - 1)*q**k/2) == 0
    M = rsb_matrix([2], [q0, q1], 4)
    assert sp.expand(_brute(M, 2) - 2*_brute(M, 2, ordered=False)) == 0


def test_n_to_zero_of_unordered_sum_is_half_of_ordered():
    # lim (1/n) sum_{a<b} q^k = -q^k/2, half of the a != b value -q^k.
    k = 3
    ordered = sum_offdiag_power("rs", [], [q], k, n)
    # a<b polynomial in n by interpolating the explicit a<b loop over the
    # explicit matrix (degree 2 in n, so 4 points are ample), independent of
    # the closed form.
    unordered = sp.interpolate([(nn, _brute(rs_matrix(nn, q), k, ordered=False))
                                for nn in range(1, 5)], n)
    assert sp.expand(n_to_zero(unordered, n) + q**k/2) == 0
    assert sp.expand(2*n_to_zero(unordered, n) - n_to_zero(ordered, n)) == 0
    # the 1/n scale matters: n_to_zero(ordered) is not the bare sum at n=0
    assert ordered.subs(n, 0) == 0 and n_to_zero(ordered, n) == -q**k


def _reason(fn):
    try:
        fn()
    except Refusal as r:
        return r.reason
    raise AssertionError("expected Refusal")


def test_n_to_zero_refuses_bad_input():
    want = "precondition:polynomial-divisible-by-n"
    assert _reason(lambda: n_to_zero(q + n*q, n)) == want      # not divisible by n
    assert _reason(lambda: n_to_zero(sp.Integer(1), n)) == want  # constant
    assert _reason(lambda: n_to_zero(sp.exp(n) - 1, n)) == want  # non-polynomial


def test_rsb_matrix_refuses_bad_block_sizes():
    pos = "precondition:positive-integer-block-sizes"
    assert _reason(lambda: rsb_matrix([0], [q0, q1], 4)) == pos
    assert _reason(lambda: rsb_matrix([-2], [q0, q1], 4)) == pos
    assert _reason(lambda: rsb_matrix([sp.Rational(1, 2)], [q0, q1], 4)) == pos
    assert _reason(lambda: rsb_matrix([m], [q0, q1], 4)) == pos       # symbolic m
    assert _reason(lambda: rsb_matrix([3], [q0, q1], 4)) == \
        "precondition:nested-block-sizes"
    assert _reason(lambda: rsb_matrix([2, 4], [q0, q1, q2], 8)) == \
        "precondition:non-increasing-block-sizes"
    for bad_n in (0, -3, n, 2.0):
        assert _reason(lambda: rsb_matrix([1], [q0, q1], bad_n)) == \
            "precondition:positive-integer-n"


def test_sum_offdiag_power_validates_numeric_ms_only():
    assert _reason(lambda: sum_offdiag_power("rsb", [0], [q0, q1], 2, n)) == \
        "precondition:positive-integer-block-sizes"
    assert _reason(lambda: sum_offdiag_power("rsb", [2, 4], [q0, q1, q2], 2, n)) == \
        "precondition:non-increasing-block-sizes"
    sum_offdiag_power("rsb", [m1, m2], [q0, q1, q2], 2, n)   # symbolic ok
    sum_offdiag_power("rsb", [4, m2], [q0, q1, q2], 2, n)    # mixed ok
