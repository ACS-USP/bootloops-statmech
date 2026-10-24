"""Replica-symmetry ansatze and the n -> 0 rules.

  rs_matrix(n, q)             -> n x n overlap matrix (RS)
  rsb_matrix(ms, qs, n)       -> n x n Parisi hierarchy matrix (integer n)
  sum_offdiag_power(...)      -> closed form of sum_{a != b} q_ab^k
  n_to_zero(expr, n)          -> lim_{n->0} expr / n

Conventions.  Matrices have q_aa = 1 on the diagonal; the sums below run over
ORDERED pairs a != b (each unordered pair twice: sum_{a != b} = 2 sum_{a < b}).
Parisi nesting: ms = [m1 >= ... >= mK], qs = [q0, ..., qK] ordered outermost
to innermost.  With m_0 = n and m_{K+1} = 1, replicas form blocks of size m_j
nested inside blocks of size m_{j-1}; two distinct replicas whose smallest
common block has size m_j (and which sit in different blocks of size m_{j+1})
have overlap q_j.  K = 0 is RS; K = 1 is 1RSB (q0 between blocks of size m,
q1 within a block).

Counting ordered pairs: replica a has m_j - m_{j+1} partners at level j, so

    sum_{a != b} q_ab^k = n * sum_{j=0..K} (m_j - m_{j+1}) q_j^k,

a polynomial in n and the m's: RS gives n(n-1) q^k, 1RSB gives
n(m-1) q1^k + n(n-m) q0^k.  Validated by brute force in the tests (integer n,
divisible block sizes); the continuation to n -> 0 substitutes the polynomial.
The RS/RSB ansatze and the n -> 0 continuation itself are physics-level
prescriptions (Parisi), not proven here: the tests validate the integer-n
counting, not the analytic continuation.
"""
import sympy as sp

from .core import Refusal


def rs_matrix(n, q):
    """n x n RS overlap matrix: 1 on the diagonal, q elsewhere."""
    return sp.Matrix(n, n, lambda a, b: 1 if a == b else q)


def _check_hierarchy(ms, qs):
    if len(qs) != len(ms) + 1:
        raise ValueError(f"need len(qs) == len(ms) + 1, got {len(qs)}, {len(ms)}")


def _check_numeric_ms(ms):
    """Numeric block sizes must be positive and non-increasing (symbolic pass)."""
    nums = [m for m in ms if sp.sympify(m).is_number]
    if any(sp.sympify(m) <= 0 for m in nums):
        raise Refusal("precondition:positive-integer-block-sizes", f"ms = {list(ms)!r}")
    if len(nums) == len(ms) and any(a < b for a, b in zip(ms, ms[1:])):
        raise Refusal("precondition:non-increasing-block-sizes", f"ms = {list(ms)!r}")


def rsb_matrix(ms, qs, n):
    """Explicit Parisi matrix for integer n and integer, nested ms (m_{j+1} | m_j | n)."""
    _check_hierarchy(ms, qs)
    if not (isinstance(n, int) and n >= 1):
        raise Refusal("precondition:positive-integer-n", f"n = {n!r}")
    _check_numeric_ms(ms)
    sizes = [n, *ms, 1]
    for big, small in zip(sizes, sizes[1:]):
        if not (isinstance(small, int) and small >= 1):
            raise Refusal("precondition:positive-integer-block-sizes", f"ms = {list(ms)!r}")
        if big % small:
            raise Refusal("precondition:nested-block-sizes",
                          f"{small} does not divide {big}")

    def entry(a, b):
        if a == b:
            return 1
        # deepest level whose block of size m_j still contains both replicas
        j = max(j for j in range(len(sizes) - 1) if a // sizes[j] == b // sizes[j])
        return qs[j]
    return sp.Matrix(n, n, entry)


def sum_offdiag_power(kind, ms, qs, k, n):
    """Closed form of sum_{a != b} q_ab^k (ordered pairs), polynomial in n, ms.

    kind "rs" takes ms=[] and qs=[q]; kind "rsb" takes the Parisi lists above.
    """
    if kind == "rs":
        if len(ms) != 0 or len(qs) != 1:
            raise ValueError('kind "rs" takes ms=[] and qs=[q]')
    elif kind == "rsb":
        if not ms:
            raise ValueError('kind "rsb" needs at least one m')
        _check_hierarchy(ms, qs)
        _check_numeric_ms(ms)
    else:
        raise ValueError(f'unknown ansatz kind {kind!r} (use "rs" or "rsb")')
    sizes = [n, *ms, 1]
    return sp.expand(n * sum((sizes[j] - sizes[j + 1]) * qs[j]**k
                             for j in range(len(qs))))


def n_to_zero(expr, n):
    """lim_{n->0} expr / n, for expr a polynomial in n with expr(0) = 0.

    The 1/n is part of the rule: the bare a != b sum vanishes at n = 0, its
    per-replica limit does not (RS: -q^k).  Anything else (not divisible by n,
    constant, non-polynomial) is a Refusal rather than an infinity or a
    silently different limit.  The continuation n -> 0 is a physics-level
    prescription, not proven here.
    """
    e = sp.expand(expr)
    if not e.is_polynomial(n) or e.subs(n, 0) != 0:
        raise Refusal("precondition:polynomial-divisible-by-n", f"expr = {expr}")
    return sp.expand(sp.cancel(e / n).subs(n, 0))
