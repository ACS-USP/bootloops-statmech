"""Banked coefficients of the 2D Ising high-temperature susceptibility (the E8 stretch check).

series/data/a002906.txt: the first 420 terms of OEIS A002906 ("High temperature series for spin-1/2
Ising magnetic susceptibility on 2D square lattice"; chi = (m^2/kT) sum_n a(n) v^n, v = tanh(J/kT)),
copied from the b-file https://oeis.org/A002906/b002906.txt (fetched 2026-10-25; the entry says it was
obtained from the Guttmann-Jensen series, terms up to n = 116 from F. Hucht).  The entry cites Nickel,
J. Phys. A 32, 3889 (1999), for the singularity structure (the natural boundary |v| = 1 of chi; not
fetched).  These numbers are inputs to a "no operator within bounds" check only, never a proof.
"""
import os

from ..core import Refusal

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "a002906.txt")


def susceptibility_series(n):
    """[a(0), ..., a(n-1)] (ints) of OEIS A002906; n <= 420."""
    rows = [ln.split() for ln in open(_PATH) if ln.strip() and not ln.startswith("#")]
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= len(rows):
        raise Refusal("precondition:order", f"n={n!r}: 1 .. {len(rows)} banked terms")
    assert all(int(r[0]) == i for i, r in enumerate(rows[:n]))
    return [int(r[1]) for r in rows[:n]]


__all__ = ["susceptibility_series"]
