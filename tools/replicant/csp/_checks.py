"""Input checks shared by the csp/ modules (named refusals, never a silent fallback)."""
import numpy as np

from ..core import Refusal


def seed(s):
    """A named seed: a non-negative int or numpy integer, not a bool (numpy's default_rng
    refuses negative seeds)."""
    if isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 0:
        raise Refusal("precondition:named-seeds",
                      f"seed {s!r}: seeds must be non-negative integers")
    return int(s)


def seeds(ss):
    """A non-empty, non-string iterable of named seeds -> tuple of ints."""
    if isinstance(ss, (str, bytes)):
        raise Refusal("precondition:named-seeds",
                      f"seeds {ss!r}: seeds must be non-negative integers (a list)")
    try:
        out = tuple(ss)
    except TypeError:
        raise Refusal("precondition:named-seeds",
                      f"seeds {ss!r}: seeds must be non-negative integers (a list)") from None
    if not out:
        raise Refusal("precondition:named-seeds", "give at least one named seed (non-negative integers)")
    return tuple(seed(x) for x in out)


__all__ = ["seed", "seeds"]
