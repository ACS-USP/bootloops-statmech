"""Result types shared by the level-(b) families (SK, p-spin, REM, ...)."""
from dataclasses import dataclass

CAVEAT = ("physics-level: replica n->0 continuation and RS/1RSB ansatz "
          "(Parisi), not proven; numbers carry quadrature error only")


@dataclass(frozen=True)
class RSSolution:
    q: object
    f: object
    e: object
    stable: bool
    replicon: object = None
    caveat: str = CAVEAT


@dataclass(frozen=True)
class RSBSolution:
    """1RSB: q0 between blocks, q1 inside a block of size m; f, e per spin."""
    q0: object
    q1: object
    m: object
    f: object
    e: object
    caveat: str = CAVEAT
