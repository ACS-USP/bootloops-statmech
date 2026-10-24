"""Result types shared by the level-(b) families (SK, p-spin, REM, ...) and by
the cavity route (cavity/), which must not import the replica families."""
from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class HopfieldSolution:
    m: object
    q: object
    r: object
    branch: str
    f: object = None
    stable: bool = None            # mode_stable and replicon > 0
    residual: object = None        # max |equation| over (m, q, r)
    grad: tuple = ()               # analytic (df/dm, df/dq, df/dr) at the point
    mode_stable: bool = None       # m-mode (retrieval, spin glass) or q-mode (paramagnet)
    mode_eig: object = None
    replicon: object = None        # lambda_AT = 1 - alpha beta^2 E sech^4 / (1 - C)^2
    note: str = ""
    caveat: str = CAVEAT


@dataclass(frozen=True)
class Capacity:
    alpha_c: object
    stable: bool          # the ansatz's alpha_c is self-consistent (see note)
    ansatz: str
    variables: str = "spherical"
    kappa: object = 0
    replicon: object = None
    note: str = ""
    point: object = field(default=None, compare=False)   # RSPoint at the threshold, if any
    caveat: str = CAVEAT
