"""ModelSpec: the declarative description of a disordered model."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    """Known `variables`: "ising", "spherical".  Known `disorder`:
    "gaussian", "pm_one", "none".  Both are deliberately not enforced here:
    the set is open to extension and steps refuse on what they cannot handle.

    Frozen, so equality works; `params` is a dict, hence a ModelSpec is not
    hashable.  Do not use it as a dict key or set member.
    """
    family: str
    variables: str
    disorder: str
    constraint: str | None = None
    prior: str | None = None
    params: dict = field(default_factory=dict)
