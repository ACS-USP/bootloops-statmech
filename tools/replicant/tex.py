"""TeX blocks: the unit of a LaTeX derivation, and the guard that keeps them honest.

  TexBlock(kind, title, body, expr=None, source=None, check=None)
      kind "equation" | "align" (math bodies, not escaped) | "text" (escaped on output)
      expr    the sympy object the body was rendered from (body == render(expr)):
              an Expr, an Eq, or a list of Eqs (an align)
      source  the citation of a hand-written equation string (a body with no expr)
      check   Check: how to evaluate `expr` against the family code
  sym(title, obj, check=None, source=None)   rendered from a sympy object by sympy.latex
  cited(title, body, source)       a LaTeX string that mirrors the code, with its paper
  text(title, body, source=None)   prose (escaped), optionally cited
  invented(block) -> bool          a math block with neither expr nor source
  verify_block(block)              failures of block.check (empty list: agrees)

Consistency guard.  Check(args, impl, points, part, slow): at each point (values for
the sympy symbols `args`), the displayed expression (or its rhs / lhs / lhs - rhs when
expr is an Eq) is evaluated from the sympy object, GaussE expectations by tanh-sinh
quadrature at 16 digits on |z| <= 30 (an independent rule, not solve.gauss_expect), and
compared with impl(*point) -- a call into the family code -- to 1e-12 relative to
max(1, |impl|).  `points`, or each point, may be a callable (computed when the check
runs: e.g. the overlap of a solved saddle); `slow` marks checks that need a solve per
point (the default test tier checks the first point of every fast check; --full checks
every point of every check).  mpmath's global precision is restored.
"""
from dataclasses import dataclass
from typing import Callable

import mpmath as mp
import sympy as sp

KINDS = ("equation", "align", "text")
TOL = mp.mpf("1e-12")
_SPECIAL = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "_": r"\_", "%": r"\%",
            "&": r"\&", "#": r"\#", "$": r"\$", "^": r"\^{}", "~": r"\~{}"}


def escape(s) -> str:
    """Text-mode escape of the TeX specials _ % & # $ ^ ~ { } and backslash."""
    return "".join(_SPECIAL.get(c, c) for c in str(s))


@dataclass(frozen=True, eq=False)
class Check:
    args: tuple
    impl: Callable
    points: object                 # tuple of points (a point may be a callable), or a callable
    part: str = "whole"            # "whole" | "rhs" | "lhs" | "residual" (lhs - rhs)
    slow: bool = False

    def resolved_points(self, limit=None):
        pts = (self.points() if callable(self.points) else self.points)[:limit]
        return tuple(tuple(p() if callable(p) else p) for p in pts)


@dataclass(frozen=True, eq=False)
class TexBlock:
    kind: str
    title: str
    body: str
    expr: object = None
    source: str | None = None
    check: Check | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"TexBlock kind {self.kind!r} not in {KINDS}")


def render(obj) -> str:
    """sympy.latex of an Expr; `lhs = rhs` of an Eq; align rows `lhs &= rhs` of a list."""
    if isinstance(obj, (list, tuple)):
        return " \\\\\n".join(f"{sp.latex(e.lhs)} &= {sp.latex(e.rhs)}" for e in obj)
    if isinstance(obj, sp.Equality):
        return f"{sp.latex(obj.lhs)} = {sp.latex(obj.rhs)}"
    return sp.latex(obj)


def sym(title, obj, check=None, kind=None, source=None):
    """A block rendered from a sympy object; `source` optionally cites the formula."""
    kind = kind or ("align" if isinstance(obj, (list, tuple)) else "equation")
    return TexBlock(kind, title, render(obj), obj, source=source, check=check)


def cited(title, body, source, kind="equation"):
    if not (source and source.strip()):
        raise ValueError(f"cited block {title!r} needs a source")
    return TexBlock(kind, title, body, source=source)


def text(title, body, source=None):
    return TexBlock("text", title, body, source=source)


def invented(b: TexBlock) -> bool:
    """A displayed equation that is neither rendered from a sympy object the code holds
    (body == render(expr)) nor a cited string."""
    if b.kind == "text":
        return False
    if b.expr is not None:
        return b.body != render(b.expr)
    return not (b.source and b.source.strip())


# ---- numerical evaluation of a displayed expression ------------------------------------
def evaluate(expr, args, point, dps=16):
    """expr (may contain gauss.GaussE atoms) at args = point, as an mpf."""
    from .gauss import GaussE
    with mp.workdps(dps + 5):
        pt = [mp.mpf(str(v)) if not isinstance(v, mp.mpf) else v for v in point]
        atoms = list(sp.sympify(expr).atoms(GaussE))
        vals = []
        for g in atoms:
            fn = sp.lambdify(tuple(args) + (g.z,), g.f, "mpmath")
            c = 1 / mp.sqrt(2 * mp.pi)       # |z| > 30 weighs < 1e-195: dropped
            vals.append(mp.quad(lambda z, fn=fn: fn(*pt, z) * c * mp.exp(-z * z / 2),
                                [-30, -6, 0, 6, 30]))
        dummies = [sp.Dummy() for _ in atoms]
        outer = sp.sympify(expr).xreplace(dict(zip(atoms, dummies)))
        v = sp.lambdify(tuple(args) + tuple(dummies), outer, "mpmath")(*pt, *vals)
        return mp.mpf(mp.re(v)) if mp.im(v) == 0 else mp.mpc(v)


def _part(expr, part):
    if part == "whole":
        return expr
    if not isinstance(expr, sp.Equality):
        raise ValueError(f"check part {part!r} needs an Eq")
    return {"rhs": expr.rhs, "lhs": expr.lhs, "residual": expr.lhs - expr.rhs}[part]


def verify_block(b: TexBlock, *, tol=TOL, max_points=None) -> list:
    """[(point, displayed value, family value)] where they differ by more than tol
    (relative to max(1, |family value|)); [] when every point agrees.  max_points:
    check only the first points (the default test tier uses 1, --full all)."""
    if b.expr is None or b.check is None:
        raise ValueError(f"block {b.title!r} has no expr / check")
    c = b.check
    e = _part(b.expr, c.part)
    bad = []
    for p in c.resolved_points(max_points):
        want = mp.mpf(c.impl(*p))          # the family code, at its own precision
        with mp.workdps(30):
            got = evaluate(e, c.args, p)
            if not abs(got - want) <= tol * max(1, abs(want)):
                bad.append((p, mp.nstr(got, 17), mp.nstr(want, 17)))
    return bad


__all__ = ["Check", "TexBlock", "cited", "escape", "evaluate", "invented", "render", "sym",
           "text", "verify_block"]
