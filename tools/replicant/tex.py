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
quadrature at 16 digits on |z| <= 14 (an independent rule, not solve.gauss_expect), and
compared with impl(*point) -- a call into the family code -- to 1e-12 relative to
max(1, |impl|).  `points`, or each point, may be a callable (computed when the check
runs: e.g. the overlap of a solved saddle); `slow` marks checks that need a solve per
point (the default test tier checks the first point of every fast check; --full checks
every point of every check).  mpmath's global precision is restored.
"""
import re
from dataclasses import dataclass
from typing import Callable

import mpmath as mp
import sympy as sp

KINDS = ("equation", "align", "text")
TOL = mp.mpf("1e-12")
_SPECIAL = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "_": r"\_", "%": r"\%",
            "&": r"\&", "#": r"\#", "$": r"\$", "^": r"\^{}", "~": r"\~{}",
            "<": r"\textless{}", ">": r"\textgreater{}", "|": r"\textbar{}"}


def escape(s) -> str:
    """Text-mode escape of the TeX specials _ % & # $ ^ ~ { } and backslash, and of < > |
    (which the OT1 encoding of a pdflatex run would print as other glyphs)."""
    return "".join(_SPECIAL.get(c, c) for c in str(s))


def prose(s) -> str:
    """escape() of prose that may quote code notation: Python's ** becomes ^ and a * between
    two operands a space (beta**2*q -> beta^2 q), so trace notes read as text."""
    s = re.sub(r"\*\*", "^", str(s))
    return escape(re.sub(r"(?<=[\w)\]])\*(?=[\w(])", " ", s))


@dataclass(frozen=True, eq=False)
class Check:
    args: tuple
    impl: Callable
    points: object                 # tuple of points (a point may be a callable), or a callable
    part: str = "whole"            # "whole" | "rhs" | "lhs" | "residual" (lhs - rhs)
    slow: bool = False
    # a cited block (no sympy expr): a hand-coded evaluation of the displayed formula,
    # display(*point), compared with impl(*point); tol overrides TOL (e.g. nested quadrature)
    display: Callable | None = None
    tol: object = None

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
    """tex_print.latex of an Expr; `lhs = rhs` of an Eq; align rows `lhs &= rhs` of a list.
    A display wider than tex_print.WIDTH is broken at top-level + / - terms (a multline
    for an equation, `\\\\ &\\quad ...` rows in an align): typesetting only."""
    from .tex_print import lines
    if isinstance(obj, (list, tuple)):
        rows = []
        for e in obj:
            ls = lines(e.lhs, e.rhs)
            rows.append(" \\\\\n".join([ls[0].replace(" = ", " &= ", 1)]
                                         + [f"&{x}" for x in ls[1:]]))
        return " \\\\\n".join(rows)
    if isinstance(obj, sp.Equality):
        return " \\\\\n".join(lines(obj.lhs, obj.rhs))
    return " \\\\\n".join(lines(None, obj))


def sym(title, obj, check=None, kind=None, source=None):
    """A block rendered from a sympy object; `source` optionally cites the formula."""
    kind = kind or ("align" if isinstance(obj, (list, tuple)) else "equation")
    return TexBlock(kind, title, render(obj), obj, source=source, check=check)


def cited(title, body, source, kind="equation", check=None):
    """A LaTeX string mirroring the code, with its paper; `check` (with display=) is a
    hand-coded evaluation of the displayed formula against the code."""
    if not (source and source.strip()):
        raise ValueError(f"cited block {title!r} needs a source")
    if check is not None and check.display is None:
        raise ValueError(f"cited block {title!r}: a check needs a display function")
    return TexBlock(kind, title, body, source=source, check=check)


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
            c = 1 / mp.sqrt(2 * mp.pi)       # |z| > 14 weighs < 1e-43: dropped
            vals.append(mp.quad(lambda z, fn=fn: fn(*pt, z) * c * mp.exp(-z * z / 2),
                                [-14, -5, 0, 5, 14]))
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
    c = b.check
    if c is None or (b.expr is None and c.display is None):
        raise ValueError(f"block {b.title!r} has no expr / check")
    e = None if b.expr is None else _part(b.expr, c.part)
    tol = mp.mpf(c.tol) if c.tol is not None else tol
    bad = []
    for p in c.resolved_points(max_points):
        want = mp.mpf(c.impl(*p))          # the family code, at its own precision
        with mp.workdps(30):
            got = evaluate(e, c.args, p) if e is not None else mp.mpf(c.display(*p))
            if not abs(got - want) <= tol * max(1, abs(want)):
                bad.append((p, mp.nstr(got, 17), mp.nstr(want, 17)))
    return bad


__all__ = ["Check", "TexBlock", "cited", "escape", "evaluate", "invented", "render", "sym",
           "text", "verify_block"]
