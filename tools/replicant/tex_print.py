"""Typesetting of sympy objects for the LaTeX writer: a faithful printer and line breaking.

  latex(expr)           sympy.latex with one change: a Sum over several indices prints as
                        nested sums, and the replica pair sum sum_{b=a+1}^{n} sum_{a=1}^{n}
                        as \\sum_{a < b \\leq n} (the same sum; no \\substack)
  width(tex)            visible width of a math string, in characters (a measured proxy:
                        fractions count their wider part, sub/superscripts 0.6, spacing 0)
  lines(lhs, rhs)       [line, ...]: the rhs split at top-level + / - terms (and inside a
                        coefficient times a wide sum, with \\Bigl( \\Bigr)) so that no line is
                        wider than WIDTH; [lhs = rhs] when it fits; a row that is still
                        wider (one indivisible term) is scaled to the line width as a box

Only the typesetting changes: the terms are printed one by one with the same printer,
in sympy's own order and with its sign convention, so the displayed object is the one
the code holds (tests/test_latex.py checks the breaks on a synthetic expression and that
compiled documents have no overfull display lines).
"""
import re

import sympy as sp
from sympy.printing.latex import LatexPrinter

WIDTH = 52          # "characters" per display line: calibrated on compiled documents (11pt,
                    # 2 cm margins: ~8.5 pt per counted character, text block ~500 pt)
_ZERO = {"left", "right", "Bigl", "Bigr", "bigl", "bigr", "operatorname", "mathbb",
         "mathrm", "hat", ",", ";", "quad", "qquad", "displaystyle"}


class _Printer(LatexPrinter):
    def _print_Sum(self, expr):
        lims, out, k = list(expr.limits), [], len(expr.limits) - 1
        while k >= 0:                       # limits are innermost first: print outermost first
            v, lo, hi = lims[k]
            if k >= 1:
                w, lo2, hi2 = lims[k - 1]
                if lo == 1 and lo2 == v + 1 and hi2 == hi:
                    out.append(rf"\sum_{{{self._print(v)} < {self._print(w)} \leq "
                               rf"{self._print(hi)}}}")
                    k -= 2
                    continue
            out.append(rf"\sum_{{{self._print(v)}={self._print(lo)}}}^{{{self._print(hi)}}}")
            k -= 1
        body = self._print(expr.function)
        if expr.function.is_Add:
            body = rf"\left({body}\right)"
        return " ".join(out) + " " + body


_P = _Printer()


def latex(expr) -> str:
    return _P.doprint(expr)


def _group(s, i):
    """(content, end) of the brace group or single token starting at s[i]."""
    if s[i] == "{":
        depth, j = 1, i + 1
        while depth:
            depth += {"{": 1, "}": -1}.get(s[j], 0)
            j += 1
        return s[i + 1:j - 1], j
    if s[i] == "\\":
        m = re.match(r"\\[a-zA-Z]+", s[i:])
        return m.group(0), i + len(m.group(0))
    return s[i], i + 1


def width(s: str) -> float:
    w, i = 0.0, 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            m = re.match(r"\\([a-zA-Z]+|.)", s[i:])
            name, i = m.group(1), i + len(m.group(0))
            if name == "frac":
                a, i = _group(s, i)
                b, i = _group(s, i)
                w += max(width(a), width(b)) + 0.5
            elif name == "\\":
                w = 0.0                     # a line break inside the string
            elif name not in _ZERO:
                w += 1
        elif c in "^_":
            g, i = _group(s, i + 1)
            w += 0.6 * width(g)
        elif c == "{":
            g, i = _group(s, i)
            w += width(g)
        else:
            w += 0 if c in " }&" else 1
            i += 1
    return w


def _signed(t):
    return ("-", -t) if t.could_extract_minus_sign() else ("+", t)


def _pieces(expr, budget):
    """Signed pieces of a sum, each narrower than budget where the structure allows."""
    out = []
    for sign, t in map(_signed, _P._as_ordered_terms(expr, order=None) if expr.is_Add
                       else [expr]):
        tex = latex(t)
        adds = [f for f in sp.Mul.make_args(t) if f.is_Add] if t.is_Mul else []
        if width(tex) <= budget or not adds:
            out.append((sign, tex))
            continue
        inner = max(adds, key=lambda f: width(latex(f)))
        rest = t / inner
        sub = _pieces(inner, budget - 4)
        head = "" if rest == 1 else latex(rest) + r" "
        for k, (s2, tx) in enumerate(sub):
            lead = (head + r"\Bigl(" + ("- " if s2 == "-" else "")) if k == 0 else f"{s2} "
            tail = r" \Bigr)" if k == len(sub) - 1 else ""
            out.append((sign if k == 0 else "", lead + tx + tail))
    return out


def lines(lhs, rhs, width_=WIDTH):
    """Display lines of `lhs = rhs` (lhs None: the expression alone).  When the lhs is the
    wide side (e.g. a stationarity equation `... = 0`), the lhs is broken and `= rhs`
    ends the last line."""
    if lhs is not None and width(latex(lhs)) > width_ and \
            width(latex(lhs)) > width(latex(rhs)):
        rows = lines(None, lhs, width_)
        rows[-1] += " = " + latex(rhs)
        return rows
    left = "" if lhs is None else latex(lhs) + " = "
    whole = left + latex(rhs)
    if width(whole) <= width_:
        return [whole]
    rows, cur, w = [], left, width(left)
    for k, (sign, tex) in enumerate(_pieces(rhs, width_ - 6)):
        if tex.startswith("- "):              # normalise: never "+ - x"
            sign, tex = {"-": "+", "+": "-", "": "-"}[sign], tex[2:]
        piece = ((("- " if sign == "-" else "") if k == 0 else (f"{sign} " if sign else ""))
                 + tex)
        wp = width(piece)
        if cur != left and w + wp > width_:
            rows.append(cur.rstrip())
            cur, w = r"\quad " + piece + " ", wp
        else:
            cur, w = cur + piece + " ", w + wp
    rows.append(cur.rstrip())
    return [_fit(r, left if k == 0 else "", width_) for k, r in enumerate(rows)]


def _fit(row, left, width_):
    """A row still wider than the line (an indivisible term, e.g. exp of a long sum) is
    scaled to the line width as one box (the lhs and its "=" stay outside the box)."""
    if width(row) <= width_:
        return row
    body = row[len(left):] if left and row.startswith(left) else row
    head = left if left and row.startswith(left) else ""
    frac = "0.85" if head else "0.9"         # shrink only: never scale a narrow row up
    return (head + r"\resizebox{\ifdim\width>" + frac + r"\linewidth " + frac
            + r"\linewidth\else\width\fi}{!}{$\displaystyle " + body + "$}")
