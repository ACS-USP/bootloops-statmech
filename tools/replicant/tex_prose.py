"""Prose of a LaTeX document: the package's code notation in notes, typeset.

  prose(text) -> LaTeX text

The step notes, trace summaries and labels are written in the package's ASCII code
notation (q_ab, s_i^a, sqrt(N), n -> 0, beta**2*q, ansatz.n_to_zero, families/sk.py).
prose() converts it word by word, conservatively:
  - a word in math notation becomes inline math: x_ab -> $x_{ab}$, s_i^a -> $s_{i}^{a}$,
    sqrt(N) -> $\\sqrt{N}$, -> and n->0 -> \\to, ** -> ^, * -> thin space, != <= >= ->
    \\neq \\leq \\geq; Greek names (alpha, beta, xi, ...) -> \\alpha ...; qhat -> \\hat{q};
    ln, log, exp, tanh, cosh, sech, sinh, lim -> \\ln ...;
  - a code reference (dotted name, file path, or an identifier with a long subscript such
    as rs_ground_energy or N_scale) becomes \\texttt{...} (escaped);
  - a word with notation that is not recognised, or whose brackets do not balance, falls
    back to \\texttt{...}: never raw escaped code;
  - every other word is escaped text (tex.escape).
Math mode is only produced from the characters of the word itself, so a converted note
says what the code said.
"""
import re

from .tex import escape

GREEK = {"alpha", "beta", "gamma", "delta", "epsilon", "kappa", "lambda", "mu", "nu", "xi",
         "pi", "rho", "sigma", "tau", "phi", "chi", "psi", "omega", "Phi", "Psi", "Omega",
         "Delta", "Gamma", "Lambda", "Xi", "Sigma", "Theta"}
FUNCS = {"ln", "log", "exp", "tanh", "cosh", "sinh", "sech", "lim", "det", "erf", "max", "min"}
_SUB_OK = {"RS", "AT", "EA", "SG", "RSB"}
_MATHY = re.compile(r"_|\^|\*\*|->|sqrt\(|!=|<=|>=|(?<=[\w)])\*(?=[\w(])")
_CODE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+(\(.*)?$|^[\w./-]*\w\.py$")


def _balanced(s):
    depth = {"(": 0, "[": 0, "{": 0}
    close = {")": "(", "]": "[", "}": "{"}
    for c in s:
        if c in depth:
            depth[c] += 1
        elif c in close:
            depth[close[c]] -= 1
            if depth[close[c]] < 0:
                return False
    return not any(depth.values())


def _code(word):
    return r"\texttt{" + escape(word) + "}"


def _sqrt(s):
    """sqrt(X) -> \\sqrt{X} with matched parentheses."""
    out, i = "", 0
    while True:
        j = s.find("sqrt(", i)
        if j < 0:
            return out + s[i:]
        k, depth = j + 5, 1
        while k < len(s) and depth:
            depth += {"(": 1, ")": -1}.get(s[k], 0)
            k += 1
        if depth:
            raise ValueError("unbalanced sqrt")
        out += s[i:j] + r"\sqrt{" + _sqrt(s[j + 5:k - 1]) + "}"
        i = k


def math_word(word):
    """A word in math notation as inline math, or None when it is not recognised."""
    subs = re.findall(r"_\{?(\w+)", word)
    if any(len(x) > 2 and x not in _SUB_OK and x not in GREEK and not x.isdigit()
           for x in subs):
        return None                          # rs_free, N_scale: identifiers, not math
    s = word.replace("**", "^")
    s = re.sub(r"\s*->\s*", r" \\to ", s)
    s = s.replace("!=", r" \neq ").replace("<=", r" \leq ").replace(">=", r" \geq ")
    try:
        s = _sqrt(s)
    except ValueError:
        return None
    if re.search(r"(?<![A-Za-z])[A-Za-z]*\d[A-Za-z\d]*_|[A-Za-z\d]_\w+_", s):
        return None                          # e0_rs, q_rs_T0: value keys, not math
    if any(len(a) > 3 and a not in _MATH_NAMES for a in _ALPHA.findall(s)):
        return None                          # an English or code word inside: code font
    s = re.sub(r"(?<=\w)\*(?=\))", "^*", s)                  # y* (a starred value)
    s = re.sub(r"(?<=[\w)\]}])\s*\*\s*(?=[\w(\\])", r"\,", s)   # a*b: a product
    s = re.sub(r"_(\w+)", r"_{\1}", s)
    s = re.sub(r"\^(\w+)", r"^{\1}", s)
    s = re.sub(r"\^\(([^()]*)\)", r"^{\1}", s)

    def name(m):
        w = m.group(0)
        if w == "qhat":
            return r"\hat{q}"
        if w in ("sum", "prod", "int"):
            return "\\" + w + " "
        if w == "inf":
            return r"\infty "
        if w in ("erfc", "Erfc"):
            return r"\operatorname{erfc}"
        if w in GREEK or w in FUNCS:
            return "\\" + w + " "
        return w
    s = re.sub(r"(?<![\\A-Za-z])[A-Za-z]+(?![A-Za-z])", name, s)
    if re.search(r"(?<!\^)\*", s) or any(c in s for c in "#%&$") or not _balanced(s):
        return None
    if len(s) > 60:                         # a long inline formula may break at products
        s = s.replace(r"\,", r"\,\allowbreak ")
    return "$" + s.strip() + "$"


def _word(w):
    if not w:
        return ""
    if w in ("<", ">", "=", "+", "-", "+-"):          # a free-standing relation or sign
        return {"+-": r"$\pm$"}.get(w, f"${w}$") if w in ("<", ">", "+-") else w
    lead = trail = ""
    while w and w[-1] in ",;:." and not re.search(r"\w\.py[,;:.]*$", w):
        trail, w = w[-1] + trail, w[:-1]
    while w and w[0] in "([" and w.count("(") + w.count("[") > w.count(")") + w.count("]"):
        lead, w = lead + w[0], w[1:]
    while w and w[-1] in ")]" and w.count(")") + w.count("]") > w.count("(") + w.count("["):
        trail, w = w[-1] + trail, w[:-1]
    if _CODE.match(w):
        core = _code(w)
    elif _MATHY.search(w):
        core = math_word(w) or _code(w.replace("**", "^"))
    else:
        core = escape(w)
    return escape(lead) + core + escape(trail)


_ALPHA = re.compile(r"[A-Za-z]+")
_MATH_NAMES = GREEK | FUNCS | {"sqrt", "qhat", "sum", "prod", "int", "inf", "erfc", "Erfc",
                               "single"}


def _math_like(word):
    """A space-free piece of a formula (no English word: alphabetic runs are short or
    known names)."""
    return all(len(a) <= 3 or a in _MATH_NAMES for a in _ALPHA.findall(word))


def _open(word):
    return word.count("(") + word.count("[") - word.count(")") - word.count("]")


def prose(text) -> str:
    """Code notation in `text` typeset (module docstring); whitespace is kept.  A formula
    written with spaces inside its brackets (sqrt(1 - q)) is taken as one token when every
    piece of it is math-like; otherwise word by word."""
    text = re.sub(r"(?<=\S) -> (?=\S)", " \x00 ", str(text))    # a free-standing arrow
    parts = re.split(r"(\s+)", text)
    out, i = [], 0
    while i < len(parts):
        part = parts[i]
        if part == "\x00":
            out.append(r"$\to$")
        elif part.isspace() or not part:
            out.append(part)
        else:
            j, tok = i, part
            if _open(part) > 0 and _math_like(part):
                k, cand = i, part
                while k + 2 < len(parts) and _open(cand) > 0 and k - i < 60:
                    nxt = parts[k + 2]
                    if not _math_like(nxt) or nxt == "\x00":
                        break
                    cand, k = cand + parts[k + 1] + nxt, k + 2
                if _open(cand) == 0 and _MATHY.search(cand):
                    j, tok = k, cand
            out.append(_word(tok))
            i = j
        i += 1
    return "".join(out)
