"""The result sections of a LaTeX derivation document (Solution, Stability, Reference
check) and the value formatter; split out of latex.py, which assembles the document.
"""
import mpmath as mp

from .core import Refusal
from .tex import escape, prose

_STAB = ("replicon", "stability", "capacity_stable", "stable", "mode_eig")


def _num(v):
    if isinstance(v, (mp.mpf, mp.mpc)):
        return mp.nstr(v, 15)
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _solution(res, bib):
    stat = set(res.statistical)
    rows = []
    for k, v in res.values.items():
        st = res.finality.get(k)
        if st is None:
            continue
        val = escape(_num(v))
        if k in stat:
            val += f" $\\pm$ {escape(_num(res.errors.get(k)))} (statistical)"
        elif res.rigor == "certified" and k in res.errors:
            val += f" $\\pm$ {escape(_num(res.errors[k]))} (certified ball)"
        rows.append(f"{escape(k)} & {val} & {escape(st.state)} ({escape(st.ansatz)}) "
                    f"{bib.cite(st.cite)} & {'yes' if st.final else 'no'} \\\\")
    meta = [f"{prose(k)} = {prose(_num(v))}" for k, v in res.values.items()
            if k not in res.finality]
    errs = [f"{prose(k)} = {prose(_num(v))}" for k, v in res.errors.items() if k not in stat]
    notes = [n for n in res.notes if "NOT final" in n]
    out = ("\\section{Solution}\nValues at the requested parameters (" +
           prose(", ".join(f"{k} = {_num(v)}" for k, v in res.params.items()) or "none") +
           "), each with its finality: only stable and exact-by-literature values are final.\n"
           "\\begin{center}\n\\begin{tabular}{llll}\nkey & value & finality (ansatz) & final \\\\"
           "\n\\hline\n" + "\n".join(rows) + "\n\\end{tabular}\n\\end{center}\n")
    if notes:
        out += "\\begin{itemize}\n" + "".join(f"\\item {prose(n)}\n" for n in notes) + \
            "\\end{itemize}\n"
    if meta:
        out += "Other outputs (parameters, labels, diagnostics): " + "; ".join(meta) + ".\n\n"
    if errs:
        out += "Numerical error estimates: " + "; ".join(errs) + ".\n"
    return out


def _stability(res):
    comp = [f"{prose(k)} = {prose(_num(v))}" for k, v in res.values.items()
            if any(s in k for s in _STAB)]
    by = {}
    for k, st in res.finality.items():
        by.setdefault(st.state, []).append(k)
    out = "\\section{Stability}\n"
    out += ("Computed stability quantities (replicon / AT / cavity susceptibility): "
            + "; ".join(comp) + ".\n\n") if comp else \
        "No stability quantity (replicon / AT) is computed by this run.\n\n"
    out += "".join(f"{prose(s)}: {prose(', '.join(ks))}.\n\n" for s, ks in sorted(by.items()))
    return out


def _references(res, bib, slow):
    from .refdata import load
    out = "\\section{Reference check}\n"
    if res.entry is None:
        near = res.report.nearest if res.report else None
        return out + prose(f"No banked keys: a composed model is not a catalog entry, so "
                            f"nothing is compared here (nearest entry: {near}).  This document "
                            "does not certify its numbers: a composed model is validated only "
                            "by an independent route (enumeration at small N, or a limit with "
                            "a known answer) in the test suite, where such a test exists.") + "\n"
    try:
        done = {(o.file, o.key): o for o in res.reference_checks(slow=slow)}
    except Refusal as r:
        return out + prose(f"refused: {r.reason}: {r.detail}") + "\n"
    rows, stat = [], []
    for ref in res.entry.refs:
        if ref.route != res.route:
            continue
        want = load(ref.file)[ref.key]
        o = done.get((ref.file, ref.key))
        if o is None:
            got, verdict = "not run (" + ("statistical" if ref.statistical else "slow") + \
                ")", "--"
        else:
            got, verdict = escape(_short(o.got)), ("agrees" if o.ok else "DISAGREES")
            if o.statistical:
                got += " (statistical)"
                stat.append(f"\\item {prose(ref.key)}: {prose(o.detail)}\n")
        rows.append(f"{escape(ref.key)} & {got} & {escape(_short(want.value))} & "
                    f"{escape(want.tol)} & {verdict} & {bib.cite(want.cite)} \\\\")
    return (out + "Banked values (reference/*.json) against this pipeline's output; numbers "
            "are shown to 16 significant digits, the comparison uses them in full; slow and "
            "statistical rows run only with --full (\\texttt{render\\_latex(slow=True)}).\n"
            "\\begin{center}\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n" + _FIT +
            "{\\begin{tabular}{llllll}\n"
            "key & computed & banked & tolerance & verdict & source \\\\\n\\hline\n" +
            "\n".join(rows) + "\n\\end{tabular}}\n\\end{center}\n" +
            ("Statistical comparisons:\n\\begin{itemize}\n" + "".join(stat) +
             "\\end{itemize}\n" if stat else ""))


_FIT = "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}"   # shrink if wide


def _short(s, digits=16):
    """A decimal string to `digits` significant digits (display only); other text as is."""
    with mp.workdps(40):
        try:
            x = mp.mpf(str(s))
        except (ValueError, TypeError):
            return str(s)
        return mp.nstr(x, digits) if len(str(s)) > digits + 3 else str(s)
