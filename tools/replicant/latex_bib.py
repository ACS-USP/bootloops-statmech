"""Bibliography of a LaTeX document: cite strings -> \\cite keys and a thebibliography
(no BibTeX, so a document compiles offline)."""
import re

from .tex import escape


_JOURNAL = re.compile(r"(J\. |Phys\. |PRL |Ann\. |Eur\. |Z\. |Europhys|PNAS )")
_IDENT = re.compile(r"^([^,]+),.*?\b(\d+),\s*(L?\d+)\b")


class _Bib:
    """Cite strings -> \\cite keys.  Items are split at top-level '; '; an item that starts
    with a journal inherits the previous item's authors ("Parisi, PRL 43, 1754 (1979); J.
    Phys. A 13, L115 (1980)"); items naming the same authors, volume and first page are one
    reference (the longest wording is printed), so a paper cited with different
    annotations, e.g. "(RS)" or a doi, appears once."""
    def __init__(self):
        self.keys = {}          # identity -> key
        self.text = {}          # key -> printed text

    @staticmethod
    def items(cite):
        out, depth, cur = [], 0, ""
        for i, ch in enumerate(cite):
            depth += {"(": 1, ")": -1}.get(ch, 0)
            if ch == ";" and depth == 0 and cite[i + 1:i + 2] == " ":
                out.append(cur.strip())
                cur = ""
            else:
                cur += ch
        return [x for x in out + [cur.strip()] if x]

    def cite(self, cite):
        if not cite:
            return ""
        keys, prev = [], None
        for it in self.items(cite):
            if prev and _JOURNAL.match(it):
                it = f"{prev}, {it}"
            m = _IDENT.match(it)
            prev = m.group(1) if m else None
            ident = (m.group(1).lower(), m.group(2), m.group(3)) if m else it
            k = self.keys.setdefault(ident, f"b{len(self.keys) + 1}")
            if len(it) > len(self.text.get(k, "")):
                self.text[k] = it
            if k not in keys:
                keys.append(k)
        return f"\\cite{{{','.join(keys)}}}"

    def render(self):
        if not self.keys:
            return ""
        rows = "\n".join(f"\\bibitem{{{k}}} {escape(self.text[k])}" for k in self.keys.values())
        return f"\n\\begin{{thebibliography}}{{99}}\n{rows}\n\\end{{thebibliography}}\n"
