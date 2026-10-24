"""PDF compilation of a written document, and the overfull-display audit.

  compile_pdf(tex) -> Path       tectonic (preferred, --keep-logs) or pdflatex (twice);
                                 no engine: Refusal precondition:tex-engine; a failed run:
                                 tex-engine:failed, or tex-engine:unavailable when the log
                                 says the engine could not fetch its packages
  overfull(log, min_pt=1.0)      [(pt, line, kind)] of the log's "Overfull \\hbox" lines;
                                 kind "display" (math display: "detected at line N") or
                                 "paragraph" ("in paragraph at lines a--b")

A display wider than the text block (min_pt or more) is reported as an OverfullDisplay
warning (warnings.warn) after a successful compile: the content may be cut at the paper
edge, so callers (the CLI, the tests) treat it as a defect of the document.
"""
import re
import shutil
import subprocess
import warnings
from pathlib import Path

from .core import Refusal

_OFFLINE = re.compile(r"network|download|resolve host|could not connect|offline|"
                      r"failed to fetch", re.I)
_OVER = re.compile(r"Overfull \\hbox \(([\d.]+)pt too wide\) (detected at line (\d+)|"
                   r"in paragraph at lines (\d+)--\d+|in alignment at lines (\d+)--\d+)")


class OverfullDisplay(UserWarning):
    pass


def overfull(log, min_pt=1.0):
    text = Path(log).read_text(errors="replace") if not isinstance(log, str) or \
        "\n" not in log else log
    out = []
    for m in _OVER.finditer(text):
        pt = float(m.group(1))
        if pt < min_pt:
            continue
        if m.group(3):
            out.append((pt, int(m.group(3)), "display"))
        elif m.group(5):
            out.append((pt, int(m.group(5)), "display"))     # an align row
        else:
            out.append((pt, int(m.group(4)), "paragraph"))
    return out


def compile_pdf(tex: Path, *, min_pt=1.0) -> Path:
    tex = Path(tex)
    engine = shutil.which("tectonic") or shutil.which("pdflatex")
    if engine is None:
        raise Refusal("precondition:tex-engine", "neither tectonic nor pdflatex is on PATH")
    d = str(tex.parent)
    if Path(engine).name == "tectonic":
        runs = [[engine, "--chatter", "minimal", "--keep-logs", "--outdir", d, str(tex)]]
    else:
        runs = [[engine, "-interaction=nonstopmode", "-halt-on-error",
                 f"-output-directory={d}", str(tex)]] * 2      # twice: citations
    for cmd in runs:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if p.returncode != 0:
            log = (p.stdout + p.stderr)[-1500:]
            reason = "tex-engine:unavailable" if _OFFLINE.search(log) else "tex-engine:failed"
            raise Refusal(reason, f"{Path(engine).name} exit {p.returncode}: {log}")
    pdf = tex.with_suffix(".pdf")
    if not pdf.exists():
        raise Refusal("tex-engine:failed", f"{engine} wrote no {pdf.name}")
    log = tex.with_suffix(".log")
    if log.exists():
        bad = [o for o in overfull(log, min_pt) if o[2] == "display"]
        if bad:
            warnings.warn(OverfullDisplay(
                f"{tex.name}: {len(bad)} display(s) wider than the text block: " +
                ", ".join(f"{pt:.1f} pt at line {ln}" for pt, ln, _ in bad)))
    return pdf
