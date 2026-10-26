"""Catalog entry `teacher_student` (spherical teacher and student, noiseless labels;
learning gibbs|bayes): the replica route (families/teacher_student.py) and the amp route
(catalog_amp.py).  Split out of catalog_entries.py; conventions as stated there.
"""
from .catalog import Out, RefCheck, Stability, entry, fstep, has, lacks
from .catalog_amp import ts_amp_route
from .families.common import CAVEAT
from .families.teacher_student import asymptotic_constant, teacher_student_error
from .spec import ModelSpec

TS_CITE = ("Gyorgyi, Phys. Rev. A 41, 7097 (1990); Seung, Sompolinsky & Tishby, Phys. Rev. A "
           "45, 6056 (1992); Opper & Haussler, PRL 66, 2677 (1991)")


def _saddle(ansatz, stable, **values):
    """The values of one saddle point (shared with catalog_entries.py)."""
    return {"ansatz": ansatz, "stable": stable, **values}


# ---- teacher-student -------------------------------------------------------------------
def _ts_const(p):
    return Out({"gibbs_asymptotic_constant": asymptotic_constant("gibbs"),
                "bayes_asymptotic_constant": asymptotic_constant("bayes")},
               "lim alpha eps(alpha): sqrt(2)/I (Gibbs), 1/I (Bayes)")


def _ts_curve(p):
    learning = p.get("learning", "gibbs")
    eps = teacher_student_error(p["alpha"], learning)
    return Out(_saddle("rs", None, eps=eps, learning=learning),
               "RS on the Nishimori line q = R; RS stability is argued (convex version "
               "space) but not computed by the family")


def teacher_student():
    steps = [fstep("ts-asymptotic-constants", TS_CITE, _ts_const, when=lacks("alpha"),
                   skip="alpha given (state point)"),
             fstep("ts-learning-curve", TS_CITE, _ts_curve, when=has("alpha"),
                   skip="no alpha given")]
    refs = [RefCheck("perceptron", "gibbs_asymptotic_constant",
                     lambda v: v["gibbs_asymptotic_constant"]),
            RefCheck("perceptron", "bayes_asymptotic_constant",
                     lambda v: v["bayes_asymptotic_constant"])]
    spec = ModelSpec("teacher_student", "spherical", "gaussian", prior="teacher")
    exact = Stability("exact-by-literature", "rs",
                      "Barbier, Krzakala, Macris, Miolane & Zdeborova, PNAS 116, 5451 (2019) "
                      "(the RS formula is proved for Bayes-optimal generalized linear models)",
                      "Gibbs sampling of the version space is Bayes-consistent with the teacher "
                      "prior (Nishimori line q = R), so both curves are on the proved RS branch. "
                      "Gibbs learning on the version space with a uniform prior is posterior "
                      "sampling, so the Nishimori identity applies (overlap concentration of "
                      "Barbier et al.)")
    fin = {k: exact for k in ("gibbs_asymptotic_constant", "bayes_asymptotic_constant", "eps")}
    amp = ts_amp_route()
    return entry("teacher_student", spec, TS_CITE, {"replica": steps, "amp": amp.steps}, refs,
                 {"replica": CAVEAT, "amp": amp.caveat},
                 "teacher-student perceptron: Gibbs and Bayes learning curves; GAMP / SE",
                 finality={**fin, **amp.finality}, meta=("learning", *amp.meta),
                 parameters={"replica": ("alpha", "learning"), "amp": amp.params})


__all__ = ["teacher_student"]
