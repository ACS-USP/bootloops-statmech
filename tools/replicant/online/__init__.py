"""Online learning (O1-O6): order-parameter ODEs in alpha = examples / N.

  fields.py    Gaussian averages over the teacher and student local fields: closed forms and an
               independent polar quadrature
  odes.py      the perceptron flow dR/dalpha, dQ/dalpha (rule, learning rate or schedule, output
               noise), its integrator, the Hebb closed form, the asymptotic scaling
  optimal.py   the variationally optimal modulation (Kinouchi-Caticha 1992; VKC 1998 Eq. (25)),
               its asymptotic constant and the noise-robustness boundaries (Copelli et al. 1997)
  simulate.py  finite-N online learning over named seeds (shares no code with the ODEs)
  linear.py    (O2) the linear perceptron with quadratic error: standard and optimal rules, sphere or
               unconstrained, greedy eta, closed forms (Kinouchi-Caticha 1995)
  linear_offline.py  zero-temperature offline solutions of the linear perceptron and the ridge route
  simulate_linear.py finite-N online and offline linear learning (shares no code with the above)
  drift.py     (O3) a drifting teacher (random or deterministic, jumps, schedules) composed onto the
               flow; stationary states, small-drift laws, the KC93 bound (Kinouchi-Caticha 1993; VKC 1998)
  drift_rules.py     VKC's annealed Hebb, step, symmetric modulations, their averages and closed forms
  simulate_drift.py  finite-N online learning with the teacher drifting (shares no code with the ODEs)

Convention (Biehl & Riegler, Europhys. Lett. 28, 525 (1994)): |B| = 1, xi_j i.i.d. N(0, 1),
J <- J + (eta / N) F(h, b, Q) b xi, h = J . xi, R = J . B, Q = J . J, eps = arccos(R / sqrt Q) / pi.
"""
