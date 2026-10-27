"""The k-core's birth point, certified: the global minimum of H(s) = s / Phi(s)^m, its location and the
core's size at the jump.

Threshold (K = 2: DGM PRL 96, 040601, Eqs. (6)-(8): p_c(k) = 1/max f_k(R), f_k = (1 - Phi_k)/(1 - R);
Janson, arXiv:0804.1656 (J09) Eq. (4.9): pi_c = inf_(0 < p <= 1) lambda p^2 / h(p) = inf p / Psi(p),
the same for site (Thm 4.5) and bond (Thm 4.6) dilution; Erdos-Renyi in the mean degree: Pittel,
Spencer & Wormald, as stated by Janson & Luczak, arXiv:math/0508453 (JL07), Thm 1.1:
lambda_k = min_(mu > 0) mu / psi_(k-1)(mu), psi_j(mu) = P(Po(mu) >= j)):
  in the occupation t, at a fixed law: H(y) = y / Psi(y)^m on (0, 1], m = K - 1; site t_c = H*^(1/m),
      bond t_c = H* (K = 2: both min y / Psi(y); K >= 3 derived here from y = b x^m, x = a Psi(y))
  in the Poisson mean degree c, at a fixed t: H(mu) = mu / P(Po(mu) >= k - 1)^m on (0, oo); site
      c_k = H* / t^m, bond c_k = H* / t (JL07's lambda_k at K = 2, t = 1; at K >= 3, k = 2 the
      XORSAT 2-core density alpha_d = c_k / K of csp/xorsat.py)
  at the jump (the t -> t_c+ limit of the selected root, derived here): y* = argmin, site x* = t_c
      Psi(y*), V* = t_c D(y*), bond x* = Psi(y*), V* = D(y*); D(y) = sum_d p_d P(Bin(d, y) >= k)
      (Poisson P(Po(c y) >= k)); hyperedges (<d>/K) b x*^K.
Certification.  With G(s) = Phi(s) - m s Phi'(s) (H' has the sign of G): an interval-Newton zero s*
of G with G' > 0 certified on a box J around it is the only minimum in J; outside J a branch and
bound proves H > upper(H(s*)) using only that Phi is increasing (H >= a / Phi(b)^m on [a, b]), the
bound Phi(s) <= B s^(k-1) near 0 (H >= s^(1 - (k-1) m) / B^m, finite since (k-1) m >= 2), and, on
(0, oo), H >= s for s large.  A minimum at the end s = 1 is certified by G < 0 on [1 - delta, 1].
A box that can be neither discarded nor resolved at width 1e-25 (two minima of nearly equal height)
is refused undetermined:threshold-minimum.  Further first-order transitions above the birth point
(J09 Remark 4.7: other local maxima of h(p)/p^2) are not enumerated here; the selected root at a
run point (kcore.py) is always the largest one.  The continuous case k = 2, K = 2 has t_c = 1/G1'(1)
(an exact rational, J09 Remark 4.9) and no jump.
_HOOKS["threshold_offset"] (0): a planted offset added to s* (tests).
"""
from dataclasses import dataclass
from fractions import Fraction

from ..core import Refusal
from ..lattice.ball import Ball, arb_q, ball_of, baller
from .kcore import PREC, Model
from .laws import parse_law

_HOOKS = {"threshold_offset": 0}
SCAN = 96


@dataclass(frozen=True)
class Threshold:
    variable: str          # "occupation" or "mean degree"
    value: Ball            # t_c or c_k
    s_star: Ball           # argmin: y* (occupation) or mu* (mean degree)
    x: Ball | None         # x at the jump
    core: Ball | None      # vertices in the core at the jump
    edges: Ball | None
    endpoint: bool         # the minimum sits at s = 1
    continuous: bool
    boxes: int             # branch-and-bound boxes examined


class _H:
    def __init__(self, phi, dphi, d2phi, m):
        self.phi, self.dphi, self.d2phi, self.m = phi, dphi, d2phi, m

    def H(self, s):
        return s / self.phi(s) ** self.m

    def G(self, s):
        return self.phi(s) - self.m * s * self.dphi(s)

    def dG(self, s):
        return (1 - self.m) * self.dphi(s) - self.m * s * self.d2phi(s)


def _scan(h, top):
    from flint import arb, ctx
    best = None
    with baller().hygiene.ctx_guard(prec=64):
        for i in range(1, SCAN + 1):
            s = arb(top) * i / SCAN
            p = h.phi(s)
            if not p > 0:
                continue
            v = float((s / p ** h.m).mid())
            if best is None or v < best[0]:
                best = (v, i)
    if best is None:
        raise Refusal("precondition:core-order", "Phi = 0 on the whole scan: the core never appears")
    return best[1]


def _interior(h, s0):
    from flint import arb
    x = arb(s0)
    for _ in range(200):
        xn = (x - h.G(x) / h.dG(x)).mid()
        done = abs(xn - x) < arb(2) ** (16 - PREC)
        x = xn
        if done:
            break
    r = arb(2) ** (-(PREC // 2))
    for _ in range(40):
        box = arb(x, r)
        N = x - h.G(arb(x)) / h.dG(box)
        if box.contains_interior(N):
            return N
        r *= 4
    raise Refusal("undetermined:threshold-minimum", f"G = Phi - m s Phi' has no certified zero near {x.str(12)}")


def _increasing(h, lo, hi, pieces=8):
    """G' > 0 on [lo, hi], certified on `pieces` sub-boxes (H then has one minimum there, at G = 0)."""
    w = (hi - lo) / pieces
    return all(h.dG((lo + i * w).union(lo + (i + 1) * w)).lower() > 0 for i in range(pieces))


def _minimise(h, top, B, r):
    """(s*, H*, J = (lo, hi), endpoint, boxes): the certified global minimum of H on (0, top]
    (top None: (0, oo))."""
    from flint import arb
    T = arb(top) if top is not None else arb(8 * r + 40)
    i = _scan(h, T)
    endpoint = top is not None and i == SCAN
    if endpoint:
        s = arb(1)
        delta = arb(1) / 64
        while not h.G((arb(1) - delta).union(arb(1))).upper() < 0:      # H decreasing on [1 - delta, 1]
            delta /= 4
            if delta < arb(2) ** -60:
                raise Refusal("undetermined:threshold-minimum", "the minimum at s = 1 is not certified")
        J = (arb(1) - delta, arb(1))
    else:
        s = _interior(h, (T * i / SCAN).mid())
        s = s + arb_q(_HOOKS["threshold_offset"])
        delta = s.mid() / 2 if top is None else min(s.mid(), arb(top) - s.mid()) / 2
        while not _increasing(h, s.mid() - delta, s.mid() + delta):
            delta /= 4
            if delta < arb(2) ** -60:
                raise Refusal("undetermined:threshold-minimum", f"G' is not certified > 0 near s* = {s.str(12)}")
        J = (s.mid() - delta, s.mid() + delta)
    Hs = h.H(s)
    cap = Hs.upper()
    m = h.m
    e = r * m - 1
    j = 1
    while not arb(2) ** (j * e) / arb_q(B) ** m > cap:
        j += 1
    a0 = arb(2) ** (-j)
    hi_end = arb(top) if top is not None else None
    if hi_end is None:
        L = arb(2)
        while not L > cap:
            L *= 2
        hi_end = L                          # [L, oo): H >= mu / 1 >= L > H*
    stack, boxes = [], 0
    if J[0] > a0:
        stack.append((a0, J[0]))
    if J[1] < hi_end:
        stack.append((J[1], hi_end))
    eps = arb(10) ** -25
    while stack:
        a, b = stack.pop()
        boxes += 1
        if (a / h.phi(b).upper() ** m).lower() > cap:
            continue
        if b - a < eps or boxes > 200000:
            raise Refusal("undetermined:threshold-minimum", f"H = s / Phi(s)^{m} on [{a.str(12)}, "
                          f"{b.str(12)}] is not separated from the candidate minimum {Hs.str(15)}: two "
                          "minima of nearly equal height, or a wrong candidate")
        mid = ((a + b) / 2).mid()
        stack += [(a, mid), (mid, b)]
    return s, Hs, endpoint, boxes


def _h_of(M):
    return _H(M.psi, M.dpsi, M.d2psi, M.m)


def occupation_threshold(law, K, k, kind) -> Threshold:
    """t_c in the occupation for site or bond dilution of `law`."""
    M = Model(law, K, k, ("site", 1))
    if kind not in ("site", "bond"):
        raise Refusal("precondition:dilution", f"{kind!r}: need 'site' or 'bond'")
    if M.continuous:
        br = parse_law(law).branching()
        with baller().hygiene.ctx_guard(prec=PREC):
            tc = ball_of(arb_q(1 / br)) if br else None
            z = ball_of(arb_q(0))
        if tc is None:
            raise Refusal("precondition:core-order", "G1'(1) = 0: no 2-core for any occupation")
        return Threshold("occupation", tc, z, z, z, z, False, True, 0)
    with baller().hygiene.ctx_guard(prec=PREC):
        s, Hs, endpoint, boxes = _minimise(_h_of(M), 1, M.bound_B(), M.k - 1)
        tc = Hs if kind == "bond" or M.m == 1 else Hs ** (1 / arb_q(M.m))
        if kind == "site":
            x, core, b = tc * M.psi(s), tc * M.core(s), 1
        else:
            x, core, b = M.psi(s), M.core(s), tc
        edges = arb_q(M.z) / M.K * b * x ** M.K
        out = Threshold("occupation", ball_of(tc), ball_of(s), ball_of(x), ball_of(core),
                        ball_of(edges), endpoint, False, boxes)
    return out


def degree_threshold(K, k, kind, t) -> Threshold:
    """c_k: the Poisson mean degree at which the k-core appears, at occupation t."""
    t = Fraction(t)
    M = Model(("poisson", 1), K, k, ("site", 1))
    if M.continuous:
        with baller().hygiene.ctx_guard(prec=PREC):
            z = ball_of(arb_q(0))
            if t == 0:
                raise Refusal("precondition:occupation", "t = 0: no core for any mean degree")
            return Threshold("mean degree", ball_of(arb_q(1 / t)), z, z, z, z, False, True, 0)
    if t == 0:
        raise Refusal("precondition:occupation", "t = 0: no core for any mean degree")
    with baller().hygiene.ctx_guard(prec=PREC):
        s, Hs, endpoint, boxes = _minimise(_h_of(M), None, M.bound_B(), M.k - 1)
        T = arb_q(t)
        ck = Hs / T ** M.m if kind == "site" else Hs / T
        if kind == "site":
            x, core, b = T * M.psi(s), T * M.core(s), 1
        else:
            x, core, b = M.psi(s), M.core(s), T
        edges = ck / M.K * b * x ** M.K
        return Threshold("mean degree", ball_of(ck), ball_of(s), ball_of(x), ball_of(core),
                         ball_of(edges), endpoint, False, boxes)


def bootstrap_closed_form(d, ell):
    """(q_c, y*) of bootstrap percolation on the random d-regular graph with threshold ell, as exact
    rationals, where a closed form is printed: Bartha & Pete, arXiv:1509.08454v1, Sec. 3 below
    Eq. (3.1), stating Balogh & Pittel (2007): ell = 2, d > 3: y* = (d-1)(d-3)/(d-2)^2, p* = 1 -
    (d-2)^(2d-5) / ((d-1)^(d-2) (d-3)^(d-3)); ell = d - 1: p* = 1 - 1/(d-1) (no interior y*).
    Their p* is Janson's q_c (arXiv:0804.1656, Eq. (5.1)).  None where no closed form is printed."""
    if ell == 2 and d > 3:
        return (1 - Fraction((d - 2) ** (2 * d - 5), (d - 1) ** (d - 2) * (d - 3) ** (d - 3)),
                Fraction((d - 1) * (d - 3), (d - 2) ** 2))
    if ell == d - 1:
        return 1 - Fraction(1, d - 1), None
    return None


__all__ = ["Threshold", "bootstrap_closed_form", "degree_threshold", "occupation_threshold"]
