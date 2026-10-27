"""Exact finite tori of the square, triangular and honeycomb lattices, and the star-triangle and
duality relations as exact finite-lattice identities (no bulk formula is used here).

Geometry.  A torus is Lx rows of Ly cells; a cell holds `t` sites (kind 0 .. t-1); a unit edge
(kp, kq, dj, di, c) joins (i, kp, j) to (i + di, kq, j + dj) (indices mod Lx, Ly), class c:
  square          t = 1: h (0, 0, 1, 0, 0), v (0, 0, 0, 1, 1)
  triangular      t = 1: h (0, 0, 1, 0, 0), v (0, 0, 0, 1, 1), d (0, 0, 1, 1, 2)
  honeycomb       t = 2, A = kind 0 (the triangular torus's sites), B = kind 1 (one per up-triangle
                  {(i, j), (i, j+1), (i+1, j+1)}): legs a (1, 0, 0, 0, 0), b (1, 0, 1, 0, 1),
                  c (1, 0, 1, 1, 2).  Decimating B gives the triangular torus, the leg opposite
                  each triangle edge being c (h), a (v), b (d).
  honeycomb_dual  t = 2: the dual of the triangular torus, kind 0 = up-triangle (i, j), kind 1 =
                  down-triangle {(i, j), (i+1, j), (i+1, j+1)}; each dual edge has the class of
                  the triangular edge it crosses: (0, 1, 0, 0, 2) crosses d, (0, 1, 1, 0, 1) v,
                  (1, 0, 0, 1, 0) h.
A sector (a, b) multiplies the coupling of every edge wrapping the i (j) direction by (-1)^a
((-1)^b): antiperiodic seams.
Two exact routes for Z, sharing only the unit description:
  counts(geom, Lx, Ly, sector) -> {(j0, j1, j2): n}   enumeration of all 2^n configurations
      (numpy, exact integers; n <= 22): n configurations with j_c unsatisfied class-c bonds
      (unsatisfied: sign * s s' = -1), so Z = sum n prod_c e^{K_c (E_c - 2 j_c)};
  transfer_z(geom, Lx, Ly, Ks, sector) -> arb          Tr T^(Lx-1) T_seam with arb_mat entries
      e^{sum K s s'} (rows of t Ly <= 8 spins): a ball of the exact finite Z at any couplings;
  kronecker(geom, Lx, Ly, sector) -> (c_0, ..., c_E)   the univariate polynomial (all couplings
      equal) from the transfer matrix in exact integers (Kronecker substitution, rows <= 6 spins).
Identities (derived here; Baxter, Exactly Solved Models (1982), Secs. 6.3, 6.4 for the bulk forms):
  star-triangle: Z_honeycomb(L1, L2, L3) = R^N Z_triangular(h = K3, v = K1, d = K2) exactly on the
      matched tori (every B spin sits in one star; Baxter (6.4.3)-(6.4.7)), N = Lx Ly;
  duality: the high-temperature expansion of the triangular torus summed over the four sectors keeps
      the homologically trivial polygons, which are the domain walls of the dual honeycomb (two
      configurations each).  With e^{-2L_c} = tanh K_c = (1 - y_c)/(1 + y_c), y_c = e^{-2K_c}:
        sum_j n^H_j prod_c (1 - y_c)^{j_c} (1 + y_c)^{E_c - j_c} = 2^{E - N - 1} sum_{a,b} P_ab(y),
      P_ab(y) = sum n^T_ab(j) prod y_c^{j_c}, E = 3N: an identity of integer polynomials.  The
      periodic sector alone (Baxter's (6.3.7) as a finite statement) is not exact on the torus: the
      difference is the topology term.
  duality_identity(Lx, Ly) -> DualityCheck;  star_triangle_check(Lx, Ly, L) -> (Ball, Ball)
_HOOKS: "edge_scale" ((1, 1, 1)): the Boltzmann weight's coupling per class (ints or Fractions), in
every route (planted (2, 1, 1) or (51/50, 1, 1): a wrong lattice weight, invisible to counts =
transfer; seen by the high-temperature torus = bulk identity, the star-triangle identity and the
cylinders).  The duality map is lattice/anisotropic._HOOKS["dual_mobius"].
"""
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import comb

import numpy as np

from ..core import Refusal
from . import anisotropic as A
from .ball import arb_q, certify

GEOM = {"square": (1, ((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))),
        "triangular": (1, ((0, 0, 1, 0, 0), (0, 0, 0, 1, 1), (0, 0, 1, 1, 2))),
        "honeycomb": (2, ((1, 0, 0, 0, 0), (1, 0, 1, 0, 1), (1, 0, 1, 1, 2))),
        "honeycomb_dual": (2, ((0, 1, 0, 0, 2), (0, 1, 1, 0, 1), (1, 0, 0, 1, 0)))}
MAX_ENUM, MAX_ROW, MAX_KRON_ROW = 22, 8, 6
_HOOKS = {"edge_scale": (1, 1, 1)}


def _check(geom, Lx, Ly, row_max=None):
    if geom not in GEOM:
        raise Refusal("precondition:lattice", f"lattice {geom!r}: one of {', '.join(GEOM)}")
    for name, L in (("Lx", Lx), ("Ly", Ly)):
        if isinstance(L, bool) or not isinstance(L, int) or not 3 <= L <= 64:
            raise Refusal("precondition:size", f"{name} = {L!r}: need an int in 3 .. 64")
    t = GEOM[geom][0]
    if row_max is not None and t * Ly > row_max:
        raise Refusal("precondition:size", f"{geom}: a row of {t * Ly} spins exceeds {row_max}")
    return t


def edges(geom, Lx, Ly, sector=(0, 0)):
    """[(p, q, c, sign)] of the torus; site (i, kind, j) is i t Ly + kind Ly + j."""
    t = _check(geom, Lx, Ly)
    out = []
    for i in range(Lx):
        for j in range(Ly):
            for kp, kq, dj, di, c in GEOM[geom][1]:
                wi, wj = i + di >= Lx, not 0 <= j + dj < Ly
                p = i * t * Ly + kp * Ly + j
                q = ((i + di) % Lx) * t * Ly + kq * Ly + (j + dj) % Ly
                out.append((p, q, c, (-1) ** (sector[0] * wi + sector[1] * wj)))
    return out


def class_sizes(geom, Lx, Ly):
    n = [0, 0, 0]
    for _, _, c, _ in edges(geom, Lx, Ly):
        n[c] += 1
    return tuple(n)


# ---- route 1: enumeration ---------------------------------------------------------------------
def counts(geom, Lx, Ly, sector=(0, 0)):
    """{(j0, j1, j2): n} by enumeration; memoised (exact integers, no hook enters), a fresh dict."""
    return dict(_counts(geom, Lx, Ly, tuple(sector)))


@lru_cache(maxsize=64)
def _counts(geom, Lx, Ly, sector):
    t = _check(geom, Lx, Ly)
    n = t * Lx * Ly
    if n > MAX_ENUM:
        raise Refusal("precondition:size", f"{n} sites: enumeration is limited to {MAX_ENUM}")
    es = edges(geom, Lx, Ly, sector)
    E = class_sizes(geom, Lx, Ly)
    base = (E[0] + 1, E[1] + 1)
    hist = np.zeros(base[0] * base[1] * (E[2] + 1), dtype=np.int64)
    chunk = 1 << min(n, 20)
    for start in range(0, 1 << n, chunk):
        cfg = np.arange(start, start + chunk, dtype=np.int64)
        bits = [((cfg >> i) & 1).astype(np.uint8) for i in range(n)]
        j = [np.zeros(chunk, dtype=np.uint8) for _ in range(3)]       # j_c <= E_c <= 66 < 256
        for p, q, c, sg in es:
            un = bits[p] ^ bits[q]
            j[c] += un if sg > 0 else 1 - un
        idx = j[0].astype(np.int32) + base[0] * (j[1].astype(np.int32) + base[1] * j[2].astype(np.int32))
        hist += np.bincount(idx, minlength=hist.size)
    out = {}
    for idx in np.nonzero(hist)[0]:
        idx = int(idx)
        out[(idx % base[0], (idx // base[0]) % base[1], idx // (base[0] * base[1]))] = int(hist[idx])
    return out


def z_of_counts(cnt, E, Ks):
    """sum n prod_c e^{K_c (E_c - 2 j_c)} (arb couplings Ks)."""
    Ks = [Ks[c] * arb_q(Fraction(_HOOKS["edge_scale"][c])) for c in range(3)]
    tot = 0 * Ks[0]
    for js, m in cnt.items():
        tot += m * sum(((E[c] - 2 * js[c]) * Ks[c] for c in range(3)), 0 * Ks[0]).exp()
    return tot


# ---- route 2: the transfer matrix --------------------------------------------------------------
def _row_data(geom, Ly, sector):
    t, unit = GEOM[geom]
    m = t * Ly
    S = 1 - 2 * ((np.arange(1 << m)[:, None] >> np.arange(m)) & 1)
    intra = np.zeros((1 << m, 3), dtype=np.int64)
    inter = np.zeros((1 << m, 1 << m, 3), dtype=np.int64)
    inter_seam = np.zeros_like(inter)
    for j in range(Ly):
        for kp, kq, dj, di, c in unit:
            wj = not 0 <= j + dj < Ly
            sg = (-1) ** (sector[1] * wj)
            p, q = kp * Ly + j, kq * Ly + (j + dj) % Ly
            if di == 0:
                intra[:, c] += sg * S[:, p] * S[:, q]
            else:
                o = np.outer(S[:, p], S[:, q])
                inter[:, :, c] += sg * o
                inter_seam[:, :, c] += sg * (-1) ** sector[0] * o
    return m, intra, inter, inter_seam


def transfer_z(geom, Lx, Ly, Ks, sector=(0, 0)):
    """arb: Tr T^(Lx-1) T_seam, T[s, s'] = exp(sum_c K_c (intra_c(s) + inter_c(s, s')))."""
    from flint import arb_mat
    _check(geom, Lx, Ly, MAX_ROW)
    Ks = [Ks[c] * arb_q(Fraction(_HOOKS["edge_scale"][c])) for c in range(3)]
    m, intra, inter, seam = _row_data(geom, Ly, sector)
    off = 4 * m + 1                     # |n_c| <= 3 m: shift to a non-negative digit

    def mat(X):
        tot = X + intra[:, None, :] + off
        code = tot[:, :, 0] + 2 * off * (tot[:, :, 1] + 2 * off * tot[:, :, 2])
        uniq, inv = np.unique(code, return_inverse=True)
        ws = []
        for u in uniq.tolist():
            n = (u % (2 * off) - off, (u // (2 * off)) % (2 * off) - off, u // (4 * off * off) - off)
            ws.append(sum((n[c] * Ks[c] for c in range(3)), 0 * Ks[0]).exp())
        idx = inv.reshape(code.shape).tolist()
        return arb_mat([[ws[k] for k in row] for row in idx])
    T, Ts = mat(inter), mat(seam)
    P = T ** (Lx - 1) * Ts
    return sum((P[s, s] for s in range(1 << m)), 0 * Ks[0])


@A.memo
def cylinder_free_energy(geom, Ly, Ks, prec=128, max_squarings=40):
    """Ball of ln lambda_0 / (t Ly), lambda_0 the Perron root of the row transfer matrix of the
    infinite cylinder of circumference Ly (no 1/N terms, unlike a torus).  Enclosure by
    Collatz-Wielandt: for the positive matrix T and x = T^(2^k) 1 > 0, min_i (T x)_i / x_i <=
    lambda_0 <= max_i (T x)_i / x_i; k grows until the relative width is below 1e-12 (or
    max_squarings) at 128 bits: the sequence it feeds is a 1e-4-level consistency check."""
    from flint import arb, arb_mat
    from .ball import ball_of, baller
    t = GEOM[geom][0]
    if isinstance(Ly, bool) or not isinstance(Ly, int) or not 2 <= Ly or t * Ly > MAX_ROW:
        raise Refusal("precondition:size", f"Ly = {Ly!r}: need 2 <= Ly, {t} Ly <= {MAX_ROW}")
    with baller().hygiene.ctx_guard(prec=prec):
        Kb = [arb_q(k) * arb_q(Fraction(s)) for k, s in zip(Ks, _HOOKS["edge_scale"])]
        m, intra, inter, _ = _row_data(geom, Ly, (0, 0))
        n = 1 << m
        code_w = {}

        def w(a, b, c):
            if (a, b, c) not in code_w:
                code_w[(a, b, c)] = (a * Kb[0] + b * Kb[1] + c * Kb[2]).exp()
            return code_w[(a, b, c)]
        tot = inter + intra[:, None, :]
        T = arb_mat([[w(*(int(x) for x in tot[s, u])) for u in range(n)] for s in range(n)])
        P = T
        lo = hi = None
        for _ in range(max_squarings):
            P = P * P
            P = P / P[0, 0]                         # rescale (the ratios are scale-free)
            x = P * arb_mat([[1]] * n)
            y = T * x
            r = [y[i, 0] / x[i, 0] for i in range(n)]
            lo = min((v.lower() for v in r), key=lambda v: v.mid())
            hi = max((v.upper() for v in r), key=lambda v: v.mid())
            if hi - lo < arb("1e-12") * hi:
                break
        return ball_of(lo.union(hi).log() / (t * Ly))


@A.memo
def torus_free_energy(geom, Lx, Ly, Ks):
    """Ball of ln Z / n of the torus (Ks: three Fractions), by the transfer matrix."""
    t = _check(geom, Lx, Ly, MAX_ROW)
    n = t * Lx * Ly
    return certify(lambda: transfer_z(geom, Lx, Ly, [arb_q(k) for k in Ks]).log() / n)


# ---- route 3: exact integers (one coupling) -----------------------------------------------------
def _matmul(X, Y):
    Yt = list(zip(*Y))
    return [[sum(a * b for a, b in zip(r, col)) for col in Yt] for r in X]


def kronecker(geom, Lx, Ly, sector=(0, 0)):
    t = _check(geom, Lx, Ly, MAX_KRON_ROW)
    n, E = t * Lx * Ly, sum(class_sizes(geom, Lx, Ly))
    B = n + 1
    m, intra, inter, seam = _row_data(geom, Ly, sector)
    k_intra = sum(1 for e in GEOM[geom][1] if e[3] == 0) * Ly     # intra edges per row
    k_inter = sum(1 for e in GEOM[geom][1] if e[3] == 1) * Ly
    # unsatisfied bonds of a row pair: (number of edges - sum of signed products) / 2
    ui = (k_intra - intra.sum(axis=1)) // 2

    def mat(X):
        u = ui[:, None] + (k_inter - X.sum(axis=2)) // 2
        return [[1 << (B * int(u[s, v])) for v in range(1 << m)] for s in range(1 << m)]
    T, Ts = mat(inter), mat(seam)
    R, P, e = None, T, Lx - 1
    while e:
        if e & 1:
            R = P if R is None else _matmul(R, P)
        e >>= 1
        if e:
            P = _matmul(P, P)
    R = Ts if R is None else _matmul(R, Ts)
    z = sum(R[s][s] for s in range(1 << m))
    out = []
    for _ in range(E + 1):
        out.append(z & ((1 << B) - 1))
        z >>= B
    if z:
        raise Refusal("no-solution:audit", f"{geom} {Lx} x {Ly}: Kronecker digits overflow")
    return tuple(out)


def univariate(cnt, E):
    """The one-coupling polynomial (c_0 .. c_E) of a counts dict (E bonds)."""
    out = [0] * (E + 1)
    for k, m in cnt.items():
        out[sum(k)] += m
    return tuple(out)


# ---- the identities ----------------------------------------------------------------------------
def _poly_pow_pair(j, E, mob):
    """Coefficients of (q + p y)^j (s + r y)^(E - j)."""
    p, q, r, s = mob
    a = [comb(j, i) * p ** i * q ** (j - i) for i in range(j + 1)]
    b = [comb(E - j, i) * r ** i * s ** (E - j - i) for i in range(E - j + 1)]
    out = [0] * (E + 1)
    for i, x in enumerate(a):
        for k, y in enumerate(b):
            out[i + k] += x * y
    return out


@dataclass(frozen=True)
class DualityCheck:
    size: tuple
    exact: bool                 # four-sector identity holds coefficient by coefficient
    terms: int                  # nonzero monomials (of either side) compared
    periodic_only_mismatch: int  # coefficients where the periodic sector alone fails
    mobius: tuple


def duality_identity(Lx=3, Ly=3) -> DualityCheck:
    return _duality_identity(Lx, Ly, tuple(A._HOOKS["dual_mobius"]), _HOOKS["edge_scale"])


@lru_cache(maxsize=16)
def _duality_identity(Lx, Ly, mob, _scale):
    N, E = Lx * Ly, class_sizes("triangular", Lx, Ly)
    hc = counts("honeycomb_dual", Lx, Ly)
    # LHS[e] = sum_j n_j prod_c M_c[j_c, e_c], M_c[j, .] the coefficients of (q + p y)^j (s + r y)^(E_c - j):
    # three tensor contractions in exact Python integers (object arrays)
    T3 = np.zeros(tuple(e + 1 for e in E), dtype=object)
    T3[...] = 0
    for js, m in hc.items():
        T3[js] = m
    for c in range(3):
        M = np.array([_poly_pow_pair(j, E[c], mob) for j in range(E[c] + 1)], dtype=object)
        T3 = np.moveaxis(np.tensordot(T3, M, axes=([c], [0])), -1, c)
    lhs = {tuple(int(i) for i in k): int(v) for k, v in np.ndenumerate(T3) if v}
    scale = 2 ** (sum(E) - N - 1)
    rhs, per = {}, {}
    for a in (0, 1):
        for b in (0, 1):
            for js, m in counts("triangular", Lx, Ly, (a, b)).items():
                rhs[js] = rhs.get(js, 0) + scale * m
                if (a, b) == (0, 0):
                    per[js] = 4 * scale * m
    keys = set(lhs) | set(rhs)
    exact = all(lhs.get(k, 0) == rhs.get(k, 0) for k in keys)
    mism = sum(1 for k in set(lhs) | set(per) if lhs.get(k, 0) != per.get(k, 0))
    return DualityCheck((Lx, Ly), exact, len(keys), mism, mob)


def duality_identity_univariate(Lx, Ly) -> DualityCheck:
    """The same identity with one coupling (y_c = y), every polynomial from the integer transfer
    matrix (rows of 2 Ly <= 6 spins on the dual: Ly = 3), no enumeration."""
    mob = tuple(A._HOOKS["dual_mobius"])
    N, E = Lx * Ly, sum(class_sizes("triangular", Lx, Ly))
    hc = kronecker("honeycomb_dual", Lx, Ly)
    lhs = [0] * (E + 1)
    for j, m in enumerate(hc):
        if m:
            for e, x in enumerate(_poly_pow_pair(j, E, mob)):
                lhs[e] += m * x
    scale = 2 ** (E - N - 1)
    sec = {s: kronecker("triangular", Lx, Ly, s) for s in ((0, 0), (1, 0), (0, 1), (1, 1))}
    rhs = [scale * sum(p[e] for p in sec.values()) for e in range(E + 1)]
    per = [4 * scale * sec[(0, 0)][e] for e in range(E + 1)]
    return DualityCheck((Lx, Ly), lhs == rhs, E + 1, sum(1 for a, b in zip(lhs, per) if a != b), mob)


@A.memo
def star_triangle_check(Lx, Ly, L):
    """(Z_H(L), R^N Z_T(K_h = K3, K_v = K1, K_d = K2)) as Balls (L: three Fractions), both by the
    transfer matrix on the matched tori."""
    from .tri_honey import star_triangle
    N = Lx * Ly

    def zh():
        return transfer_z("honeycomb", Lx, Ly, [arb_q(x) for x in L])

    def zt():
        K1, K2, K3, R = star_triangle(*[arb_q(x) for x in L])
        return R ** N * transfer_z("triangular", Lx, Ly, [K3, K1, K2])
    return certify(zh), certify(zt)


__all__ = ["DualityCheck", "GEOM", "class_sizes", "counts", "duality_identity", "edges",
           "kronecker", "star_triangle_check", "torus_free_energy", "transfer_z", "univariate",
           "z_of_counts"]
