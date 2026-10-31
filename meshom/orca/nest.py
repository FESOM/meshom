# SPDX-License-Identifier: GPL-3.0-or-later
"""A finer ORCA nested in a box of an ORCA grid (ORCA12 in ORCA025; ORCA36 in ORCA12 for m16_orca's `nest.inner`,
B230), joined by a structured frame built in ORCA's own index space: the nest (decisions B172-B176; m16_orca `nest`).

Coordinates: fine F indices. The box is given by four coarse F lines (the fine box: columns c0..c1, rows r0..r1 of
the coarse grid); its edges are lines of both grids (a coarse F point (r, c) is the fine F point (k r + b_j, k c + b_i),
meshom.orca.mesh.nesting). The coarse box is the fine box widened by H / k coarse cells on each side. Local
coordinates (x, y) count fine steps from the coarse box's south-west corner: the coarse box is [0, Lx] x [0, Ly],
the fine box [H, Lx - H] x [H, Ly - H] (Lx, Ly and H multiples of k = 3).

- Rings (B172): n + 1 rectangles j = 0 .. n offset d_j from the coarse box (meshom.orca.ladder.rows: ln of the band
  height linear in d, 3 -> 1 fine steps; d_0 = 0, d_n = H). Ring 0 is the coarse grid's F points, ring n the fine
  grid's; band j lies between rings j and j + 1. Each side of a ring runs between two ring corners, which lie on the
  box's diagonals.
- Corners (B173): the tensor corner. In every band each corner holds one quad: the outer ring's corner, its two
  neighbours on the outer ring and the inner ring's corner (a square of side d_{j+1} - d_j in index space). The ring
  corners have valence 4; lines that meet the diagonal end there, one per band and side.
- Sides (B174): each side (local coordinate t along it from its first corner, depth d into the frame; the west and
  east sides are the south side turned by 90 deg) holds three parts. The middle is the ladder (B115-B116, not
  periodic): every coarse line t = 3 m with H < 3 m < L - H becomes the fine lines 3 m - 1, 3 m, 3 m + 1 by its two
  staggered Y-splits. The two end parts, between a ring corner and the ladder's first (last) line, follow the band
  rule: the first parent of a band ends at the inner ring's corner; the others go straight on (one child) or split
  (two children, a Y with one triangle), as many splits as the end part needs to keep the ladder's column density
  (the ladder's mean nodes per coarse column at that ring, times the end part's length); the splits go to the widest
  columns, never next to a split of the last GAP bands, and their arms open by min(w / 4, h tan THETA_END_DEG).
- Relaxation (B175): per side, the nodes of rings 1 .. n - 1 slide along their ring (the depth fixed: rings stay
  index lines) by least squares, as the ladder's (B115) with weights RELAX: widths change slowly along a ring, lines
  keep small tilts, the Y-split triangles keep their base. Corners and rings 0 and n are fixed.
- Counts: triangles = N_n - N_0 + 8 n (N_j = ring j's node count: each corner square takes two outer edges and gives
  no inner one), every triangle has one valence-5 node (its Y node), no node of valence 3.

On the sphere (`build`): rings 0 and n are the grids' own F points (exact, node for node); the other nodes are the
map (meshom.orca.generator.lattice) sampled along each ring every 1 / SUB fine steps and interpolated linearly.
"""
from dataclasses import dataclass, field

import numpy as np

from meshom.orca.ladder import SUB, THETA_DEG, _open, event_zones, rows

H_DEFAULT = 39          # B172: the frame's width in fine steps (13 ORCA025 cells, n = 24 rings for k = 3)
THETA_END_DEG = 15.0    # B174: an end part's split arm opens by h tan(theta) at most
GAP = 2                 # B174: an end part's split is never next to a line that split in the last GAP bands
RELAX = (1.0, 3.0, 3.0)     # B175: relaxation weights (widths along a ring, tilts, triangle bases)
CONTEXT_CELLS = 2       # B176: the ring of each part's own grid across a box edge that its clean-up sees as context
SMOOTH_W = 0.3          # B174: the 1-D smoothing of an end part's new ring (second differences against the targets)


@dataclass
class FrameTopo:
    """The frame in local fine index space (module docstring)."""
    x: np.ndarray               # (N,) local fine column
    y: np.ndarray               # (N,) local fine row
    ring: np.ndarray            # (N,)
    side: np.ndarray            # (N,) 0 S, 1 E, 2 N, 3 W; -1 a ring corner
    t: np.ndarray               # (N,) coordinate along the side (corners: nan)
    quads: np.ndarray           # (Q, 4) CCW (x east, y north)
    tris: np.ndarray            # (T, 3), the Y node first
    d: np.ndarray               # (n + 1,) ring offsets
    corners: np.ndarray         # (n + 1, 4) node ids of the ring corners (SW, SE, NE, NW)
    ring_seq: dict              # (j, s) -> node ids of side s of ring j, corners excluded, along t
    split_nodes: np.ndarray     # (T,) the Y nodes
    info: dict = field(default_factory=dict)


def _smooth(tgt: np.ndarray, ta: float, tb: float, w: float) -> np.ndarray:
    """1-D least squares: positions near tgt with small second differences, the ends ta, tb fixed."""
    m = len(tgt)
    if m == 0:
        return tgt
    D = np.zeros((m, m + 2))
    for i in range(m):
        D[i, i:i + 3] = (1.0, -2.0, 1.0)
    Df, Db = D[:, 1:-1], D[:, [0, -1]]
    return np.linalg.solve(np.eye(m) + w * Df.T @ Df, tgt - w * Df.T @ (Db @ np.array([ta, tb])))


def _end_part(t0: np.ndarray, d: np.ndarray, x_lad: list, dens: np.ndarray, a_open, gap: int = GAP,
              smooth_w: float = SMOOTH_W):
    """The band rule of one end part of a side (B174), in its own coordinate u (0 at the ring corner's side, growing
    towards the ladder): t0 the end part's nodes on ring 0 (u), d the ring offsets (the corner of ring j at u = d_j),
    x_lad[j] the u of the ladder's nearest node on ring j, dens[j] the column density (columns per fine step) wanted
    on ring j, a_open(j, u) an arm's largest opening in band j. Returns (pos, kids, split): pos[j] the end part's
    nodes on ring j (u, increasing), kids[j][p] the children of parent p of band j as indices into pos[j + 1] (-1 = the
    inner ring's corner), split[j] the parents that split."""
    n = len(d) - 1
    pos = [np.asarray(t0, float)]
    age = [np.full(len(t0), 99)]
    kids, split_l = [], []
    for j in range(n):
        tp, ag = pos[j], age[j]
        P = len(tp)
        last = j + 1 == n
        if last:                                      # the fine ring: the integer lines between its corner and the ladder
            tin = np.arange(np.ceil(d[n] - 1e-9) + 1.0, x_lad[n] - 0.5)
            C = len(tin)
        else:
            want = int(round((x_lad[j + 1] - d[j + 1]) * dens[j + 1])) - 1
            C = int(np.clip(want, max(P - 1, 2), min(2 * (P - 1), n - j)))
        S = C - (P - 1)
        if S < 0 or S > P - 1:
            raise ValueError(f"nest: an end part cannot reach its ring {j + 1} ({P} parents, {C} children)")
        ext = np.r_[d[j], tp, x_lad[j]]
        w = 0.5 * (ext[2:] - ext[:-2])
        near = np.minimum(ag, np.minimum(np.r_[99, ag[:-1]], np.r_[ag[1:], 99]))
        score = w - np.where(near < gap, 10.0 * (gap - near), 0.0)
        score[0] = -np.inf                            # the first parent ends at the corner
        sp = np.zeros(P, bool)
        order = np.argsort(-score, kind="stable")
        for adjacent in (False, True):                # never two splits side by side, unless there is no room
            for p in order:
                if sp.sum() >= S:
                    break
                if p == 0 or sp[p] or (not adjacent and ((p > 0 and sp[p - 1]) or (p < P - 1 and sp[p + 1]))):
                    continue
                sp[p] = True
        ch, tgt, m = [[-1]], [], 0
        for p in range(1, P):
            if sp[p]:
                a = min(w[p] / 4.0, a_open(j, tp[p]))
                tgt += [tp[p] - a, tp[p] + a]
                ch.append([m, m + 1])
                m += 2
            else:
                tgt.append(tp[p])
                ch.append([m])
                m += 1
        new = tin if last else _smooth(np.array(tgt), d[j + 1], x_lad[j + 1], smooth_w)
        if np.any(np.diff(np.r_[d[j + 1], new, x_lad[j + 1]]) <= 0):
            raise ValueError(f"nest: an end part's ring {j + 1} is not ordered")
        na = np.empty(len(new), np.int64)
        for p, cl in enumerate(ch):
            for c in cl:
                if c >= 0:
                    na[c] = 0 if len(cl) == 2 else min(ag[p] + 1, 99)
        pos.append(new)
        age.append(na)
        kids.append(ch)
        split_l.append(np.flatnonzero(sp))
    return pos, kids, split_l


def _ladder_part(L: int, H: int, d: np.ndarray, a_open, k: int = 3):
    """The ladder of one side (B115-B116 without the wrap): the coarse lines t = 3 m, m = H/3 + 1 .. (L - H)/3 - 1.
    Returns (pos, keys, kids, tri, info): pos[j] (t, increasing) and keys[j] of its nodes on ring j, kids[j][p] the
    children of node p of ring j as indices into ring j + 1, tri[j] (parent index, left, right) of its splits."""
    n = len(d) - 1
    hk = np.diff(d)
    ms = np.arange(H // k + 1, (L - H) // k)
    if not len(ms):
        raise ValueError(f"nest: a side of {L} fine steps leaves no room for the ladder between frames of {H}")
    z1, z2, p1, p2, m1, m2, (P1, P2), (s1, s2) = event_zones(hk, len(ms))
    k1 = z1[p1[ms % P1]]
    k2 = z2[p2[ms % P2]]
    ka, kb = z1[-1] + 2, z2[0]
    tt = np.clip((np.arange(n + 1) - ka) / max(kb - ka, 1), 0.0, 1.0)
    drift = -0.25 * tt * tt * (3.0 - 2.0 * tt)
    X = k * ms.astype(float)
    a1e1 = np.array([a_open(int(k1[i]), X[i]) for i in range(len(ms))])
    a1e2 = np.array([a_open(int(k2[i]), X[i] + 0.5) for i in range(len(ms))])
    pos, keys = [], []
    for j in range(n + 1):
        pj, kj = [], []
        for i in range(len(ms)):
            e1, e2 = k1[i], k2[i]
            if j <= e1:
                kj += [("L", i)]
                pj += [X[i]]
            elif j <= e2:
                o = _open(0.75, a1e1[i], j - e1, e2 - e1)
                kj += [("A-", i), ("A+", i)]
                pj += [X[i] - o + drift[j], X[i] + o + drift[j]]
            else:
                o2 = _open(0.5, a1e2[i], j - e2, n - e2)
                kj += [("A-", i), ("B-", i), ("B+", i)]
                pj += [X[i] - 0.75 + drift[j], X[i] + 0.5 - o2, X[i] + 0.5 + o2]
        pos.append(np.array(pj))
        keys.append(kj)
    kids, tri = [], []
    for j in range(n):
        idx = {kv: a for a, kv in enumerate(keys[j + 1])}
        kj, tj = [], []
        for p, (kind, i) in enumerate(keys[j]):
            if kind == "L":
                cl = [idx[("L", i)]] if j < k1[i] else [idx[("A-", i)], idx[("A+", i)]]
            elif kind == "A+" and j == k2[i]:
                cl = [idx[("B-", i)], idx[("B+", i)]]
            else:
                cl = [idx[(kind, i)]]
            kj.append(cl)
            if len(cl) == 2:
                tj.append((p, cl[0], cl[1]))
        kids.append(kj)
        tri.append(tj)
    info = {"lines": int(len(ms)), "zone1_rows": [int(z1[0]), int(z1[-1])], "zone2_rows": [int(z2[0]), int(z2[-1])],
            "periods": [int(P1), int(P2)], "zone_steps": [int(s1), int(s2)], "stagger_m": [int(m1), int(m2)],
            "drift_rows": [int(ka), int(kb)]}
    # the column density of the ladder on each ring (nodes per fine step): 1 + the share of lines past each event
    f1 = np.array([(z1 < j).mean() for j in range(n + 1)])
    f2 = np.array([(z2 < j).mean() for j in range(n + 1)])
    return pos, keys, kids, tri, info, (1.0 + f1 + f2) / k


def _side(L: int, H: int, d: np.ndarray, a_open, k: int = 3) -> dict:
    """One side of the frame (B174): per ring its nodes along t (corners excluded) and per band the children of
    every parent (-1 / -2: the inner ring's first / last corner) and the triangles."""
    n = len(d) - 1
    lpos, _, lkids, ltri, linfo, dens = _ladder_part(L, H, d, a_open, k)
    t0 = k * np.arange(1, H // k + 1, dtype=float)                       # the end parts' coarse lines on ring 0
    f = np.tan(np.radians(THETA_END_DEG)) / np.tan(np.radians(THETA_DEG))    # the end parts' smaller cap
    wpos, wkids, wsplit = _end_part(t0, d, [p[0] for p in lpos], dens, lambda j, u: f * a_open(j, u))
    epos_u, ekids, esplit = _end_part(t0, d, [L - p[-1] for p in lpos], dens, lambda j, u: f * a_open(j, L - u))
    seq, kids, tris, base = [], [], [], []
    for j in range(n + 1):
        seq.append(np.r_[wpos[j], lpos[j], (L - epos_u[j])[::-1]])
    for j in range(n):
        nw, nl = len(wpos[j + 1]), len(lpos[j + 1])
        ne = len(epos_u[j + 1])
        kj, tj = [], []
        for p, cl in enumerate(wkids[j]):                                # west: -1 = the start corner
            kj.append([c if c >= 0 else -1 for c in cl])
        for cl in lkids[j]:
            kj.append([nw + c for c in cl])
        for cl in ekids[j][::-1]:                                        # east, reversed: u -> t
            kj.append([nw + nl + (ne - 1 - c) if c >= 0 else -2 for c in cl[::-1]])
        for p, cl in enumerate(kj):
            if len(cl) == 2:
                tj.append((p, cl[0], cl[1]))
        kids.append(kj)
        tris.append(tj)
    nsplit = sum(len(s) for s in wsplit) + sum(len(s) for s in esplit)
    return {"seq": seq, "kids": kids, "tris": tris, "ladder": linfo, "end_splits": int(nsplit),
            "ladder_splits": int(sum(len(t) for t in ltri))}


def _to_xy(s: int, t, dd, Lx: int, Ly: int):
    """Side-local (t, depth) -> local (x, y) (rotations: orientation kept)."""
    t, dd = np.asarray(t, float), np.asarray(dd, float)
    if s == 0:
        return t, dd
    if s == 1:
        return Lx - dd, t
    if s == 2:
        return Lx - t, Ly - dd
    return dd, Ly - t


def relax_side(t: np.ndarray, d: np.ndarray, seqs: list, kids: list, tris: list, corner_t: list,
               w=RELAX, base: list | None = None) -> np.ndarray:
    """(N,) new t of one side's nodes (B175): rings 1 .. n - 1 free, least squares of alpha (second differences along a
    ring, its corners as fixed ends) + beta (child - parent along every line, the corners' children included) + gamma
    (each Y triangle's base minus its nominal base), each over the band's height. seqs[j]: node ids of ring j;
    kids[j][p]: global ids of parent p's children (a corner given as ('c', t)); tris[j]: (left, right) ids; corner_t[j]
    = (t of the start corner, t of the end corner); base[j]: nominal bases (None: the current ones)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    alpha, beta, gamma = w
    n = len(d) - 1
    hk = np.diff(d)
    free_ids = np.concatenate([seqs[j] for j in range(1, n)]) if n > 1 else np.zeros(0, np.int64)
    col = {int(v): i for i, v in enumerate(free_ids)}
    R, C, V, B = [], [], [], []
    r = 0

    def term(nodes, coefs, const, wt):
        nonlocal r
        sw = np.sqrt(wt)
        rhs = -const
        for nd, cf in zip(nodes, coefs):
            if isinstance(nd, tuple):                 # a fixed corner value
                rhs -= cf * nd[1]
            elif int(nd) in col:
                R.append(r)
                C.append(col[int(nd)])
                V.append(sw * cf)
            else:
                rhs -= cf * t[nd]
        B.append(sw * rhs)
        r += 1

    for j in range(1, n):
        h = hk[j]
        full = [("c", corner_t[j][0])] + list(seqs[j]) + [("c", corner_t[j][1])]
        for a in range(len(full) - 2):
            term(full[a:a + 3], (1.0 / h, -2.0 / h, 1.0 / h), 0.0, alpha)
    for j in range(n):
        h = hk[j]
        for p, cl in zip(seqs[j], kids[j]):
            for c in cl:
                term([c, p], (1.0 / h, -1.0 / h), 0.0, beta)
        for i, (lft, rgt) in enumerate(tris[j]):
            b0 = base[j][i] if base is not None else t[rgt] - t[lft]
            term([rgt, lft], (1.0 / h, -1.0 / h), -b0 / h, gamma)
    out = t.copy()
    if not len(free_ids):
        return out
    A = coo_matrix((V, (R, C)), shape=(r, len(free_ids))).tocsc()
    out[free_ids] = spsolve((A.T @ A).tocsc(), A.T @ np.array(B))
    return out


def topology(Lx: int, Ly: int, H: int = H_DEFAULT, a_open=None, k: int = 3, relax_w=RELAX) -> FrameTopo:
    """The frame in local fine index space (module docstring). a_open(s, j, t) -> an arm's first-row opening (fine
    steps) on side s, band j, at t (None: the band's height times tan THETA_DEG for the ladder, THETA_END_DEG for
    the end parts' cap)."""
    if k != 3:
        raise ValueError("nest: only a refinement by 3 (ORCA025 -> ORCA12)")
    if H % k or Lx % k or Ly % k:
        raise ValueError(f"nest: Lx {Lx}, Ly {Ly} and H {H} must be multiples of {k}")
    d = rows(0.0, float(H), k)
    d[0], d[-1] = 0.0, float(H)                                          # the two lattices exactly
    n = len(d) - 1
    hk = np.diff(d)
    if a_open is None:
        a_open = lambda s, j, t: hk[min(j, n - 1)] * np.tan(np.radians(THETA_DEG))   # noqa: E731
    xs, ys, rg, sd, ts = [], [], [], [], []

    def add(x, y, j, s, tv):
        xs.append(float(x)); ys.append(float(y)); rg.append(j); sd.append(s); ts.append(tv)   # noqa: E702
        return len(xs) - 1

    corners = np.zeros((n + 1, 4), np.int64)
    for j in range(n + 1):
        dj = d[j]
        for c, (cx, cy) in enumerate(((dj, dj), (Lx - dj, dj), (Lx - dj, Ly - dj), (dj, Ly - dj))):
            corners[j, c] = add(cx, cy, j, -1, np.nan)
    sides, ring_seq = [], {}
    quads, tris, split = [], [], []
    for s in range(4):
        L = Lx if s % 2 == 0 else Ly
        sd_ = _side(L, H, d, lambda j, t, s=s: a_open(s, j, t), k)
        ids = []
        for j in range(n + 1):
            x, y = _to_xy(s, sd_["seq"][j], np.full(len(sd_["seq"][j]), d[j]), Lx, Ly)
            ids.append(np.array([add(x[i], y[i], j, s, sd_["seq"][j][i]) for i in range(len(x))], np.int64))
            ring_seq[(j, s)] = ids[j]
        sd_["ids"] = ids
        sides.append(sd_)
    t = np.array(ts, float)
    for s in range(4):
        sd_, ids = sides[s], sides[s]["ids"]
        L = Lx if s % 2 == 0 else Ly
        c0, c1 = s, (s + 1) % 4
        gk, gt, base = [], [], []
        for j in range(n):
            kj = []
            for cl in sd_["kids"][j]:
                kj.append([corners[j + 1, c0] if c == -1 else corners[j + 1, c1] if c == -2 else ids[j + 1][c]
                           for c in cl])
            gk.append(kj)
            gt.append([(ids[j + 1][lft], ids[j + 1][rgt]) for _, lft, rgt in sd_["tris"][j]])
        if relax_w:
            ct = [(d[j], L - d[j]) for j in range(n + 1)]
            kk = [[[("c", d[j + 1]) if c == corners[j + 1, c0] else ("c", L - d[j + 1]) if c == corners[j + 1, c1]
                    else c for c in cl] for cl in gk[j]] for j in range(n)]
            t = relax_side(t, d, ids, kk, gt, ct, relax_w)
        for j in range(n):
            par = ids[j]
            for a in range(len(par) - 1):
                quads.append((par[a], par[a + 1], gk[j][a + 1][0], gk[j][a][-1]))
            for p, lft, rgt in sd_["tris"][j]:
                tris.append((par[p], ids[j + 1][rgt], ids[j + 1][lft]))
                split.append(par[p])
    x, y = np.array(xs), np.array(ys)
    sdv, rgv = np.array(sd), np.array(rg)
    for s in range(4):                                                   # positions from the relaxed t
        sel = sdv == s
        x[sel], y[sel] = _to_xy(s, t[sel], d[rgv[sel]], Lx, Ly)
    for j in range(n):                                                   # the corner squares
        for c in range(4):
            sp, sn = (c - 1) % 4, c
            quads.append((ring_seq[(j, sp)][-1], corners[j, c], ring_seq[(j, sn)][0], corners[j + 1, c]))
    N = [sum(len(ring_seq[(j, s)]) for s in range(4)) + 4 for j in range(n + 1)]
    info = {"n": int(n), "H": int(H), "Lx": int(Lx), "Ly": int(Ly), "k": int(k),
            "band_first_last": [float(hk[0]), float(hk[-1])], "ring_nodes_first_last": [int(N[0]), int(N[-1])],
            "tris_expected": int(N[-1] - N[0] + 8 * n), "ladder": sides[0]["ladder"],
            "ladder_splits": int(sum(sd_["ladder_splits"] for sd_ in sides)),
            "end_splits": int(sum(sd_["end_splits"] for sd_ in sides)), "relax": list(relax_w) if relax_w else None}
    return FrameTopo(x, y, rgv, sdv, t, np.array(quads, np.int64), np.array(tris, np.int64).reshape(-1, 3), d,
                     corners, ring_seq, np.array(split, np.int64), info)


def index_quality(ft: FrameTopo, sx: float = 1.0, sy: float = 1.0, corner: bool | None = None) -> dict:
    """Angles, size ratio and valences of the frame in index space (x scaled by sx, y by sy). corner: None = every
    element, True = the elements whose centroid lies in a corner square ([0, H]^2 and its mirrors), False = the rest."""
    from meshom.mesh import Mesh, build_topology

    P = np.c_[ft.x * sx, ft.y * sy]
    H, Lx, Ly = ft.info["H"], ft.info["Lx"], ft.info["Ly"]

    def ang(c):
        q = P[c]
        a = np.roll(q, -1, axis=1) - q
        b = np.roll(q, 1, axis=1) - q
        cr = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        return np.degrees(np.mod(np.arctan2(cr, (a * b).sum(-1)), 2 * np.pi))

    def inq(c):
        cx, cy = ft.x[c].mean(axis=1), ft.y[c].mean(axis=1)
        return ((cx < H) | (cx > Lx - H)) & ((cy < H) | (cy > Ly - H))

    qa, ta = ang(ft.quads), ang(ft.tris)
    qe = np.linalg.norm(P[np.roll(ft.quads, -1, axis=1)] - P[ft.quads], axis=-1)
    te = np.linalg.norm(P[np.roll(ft.tris, -1, axis=1)] - P[ft.tris], axis=-1)
    top = build_topology(Mesh(np.zeros((len(ft.x), 3)), ft.tris, ft.quads, np.zeros(len(ft.x), np.int8)))
    me = np.r_[te.mean(axis=1), qe.mean(axis=1)]
    ee = top.edge_elems[top.edge_elems[:, 1] >= 0]
    sel_e = np.ones(len(me), bool)
    if corner is not None:
        sel_e = np.r_[inq(ft.tris), inq(ft.quads)] == corner
    ee = ee[sel_e[ee[:, 0]] & sel_e[ee[:, 1]]]
    ratio = np.maximum(me[ee[:, 0]], me[ee[:, 1]]) / np.minimum(me[ee[:, 0]], me[ee[:, 1]])
    T = len(ft.tris)
    qsel, tsel = sel_e[T:], sel_e[:T]
    qm = qa[qsel].max(axis=1)
    val = np.bincount(top.edges.ravel(), minlength=len(ft.x))
    interior = ~top.boundary_node
    return {"quads": int(qsel.sum()), "tris": int(tsel.sum()), "reflex_quads": int((qa[qsel] >= 180).any(axis=1).sum()),
            "quad_max_p99": float(np.percentile(qm, 99)), "quad_max_p99_9": float(np.percentile(qm, 99.9)),
            "quad_max": float(qm.max()), "quads_over_110_pct": float(100 * (qm > 110).mean()),
            "quad_min": float(qa[qsel].min()),
            "tri_min": float(ta[tsel].min()) if tsel.any() else None,
            "tri_max": float(ta[tsel].max()) if tsel.any() else None,
            "ratio_p99": float(np.percentile(ratio, 99)), "ratio_max": float(ratio.max()),
            "valence_interior": {int(v): int(c) for v, c in zip(*np.unique(val[interior], return_counts=True))}}


# ---- on the sphere ---------------------------------------------------------------------------------------------

def box_from_lonlat(g, lonlat_box) -> tuple[int, int, int, int]:
    """(c0, c1, r0, r1) the fine box as coarse F lines from a lon-lat box [lon0, lat0, lon1, lat1] (degrees; lon0 >
    lon1 crosses the antimeridian): the smallest box of F lines that holds every cell whose T point lies in the
    lon-lat box (columns unwrapped: c0 < c1, c1 may pass the column count)."""
    lon0, lat0, lon1, lat1 = map(float, lonlat_box)
    lo, la = g.lon["t"][:g.j_pole - 1, :-2], g.lat["t"][:g.j_pole - 1, :-2]
    ni = lo.shape[1]
    dl = np.mod(lo - lon0, 360.0)
    inside = (dl <= np.mod(lon1 - lon0, 360.0)) & (la >= lat0) & (la <= lat1)
    rr, cc = np.nonzero(inside)
    if not len(rr):
        raise ValueError(f"nest: no T point lies in {lonlat_box}")
    cols = np.unique(cc)                                           # the columns, unwrapped after their largest gap
    gaps = np.diff(np.r_[cols, cols[0] + ni])                      # (also across the grid's own column seam)
    start = int(cols[(int(np.argmax(gaps)) + 1) % len(cols)])
    cu = start + np.mod(cc - start, ni)
    return int(cu.min() - 1), int(cu.max()), int(rr.min() - 1), int(rr.max())


def boxes(box, H: int, k: int, nest: dict) -> dict:
    """The fine box (coarse F lines, unwrapped columns), the coarse box (H / k coarse cells wider on every side), the
    local frame's size and origin in fine F indices (unwrapped)."""
    c0, c1, r0, r1 = map(int, box)
    w = H // k
    Ca, Cb, Ra, Rb = c0 - w, c1 + w, r0 - w, r1 + w
    b_i, b_j = nest["col_offset"], nest["row_offset"]
    return {"fine_box": [c0, c1, r0, r1], "coarse_box": [Ca, Cb, Ra, Rb], "Lx": k * (Cb - Ca), "Ly": k * (Rb - Ra),
            "X0": k * Ca + b_i, "Y0": k * Ra + b_j}


def _line_samples(gf, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """(len(Y), len(X), 3) the fine map at the fine F indices X (columns, wrapped here) x Y (rows)."""
    from meshom.geo import lonlat_to_xyz
    from meshom.orca.generator import lattice, row_map

    p, c, q = gf.params, gf.coef, gf.q
    nf = gf.lon["f"].shape[1] - 2
    zi = gf.info["i_offset"] + (np.mod(np.asarray(X, float), nf) + 0.5) / q
    jj = gf.info["j_offset"] + (np.asarray(Y, float) + 0.5) / q
    lon, lat = lattice(p, c, row_map(p, c), {"f": zi}, {"f": jj})
    return lonlat_to_xyz(lon["f"].ravel(), lat["f"].ravel()).reshape(len(jj), len(zi), 3)


def _metric(gc, k: int, b_i: int, b_j: int):
    """(sx, sy) per fine step at fine F index (X, Y): the coarse grid's F spacing around the nearest coarse F point
    over k (chord lengths on the unit sphere)."""
    from meshom.geo import lonlat_to_xyz

    lon, lat = gc.lon["f"][:, :-2], gc.lat["f"][:, :-2]
    nc = lon.shape[1]

    def f(X, Y):
        r = int(round((Y - b_j) / k))
        c = int(round((X - b_i) / k))
        p = lonlat_to_xyz(lon[[r, r, r + 1], [c % nc, (c + 1) % nc, c % nc]],
                          lat[[r, r, r + 1], [c % nc, (c + 1) % nc, c % nc]])
        return np.linalg.norm(p[1] - p[0]) / k, np.linalg.norm(p[2] - p[0]) / k
    return f


def build(gc, gf, box, nest: dict, H: int = H_DEFAULT, theta_deg: float = THETA_DEG, relax_w=RELAX,
          sub: int = SUB):
    """The frame between the coarse grid gc and its refinement gf (nest: meshom.orca.mesh.nesting, k = 3) around the
    fine box `box` = (c0, c1, r0, r1) coarse F lines, as a Mesh on the sphere (module docstring), its topology and
    record. An arm's first-row opening tilts it by theta_deg on the sphere (the map's local metric)."""
    from meshom.geo import lonlat_to_xyz
    from meshom.mesh import Mesh

    k, b_i, b_j = nest["k"], nest["col_offset"], nest["row_offset"]
    bx = boxes(box, H, k, nest)
    Lx, Ly, X0, Y0 = bx["Lx"], bx["Ly"], bx["X0"], bx["Y0"]
    nc = gc.lon["f"].shape[1] - 2
    nf = gf.lon["f"].shape[1] - 2
    Ca, Cb, Ra, Rb = bx["coarse_box"]
    if Ra < 1 or Rb > gc.j_pole - 3 or Cb - Ca >= nc:
        raise ValueError(f"nest: the coarse box {bx['coarse_box']} must lie below the fold and span less than the globe")
    metric = _metric(gc, k, b_i, b_j)
    tan_t = np.tan(np.radians(theta_deg))
    d = rows(0.0, float(H), k)
    d[0], d[-1] = 0.0, float(H)
    hk = np.diff(d)
    n = len(d) - 1

    def a_open(s, j, t):
        x, y = _to_xy(s, t, d[min(j, n - 1)], Lx, Ly)
        sx, sy = metric(X0 + float(x), Y0 + float(y))
        m = sy / sx if s % 2 == 0 else sx / sy                    # across the side over along it
        return hk[min(j, n - 1)] * m * tan_t

    ft = topology(Lx, Ly, H, a_open, k, relax_w)
    pts = np.full((len(ft.x), 3), np.nan)
    on0, onn = ft.ring == 0, ft.ring == n
    X = X0 + ft.x
    Y = Y0 + ft.y
    r = np.round((Y[on0] - b_j) / k).astype(np.int64)
    c = np.round((X[on0] - b_i) / k).astype(np.int64) % nc
    pts[on0] = lonlat_to_xyz(gc.lon["f"][r, c], gc.lat["f"][r, c])
    yi = np.round(Y[onn]).astype(np.int64)
    xi = np.round(X[onn]).astype(np.int64) % nf
    pts[onn] = lonlat_to_xyz(gf.lon["f"][yi, xi], gf.lat["f"][yi, xi])
    mid = ~on0 & ~onn
    rows_sn = (ft.side % 2 == 0) | (ft.side < 0)                  # S / N sides and the corners: on rows
    for along_x, sel in ((True, mid & rows_sn), (False, mid & ~rows_sn)):
        if not sel.any():
            continue
        if along_x:
            lines = np.unique(ft.y[sel])
            s = np.arange(Lx * sub + 1) / sub
            S = _line_samples(gf, X0 + s, Y0 + lines)                # (lines, samples, 3)
            li = np.searchsorted(lines, ft.y[sel])
            u = ft.x[sel] * sub
        else:
            lines = np.unique(ft.x[sel])
            s = np.arange(Ly * sub + 1) / sub
            S = np.transpose(_line_samples(gf, X0 + lines, Y0 + s), (1, 0, 2))
            li = np.searchsorted(lines, ft.x[sel])
            u = ft.y[sel] * sub
        j = np.clip(np.floor(u).astype(np.int64), 0, S.shape[1] - 2)
        w = (u - j)[:, None]
        p = (1.0 - w) * S[li, j] + w * S[li, j + 1]
        pts[sel] = p / np.linalg.norm(p, axis=1, keepdims=True)
    mesh = Mesh(pts, ft.tris, ft.quads, np.zeros(len(pts), np.int8))
    q = index_quality(ft)
    info = dict(ft.info, **bx, theta_deg=float(theta_deg), sub=int(sub), nodes=int(mesh.n_nodes),
                quads=int(mesh.n_quads), tris=int(mesh.n_tris), ring0_nodes=int(on0.sum()), ringn_nodes=int(onn.sum()),
                index_quality={kq: (round(v, 3) if isinstance(v, float) else v) for kq, v in q.items()})
    return mesh, ft, info
