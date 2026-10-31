"""One structured block between two fixed join chains and two sliding banks (decision B161).

Logical frame: i along the strait (row 0 = join A, row M = join B), k across (k = 0 on the right bank "S", k = N_i
on the left bank "N"). Steps:

1. Harmonic map: a fine Winslow (inverse Laplace) lattice of the region, the join sides Dirichlet (the host
   chains, parametrised by node index: node k of a chain of n edges at eta = k / n), the banks sliding (every
   SLIDE_EVERY sweeps a bank node moves halfway to the foot of its first interior neighbour on the bank polyline:
   grid lines meet the coast at right angles); three levels (4, 2, 1); the row count of the coarse level by
   ASPECT_ROUNDS rounds of M <- M x (mean along / across aspect), so cells are square on average.
2. Size target along the block: h_req = row length / N_neck, limited at slope G_LAYOUT along the centreline, capped
   by the host size at the joins (linear between them in xi), so the size ratio at a join is ~1 when the room rule
   holds.
3. Rows at equal steps of the integral of ds / h, at least as many as the count changes need; counts N_i = round(row
   length / h) with hysteresis HYST, N_0 and N_M the chains' edge counts, a row-to-row change of at most
   max(1, BIRTH_FRAC N_i).
4. Topology: between two rows of different counts the shorter row splits |dN| interior nodes (a Y-split: one node
   joined to two of the next row, one triangle, one valence-5 node); split columns at golden-ratio positions, no
   column or its neighbours split again within GAP_ROWS rows (the ORCA ladder's stagger idea, B116).
5. Across: eta per node by least squares (spacing 1 / N_i along a row, column lines continuous, a split's children
   at -/+ half a spacing); nodes = the harmonic map at (xi_i, eta); join rows = the host nodes exactly.
6. Size floor (B165): the harmonic map compresses cells at a cape (both ways, to 0.4-0.6 of the target at Tarifa and
   at the Westfjords), so a target >= h_min does not give cells >= h_min. With a cell floor, every round measures
   the cells (mean great-circle edge length) and, in each row interval holding a cell below FLOOR_MARGIN x the
   floor, raises the target by the deficit (a per-row floor on the target, its slope limited to G_LAYOUT, never above
   the host size at the joins: step 2's cap), then lays the rows out again (steps 3-5), at most FLOOR_ROUNDS rounds.
   The rows near a cape get fewer, larger cells; the caller checks the result and refuses a cell still below the
   floor.
7. Add columns (B207, an option per side): on a side with a set join count the count target is held only at or above the
   smaller of the neck's and the join's counts (not between them, B200), and the rounds of step 6 also lower the
   target (a per-row ceiling, its slope limited to the side's slope) in each row interval holding a cell above
   ADD_COLUMNS_CAP x the host size, at most ADD_COLUMNS_ROUNDS rounds (the floor wins over it): where the harmonic
   map fans out (the outer bank of a bend, a convex bank corner) the rows get more columns and more rows, and the
   counts come back to the join's count before the join (the birth rule of step 3).
8. Split corners (B208, an option): each corner triangle of a fixed row (B180, B206) is split at the midpoint of its
   diagonal into two triangles, and the midpoint takes the corner node's place in the row the block is built on
   (step 1's Dirichlet chain, the zipped row 0 / M): the block's first row meets the row's full count of segments,
   the diagonal halves, not the diagonals. Join margin (B209, an option): step 2's cap at the joins is that factor
   times the host size, so the outermost rows are a little finer than the host.
"""
from dataclasses import dataclass, field

import numba
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

from meshom.insert.geom import Polyline

SLIDE_EVERY = 10             # B161: sweeps between two bank slides
SWEEPS = (3000, 2000, 1500)  # B161: Winslow sweeps on the levels 4, 2, 1
ASPECT_ROUNDS = 4            # B161: rounds of the row-count iteration on the coarse level
FINE_PER_CELL = 8            # B161: fine lattice columns per cell of the widest row (at least FINE_MIN)
FINE_MIN = 64
FINE_MAX_NODES = 1_500_000   # B161: cap of the fine lattice (columns reduced to keep under it)
G_LAYOUT = 0.12              # B161: slope of the size target along the block (dh/ds; per-cell ratio ~ e^0.12)
BIRTH_FRAC = 0.2             # B161: at most max(1, BIRTH_FRAC N_i) column births / deaths between two rows
HYST = 0.75                  # B161: a row's count changes only when its target moved by more than this
NECK_BAND = 0.5              # B205: rows whose own count target is within this of the narrowest row's keep it unsmoothed
SMOOTH_ROWS = 3.0            # B161: Gaussian (in rows) on the count target before rounding
GAP_ROWS = 2                 # B161 / B116: a column and its neighbours do not split again within this many rows
GOLDEN = 0.6180339887
CORNER_TURN_DEG = 60.0       # B180: a fixed row's node is a convex bend (corner triangle) when it turns this much ...
CORNER_W = 2                 # B180: ... between the chords over this many edges before and after it
FLOOR_ROUNDS = 10            # B165: layout rounds that raise the target where the map compresses cells below the floor
FLOOR_MARGIN = 1.02          # B165: the rounds aim at cells >= this x the floor (the final check has no tolerance)
ADD_COLUMNS_CAP = 1.1        # B207: with add_columns, rounds lower the target where a cell exceeds this x the host size
ADD_COLUMNS_ROUNDS = 6       # B207: at most this many layout rounds for it


@numba.njit(cache=True)
def _sweeps(X, n):
    """n Gauss-Seidel Winslow sweeps on the interior of X (M+1, N+1, 2), uniform computational spacing."""
    M, N = X.shape[0] - 1, X.shape[1] - 1
    for _ in range(n):
        for i in range(1, M):
            for j in range(1, N):
                ax = 0.5 * (X[i, j + 1, 0] - X[i, j - 1, 0])
                ay = 0.5 * (X[i, j + 1, 1] - X[i, j - 1, 1])
                bx = 0.5 * (X[i + 1, j, 0] - X[i - 1, j, 0])
                by = 0.5 * (X[i + 1, j, 1] - X[i - 1, j, 1])
                al = ax * ax + ay * ay
                ga = bx * bx + by * by
                be = ax * bx + ay * by
                den = 2.0 * (al + ga)
                if den <= 0.0:
                    continue
                for c in range(2):
                    cr = 0.25 * (X[i + 1, j + 1, c] - X[i + 1, j - 1, c] - X[i - 1, j + 1, c] + X[i - 1, j - 1, c])
                    X[i, j, c] = (al * (X[i - 1, j, c] + X[i + 1, j, c]) + ga * (X[i, j - 1, c] + X[i, j + 1, c])
                                  - 2.0 * be * cr) / den


def tfi(W, E, S, Nn):
    """Transfinite interpolation. W, E (N+1,2) at i = 0, M; S, Nn (M+1,2) at k = 0, N."""
    M, N = len(S) - 1, len(W) - 1
    u = np.linspace(0, 1, M + 1)[:, None, None]
    v = np.linspace(0, 1, N + 1)[None, :, None]
    return ((1 - u) * W[None] + u * E[None] + (1 - v) * S[:, None] + v * Nn[:, None]
            - (1 - u) * (1 - v) * S[0] - u * (1 - v) * S[-1] - (1 - u) * v * Nn[0] - u * v * Nn[-1])


def winslow(X, bS: Polyline, bN: Polyline, n_iter: int):
    X = np.ascontiguousarray(X, dtype=np.float64)
    N = X.shape[1] - 1
    done = 0
    while done < n_iter:
        k = min(SLIDE_EVERY, n_iter - done)
        _sweeps(X, k)
        done += k
        for pl, j, jn in ((bS, 0, 1), (bN, N, N - 1)):
            s_old = pl.project(X[1:-1, j])
            s_new = pl.project(X[1:-1, jn])
            s = np.maximum.accumulate(np.clip(s_old + 0.5 * (s_new - s_old), 1e-6 * pl.L, pl.L * (1 - 1e-6)))
            X[1:-1, j] = pl.at(s)
    return X


def resample_chain(xy, nf):
    """Chain of n+1 nodes -> nf+1 points at the index parameter t = k n / nf (node k at t = k), linear between."""
    n = len(xy) - 1
    t = np.linspace(0, n, nf + 1)
    k = np.minimum(t.astype(int), n - 1)
    f = (t - k)[:, None]
    return (1 - f) * xy[k] + f * xy[k + 1]


def cell_aspect(X):
    a = 0.5 * (np.linalg.norm(X[1:, :-1] - X[:-1, :-1], axis=-1) + np.linalg.norm(X[1:, 1:] - X[:-1, 1:], axis=-1))
    b = 0.5 * (np.linalg.norm(X[:-1, 1:] - X[:-1, :-1], axis=-1) + np.linalg.norm(X[1:, 1:] - X[1:, :-1], axis=-1))
    return float(np.exp(np.log(np.maximum(a, 1e-9) / np.maximum(b, 1e-9)).mean()))


def harmonic_map(xyA, xyB, bS: Polyline, bN: Polyline, n_cells: int):
    """Fine Winslow lattice on [0,1]^2 (module docstring, step 1): (X (Mf+1, Nf+1, 2), interpolator, info)."""
    nA, nB = len(xyA) - 1, len(xyB) - 1
    Nf = max(FINE_PER_CELL * max(nA, nB, n_cells), FINE_MIN)
    LA = np.linalg.norm(np.diff(xyA, axis=0), axis=1).sum()
    LB = np.linalg.norm(np.diff(xyB, axis=0), axis=1).sum()
    ratio = (bS.L + bN.L) / max(LA + LB, 1e-9)
    while Nf > FINE_MIN and Nf * Nf * ratio > FINE_MAX_NODES:
        Nf = int(Nf * 0.8)
    Nf = 4 * int(np.ceil(Nf / 4))
    levels = (4, 2, 1)
    X, rounds = None, []
    Ml = max(int(round(Nf / 4 * ratio)), 2)
    Mf = None
    for lev, it in zip(levels, SWEEPS):
        Nl = Nf // lev
        W, E = resample_chain(xyA, Nl), resample_chain(xyB, Nl)
        if X is None:
            for _ in range(ASPECT_ROUNDS):
                X = winslow(tfi(W, E, bS.at(np.linspace(0, bS.L, Ml + 1)), bN.at(np.linspace(0, bN.L, Ml + 1))),
                            bS, bN, it)
                asp = cell_aspect(X)
                rounds.append((Ml, asp))
                Mn = max(2, int(round(Ml * asp)))
                Mn = min(Mn, max(2, int(FINE_MAX_NODES / (Nf + 1) / lev)))
                if Mn == Ml:
                    break
                Ml = Mn
            Mf = Ml * lev
        else:
            Mp, Np = X.shape[0] - 1, X.shape[1] - 1
            ip = RegularGridInterpolator((np.linspace(0, 1, Mp + 1), np.linspace(0, 1, Np + 1)), X)
            Ml = Mf // lev
            g = np.stack(np.meshgrid(np.linspace(0, 1, Ml + 1), np.linspace(0, 1, Nl + 1), indexing="ij"), -1)
            X = ip(g.reshape(-1, 2)).reshape(Ml + 1, Nl + 1, 2)
            X[0], X[-1] = W, E
            X = winslow(X, bS, bN, it)
    Mf, Nf = X.shape[0] - 1, X.shape[1] - 1
    phi = RegularGridInterpolator((np.linspace(0, 1, Mf + 1), np.linspace(0, 1, Nf + 1)), X)
    return X, phi, {"fine": [int(Mf + 1), int(Nf + 1)], "aspect_rounds": [[int(m), float(a)] for m, a in rounds]}


def side_slopes(L, g) -> np.ndarray:
    """B201: the size slope per fine row: g, or a pair (side A, side B) split at the narrowest row."""
    if np.ndim(g) == 0:
        return np.full(len(L), float(g))
    i_n = int(np.argmin(L))
    return np.where(np.arange(len(L)) <= i_n, float(g[0]), float(g[1]))


def size_target(X, n_neck, h_cap_a, h_cap_b, g=G_LAYOUT, h_min=0.0):
    """Along the fine lattice: (h (Mf+1,), row length L, centreline arc s, row-mean step ds) (module docstring,
    step 2); never below h_min (the insert's size floor, B165). g: the slope, or (side A, side B) (B201)."""
    L = np.linalg.norm(np.diff(X, axis=1), axis=-1).sum(1)
    ds = np.linalg.norm(np.diff(X, axis=0), axis=-1).mean(1)
    s = np.r_[0, np.cumsum(ds)]
    hreq = L / n_neck
    gr = side_slopes(L, g)
    h = np.min(hreq[None, :] + gr[:, None] * np.abs(s[:, None] - s[None, :]), axis=1)
    xi = np.linspace(0, 1, len(L))
    return np.maximum(np.minimum(h, h_cap_a + (h_cap_b - h_cap_a) * xi), h_min), L, s, ds


def birth_rows(a: int, b: int) -> int:
    """Rows needed to go from a cells across to b with at most max(1, BIRTH_FRAC N) births or deaths per row."""
    k = 0
    while a != b:
        lim = max(1, int(BIRTH_FRAC * a))
        a += int(np.clip(b - a, -lim, lim))
        k += 1
    return k


def monotone_counts(raw: np.ndarray, nA: int, nB: int, sides=(False, False), i_n: int | None = None,
                    add_columns=(False, False)) -> np.ndarray:
    """B200: the count target of a side with a set join count runs monotonically from the neck (row i_n, the
    shortest row; default the row with the fewest cells) to the join's count: clipped to between the two and made
    monotone outward from the neck, so equal counts only widen the rows (no columns born and killed again).
    add_columns (side A, side B; B207): on that side the target is only held at or above the smaller of the two
    counts, so it rises above both where the rows are long and comes back to the join's count before the join."""
    raw = raw.copy()
    i_n = int(np.argmin(raw)) if i_n is None else int(i_n)
    rn = float(raw[i_n])
    for on, wd, idx, nj in ((sides[0], add_columns[0], np.arange(i_n, -1, -1), nA),
                            (sides[1], add_columns[1], np.arange(i_n, len(raw)), nB)):
        if not on:
            continue
        lo, hi = min(rn, nj), max(rn, nj)
        if wd:
            raw[idx] = np.maximum(raw[idx], lo)
            continue
        v = np.clip(raw[idx], lo, hi)
        v = np.maximum.accumulate(v) if nj >= rn else np.minimum.accumulate(v)
        raw[idx] = v
    return raw


def layout(h, L, ds, nA, nB, fixed_count: bool = False, monotone=(False, False), rows: int | None = None,
           add_columns=(False, False)):
    """Rows xi_i and counts N_i (module docstring, step 3). The row count is at least what the birth rule needs to
    go from the join A count down to the fewest cells the target asks for and up to the join B count (with a
    floor-raised target, Denmark's default-floor block got 3 rows for 17 -> 14 -> 17 -> 21 cells, its
    Y-splits crowded and a quad folded at 350 deg). rows: the row count asked for (B204; the birth rule may add
    rows), else by the size target, at least 2. add_columns: (side A, side B) of monotone_counts (B207)."""
    Mf = len(L) - 1
    xi_f = np.linspace(0, 1, Mf + 1)
    q = np.r_[0, np.cumsum(ds / (0.5 * (h[1:] + h[:-1])))]
    M = max(2, int(round(q[-1]))) if rows is None else max(1, int(rows))
    if not fixed_count:
        n_low = int(np.clip(np.round(np.min(L / h)), 2, min(nA, nB)))
        M = max(M, birth_rows(nA, n_low) + birth_rows(n_low, nB))
    xi = np.interp(np.linspace(0, q[-1], M + 1), q, xi_f)
    if fixed_count:
        return xi, np.full(M + 1, nA, dtype=int)
    Lr, hr = np.interp(xi, xi_f, L), np.interp(xi, xi_f, h)
    raw = gaussian_filter1d(Lr / hr, SMOOTH_ROWS, mode="nearest") if SMOOTH_ROWS else Lr / hr
    unsm = Lr / hr                                # B205: the smoothing must not lift the count at the neck: the
    i_short = int(np.argmin(Lr))                  # narrowest row and its neighbours whose own count is within
    neck = np.zeros(len(unsm), bool)              # NECK_BAND of it keep their own (unsmoothed) count, no hysteresis
    lo = hi = i_short
    while lo > 0 and unsm[lo - 1] < unsm[i_short] + NECK_BAND:
        lo -= 1
    while hi < len(unsm) - 1 and unsm[hi + 1] < unsm[i_short] + NECK_BAND:
        hi += 1
    neck[lo:hi + 1] = True
    raw = np.where(neck, np.minimum(raw, unsm), raw)
    if any(monotone):
        raw = monotone_counts(raw, nA, nB, monotone, int(np.argmin(Lr)), add_columns)
    raw[0], raw[-1] = nA, nB
    want = np.empty(M + 1, int)
    want[0] = nA
    for i in range(1, M + 1):
        keep = abs(raw[i] - want[i - 1]) <= HYST and not neck[i]           # B205: no hysteresis at the neck
        want[i] = want[i - 1] if keep else int(np.round(raw[i]))
    want = np.maximum(2, want)
    want[-1] = nB
    N = want.copy()
    for _ in range(50):
        old = N.copy()
        for i in range(1, M + 1):
            lim = max(1, int(BIRTH_FRAC * N[i - 1]))
            N[i] = np.clip(N[i], N[i - 1] - lim, N[i - 1] + lim)
        N[-1] = nB
        for i in range(M - 1, -1, -1):
            lim = max(1, int(BIRTH_FRAC * N[i + 1]))
            N[i] = np.clip(N[i], N[i + 1] - lim, N[i + 1] + lim)
        N[0] = nA
        if (N == old).all():
            break
    N = np.maximum(N, 1)
    return xi, N


def bend_corners(xy: np.ndarray, region_xy: np.ndarray, deg: float | None = None, w: int | None = None,
                 gap: int = 3) -> list:
    """B180: the convex bends of a fixed row (a join chain that turns a corner: an L-join, a strip's or a corridor's
    host chain): nodes where the chain turns toward the block's side by at least CORNER_TURN_DEG, measured between
    the chords over CORNER_W edges before and after the node (a host staircase alternates +-90 deg turns node by
    node; over 2 edges it does not turn); local maxima at least 3 nodes apart, never the two end nodes. The block's
    side: the sign of the chain's total turning when it turns by more than 45 deg, else the side of region_xy (a
    point inside the block); with w < CORNER_W the total turning is taken between the chords over CORNER_W edges at
    both ends, so a staircase whose first and last single steps differ does not read as a turn (B206). gap: the least
    index distance of two bends (B206: 2 with w = 1, every step of a staircase). Returns the node indices."""
    deg = CORNER_TURN_DEG if deg is None else deg
    w = CORNER_W if w is None else w
    n = len(xy) - 1
    if n < 2 * w:
        return []
    ws = min(CORNER_W, n) if w < CORNER_W else 1
    d = xy[ws:] - xy[:-ws]
    th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    total = float(th[-1] - th[0])
    if abs(total) > np.radians(45.0):
        side = np.sign(total)
    else:
        mid = xy[n // 2]
        c = xy[-1] - xy[0]
        side = np.sign(c[0] * (region_xy[1] - mid[1]) - c[1] * (region_xy[0] - mid[0]))
    turn = np.zeros(n + 1)
    for k in range(1, n):
        a, b = xy[k] - xy[max(k - w, 0)], xy[min(k + w, n)] - xy[k]
        turn[k] = np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], (a * b).sum())) * side
    out = []
    for k in np.argsort(-turn):
        if turn[k] < deg or k in (0, n):
            break
        if all(abs(int(k) - j) >= gap for j in out):
            out.append(int(k))
    return sorted(out)


def _virtual(ids: np.ndarray, corners, first: bool = False) -> tuple[np.ndarray, list]:
    """A fixed row without its corner nodes (B180) and the corner triangles cut off at them; on row 0 (first) the
    block lies on the other side of the row, so its triangles run the other way (one orientation, B186)."""
    keep = np.ones(len(ids), bool)
    tris = []
    for k in corners:
        keep[k] = False
        tris.append([ids[k + 1], ids[k], ids[k - 1]] if first else [ids[k - 1], ids[k], ids[k + 1]])
    return ids[keep], tris


def split_chain(xy: np.ndarray, corners) -> np.ndarray:
    """B208: a fixed row with each corner node replaced by the midpoint of the chord between its two neighbours
    (corners at least 2 nodes apart, never an end node)."""
    out = np.array(xy, dtype=np.float64, copy=True)
    for k in corners:
        out[k] = 0.5 * (xy[k - 1] + xy[k + 1])
    return out


def split_corner_tris(ids: np.ndarray, corners, first: bool, nxt: int) -> tuple[list, list]:
    """B208: the two triangles of each split corner of a fixed row (ids: the row's node ids, the corner's place
    holding the midpoint node; the corner node itself gets the new id nxt, nxt + 1, ...), oriented as _virtual's
    corner triangle. Returns (triangles, corner node ids)."""
    tris, cn = [], []
    for j, k in enumerate(corners):
        c, m, a, b = nxt + j, ids[k], ids[k - 1], ids[k + 1]
        tris += [[b, c, m], [m, c, a]] if first else [[a, c, m], [m, c, b]]
        cn.append(c)
    return tris, cn


def topology(N, corners_a=(), corners_b=()):
    """(row node ids, quads, tris, vertical links) in labels (module docstring, step 4). corners_a / corners_b: node
    indices of row 0 / row M that are convex bends (B180): the corner node is cut off by a triangle (its two
    neighbours and itself), and the row without it is zipped with the next row as usual (its count one less per
    corner)."""
    M = len(N) - 1
    ids, c = [], 0
    for i in range(M + 1):
        ids.append(np.arange(c, c + N[i] + 1))
        c += N[i] + 1
    rows = list(ids)
    pos = [np.arange(N[i] + 1) for i in range(M + 1)]          # index in the real row of each zipped node
    corner_tris = []
    for i, cs in ((0, corners_a), (M, corners_b)):
        if len(cs):
            rows[i], t = _virtual(ids[i], cs, first=i == 0)
            keep = np.ones(N[i] + 1, bool)
            keep[list(cs)] = False
            pos[i] = np.flatnonzero(keep)
            corner_tris += t
    quads, tris, vert, recent = [], [], [], []
    t = 0
    for i in range(M):
        a, b = len(rows[i]) - 1, len(rows[i + 1]) - 1
        flip = b < a
        lo, hi = (b, a) if flip else (a, b)
        m = hi - lo
        bad = set()
        for r in recent[-GAP_ROWS:]:
            for kk in r:
                bad.update({kk - 1, kk, kk + 1})
        valid = list(range(1, lo)) if lo >= 2 else list(range(0, lo + 1))
        assert m <= len(valid), f"row {i}: {m} splits on a row of {lo} cells"
        cand, tries = [], 0
        while len(cand) < m:
            u = (0.37 + GOLDEN * t) % 1.0
            t += 1
            tries += 1
            k = valid[min(int(u * len(valid)), len(valid) - 1)]
            if k in cand or (k in bad and tries <= 2000):
                continue
            cand.append(k)
        split = np.zeros(lo + 1, bool)
        split[cand] = True
        f = np.arange(lo + 1) + np.r_[0, np.cumsum(split)[:-1]]
        rs, rb = (rows[i + 1], rows[i]) if flip else (rows[i], rows[i + 1])
        ps, pb = (pos[i + 1], pos[i]) if flip else (pos[i], pos[i + 1])
        for cc in range(lo):
            left = f[cc] + (1 if split[cc] else 0)
            right = f[cc + 1]
            q = [rs[cc], rs[cc + 1], rb[right], rb[left]]
            quads.append([q[3], q[0], q[1], q[2]] if flip else [q[0], q[3], q[2], q[1]])
        for k in range(lo + 1):
            if split[k]:
                tri = [rs[k], rb[f[k]], rb[f[k] + 1]]
                tris.append([tri[0], tri[2], tri[1]] if flip else tri)
        for k in range(lo + 1):
            kids = [f[k], f[k] + 1] if split[k] else [f[k]]
            for j, kb in enumerate(kids):
                off = 0.0 if len(kids) == 1 else (-0.5 if j == 0 else 0.5)
                vert.append((i + 1, int(ps[k]), i, int(pb[kb]), off) if flip else (i, int(ps[k]), i + 1, int(pb[kb]),
                                                                                    off))
        recent.append(cand)
    tris += corner_tris
    return ids, np.array(quads, dtype=np.int64).reshape(-1, 4), np.array(tris, dtype=np.int64).reshape(-1, 3), vert


def eta_solve(N, vert, etaA, etaB):
    """Least-squares eta per node (module docstring, step 5); rows 0 and M fixed, banks 0 and 1."""
    M = len(N) - 1
    off = np.r_[0, np.cumsum(np.asarray(N) + 1)]
    n = off[-1]
    fixed = np.full(n, np.nan)
    for i in range(M + 1):
        fixed[off[i]] = 0.0
        fixed[off[i] + N[i]] = 1.0
    fixed[off[0]:off[0] + N[0] + 1] = etaA
    fixed[off[M]:off[M] + N[M] + 1] = etaB
    free = np.isnan(fixed)
    col = -np.ones(n, int)
    col[free] = np.arange(free.sum())
    rows, cols, vals, rhs = [], [], [], []
    r = 0

    def add(terms, b):
        nonlocal r
        for node, cf in terms:
            if free[node]:
                rows.append(r)
                cols.append(col[node])
                vals.append(cf)
            else:
                b -= cf * fixed[node]
        rhs.append(b)
        r += 1

    for i in range(M + 1):
        for k in range(N[i]):
            add([(off[i] + k + 1, 1.0), (off[i] + k, -1.0)], 1.0 / N[i])
    for (i, k, j, kb, o) in vert:
        add([(off[j] + kb, 1.0), (off[i] + k, -1.0)], o / max(N[j], N[i]))
    eta = fixed.copy()
    if free.any():
        A = coo_matrix((vals, (rows, cols)), shape=(r, int(free.sum()))).tocsr()
        eta[free] = lsqr(A, np.array(rhs), atol=1e-12, btol=1e-12, iter_lim=20000)[0]
    return eta, off


ZIP_TRI_DEG = 30.0           # B181: in a zip a triangle costs as much as a quad's corner this far above 90 deg


def _qmax(p):
    a = np.roll(p, -1, 0) - p
    b = np.roll(p, 1, 0) - p
    ang = np.degrees(np.mod(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(-1)), 2 * np.pi))
    ang = np.where(ang > 180, 360 - ang, ang) if (ang > 180).sum() >= 2 else ang
    return float(ang.max())


def zip_rows(ea, eb, ia, ib, xy=None):
    """B181: the cells between two rows by their parameter (ea, eb increasing; ia, ib node ids): a strip of
    triangles advancing on the row whose next node has the smaller parameter, then neighbouring triangles that
    advanced on different rows merged into quads, chosen by dynamic programming: a quad costs its largest corner
    above 90 deg (in the plane, xy given), a triangle left alone ZIP_TRI_DEG (equal, aligned counts give quads only;
    an extra node of one row leaves one triangle, a Y-split)."""
    n, m = len(ea) - 1, len(eb) - 1
    i = j = 0
    seq = []
    while i < n or j < m:
        if i < n and (j == m or ea[i + 1] <= eb[j + 1]):
            seq.append(("a", i, j))
            i += 1
        else:
            seq.append(("b", i, j))
            j += 1
    L = len(seq)

    def quad(k):
        t, i, j = seq[k]
        return [ia[i], ia[i + 1], ib[j + 1], ib[j]]

    best = np.zeros(L + 2)
    take = np.zeros(L + 1, bool)
    for k in range(L - 1, -1, -1):
        best[k] = best[k + 1] + ZIP_TRI_DEG
        if k + 1 < L and seq[k + 1][0] != seq[k][0]:
            if xy is None:
                c = 0.0
            else:
                c = max(0.0, _qmax(xy[quad(k)]) - 90.0)
            if c + best[k + 2] < best[k]:
                best[k] = c + best[k + 2]
                take[k] = True
    quads, tris = [], []
    k = 0
    while k < L:
        if take[k]:
            quads.append(quad(k))
            k += 2
            continue
        t, i, j = seq[k]
        tris.append([ia[i], ia[i + 1], ib[j]] if t == "a" else [ia[i], ib[j + 1], ib[j]])
        k += 1
    return quads, tris


@dataclass
class Block:
    xy: np.ndarray               # (n,2) chart coordinates
    quads: np.ndarray            # (Q,4) local ids
    tris: np.ndarray             # (T,3) local ids
    row: np.ndarray              # (n,) row index i of each node
    off: np.ndarray              # row offsets (row i = off[i] .. off[i] + N[i])
    N: np.ndarray                # (M+1,) cells per row
    xi: np.ndarray               # (M+1,) row positions in the parameter domain
    fine_xy: np.ndarray          # (Mf+1, Nf+1, 2) the harmonic map lattice
    fine_h: np.ndarray           # (Mf+1,) the size target per fine row (m, chart)
    info: dict = field(default_factory=dict)
    join_a: np.ndarray | None = None   # B208: ids of join A's host nodes in chain order (None: row 0)
    join_b: np.ndarray | None = None   # B208: ids of join B's host nodes in chain order (None: row M)

    def joins(self) -> tuple[np.ndarray, np.ndarray]:
        """Local ids of the two join chains' host nodes (chain order)."""
        M = self.M
        ia = np.arange(self.off[0], self.off[0] + self.N[0] + 1) if self.join_a is None else self.join_a
        ib = np.arange(self.off[M], self.off[M] + self.N[M] + 1) if self.join_b is None else self.join_b
        return ia, ib

    @property
    def M(self) -> int:
        return len(self.N) - 1


def cell_sizes(xy, tris, quads, scale=None) -> np.ndarray:
    """Mean edge length per cell (tris first, then quads), in sphere metres: chart length / scale at the edge's
    midpoint (scale None: chart metres)."""
    out = []
    for c in (tris, quads):
        if not len(c):
            out.append(np.zeros(0))
            continue
        p = xy[c]
        q = np.roll(p, -1, axis=1)
        e = np.linalg.norm(q - p, axis=-1)
        if scale is not None:
            e = e / np.asarray(scale(0.5 * (p + q).reshape(-1, 2))).reshape(e.shape)
        out.append(e.mean(1))
    return np.concatenate(out)


def _upper_envelope(f, s, g):
    """max over j of f_j - g |s - s_j| (a floor profile whose slope along s is at most g; g per row allowed)."""
    g = np.broadcast_to(np.asarray(g, dtype=float), s.shape)
    return np.max(f[None, :] - g[:, None] * np.abs(s[:, None] - s[None, :]), axis=1)


def build(xyA, xyB, bS: Polyline, bN: Polyline, n_neck: int, fixed_count: bool = False, h_min: float = 0.0,
          cell_floor: float = 0.0, scale=None, corners=False, growth=G_LAYOUT, monotone=(False, False),
          rows: int | None = None, steps=(False, False), add_columns=False, split=False, margin: float = 1.0) -> Block:
    """The block of the module docstring: xyA, xyB (n+1,2) the join chains from bank S to bank N (exact host node
    positions), bS / bN the banks from join A to join B. h_min: the size target's floor (m); cell_floor: the floor of
    the cells' mean great-circle edge length (m, step 6; 0 = none); scale: the chart's scale (chart m per sphere m)
    for both (None: chart metres). corners: cut the convex bends of the fixed rows by corner triangles (B180): a bool
    for both rows or a pair (row A, row B). growth: the size target's slope, or (side A, side B) (B201); monotone:
    (side A, side B) counts run monotonically from the neck to the join's count (B200); rows: the row count (B204;
    None: by the size target); steps: (row A, row B) cut every convex step of a staircase row, not only its bends
    over CORNER_W edges (B206; with corners on); add_columns: a bool for both sides or a pair (side A, side B): on that
    side the counts are not held between the neck's and the join's (B200) and the layout rounds lower the target
    where a cell exceeds ADD_COLUMNS_CAP x the host size (step 7, B207); split: a bool for both rows or a pair (row
    A, row B): split the corner triangles at their diagonal's midpoint, a node of the block's row (step 8, B208);
    margin: the cap at the joins in host sizes (step 8, B209)."""
    nA, nB = len(xyA) - 1, len(xyB) - 1
    add_columns = add_columns if isinstance(add_columns, tuple) else (bool(add_columns), bool(add_columns))
    LA = np.linalg.norm(np.diff(xyA, axis=0), axis=1).sum()
    LB = np.linalg.norm(np.diff(xyB, axis=0), axis=1).sum()
    ca_on, cb_on = corners if isinstance(corners, tuple) else (corners, corners)
    sa, sb = steps
    cA = (bend_corners(xyA, xyB.mean(0), w=1, gap=2) if sa else bend_corners(xyA, xyB.mean(0))) if ca_on else []
    cB = (bend_corners(xyB, xyA.mean(0), w=1, gap=2) if sb else bend_corners(xyB, xyA.mean(0))) if cb_on else []
    spa, spb = split if isinstance(split, tuple) else (split, split)
    sA, sB = (cA if spa else []), (cB if spb else [])        # B208: the split corners
    tA, tB = ([] if spa else cA), ([] if spb else cB)        # the corners cut by one triangle (B180)
    xyA_r = split_chain(xyA, sA) if len(sA) else xyA         # B208: the rows the block is built on
    xyB_r = split_chain(xyB, sB) if len(sB) else xyB
    nA_v, nB_v = nA - len(tA), nB - len(tB)                  # the rows zipped next to them (B180)
    X, phi, info = harmonic_map(xyA_r, xyB_r, bS, bN, n_neck)
    sc_row = np.asarray(scale(X.reshape(-1, 2))).reshape(X.shape[:2]).max(1) if scale is not None else 1.0
    h0, L, s, ds = size_target(X, n_neck, margin * LA / nA, margin * LB / nB, g=growth, h_min=h_min * sc_row)
    g_rows = side_slopes(L, growth)
    Mf = len(L) - 1
    xi_f = np.linspace(0, 1, Mf + 1)
    host_f = LA / nA + (LB / nB - LA / nA) * xi_f              # the host size at the joins, linear between them
    cap = margin * host_f                                    # step 2's cap (B209: margin x the host size)
    floor = np.zeros(Mf + 1)
    ceil = np.full(Mf + 1, np.inf)                           # B207: the target's ceiling from the add_columns rounds
    i_nf = int(np.argmin(L))
    wide = np.zeros(Mf + 1, bool)                            # B207: the fine rows of a side with add_columns
    wide[:i_nf + 1] |= add_columns[0]
    wide[i_nf:] |= add_columns[1]
    rounds = max(FLOOR_ROUNDS if cell_floor > 0 else 1, ADD_COLUMNS_ROUNDS if wide.any() else 1)
    for rnd in range(rounds):
        h = np.maximum(np.maximum(np.minimum(h0, ceil), h_min * sc_row),
                       np.minimum(floor, cap))               # never coarser than the host (then the caller refuses)
        xi, N = layout(h, L, ds, nA_v, nB_v, fixed_count=fixed_count and nA == nB, monotone=monotone, rows=rows,
                       add_columns=add_columns)
        N = N.copy()
        N[0], N[-1] = nA, nB
        row = np.concatenate([np.full(N[i] + 1, i) for i in range(len(N))])
        M = len(N) - 1
        ids, quads, tris, vert = topology(N, tA, tB)
        eta, off = eta_solve(N, vert, np.arange(nA + 1) / nA, np.arange(nB + 1) / nB)
        xy = phi(np.c_[np.clip(xi[row], 0, 1), np.clip(eta, 0, 1)])
        xy[off[0]:off[0] + nA + 1] = xyA_r
        xy[off[M]:off[M] + nB + 1] = xyB_r
        join_a = join_b = None
        if len(sA) or len(sB):                               # B208: the corner nodes appended, two triangles each
            ta, na = split_corner_tris(ids[0], sA, True, len(xy))
            tb, nb = split_corner_tris(ids[M], sB, False, len(xy) + len(sA))
            xy = np.vstack([xy, xyA[sA].reshape(-1, 2), xyB[sB].reshape(-1, 2)])
            row = np.r_[row, np.zeros(len(sA), int), np.full(len(sB), M)]
            tris = np.vstack([tris, np.array(ta + tb, dtype=np.int64).reshape(-1, 3)])
            join_a, join_b = ids[0].copy(), ids[M].copy()
            join_a[list(sA)] = na
            join_b[list(sB)] = nb
        if cell_floor <= 0 and not wide.any():
            break
        sz = cell_sizes(xy, tris, quads, scale)
        rc = np.r_[row[tris].min(1) if len(tris) else np.zeros(0, int), row[quads].min(1)]
        if cell_floor > 0:
            info["cell_min_m"] = float(sz.min())
            info["floor_rounds"] = rnd
        low = cell_floor > 0 and sz.min() < FLOOR_MARGIN * cell_floor
        excess = np.zeros(M)                                 # B207: per row interval, largest cell / its cap
        if wide.any():
            xm = 0.5 * (xi[1:] + xi[:-1])
            host = np.interp(xm, xi_f, host_f) / np.interp(xm, xi_f, np.broadcast_to(sc_row, xi_f.shape))
            np.maximum.at(excess, np.minimum(rc, M - 1), sz / (ADD_COLUMNS_CAP * host[np.minimum(rc, M - 1)]))
            excess[~np.interp(xm, xi_f, wide.astype(float)).astype(bool)] = 0.0
            info["add_columns_rounds"] = rnd
            info["cell_max_host"] = float(ADD_COLUMNS_CAP * excess.max()) if excess.max() > 0 else None
        more = excess.max() > 1.0 and rnd < ADD_COLUMNS_ROUNDS - 1
        if not (low or more) or rnd == rounds - 1:
            break
        if low:
            deficit = np.zeros(M)
            np.maximum.at(deficit, np.minimum(rc, M - 1), FLOOR_MARGIN * cell_floor / sz)
            need = np.zeros(Mf + 1)
            for i in np.flatnonzero(deficit > 1.0):        # the fine rows of row interval i, one fine row beyond
                lo = max(int(np.searchsorted(xi_f, xi[i])) - 1, 0)
                hi = min(int(np.searchsorted(xi_f, xi[i + 1])) + 1, Mf)
                need[lo:hi + 1] = np.maximum(need[lo:hi + 1], h[lo:hi + 1] * deficit[i])
            floor = _upper_envelope(np.maximum(floor, need), s, g_rows)
        if more:                                           # B207: the same rows, the target lowered by the excess
            top = np.full(Mf + 1, -np.inf)
            for i in np.flatnonzero(excess > 1.0):
                lo = max(int(np.searchsorted(xi_f, xi[i])) - 1, 0)
                hi = min(int(np.searchsorted(xi_f, xi[i + 1])) + 1, Mf)
                top[lo:hi + 1] = np.maximum(top[lo:hi + 1], -h[lo:hi + 1] / excess[i])
            ceil = np.minimum(ceil, -_upper_envelope(top, s, g_rows))
    info.update(rows=int(M), counts=[int(x) for x in N], n_neck_target=int(n_neck),
                h_min_m=float(h.min()), h_max_m=float(h.max()), neck_row_len_m=float(L.min()),
                y_splits=int(len(tris) - len(tA) - len(tB) - 2 * (len(sA) + len(sB))),
                corners=[[int(k) for k in cA], [int(k) for k in cB]], split=[bool(len(sA)), bool(len(sB))],
                margin=float(margin),
                floor_raised=bool(floor.max() > 0), ceiling_lowered=bool(np.isfinite(ceil).any()),
                growth=[float(g) for g in np.atleast_1d(growth)], monotone=[bool(m) for m in monotone],
                add_columns=[bool(w) for w in add_columns])
    return Block(xy, quads, tris, row, off, N, xi, X, h, info, join_a, join_b)
