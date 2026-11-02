"""Group inserts (kind = "group", decisions B240-B249, B252): several channels, the basin and junction patches
between them and their islands, in one envelope of host cells.

The envelope is a polygon: the host cells whose centroid lies inside it are the region R, replaced as a whole. Every
step works in the stereographic chart at the envelope's centre:

1. R's outer boundary: runs of join edges (host edges between R and the host cells outside it: the joins, kept node
   for node, as many as there are) and runs of host coast between them; host islands inside R are allowed (their
   coast is replaced like the rest);
2. the new outer boundary: each coast run replaced by the real coast between the two joins' end nodes
   (RealCoast.arc, blended onto the end nodes over BLEND_H join cells, as a strait's banks, B164); a passage that the
   host closes opens when its real water lies inside the envelope;
3. islands: the closed real coast loops inside the new boundary. Smaller than island_km2: dropped (water); then one
   closer than gap_km to the coast or to another island is joined to it (the water between them closed, B241);
   the others are holes;
4. channels: each a structured block (meshom.insert.block) between two cross lines snapped onto the coast, its count
   of cells across held from one cross line to the other (no triangle inside a block, B242); its banks are the coast
   between the cross lines' ends;
5. patches: the water outside the blocks, traced from the pieces (joins, coast pieces, the cross lines, which a patch
   and a block share node for node), each meshed by m02 (quad-dominant: packing along a cross field, Delaunay,
   recombination; its loop nodes kept exactly) at the size GroupH (B243; each fine sample's slope reaches the host's
   size at every join node, B252);
6. blocks and patches merged by node positions and put in place of R (meshom.insert.assemble.replace_part: the join
   nodes keep their host ids, every host cell outside R stays);
7. after the splice and fix_rings, the area pass on the patches, the blocks and two host rings (even_group,
   meshom.insert.even, B252), the obtuse patch quads split (split_obtuse, B243) and the pass again.

The depth of a group: the insert's own rule on its nodes (meshom.insert.core.insert_depth), then each channel's own
sill (branch_depth, B244), or, when the group names its routes (the straits through it, from basin to basin), the
channel raise, the dam and the section areas along each route across blocks and patches (route_depth, B246).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from meshom.geo import R_EARTH_M, xyz_to_lonlat
from meshom.insert.geom import Chart, Polyline, point_in_polygon
from meshom.insert.region import MIN_CELLS_ACROSS, InsertError
from meshom.mesh import Mesh, build_topology, signed_area

GAP_JOIN_CELLS = 5           # B241: islands closer than this many floor cells (h_min) join the land by default
GAP_CELLS = 3                # B243: every passage between two boundary pieces gets at least this many fill cells
                             # across (3 at a narrowest where 5 would force small cells)
ISLAND_KM2 = 10.0            # B241: islands smaller than this are dropped (water) unless asked otherwise
GROWTH = 0.12                # B243: the size slope of the fill (as the block's G_LAYOUT)
GROWTH_MAX = 0.5             # B252: the steepest slope a fine sample's cone takes to reach the host's size at the joins
BRIDGE_REACH = 1.5           # B241: the coast cut off by a joined island: its points within the island's distance
                             # + half gap_km of it, at most this x gap_km
ISLAND_SIGMA = 0.03          # B241: an island's coast is smoothed by at most this fraction of its perimeter (the bank
                             # smoothing of the local cell size shrank a 198 km^2 island to 67 km^2)
ROUTE_END_H = 2.0            # B246: a route's end: the deepest box mean within this many h_fine of the given point
MERGE_CHORD = 1e-9           # the parts' shared nodes are matched within this chord (unit sphere; ~6 mm)
FILL_OPTS = {"price": 1.0 / 3.0, "recombine": "exact", "size_iters": 30, "angle_iters": 3}   # B243: m02 on patches
FILL_FLOOR = 1.2             # B256: the patches' size floor in h_min (m02's cells come out up to ~10 % below their
                             # target where the water is narrow, fix_rings shrinks some by a few %; the floor check
                             # stays at h_min)


@dataclass
class Channel:
    name: str
    a: np.ndarray            # (2,3) unit vectors: the cross line at end A (bank to bank)
    b: np.ndarray            # (2,3): the cross line at end B
    cells: int = 5           # cells across, held from A to B


@dataclass
class Piece:
    """A piece of the new boundary: a join (host node ids), a coast polyline (chart), an island loop, a cross line."""
    kind: str                            # "join" | "coast" | "island" | "cross"
    xy: np.ndarray                       # chart points (a join: its host nodes; an island: closed, first not repeated)
    ids: np.ndarray | None = None        # a join's host node ids
    start: tuple | None = None           # key of the first point (None: a closed loop)
    end: tuple | None = None
    bank: str = ""                       # "c<k>S" / "c<k>N": a block's bank (not a patch's)
    nodes: np.ndarray | None = None      # the mesh nodes (xyz) of the piece, from start to end (closed: all)


def _shoelace(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def directed_edges(mesh: Mesh, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The cells' edges as (a, b) with the cell on the left (cells taken counter-clockwise on the sphere), and the
    element of each; mask (T+Q,) selects the cells."""
    T = mesh.n_tris
    sa = signed_area(mesh)
    out, el = [], []
    for c, off in ((mesh.tris, 0), (mesh.quads, T)):
        if not len(c):
            continue
        idx = np.arange(len(c)) if mask is None else np.flatnonzero(mask[off:off + len(c)])
        cc = c[idx]
        neg = sa[off + idx] < 0
        cc = np.where(neg[:, None], cc[:, ::-1], cc)
        out.append(np.stack([cc, np.roll(cc, -1, axis=1)], -1).reshape(-1, 2))
        el.append(np.repeat(off + idx, c.shape[1]))
    return (np.vstack(out) if out else np.zeros((0, 2), np.int64)), (np.concatenate(el) if el else np.zeros(0, int))


def envelope_region(win, chart: Chart, env_xy: np.ndarray) -> np.ndarray:
    """(T+Q,) bool: the window's cells whose centroid lies inside the envelope polygon (chart); refuses an empty
    region, one in more than one piece, or one touching the window's cut."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    m = win.mesh
    cen = np.r_[m.points[m.tris].mean(1) if m.n_tris else np.zeros((0, 3)), m.points[m.quads].mean(1)]
    reg = point_in_polygon(chart.fwd(cen), env_xy)
    if not reg.any():
        raise InsertError("the envelope holds no host cell")
    topo = build_topology(m)
    ee = topo.edge_elems
    inn = (ee[:, 1] >= 0) & reg[np.maximum(ee[:, 0], 0)] & reg[np.maximum(ee[:, 1], 0)]
    E = m.n_elems
    n, lab = connected_components(coo_matrix((np.ones(inn.sum()), (ee[inn, 0], ee[inn, 1])), shape=(E, E)),
                                  directed=False)
    if len(np.unique(lab[reg])) > 1:
        raise InsertError(f"the envelope's host cells are {len(np.unique(lab[reg]))} pieces (one is needed)")
    used = np.unique(np.r_[m.tris[reg[:m.n_tris]].ravel(), m.quads[reg[m.n_tris:]].ravel()])
    if (win.kind[used] == 2).any():
        raise InsertError("the envelope reaches the window's cut (a larger window is needed)")
    return reg


def boundary_loops(mesh: Mesh, reg: np.ndarray) -> list:
    """R's boundary as loops [(nodes (k,), join (k,) bool: edge i = nodes[i] -> nodes[i+1] has a host cell outside R
    across it)], R on the left; refuses a pinch node."""
    de, _ = directed_edges(mesh, reg)
    n = mesh.n_nodes
    key = de[:, 0] * n + de[:, 1]
    bnd = de[~np.isin(de[:, 1] * n + de[:, 0], key)]
    wde, _ = directed_edges(mesh)
    wkey = wde[:, 0] * n + wde[:, 1]
    join = np.isin(bnd[:, 1] * n + bnd[:, 0], wkey)
    nxt = {}
    for (a, b), j in zip(bnd.tolist(), join.tolist()):
        if a in nxt:
            raise InsertError(f"the envelope's boundary pinches at a host node (two outgoing boundary edges at {a})")
        nxt[a] = (b, j)
    loops, seen = [], set()
    for a0 in list(nxt):
        if a0 in seen:
            continue
        nodes, jn, a = [], [], a0
        while a not in seen:
            seen.add(a)
            b, j = nxt[a]
            nodes.append(a)
            jn.append(j)
            a = b
        loops.append((np.array(nodes), np.array(jn)))
    return loops


def runs(nodes: np.ndarray, join: np.ndarray) -> list:
    """The outer loop split into runs [("join" | "coast", node ids from its first to its last node)], starting with
    a join; consecutive runs share their end node."""
    k = len(nodes)
    if not join.any():
        raise InsertError("the envelope's boundary has no join edge (the envelope is an island?)")
    if join.all():
        return [("join", np.r_[nodes, nodes[:1]])]
    s = int(np.flatnonzero(join & ~np.roll(join, 1))[0])
    nodes, join = np.roll(nodes, -s), np.roll(join, -s)
    out, i = [], 0
    while i < k:
        j = i
        while j < k and join[j] == join[i]:
            j += 1
        out.append(("join" if join[i] else "coast", np.r_[nodes[i:j], nodes[j % k]]))
        i = j
    return out


def _resample_h(xy: np.ndarray, h_xy, closed: bool = False, step: float | None = None) -> np.ndarray:
    """Points along a polyline at equal steps of the integral of ds / h (h_xy: chart points -> local size in chart
    metres): an open one keeps its ends (at least one step), a closed one gets at least 3 points (first not
    repeated)."""
    q = np.vstack([xy, xy[:1]]) if closed else xy
    pl = Polyline(q)
    if pl.L <= 0:
        return xy[:1]
    if step is None:
        step = max(pl.L / 4000.0, 1.0)
    t = np.linspace(0.0, pl.L, max(int(np.ceil(pl.L / step)), 2) + 1)
    p = pl.at(t)
    hv = np.asarray(h_xy(p), dtype=float)
    w = np.r_[0.0, np.cumsum(np.diff(t) / (0.5 * (hv[1:] + hv[:-1])))]
    m = max(int(round(w[-1])), 3 if closed else 1)
    tt = np.interp(np.linspace(0.0, w[-1], m + 1), w, t)
    out = pl.at(tt)
    out[0], out[-1] = q[0], q[-1]
    return out[:-1] if closed else out


def _near_run(P: np.ndarray, tree_b: cKDTree, reach: float, closed: bool) -> tuple[int, int] | None:
    """The run of P's points within reach of B around P's point nearest to B: (i0, i1) (closed P: i1 may wrap)."""
    d, _ = tree_b.query(P)
    i = int(np.argmin(d))
    if d[i] > reach:
        return None
    n = len(P)
    i0 = i1 = i
    for _ in range(n - 1):
        k = (i0 - 1) % n if closed else i0 - 1
        if k < 0 or d[k] > reach or (closed and k == i1):
            break
        i0 = k
    for _ in range(n - 1):
        k = (i1 + 1) % n if closed else i1 + 1
        if k >= n or d[k] > reach or (closed and k == i0):
            break
        i1 = k
    return i0, i1


def _forward(B: np.ndarray, j0: int, j1: int) -> np.ndarray:
    return B[j0:j1 + 1] if j1 >= j0 else np.vstack([B[j0:], B[:j1 + 1]])


def join_island(P: np.ndarray, B: np.ndarray, reach: float, closed: bool) -> np.ndarray | None:
    """B241: the island loop B (water on its left) joined to the boundary P (an open coast polyline or another island
    loop, water on its left): the points of P within `reach` of B are cut out and replaced by B's far side, from B's
    point nearest to the run's first point round to the one nearest to its last (B's own direction keeps the water on
    the left): the water between them is closed. None when no point of P lies within reach."""
    tb = cKDTree(B)
    r = _near_run(P, tb, reach, closed)
    if r is None:
        return None
    n = len(P)
    while closed and ((r[1] - r[0]) % n) > n // 2 and reach > 1.0:   # a closed P keeps its far half at least
        reach *= 0.7
        r = _near_run(P, tb, reach, closed) or r
    i0, i1 = r
    if not closed:
        i0, i1 = max(i0, 1), min(i1, n - 2)
        if i1 < i0:
            i0 = i1 = min(max(r[0], 1), n - 2)
    j0 = int(tb.query(P[i0])[1])
    j1 = int(tb.query(P[i1])[1])
    far = _forward(B, j0, j1)
    if len(far) < 3:                                       # the near run touched B at one point: go round all of it
        far = np.vstack([B[j0:], B[:j0 + 1]])
    if closed:
        rest = _forward(P, i1, i0) if i1 != i0 else np.vstack([P[i0:], P[:i0 + 1]])   # one near point: all of P
        return np.vstack([rest, far])
    return np.vstack([P[:i0 + 1], far, P[i1:]])


def _seg_cross(a, b, c, d) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def crossings(loops: list) -> list:
    """Pairs of crossing segments among the closed polylines `loops` (chart), as midpoints: a quick check that the
    new boundary is a set of simple, disjoint loops."""
    segs, own = [], []
    for k, L in enumerate(loops):
        q = np.vstack([L, L[:1]])
        segs.append(np.stack([q[:-1], q[1:]], axis=1))
        own.append(np.full(len(q) - 1, k))
    S = np.concatenate(segs)
    mid = S.mean(1)
    half = 0.5 * np.linalg.norm(S[:, 1] - S[:, 0], axis=1)
    tree = cKDTree(mid)
    out = []
    for i, j in tree.query_pairs(2.0 * float(half.max()) + 1.0):
        if np.linalg.norm(mid[i] - mid[j]) > half[i] + half[j]:
            continue
        a, b = S[i]
        c, d = S[j]
        if np.allclose(a, c) or np.allclose(a, d) or np.allclose(b, c) or np.allclose(b, d):
            continue
        if _seg_cross(a, b, c, d):
            out.append(0.5 * (mid[i] + mid[j]))
    return out


class SlopeCone:
    """h(x) = min(cap(x), min_k h_k + g_k |x - p_k|) (chord distance) with a slope g_k per sample (B252), exact: the
    K_NEAR nearest samples give a first bound; a farther sample gives at least min(h_k) + min(g_k) d_K, so only the
    queries where that can beat the bound search the samples within its radius (as assemble.InsertH). Picklable."""

    K_NEAR = 16

    def __init__(self, cap, pts: np.ndarray, hk: np.ndarray, gk: np.ndarray):
        self.cap, self.pts, self.hk, self.gk = cap, np.asarray(pts), np.asarray(hk, float), np.asarray(gk, float)
        self._t = None

    def __call__(self, xyz):
        if self._t is None:
            self._t = cKDTree(self.pts)
        q = np.atleast_2d(xyz)
        cap = np.asarray(self.cap(q), dtype=float)
        G = self.gk * R_EARTH_M
        k = min(self.K_NEAR, len(self.pts))
        d, j = self._t.query(q, k=k)
        d, j = d.reshape(len(d), k), j.reshape(len(j), k)
        best = np.min(self.hk[j] + G[j] * d, axis=1)
        bound = np.minimum(best, cap)
        hmin, gmin = float(self.hk.min()), float(G.min())
        if k < len(self.pts):
            need = np.flatnonzero(hmin + gmin * d[:, -1] < bound)
            if len(need):
                r = (bound[need] - hmin) / gmin
                for i, idx in zip(need, self._t.query_ball_point(q[need], r)):
                    if idx:
                        idx = np.asarray(idx)
                        best[i] = min(best[i], float(np.min(self.hk[idx] + G[idx] * np.linalg.norm(
                            self.pts[idx] - q[i], axis=1))))
        return np.minimum(best, cap)

    def __getstate__(self):
        return {"cap": self.cap, "pts": self.pts, "hk": self.hk, "gk": self.gk}

    def __setstate__(self, s):
        self.__dict__.update(s)
        self._t = None


def join_slopes(pts: np.ndarray, hk: np.ndarray, fine: np.ndarray, join_pts: np.ndarray, join_h: np.ndarray,
                g: float, g_max: float = None) -> np.ndarray:
    """B252: each fine sample's slope (pts, hk: all samples; fine: the samples that are no join node): the least slope
    >= g with which its cone reaches every join node's host size, h_k + g_k |p_j - p_k| >= H_j, at most g_max (default
    GROWTH_MAX); the join samples keep g. So the field reaches the host's size at the joins however near a channel or
    a narrow gap lies to them, and the size steps are spread over the whole way from there (a plain slope g left
    the patch cells along the joins 1.6-2.5 times finer than the host's, ORCA025 v5: a fan of triangles on every
    join edge)."""
    gk = np.full(len(pts), float(g))
    g_max = GROWTH_MAX if g_max is None else float(g_max)
    if not len(join_pts) or not fine.any():
        return gk
    from scipy.spatial.distance import cdist

    d = R_EARTH_M * cdist(pts[fine], join_pts)
    need = (np.asarray(join_h)[None, :] - hk[fine][:, None]) / np.maximum(d, 1.0)
    gk[fine] = np.clip(need.max(1), g, max(g, g_max))
    return gk


def chord_test(pieces: list, closed: list):
    """B256: the bay search's chord test for a group (meshom.insert.realcoast.RealCoast.prepare's chord_ok): a pocket
    of piece k whose mouth chord a -> b crosses another piece (an island, the other bank of a channel) is no bay and
    is not bridged (the chord would cut through the land beside it and close the passage between)."""
    segs, own = [], []
    for k, (p, cl) in enumerate(zip(pieces, closed)):
        q = np.vstack([p, p[:1]]) if cl else p
        if len(q) >= 2:
            segs.append(np.stack([q[:-1], q[1:]], axis=1))
            own.append(np.full(len(q) - 1, k))
    S, own = np.concatenate(segs), np.concatenate(own)
    mid = S.mean(1)
    half = float(0.5 * np.linalg.norm(S[:, 1] - S[:, 0], axis=1).max())
    tree = cKDTree(mid)

    def ok(k, a, b):
        c = 0.5 * (a + b)
        for i in tree.query_ball_point(c, 0.5 * float(np.linalg.norm(b - a)) + half + 1.0):
            if own[i] != k and _seg_cross(a, b, S[i, 0], S[i, 1]):
                return False
        return True
    return ok


def fill_coast(rc, lines: list, step: float, scale: float) -> list:
    """B257: a group's fill lines (chart polylines, drawn in the water from coast to coast). Each line's ends go to
    the nearest points of the main-coast piece nearest to both ends; the coast between them (on a closed piece the
    shorter way round, on an open one the run between) is replaced by the line, the coast's direction kept: the water
    behind the line becomes land (a closed inlet, a narrowed channel). Changes rc.main (and rc.base) and their
    polylines in place. scale: the chart's scale (chart m per sphere m). Returns one record per line."""
    from meshom.insert.realcoast import _resample

    out = []
    for li, F0 in enumerate(lines):
        F = _resample(np.asarray(F0, dtype=float), False, step)
        best = None
        for k, m in enumerate(rc.main):
            if len(m) < 4:
                continue
            d = max(float(np.min(np.linalg.norm(m - F[0], axis=1))), float(np.min(np.linalg.norm(m - F[-1], axis=1))))
            if best is None or d < best[0]:
                best = (d, k)
        k = best[1]
        m, cl = rc.main[k], rc.closed[k]
        ia, ib = int(np.argmin(np.linalg.norm(m - F[0], axis=1))), int(np.argmin(np.linalg.norm(m - F[-1], axis=1)))
        n = len(m)
        if cl:
            if (ib - ia) % n > (ia - ib) % n:              # the shorter way round runs from ib to ia
                F, ia, ib = F[::-1], ib, ia
            R = np.roll(m, -ia, axis=0)
            j = (ib - ia) % n
            cut, F = R[:j + 1], np.vstack([R[:1], F[1:-1], R[j:j + 1]])
            new = np.vstack([F, R[j + 1:]])
        else:
            if ia > ib:
                F, ia, ib = F[::-1], ib, ia
            cut, F = m[ia:ib + 1], np.vstack([m[ia:ia + 1], F[1:-1], m[ib:ib + 1]])
            new = np.vstack([m[:ia], F, m[ib + 1:]])
        a_km2 = abs(_shoelace(np.vstack([cut, F[::-1][1:-1]]))) / scale ** 2 / 1e6
        rc.main[k] = new
        rc._pl[k] = Polyline(np.vstack([new, new[:1]]) if cl else new)
        if rc.base:
            rc.base[k] = new
            rc._plb[k] = Polyline(np.vstack([new, new[:1]]) if cl else new)
        out.append({"line": li, "piece": int(k), "closed_piece": bool(cl), "coast_km": round(
            float(np.linalg.norm(np.diff(cut, axis=0), axis=1).sum()) / scale / 1e3, 2), "land_km2": round(a_km2, 2),
            "snap_km": round(best[0] / scale / 1e3, 2)})
    return out


class GroupH:
    """B243: the group's size field (sphere metres): h(x) = max(h_min, min(cap(x), min_k h_k + g_k |x - p_k|)), the
    cone over the samples p_k (the cross lines' nodes at their spacing, the joins' nodes at their host edge length, the
    narrow gaps' points at their width / GAP_CELLS), capped by the host's cell size (MeshH of the host window); with
    the join samples given (`join`: (K,) bool), each other sample's slope g_k from join_slopes (B252), else g.
    Picklable."""

    def __init__(self, cap, pts: np.ndarray, hk: np.ndarray, g: float, h_min: float, join: np.ndarray | None = None,
                 g_max: float | None = None):
        pts, hk = np.asarray(pts), np.asarray(hk, float)
        if join is None or not np.any(join):
            gk = np.full(len(pts), float(g))
        else:
            join = np.asarray(join, bool)
            gk = join_slopes(pts, hk, ~join, pts[join], hk[join], g, g_max)
        self.gk = gk
        self.cone = SlopeCone(cap, pts, hk, gk)
        self.h_min = float(h_min)

    def __call__(self, xyz):
        return np.maximum(self.cone(np.atleast_2d(xyz)), self.h_min)


@dataclass
class GroupResult:
    ok: bool
    record: dict
    window: Mesh | None = None
    pieces: list = field(default_factory=list)
    outline: np.ndarray | None = None
    blocks_xyz: list = field(default_factory=list)       # per channel: the block's nodes (xyz)
    cross_xyz: list = field(default_factory=list)        # per channel: (cross A nodes, cross B nodes) (xyz)
    joins_xy: list = field(default_factory=list)
    h: object = None
    real: object = None
    holes: list = field(default_factory=list)            # the island loops kept as holes (chart)


def gap_samples(loops: list, query: list, h_min: float, scale, gap_cells: int = GAP_CELLS):
    """B243: the narrow gaps of the water. loops: closed dense chart polylines with the water on their left (the outer
    boundary in loop order, the islands); query: per loop, its points that are coast (not a join: a join is open
    water, never a wall). For each coast point, the distance to the nearest coast point across the water: of another
    loop, or of its own loop farther along it than 3 x that distance plus 2 h_min (the other side of a bay or a
    channel), each lying on the other's water side (a narrow cape or island is not a gap). Returns (points chart,
    width / gap_cells in sphere metres, at least h_min)."""
    P = np.vstack(loops)
    lid = np.concatenate([np.full(len(L), k) for k, L in enumerate(loops)])
    s, per, nrm = [], [], []
    for L in loops:
        q = np.vstack([L, L[:1]])
        c = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        s.append(c[:-1])
        per.append(np.full(len(L), c[-1]))
        t = np.roll(L, -1, axis=0) - np.roll(L, 1, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        nrm.append(np.c_[-t[:, 1], t[:, 0]])               # the water side (left)
    s, per, nrm = np.concatenate(s), np.concatenate(per), np.vstack(nrm)
    qm = np.concatenate(query)
    tree = cKDTree(P[qm])
    sc = np.asarray(scale(P))
    w = np.full(len(P), np.inf)
    idx = np.flatnonzero(qm)
    k = min(64, len(idx))
    d, j = tree.query(P[qm], k=k)
    j = idx[j]
    si, li, sci, ni, pi = s[qm], lid[qm], sc[qm], nrm[qm], P[qm]
    wq = np.full(qm.sum(), np.inf)
    for c in range(1, k):
        jj = j[:, c]
        ds = np.abs(s[jj] - si)
        ds = np.minimum(ds, per[jj] - ds)
        far = (lid[jj] != li) | (ds > 3.0 * d[:, c] + 2.0 * h_min * sci)
        v = P[jj] - pi
        wet = (np.sum(v * ni, 1) > 0) & (np.sum(-v * nrm[jj], 1) > 0)
        wq = np.where(far & wet & (d[:, c] < wq), d[:, c], wq)
    w[qm] = wq
    ok = np.isfinite(w)
    return P[ok], np.maximum(w[ok] / sc[ok] / gap_cells, h_min)


def _dense(xy: np.ndarray, step: float) -> np.ndarray:
    """A polyline with points at most `step` apart (its own points kept)."""
    out = [xy[:1]]
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        t = np.linspace(0.0, 1.0, n + 1)[1:, None]
        out.append(a + t * (b - a))
    return np.vstack(out)


def coast_arc(rc, xa: np.ndarray, xb: np.ndarray, guide: np.ndarray, env_xy: np.ndarray, h: float):
    """B240: the real coast between a coast run's two end nodes xa, xb. Candidates: the smoothed coast pieces with a
    point inside the envelope whose larger distance to the two ends is within 2 h (the host's cell) of the nearest
    such piece's; on each, the way(s) from the point nearest xa to the point nearest xb (both ways round a closed
    one). Taken: the nearest piece by that distance in whole host cells, then the way that lies most inside the
    envelope (the host's coast may lack a whole cape, so the host's run is no guide there), then the way nearer to the
    host's run. Returns (polyline or None, note)."""
    pls = rc._pls("main")
    cand = []
    for k, pl in enumerate(pls):
        if pl.L <= 0 or not point_in_polygon(pl.xy, env_xy).any():
            continue
        sa, sb = float(pl.project(xa[None])[0]), float(pl.project(xb[None])[0])
        da, db = float(np.linalg.norm(pl.at(sa)[0] - xa)), float(np.linalg.norm(pl.at(sb)[0] - xb))
        cand.append((max(da, db), k, sa, sb, da, db))
    if not cand:
        return None, "no coast piece inside the envelope", -1
    d0 = min(c[0] for c in cand)
    gp = Polyline(guide)
    best = None
    for dmax, k, sa, sb, da, db in cand:
        if dmax > d0 + 2.0 * h:
            continue
        for w in rc._ways(k, sa, sb):
            sc = (int(dmax / h), -float(np.mean(point_in_polygon(w, env_xy))),
                  float(np.mean(Polyline(w).distance(gp.xy))) + float(np.mean(gp.distance(w))))
            if best is None or sc < best[0]:
                best = (sc, w, k, da, db)
    sc, w, k, da, db = best
    return w, (f"piece {k} ({'closed' if rc.closed[k] else 'open'}), {100 * -sc[1]:.0f} % inside the envelope, ends "
               f"{da / 1e3:.1f} / {db / 1e3:.1f} km off"), k


def bank_sides(c_xy: list) -> list:
    """B255: per channel, the chart points of its two banks' ends: (S ends (2,2), N ends (2,2)); bank S is the end on
    the right looking from cross line a to b (as build's blocks take them)."""
    out = []
    for A, B in c_xy:
        t = B.mean(0) - A.mean(0)
        s, n = [], []
        for line in (A, B):
            cr = [t[0] * (p[1] - line.mean(0)[1]) - t[1] * (p[0] - line.mean(0)[0]) for p in line]
            iS = int(np.argmin(cr))
            s.append(line[iS])
            n.append(line[1 - iS])
        out.append((np.array(s), np.array(n)))
    return out


def join_raw(rc, env_xy: np.ndarray, gap_m: float, island_km2: float, chart: Chart, apart: list = ()) -> dict:
    """B241, on the raw coast pieces before the smoothing (an island narrower than the smoothing's width would vanish
    in it): the islands (closed pieces inside the envelope) smaller than island_km2 are removed; then, closest pair
    first, an island closer than gap_m (sphere metres) to another piece is joined to it (join_island; to an island:
    the two become one island; to any other piece: the island becomes part of that coast). B255: apart = per channel
    (its bank S ends, its bank N ends) in chart metres (bank_sides): the pieces nearest to a channel's S ends and those
    nearest to its N ends are never joined to each other, nor are the pieces joined into them (a channel narrower than
    gap_m stays open). Changes rc.pieces / rc.closed in place; returns the record (joined, dropped, apart)."""
    sc = float(np.mean(chart.scale(env_xy)))
    gap, reach = gap_m * sc, BRIDGE_REACH * gap_m * sc

    def info(p):
        lo, la = xyz_to_lonlat(chart.inv(p.mean(0)[None]))
        return {"area_km2": round(abs(_shoelace(p)) / sc ** 2 / 1e6, 2),
                "lonlat": [round(float(lo[0]), 4), round(float(la[0]), 4)]}

    pcs = [(p if (not cl or _shoelace(p) < 0) else p[::-1]) for p, cl in zip(rc.pieces, rc.closed)]
    cls = list(rc.closed)
    isl = [cl and len(p) >= 3 and bool(point_in_polygon(p, env_xy).all()) for p, cl in zip(pcs, cls)]
    dropped, joined = [], []
    keep = []
    for k, p in enumerate(pcs):
        if isl[k] and abs(_shoelace(p)) / sc ** 2 / 1e6 < island_km2:
            dropped.append(info(p) | {"why": "raw area below island_km2"})
            continue
        keep.append(k)
    pcs, cls, isl = [pcs[k] for k in keep], [cls[k] for k in keep], [isl[k] for k in keep]
    trees = [cKDTree(p) for p in pcs]
    tags = [set() for _ in pcs]                            # B255: (channel, side) of the banks each piece holds
    for ci, sides in enumerate(apart):
        for side, ends in zip("SN", sides):
            for q in np.atleast_2d(ends):
                tags[int(np.argmin([tr.query(q)[0] for tr in trees]))].add((ci, side))
    other = {"S": "N", "N": "S"}

    def opposite(k, t):
        return any((ci, other[sd]) in tags[t] for ci, sd in tags[k])

    kept_apart = set()
    while True:
        best = None
        for k in range(len(pcs)):
            if not isl[k]:
                continue
            for t in range(len(pcs)):
                if t == k:
                    continue
                d = float(trees[k].query(pcs[t], distance_upper_bound=gap)[0].min())
                if d < gap and opposite(k, t):             # B255: two banks of one channel
                    kept_apart |= {ci for ci, sd in tags[k] if (ci, other[sd]) in tags[t]}
                    continue
                if d < gap and (best is None or d < best[0]):
                    best = (d, k, t)
        if best is None:
            break
        d, k, t = best
        if isl[t] and abs(_shoelace(pcs[t])) < abs(_shoelace(pcs[k])):   # two islands: the smaller joins the larger
            k, t = t, k
        new = join_island(pcs[t], pcs[k], min(reach, d + 0.5 * gap), closed=cls[t])
        joined.append(info(pcs[k]) | {"to": "island" if isl[t] else "coast", "gap_km": round(d / sc / 1e3, 2),
                                      "to_lonlat": info(pcs[t])["lonlat"]})
        pcs[t] = new
        trees[t] = cKDTree(new)
        tags[t] |= tags[k]
        del pcs[k], cls[k], isl[k], trees[k], tags[k]
    rc.pieces, rc.closed = pcs, cls
    return {"joined": joined, "dropped": dropped, "apart": sorted(kept_apart)}


def build(win, chart: Chart, env_xy: np.ndarray, channels: list, real, h_min: float, gap_m: float,
          island_km2: float, growth: float, bank_sigma_cells: float = 1.5, log=print, name: str = "group",
          fill_opts: dict | None = None, land_cells: float = 0.0, gap_cells: int = GAP_CELLS,
          bank_smooth: float = 0.0, state: dict | None = None, blend_cells: float | None = None,
          growth_max: float | None = None, fill: list | None = None) -> GroupResult:
    """The group insert in the host window `win` (module docstring). real: the run's coast loops ((K,3), water on the
    left; None: the host's coast). Returns a GroupResult (window: the window mesh with the group in place of R).
    state: a dict that receives the real coast, the outline and the holes as they are built (for the maps of a
    refusal). fill: chart polylines whose water behind becomes land (B257, fill_coast)."""
    state = {} if state is None else state
    for ch in channels:                                    # B258: no 1-cell channel
        if ch.cells < MIN_CELLS_ACROSS:
            raise InsertError(f"channel {ch.name}: {ch.cells} cell(s) across; a channel needs at least "
                              f"{MIN_CELLS_ACROSS} cells across (one cell carries no flow, B258)")
    from meshom.insert import block as blockm
    from meshom.insert import measure
    from meshom.insert import realcoast as rcm
    from meshom.insert.assemble import MeshH, replace_part, split_nonconvex
    from meshom.insert.realcoast import BLEND_H, RealCoast
    from meshom.methods import m02_linf_frontal as m02
    from meshom.domain import Domain

    mesh = win.mesh
    xy = chart.fwd(mesh.points)
    rec = {"name": name, "timing": {}}
    reg = envelope_region(win, chart, env_xy)
    loops = boundary_loops(mesh, reg)
    area = [_shoelace(xy[nodes]) for nodes, _ in loops]
    io = int(np.argmax(area))
    for k, (nodes, jn) in enumerate(loops):
        if k != io and jn.any():
            raise InsertError("the envelope leaves host water cells inside it (an inner boundary with join edges)")
    rs = runs(*loops[io])
    capH = MeshH(mesh)
    rec["region"] = {"cells": int(reg.sum()), "joins": sum(r[0] == "join" for r in rs),
                     "host_islands_inside": len(loops) - 1}
    log(f"[insert {name}] envelope: {int(reg.sum())} host cells, {rec['region']['joins']} joins, "
        f"{len(loops) - 1} host island(s) inside", flush=True)

    # --- the real coast, smoothed at the group's first size estimate (the channels' necks, the host's size)
    h_host = np.asarray(capH(mesh.points))
    centre_h = float(np.median(h_host[np.unique(np.r_[mesh.tris[reg[:mesh.n_tris]].ravel(),
                                                       mesh.quads[reg[mesh.n_tris:]].ravel()])]))
    c_xy = [(chart.fwd(ch.a), chart.fwd(ch.b)) for ch in channels]
    rc = None
    if real is not None:
        win_r = float(np.max(np.linalg.norm(xy, axis=1))) / float(chart.scale(np.zeros((1, 2)))[0])
        rc = RealCoast.from_loops(real, chart, chart.c, win_r + 2 * centre_h, "run")
        if not rc.pieces:
            rc = None
    necks = []
    for (A, B), ch in zip(c_xy, channels):
        w = 0.5 * (np.linalg.norm(A[1] - A[0]) + np.linalg.norm(B[1] - B[0]))
        necks.append((0.25 * (A[0] + A[1] + B[0] + B[1]), w / max(ch.cells, 1)))
    tree_h = cKDTree(xy)

    def h0_xy(q):                                          # chart m: the first estimate (coast smoothing, B243)
        q = np.atleast_2d(q)
        cap = h_host[tree_h.query(q)[1]] * chart.scale(q)
        h = cap.copy()
        for p, hc in necks:
            h = np.minimum(h, hc + growth * np.linalg.norm(q - p, axis=1))
        return np.maximum(h, h_min * chart.scale(q))

    pre = {"joined": [], "dropped": []}
    inlets = []
    if rc is not None:
        pre = join_raw(rc, env_xy, gap_m, island_km2, chart, apart=bank_sides(c_xy))     # B255: banks kept apart
        def sig_mult(q):                                   # B241: an island keeps its size
            per = float(np.linalg.norm(np.diff(np.vstack([q, q[:1]]), axis=0), axis=1).sum())
            sig = bank_sigma_cells * float(np.median(h0_xy(q)))
            return np.full(len(q), min(1.0, ISLAND_SIGMA * per / max(sig, 1.0)))

        sc_e = float(np.mean(chart.scale(env_xy)))
        rc.prepare(h0_xy, lambda q: np.full(len(np.atleast_2d(q)), np.inf), h_min, bank_sigma_cells, bays=True,
                   bridge="narrow", sigma_mult=sig_mult, land_cells=land_cells,    # B245: capes cut when asked
                   chord_ok=chord_test(rc.pieces, rc.closed))                     # B256: no chord across land
        for k, (m, cl) in enumerate(zip(rc.smooth, rc.closed)):   # B241: an island inside the envelope keeps its
            if cl and len(m) >= 3 and point_in_polygon(m, env_xy).all():   # smoothed loop: the bay search treats a
                b = rc.main[k]                                         # loop as open and bridged a whole island
                if len(b) >= 3 and abs(_shoelace(b)) > abs(_shoelace(m)) and not crossings([b]):
                    lo, la = xyz_to_lonlat(chart.inv(b.mean(0)[None]))     # B256: unless the search only bridged
                                                                           # its inlets (the island grows, simple)
                    inlets.append({"lonlat": [round(float(lo[0]), 4), round(float(la[0]), 4)],
                                   "area_km2": round(abs(_shoelace(b)) / sc_e ** 2 / 1e6, 1),
                                   "inlets_km2": round((abs(_shoelace(b)) - abs(_shoelace(m))) / sc_e ** 2 / 1e6, 1)})
                    continue
                rc.main[k] = m
                rc._pl[k] = Polyline(np.vstack([m, m[:1]]))
                if rc.base:
                    rc.base[k] = m
                    rc._plb[k] = Polyline(np.vstack([m, m[:1]]))
        if fill:                                           # B257: the water behind each fill line becomes land
            rec["fill"] = fill_coast(rc, fill, max(rcm.STEP_CELLS * h_min * sc_e, 1.0), sc_e)
            for f in rec["fill"]:
                log(f"[insert {name}] fill line {f['line']}: {f['coast_km']} km of coast replaced, "
                    f"{f['land_km2']} km^2 of water made land", flush=True)

    # --- the new outer boundary: joins kept, coast runs replaced by the real coast
    pieces, fallback, used_pieces = [], [], set()
    sc_c = float(np.mean(chart.scale(env_xy)))
    hj = []
    for kind, ids in rs:
        if kind == "join":
            L = np.linalg.norm(np.diff(xy[ids], axis=0), axis=1)
            hj.append(float(L.mean()))
    ji = 0
    for k, (kind, ids) in enumerate(rs):
        if kind == "join":
            pieces.append(Piece("join", xy[ids], ids=ids, start=("h", int(ids[0])), end=("h", int(ids[-1]))))
            ji += 1
            continue
        bl = BLEND_H if blend_cells is None else float(blend_cells)    # B247: the blend's length in join cells
        la, lb = bl * hj[(ji - 1) % len(hj)], bl * hj[ji % len(hj)]
        arc, why, kp = (None, "no real coast", -1) if rc is None else coast_arc(rc, xy[ids[0]], xy[ids[-1]], xy[ids], env_xy,
                                                                      centre_h * sc_c)
        log(f"[insert {name}] coast run {k}: {why}", flush=True)
        if arc is not None:
            used_pieces.add(int(kp))
        if arc is None:
            fallback.append({"run": k, "reason": why, "ends_lonlat": [[round(float(v[0]), 4) for v in xyz_to_lonlat(
                mesh.points[i][None])] for i in (ids[0], ids[-1])]})
            log(f"[insert {name}] WARNING: coast run {k} keeps the host's coast: {why}", flush=True)
            arc = xy[ids]
        else:
            arc = rcm.blend_ends(arc, xy[ids[0]], xy[ids[-1]], la, lb)
        pieces.append(Piece("coast", arc, start=("h", int(ids[0])), end=("h", int(ids[-1]))))
    rec["coast_fallback"] = fallback
    outline = np.vstack([p.xy[:-1] for p in pieces])
    state.update(rc=rc, outline=outline, pieces=pieces)

    # --- islands: the closed smoothed pieces inside the new boundary are holes (the raw ones were joined and dropped
    # before the smoothing, join_raw); one the smoothing shrank below island_km2 is dropped too
    sc0 = float(np.mean(chart.scale(outline)))
    holes, hole_info, dropped = [], [], list(pre["dropped"])
    if rc is not None:
        for km, (m, cl) in enumerate(zip(rc.main, rc.closed)):
            if not cl or len(m) < 3:
                continue
            if km in used_pieces:                          # a coast arc of the boundary runs along it
                lo, la = xyz_to_lonlat(chart.inv(m.mean(0)[None]))
                log(f"[insert {name}]   island on the boundary's coast: {abs(_shoelace(m)) / sc0 ** 2 / 1e6:.0f} km^2 "
                    f"at [{float(lo[0]):.3f}, {float(la[0]):.3f}]", flush=True)
                continue
            inside = point_in_polygon(m, outline)
            if not inside.any():
                continue
            a_km2 = abs(_shoelace(m)) / sc0 ** 2 / 1e6
            lo, la = xyz_to_lonlat(chart.inv(m.mean(0)[None]))
            info = {"area_km2": round(a_km2, 2), "lonlat": [round(float(lo[0]), 4), round(float(la[0]), 4)]}
            if not inside.all():
                if a_km2 < island_km2:                     # a small one across a join: dropped
                    dropped.append(info | {"why": "crosses the boundary, below island_km2"})
                    continue
                raise InsertError(f"an island of the real coast crosses the group's boundary at {info['lonlat']} "
                                  f"({info['area_km2']} km^2, {int(inside.sum())}/{len(m)} points inside: move the "
                                  "envelope off it)")
            if a_km2 < island_km2:
                dropped.append(info | {"why": "smoothed at the local cell size"})
                continue
            holes.append(m if _shoelace(m) < 0 else m[::-1])
            hole_info.append(info)
    state.update(holes=holes)
    rec["islands"] = {"holes": hole_info, "joined": pre["joined"], "dropped": dropped,
                      "banks_apart": [channels[ci].name for ci in pre.get("apart", [])], "inlets_closed": inlets,
                      "gap_km": gap_m / 1e3, "island_km2": island_km2}
    log(f"[insert {name}] islands: {len(holes)} hole(s), {len(pre['joined'])} joined to land or to each other "
        f"(gap < {gap_m / 1e3:.1f} km), {len(dropped)} dropped (< {island_km2} km^2)", flush=True)
    if rec["islands"]["banks_apart"]:
        log(f"[insert {name}]   banks kept apart (B255): {rec['islands']['banks_apart']}", flush=True)
    for hi in hole_info:
        log(f"[insert {name}]   hole {hi['area_km2']} km^2 at {hi['lonlat']}", flush=True)
    for dr in dropped:
        if dr["area_km2"] > island_km2:
            log(f"[insert {name}]   dropped {dr}", flush=True)
    for j in pre["joined"]:
        log(f"[insert {name}]   joined {j['area_km2']} km^2 at {j['lonlat']} to the {j['to']} at {j['to_lonlat']} "
            f"(gap {j['gap_km']} km)", flush=True)
    outline = np.vstack([p.xy[:-1] for p in pieces])
    bad = crossings([outline] + holes)
    if bad:
        ll = [[round(float(v[0]), 4) for v in xyz_to_lonlat(chart.inv(b[None]))] for b in bad[:5]]
        raise InsertError(f"the group's boundary crosses itself at {len(bad)} place(s), first at {ll} (the real "
                          f"coast arcs overlap a join or each other: move the envelope)")

    # --- the channels' cross lines snapped onto the boundary
    loopsets = [("coast", pi, p.xy) for pi, p in enumerate(pieces) if p.kind == "coast"]
    loopsets += [("island", k, np.vstack([B, B[:1]])) for k, B in enumerate(holes)]
    pls = [Polyline(L) for _, _, L in loopsets]

    def snap(q):
        best = None
        for li, pl in enumerate(pls):
            s = float(pl.project(q[None])[0])
            d = float(np.linalg.norm(pl.at(s)[0] - q))
            if best is None or d < best[0]:
                best = (d, li, s)
        return best

    splits = {li: [] for li in range(len(loopsets))}
    ends = []
    for ci, ((A, B), ch) in enumerate(zip(c_xy, channels)):
        t = B.mean(0) - A.mean(0)
        e = {}
        for nm, line in (("A", A), ("B", B)):
            cr = [t[0] * (p[1] - line.mean(0)[1]) - t[1] * (p[0] - line.mean(0)[0]) for p in line]
            iS = int(np.argmin(cr))                        # bank S: on the right looking from A to B
            for side, p in (("S", line[iS]), ("N", line[1 - iS])):
                d, li, s = snap(p)
                if d > 3.0 * centre_h * sc0:
                    raise InsertError(f"channel {ch.name}: the end {side} of cross line {nm} lies {d / sc0 / 1e3:.0f} "
                                      "km from the coast")
                key = ("x", ci, nm + side)
                splits[li].append((s, key))
                e[nm + side] = (li, s, pls[li].at(s)[0])
        ends.append(e)

    # --- sub-pieces: coast pieces and islands split at the cross lines' ends
    subs = []                                              # in loop order per loop: (loop index, Piece)
    order = {}
    for li, (kind, idx, L) in enumerate(loopsets):
        pl = pls[li]
        sp = sorted(splits[li])
        if kind == "coast":
            p0 = pieces[idx]
            cuts = [(0.0, p0.start)] + sp + [(pl.L, p0.end)]
            seq = []
            for (s0, k0), (s1, k1) in zip(cuts[:-1], cuts[1:]):
                if s1 - s0 <= 1e-6:
                    raise InsertError(f"two cross line ends meet on the coast at {k0}, {k1}")
                seq.append(Piece("coast", _sub(pl, s0, s1), start=k0, end=k1))
            order[("coast", idx)] = seq
        else:
            if not sp:
                order[("island", idx)] = [Piece("island", L[:-1])]
                continue
            seq = []
            for a_, b_ in zip(sp, sp[1:] + [(sp[0][0] + pl.L, sp[0][1])]):
                seq.append(Piece("coast", _sub(pl, a_[0], b_[0], wrap=pl.L), start=a_[1], end=b_[1]))
            order[("island", idx)] = seq
    outer_seq = []
    for pi, p in enumerate(pieces):
        outer_seq += [p] if p.kind == "join" else order[("coast", pi)]
    seqs = [outer_seq] + [order[("island", k)] for k in range(len(holes))]

    # --- banks: S from AS to BS along the loop, N from BN to AN
    banks = []
    for ci, (e, ch) in enumerate(zip(ends, channels)):
        bk = {}
        for side, k0, k1 in (("S", ("x", ci, "AS"), ("x", ci, "BS")), ("N", ("x", ci, "BN"), ("x", ci, "AN"))):
            seq = next((sq for sq in seqs if any(p.start == k0 for p in sq)), None)
            if seq is None or not any(p.end == k1 for p in seq):
                raise InsertError(f"channel {ch.name}: the bank {side}'s two ends are not on one coast piece or "
                                  "island (cross line ends on different land)")
            i = next(i for i, p in enumerate(seq) if p.start == k0)
            path = []
            for _ in range(len(seq)):
                p = seq[i % len(seq)]
                if p.kind == "join":
                    raise InsertError(f"channel {ch.name}: bank {side} runs along a join (a cross line end on the "
                                      "wrong side)")
                if p.bank:
                    raise InsertError(f"channel {ch.name}: bank {side} overlaps another channel's bank")
                p.bank = f"c{ci}{side}"
                path.append(p.xy)
                if p.end == k1:
                    break
                i += 1
            xyb = np.vstack([path[0]] + [q[1:] for q in path[1:]])
            bk[side] = xyb if side == "S" else xyb[::-1]
        banks.append(bk)

    # --- channel blocks (fixed count: no triangle inside, B242)
    blocks, cross_xy, cross_xyz, cross_h = [], [], [], []
    for ci, (e, ch) in enumerate(zip(ends, channels)):
        n = int(ch.cells)
        tA = np.linspace(0.0, 1.0, n + 1)[:, None]
        xA = e["AS"][2] + tA * (e["AN"][2] - e["AS"][2])
        xB = e["BS"][2] + tA * (e["BN"][2] - e["BS"][2])
        hc = 0.5 * (np.linalg.norm(xA[-1] - xA[0]) + np.linalg.norm(xB[-1] - xB[0])) / n
        bS, bN = (smooth_bank_ends(banks[ci][sd], bank_smooth * hc) for sd in ("S", "N"))
        blk = blockm.build(xA, xB, Polyline(bS), Polyline(bN), n, fixed_count=True,
                           h_min=h_min, cell_floor=h_min, scale=chart.scale, growth=growth)
        zA, zB = chart.inv(xA), chart.inv(xB)
        P = chart.inv(blk.xy)
        M = blk.M
        P[blk.off[0]:blk.off[0] + n + 1] = zA
        P[blk.off[M]:blk.off[M] + n + 1] = zB
        blocks.append((blk, P))
        cross_xy.append((xA, xB))
        cross_xyz.append((zA, zB))
        hA = np.linalg.norm(np.diff(xA, axis=0), axis=1) / chart.scale(0.5 * (xA[1:] + xA[:-1]))
        hB = np.linalg.norm(np.diff(xB, axis=0), axis=1) / chart.scale(0.5 * (xB[1:] + xB[:-1]))
        cross_h.append((np.r_[hA[0], 0.5 * (hA[1:] + hA[:-1]), hA[-1]], np.r_[hB[0], 0.5 * (hB[1:] + hB[:-1]),
                                                                               hB[-1]]))
        sz = blockm.cell_sizes(blk.xy, blk.tris, blk.quads, chart.scale)
        log(f"[insert {name}] channel {ch.name}: block {M} rows x {n} cells, cells {sz.min() / 1e3:.2f}-"
            f"{sz.max() / 1e3:.2f} km, cross lines {hA.sum() / 1e3:.1f} / {hB.sum() / 1e3:.1f} km, "
            f"{len(blk.tris)} triangles", flush=True)

    # --- the size field of the patches (B243)
    sp_x, sp_h, sp_j = [], [], []
    for (zA, zB), (hA, hB) in zip(cross_xyz, cross_h):
        sp_x += [zA, zB]
        sp_h += [hA, hB]
        sp_j += [np.zeros(len(zA), bool), np.zeros(len(zB), bool)]
    for p in pieces:
        if p.kind == "join":
            z = mesh.points[p.ids]
            from meshom.geo import arc_length
            L = R_EARTH_M * arc_length(z[1:], z[:-1])
            sp_x.append(z)
            sp_h.append(np.r_[L[0], 0.5 * (L[1:] + L[:-1]), L[-1]] if len(L) > 1 else np.r_[L, L])
            sp_j.append(np.ones(len(z), bool))
    step_d = max(0.25 * h_min * sc0, 1.0)
    o_pts, o_q = [], []
    for p in outer_seq:
        q = _dense(p.xy, step_d)[:-1]
        o_pts.append(q)
        o_q.append(np.full(len(q), p.kind != "join"))
    gl, gq = [np.vstack(o_pts)], [np.concatenate(o_q)]
    for B in holes:
        q = _dense(np.vstack([B, B[:1]]), step_d)[:-1]
        gl.append(q)
        gq.append(np.ones(len(q), bool))
    gp, gh = gap_samples(gl, gq, h_min, chart.scale, gap_cells)
    keep = gh < np.asarray(capH(chart.inv(gp)))
    sp_x.append(chart.inv(gp[keep]))
    sp_h.append(gh[keep])
    sp_j.append(np.zeros(int(keep.sum()), bool))
    H = GroupH(capH, np.vstack(sp_x), np.concatenate(sp_h), growth, FILL_FLOOR * h_min,     # B256: floor margin
               join=np.concatenate(sp_j), g_max=growth_max)
    fine = ~np.concatenate(sp_j)
    rec["size"] = {"growth": growth, "gap_samples": int(keep.sum()),
                   "gap_h_min_km": float(gh[keep].min() / 1e3) if keep.any() else None,
                   "growth_max": GROWTH_MAX if growth_max is None else float(growth_max),
                   "slope_fine": [float(H.gk[fine].min()), float(np.median(H.gk[fine])), float(H.gk[fine].max())]}
    log(f"[insert {name}] size: slope {growth:g}, the fine samples' slopes to the joins' host size "
        f"{H.gk[fine].min():.3f} / {np.median(H.gk[fine]):.3f} / {H.gk[fine].max():.3f} (min / median / max)",
        flush=True)

    def h_xy(q):
        q = np.atleast_2d(q)
        return np.asarray(H(chart.inv(q))) * chart.scale(q)

    # --- the patches: trace the loops of the pieces outside the blocks
    key_xyz = {}
    for p in pieces:
        if p.kind == "join":
            key_xyz[p.start] = mesh.points[p.ids[0]]
            key_xyz[p.end] = mesh.points[p.ids[-1]]
    for ci, (zA, zB) in enumerate(cross_xyz):
        key_xyz[("x", ci, "AS")], key_xyz[("x", ci, "AN")] = zA[0], zA[-1]
        key_xyz[("x", ci, "BS")], key_xyz[("x", ci, "BN")] = zB[0], zB[-1]
    free = []
    for sq in seqs:
        for p in sq:
            if p.bank:
                continue
            if p.kind == "join":
                p.nodes = mesh.points[p.ids]
            elif p.kind == "island":
                p.nodes = chart.inv(_resample_h(p.xy, h_xy, closed=True))
            else:
                r = _resample_h(p.xy, h_xy)
                p.nodes = np.vstack([key_xyz[p.start][None], chart.inv(r[1:-1]), key_xyz[p.end][None]])
            free.append(p)
    for ci, (zA, zB) in enumerate(cross_xyz):
        free.append(Piece("cross", cross_xy[ci][0], start=("x", ci, "AS"), end=("x", ci, "AN"), nodes=zA))
        free.append(Piece("cross", cross_xy[ci][1][::-1], start=("x", ci, "BN"), end=("x", ci, "BS"),
                          nodes=zB[::-1]))
    by_start = {}
    closed_loops = []
    for p in free:
        if p.start is None:
            closed_loops.append(p.nodes)
            continue
        if p.start in by_start:
            raise InsertError(f"the patches cannot be traced: two pieces start at {p.start}")
        by_start[p.start] = p
    used, cycles = set(), []
    for k0, p0 in by_start.items():
        if k0 in used:
            continue
        nodes, k = [], k0
        while k not in used:
            used.add(k)
            p = by_start.get(k)
            if p is None:
                raise InsertError(f"the patches cannot be traced: no piece starts at {k}")
            nodes.append(p.nodes[:-1])
            k = p.end
        if k != k0:
            raise InsertError("the patches cannot be traced: a loop does not close")
        cycles.append(np.vstack(nodes))
    outers, inner = [], list(closed_loops)
    for cyc in cycles:
        (outers if _shoelace(chart.fwd(cyc)) > 0 else inner).append(cyc)
    patch_loops = [[o] for o in outers]
    for hl in inner:
        q = chart.fwd(hl[:1])
        k = next((i for i, o in enumerate(outers) if point_in_polygon(q, chart.fwd(o))[0]), None)
        if k is None:
            raise InsertError("an island lies in no patch")
        patch_loops[k].append(hl)

    # --- fill the patches (m02), then merge with the blocks
    parts, prec = [], []
    for k, lp in enumerate(patch_loops):
        dom = Domain(lp, f"{name}_patch{k}", [True] * len(lp))
        pm = m02.generate(dom, H, dict(FILL_OPTS, **(fill_opts or {})))
        nb = sum(len(L) for L in lp)
        d, _ = cKDTree(pm.points).query(np.vstack(lp))
        if d.max() > MERGE_CHORD:
            raise InsertError(f"patch {k}: the fill moved {int((d > MERGE_CHORD).sum())} of its {nb} loop nodes")
        pm, split = split_nonconvex(pm)                    # B243: a non-convex fill quad becomes two triangles
        parts.append(pm)
        prec.append({"loops": len(lp), "loop_nodes": nb, "quads": int(pm.n_quads), "tris": int(pm.n_tris),
                     "split_nonconvex": int(len(split))})
        log(f"[insert {name}] patch {k}: {len(lp)} loop(s), {nb} loop nodes -> {pm.n_quads} quads, {pm.n_tris} "
            f"triangles", flush=True)
    for (blk, P), ch in zip(blocks, channels):
        parts.append(Mesh(P, blk.tris.astype(np.int64), blk.quads.astype(np.int64), np.zeros(len(P), np.int8)))
    part = merge_parts(parts)
    joins_ids = np.unique(np.concatenate([p.ids for p in pieces if p.kind == "join"]))
    d, j = cKDTree(mesh.points[joins_ids]).query(part.points)
    snap_ = d < MERGE_CHORD
    part.points[snap_] = mesh.points[joins_ids[j[snap_]]]
    out = replace_part(mesh, reg, part, joins_ids)
    v = measure.validity(out)
    rec["patches"] = prec
    rec["channels"] = [{"name": ch.name, "cells": int(ch.cells), "rows": int(blk.M), "triangles": int(len(blk.tris)),
                        "cell_min_km": float(blockm.cell_sizes(blk.xy, blk.tris, blk.quads, chart.scale).min() / 1e3),
                        "cell_max_km": float(blockm.cell_sizes(blk.xy, blk.tris, blk.quads, chart.scale).max() / 1e3),
                        "info": {k: blk.info.get(k) for k in ("rows", "counts", "h_min_m", "h_max_m",
                                                              "neck_row_len_m", "floor_rounds")}}
                       for (blk, _), ch in zip(blocks, channels)]
    rec["new"] = {"quads": int(part.n_quads), "tris": int(part.n_tris), "nodes": int(part.n_nodes)}
    rec["validity_window"] = v
    why = []
    if not (v["inverted"] == 0 and v["nonconvex"] == 0 and v["nonmanifold_edges"] == 0
            and v["same_direction_edges"] == 0):
        why.append(f"the window has {v['inverted']} inverted / {v['nonconvex']} non-convex cells, "
                   f"{v['nonmanifold_edges']} edges with > 2 cells, {v['same_direction_edges']} shared edges "
                   "traversed the same way by both cells")
    rec["valid"] = not why
    if why:
        rec["invalid"] = why
        log(f"[insert {name}] INVALID result: {'; '.join(why)}", flush=True)
    fl = measure.floor_check(mesh, out, h_min)
    rec["floor"] = fl
    log(f"[insert {name}] new cells: {part.n_quads} quads, {part.n_tris} triangles; smallest {fl['smallest_m'] / 1e3:.3f}"
        f" km ({fl['below']} below the floor {h_min / 1e3:.2f} km)", flush=True)
    res = GroupResult(not fl["below"], rec, out, pieces, outline, [P for _, P in blocks], cross_xyz,
                      [xy[p.ids] for p in pieces if p.kind == "join"], H, rc, holes)
    if fl["below"]:
        rec["refused"] = [f"size floor: {fl['below']} cell(s) below h_min_m {h_min:.0f} m (smallest "
                          f"{fl['smallest_m']:.0f} m at {fl['at']})"]
    return res


def _sub(pl: Polyline, s0: float, s1: float, wrap: float | None = None) -> np.ndarray:
    from meshom.insert.realcoast import _sub as sub

    return sub(pl, s0, s1, wrap=wrap)


def smooth_bank_ends(xy: np.ndarray, sigma: float) -> np.ndarray:
    """B245: a block's bank smoothed by a Gaussian of `sigma` (chart metres) along it, its two ends held (the cross
    lines' ends on the coast): a cape or a pocket narrower than ~sigma is cut, the block covers the cut land or the
    filled water (the depth rule keeps the section's area); sigma 0: the bank as it is."""
    from meshom.insert.realcoast import _arc, _resample, _smooth_var

    if sigma <= 0 or len(xy) < 4:
        return xy
    q = _resample(xy, False, max(Polyline(xy).L / 400.0, 1.0))
    s = _arc(q, False)
    return _smooth_var(q, s, np.full(len(q), float(sigma)), False, float(s[-1]))


def merge_parts(parts: list) -> Mesh:
    """The parts (meshes on the sphere) as one mesh: nodes within MERGE_CHORD of each other are one node (the first
    one's position); every cell counter-clockwise."""
    from meshom.mesh import orient_ccw

    P = np.vstack([p.points for p in parts])
    off = np.cumsum([0] + [p.n_nodes for p in parts])
    tree = cKDTree(P)
    rep = np.arange(len(P))
    for i, j in sorted(tree.query_pairs(MERGE_CHORD)):
        a, b = rep[i], rep[j]
        while rep[a] != a:
            a = rep[a]
        while rep[b] != b:
            b = rep[b]
        if a != b:
            rep[max(a, b)] = min(a, b)
    for i in range(len(rep)):
        r = i
        while rep[r] != r:
            r = rep[r]
        rep[i] = r
    uniq, inv = np.unique(rep, return_inverse=True)
    tris = np.vstack([inv[p.tris + off[k]] for k, p in enumerate(parts) if p.n_tris] or [np.zeros((0, 3), np.int64)])
    quads = np.vstack([inv[p.quads + off[k]] for k, p in enumerate(parts) if p.n_quads] or [np.zeros((0, 4), np.int64)])
    m = Mesh(P[uniq].copy(), tris.astype(np.int64), quads.astype(np.int64), np.zeros(len(uniq), np.int8))
    return orient_ccw(m)


def channel_cells(full: Mesh, block_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(cells (T+Q,) bool, nodes) of one channel's block in the spliced mesh: the cells whose nodes are all the
    block's nodes."""
    isb = np.zeros(full.n_nodes, bool)
    isb[block_ids] = True
    T = full.n_tris
    return np.r_[isb[full.tris].all(1) if T else np.zeros(0, bool), isb[full.quads].all(1)], isb


def branch_depth(full: Mesh, depth: np.ndarray, blocks: list, cfg, sill_level: str, depth_channel: str,
                 h_fine: list, log=print, tag: str = "") -> tuple[np.ndarray, list]:
    """B244: each channel's own sill: on the block's cells and one ring of cells round them (the patches beyond its
    cross lines), the channel line between its two cross lines (meshom.insert.channel.channel_line: the cross
    lines are the line's two joins, the block's banks its banks) and, with sill_level = "grid", the grid's saddle
    between them held at its level by a dam of the block's nodes (channel.grid_sill); with depth_channel the channel
    raise on the block's nodes (channel.line_raise). blocks: per channel (block node ids in `full`, cross-line node
    ids in `full`). Returns (depth, records)."""
    from pathlib import Path

    from meshom.bathymetry import levels
    from meshom.insert import channel as chan
    from meshom.mesh import signed_area as sarea

    b = cfg.bathymetry
    out = np.asarray(depth, dtype=np.float64).copy()
    recs = []
    T = full.n_tris
    for k, (bids, xids) in enumerate(blocks):
        cin, isb = channel_cells(full, bids)
        near = np.zeros(full.n_nodes, bool)
        near[np.r_[full.tris[cin[:T]].ravel(), full.quads[cin[T:]].ravel()]] = True
        ring = np.r_[near[full.tris].any(1) if T else np.zeros(0, bool), near[full.quads].any(1)]
        wid = np.unique(np.r_[full.tris[ring[:T]].ravel(), full.quads[ring[T:]].ravel()])
        loc = np.full(full.n_nodes, -1, np.int64)
        loc[wid] = np.arange(len(wid))
        sub = Mesh(full.points[wid], loc[full.tris[ring[:T]]].reshape(-1, 3), loc[full.quads[ring[T:]]].reshape(-1, 4),
                   np.zeros(len(wid), np.int8))
        ins = isb[wid].copy()
        ins[loc[xids]] = False                              # the cross lines are the channel's two joins
        lo_, la_ = xyz_to_lonlat(sub.points)
        r_max = float(np.sqrt(np.abs(sarea(sub)).max()))
        grid = chan.read_window(Path(b.source), b.variable, lo_, la_, 2.0 * r_max)
        r = {"channel": k, "nodes": int(ins.sum()), "cross_nodes": int(len(xids)), "cells": int(cin.sum()),
             "joins": [int(len(p[0])) for p in chan._host_joins(sub, ins)]}
        line = chan.channel_line(sub, ins, grid, h_fine[k])
        if line is None:
            r["skipped"] = "no two cross lines"
            recs.append(r)
            continue
        r["line"] = line.record
        d = out[wid]
        if depth_channel:
            d, r["raise"] = chan.line_raise(sub, d, ins, grid, line, depth_channel)
        if sill_level in ("grid", "nearest"):
            d, r["sill"] = chan.grid_sill(sub, d, ins, line, levels(b.levels), nearest=sill_level == "nearest")
        out[wid[ins]] = d[ins]
        s = r.get("sill", {})
        log(f"{tag} channel {k}: line {line.record['km']:.0f} km, deepest connection "
            f"{line.record['deepest_connection_box_m']:.0f} m, grid saddle {line.saddle[0]:.0f} m at "
            f"{line.record['grid_saddle']['at']}; {('dam ' + str(s.get('chain_nodes')) + ' nodes at ' + str(round(s['cap_m'], 1)) + ' m') if 'cap_m' in s else s}",
            flush=True)
        recs.append(r)
    return out, recs


def insert_group(win, spec, log=print, real=None, h_min: float | None = None):
    """The group insert of a Spec (kind "group": spec.path = the envelope, spec.channels) in the host window, as a
    meshom.insert.core.Result for the run-folder wrapper (core.run): window = the window mesh with the group in place
    of the envelope region; band = the maps' centre, joins and fine size; group = the GroupResult."""
    from meshom.insert.core import Result, _path_frame

    c, _ = _path_frame(spec.path)
    chart = Chart(c)
    env_xy = chart.fwd(spec.path)
    h_min = float(spec.h_min_m if spec.h_min_m is not None else (h_min or 0.0))
    gap_m = spec.gap_km * 1e3 if spec.gap_km is not None else GAP_JOIN_CELLS * h_min
    growth = float(spec.growth if spec.growth is not None else GROWTH)
    lo, la = xyz_to_lonlat(c[None])
    state = {}
    try:
        g = build(win, chart, env_xy, spec.channels, real if spec.coast == "real" else None, h_min, gap_m,
                  spec.island_km2, growth, spec.bank_sigma_cells, log, spec.name, land_cells=spec.bank_land_cells,
                  gap_cells=spec.gap_cells, bank_smooth=spec.bank_smooth_cells, state=state,
                  blend_cells=spec.blend_cells, growth_max=spec.growth_max,
                  fill=[chart.fwd(f) for f in spec.fill])
    except InsertError as ex:
        log(f"[insert {spec.name}] REFUSED {ex}", flush=True)
        res = Result(False, {"name": spec.name, "refused": [str(ex)]}, host_window=win, chart=chart,
                     outline=state.get("outline"))
        res.real, res.state = state.get("rc"), state
        return res
    rec = g.record
    from meshom.insert.assemble import MeshH

    h_host = float(np.median(MeshH(win.mesh)(chart.inv(env_xy))))
    rec["profile"] = {"h_host_km": h_host / 1e3, "neck_lonlat": [float(lo[0]), float(la[0])],
                      "h_min_km": h_min / 1e3, "coast": "real" if g.real is not None else "host"}
    res = Result(g.ok, rec, g.window, g.window, win, chart, None, [], [], g.outline, g.h, None)
    res.real = g.real
    fine = [ch["cell_min_km"] * 1e3 for ch in rec["channels"]] or [h_host]
    zooms = []
    for ch in spec.channels:
        a, b = chart.fwd(ch.a), chart.fwd(ch.b)
        r = 0.75 * max(np.linalg.norm(a[1] - a[0]), np.linalg.norm(b[1] - b[0]), np.linalg.norm(b.mean(0) - a.mean(0)))
        zooms.append((ch.name, 0.5 * (a.mean(0) + b.mean(0)), float(r)))
    res.band = {"mid": env_xy.mean(0), "chains_xy": g.joins_xy, "h_fine": float(min(fine)), "zooms": zooms}
    res.group = g
    return res


def ids_in(full: Mesh, g: GroupResult) -> list:
    """B244: per channel, (the block's node ids, its two cross lines' node ids) in the spliced mesh `full`, by
    position (before any node moves)."""
    tree = cKDTree(full.points)
    out = []
    for P, (zA, zB) in zip(g.blocks_xyz, g.cross_xyz):
        d, b = tree.query(P)
        dx, x = tree.query(np.vstack([zA, zB]))
        assert d.max() < MERGE_CHORD and dx.max() < MERGE_CHORD
        out.append((np.unique(b), np.unique(x)))
    return out


def route_banks(sub: Mesh, ins: np.ndarray, line):
    """B246: the two banks of a route through a group, for the dam (meshom.insert.channel.grid_sill's `sides`): the
    graph of the nodes of the group's cells (neighbours: nodes of one cell) and the coast nodes of the group's cells
    on each side of the route's line (the side of the line's local direction over CHANNEL_DIR_M at the nearest line
    point; coast nodes nearest the line's two end points belong to neither). Only the open coast pieces (the outer
    boundary between the joins) are banks: an island's closed coast lies between the passages, and a chain from bank
    to bank that crosses it cuts every passage beside it. Returns (graph, bank A mask, bank B mask) or a reason
    (str)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from meshom.geo import lonlat_to_xyz
    from meshom.insert.channel import CHANNEL_DIR_M

    n = sub.n_nodes
    T = sub.n_tris
    cells = [c[ins[c].any(1)] for c in (sub.tris, sub.quads) if len(c)]
    pairs = np.vstack([np.c_[c[:, i], c[:, j]] for c in cells for i in range(c.shape[1])
                       for j in range(i + 1, c.shape[1])])
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()
    g = ((g + g.T) > 0).tocsr()
    topo = build_topology(sub)
    bnd = topo.edge_elems[:, 1] < 0
    elem_ins = np.r_[ins[sub.tris].any(1) if T else np.zeros(0, bool), ins[sub.quads].any(1)]
    ce = topo.edges[bnd][elem_ins[topo.edge_elems[bnd, 0]]]
    if not len(ce):
        return "the group has no coast edges"
    nc, lab = connected_components(coo_matrix((np.ones(len(ce)), (ce[:, 0], ce[:, 1])), shape=(n, n)),
                                   directed=False)
    on = np.zeros(n, bool)
    on[ce.ravel()] = True
    deg = np.bincount(ce.ravel(), minlength=n)
    open_ = np.zeros(nc, bool)
    open_[lab[on & (deg != 2)]] = True                       # a piece with an end: not a closed island loop
    bank = on & open_[lab]
    P = lonlat_to_xyz(line.lon, line.lat)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))] * R_EARTH_M
    _, j = cKDTree(P).query(sub.points)
    ia = np.clip(np.searchsorted(s, s[j] - CHANNEL_DIR_M, side="left"), 0, len(P) - 1)
    ib = np.clip(np.searchsorted(s, s[j] + CHANNEL_DIR_M, side="right") - 1, 0, len(P) - 1)
    side = np.einsum("ij,ij->i", np.cross(P[j], P[ib] - P[ia]), sub.points - P[j])
    inner = (j > 0) & (j < len(P) - 1)
    left, right = bank & inner & (side > 0), bank & inner & (side < 0)
    if not left.any() or not right.any():
        return "the route's line has open coast on one side only"
    return g, left, right


def route_depth(full: Mesh, depth: np.ndarray, own: np.ndarray, routes: list, cfg, sill_level: str,
                depth_channel: str, depth_area: bool, h_fine: float, log=print, tag: str = "") -> tuple:
    """B246: the depth rules along each route through the group (routes: (name, end a (3,), end b (3,)), the ends in
    the basins), on the group's own nodes `own` (of `full`), computed on the group's cells and one cell ring round
    them. Per route, on the grid's box means at h_fine (half-width h_fine / 2): the route's line, the deepest path
    from the wet cell nearest end a to the one nearest end b (as the channel line, B213: cost per km (D_ref /
    D)^CHANNEL_POWER) in the corridor of the group's cells and the ring widened by h_fine (the whole window when the
    corridor does not join them), the grid's saddle between the ends (on the grid's cells) and the line's narrows
    round its narrowest point in the group's cells; then with depth_channel the channel raise on the group's nodes
    along the narrows (meshom.insert.channel.line_raise), across blocks and patches alike; with sill_level = "grid"
    the dam (channel.grid_sill) from bank to bank through the saddle, its banks the open coast on each side of the
    line (route_banks), so that it cuts every passage of the group beside islands; with depth_area the section-area
    rule (meshom.insert.area.area_depth, B250) along the route's line, its dam's cap kept, the deepest connection
    between the route's ends within a level of its grid saddle's (B251) and the other routes' dam nodes left alone. Each end is the deepest box
    mean within ROUTE_END_H x h_fine of the given point (in the corridor when it has one there). A route with a
    corridor width keeps its line, saddle and narrows within half that width of the polyline a, via..., b (a strait
    beside a deeper one would otherwise take the deeper one's path). Returns (depth, records)."""
    from pathlib import Path

    from scipy import ndimage

    from meshom import run as runmod
    from meshom.bathymetry import levels
    from meshom.geo import lonlat_to_xyz
    from meshom.insert import channel as chan
    from meshom.insert.area import area_depth
    from meshom.straits import locate

    b = cfg.bathymetry
    T = full.n_tris

    def cells_of(m):
        return np.r_[m[full.tris].any(1) if T else np.zeros(0, bool), m[full.quads].any(1)]

    near = np.zeros(full.n_nodes, bool)
    c = cells_of(own)
    near[full.tris[c[:T]].ravel()] = True
    near[full.quads[c[T:]].ravel()] = True
    c = cells_of(near)
    wid = np.unique(np.r_[full.tris[c[:T]].ravel(), full.quads[c[T:]].ravel()])
    loc = np.full(full.n_nodes, -1, np.int64)
    loc[wid] = np.arange(len(wid))
    sub = Mesh(full.points[wid], loc[full.tris[c[:T]]].reshape(-1, 3), loc[full.quads[c[T:]]].reshape(-1, 4),
               np.zeros(len(wid), np.int8))
    ins = own[wid]
    zbar = levels(b.levels)
    floor = runmod.depth_floor_m(cfg, cfg.mesh.format)
    ends = np.array([p for _, pa, pb, *_ in routes for p in (pa, pb)])
    lo_, la_ = xyz_to_lonlat(np.vstack([sub.points, ends]))
    r_max = float(np.sqrt(np.abs(signed_area(sub)).max()))
    grid = chan.read_window(Path(b.source), b.variable, lo_, la_, 2.0 * r_max)
    D = grid.smoothed(0.5 * h_fine)
    fp_own = chan.footprint(sub, ins, grid)
    fp_all = chan.footprint(sub, np.ones(sub.n_nodes, bool), grid)
    dy = R_EARTH_M * np.radians(abs(grid.dlat))
    dx = R_EARTH_M * np.radians(abs(grid.dlon)) * max(np.cos(np.radians(float(np.mean(grid.lat)))), 1e-6)
    ki, kj = int(np.ceil(h_fine / dy)), int(np.ceil(h_fine / dx))
    corr0 = ndimage.maximum_filter(fp_all, size=(2 * ki + 1, 2 * kj + 1))
    sq = np.ones((3, 3), bool)
    d = np.asarray(depth, dtype=np.float64)[wid].copy()
    lines, recs = [], []
    II, JJ = np.meshgrid(np.arange(len(grid.lat)), np.arange(len(grid.lon)), indexing="ij")

    def end_cell(p, cm):                                   # the deepest box mean near the given end
        lo, la = xyz_to_lonlat(p[None])
        i, j = chan._wet_cell(grid, D, float(lo[0]), float(la[0]))
        r = ROUTE_END_H * h_fine
        m = np.isfinite(D) & (((II - i) * dy) ** 2 + ((JJ - j) * dx) ** 2 <= r * r)
        if (m & cm).any():
            m &= cm
        k = int(np.argmax(np.where(m, D, -1.0)))
        return int(II.ravel()[k]), int(JJ.ravel()[k])

    for name, pa, pb, via, width in routes:
        corr = corr0.copy()
        if width:                                          # the route's own corridor round a, via..., b
            from meshom import straits as stm

            pl = stm.densify(np.vstack([pa[None], via, pb[None]]) if via is not None else np.vstack([pa, pb]),
                             0.25 * h_fine)
            cell = lonlat_to_xyz(grid.lon[JJ.ravel()], grid.lat[II.ravel()])
            dd, _ = cKDTree(pl).query(cell, distance_upper_bound=2.0 * np.sin(0.25 * width / R_EARTH_M))
            corr &= np.isfinite(dd).reshape(corr.shape)
        ea, eb = end_cell(pa, corr), end_cell(pb, corr)
        corr[ea] = corr[eb] = True
        Dc = np.where(corr, D, np.nan)
        lab, _ = ndimage.label(np.isfinite(Dc), structure=sq)
        if not lab[ea] or lab[ea] != lab[eb]:
            Dc, corr = D, None
        d_ref, _ = chan._maximin(Dc, ea, eb, sq)
        ii, jj = chan._path(grid, Dc, ea, eb, d_ref)
        k = int(np.argmin(D[ii, jj]))
        sd = chan._saddle(grid, ea, eb, corr, near=(int(ii[k]), int(jj[k])))
        plon, plat = grid.lon[jj], grid.lat[ii]
        narrow, nrec = chan.narrows(grid, plon, plat, fp_own[ii, jj])
        xyz = lonlat_to_xyz(plon, plat)
        km = float(R_EARTH_M / 1e3 * np.arccos(np.clip(np.sum(xyz[1:] * xyz[:-1], 1), -1, 1)).sum())
        pt = lambda cc: [round(float(grid.lon[cc[1]]), 4), round(float(grid.lat[cc[0]]), 4)]   # noqa: E731
        lrec = {"ends": [pt(ea), pt(eb)], "end_box_m": [float(D[ea]), float(D[eb])], "corridor": corr is not None,
                "box_half_width_m": 0.5 * h_fine, "cells": int(len(ii)), "km": km,
                "deepest_connection_box_m": float(d_ref),
                "shallowest_on_line": {"m": float(D[ii[k], jj[k]]),
                                       "at": [round(float(plon[k]), 4), round(float(plat[k]), 4)]},
                "grid_saddle": {"m": sd[0], "at": [round(sd[1], 4), round(sd[2], 4)]}, "narrows": nrec}
        line = chan.Line(plon, plat, D[ii, jj], sd, lrec, narrow)
        r = {"route": name, "line": lrec}
        if depth_channel:
            d, r["raise"] = chan.line_raise(sub, d, ins, grid, line, depth_channel)
        lines.append(line)
        recs.append(r)
    held = [None] * len(routes)
    if sill_level in ("grid", "nearest"):
        for i, (line, r) in enumerate(zip(lines, recs)):
            sides = route_banks(sub, ins, line)
            d, r["sill"] = (chan.grid_sill(sub, d, ins, line, zbar, sides=sides, nearest=sill_level == "nearest")
                            if not isinstance(sides, str) else (d, {"skipped": sides}))
            held[i] = r["sill"].pop("held", None)
    if depth_area:
        on_line = np.zeros(sub.n_nodes, bool)
        Ts = sub.n_tris
        for line in lines:
            el = locate(sub, lonlat_to_xyz(line.lon, line.lat))
            el = np.unique(el[el >= 0])
            on_line[sub.tris[el[el < Ts]].ravel()] = True
            on_line[sub.quads[el[el >= Ts] - Ts].ravel()] = True
        for i, (line, r) in enumerate(zip(lines, recs)):
            other = np.zeros(sub.n_nodes, bool)
            for j_, h in enumerate(held):
                if h is not None and j_ != i:
                    other[np.asarray(h[0], np.int64)] = True
            ax = lonlat_to_xyz(line.lon, line.lat)
            d, r["area"] = area_depth(sub, d, ins & ~other, ax, zbar, floor, Path(b.source), b.variable, h_fine,
                                      held[i], on_line, f"{tag} route {r['route']}", log,
                                      saddle_m=line.saddle[0])                 # B251: the band, the route's ends
    for line, r in zip(lines, recs):
        s, lr = r.get("sill", {}), r["line"]
        raise_ = f"; raise {r['raise']['nodes_raised']} nodes" if "raise" in r else ""
        dam = (f"; dam {s.get('chain_nodes')} nodes at {s['cap_m']:.1f} m" if "cap_m" in s
               else (f"; {s}" if s else ""))
        log(f"{tag} route {r['route']}: line {lr['km']:.0f} km, deepest connection "
            f"{lr['deepest_connection_box_m']:.0f} m, grid saddle {line.saddle[0]:.0f} m at {lr['grid_saddle']['at']};"
            f" narrows {lr['narrows']['from']} - {lr['narrows']['to']}{raise_}{dam}", flush=True)
    out = np.asarray(depth, dtype=np.float64).copy()
    out[wid[ins]] = d[ins]
    return out, recs


def split_obtuse(full: Mesh, window: Mesh, own: np.ndarray, blocks: list, deg: float) -> tuple:
    """B243 (`split_deg`): every patch quad of the spliced mesh (a quad of the insert's window part, meshom.insert.core
    ._part, with a node of `own`, not a block's cell: blocks as ids_in) whose largest corner (tangent plane) exceeds
    `deg` becomes two triangles along the diagonal from that corner (orientation kept; no node added or moved, so the
    node ids and the depth bookkeeping stay). The triangles go to the end of the triangle list and the window's part
    stays the tail of both lists. Returns (mesh, the window part as a mesh on the same points, record)."""
    from meshom.quality.kernel import element_geometry

    T, Q = full.n_tris, full.n_quads
    q0 = Q - window.n_quads
    isb = np.zeros(full.n_nodes, bool)
    for b, _ in blocks:
        isb[b] = True
    cand = np.arange(q0, Q)
    cq = full.quads[cand]
    cand = cand[own[cq].any(1) & ~isb[cq].all(1)]
    rec = {"deg": float(deg), "split": 0, "worst_deg": 0.0}
    if not len(cand):
        return full, window, rec
    sub = Mesh(full.points, np.zeros((0, 3), np.int64), full.quads[cand], full.node_flag)
    qa = element_geometry(sub).quad_angles
    worst = np.degrees(qa.max(1))
    bad = worst > deg
    rec["worst_deg"] = float(worst.max())
    if not bad.any():
        return full, window, rec
    q = full.quads[cand[bad]]
    r = np.argmax(qa[bad], axis=1)
    i = np.arange(len(q))
    a, b, c, d = q[i, r], q[i, (r + 1) % 4], q[i, (r + 2) % 4], q[i, (r + 3) % 4]
    keep = np.ones(Q, bool)
    keep[cand[bad]] = False
    tris = np.vstack([full.tris, np.c_[a, b, c], np.c_[a, c, d]]).astype(np.int64)
    tz = None if full.tri_zone is None else np.r_[full.tri_zone, np.zeros(2 * len(q), bool)]
    out = Mesh(full.points, tris, full.quads[keep], full.node_flag, tz)
    nt, nq = window.n_tris + 2 * len(q), window.n_quads - len(q)
    win = Mesh(full.points, out.tris[out.n_tris - nt:], out.quads[out.n_quads - nq:], full.node_flag)
    rec.update(split=int(len(q)), worst_after_deg=float(worst[~bad].max()) if (~bad).any() else 0.0)
    return out, win, rec


EVEN_RINGS = 2               # B252: the host cell rings round the insert whose nodes the area pass may move too
EVEN_BLOCKS = ""             # B252: the blocks' nodes in the area pass by default (`even_blocks`): "" free, "cross"
                             # the cross lines held, "block" every block node held


def even_group(full: Mesh, window: Mesh, own: np.ndarray, blocks: list, fixed: np.ndarray, h_min: float,
               sweeps: int, rings: int = EVEN_RINGS, limit_deg: float = 120.0, log=print, tag: str = "",
               verbose: bool = False, hold: str | None = None) -> tuple:
    """B252: meshom.insert.even.even_areas on the group's patches and the host rings round them, in the spliced mesh:
    the free nodes are the insert's own nodes (`own`; hold (default EVEN_BLOCKS) "cross": not the cross lines' nodes,
    "block": no block's node, `blocks` as ids_in; "": all, as fix_rings moves them; a block's banks are coast and
    stay, its cells stay its cells) and the host nodes of the cells within `rings` element rings of the own cells
    (the size steps then spread into the host's cells next to the joins); `fixed` (the window's cut) and the mesh's
    boundary (the coast) never move. Returns (mesh, the window part with the new positions, record)."""
    from meshom.insert.even import even_areas

    T = full.n_tris
    isb = np.zeros(full.n_nodes, bool)
    hold = hold or EVEN_BLOCKS
    for b, x in blocks:
        if hold in ("cross", "block"):                     # the cross lines' nodes stay
            isb[x] = True
        if hold == "block":                                # and the blocks' own nodes
            isb[b] = True

    def touch(mask):
        return np.r_[mask[full.tris].any(1) if T else np.zeros(0, bool), mask[full.quads].any(1)]

    reach = own.copy()
    for _ in range(int(rings)):
        c = touch(reach)
        reach = reach.copy()
        reach[np.r_[full.tris[c[:T]].ravel(), full.quads[c[T:]].ravel()]] = True
    c = touch(np.asarray(fixed, bool))                       # no node of a cell on the window's cut
    bad = np.zeros(full.n_nodes, bool)
    bad[np.r_[full.tris[c[:T]].ravel(), full.quads[c[T:]].ravel()]] = True
    mov = reach & ~isb & ~bad
    d, gid = cKDTree(full.points).query(window.points)
    out, rec = even_areas(full, mov, h_min, limit_deg, sweeps, log=lambda s: log(f"{tag} {s}", flush=True),
                          verbose=verbose)
    rec["rings"] = int(rings)
    rec["host_nodes_free"] = int((mov & ~own).sum())
    win = Mesh(out.points[gid], window.tris, window.quads, window.node_flag, window.tri_zone)
    return out, win, rec
