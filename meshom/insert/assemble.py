"""Fitting a block into the host window: the host cells between the joins, the banks, the merge by shared node
ids, the insert's h layer and the cleaning of the join ring with the mesh-fixing pipeline (decisions B161, B162).
"""
import dataclasses
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from meshom.geo import R_EARTH_M
from meshom.insert.region import InsertError
from meshom.mesh import Mesh, build_topology, signed_area

COAST_RULE_H = 0.25          # B161: the coast rule: a coast node stays within this x h_run of the host coast
COAST_CAP_H = 0.2            # B161: a smoothed bank stays within this x h_run of the host coast (margin to the rule)
BANK_SIGMA_H = 1.5           # B161: bank smoothing (Gaussian along the arc) in host coast cells
CLEAN_RINGS = 3              # B162: the cleaned ring = elements within this many element rings of a join node...
COLLAR_RINGS = 10            # B192: the collar's default width on an irregular host (host element rings)
COLLAR_CORE = 4              # B192: ... and it reaches this many cell rings into the block's core
COLLAR_SMOOTH = 0            # B192: sweeps of meshom.smooth (kind "size") on the collar before the stages (20 sweeps
                             # gained ~0.01-0.02 in the size ratio p99 and cost up to 24 deg at a corner)
FIXED_CORE = 8               # B193: a structured host kept: the block's cells this many rings into its core ...
FIXED_SMOOTH = 20            # B193: ... relaxed by this many sweeps of meshom.smooth before the stages


def collar_opts(host_fixed: bool) -> dict:
    """The cleaning's collar options (B192, B193): core rings and smoothing sweeps."""
    if host_fixed:
        return {"core_rings": FIXED_CORE, "smooth_iters": FIXED_SMOOTH, "host_fixed": True}
    return {"core_rings": COLLAR_CORE, "smooth_iters": COLLAR_SMOOTH, "host_fixed": False}
RING_ROWS = 2                # B162: ... and the block's first RING_ROWS rows at each join; the core is fixed
STAGES = "d,b,e,a"           # B162: the pipeline's stages on the ring (B149's default order)
HOST_SHIFT_H = 0.25          # B162 / B168: the host's coast outside the insert moves at most this x h_run (a gate)
MATCH_CHORD = 1e-12
KEEP_LIMIT_DEG = 120.0       # B220: the angle excess counts each quad's largest corner above this
KEEP_DEG = 0.5               # B220: fix_rings keeps its result unless its angle excess (or, tied, its worst corner)
                             # grows by more than this ...
KEEP_RATIO = 0.005           # B220: ... ties of the angles go to the sizes: ratio p99 differences above this count,
KEEP_CELL_HOST = 0.01        # B220: ... cell / host differences above this,
KEEP_SMALLEST = 0.01         # B220: ... and smallest-cell differences above this fraction


def _key(e: np.ndarray, n: int) -> np.ndarray:
    return np.minimum(e[:, 0], e[:, 1]) * n + np.maximum(e[:, 0], e[:, 1])


def elem_nodes(mesh: Mesh) -> list:
    return [mesh.tris, mesh.quads]


def region_between(mesh: Mesh, seed: int, chains: list, kind: np.ndarray) -> np.ndarray:
    """(T+Q,) bool: the elements reachable from `seed` across interior edges that are not chain edges. Refuses a
    region that leaks to the window's cut (the joins do not close it), is not a disc (an island inside), or whose
    boundary does not hold both chains."""
    topo = build_topology(mesh)
    n = mesh.n_nodes
    walls = np.concatenate([_key(np.c_[c[:-1], c[1:]], n) for c in chains])
    ek = _key(topo.edges, n)
    inner = (topo.edge_elems[:, 1] >= 0) & ~np.isin(ek, walls)
    a, b = topo.edge_elems[inner, 0], topo.edge_elems[inner, 1]
    E = mesh.n_elems
    A = coo_matrix((np.ones(len(a)), (a, b)), shape=(E, E))
    _, lab = connected_components(A, directed=False)
    reg = lab == lab[seed]
    T = mesh.n_tris
    sub = Mesh(mesh.points, mesh.tris[reg[:T]], mesh.quads[reg[T:]], mesh.node_flag)
    used = np.unique(np.r_[sub.tris.ravel(), sub.quads.ravel()])
    if (kind[used] == 2).any():
        raise InsertError("the joins do not close the strait: the host cells between them reach the window's edge "
                          "(a coast gap or a second channel around the joins)")
    st = build_topology(sub)
    be = st.edges[st.edge_elems[:, 1] < 0]
    chi = len(used) - len(st.edges) + int(reg.sum())
    if chi != 1:
        raise InsertError(f"the region between the joins is not a disc (Euler characteristic {chi}: an island or a "
                          "lake inside it); step 1 builds single blocks only")
    if not np.isin(walls, _key(be, n)).all():
        raise InsertError("a join is not on the region's boundary (the two joins cross or touch)")
    return reg


def banks(mesh: Mesh, reg: np.ndarray, cA: np.ndarray, cB: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two coast node paths of the region's boundary: S from cA[0] to cB[0], N from cA[-1] to cB[-1]."""
    T = mesh.n_tris
    sub = Mesh(mesh.points, mesh.tris[reg[:T]], mesh.quads[reg[T:]], mesh.node_flag)
    st = build_topology(sub)
    be = st.edges[st.edge_elems[:, 1] < 0]
    n = mesh.n_nodes
    walls = np.concatenate([_key(np.c_[c[:-1], c[1:]], n) for c in (cA, cB)])
    be = be[~np.isin(_key(be, n), walls)]
    nb = {}
    for x, y in be:
        nb.setdefault(int(x), []).append(int(y))
        nb.setdefault(int(y), []).append(int(x))

    def walk(a, goal):
        path, prev = [int(a)], -1
        while path[-1] != goal:
            nxt = [v for v in nb.get(path[-1], []) if v != prev]
            if len(nxt) != 1 and not (len(path) == 1 and len(nxt) >= 1):
                raise InsertError("the region's coast is not a simple path between the joins (a pinch node)")
            prev = path[-1]
            path.append(nxt[0])
            if len(path) > len(be) + 2:
                raise InsertError("the region's coast does not connect the joins' ends as expected")
        return np.array(path)
    return walk(cA[0], int(cB[0])), walk(cA[-1], int(cB[-1]))


@dataclass
class Merged:
    mesh: Mesh                   # the window with the block in place of the region
    block_t: np.ndarray          # (T,) bool: the block's triangles
    block_q: np.ndarray          # (Q,) bool: the block's quads
    core_t: np.ndarray           # (T,) bool: the block's core (rows RING_ROWS .. M - RING_ROWS)
    core_q: np.ndarray
    join_nodes: np.ndarray       # node ids of the two chains (merged numbering)
    inverted: int
    host_node: np.ndarray | None = None   # (N,) bool: the node is a host window node at its own position
    join_ends: np.ndarray | None = None   # the joins' coast end nodes (cA[0], cA[-1], cB[0], cB[-1])


def merge(win_mesh: Mesh, reg: np.ndarray, cA: np.ndarray, cB: np.ndarray, blk, new_xyz: np.ndarray,
          ring_rows: int = RING_ROWS) -> Merged:
    """merge_net of a single block (its core = rows ring_rows .. M - ring_rows)."""
    from meshom.insert.network import from_block

    return merge_net(win_mesh, reg, cA, cB, from_block(blk, ring_rows, lambda p: np.ones(len(p))), new_xyz)


def merge_net(win_mesh: Mesh, reg: np.ndarray, cA: np.ndarray, cB: np.ndarray, net, new_xyz: np.ndarray) -> Merged:
    """The window without the region's elements plus the network's (join rows = the chains' host nodes, other
    network nodes appended, host nodes no element uses any more dropped); cells oriented CCW."""
    T = win_mesh.n_tris
    cB = np.asarray(cB, dtype=np.int64)
    loc = np.full(len(net.xy), -1, np.int64)
    loc[net.ids_a] = cA
    if len(cB):                                    # a coastal strip has one join chain (B181)
        loc[net.ids_b] = cB
    inner = loc < 0
    n0 = win_mesh.n_nodes
    loc[inner] = n0 + np.arange(inner.sum())
    pts = np.vstack([win_mesh.points, new_xyz[inner]])
    bq, bt = loc[net.quads], loc[net.tris]
    tmp = Mesh(pts, bt, bq, np.zeros(len(pts), np.int8))
    a = signed_area(tmp)
    if (a < 0).sum() > len(a) / 2:
        bq, bt = bq[:, ::-1].copy(), bt[:, ::-1].copy()
        a = -a
    ct, cq = np.asarray(net.core_t, bool), np.asarray(net.core_q, bool)
    tris = np.vstack([win_mesh.tris[~reg[:T]], bt]).astype(np.int64)
    quads = np.vstack([win_mesh.quads[~reg[T:]], bq]).astype(np.int64)
    used = np.zeros(len(pts), bool)
    used[tris.ravel()] = True
    used[quads.ravel()] = True
    keep = np.flatnonzero(used)
    re = np.full(len(pts), -1, np.int64)
    re[keep] = np.arange(len(keep))
    out = Mesh(pts[keep], re[tris].reshape(-1, 3), re[quads].reshape(-1, 4), np.zeros(len(keep), np.int8))
    out.node_flag = build_topology(out).boundary_node.astype(np.int8)
    kt, kq = int((~reg[:T]).sum()), int((~reg[T:]).sum())
    return Merged(out, np.r_[np.zeros(kt, bool), np.ones(len(bt), bool)],
                  np.r_[np.zeros(kq, bool), np.ones(len(bq), bool)], np.r_[np.zeros(kt, bool), ct],
                  np.r_[np.zeros(kq, bool), cq], re[np.r_[cA, cB]], int((a <= 0).sum()), keep < n0,
                  re[np.array([c[e] for c in (cA, cB) if len(c) for e in (0, -1)])])


class InsertH:
    """The insert's h layer composed with the run's h (B162: h stays authoritative, the insert's zoom is a
    layer, never an h read back from the cells): h(x) = min(h_run(x), min over ALL the block's size-target samples
    p_k of h_k + G x |x - p_k|) (chord distance). Exact: the K_NEAR nearest samples give a first bound; a sample
    farther than the K-th one gives at least min(h_k) + G x d_K, so only queries where that can beat the bound search
    the samples within the bound's radius (the K nearest alone made h jump 33 m over 1 m and overestimate by
    33 %). Picklable (the pipeline's workers unpickle it)."""

    K_NEAR = 16

    def __init__(self, h_run, pts: np.ndarray, hk: np.ndarray, g: float):
        self.h_run, self.pts, self.hk, self.g = h_run, np.asarray(pts), np.asarray(hk), float(g)
        self._t = None

    def layer(self, xyz, cap=None):
        """The cone minimum at xyz; with cap (e.g. h_run there), min(cone, cap), searching only as far as needed."""
        if self._t is None:
            self._t = cKDTree(self.pts)
        q = np.atleast_2d(xyz)
        G = self.g * R_EARTH_M
        k = min(self.K_NEAR, len(self.pts))
        d, j = self._t.query(q, k=k)
        d, j = d.reshape(len(d), k), j.reshape(len(j), k)
        best = np.min(self.hk[j] + G * d, axis=1)
        bound = best if cap is None else np.minimum(best, cap)
        hmin = float(self.hk.min())
        if k < len(self.pts):
            need = np.flatnonzero(hmin + G * d[:, -1] < bound)
            if len(need):
                r = (bound[need] - hmin) / G
                for i, idx in zip(need, self._t.query_ball_point(q[need], r)):
                    if idx:
                        idx = np.asarray(idx)
                        best[i] = min(best[i], float(np.min(self.hk[idx] + G * np.linalg.norm(self.pts[idx] - q[i],
                                                                                              axis=1))))
        return best if cap is None else np.minimum(best, cap)

    def __call__(self, xyz):
        xyz = np.atleast_2d(xyz)
        return self.layer(xyz, cap=np.asarray(self.h_run(xyz), dtype=float))

    def __getstate__(self):
        return {"h_run": self.h_run, "pts": self.pts, "hk": self.hk, "g": self.g}

    def __setstate__(self, s):
        self.__dict__.update(s)
        self._t = None


class MeshH:
    """The mesh's own node sizes (mean incident edge length) at the nearest node: the cleaning then repairs shapes,
    not sizes (option clean_h = "mesh", B162). Picklable."""

    def __init__(self, mesh: Mesh):
        from meshom.geo import arc_length

        e = build_topology(mesh).edges
        L = R_EARTH_M * arc_length(mesh.points[e[:, 0]], mesh.points[e[:, 1]])
        deg = np.bincount(e.ravel(), minlength=mesh.n_nodes)
        self.h = np.bincount(e.ravel(), weights=np.repeat(L, 2), minlength=mesh.n_nodes) / np.maximum(deg, 1)
        self.p = mesh.points.copy()
        self._t = None

    def __call__(self, xyz):
        if self._t is None:
            self._t = cKDTree(self.p)
        return self.h[self._t.query(np.atleast_2d(xyz))[1]]

    def __getstate__(self):
        return {"h": self.h, "p": self.p}

    def __setstate__(self, s):
        self.h, self.p, self._t = s["h"], s["p"], None


def replace_part(mesh: Mesh, removed: np.ndarray, part: Mesh, anchors: np.ndarray) -> Mesh:
    """mesh without the elements `removed` (T+Q bool) plus the elements of `part`; part nodes at the exact position
    of an anchor node (mesh ids) take its id, the others are appended; unused nodes dropped."""
    T = mesh.n_tris
    d, j = cKDTree(mesh.points[anchors]).query(part.points)
    gid = np.where(d < MATCH_CHORD, anchors[j], -1)
    fresh = gid < 0
    gid[fresh] = mesh.n_nodes + np.arange(fresh.sum())
    pts = np.vstack([mesh.points, part.points[fresh]])
    tris = np.vstack([mesh.tris[~removed[:T]], gid[part.tris]]).astype(np.int64)
    quads = np.vstack([mesh.quads[~removed[T:]], gid[part.quads]]).astype(np.int64)
    used = np.zeros(len(pts), bool)
    used[tris.ravel()] = True
    used[quads.ravel()] = True
    keep = np.flatnonzero(used)
    re = np.full(len(pts), -1, np.int64)
    re[keep] = np.arange(len(keep))
    out = Mesh(pts[keep], re[tris].reshape(-1, 3), re[quads].reshape(-1, 4), np.zeros(len(keep), np.int8))
    out.node_flag = build_topology(out).boundary_node.astype(np.int8)
    return out


def collar_of(collar_rings, g, orca: bool | None = None) -> tuple[int, bool, dict]:
    """B192 / B193: (rings, host_fixed, record) of the collar: on a structured host (an ORCA run, orca = True, whose
    window is regular by meshom.insert.region.structured; orca = None: by the window alone) the host's nodes and
    cells stay (rings 0) unless collar_rings asks for a collar; elsewhere collar_rings or COLLAR_RINGS. meshom's own
    block meshes are as regular as ORCA in open water (core2_A2 at the Denmark Strait: every interior node), so the
    run's method decides."""
    from meshom.insert.region import structured

    st, share = structured(g)
    st = st and orca is not False
    rings = collar_rings if collar_rings is not None else (0 if st else COLLAR_RINGS)
    return (max(int(rings), 1), rings == 0, {"structured": bool(st), "regular_share": round(share, 4),
                                             "collar_rings": int(rings)})


def ring_case(m: Merged, coast_keys: np.ndarray, h, tol, gates, name: str, dist_seg: np.ndarray,
              ref_host: bool = False, rings: int = CLEAN_RINGS, host_fixed: bool = False, core_rings: int = 0):
    """The meshfix case of the join ring (B162): the elements within CLEAN_RINGS element rings of a join node minus
    the block's core; its boundary edges that are coast edges of the window are coast (kind 1), the rest (the cut
    around the ring and the core's outline) artificial (kind 2: fixed, no defect sink). Held as kind 2 too:
    the join chains' coast end nodes and the host's coast nodes left in the window, so the
    outside keeps the host's coast exactly and only the insert's own banks slide; the inner join nodes stay free (the
    cleaning repairs the join across it). Elements at a pinch node of the cut-out are left out (fixed) until none is
    left. Returns (case, keep mask (T+Q,))."""
    from meshom.meshfix.cases import Case, _basin_points

    mesh = m.mesh
    T = mesh.n_tris
    n = mesh.n_nodes
    near = np.zeros(n, bool)
    near[m.join_nodes] = True
    keep = np.zeros(mesh.n_elems, bool)
    for _ in range(rings):
        hit = np.r_[near[mesh.tris].any(1) if T else np.zeros(0, bool), near[mesh.quads].any(1)]
        keep |= hit
        near[mesh.tris[keep[:T]].ravel()] = True
        near[mesh.quads[keep[T:]].ravel()] = True
    keep |= np.r_[m.block_t & ~m.core_t, m.block_q & ~m.core_q]   # B166: every non-core block cell (the bays)
    keep &= ~np.r_[m.core_t, m.core_q]
    if host_fixed:                                          # B193: a structured host (ORCA) keeps every node and cell
        keep &= np.r_[m.block_t, m.block_q]
    for _ in range(core_rings):                             # B192: the collar reaches into the core
        near = np.zeros(n, bool)
        near[mesh.tris[keep[:T]].ravel()] = True
        near[mesh.quads[keep[T:]].ravel()] = True
        keep |= np.r_[near[mesh.tris].any(1) if T else np.zeros(0, bool), near[mesh.quads].any(1)] & \
            np.r_[m.core_t, m.core_q]
    for _ in range(1000):
        sub = Mesh(mesh.points, mesh.tris[keep[:T]], mesh.quads[keep[T:]], mesh.node_flag)
        st = build_topology(sub)
        be = st.edges[st.edge_elems[:, 1] < 0]
        cnt = np.bincount(be.ravel(), minlength=n)
        pinch = cnt > 2
        if not pinch.any():
            break
        keep &= ~np.r_[pinch[mesh.tris].any(1) if T else np.zeros(0, bool), pinch[mesh.quads].any(1)]
    used = np.unique(np.r_[mesh.tris[keep[:T]].ravel(), mesh.quads[keep[T:]].ravel()])
    loc = np.full(n, -1, np.int64)
    loc[used] = np.arange(len(used))
    sub = Mesh(mesh.points[used].copy(), loc[mesh.tris[keep[:T]]].reshape(-1, 3),
               loc[mesh.quads[keep[T:]]].reshape(-1, 4), np.zeros(len(used), np.int8))
    st = build_topology(sub)
    be = st.edges[st.edge_elems[:, 1] < 0]
    real = np.isin(_key(used[be], n), coast_keys)
    kind = np.zeros(sub.n_nodes, np.int8)
    kind[be[real].ravel()] = 1
    kind[be[~real].ravel()] = 2
    if m.join_ends is not None:
        je = loc[m.join_ends]
        kind[je[je >= 0]] = 2                               # the joins' coast end nodes stay where the host has them
    if m.host_node is not None:
        kind[(kind == 1) & m.host_node[used]] = 2           # the host's coast outside the insert
    sub.node_flag = st.boundary_node.astype(np.int8)
    seg = np.stack([sub.points[be[real][:, 0]], sub.points[be[real][:, 1]]], axis=1)
    if ref_host and len(dist_seg):                          # B162: slides measured against the host's coast
        m_ = 0.5 * (dist_seg[:, 0] + dist_seg[:, 1])
        d_, _ = cKDTree(sub.points).query(m_ / np.linalg.norm(m_, axis=1, keepdims=True))
        ext = np.sqrt(np.abs(signed_area(sub)).max()) / R_EARTH_M
        seg = dist_seg[d_ < 6 * ext]
    case = Case(name, sub, kind, h, np.asarray(h(sub.points)), seg, dist_seg, _basin_points(sub), be[~real],
                tol, gates or {}, {}, {"kind": "insert_ring"})
    return case, keep


def split_nonconvex(mesh: Mesh) -> tuple[Mesh, np.ndarray]:
    """B162 (step 2): every quad with a corner >= 180 deg (tangent plane) split into two triangles along the diagonal
    from that corner (both keep the orientation); the geometry stage needs a valid start (it refused to move any
    node when one ring quad was non-convex: Gibraltar, Bonifacio, Denmark). Returns (mesh, centroids of the split
    quads)."""
    from meshom.quality.kernel import element_geometry

    if not mesh.n_quads:
        return mesh, np.zeros((0, 3))
    g = element_geometry(mesh)
    qa = g.quad_angles
    bad = np.flatnonzero((qa >= np.pi).any(1))
    if not len(bad):
        return mesh, np.zeros((0, 3))
    r = np.argmax(qa[bad], axis=1)
    q = mesh.quads[bad]
    i = np.arange(len(bad))
    a, b, c, d = q[i, r], q[i, (r + 1) % 4], q[i, (r + 2) % 4], q[i, (r + 3) % 4]
    keep = np.ones(mesh.n_quads, bool)
    keep[bad] = False
    tris = np.vstack([mesh.tris, np.c_[a, b, c], np.c_[a, c, d]]).astype(np.int64)
    out = Mesh(mesh.points, tris, mesh.quads[keep], mesh.node_flag)
    return out, g.centroid[mesh.n_tris + bad]


def clean(m: Merged, coast_keys: np.ndarray, h, tol, gates, stages: str, dist_seg: np.ndarray, name: str,
          opts: dict | None = None, verbose: bool = True, ref_host: bool = False, rings: int = CLEAN_RINGS,
          clean_h: str = "insert", smooth_iters: int = 0, host_fixed: bool = False, core_rings: int = 0):
    """Run the pipeline stages on the ring case and put the result back into the merged window: (window mesh,
    record, case, cleaned case mesh). smooth_iters: sweeps of meshom.smooth (kind "size", against the case's h)
    on the case's free nodes before the stages (B192); host_fixed: the host's cells stay out of the case (B193)."""
    from meshom.meshfix.pipeline import parse_opts, run_stages

    if clean_h == "mesh":
        h = MeshH(m.mesh)
    case, keep = ring_case(m, coast_keys, h, tol, gates, name, dist_seg, ref_host, rings, host_fixed, core_rings)
    st = [s.strip() for s in stages.split(",") if s.strip()]
    rec = {"stages": st, "ring_elems": int(keep.sum()), "ring_nodes": int(case.mesh.n_nodes),
           "ring_fixed_nodes": int((case.kind == 2).sum()), "ring_sliding_nodes": int((case.kind == 1).sum())}
    if not st:
        return m.mesh, rec, case, case.mesh
    case0 = case
    sm, where = split_nonconvex(case.mesh)
    if len(where):
        from meshom.geo import xyz_to_lonlat

        case = dataclasses.replace(case, mesh=sm)
        rec["split_nonconvex"] = [[round(float(v), 4) for v in ll] for ll in np.c_[xyz_to_lonlat(where)]]
    if smooth_iters:                                        # B192: the collar relaxed toward h before the stages
        from meshom.smooth import smooth

        sm = smooth(case.mesh, case.h, case.kind != 0, iters=smooth_iters, kind="size")
        case = dataclasses.replace(case, mesh=sm, h_node=np.asarray(case.h(sm.points)))
    out, recs = run_stages(case, st, parse_opts(opts or []) if isinstance(opts, list) else (opts or {}),
                           verbose=verbose)
    rec["per_stage"] = [{k: v for k, v in r.items() if k != "stats"} for r in recs]
    art = np.flatnonzero(case.kind == 2)
    d, j = cKDTree(m.mesh.points).query(case.mesh.points[art])
    assert d.max() < MATCH_CHORD
    win = replace_part(m.mesh, keep, out, j)
    return win, rec, case0, out


def coast_keys_of(mesh: Mesh, cut_xyz: np.ndarray) -> np.ndarray:
    """Keys of the boundary edges of mesh that are coast (not both ends at a window-cut position)."""
    st = build_topology(mesh)
    be = st.edges[st.edge_elems[:, 1] < 0]
    cut = np.zeros(mesh.n_nodes, bool)
    if len(cut_xyz):
        d, _ = cKDTree(cut_xyz).query(mesh.points)
        cut = d < MATCH_CHORD
    be = be[~(cut[be[:, 0]] & cut[be[:, 1]])]
    return _key(be, mesh.n_nodes), be


def with_h(case, h):
    return dataclasses.replace(case, h=h, h_node=np.asarray(h(case.mesh.points)))


class HMin:
    """B210: h = min(a(xyz), b(xyz)) of two size fields (MeshH of the spliced mesh and of the host): the fixer repairs
    shapes and holds sizes, and pulls cells larger than the host's toward it."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def __call__(self, xyz):
        return np.minimum(self.a(xyz), self.b(xyz))


def objectives(cm: Mesh, kind, host: Mesh, W: Mesh, h_min: float, hostH) -> dict:
    """B220: the insert's objectives on the case `cm` (its cells) and the window part `W` it belongs to: the angle
    excess (the sum over the quads of their largest corner's excess over KEEP_LIMIT_DEG 120 deg), the worst quad
    corner and the quads above 120 deg (measure.cells), the smallest changed cell (measure.floor_check against
    the host window, and how many lie below the floor), the largest cell / host cell (the cell's mean edge over the
    host's node size at its centre, MeshH), the neighbour size ratio p99, and the invalid cells (inverted or
    non-convex in the case, inverted or non-convex in the part)."""
    from meshom.insert import measure
    from meshom.quality.kernel import element_geometry

    c = measure.cells(cm, kind)
    fl = measure.floor_check(host, W, h_min)
    g = element_geometry(cm)
    me = np.r_[g.tri_edges.mean(1) if cm.n_tris else np.zeros(0), g.quad_edges.mean(1) if cm.n_quads else np.zeros(0)]
    v = measure.validity(W)
    qa = np.degrees(g.quad_angles).max(1) if cm.n_quads else np.zeros(0)
    return {"excess_deg": float(np.maximum(qa - KEEP_LIMIT_DEG, 0.0).sum()),
            "qmax": c["qmax"] or 0.0, "q120": c["q120"], "smallest_m": fl["smallest_m"] or 0.0,
            "below_floor": fl["below"],
            "cell_host": float((me / np.asarray(hostH(g.centroid))).max()) if len(me) else 0.0,
            "ratio_p99": c["ratio_p99"] or 1.0, "invalid": int(c["invalid"] + v["inverted"] + v["nonconvex"])}


def keep_after(b: dict, a: dict) -> tuple[bool, str]:
    """B220: whether fix_rings keeps its result `a` over its input `b` (objectives), and why. In this order: more
    invalid cells or more cells below the floor than before: the input; then the angle excess above 120 deg (the sum
    over the quads): lower by more than KEEP_DEG wins; then the worst corner: lower by more than KEEP_DEG wins; then,
    the angles tied, the sizes: ratio p99 (by more than KEEP_RATIO), the largest cell / host (KEEP_CELL_HOST) and the
    smallest cell (by more than KEEP_SMALLEST of it) vote, the result is kept unless the input wins more votes."""
    if a["invalid"] > b["invalid"]:
        return False, f"the result has more invalid cells ({a['invalid']} > {b['invalid']})"
    if a["below_floor"] > b["below_floor"]:
        return False, f"the result has more cells below the floor ({a['below_floor']} > {b['below_floor']})"
    if abs(a["excess_deg"] - b["excess_deg"]) > KEEP_DEG:
        ok = a["excess_deg"] < b["excess_deg"]
        return ok, (f"angle excess above 120 deg {b['excess_deg']:.1f} -> {a['excess_deg']:.1f} (quads > 120 deg "
                    f"{b['q120']} -> {a['q120']}, worst {b['qmax']:.1f} -> {a['qmax']:.1f} deg)")
    if abs(a["qmax"] - b["qmax"]) > KEEP_DEG:
        ok = a["qmax"] < b["qmax"]
        return ok, f"worst quad {b['qmax']:.2f} -> {a['qmax']:.2f} deg (angle excess tied)"
    vote = 0
    for k, tol, sign in (("ratio_p99", KEEP_RATIO, 1), ("cell_host", KEEP_CELL_HOST, 1),
                         ("smallest_m", KEEP_SMALLEST * max(b["smallest_m"], 1.0), -1)):
        d = sign * (a[k] - b[k])
        vote += (d < -tol) - (d > tol)
    return vote >= 0, (f"angles tied (worst {b['qmax']:.2f} / {a['qmax']:.2f} deg); sizes vote {vote:+d} for the "
                       f"result (ratio p99 {b['ratio_p99']:.4f} -> {a['ratio_p99']:.4f}, cell/host "
                       f"{b['cell_host']:.3f} -> {a['cell_host']:.3f}, smallest {b['smallest_m']:.0f} -> "
                       f"{a['smallest_m']:.0f} m)")


def fix_rings(host: Mesh, mesh: Mesh, part: np.ndarray, cut: np.ndarray, rings: int, h_min: float, h_host: float,
              centre: np.ndarray, name: str) -> tuple[Mesh, dict, np.ndarray]:
    """B210: meshom.meshfix.geom (stage a, free-node geometry) on the cells of the spliced mesh `mesh` that the insert
    created or changed (measure.changed_cells against `host`, the host window it was fitted to) and `rings` element
    rings round them (grown through shared nodes), every other node held. part: (T+Q,) bool, the elements of `mesh`
    that replaced the host window (the case is taken from them); cut: (N,) bool, the nodes of `mesh` on the window's
    cut (cells touching it stay out of the case). Elements at a pinch node of the case's outline are left out until
    none is left (as ring_case). The case's boundary edges on the mesh's boundary are coast (kind 1, sliding: the
    insert's banks), the rest fixed (kind 2); the host's coast nodes are fixed too. The size field is HMin of the
    spliced mesh's and the host's node sizes (MeshH). B220: the stage's result is kept only when it is not worse
    than its input by the insert's objectives (objectives, keep_after); else every node stays. Returns (the mesh
    with the new node positions, the record: the case, the stage's info, which mesh was kept and why, the
    objectives of both, the case's cells before / after (measure.cells; after = the mesh returned) and of the
    stage's result, the host nodes moved (count, the largest move in host cells), the nodes moved and the largest
    move, the size floor (measure.floor_check) and the validity of the window part before / after; the ids in
    `mesh` of the host nodes moved)."""
    import time

    from meshom.geo import xyz_to_lonlat
    from meshom.insert import measure
    from meshom.meshfix.cases import Case, _basin_points
    from meshom.meshfix.geom import fix as geom_fix

    t0 = time.perf_counter()
    T0 = mesh.n_tris
    pt, pq = part[:T0], part[T0:]
    wid = np.unique(np.r_[mesh.tris[pt].ravel(), mesh.quads[pq].ravel()])
    loc = np.full(mesh.n_nodes, -1, np.int64)
    loc[wid] = np.arange(len(wid))
    W = Mesh(mesh.points[wid].copy(), loc[mesh.tris[pt]].reshape(-1, 3), loc[mesh.quads[pq]].reshape(-1, 4),
             np.zeros(len(wid), np.int8))
    T = W.n_tris
    d, _ = cKDTree(host.points).query(W.points)
    is_host = d < MATCH_CHORD
    wcut = np.asarray(cut, bool)[wid]

    def touch(mask):
        return np.r_[mask[W.tris].any(1) if T else np.zeros(0, bool), mask[W.quads].any(1)]

    keep = measure.changed_cells(host, W)
    for _ in range(rings):
        near = np.zeros(W.n_nodes, bool)
        near[W.tris[keep[:T]].ravel()] = True
        near[W.quads[keep[T:]].ravel()] = True
        keep |= touch(near)
    keep &= ~touch(wcut)
    for _ in range(1000):                                   # no pinch node on the case's outline (as ring_case)
        st = build_topology(Mesh(W.points, W.tris[keep[:T]], W.quads[keep[T:]], W.node_flag))
        be = st.edges[st.edge_elems[:, 1] < 0]
        pinch = np.bincount(be.ravel(), minlength=W.n_nodes) > 2
        if not pinch.any():
            break
        keep &= ~touch(pinch)
    used = np.unique(np.r_[W.tris[keep[:T]].ravel(), W.quads[keep[T:]].ravel()])
    lw = np.full(W.n_nodes, -1, np.int64)
    lw[used] = np.arange(len(used))
    cm = Mesh(W.points[used].copy(), lw[W.tris[keep[:T]]].reshape(-1, 3), lw[W.quads[keep[T:]]].reshape(-1, 4),
              np.zeros(len(used), np.int8))
    wtopo = build_topology(W)
    wbe = wtopo.edges[wtopo.edge_elems[:, 1] < 0]
    ct = build_topology(cm)
    cbe = ct.edges[ct.edge_elems[:, 1] < 0]
    coast = np.isin(_key(used[cbe], W.n_nodes), _key(wbe, W.n_nodes))    # on the mesh's boundary (the cut is out)
    kind = np.zeros(cm.n_nodes, np.int8)
    kind[cbe[coast].ravel()] = 1
    kind[cbe[~coast].ravel()] = 2
    kind[(kind == 1) & is_host[used]] = 2                   # the host's coast stays where it is
    cm.node_flag = ct.boundary_node.astype(np.int8)
    h = HMin(MeshH(W), MeshH(host))
    seg = np.stack([cm.points[cbe[coast][:, 0]], cm.points[cbe[coast][:, 1]]], axis=1)
    dist_seg = np.stack([W.points[wbe[:, 0]], W.points[wbe[:, 1]]], axis=1)
    case = Case(name, cm, kind, h, np.asarray(h(cm.points)), seg, dist_seg, _basin_points(cm), cbe[~coast],
                0.20, {}, {}, {"kind": "insert_fix_rings"})
    t1 = time.perf_counter()
    out, frec = geom_fix(case, {})
    wall_a = time.perf_counter() - t1
    P = mesh.points.copy()
    P[wid[used]] = out.points
    W2 = Mesh(P[wid], W.tris, W.quads, W.node_flag)
    hostH = MeshH(host)
    obj = {"before": objectives(cm, kind, host, W, h_min, hostH),
           "after": objectives(out, None, host, W2, h_min, hostH)}
    after, why = keep_after(obj["before"], obj["after"])
    fixer = measure.cells(out)
    if not after:                                           # B220: the smoothing made the insert worse: undone
        out, P, W2 = Mesh(cm.points.copy(), cm.tris, cm.quads, cm.node_flag), mesh.points.copy(), W
    moved = np.linalg.norm(out.points - cm.points, axis=1) > 0
    res = Mesh(P, mesh.tris, mesh.quads, mesh.node_flag, mesh.tri_zone)
    hm = is_host[used] & moved
    dist = R_EARTH_M * 2 * np.arcsin(np.minimum(1.0, 0.5 * np.linalg.norm(out.points - cm.points, axis=1)))
    reach = R_EARTH_M * np.arccos(np.clip(cm.points @ (centre / np.linalg.norm(centre)), -1.0, 1.0)).max()
    far = int(np.argmax(dist))
    rec = {"rings": int(rings), "h": "min(MeshH(spliced), MeshH(host))",
           "case": {"nodes": int(cm.n_nodes), "free": int((kind == 0).sum()), "sliding": int((kind == 1).sum()),
                    "fixed": int((kind == 2).sum()), "elems": int(cm.n_elems), "quads": int(cm.n_quads),
                    "tris": int(cm.n_tris), "reach_km": float(reach / 1e3)},
           "stage_a": {"wall_s": wall_a, "info": {k: v for k, v in frec["info"].items() if k != "opts"}},
           "kept": "after" if after else "before", "keep_rule": why, "objectives": obj, "fixer_result": fixer,
           "before": measure.cells(cm, kind), "after": measure.cells(out),
           "host_moved": int(hm.sum()),
           "host_moved_max_cells": float(dist[hm].max() / h_host) if hm.any() else 0.0,
           "nodes_moved": int(moved.sum()), "moved_max_km": float(dist.max() / 1e3),
           "moved_max_at": [round(float(v[0]), 4) for v in xyz_to_lonlat(out.points[far][None])],
           "floor": measure.floor_check(host, W2, h_min),
           "validity": {"before": measure.validity(W), "after": measure.validity(W2)},
           "wall_s": time.perf_counter() - t0}
    return res, rec, wid[used][hm]
