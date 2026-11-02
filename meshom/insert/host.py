"""The host side of an insert: a window cut out of a finished mesh (with which of its boundary edges are the real
coast and which the window's cut), the host edge graph, and the splice of an edited window back into the mesh.

Node ids: the splice keeps every host node the window did not change (matched by exact position) with its id and
order; moved and new nodes are appended; nodes no element uses any more are dropped (B163).
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from meshom.geo import R_EARTH_M, arc_length
from meshom.mesh import Mesh, build_topology

MATCH_CHORD = 1e-12          # exact position match (unit-vector chord) of an unchanged node


@dataclass
class Window:
    mesh: Mesh                   # the window's elements, local node ids
    old_id: np.ndarray           # (n,) host node id per local node
    elems: np.ndarray            # (k,) host element ids in the window (tris first, then T + quad index)
    kind: np.ndarray             # (n,) int8: 0 interior, 1 real coast, 2 window cut
    coast_edges: np.ndarray      # (C,2) local ids of the real coast edges
    cut_edges: np.ndarray        # (D,2) local ids of the window's cut edges


def _sub(points, tris, quads, keep_t, keep_q):
    used = np.zeros(len(points), bool)
    used[tris[keep_t].ravel()] = True
    used[quads[keep_q].ravel()] = True
    old = np.flatnonzero(used)
    loc = np.full(len(points), -1, np.int64)
    loc[old] = np.arange(len(old))
    return Mesh(points[old], loc[tris[keep_t]].reshape(-1, 3), loc[quads[keep_q]].reshape(-1, 4),
                np.zeros(len(old), np.int8)), old


def _bkeys(mesh: Mesh, ids: np.ndarray | None = None):
    topo = build_topology(mesh)
    be = topo.edges[topo.edge_elems[:, 1] < 0]
    g = be if ids is None else ids[be]
    return be, np.minimum(g[:, 0], g[:, 1]) * (1 << 32) + np.maximum(g[:, 0], g[:, 1]), topo


def cut_window(mesh: Mesh, centre: np.ndarray, radius_m: float) -> Window:
    """The host elements whose nodes all lie within radius_m (great circle) of the unit vector centre. A window
    boundary edge is real coast when it is a boundary edge of the host too (tested on the halo: every host element
    touching a window node)."""
    c = centre / np.linalg.norm(centre)
    inside = mesh.points @ c >= np.cos(min(radius_m / R_EARTH_M, np.pi))
    kt = inside[mesh.tris].all(1) if mesh.n_tris else np.zeros(0, bool)
    kq = inside[mesh.quads].all(1)
    win, old = _sub(mesh.points, mesh.tris, mesh.quads, kt, kq)
    near = np.zeros(mesh.n_nodes, bool)
    near[old] = True
    ht = near[mesh.tris].any(1) if mesh.n_tris else np.zeros(0, bool)
    hq = near[mesh.quads].any(1)
    halo, hold = _sub(mesh.points, mesh.tris, mesh.quads, ht, hq)
    be, wkey, _ = _bkeys(win, old)
    _, hkey, _ = _bkeys(halo, hold)
    real = np.isin(wkey, hkey)
    kind = np.zeros(win.n_nodes, np.int8)
    kind[be[real].ravel()] = 1
    kind[be[~real].ravel()] = 2
    win.node_flag = np.zeros(win.n_nodes, np.int8)
    win.node_flag[be.ravel()] = 1
    elems = np.r_[np.flatnonzero(kt), mesh.n_tris + np.flatnonzero(kq)]
    return Window(win, old, elems, kind, be[real], be[~real])


def splice(mesh: Mesh, win: Window, new: Mesh, tri_zone_new: np.ndarray | None = None) -> tuple[Mesh, dict]:
    """The host mesh with the window's elements replaced by `new` (a mesh of the window's region whose cut nodes are
    unmoved): nodes of `new` at the exact position of a window node take its host id, the others are appended; host
    nodes left without an element are dropped, so later host ids shift down (info["host_to_new"] maps them).
    Returns (mesh, info)."""
    d, j = cKDTree(win.mesh.points).query(new.points)
    gid = np.where(d < MATCH_CHORD, win.old_id[j], -1)
    cut = np.flatnonzero(win.kind == 2)
    assert np.isin(win.old_id[cut], gid).all(), "a window cut node moved"
    T = mesh.n_tris
    rm = np.zeros(mesh.n_elems, bool)
    rm[win.elems] = True
    n0 = mesh.n_nodes
    fresh = gid < 0
    gid[fresh] = n0 + np.arange(fresh.sum())
    pts = np.vstack([mesh.points, new.points[fresh]])
    tris = np.vstack([mesh.tris[~rm[:T]], gid[new.tris]]).astype(np.int64)
    quads = np.vstack([mesh.quads[~rm[T:]], gid[new.quads]]).astype(np.int64)
    tz = None
    if mesh.tri_zone is not None or tri_zone_new is not None:
        a = np.zeros(int((~rm[:T]).sum()), bool) if mesh.tri_zone is None else np.asarray(mesh.tri_zone)[~rm[:T]]
        b = np.zeros(new.n_tris, bool) if tri_zone_new is None else np.asarray(tri_zone_new, bool)
        tz = np.r_[a, b]
    used = np.zeros(len(pts), bool)
    used[tris.ravel()] = True
    used[quads.ravel()] = True
    keep = np.flatnonzero(used)
    loc = np.full(len(pts), -1, np.int64)
    loc[keep] = np.arange(len(keep))
    out = Mesh(pts[keep], loc[tris].reshape(-1, 3), loc[quads].reshape(-1, 4), np.zeros(len(keep), np.int8), tz)
    topo = build_topology(out)
    out.node_flag = topo.boundary_node.astype(np.int8)
    info = {"host_nodes": int(n0), "nodes": int(out.n_nodes), "appended_nodes": int(fresh.sum()),
            "dropped_host_nodes": int(n0 - (keep < n0).sum()), "removed_elems": int(rm.sum()),
            "added_elems": int(new.n_elems),
            # host node id -> its id in the result (-1: dropped); the order is kept, the numbers shift past every
            # dropped node, so per-node data of the host must be carried over with this map
            "host_to_new": loc[:n0]}
    return out, info


def drop_deadends(mesh: Mesh) -> tuple[Mesh, np.ndarray, dict]:
    """FESOM3's dead-end rule (B260, as meshom's ORCA rule 12): passes until no element has fewer than two edge
    neighbours, each pass removing every such element; the nodes of a removed element that stay get flag 1, nodes no
    element uses any more go; nodes and elements keep their order. Returns (mesh, old_to_new (-1: dropped), record
    {"passes", "removed", "nodes_removed", "at": the removed elements' centres [lon, lat]})."""
    from meshom.geo import xyz_to_lonlat

    o2n = np.arange(mesh.n_nodes, dtype=np.int64)
    passes, at = [], []
    while True:
        dead = build_topology(mesh).elem_neighbours < 2
        if not dead.any():
            break
        T = mesh.n_tris
        passes.append(int(dead.sum()))
        gone = [mesh.tris[i] if i < T else mesh.quads[i - T] for i in np.flatnonzero(dead)]
        lo, la = xyz_to_lonlat(np.array([mesh.points[e].mean(0) for e in gone]))
        at += np.round(np.c_[lo, la], 5).tolist()
        flag = mesh.node_flag.copy()
        flag[np.concatenate(gone)] = 1
        tris, quads = mesh.tris[~dead[:T]], mesh.quads[~dead[T:]]
        used = np.zeros(mesh.n_nodes, bool)
        used[tris.ravel()] = True
        used[quads.ravel()] = True
        new = np.full(mesh.n_nodes, -1, np.int64)
        new[used] = np.arange(int(used.sum()))
        tz = None if mesh.tri_zone is None else np.asarray(mesh.tri_zone)[~dead[:T]]
        mesh = Mesh(mesh.points[used], new[tris].reshape(-1, 3), new[quads].reshape(-1, 4), flag[used], tz)
        o2n = np.where(o2n >= 0, new[np.maximum(o2n, 0)], -1)
    return mesh, o2n, {"passes": passes, "removed": int(sum(passes)), "nodes_removed": int((o2n < 0).sum()),
                       "at": at}


@dataclass
class Graph:
    edges: np.ndarray            # (E,2)
    length_m: np.ndarray         # (E,) great-circle length
    boundary: np.ndarray         # (n,) bool
    elems_per_node: np.ndarray   # (n,)
    tris_per_node: np.ndarray    # (n,)
    node_quads: list             # per node: list of quad ids


def graph(mesh: Mesh) -> Graph:
    topo = build_topology(mesh)
    L = R_EARTH_M * arc_length(mesh.points[topo.edges[:, 0]], mesh.points[topo.edges[:, 1]])
    nq = [[] for _ in range(mesh.n_nodes)]
    for k, q in enumerate(mesh.quads):
        for v in q:
            nq[v].append(k)
    tpn = np.bincount(mesh.tris.ravel(), minlength=mesh.n_nodes) if mesh.n_tris else np.zeros(mesh.n_nodes, int)
    return Graph(topo.edges, L, topo.boundary_node, topo.elems_per_node, tpn, nq)


def coast_neighbours(coast_edges: np.ndarray, n: int) -> list:
    nb = [[] for _ in range(n)]
    for a, b in coast_edges:
        nb[a].append(int(b))
        nb[b].append(int(a))
    return nb


def coast_walk(nb: list, start: int, first: int, length_m: float, xy: np.ndarray) -> int:
    """Walk along the coast from `start` through its neighbour `first` until `length_m` (chart) is covered; returns
    the node reached (stops at a pinch or a dead end)."""
    prev, cur, acc = start, first, float(np.linalg.norm(xy[first] - xy[start]))
    while acc < length_m:
        nxt = [v for v in nb[cur] if v != prev]
        if len(nb[cur]) != 2 or not nxt:
            break
        prev, cur = cur, nxt[0]
        acc += float(np.linalg.norm(xy[cur] - xy[prev]))
    return cur


def coast_path(nb: list, start: int, first: int, xy: np.ndarray, stop, max_len: float) -> np.ndarray:
    """The coast nodes from `start` through `first` until stop(node) holds, max_len (chart) is covered, or a pinch
    or a dead end is reached."""
    path, acc = [start, first], float(np.linalg.norm(xy[first] - xy[start]))
    while acc < max_len and not stop(path[-1]):
        nxt = [v for v in nb[path[-1]] if v != path[-2]]
        if len(nb[path[-1]]) != 2 or not nxt or nxt[0] == start:
            break
        path.append(nxt[0])
        acc += float(np.linalg.norm(xy[path[-1]] - xy[path[-2]]))
    return np.array(path)
