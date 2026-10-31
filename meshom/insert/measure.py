"""Scores of an insert by zone, on the sphere (meshom.quality.kernel's tangent planes), the same definitions before
(the host) and after (decision B163).

Zones by position, so the host and the result are cut alike: **core** = elements whose centroid lies inside the
insert's outline (join A, bank N, join B, bank S) and outside the ring; **ring** = elements whose centroid lies
within RING_H join cells of a join chain, both sides; **outside** = the other elements within OUT_H host cells of
the outline. Per zone: quads, triangles, quad max angle (> 110, > 120 deg, p99, max), neighbour size ratio (larger /
smaller mean edge length across an interior edge with both elements in the zone; p99, max), aspect (longer / shorter
sum of opposite edges; p99, max), irregular interior nodes (pure-quad nodes with a quad count other than 4; nodes
touching a triangle apart), inverted or non-convex cells. Across the join: the ratio over the interior edges with
one element inside the outline and one outside, both in the ring. Coast: the distance of every boundary node in the
core or the ring to the host's coast (km and in units of the run's h; the rule: <= 0.25 h, B161). Core size: |ln s /
h| of its edges against the insert's h (B162).
"""
import numpy as np

from meshom.geo import R_EARTH_M, arc_length
from meshom.insert.geom import point_in_polygon, seg_distance
from meshom.mesh import Mesh, build_topology
from meshom.quality.kernel import element_geometry

RING_H = 2.0                 # B163: the ring zone: centroids within this many join cells of a join chain
ON_OUTLINE_M = 100.0         # B168: a coast node within this (m) of the insert's outline counts as the insert's
OUT_H = 6.0                  # B163: the outside zone: centroids within this many host cells of the outline


def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


def _mx(x):
    return float(np.max(x)) if len(x) else None


def zones(mesh: Mesh, chart, outline_xy: np.ndarray, chains_xy: list, h_join: list, h_host: float) -> dict:
    """{zone: (T+Q,) bool} of the module docstring."""
    g = element_geometry(mesh)
    c = chart.fwd(g.centroid)
    inside = point_in_polygon(c, outline_xy)
    ring = np.zeros(len(c), bool)
    for xy, hj in zip(chains_xy, h_join):
        seg = np.stack([xy[:-1], xy[1:]], axis=1)
        ring |= seg_distance(c, seg) <= RING_H * hj
    oseg = np.stack([outline_xy, np.roll(outline_xy, -1, axis=0)], axis=1)
    near = seg_distance(c, oseg) <= OUT_H * h_host
    return {"core": inside & ~ring, "ring": ring, "outside": near & ~inside & ~ring, "inside": inside}


def metrics(mesh: Mesh, z: dict, chart=None, coast_seg_xy=None, h_run=None, h_ins=None) -> dict:
    g = element_geometry(mesh)
    topo = build_topology(mesh)
    T, Q = mesh.n_tris, mesh.n_quads
    qa = np.degrees(g.quad_angles)
    qmax = qa.max(1) if Q else np.zeros(0)
    area = np.r_[g.tri_area, g.quad_area]
    mean_edge = np.r_[g.tri_edges.mean(1) if T else np.zeros(0), g.quad_edges.mean(1) if Q else np.zeros(0)]
    ee = topo.edge_elems
    inn = ee[:, 1] >= 0
    e0, e1 = ee[inn, 0], ee[inn, 1]
    r = np.maximum(mean_edge[e0], mean_edge[e1]) / np.maximum(np.minimum(mean_edge[e0], mean_edge[e1]), 1e-9)
    qe = g.quad_edges
    asp = (np.maximum(qe[:, 0] + qe[:, 2], qe[:, 1] + qe[:, 3]) /
           np.maximum(np.minimum(qe[:, 0] + qe[:, 2], qe[:, 1] + qe[:, 3]), 1e-9)) if Q else np.zeros(0)
    nq = np.bincount(mesh.quads.ravel(), minlength=mesh.n_nodes) if Q else np.zeros(mesh.n_nodes, int)
    nt = np.bincount(mesh.tris.ravel(), minlength=mesh.n_nodes) if T else np.zeros(mesh.n_nodes, int)
    interior = ~topo.boundary_node
    out = {}
    for name in ("core", "ring", "outside"):
        em = z[name]
        qm = em[T:]
        both = em[e0] & em[e1]
        nodes = np.zeros(mesh.n_nodes, bool)
        nodes[mesh.quads[qm].ravel()] = True
        if T:
            nodes[mesh.tris[em[:T]].ravel()] = True
        pure = nodes & interior & (nt == 0)
        irr = pure & (nq != 4)
        d = {"quads": int(qm.sum()), "tris": int(em[:T].sum()),
             "q110": int((qmax[qm] > 110).sum()), "q120": int((qmax[qm] > 120).sum()),
             "qmax_p99": _pct(qmax[qm], 99), "qmax": _mx(qmax[qm]),
             "ratio_p99": _pct(r[both], 99), "ratio_max": _mx(r[both]),
             "aspect_p99": _pct(asp[qm], 99), "aspect_max": _mx(asp[qm]),
             "irregular": {int(k): int(v) for k, v in zip(*np.unique(nq[irr], return_counts=True))},
             "tri_nodes": int((nodes & interior & (nt > 0)).sum()),
             "invalid": int((area[em] <= 0).sum() + (qa[qm] >= 180).any(1).sum())}
        if qm.any():                                       # where the worst quad of the zone is (lon, lat)
            from meshom.geo import xyz_to_lonlat

            k = np.flatnonzero(qm)[int(np.argmax(qmax[qm]))]
            d["qmax_at"] = [round(float(v), 4) for v in np.c_[xyz_to_lonlat(g.centroid[T + k][None])][0]]
        if T and em[:T].any():
            ta = np.degrees(g.tri_angles[em[:T]])
            d["tri_angle_min"], d["tri_angle_max"] = float(ta.min()), float(ta.max())
        out[name] = d
    across = z["ring"][e0] & z["ring"][e1] & (z["inside"][e0] != z["inside"][e1])
    out["across_join"] = {"edges": int(across.sum()), "ratio_p99": _pct(r[across], 99),
                          "ratio_max": _mx(r[across])}
    if chart is not None and coast_seg_xy is not None:
        zone_nodes = np.zeros(mesh.n_nodes, bool)
        em = z["core"] | z["ring"]
        zone_nodes[mesh.quads[em[T:]].ravel()] = True
        if T:
            zone_nodes[mesh.tris[em[:T]].ravel()] = True
        cn = np.flatnonzero(zone_nodes & topo.boundary_node)
        if len(cn):
            dist = seg_distance(chart.fwd(mesh.points[cn]), coast_seg_xy) / chart.scale(chart.fwd(mesh.points[cn]))
            hh = np.asarray(h_run(mesh.points[cn])) if h_run is not None else np.full(len(cn), np.nan)
            out["coast"] = {"nodes": int(len(cn)), "shift_km_max": float(dist.max() / 1e3),
                            "shift_km_p99": _pct(dist / 1e3, 99), "shift_h_max": float(np.max(dist / hh)),
                            "shift_h_p99": _pct(dist / hh, 99)}
    if h_ins is not None:
        em = z["core"]
        cells = [c for c in (mesh.tris[em[:T]], mesh.quads[em[T:]]) if len(c)]
        if cells:
            e = np.unique(np.sort(np.vstack([np.c_[c.ravel(), np.roll(c, -1, 1).ravel()] for c in cells]), 1),
                          axis=0)
            s = R_EARTH_M * arc_length(mesh.points[e[:, 0]], mesh.points[e[:, 1]])
            m = mesh.points[e[:, 0]] + mesh.points[e[:, 1]]
            m /= np.linalg.norm(m, axis=1, keepdims=True)
            lr = np.abs(np.log(s / np.asarray(h_ins(m))))
            out["core"]["size_lnsh_p99"] = _pct(lr, 99)
            out["core"]["size_lnsh_max"] = _mx(lr)
    return out


def validity(mesh: Mesh) -> dict:
    """Whole-mesh checks: inverted / non-convex cells, edges with more than two elements, water components, and
    shared edges that both their cells traverse in the same direction (B186: with every cell counter-clockwise on
    the sphere, two cells on the same side of their shared edge overlap; a cell reversed to hide a fold shows here)."""
    from meshom.meshfix.cases import components

    g = element_geometry(mesh)
    area = np.r_[g.tri_area, g.quad_area]
    nonconvex = int((np.degrees(g.quad_angles) >= 180).any(1).sum()) if mesh.n_quads else 0
    de = np.concatenate([np.stack([c, np.roll(c, -1, 1)], -1).reshape(-1, 2) for c in (mesh.tris, mesh.quads)
                         if len(c)])
    ed = np.sort(de, 1)
    key = ed[:, 0] * mesh.n_nodes + ed[:, 1]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    bal = np.bincount(inv, weights=np.where(de[:, 0] < de[:, 1], 1.0, -1.0), minlength=len(cnt))
    lab = components(mesh)
    return {"inverted": int((area <= 0).sum()), "nonconvex": nonconvex, "nonmanifold_edges": int((cnt > 2).sum()),
            "same_direction_edges": int(((cnt == 2) & (bal != 0)).sum()),
            "water_components": int(lab.max() + 1) if len(lab) else 0}


def changed_cells(host: Mesh, mesh: Mesh) -> np.ndarray:
    """(T+Q,) bool: the cells of mesh the insert created or changed (B165): a node not at the exact position of a host
    node, or a node set that is no host cell."""
    from scipy.spatial import cKDTree

    d, j = cKDTree(host.points).query(mesh.points)
    hid = np.where(d < 1e-12, j, -1)
    keys = {tuple(r) for c in (host.tris, host.quads) for r in np.sort(c, axis=1).tolist()}
    out = []
    for c in (mesh.tris, mesh.quads):
        if len(c):
            h = hid[c]
            out.append(np.array([(min(r) < 0) or (tuple(r) not in keys) for r in np.sort(h, axis=1).tolist()]))
    return np.concatenate(out) if out else np.zeros(0, bool)


def floor_check(host: Mesh, mesh: Mesh, h_min: float) -> dict:
    """The size floor on the result (B165): every cell the insert created or changed has a
    mean great-circle edge length >= h_min (host_min_cell's measure), no tolerance. Returns the smallest such cell,
    the count below the floor and where the smallest one lies."""
    from meshom.geo import xyz_to_lonlat

    g = element_geometry(mesh)
    me = np.r_[g.tri_edges.mean(1) if mesh.n_tris else np.zeros(0), g.quad_edges.mean(1) if mesh.n_quads else
               np.zeros(0)]
    ch = changed_cells(host, mesh)
    if not ch.any():
        return {"h_min_m": float(h_min), "cells": 0, "smallest_m": None, "below": 0}
    k = np.flatnonzero(ch)[int(np.argmin(me[ch]))]
    return {"h_min_m": float(h_min), "cells": int(ch.sum()), "smallest_m": float(me[k]),
            "smallest_h_min": float(me[k] / h_min) if h_min > 0 else None,
            "below": int((me[ch] < h_min).sum()) if h_min > 0 else 0,
            "at": [round(float(v), 4) for v in np.c_[xyz_to_lonlat(g.centroid[k][None])][0]]}


def boundary_edges(mesh: Mesh) -> np.ndarray:
    topo = build_topology(mesh)
    return topo.edges[topo.edge_elems[:, 1] < 0]


def insert_coast_nodes(mesh: Mesh, merged: Mesh, merged_host: np.ndarray, chart) -> np.ndarray:
    """(N,) bool: the boundary nodes of mesh that belong to the insert's coast (B168): each takes the
    side of its nearest boundary edge of the merged window, an edge being the host's when both its nodes are host
    nodes (merged_host) and the insert's otherwise (block and bay banks, which meet the host at the join ends). By
    position against the outline, bay and bank nodes the cleaning slid 300 m off it counted as the host's (Dover's
    1.5 h host_shift)."""
    be = boundary_edges(merged)
    lab = ~(merged_host[be[:, 0]] & merged_host[be[:, 1]])
    mxy = chart.fwd(merged.points)
    seg = np.stack([mxy[be[:, 0]], mxy[be[:, 1]]], axis=1)
    out = np.zeros(mesh.n_nodes, bool)
    bn = np.unique(boundary_edges(mesh).ravel())
    if not lab.any() or not len(bn):
        return out
    q = chart.fwd(mesh.points[bn])
    out[bn] = seg_distance(q, seg[lab]) < (seg_distance(q, seg[~lab]) if (~lab).any() else np.inf)
    return out


def host_coast(mesh: Mesh, merged: Mesh, merged_host: np.ndarray, insert_nodes: np.ndarray, chart, host_seg_xy,
               h_run, cut_xyz=None) -> dict:
    """The host's coast outside the insert (B168, B162: the join and the outside keep the host's coast):
    forward = the mesh's coast nodes on the host's side against the host's coast; back = the host's coast nodes left
    in the merged window (outside the replaced region) against the mesh's coast edges; the window's cut nodes
    (cut_xyz) are no coast. In km and in the run's h."""
    from scipy.spatial import cKDTree

    def coast(m, nodes):
        if cut_xyz is None or not len(cut_xyz) or not len(nodes):
            return nodes
        return nodes[cKDTree(cut_xyz).query(m.points[nodes])[0] >= 1e-12]
    out = {}
    xy = chart.fwd(mesh.points)
    bn = coast(mesh, np.unique(boundary_edges(mesh).ravel()))
    f = bn[~insert_nodes[bn]]
    if len(f) and len(host_seg_xy):
        d = seg_distance(xy[f], host_seg_xy) / chart.scale(xy[f])
        r = d / np.asarray(h_run(mesh.points[f]))
        out["forward"] = {"nodes": int(len(f)), "km_max": float(d.max() / 1e3), "h_run_max": float(r.max())}
    mbn = coast(merged, np.unique(boundary_edges(merged).ravel()))
    hb = mbn[merged_host[mbn]]
    be = boundary_edges(mesh)
    if len(hb) and len(be):
        mxy = chart.fwd(merged.points[hb])
        d = seg_distance(mxy, np.stack([xy[be[:, 0]], xy[be[:, 1]]], axis=1)) / chart.scale(mxy)
        r = d / np.asarray(h_run(merged.points[hb]))
        out["back"] = {"nodes": int(len(hb)), "km_max": float(d.max() / 1e3), "h_run_max": float(r.max())}
    out["h_run_max"] = max([v["h_run_max"] for v in out.values()], default=0.0)
    return out


def coast_real(mesh: Mesh, z: dict, chart, real_seg_xy: np.ndarray, host_seg_xy: np.ndarray, h_ins, h_run,
               inside_xy=None, host_xy=None, insert_nodes=None) -> dict:
    """The coast against the real coast (B164): the insert's coast nodes (boundary nodes of the core and ring cells
    whose position lies inside the insert's outline, or insert_nodes when given: insert_coast_nodes) measured against
    the real coast (shift), the ring's other coast nodes against the host's coast (host_shift: the join and the
    outside keep the host's coast), and the real coast's own points (densified) whose nearest coast node of the mesh
    is an insert coast node, measured against the mesh's coast edges (missed: bays and fjords the insert leaves out).
    Distances on the chart over its scale (sphere metres), in km and in the local insert cell (h_ins) and the run's
    h."""
    topo = build_topology(mesh)
    T = mesh.n_tris
    em = z["core"] | z["ring"]
    zn = np.zeros(mesh.n_nodes, bool)
    zn[mesh.quads[em[T:]].ravel()] = True
    if T:
        zn[mesh.tris[em[:T]].ravel()] = True
    bn = topo.boundary_node
    xy = chart.fwd(mesh.points)
    sc = chart.scale(xy)
    ins = np.ones(mesh.n_nodes, bool)
    if insert_nodes is not None:
        ins = np.asarray(insert_nodes, bool).copy()
    elif inside_xy is not None:                            # on the outline counts as inside (the banks lie on it)
        oseg = np.stack([inside_xy, np.roll(inside_xy, -1, axis=0)], axis=1)
        ins = point_in_polygon(xy, inside_xy) | (seg_distance(xy, oseg) <= ON_OUTLINE_M)
    if host_xy is not None and len(host_xy):           # the join chains' nodes are the host's (its coast)
        from scipy.spatial import cKDTree

        d, _ = cKDTree(host_xy).query(xy)
        ins &= d > 1.0
    out = {}

    def stats(name, nodes, dist):
        if not len(nodes):
            return
        hi = np.asarray(h_ins(mesh.points[nodes])) if h_ins is not None else np.full(len(nodes), np.nan)
        hr = np.asarray(h_run(mesh.points[nodes])) if h_run is not None else np.full(len(nodes), np.nan)
        out[name] = {"nodes": int(len(nodes)), "km_max": float(dist.max() / 1e3), "km_p99": _pct(dist / 1e3, 99),
                     "km_mean": float(dist.mean() / 1e3),
                     "h_ins_max": float(np.max(dist / hi)), "h_ins_p99": _pct(dist / hi, 99),
                     "h_run_max": float(np.max(dist / hr)), "h_run_p99": _pct(dist / hr, 99)}
    a = np.flatnonzero(zn & bn & ins)
    if len(a) and len(real_seg_xy):
        stats("shift", a, seg_distance(xy[a], real_seg_xy) / sc[a])
    b = np.flatnonzero(zn & bn & ~ins)
    if len(b) and len(host_seg_xy):
        stats("host_shift", b, seg_distance(xy[b], host_seg_xy) / sc[b])
    if len(a) and len(real_seg_xy):
        from scipy.spatial import cKDTree

        be = topo.edges[topo.edge_elems[:, 1] < 0]
        mseg = np.stack([xy[be[:, 0]], xy[be[:, 1]]], axis=1)
        p = real_seg_xy[:, 0]
        coast_nodes = np.flatnonzero(bn)
        d, j = cKDTree(xy[coast_nodes]).query(p)
        mine = np.isin(coast_nodes[j], a)
        if mine.any():
            q = p[mine]
            dist = seg_distance(q, mseg) / chart.scale(q)
            qx = chart.inv(q)
            hi = np.asarray(h_ins(qx)) if h_ins is not None else np.full(len(q), np.nan)
            hr = np.asarray(h_run(qx)) if h_run is not None else np.full(len(q), np.nan)
            out["missed"] = {"points": int(len(q)), "km_max": float(dist.max() / 1e3), "km_p99": _pct(dist / 1e3, 99),
                             "h_ins_max": float(np.max(dist / hi)), "h_ins_p99": _pct(dist / hi, 99),
                             "h_run_max": float(np.max(dist / hr))}
    return out


def cells(mesh: Mesh, case_kind: np.ndarray | None = None) -> dict:
    """The collar's measures (B192) over every cell of mesh (the cleaned case: the collar's host rings, the join and
    the block's ring rows): quad max corner (max, p99.9, p99, count > 110, count and share > 120 deg), the neighbour
    size ratio across its interior edges (p99, max), triangles, and the valence (cells per node) of the interior nodes
    that are not on the case's boundary (case_kind: 0 interior)."""
    g = element_geometry(mesh)
    topo = build_topology(mesh)
    T, Q = mesh.n_tris, mesh.n_quads
    qmax = np.degrees(g.quad_angles).max(1) if Q else np.zeros(0)
    me = np.r_[g.tri_edges.mean(1) if T else np.zeros(0), g.quad_edges.mean(1) if Q else np.zeros(0)]
    ee = topo.edge_elems
    inn = ee[:, 1] >= 0
    a, b = me[ee[inn, 0]], me[ee[inn, 1]]
    r = np.maximum(a, b) / np.maximum(np.minimum(a, b), 1e-9)
    val = np.bincount(np.r_[mesh.tris.ravel(), mesh.quads.ravel()], minlength=mesh.n_nodes)
    inner = ~topo.boundary_node
    if case_kind is not None:
        inner &= np.asarray(case_kind) == 0
    vv, vc = np.unique(val[inner], return_counts=True)
    return {"quads": int(Q), "tris": int(T), "qmax": _mx(qmax), "qmax_p999": _pct(qmax, 99.9),
            "qmax_p99": _pct(qmax, 99), "q110": int((qmax > 110).sum()), "q120": int((qmax > 120).sum()),
            "q120_share": float((qmax > 120).mean()) if Q else 0.0, "q125": int((qmax > 125).sum()),
            "ratio_p99": _pct(r, 99), "ratio_max": _mx(r),
            "valence": {int(k): int(v) for k, v in zip(vv, vc)},
            "invalid": int((np.r_[g.tri_area, g.quad_area] <= 0).sum() + (qmax >= 180).sum())}


def host_moved(host: Mesh, mesh: Mesh, h_host: float, kept: np.ndarray, collar: np.ndarray | None = None) -> dict:
    """Host nodes the insert moved (B192): of the host window's nodes `kept` ((N,) bool: those a host cell outside the
    replaced region uses, the joins included), the ones that are no node of mesh at their exact position -- count
    and the largest distance to mesh's nearest node in host cells, and their ids ("ids"); with collar ((N,) bool: the
    host nodes the collar may move), the moved nodes beyond it (must be 0)."""
    from scipy.spatial import cKDTree

    d, _ = cKDTree(mesh.points).query(host.points)
    moved = (d >= 1e-12) & np.asarray(kept, bool)
    dist = R_EARTH_M * 2 * np.arcsin(np.minimum(1.0, 0.5 * d))
    out = {"moved": int(moved.sum()), "largest_host_cells": float(dist[moved].max() / h_host) if moved.any() else 0.0,
           "ids": np.flatnonzero(moved)}
    if collar is not None:
        out["beyond_collar"] = int((moved & ~np.asarray(collar, bool)).sum())
    return out
