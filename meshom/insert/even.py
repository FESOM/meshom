"""Even neighbour areas in a group insert's patches and the host rings round them (decision B252).

The patch fill (m02: packing, Delaunay, recombination) leaves cells whose area differs from their edge neighbours' by
2-3.5 times where the size grows fast and where triangles sit among quads. This pass moves the free nodes, one at a
time (Gauss-Seidel sweeps), to lower

    F = sum over the edge-neighbour pairs (a, b) round the node of ln(A_a / A_b)^2
        + TAIL_W x sum over the same pairs of max(0, |ln(A_a / A_b)| - ln TAIL_RATIO)^2
        + ANGLE_W x sum over its cells of max(0, worst corner - ANGLE_SOFT_DEG)^2 (radians)
        + LIMIT_W x sum over its cells of max(0, worst corner - (limit_deg - LIMIT_MARGIN_DEG))^2

where A is the cell's area in the tangent plane at its centroid, a triangle's counted twice (a triangle is half a
quad: a quad split into two has ratio 1). Each node tries EVEN_DIRS directions at EVEN_STEPS fractions of its cells'
mean edge in its tangent plane and takes the best trial that lowers F; a trial is refused when any of its cells
inverts or turns non-convex, when its worst corner (quad or triangle) exceeds max(limit_deg, the least it has had in
the pass), or when a cell's mean edge falls below h_min. The cells' node ids
never change (the depth's bookkeeping holds); fixed nodes never move.
"""
import numpy as np

from meshom.geo import R_EARTH_M
from meshom.mesh import Mesh, build_topology

EVEN_SWEEPS = 40             # B252: sweeps of the pass (it stops earlier when a sweep lowers F by < EVEN_TOL)
EVEN_STEPS = (0.3, 0.15, 0.07, 0.03)   # B252: the trial steps, fractions of the node's cells' mean edge
EVEN_DIRS = 8                # B252: the trial directions per step
EVEN_TOL = 1e-4              # B252: a sweep that lowers the total F by less than this fraction ends the pass
ANGLE_SOFT_DEG = 110.0       # B252: corners above this are priced (the pass keeps the quads away from the 120 limit)
ANGLE_W = 4.0                # B252: the price of a corner's excess over ANGLE_SOFT_DEG (radians squared)
LIMIT_W = 2000.0             # B252: the price of a corner's excess over limit_deg (radians squared): the quads still
                             # above it after fix_rings come down first
LIMIT_MARGIN_DEG = 0.5       # B252: the limit's price starts this far below limit_deg
TAIL_RATIO = 1.4             # B252: pair ratios above this are priced again, TAIL_W times (the tail first)
TAIL_W = 10.0


def _cells(mesh: Mesh):
    """(E, 4) node ids (a triangle's fourth = -1), (E,) corner counts."""
    T = mesh.n_tris
    el = np.full((mesh.n_elems, 4), -1, np.int64)
    if T:
        el[:T, :3] = mesh.tris
    el[T:] = mesh.quads
    nv = np.r_[np.full(T, 3), np.full(mesh.n_quads, 4)]
    return el, nv


def _geom(V: np.ndarray, m: int):
    """V (..., m, 3) unit vectors of cells with m corners: (signed area in the tangent plane at the centroid (unit
    sphere), worst corner (rad; a reflex corner counts 2 pi - angle), all corners convex and turning as the area)."""
    c = V.mean(-2)
    c = c / np.linalg.norm(c, axis=-1, keepdims=True)
    S = np.zeros(V.shape[:-2])
    for i in range(1, m - 1):
        S = S + np.einsum("...j,...j->...", np.cross(V[..., i, :] - V[..., 0, :], V[..., i + 1, :] - V[..., 0, :]), c)
    S = 0.5 * S
    worst = np.zeros(V.shape[:-2])
    convex = np.ones(V.shape[:-2], bool)
    for i in range(m):
        p, a, b = V[..., i, :], V[..., i - 1, :], V[..., (i + 1) % m, :]
        ta = a - np.einsum("...j,...j->...", a, p)[..., None] * p
        tb = b - np.einsum("...j,...j->...", b, p)[..., None] * p
        cs = np.einsum("...j,...j->...", ta, tb) / np.maximum(np.linalg.norm(ta, axis=-1) * np.linalg.norm(tb, axis=-1),
                                                              1e-300)
        th = np.arccos(np.clip(cs, -1.0, 1.0))
        turn = np.einsum("...j,...j->...", np.cross(b - p, a - p), p)
        reflex = np.sign(turn) != np.sign(S)
        th = np.where(reflex, 2.0 * np.pi - th, th)
        convex &= ~reflex
        worst = np.maximum(worst, th)
    return S, worst, convex


def _price(lr: np.ndarray) -> np.ndarray:
    """The price of the pair log ratios lr: lr^2 + TAIL_W max(0, |lr| - ln TAIL_RATIO)^2."""
    return lr * lr + TAIL_W * np.maximum(np.abs(lr) - np.log(TAIL_RATIO), 0.0) ** 2


def _corner_price(w: np.ndarray, soft: float, lim: float) -> np.ndarray:
    """The price of the cells' worst corners w (rad): ANGLE_W over `soft`, LIMIT_W more over `lim` less
    LIMIT_MARGIN_DEG (a corner priced only above the limit stays a hair above it)."""
    return (ANGLE_W * np.maximum(w - soft, 0.0) ** 2
            + LIMIT_W * np.maximum(w - lim + np.radians(LIMIT_MARGIN_DEG), 0.0) ** 2)


def _mean_edge(V: np.ndarray, m: int) -> np.ndarray:
    L = np.zeros(V.shape[:-2])
    for i in range(m):
        L = L + np.linalg.norm(V[..., (i + 1) % m, :] - V[..., i, :], axis=-1)
    return L / m


def even_areas(mesh: Mesh, movable: np.ndarray, h_min: float, limit_deg: float = 120.0, sweeps: int = EVEN_SWEEPS,
               log=None, verbose: bool = False) -> tuple[Mesh, dict]:
    """The pass of the module docstring on the nodes `movable` (N,) bool (a node on the mesh's boundary never moves).
    Returns (the mesh with the new positions, record: sweeps, nodes moved, F before / after, the largest pair ratio
    and the pairs above 1.5 round the movable nodes before / after)."""
    P = mesh.points / np.linalg.norm(mesh.points, axis=1, keepdims=True)
    el, nv = _cells(mesh)
    E = len(el)
    topo = build_topology(mesh)
    mov = np.asarray(movable, bool) & ~topo.boundary_node
    ee = topo.edge_elems
    inn = ee[:, 1] >= 0
    pa, pb = ee[inn, 0], ee[inn, 1]
    nbr = [[] for _ in range(E)]
    for a, b in zip(pa.tolist(), pb.tolist()):
        nbr[a].append(b)
        nbr[b].append(a)
    inc = [[] for _ in range(mesh.n_nodes)]
    for e in range(E):
        for k in range(nv[e]):
            inc[el[e, k]].append(e)
    lim = np.radians(limit_deg)
    soft = np.radians(ANGLE_SOFT_DEG)
    floor = h_min / R_EARTH_M
    mult = np.where(nv == 3, 2.0, 1.0)

    def cell_state(ids):
        A = np.zeros(len(ids))
        W = np.zeros(len(ids))
        for m in (3, 4):
            k = nv[ids] == m
            if k.any():
                S, w, _ = _geom(P[el[ids[k], :m]], m)
                A[k], W[k] = np.abs(S), w
        return A * mult[ids], W

    eff, worst = cell_state(np.arange(E))
    sign = np.ones(E)
    for m in (3, 4):
        k = np.flatnonzero(nv == m)
        if len(k):
            sign[k] = np.sign(_geom(P[el[k, :m]], m)[0])
    cap = np.maximum(worst, lim)                              # a corner never grows beyond max(limit, before)
    near = np.zeros(E, bool)
    for i in np.flatnonzero(mov):
        near[inc[i]] = True
    sel = near[pa] | near[pb]

    def pair_stats():
        r = np.maximum(eff[pa], eff[pb]) / np.maximum(np.minimum(eff[pa], eff[pb]), 1e-300)
        r = r[sel]
        return (float(r.max()) if len(r) else 1.0), int((r > 1.5).sum()), float(_price(np.log(r)).sum())

    rec = {"movable": int(mov.sum()), "limit_deg": float(limit_deg)}
    rec["ratio_max_before"], rec["pairs_gt15_before"], F0 = pair_stats()
    F0 += _corner_price(worst[near], soft, lim).sum()
    rec["F_before"] = F0
    nodes = np.flatnonzero(mov)
    ang = 2.0 * np.pi * np.arange(EVEN_DIRS) / EVEN_DIRS
    moved = np.zeros(mesh.n_nodes, bool)
    F_prev, done = F0, 0
    for sw in range(sweeps):
        for i in nodes:
            C = np.asarray(inc[i])
            pairs = [(c, n) for c in C for n in nbr[c] if not (n in C and n < c)]
            if not pairs:
                continue
            pc = np.array([p[0] for p in pairs])
            pn = np.array([p[1] for p in pairs])
            n_in = np.isin(pn, C)
            x = P[i]
            e1 = np.cross(x, [0.0, 0.0, 1.0] if abs(x[2]) < 0.9 else [1.0, 0.0, 0.0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(x, e1)
            h = 0.0
            for c in C:
                h += _mean_edge(P[el[c, :nv[c]]][None], nv[c])[0]
            h /= len(C)
            steps = np.array(EVEN_STEPS) * h
            dirs = (np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2)
            X = np.vstack([x[None]] + [x + s * dirs for s in steps])
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            K = len(X)
            effC = np.zeros((K, len(C)))
            worC = np.zeros((K, len(C)))
            ok = np.ones(K, bool)
            for j, c in enumerate(C):
                m = nv[c]
                V = np.repeat(P[el[c, :m]][None], K, axis=0)
                V[:, list(el[c, :m]).index(i)] = X
                S, w, cv = _geom(V, m)
                effC[:, j] = np.abs(S) * mult[c]
                worC[:, j] = w
                ok &= cv & (np.sign(S) == sign[c]) & (w <= cap[c] + 1e-9) & (_mean_edge(V, m) >= floor)
            ok[0] = True
            pos = {c: j for j, c in enumerate(C)}
            ac = effC[:, [pos[c] for c in pc]]
            an = np.where(n_in[None], effC[:, [pos.get(n, 0) for n in pn]], eff[pn][None])
            F = _price(np.log(ac / np.maximum(an, 1e-300))).sum(1) + _corner_price(worC, soft, lim).sum(1)
            F = np.where(ok, F, np.inf)
            k = int(np.argmin(F))
            if k > 0 and F[k] < F[0] - 1e-12:
                P[i] = X[k]
                eff[C], worst[C] = effC[k], worC[k]
                cap[C] = np.minimum(cap[C], np.maximum(worC[k], lim))   # a corner above the limit never grows back
                moved[i] = True
        r_max, n15, Fp = pair_stats()
        F_now = Fp + float(_corner_price(worst[near], soft, lim).sum())
        done = sw + 1
        if verbose and log is not None:
            log(f"even areas sweep {done}: F {F_now:.3f}, pairs above 1.5 {n15}, largest {r_max:.2f}")
        if F_prev - F_now < EVEN_TOL * max(F_prev, 1e-12):
            F_prev = F_now
            break
        F_prev = F_now
    rec["ratio_max_after"], rec["pairs_gt15_after"], _ = pair_stats()
    rec.update(F_after=F_prev, sweeps=done, nodes_moved=int(moved.sum()))
    if log is not None:
        log(f"even areas: {rec['movable']} free nodes, {done} sweeps, {rec['nodes_moved']} moved; pairs above 1.5 "
            f"{rec['pairs_gt15_before']} -> {rec['pairs_gt15_after']}, largest {rec['ratio_max_before']:.2f} -> "
            f"{rec['ratio_max_after']:.2f}; F {F0:.2f} -> {F_prev:.2f}")
    out = mesh.points.copy()
    out[moved] = P[moved]
    return Mesh(out, mesh.tris, mesh.quads, mesh.node_flag, mesh.tri_zone), rec
