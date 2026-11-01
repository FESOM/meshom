"""Depth per mesh node from a bathymetry grid (`[bathymetry]`, the `bathymetry` stage of `meshom run`).

node_depth(mesh, path, variable, min_depth_m) -> depth (N,) metres, positive down, for every node of the final mesh
(steps 1-3; meshom.run passes the floor of step 3 as min_depth_m), then depth_policy (step 4):

1. **Box mean of the ocean cells.** The grid is a regular lon-lat grid of elevation (positive up, ocean < 0; GEBCO
   2024: 15", 43200 x 86400 int16, cell centres). A node's depth is minus the mean elevation of the grid's ocean
   cells inside a lon-lat box around the node with half-width r = half the mean length of the node's edges (in metres
   north-south, r / cos(lat) east-west): the box has the area h^2 of the node's control cell (median dual of an
   equilateral mesh: 0.87 h^2), so the depth stands for the water the node represents, as FESOM2 uses it (element
   depth = mean of its 3 nodes, the bottom level per element); `[bathymetry] box_factor` scales r (B222: 2 = a box
   of 4 h^2, a smoother depth). Land cells (>= 0) are left
   out, so a coastal node gets the depth of its wet part, not a mix with the land height. This rule is after
   D. Sein (AWI), fesom2_mesh_gen; the nearest grid cell would sample a 450 m pixel for a 25-110 km cell.
2. **Nodes without an ocean cell** (GSHHS water where the grid has land: narrow bays, lagoons, the GSHHS / GEBCO
   coast mismatch) take the mean depth of their known neighbours, weights 1 / edge length, ring by ring inwards
   (as in fesom2_mesh_gen); a component without any known node gets min_depth_m.
3. **depth = max(depth, floor)**, floor = min_depth_of(min_depth_m, levels, min_layers) (30 m with the CORE2
   levels: what FESOM2 / FESOM3 raise shallower nodes to). No maximum (FESOM2 aborts only above 11 000 m, GEBCO's
   deepest cell is 10 919 m; depths below the last level take all levels, as in FESOM2).
4. **depth_policy**: volume-conserving smoothing (smooth_depth), an optional slope limiter (limit_slope), carving of
   the straits' depth floors with a cosine taper (meshom.straits.depth_floor), the floor again.
5. after the method's own node depths (m16_orca, meshom.run._method_node_depth): **apply_regions**, the
   `[[bathymetry.regions]]` rules by area in file order, at the floor (C1, decisions O35, O36; meshom.run.model_depth).
   `meshom write` then applies the mesh file's node edits (**apply_node_edits**, C2, O37), at the floor too.

The grid is read in bands of rows (`band_rows`, plus the box margin), never whole: per band the summed-area tables
of the ocean elevation and of the ocean count (int64, exact) answer every box of the band's nodes in O(1); longitude
is periodic (the band is padded by the largest east-west half-width), boxes are clipped at the poles. Memory: about
20 bytes per cell of a padded band (a 600-row GEBCO band: ~1.3 GB).

Levels: `levels(spec)` turns `[bathymetry] levels` into the FESOM2 level surfaces zbar (metres, positive down): a list,
a named set of LEVEL_SETS (default "core2"), or a file holding nl and then nl values (aux3d.out, depth_zlev.out).
"""
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from numba import njit

from meshom.geo import R_EARTH_M, arc_length, lonlat_to_xyz, xyz_to_lonlat
from meshom.mesh import Mesh, build_topology

# The 48 level surfaces (47 layers) of the FESOM2 CORE2 mesh: /pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2/aux3d.out,
# lines 1-49 (written there as 0, -5, ...); fesom2_mesh_gen's meshes use the same levels.
CORE2_LEVELS_M = (0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 115.0, 135.0, 160.0, 190.0,
                  230.0, 280.0, 340.0, 410.0, 490.0, 580.0, 680.0, 790.0, 910.0, 1040.0, 1180.0, 1330.0, 1500.0,
                  1700.0, 1920.0, 2150.0, 2400.0, 2650.0, 2900.0, 3150.0, 3400.0, 3650.0, 3900.0, 4150.0, 4400.0,
                  4650.0, 4900.0, 5150.0, 5400.0, 5650.0, 6000.0, 6250.0)
LEVEL_SETS = {"core2": CORE2_LEVELS_M}


def levels(spec, base: Path | None = None) -> np.ndarray:
    """[bathymetry] levels -> zbar (nl,) metres, positive down. A string is a LEVEL_SETS name or a file (relative to
    `base`): first number nl, then nl level depths of either sign."""
    if isinstance(spec, str):
        if spec in LEVEL_SETS:
            return np.asarray(LEVEL_SETS[spec], dtype=np.float64)
        p = Path(spec).expanduser()
        if not p.is_absolute() and base is not None:
            p = Path(base) / p
        tok = p.read_text().split()
        return np.abs(np.asarray(tok[1:1 + int(float(tok[0]))], dtype=np.float64))
    return np.abs(np.asarray(spec, dtype=np.float64))


def node_spacing_m(mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean edge length per node in metres, edges (E,2), edge lengths (E,) metres)."""
    e = build_topology(mesh).edges
    L = R_EARTH_M * arc_length(mesh.points[e[:, 0]], mesh.points[e[:, 1]])
    n = mesh.n_nodes
    cnt = np.bincount(e.ravel(), minlength=n)
    return np.bincount(e.ravel(), weights=np.repeat(L, 2), minlength=n) / np.maximum(cnt, 1), e, L


def _sat(a: np.ndarray) -> np.ndarray:
    """Summed-area table with a zero first row and column: S[i, j] = a[:i, :j].sum()."""
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.int64)
    np.cumsum(a, axis=0, dtype=np.int64, out=s[1:, 1:])
    np.cumsum(s[1:, 1:], axis=1, out=s[1:, 1:])
    return s


def _box(s: np.ndarray, i0, i1, j0, j1) -> np.ndarray:
    return s[i1, j1] - s[i0, j1] - s[i1, j0] + s[i0, j0]


def _grid_boxes(lon: np.ndarray, lat: np.ndarray, r_m: np.ndarray, glat: np.ndarray,
                glon: np.ndarray) -> np.ndarray:
    """(N, 4) int64 box of each node on the grid: centre row and column (ci, cj), half-widths in rows and columns
    (hi, hj). The box sums of box_means are a function of these four integers and the grid alone."""
    ny, nx = len(glat), len(glon)
    dlat, dlon = (glat[-1] - glat[0]) / (ny - 1), (glon[-1] - glon[0]) / (nx - 1)
    ci = np.clip(np.rint((lat - glat[0]) / dlat).astype(np.int64), 0, ny - 1)
    cj = np.rint((lon - glon[0]) / dlon).astype(np.int64) % nx
    hi = np.floor(r_m / (R_EARTH_M * np.radians(abs(dlat)))).astype(np.int64)
    coslat = np.maximum(np.cos(np.radians(lat)), 1e-6)
    hj = np.minimum(np.floor(r_m / (R_EARTH_M * np.radians(abs(dlon)) * coslat)), nx // 2 - 1).astype(np.int64)
    return np.column_stack([ci, cj, hi, hj])


def _band_sums(var, box: np.ndarray, band_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """(sum of the ocean elevations, count of ocean cells) in each box (N, 4) of _grid_boxes, the grid `var` read in
    bands of rows. A box is clipped at the first and last row only, so its sums do not depend on the bands or on the
    other boxes asked."""
    ny, nx = var.shape
    ci, cj, hi, hj = box.T
    ssum = np.zeros(len(box), dtype=np.int64)
    scnt = np.zeros(len(box), dtype=np.int64)
    for r0 in range(0, ny, band_rows):
        sel = np.flatnonzero((ci >= r0) & (ci < r0 + band_rows))
        if not len(sel):
            continue
        a0 = max(0, r0 - int(hi[sel].max()))
        a1 = min(ny, r0 + band_rows + int(hi[sel].max()) + 1)
        z = np.asarray(var[a0:a1, :])
        pad = int(hj[sel].max()) + 1
        z = np.concatenate([z[:, nx - pad:], z, z[:, :pad]], axis=1)
        wet = z < 0
        i0 = np.clip(ci[sel] - hi[sel], a0, a1) - a0
        i1 = np.clip(ci[sel] + hi[sel] + 1, a0, a1) - a0
        j0 = cj[sel] - hj[sel] + pad
        j1 = cj[sel] + hj[sel] + 1 + pad
        s = _sat(np.where(wet, z, 0))
        ssum[sel] = _box(s, i0, i1, j0, j1)
        del s
        s = _sat(wet)
        scnt[sel] = _box(s, i0, i1, j0, j1)
        del s, z, wet
    return ssum, scnt


def _open_grid(ds, variable: str):
    var = ds[variable]
    var.set_auto_maskandscale(False)
    glat = np.asarray(ds[var.dimensions[0]][:], dtype=np.float64)
    glon = np.asarray(ds[var.dimensions[1]][:], dtype=np.float64)
    return var, glat, glon


def box_means(lon: np.ndarray, lat: np.ndarray, r_m: np.ndarray, path: Path, variable: str = "elevation",
              band_rows: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """(sum of the ocean elevations, count of ocean cells) in each node's box; see the module docstring."""
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        var, glat, glon = _open_grid(ds, variable)
        return _band_sums(var, _grid_boxes(lon, lat, r_m, glat, glon), band_rows)


BOX_CACHE_VERSION = 1      # bump when the box rule (_grid_boxes, _band_sums) changes: older caches are then ignored


def cached_box_means(lon: np.ndarray, lat: np.ndarray, r_m: np.ndarray, path: Path, cache: Path,
                     variable: str = "elevation", band_rows: int = 600) -> tuple[np.ndarray, np.ndarray, int]:
    """box_means through a cache file of box sums (`meshom write`): the boxes (_grid_boxes) found in `cache` take
    its sums, only the others are read from the grid (one band per band of rows they fall in), and the cache gets
    them. The sums are exact integers of the box and the grid, so the result equals box_means bit for bit. The
    cache is used only when it was made from the same grid file (path, size, modification time) and variable; it
    keeps every box it was given. Returns (ssum, scnt, number of boxes read from the grid)."""
    import os

    import netCDF4

    path, cache = Path(path).resolve(), Path(cache)
    st = path.stat()
    tag = f"{BOX_CACHE_VERSION}|{path}|{variable}|{st.st_size}|{st.st_mtime_ns}"
    with netCDF4.Dataset(path) as ds:
        var, glat, glon = _open_grid(ds, variable)
        box = _grid_boxes(lon, lat, r_m, glat, glon)
        keys, csum, ccnt = np.zeros((0, 4), np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
        if cache.exists():
            with np.load(cache) as z:
                if str(z["tag"]) == tag:
                    keys, csum, ccnt = z["keys"], z["ssum"], z["scnt"]
        M = len(keys)
        uniq, inv = np.unique(np.concatenate([keys, box]), axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        slot = np.full(len(uniq), -1, dtype=np.int64)
        slot[inv[:M]] = np.arange(M)
        hit = slot[inv[M:]]
        miss = hit < 0
        ssum, scnt = np.zeros(len(box), np.int64), np.zeros(len(box), np.int64)
        ssum[~miss], scnt[~miss] = csum[hit[~miss]], ccnt[hit[~miss]]
        new = np.zeros((0, 4), np.int64)
        if miss.any():
            new, back = np.unique(box[miss], axis=0, return_inverse=True)
            s_new, c_new = _band_sums(var, new, band_rows)
            ssum[miss], scnt[miss] = s_new[back.reshape(-1)], c_new[back.reshape(-1)]
    if len(new) or M == 0:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, tag=np.array(tag), keys=np.concatenate([keys, new]),
                     ssum=np.concatenate([csum, s_new]) if len(new) else csum,
                     scnt=np.concatenate([ccnt, c_new]) if len(new) else ccnt)
        os.replace(tmp, cache)
    return ssum, scnt, len(new)


def fill_from_neighbours(depth: np.ndarray, known: np.ndarray, edges: np.ndarray, length: np.ndarray,
                         fallback: float) -> tuple[np.ndarray, int, int]:
    """Unknown nodes take the 1/length-weighted mean of their known neighbours, one ring per pass; nodes no ring
    reaches get `fallback`. Returns (depth, rings, number left to the fallback)."""
    n = len(depth)
    w = 1.0 / np.maximum(length, 1e-9)
    W = sp.csr_matrix((np.r_[w, w], (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                      shape=(n, n))
    depth, known = depth.copy(), known.copy()
    rings = 0
    while not known.all():
        k = known.astype(np.float64)
        ws = W @ k
        new = ~known & (ws > 0)
        if not new.any():
            break
        depth[new] = (W @ (depth * k))[new] / ws[new]
        known |= new
        rings += 1
    left = int((~known).sum())
    depth[~known] = fallback
    return depth, rings, left


def min_depth_of(min_depth_m: float, zbar: np.ndarray | None, min_layers: int) -> float:
    """The depth floor of a run: max(min_depth_m, zbar[min_layers]) (zbar positive down, zbar[0] = 0 the surface:
    zbar[min_layers] is the bottom of the min_layers-th layer); without levels (zbar None) or min_layers 0 just
    min_depth_m. FESOM2 and FESOM3 (fesom_ini) raise every node above zbar(thers_zbar_lev = 5) (1-based, = zbar[4]:
    30 m with the CORE2 levels) and give every element at least 4 layers, so
    min_layers = 4 writes the depth the model will use."""
    if zbar is None or min_layers <= 0:
        return float(min_depth_m)
    return float(max(min_depth_m, zbar[min(min_layers, len(zbar) - 1)]))


def node_area_m2(mesh: Mesh) -> np.ndarray:
    """(N,) the node's share of its elements' areas: area / 3 of each triangle, / 4 of each quad (FESOM's
    median-dual control volume of a triangle node is area / 3)."""
    from meshom.mesh import signed_area

    a = np.abs(signed_area(mesh))
    out = np.zeros(mesh.n_nodes)
    if mesh.n_tris:
        np.add.at(out, mesh.tris.ravel(), np.repeat(a[:mesh.n_tris] / 3.0, 3))
    if mesh.n_quads:
        np.add.at(out, mesh.quads.ravel(), np.repeat(a[mesh.n_tris:] / 4.0, 4))
    return out


def smooth_depth(depth: np.ndarray, edges: np.ndarray, area: np.ndarray, passes: int,
                 nodes: np.ndarray | None = None) -> np.ndarray:
    """`passes` passes of volume-conserving diffusion on the mesh graph: per edge (i, j) the flux
    F = beta_ij (d_j - d_i), beta_ij = min(A_i / deg_i, A_j / deg_j) / 2, d_i += sum_j F_ij / A_i. The flux is
    antisymmetric, so sum_i A_i d_i (the water volume) is unchanged; the weight on the node itself stays >= 1/2, so
    every pass is a convex combination (no new minimum or maximum: the depth floor holds). Each pass moves a node
    about half way to its neighbours' mean (fesom2_mesh_gen's pass, (d_i + sum_j d_j) / (1 + deg_i), moves it
    deg / (1 + deg) of the way and does not conserve the volume). With `nodes` (bool (N,)) only edges with both ends
    in the set exchange water: the smoothing stays inside it and conserves the volume there."""
    d = np.asarray(depth, dtype=np.float64).copy()
    if passes <= 0:
        return d
    e = edges if nodes is None else edges[nodes[edges[:, 0]] & nodes[edges[:, 1]]]
    deg = np.bincount(e.ravel(), minlength=len(d))
    a, b = e[:, 0], e[:, 1]
    beta = 0.5 * np.minimum(area[a] / np.maximum(deg[a], 1), area[b] / np.maximum(deg[b], 1))
    inv = 1.0 / area
    for _ in range(passes):
        f = beta * (d[b] - d[a])
        d += (np.bincount(a, weights=f, minlength=len(d)) - np.bincount(b, weights=f, minlength=len(d))) * inv
    return d


@njit(cache=True)
def _close_steep_edges(d, a, b, lim, inv_a, inv_b, ptr, inc, tol, max_sweeps):
    """Gauss-Seidel sweeps of limit_slope over the steep edges, in place on d; returns the number of sweeps. A sweep
    closes each steep edge of the active list in turn (forward, then backward in the next sweep); the next list holds
    the steep edges at the nodes the sweep moved (no other edge changed). Stops when no edge is steep or after
    max_sweeps sweeps."""
    m = len(a)
    seen = np.full(m, -1, np.int64)
    moved = np.full(len(d), -1, np.int64)
    nodes = np.empty(len(d), np.int64)
    active = np.empty(m, np.int64)
    na = 0
    for e in range(m):
        if abs(d[a[e]] - d[b[e]]) - lim[e] > tol * lim[e]:
            active[na] = e
            na += 1
    sweeps = 0
    while na > 0 and sweeps < max_sweeps:
        sweeps += 1
        nn = 0
        for k in range(na):
            e = active[k] if sweeps % 2 == 1 else active[na - 1 - k]
            i, j = a[e], b[e]
            step = d[i] - d[j]
            exc = abs(step) - lim[e]
            if exc > tol * lim[e]:
                f = exc / (inv_a[e] + inv_b[e])                # closes the excess: |d_i - d_j| = lim afterwards
                if step < 0.0:
                    f = -f
                d[i] -= f * inv_a[e]
                d[j] += f * inv_b[e]
                for v in (i, j):
                    if moved[v] != sweeps:
                        moved[v] = sweeps
                        nodes[nn] = v
                        nn += 1
        na = 0
        for k in range(nn):
            v = nodes[k]
            for q in range(ptr[v], ptr[v + 1]):
                e = inc[q]
                if seen[e] != sweeps:
                    seen[e] = sweeps
                    if abs(d[a[e]] - d[b[e]]) - lim[e] > tol * lim[e]:
                        active[na] = e
                        na += 1
        active[:na] = np.sort(active[:na])
    return sweeps


def limit_slope(depth: np.ndarray, edges: np.ndarray, length: np.ndarray, area: np.ndarray, max_slope: float,
                tol: float = 1e-3, max_iter: int | None = None) -> tuple[np.ndarray, dict]:
    """Volume-conserving slope limiter: wherever |d_i - d_j| > max_slope L_ij on an edge, water moves from the
    deeper node to the shallower one (the deeper gets shallower, the shallower deeper) until the step is
    max_slope L_ij. Gauss-Seidel: each steep edge in turn carries the flux F = excess / (1/A_i + 1/A_j) that closes
    its excess, sweep after sweep over the steep edges and their neighbours, until no edge exceeds max_slope L_ij
    (1 + tol). Each closure is a projection onto the convex set |d_i - d_j| <= max_slope L_ij that keeps the volume,
    and a flat depth lies in every set, so the sweeps converge; a ramp of k cells takes of the order of k^2 sweeps
    (0.5 k^2 on quads, 2 k^2 on triangles), each over the steep edges only. sum_i A_i d_i is unchanged, depths stay
    within the input's min and max (water only moves down a step, never past the other node). `max_iter` caps the
    sweeps (default 10 k^2 + 1000, k = depth range / (max_slope min L)); steep edges left raise RuntimeError, the
    depth is never returned above the limit. Returns the depth and a record."""
    d = np.asarray(depth, dtype=np.float64).copy()
    edges = np.asarray(edges, dtype=np.int64)
    a, b = edges[:, 0], edges[:, 1]
    lim = max_slope * np.asarray(length, dtype=np.float64)
    if max_iter is None:
        k = (d.max() - d.min()) / lim.min()
        max_iter = int(min(10.0 * k * k + 1000.0, 2**62))
    ends = edges.ravel()
    order = np.argsort(ends, kind="stable")
    ptr = np.zeros(len(d) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(ends, minlength=len(d)))
    inv = 1.0 / np.asarray(area, dtype=np.float64)
    it = _close_steep_edges(d, a, b, lim, inv[a], inv[b], ptr, order // 2, tol, max_iter)
    exc = np.abs(d[a] - d[b]) - lim
    rec = {"max_slope": max_slope, "iterations": int(it), "edges_steep_left": int((exc > tol * lim).sum())}
    if rec["edges_steep_left"]:
        raise RuntimeError(f"limit_slope: {rec['edges_steep_left']} steep edges left after {it} sweeps "
                           f"(max_slope {max_slope}, slope up to {float(np.max(np.abs(d[a] - d[b]) / length)):.4g})")
    return d, rec


def node_depth(mesh: Mesh, path: Path, variable: str = "elevation", min_depth_m: float = 10.0,
               band_rows: int = 600, cache: Path | None = None, box_factor: float = 1.0) -> tuple[np.ndarray, dict]:
    """Depth (N,) metres positive down per node of `mesh` and a record for stages.json (module docstring). `cache`:
    a box-sum cache file (cached_box_means; `meshom write`): only the boxes not in it are read from the grid, the
    depth is the same bit for bit; the record then also has "cache" and "n_boxes_read". `box_factor` (`[bathymetry]
    box_factor`, B222): the box's side in units of the node's mean edge length (half-width box_factor x half of
    it; 2 = a box of 4 times the area, a smoother depth)."""
    lon, lat = xyz_to_lonlat(mesh.points)
    spacing, edges, length = node_spacing_m(mesh)
    r = 0.5 * box_factor * spacing
    if cache is None:
        ssum, scnt = box_means(lon, lat, r, Path(path), variable, band_rows)
    else:
        ssum, scnt, n_read = cached_box_means(lon, lat, r, Path(path), Path(cache), variable, band_rows)
    known = scnt > 0
    raw = np.where(known, -ssum / np.maximum(scnt, 1), 0.0)
    depth, rings, left = fill_from_neighbours(raw, known, edges, length, min_depth_m)
    shallow = depth < min_depth_m
    depth = np.maximum(depth, min_depth_m)
    info = {"source": str(path), "variable": variable, "min_depth_m": min_depth_m, "n_nodes": int(mesh.n_nodes),
            "box_factor": float(box_factor),
            "box_half_width_m": {"min": float(r.min()), "median": float(np.median(r)), "max": float(r.max())},
            "cells_per_box_median": float(np.median(scnt[known])) if known.any() else 0.0,
            "n_without_ocean_cell": int((~known).sum()), "fill_rings": rings, "n_fallback": left,
            "n_raised_to_min_depth": int(shallow.sum()),
            "depth_m": {"min": float(depth.min()), "median": float(np.median(depth)), "max": float(depth.max())}}
    if cache is not None:
        info.update(cache=str(cache), n_boxes_read=n_read)
    return depth, info


def cells_deepest_depth(mesh: Mesh, path: Path, variable: str = "elevation", min_depth_m: float = 10.0,
                        band_rows: int = 600, cache: Path | None = None) -> tuple[np.ndarray, dict]:
    """`depth_at = "cells_deepest"` (B103), FESOM3's ORCA rule on meshom's cells: an element's depth is minus the mean
    elevation of the grid's ocean cells in a lon-lat box around its centroid with the element's area (half-width
    sqrt(area) / 2; land cells left out, as node_depth), and a node takes the deepest of its elements. So every
    element's nodes, and with them the element's levels in fesom_ini (the mean or the shallowest of its nodes), are
    at least as deep as the element's own water: a coastal node does not block a strait two or three elements wide.
    Nodes whose elements hold no ocean cell take node_depth's steps 2-3 (neighbours, then min_depth_m)."""
    P = mesh.points
    cent, area = [], []
    for E in (mesh.tris, mesh.quads):
        if not len(E):
            continue
        c = P[E].mean(axis=1)
        if E.shape[1] == 3:
            a = 0.5 * np.linalg.norm(np.cross(P[E[:, 1]] - P[E[:, 0]], P[E[:, 2]] - P[E[:, 0]]), axis=1)
        else:
            a = 0.5 * (np.linalg.norm(np.cross(P[E[:, 1]] - P[E[:, 0]], P[E[:, 2]] - P[E[:, 0]]), axis=1)
                       + np.linalg.norm(np.cross(P[E[:, 2]] - P[E[:, 0]], P[E[:, 3]] - P[E[:, 0]]), axis=1))
        cent.append(c / np.linalg.norm(c, axis=1, keepdims=True))
        area.append(a * R_EARTH_M ** 2)
    cent, area = np.vstack(cent), np.concatenate(area)
    lon, lat = xyz_to_lonlat(cent)
    r = 0.5 * np.sqrt(area)
    if cache is None:
        ssum, scnt = box_means(lon, lat, r, Path(path), variable, band_rows)
    else:
        ssum, scnt, _ = cached_box_means(lon, lat, r, Path(path), Path(cache), variable, band_rows)
    de = np.where(scnt > 0, -ssum / np.maximum(scnt, 1), -np.inf)
    d = np.full(mesh.n_nodes, -np.inf)
    T = mesh.n_tris
    if T:
        np.maximum.at(d, mesh.tris.ravel(), np.repeat(de[:T], 3))
    if mesh.n_quads:
        np.maximum.at(d, mesh.quads.ravel(), np.repeat(de[T:], 4))
    known = np.isfinite(d)
    _, edges, length = node_spacing_m(mesh)
    depth, rings, left = fill_from_neighbours(np.where(known, d, 0.0), known, edges, length, min_depth_m)
    shallow = depth < min_depth_m
    depth = np.maximum(depth, min_depth_m)
    info = {"source": str(path), "variable": variable, "rule": "cells_deepest", "min_depth_m": min_depth_m,
            "n_nodes": int(mesh.n_nodes), "n_elems": int(len(de)),
            "elems_without_ocean_cell": int((scnt == 0).sum()), "n_without_ocean_cell": int((~known).sum()),
            "fill_rings": rings, "n_fallback": left, "n_raised_to_min_depth": int(shallow.sum()),
            "depth_m": {"min": float(depth.min()), "median": float(np.median(depth)), "max": float(depth.max())}}
    return depth, info


def depth_policy(mesh: Mesh, depth: np.ndarray, floor_m: float, smooth_passes: int = 0, max_slope: float = 0.0,
                 straits: list | None = None, carve_taper: float = 1.0) -> tuple[np.ndarray, dict]:
    """The depth rules after the box mean, in this order: smoothing (smooth_depth, `smooth_passes` over the whole
    mesh), the slope limiter (limit_slope, `max_slope` > 0), carving of the straits' depth floors with a cosine taper
    (meshom.straits.depth_floor), then depth >= floor_m. Smoothing and limiting conserve the water volume and keep
    the depth within its range, so the floor only binds where carving or the input put it. The record says how many
    nodes moved and by how much, within the strait corridors and elsewhere."""
    d0 = np.asarray(depth, dtype=np.float64)
    _, edges, length = node_spacing_m(mesh)
    area = node_area_m2(mesh)
    rec = {"floor_m": floor_m, "smooth_passes": smooth_passes, "max_slope": max_slope}
    d = smooth_depth(d0, edges, area, smooth_passes)
    if max_slope > 0:
        d, rec["limit_slope"] = limit_slope(d, edges, length, area, max_slope)
    if straits:
        from meshom.straits import depth_floor

        d, rec["strait_depth_floor"] = depth_floor(mesh, d, straits, carve_taper)
    d = np.maximum(d, floor_m)
    near = np.zeros(mesh.n_nodes, dtype=bool)
    for s in straits or []:
        near |= s.distance_m(mesh.points, 0.5 * s.corridor_m) <= 0.5 * s.corridor_m
    dd = d - d0
    vol0 = float(area @ d0)
    for key, m in (("corridors", near), ("elsewhere", ~near)):
        a = np.abs(dd[m])
        rec[key] = {"n_nodes": int(m.sum()), "n_moved_1m": int((a > 1.0).sum()),
                    "n_moved_10pct": int((a > 0.1 * d0[m]).sum()),
                    "max_up_m": float(-dd[m].min(initial=0.0)), "max_down_m": float(dd[m].max(initial=0.0)),
                    "mean_abs_m": float(a.mean()) if len(a) else 0.0}
    rec["volume_change_rel"] = float(area @ d) / vol0 - 1.0
    rec["slope_p99_before"] = float(np.quantile(np.abs(d0[edges[:, 0]] - d0[edges[:, 1]]) / length, 0.99))
    rec["slope_p99_after"] = float(np.quantile(np.abs(d[edges[:, 0]] - d[edges[:, 1]]) / length, 0.99))
    rec["slope_max_after"] = float((np.abs(d[edges[:, 0]] - d[edges[:, 1]]) / length).max())
    return d, rec


# ------------------------------------------------------------------------- depth regions and node edits (C1, C2)

BOX_EPS_DEG = 1e-9       # a region box's edges hold the nodes on them despite the round-off of xyz -> lon, lat


def region_mask(xyz: np.ndarray, region) -> np.ndarray:
    """(N,) bool: the points xyz (N, 3) inside a `[[bathymetry.regions]]` entry: its `box` [lon_min, lat_min,
    lon_max, lat_max] in degrees, edges included, sides along meridians and parallels (lon_min > lon_max: the box
    crosses 180°, as does lon_max > 180), else its `polygon` [[lon, lat], ...] with great-circle edges
    (meshom.size.polygon_inside: within a hemisphere, any orientation, across 180° as well)."""
    if region.box:
        lo0, la0, lo1, la1 = region.box
        lon, lat = xyz_to_lonlat(xyz)
        span, e = lo1 - lo0, BOX_EPS_DEG
        inlon = np.ones(len(xyz), bool) if span >= 360.0 else (lon - lo0 + e) % 360.0 <= span % 360.0 + 2 * e
        return inlon & (lat >= la0 - e) & (lat <= la1 + e)
    from meshom.size import polygon_inside

    p = np.asarray(region.polygon, dtype=np.float64)
    return polygon_inside(xyz, lonlat_to_xyz(p[:, 0], p[:, 1]))


def apply_regions(mesh: Mesh, depth: np.ndarray, regions: list, floor_m: float) -> tuple[np.ndarray, list]:
    """C1 (decisions O35, O36): the `[[bathymetry.regions]]` rules on the node depths, in file order, each on the
    nodes inside its area (region_mask): `depth_m` sets the depth, then `min_depth_m` raises to it, then
    `max_depth_m` lowers to it (the keys given), then depth >= floor_m (the run's floor, min_depth_of: a rule below it
    gives the floor, the depth the model uses). A later region overrides an earlier one on their common nodes.
    Returns (depth, one record per region: name, n_nodes inside, n_changed, n_clamped_to_floor)."""
    d = np.array(depth, dtype=np.float64)
    recs = []
    for r in regions:
        m = region_mask(mesh.points, r)
        v = d[m]
        if r.depth_m is not None:
            v = np.full_like(v, r.depth_m)
        if r.min_depth_m is not None:
            v = np.maximum(v, r.min_depth_m)
        if r.max_depth_m is not None:
            v = np.minimum(v, r.max_depth_m)
        low = v < floor_m
        v = np.maximum(v, floor_m)
        recs.append({"name": r.name, "n_nodes": int(m.sum()), "n_changed": int((v != d[m]).sum()),
                     "n_clamped_to_floor": int(low.sum())})
        d[m] = v
    return d, recs


EDIT_OPS = {"set": lambda d, D: np.full_like(d, D), "min": np.maximum, "max": np.minimum}


def apply_node_edits(depth: np.ndarray, path: Path, floor_m: float) -> tuple[np.ndarray, dict]:
    """C2 (decisions O37): the node edits of one mesh file, `<mesh stem>.depth_edits.json` next to it ({"mesh",
    "n_nodes", "edits": [{"op": "set" | "min" | "max", "depth_m", "nodes": [0-based node ids of the mesh file],
    "outline", "date"}]}), applied in order on `depth` (the depth after C1): "set" d = depth_m, "min" d =
    max(d, depth_m), "max" d = min(d, depth_m); then the edited nodes >= floor_m (an edit below the floor gives the
    floor, as FESOM would raise the node). An edits file whose n_nodes is not the mesh's belongs to another mesh:
    ValueError. Returns (depth, record {file, n_edits, n_nodes_changed, n_clamped_to_floor, floor_m})."""
    e = json.loads(Path(path).read_text())
    d0 = np.asarray(depth, dtype=np.float64)
    if e["n_nodes"] != len(d0):
        raise ValueError(f"{path}: n_nodes {e['n_nodes']} is not the mesh's {len(d0)}: the edits of another mesh")
    d = d0.copy()
    edited = np.zeros(len(d), bool)
    for ed in e["edits"]:
        i = np.asarray(ed["nodes"], dtype=np.int64)
        d[i] = EDIT_OPS[ed["op"]](d[i], float(ed["depth_m"]))
        edited[i] = True
    low = edited & (d < floor_m)
    d[low] = floor_m
    return d, {"file": str(path), "n_edits": len(e["edits"]), "n_nodes_changed": int((d != d0).sum()),
               "n_clamped_to_floor": int(low.sum()), "floor_m": float(floor_m)}
