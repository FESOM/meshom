"""The insert's channel depth rules (decisions B213-B215), all on the bathymetry grid at the mesh's scale.

FESOM3's level finder gives an element the level surface nearest the mean of its nodes' depths, a node the deepest
level of its elements, and the model takes an element's bottom at the shallowest of its nodes' levels. With cells
about as wide as a strait's deep channel, the box mean per node (B211) puts the channel's flank nodes on the shelf
and the channel's elements come out levels too shallow. These rules act on the insert's own nodes after its grid
rule (meshom.insert.core.insert_depth):

- the channel line (`channel_line`): the deepest path between the insert's two joins on the grid's box means at the
  insert's fine size (the cost per km (D_ref / D)^CHANNEL_POWER, D_ref the deepest connection of the box means
  between the ends), and the grid's own saddle between its ends (the deepest connection on the grid's cells); its
  ends are the deepest box means along the joins, or the basin pool beyond a join too shallow for the strait (B218),
  and it stays in the insert's corridor; no depth here comes from the host's nodes;
- `depth_channel` (`line_raise`, B213): "cells": every insert node of a cell the line crosses takes at least the
  cell's channel depth (the box mean, box of the cell's area, centred on the line's points inside the cell);
  "near": every insert node within CHANNEL_NEAR of its spacings of the line takes at least the box mean centred on
  the line's nearest point (box of the node's own size, node_depth's box moved onto the channel); both only on the
  line's narrows (`narrows`: the stretch round its narrowest point whose wet width across the line is at most
  CHANNEL_WIDTH_K times the narrowest), never on its legs in the open basins;
- `depth_quantile` (`node_quantile`, B214): the insert's nodes take a quantile of the grid's ocean cells in their
  node box instead of the mean;
- `sill_level = "grid"` (`grid_sill`, B215): the grid's saddle is held at its level: the bank-to-bank chain of the
  insert's nodes through the node nearest the saddle that follows the shallowest nodes (least sum of depths), and
  every node of a cell holding one of its nodes, capped just above the midpoint between L (the deepest level surface
  not below the saddle's grid depth) and the next level.
"""
from dataclasses import dataclass

import numpy as np

from meshom.geo import R_EARTH_M, lonlat_to_xyz, xyz_to_lonlat
from meshom.mesh import Mesh

CHANNEL_POWER = 4.0          # B213: the path's cost per km (D_ref / D)^CHANNEL_POWER on the grid's box means
CHANNEL_NEAR = 1.0           # B213: "near": insert nodes within this many of their spacings of the line are raised
SILL_MARGIN_M = 1.0          # B215: the dam is capped this far above the midpoint between the two levels (as B212)
CHANNEL_WIDTH_K = 6.0        # B213: the raise acts where the line is at most this many times as wide as its narrowest
CHANNEL_DIR_M = 2000.0       # B213: the line's direction at a point: the chord between its points this far on each side
WIDTH_CAP_M = 50000.0        # B213: the widths are first measured up to this far on each side (then up to the limit)
WIDTH_TURN_DEG = 45.0        # B213: a width is the shortest wet crossing within this angle of the line's normal ...
WIDTH_STEP_DEG = 15.0        # B213: ... tried every this many degrees
END_TOL_M = 0.5              # B218: the line's search is end-limited at an end when its deepest connection lies within
                             # this of the end's box mean


@dataclass
class Grid:
    """A lon-lat window of the bathymetry grid: cell centres (lon unwrapped round the window's centre) and the
    elevation (positive up, ocean < 0), as read."""
    lon: np.ndarray
    lat: np.ndarray
    elev: np.ndarray
    centre: float

    @property
    def dlat(self) -> float:
        return float((self.lat[-1] - self.lat[0]) / (len(self.lat) - 1))

    @property
    def dlon(self) -> float:
        return float((self.lon[-1] - self.lon[0]) / (len(self.lon) - 1))

    def unwrap(self, lon: np.ndarray) -> np.ndarray:
        return (np.asarray(lon, dtype=np.float64) - self.centre + 180.0) % 360.0 - 180.0 + self.centre

    def boxes(self, lon, lat, r_m) -> tuple:
        """meshom.bathymetry._grid_boxes on the window: centre row and column, half-widths in rows and columns."""
        lon, lat, r_m = self.unwrap(lon), np.asarray(lat, dtype=np.float64), np.asarray(r_m, dtype=np.float64)
        ci = np.clip(np.rint((lat - self.lat[0]) / self.dlat).astype(np.int64), 0, len(self.lat) - 1)
        cj = np.clip(np.rint((lon - self.lon[0]) / self.dlon).astype(np.int64), 0, len(self.lon) - 1)
        hi = np.floor(r_m / (R_EARTH_M * np.radians(abs(self.dlat)))).astype(np.int64)
        coslat = np.maximum(np.cos(np.radians(lat)), 1e-6)
        hj = np.floor(r_m / (R_EARTH_M * np.radians(abs(self.dlon)) * coslat)).astype(np.int64)
        return ci, cj, hi, hj

    def box_mean(self, lon, lat, r_m) -> np.ndarray:
        """minus the mean elevation of the ocean cells in each box (node_depth's rule), NaN without one."""
        from meshom.bathymetry import _box, _sat

        if not hasattr(self, "_s"):
            wet = self.elev < 0
            self._s = (_sat(np.where(wet, self.elev, 0)), _sat(wet))
        ci, cj, hi, hj = self.boxes(lon, lat, r_m)
        ny, nx = self.elev.shape
        i0, i1 = np.clip(ci - hi, 0, ny), np.clip(ci + hi + 1, 0, ny)
        j0, j1 = np.clip(cj - hj, 0, nx), np.clip(cj + hj + 1, 0, nx)
        s, c = _box(self._s[0], i0, i1, j0, j1), _box(self._s[1], i0, i1, j0, j1)
        return np.where(c > 0, -s / np.maximum(c, 1), np.nan)

    def smoothed(self, r_m: float) -> np.ndarray:
        """(ny, nx) the box mean at every ocean cell, half-width r_m (NaN on land)."""
        ny, nx = self.elev.shape
        jj = np.arange(nx)
        out = np.full((ny, nx), np.nan)
        for i in range(ny):
            out[i] = self.box_mean(self.lon[jj], np.full(nx, self.lat[i]), np.full(nx, r_m))
        return np.where(self.elev < 0, out, np.nan)

    def cell(self, lon, lat) -> tuple[int, int]:
        ci, cj, _, _ = self.boxes(np.atleast_1d(lon), np.atleast_1d(lat), np.zeros(1))
        return int(ci[0]), int(cj[0])


def read_window(path, variable: str, lon: np.ndarray, lat: np.ndarray, margin_m: float) -> Grid:
    """The grid's cells round the points (lon, lat) with margin_m on every side, read in one piece (two across the
    grid's longitude seam)."""
    import netCDF4

    from meshom.bathymetry import _open_grid

    c = float(np.degrees(np.arctan2(np.sin(np.radians(lon)).mean(), np.cos(np.radians(lon)).mean())))
    lu = (np.asarray(lon) - c + 180.0) % 360.0 - 180.0 + c
    dlat_m = np.degrees(margin_m / R_EARTH_M)
    la0, la1 = float(np.min(lat)) - dlat_m, float(np.max(lat)) + dlat_m
    dlon_m = dlat_m / max(np.cos(np.radians(min(89.0, max(abs(la0), abs(la1))))), 1e-6)
    lo0, lo1 = float(lu.min()) - dlon_m, float(lu.max()) + dlon_m
    with netCDF4.Dataset(path) as ds:
        var, glat, glon = _open_grid(ds, variable)
        ny, nx = len(glat), len(glon)
        dla, dlo = (glat[-1] - glat[0]) / (ny - 1), (glon[-1] - glon[0]) / (nx - 1)
        i0 = max(0, int(np.floor((la0 - glat[0]) / dla)))
        i1 = min(ny, int(np.ceil((la1 - glat[0]) / dla)) + 1)
        j0 = int(np.floor((lo0 - glon[0]) / dlo))
        j1 = int(np.ceil((lo1 - glon[0]) / dlo)) + 1
        j1 = min(j1, j0 + nx)
        a, b = j0 % nx, (j1 - 1) % nx + 1
        if a < b and j1 - j0 == b - a:
            z = np.asarray(var[i0:i1, a:b])
        else:
            z = np.concatenate([np.asarray(var[i0:i1, a:]), np.asarray(var[i0:i1, :b])], axis=1)
        glat_w = np.asarray(glat[i0:i1], dtype=np.float64)
        glon_w = glon[0] + np.arange(j0, j1) * dlo
    return Grid(glon_w, glat_w, z, c)


def _wet_cell(grid: Grid, D: np.ndarray, lon: float, lat: float) -> tuple[int, int]:
    """the cell with a finite D nearest (lon, lat) (in rows and cos-scaled columns)."""
    i, j = grid.cell(lon, lat)
    ii, jj = np.nonzero(np.isfinite(D))
    k = np.argmin((ii - i) ** 2 + ((jj - j) * np.cos(np.radians(lat))) ** 2)
    return int(ii[k]), int(jj[k])


def _maximin(D: np.ndarray, a: tuple, b: tuple, structure) -> tuple[float, float]:
    """(lo, hi): the deepest connection between cells a and b through the cells with D >= it (bisection: cells with
    D >= lo connect a and b, cells with D >= hi do not)."""
    from scipy import ndimage

    D0 = np.where(np.isfinite(D), D, -1.0)
    lo, hi = 0.0, float(min(D0[a], D0[b])) + 1e-6
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lab, _ = ndimage.label(D0 >= mid, structure=structure)
        if lab[a] and lab[a] == lab[b]:
            lo = mid
        else:
            hi = mid
    return lo, hi


def _path(grid: Grid, D: np.ndarray, a: tuple, b: tuple, d_ref: float) -> tuple[np.ndarray, np.ndarray]:
    """the cheapest path of 8-connected cells with a finite D from a to b, cost per km (d_ref / D)^CHANNEL_POWER."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    ny, nx = D.shape
    wet = np.isfinite(D)
    w = (d_ref / np.maximum(np.where(wet, D, 1.0), 1.0)) ** CHANNEL_POWER
    idx = np.arange(ny * nx).reshape(ny, nx)
    dy = R_EARTH_M / 1e3 * np.radians(abs(grid.dlat))
    dx = R_EARTH_M / 1e3 * np.radians(abs(grid.dlon)) * np.cos(np.radians(grid.lat))
    rows, cols, vals = [], [], []
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
        sa = (slice(0, ny - di), slice(max(0, -dj), nx - max(0, dj)))
        sb = (slice(di, ny), slice(max(0, dj), nx + min(0, dj)))
        ok = wet[sa] & wet[sb]
        L = np.sqrt((dj * 0.5 * (dx[:ny - di] + dx[di:]))[:, None] ** 2 + (di * dy) ** 2) * np.ones(ok.shape)
        rows.append(idx[sa][ok])
        cols.append(idx[sb][ok])
        vals.append((L * 0.5 * (w[sa] + w[sb]))[ok])
    g = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ny * nx,) * 2)
    _, pred = dijkstra(g.tocsr(), directed=False, indices=idx[a], return_predecessors=True)
    p = [idx[b]]
    while p[-1] != idx[a]:
        if pred[p[-1]] < 0:
            raise RuntimeError("channel_line: the joins are not connected on the grid's box means")
        p.append(pred[p[-1]])
    return np.unravel_index(np.array(p[::-1]), (ny, nx))


def _saddle(grid: Grid, a: tuple, b: tuple, mask: np.ndarray | None = None,
            near: tuple | None = None) -> tuple[float, float, float]:
    """The grid's own saddle between cells a and b (4-connected cells, the grid's depth; only the cells of `mask`
    when given): (depth, lon, lat) of the cell of the band at the critical depth that joins the two sides (the one
    nearest the band's centroid). When no single band component joins them (B218: several barriers at the critical
    depth in series, common on flat shoals of integer depths), the band components on a route of alternating band
    and deeper components from a's side to b's are the saddles, and the one nearest the cell `near` (default: the
    first) is taken."""
    from collections import deque

    from scipy import ndimage

    wet = grid.elev < 0 if mask is None else (grid.elev < 0) & mask
    D = np.where(wet, -grid.elev.astype(np.float64), np.nan)
    lo, hi = _maximin(D, a, b, None)
    D0 = np.where(np.isfinite(D), D, -1.0)
    lab, _ = ndimage.label(D0 >= hi)
    A = ndimage.binary_dilation(lab == lab[a]) if lab[a] else np.zeros(D.shape, bool)
    B = ndimage.binary_dilation(lab == lab[b]) if lab[b] else np.zeros(D.shape, bool)
    band, nb = ndimage.label((D0 >= lo) & (D0 < hi))
    objs = ndimage.find_objects(band)
    ny, nx = D.shape

    def cells(k):
        sl = objs[k - 1]
        ii, jj = np.nonzero(band[sl] == k)
        return ii + sl[0].start, jj + sl[1].start

    def centre(k):
        ii, jj = cells(k)
        c = int(np.argmin((ii - ii.mean()) ** 2 + (jj - jj.mean()) ** 2))
        return float(D0[ii[c], jj[c]]), float(grid.lon[jj[c]]), float(grid.lat[ii[c]])

    for k in range(1, nb + 1):
        sl = objs[k - 1]
        m = band[sl] == k
        if (m & A[sl]).any() and (m & B[sl]).any():
            return centre(k)
    if not (lab[a] and lab[b]):
        return float(lo), float("nan"), float("nan")
    touch = {}                                             # band component -> the deeper components round it
    for k in range(1, nb + 1):
        sl = objs[k - 1]
        i0, i1 = max(sl[0].start - 1, 0), min(sl[0].stop + 1, ny)
        j0, j1 = max(sl[1].start - 1, 0), min(sl[1].stop + 1, nx)
        hs = np.unique(lab[i0:i1, j0:j1][ndimage.binary_dilation(band[i0:i1, j0:j1] == k)])
        hs = [int(h) for h in hs if h]
        if len(hs) >= 2:
            touch[k] = hs
    via = {}
    for k, hs in touch.items():
        for h in hs:
            via.setdefault(h, []).append(k)
    prev = {("h", int(lab[a])): None}
    q = deque([("h", int(lab[a]))])
    goal = ("h", int(lab[b]))
    while q and goal not in prev:
        t, x = q.popleft()
        nxt = [("b", k) for k in via.get(x, [])] if t == "h" else [("h", h) for h in touch[x]]
        for y in nxt:
            if y not in prev:
                prev[y] = (t, x)
                q.append(y)
    if goal not in prev:
        return float(lo), float("nan"), float("nan")
    route, y = [], goal
    while y is not None:
        if y[0] == "b":
            route.append(y[1])
        y = prev[y]
    if near is None:
        return centre(route[-1])
    dist = [float(np.min((cells(k)[0] - near[0]) ** 2 + (cells(k)[1] - near[1]) ** 2)) for k in route]
    return centre(route[int(np.argmin(dist))])


def _host_joins(sub: Mesh, ins: np.ndarray) -> list:
    """the host nodes of the insert's cells (the joins and the coast nodes at their ends) in connected pieces, the
    largest first: [(nodes, edges (k, 2): the insert cells' edges between two of them)]."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = sub.n_nodes
    rows, cols = [], []
    for c in (sub.tris, sub.quads):
        if not len(c):
            continue
        c = c[ins[c].any(1)]
        for i in range(c.shape[1]):
            a, b = c[:, i], c[:, (i + 1) % c.shape[1]]
            ok = ~ins[a] & ~ins[b]
            rows.append(a[ok])
            cols.append(b[ok])
    host = np.zeros(n, bool)
    for c in (sub.tris, sub.quads):
        if len(c):
            cc = c[ins[c].any(1)]
            host[cc.ravel()] = True
    host &= ~ins
    r, cl = np.concatenate(rows), np.concatenate(cols)
    nc, lab = connected_components(coo_matrix((np.ones(len(r)), (r, cl)), shape=(n, n)), directed=False)
    pieces = [(np.flatnonzero(host & (lab == k)), np.c_[r, cl][lab[r] == k]) for k in range(nc)]
    pieces = [p for p in pieces if len(p[0]) > 1]
    return sorted(pieces, key=lambda p: len(p[0]), reverse=True)


def footprint(sub: Mesh, ins: np.ndarray, grid: Grid) -> np.ndarray:
    """(ny, nx) bool: the grid cells whose centre lies in a cell of the insert (a cell with a node of `ins`)."""
    from meshom.straits import _elem_tree, locate

    T = sub.n_tris
    cin = np.r_[ins[sub.tris].any(1) if T else np.zeros(0, bool), ins[sub.quads].any(1)]
    m = Mesh(sub.points, sub.tris[cin[:T]], sub.quads[cin[T:]], np.zeros(sub.n_nodes, np.int8))
    lo, la = xyz_to_lonlat(sub.points[np.unique(np.r_[m.tris.ravel(), m.quads.ravel()])])
    lo = grid.unwrap(lo)
    ci, cj, _, _ = grid.boxes(np.array([lo.min(), lo.max()]), np.array([la.min(), la.max()]), np.zeros(2))
    II, JJ = np.meshgrid(np.arange(ci.min(), ci.max() + 1), np.arange(cj.min(), cj.max() + 1), indexing="ij")
    II, JJ = II.ravel(), JJ.ravel()
    out = np.zeros(grid.elev.shape, bool)
    tree = _elem_tree(m)
    for s in range(0, len(II), 200000):
        i, j = II[s:s + 200000], JJ[s:s + 200000]
        out[i, j] = locate(m, lonlat_to_xyz(grid.lon[j], grid.lat[i]), tree) >= 0
    return out


def _join_cells(grid: Grid, points: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """the grid cells (rows, columns) of samples every half grid cell along the edges (great circles)."""
    from meshom.geo import arc_length

    step = 0.5 * R_EARTH_M * np.radians(abs(grid.dlat))
    a, b = points[edges[:, 0]], points[edges[:, 1]]
    n = np.maximum(1, np.ceil(R_EARTH_M * arc_length(a, b) / step).astype(np.int64))
    t = np.concatenate([np.linspace(0.0, 1.0, k + 1) for k in n])
    e = np.repeat(np.arange(len(edges)), n + 1)
    q = a[e] + t[:, None] * (b[e] - a[e])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    lo, la = xyz_to_lonlat(q)
    ci, cj, _, _ = grid.boxes(lo, la, np.zeros(len(lo)))
    return ci, cj


def _pool(grid: Grid, D: np.ndarray, fp: np.ndarray, seeds: np.ndarray, join_xyz: np.ndarray, other_xyz: np.ndarray,
          start: tuple) -> np.ndarray:
    """B218: the basin pool beyond a join: the wet cells (finite D) outside the insert's footprint `fp` joined to the
    join's cells (`seeds`, (ny, nx) bool: its sample cells and their 8 neighbours) through such cells, on the far side
    of the join's chord (the great circle through its two farthest nodes `join_xyz`; the far side: away from the
    other join's centre, `other_xyz` its nodes) and nearer to the join's nodes than to the other's, and joined to the
    join's deepest sample `start` through cells at least as deep as it (the pool never lowers the connection).
    (ny, nx) bool."""
    from scipy import ndimage
    from scipy.spatial import cKDTree

    out = np.isfinite(D) & ~fp
    lab, _ = ndimage.label(out, structure=np.ones((3, 3), bool))
    ids = np.unique(lab[seeds & out])
    pool = np.isin(lab, ids[ids > 0])
    if not pool.any():
        return pool
    ii, jj = np.nonzero(pool)
    q = lonlat_to_xyz(grid.lon[jj], grid.lat[ii])
    dd = np.linalg.norm(join_xyz[:, None, :] - join_xyz[None, :, :], axis=2)
    a, b = np.unravel_index(int(np.argmax(dd)), dd.shape)
    nrm = np.cross(join_xyz[a], join_xyz[b])                   # the chord's great circle
    nrm /= max(float(np.linalg.norm(nrm)), 1e-15)
    if float(np.dot(nrm, other_xyz.mean(0))) > 0:
        nrm = -nrm
    beyond = q @ nrm > 0
    near = cKDTree(join_xyz).query(q)[0] <= cKDTree(other_xyz).query(q)[0]
    keep = beyond & near
    pool[ii[~keep], jj[~keep]] = False
    reach = pool | seeds
    reach[start] = True
    lab2, _ = ndimage.label(reach & (np.where(np.isfinite(D), D, -1.0) >= D[start]), structure=np.ones((3, 3), bool))
    return pool & (lab2 == lab2[start])


@dataclass
class Line:
    lon: np.ndarray          # the path's cell centres (unwrapped), from join A's end to join B's
    lat: np.ndarray
    depth: np.ndarray        # the box means along it (the channel's profile at the fine size)
    saddle: tuple            # the grid's own saddle between the ends: (depth m, lon, lat)
    record: dict
    narrow: np.ndarray | None = None     # (n,) bool: the line's narrows (`narrows`), where depth_channel acts


def line_widths(grid: Grid, lon: np.ndarray, lat: np.ndarray, cap_m: float) -> np.ndarray:
    """(n,) the wet width (m) across a line of grid cells at each of its points: the shortest run of the grid's ocean
    cells through the point along a great circle within WIDTH_TURN_DEG of the line's normal (the line's direction:
    the chord between its points CHANNEL_DIR_M before and after; every WIDTH_STEP_DEG), sampled every half cell up
    to cap_m on each side; samples outside the grid's window count as water."""
    p = lonlat_to_xyz(lon, lat)
    n = len(p)
    s = np.r_[0.0, np.cumsum(2.0 * np.arcsin(np.minimum(np.linalg.norm(np.diff(p, axis=0), axis=1) / 2.0, 1.0)))]
    s *= R_EARTH_M
    a = np.clip(np.searchsorted(s, s - CHANNEL_DIR_M, side="left"), 0, n - 1)
    b = np.clip(np.searchsorted(s, s + CHANNEL_DIR_M, side="right") - 1, 0, n - 1)
    t = p[b] - p[a]
    t -= np.sum(t * p, axis=1)[:, None] * p
    t /= np.maximum(np.linalg.norm(t, axis=1), 1e-15)[:, None]
    nrm = np.cross(p, t)
    step = 0.5 * R_EARTH_M * np.radians(abs(grid.dlat))
    ang = np.arange(1, int(np.ceil(cap_m / step)) + 1) * step / R_EARTH_M
    la0, la1 = float(grid.lat.min()), float(grid.lat.max())
    lo0, lo1 = float(grid.lon.min()), float(grid.lon.max())
    best = np.full(n, np.inf)
    for turn in np.radians(np.arange(-WIDTH_TURN_DEG, WIDTH_TURN_DEG + 1e-9, WIDTH_STEP_DEG)):
        u = np.cos(turn) * nrm + np.sin(turn) * t
        width = np.full(n, step)
        for sgn in (1.0, -1.0):
            q = p[:, None, :] * np.cos(ang)[None, :, None] + sgn * u[:, None, :] * np.sin(ang)[None, :, None]
            qlo, qla = xyz_to_lonlat(q.reshape(-1, 3))
            ci, cj, _, _ = grid.boxes(qlo, qla, np.zeros(len(qlo)))
            ul = grid.unwrap(qlo)
            out = (qla < la0) | (qla > la1) | (ul < lo0) | (ul > lo1)
            wet = (out | (grid.elev[ci, cj] < 0)).reshape(n, len(ang))
            width += np.where(wet.all(axis=1), len(ang), np.argmin(wet, axis=1)) * step
        best = np.minimum(best, width)
    return best


def narrows(grid: Grid, lon: np.ndarray, lat: np.ndarray, inside: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """B213: the line's narrows: the run of its points round its narrowest one whose wet width across the line
    (line_widths) is at most CHANNEL_WIDTH_K times the narrowest; the narrowest point is sought among the points
    `inside` ((n,) bool: in the insert's cells, B218; None or none inside: all). Returns ((n,) bool, record)."""
    if inside is None or not np.any(inside):
        inside = np.ones(len(lon), bool)
    w = line_widths(grid, lon, lat, WIDTH_CAP_M)
    lim = CHANNEL_WIDTH_K * float(w[inside].min())
    if lim > WIDTH_CAP_M:                                  # every width up to the limit measured, not capped
        w = line_widths(grid, lon, lat, lim)
    i0 = int(np.flatnonzero(inside)[np.argmin(w[inside])])
    a = i0
    while a > 0 and w[a - 1] <= lim:
        a -= 1
    b = i0
    while b < len(w) - 1 and w[b + 1] <= lim:
        b += 1
    m = np.zeros(len(w), bool)
    m[a:b + 1] = True
    xyz = lonlat_to_xyz(lon[a:b + 1], lat[a:b + 1])
    km = float(R_EARTH_M / 1e3 * np.arccos(np.clip(np.sum(xyz[1:] * xyz[:-1], 1), -1, 1)).sum())
    pt = lambda i: [round(float(lon[i]), 4), round(float(lat[i]), 4)]          # noqa: E731
    return m, {"width_min_km": float(w[i0]) / 1e3, "at": pt(i0), "limit_km": lim / 1e3, "km": km,
               "from": pt(a), "to": pt(b), "cells": int(b - a + 1)}


def channel_line(sub: Mesh, ins: np.ndarray, grid: Grid, h_fine_m: float) -> Line | None:
    """The deepest channel between the insert's two joins (module docstring; B213, its ends B218). The joins: the two
    largest pieces of host nodes on the insert's cells, sampled along their edges every half grid cell; an end is
    the join's sample with the deepest box mean (the grid's ocean cells, half-width h_fine_m / 2); when the deepest
    connection between the ends reaches an end's depth (within END_TOL_M: the search is end-limited there), that end
    moves to the deepest cell of the basin pool beyond its join (_pool) when it is deeper. The path, the deepest
    connection and the grid's saddle stay in the corridor: the insert's cells widened by one fine cell, the joins'
    cells and the pools (the whole window when the corridor does not join the ends); when the grid's own depth limits
    the saddle's search at an end (no saddle cell found), the saddle is sought again from the deepest grid cell of
    each leg of the line (before and after its shallowest box mean); the narrows' narrowest point lies in the
    insert's cells. None when the insert has no two joins."""
    from scipy import ndimage

    pieces = _host_joins(sub, ins)
    if len(pieces) < 2:
        return None
    D = grid.smoothed(0.5 * h_fine_m)
    fp = footprint(sub, ins, grid)
    sq = np.ones((3, 3), bool)
    seeds, ends, src, join_m = [], [], [], []
    for _, edges in pieces[:2]:
        ci, cj = _join_cells(grid, sub.points, edges)
        m = np.zeros(D.shape, bool)
        m[ci, cj] = True
        seeds.append(ndimage.binary_dilation(m, sq))
        d = D[ci, cj]
        if np.isfinite(d).any():
            k = int(np.nanargmax(d))
            ends.append((int(ci[k]), int(cj[k])))
            src.append("join")
            join_m.append(float(d[k]))
        else:                                              # no wet sample on the join: the wet cell nearest its middle
            k = len(ci) // 2
            ends.append(_wet_cell(grid, D, float(grid.lon[cj[k]]), float(grid.lat[ci[k]])))
            src.append("nearest wet cell")
            join_m.append(float("nan"))
    dy = R_EARTH_M * np.radians(abs(grid.dlat))
    dx = R_EARTH_M * np.radians(abs(grid.dlon)) * max(np.cos(np.radians(float(np.mean(grid.lat)))), 1e-6)
    ki, kj = int(np.ceil(h_fine_m / dy)), int(np.ceil(h_fine_m / dx))
    wide = ndimage.maximum_filter(fp, size=(2 * ki + 1, 2 * kj + 1)) | seeds[0] | seeds[1]
    pools = [np.zeros(D.shape, bool), np.zeros(D.shape, bool)]

    def search():
        corr = wide | pools[0] | pools[1]
        Dc = np.where(corr, D, np.nan)
        lab, _ = ndimage.label(np.isfinite(Dc), structure=sq)
        a, b = ends
        if not lab[a] or lab[a] != lab[b]:                 # the corridor does not join the ends: the whole window
            Dc, corr = D, None
        d_ref, _ = _maximin(Dc, a, b, sq)
        return Dc, corr, d_ref, [bool(d_ref >= D[e] - END_TOL_M) for e in ends]

    Dc, corr, d_ref, lim = search()
    join_lim = list(lim)
    if any(lim):
        for s in (0, 1):
            if not lim[s]:
                continue
            pools[s] = _pool(grid, D, fp, seeds[s], sub.points[pieces[s][0]], sub.points[pieces[1 - s][0]], ends[s])
            if pools[s].any():
                pi, pj = np.nonzero(pools[s])
                k = int(np.argmax(D[pi, pj]))
                if D[pi[k], pj[k]] > D[ends[s]] + END_TOL_M:
                    ends[s] = (int(pi[k]), int(pj[k]))
                    src[s] = "pool"
        Dc, corr, d_ref, lim = search()
    a, b = ends
    ii, jj = _path(grid, Dc, a, b, d_ref)
    k = int(np.argmin(D[ii, jj]))
    sd = _saddle(grid, a, b, corr, near=(int(ii[k]), int(jj[k])))
    saddle_from = "the line's ends"
    if not np.isfinite(sd[1]) and 0 < k < len(ii) - 1:
        # the grid's own depth limits the saddle's search at an end (an end cell no deeper than the saddle, though
        # its box mean is): from the deepest grid cell of each leg of the line (before / after its shallowest point)
        raw = np.where(grid.elev[ii, jj] < 0, -grid.elev[ii, jj].astype(np.float64), -1.0)
        la_, lb_ = int(np.argmax(raw[:k])), k + 1 + int(np.argmax(raw[k + 1:]))
        sd2 = _saddle(grid, (int(ii[la_]), int(jj[la_])), (int(ii[lb_]), int(jj[lb_])), corr,
                      near=(int(ii[k]), int(jj[k])))
        if np.isfinite(sd2[1]):
            sd, saddle_from = sd2, "the deepest grid cell of each leg of the line"
    plon, plat = grid.lon[jj], grid.lat[ii]
    xyz = lonlat_to_xyz(plon, plat)
    km = float(R_EARTH_M / 1e3 * np.arccos(np.clip(np.sum(xyz[1:] * xyz[:-1], 1), -1, 1)).sum())
    narrow, nrec = narrows(grid, plon, plat, fp[ii, jj])
    pt = lambda c: [round(float(grid.lon[c[1]]), 4), round(float(grid.lat[c[0]]), 4)]          # noqa: E731
    rec = {"ends": [pt(a), pt(b)], "end_from": src, "join_deepest_box_m": join_m,
           "end_box_m": [float(D[a]), float(D[b])], "join_limited": join_lim, "end_limited": lim,
           "corridor": corr is not None,
           "pool_cells": [int(p.sum()) for p in pools],
           "box_half_width_m": 0.5 * h_fine_m, "cells": int(len(ii)), "km": km,
           "deepest_connection_box_m": float(d_ref),
           "shallowest_on_line": {"m": float(D[ii[k], jj[k]]), "at": [round(float(plon[k]), 4), round(float(plat[k]), 4)]},
           "grid_saddle": {"m": sd[0], "at": [round(sd[1], 4), round(sd[2], 4)], "from": saddle_from},
           "narrows": nrec}
    return Line(plon, plat, D[ii, jj], sd, rec, narrow)


def line_raise(sub: Mesh, depth: np.ndarray, ins: np.ndarray, grid: Grid, line: Line, mode: str) -> tuple:
    """B213 (module docstring): "cells" or "near", on the line's narrows (line.narrow; None: the whole line).
    Returns (depth, record)."""
    from scipy.spatial import cKDTree

    from meshom.bathymetry import node_spacing_m
    from meshom.mesh import signed_area
    from meshom.straits import locate

    d = np.asarray(depth, dtype=np.float64).copy()
    keep = line.narrow if line.narrow is not None else np.ones(len(line.lon), bool)
    lon, lat = line.lon[keep], line.lat[keep]
    pts = lonlat_to_xyz(lon, lat)
    if mode == "cells":
        el = locate(sub, pts)
        area = np.abs(signed_area(sub))
        T = sub.n_tris
        for e in np.unique(el[el >= 0]):
            nd = sub.tris[e] if e < T else sub.quads[e - T]
            if not ins[nd].any():
                continue
            k = el == e
            bm = grid.box_mean(lon[k], lat[k], np.full(k.sum(), 0.5 * np.sqrt(area[e])))
            bm = bm[np.isfinite(bm)]
            if len(bm):
                m = nd[ins[nd]]
                d[m] = np.maximum(d[m], float(bm.mean()))
    else:
        spacing, _, _ = node_spacing_m(sub)
        dist, j = cKDTree(pts).query(sub.points)
        arc = 2.0 * np.arcsin(np.minimum(dist / 2.0, 1.0)) * R_EARTH_M
        near = ins & (arc <= CHANNEL_NEAR * spacing)
        bm = grid.box_mean(lon[j[near]], lat[j[near]], 0.5 * spacing[near])
        idx = np.flatnonzero(near)
        ok = np.isfinite(bm)
        d[idx[ok]] = np.maximum(d[idx[ok]], bm[ok])
    up = d - depth
    return d, {"mode": mode, "nodes_raised": int((up > 1e-6).sum()), "raise_max_m": float(up.max()),
               "raise_mean_m": float(up[up > 1e-6].mean()) if (up > 1e-6).any() else 0.0}


def node_quantile(sub: Mesh, depth: np.ndarray, ins: np.ndarray, grid: Grid, q: float, floor_m: float) -> tuple:
    """B214: the insert's nodes take the q-quantile of the depths of the grid's ocean cells in their node box
    (node_depth's box: half-width half the node's mean edge length); nodes whose box holds no ocean cell keep
    `depth`; then the floor. Returns (depth, record)."""
    from meshom.bathymetry import node_spacing_m

    spacing, _, _ = node_spacing_m(sub)
    lon, lat = xyz_to_lonlat(sub.points)
    idx = np.flatnonzero(ins)
    ci, cj, hi, hj = grid.boxes(lon[idx], lat[idx], 0.5 * spacing[idx])
    d = np.asarray(depth, dtype=np.float64).copy()
    kept = 0
    for k, i in enumerate(idx):
        z = grid.elev[max(0, ci[k] - hi[k]):ci[k] + hi[k] + 1, max(0, cj[k] - hj[k]):cj[k] + hj[k] + 1]
        z = z[z < 0]
        if len(z):
            d[i] = max(float(np.quantile(-z.astype(np.float64), q)), floor_m)
        else:
            kept += 1
    return d, {"quantile": q, "nodes": int(len(idx)), "kept_without_ocean_cell": kept,
               "change_m": {"min": float((d - depth)[idx].min()), "max": float((d - depth)[idx].max())}}


def banks(sub: Mesh, ins: np.ndarray):
    """The graph of the nodes of the insert's cells (neighbours: nodes of one insert cell) and the two banks (the two
    largest components of the insert cells' coast edges): (g csr, bank A mask, bank B mask) or a reason (str)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from meshom.mesh import build_topology

    T = sub.n_tris
    cells = [c for c in (sub.tris, sub.quads) if len(c)]
    in_ins = [c[ins[c].any(1)] for c in cells]
    pairs = np.vstack([np.c_[c[:, i], c[:, j]] for c in in_ins for i in range(c.shape[1])
                       for j in range(i + 1, c.shape[1])])
    n = sub.n_nodes
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()
    g = ((g + g.T) > 0).tocsr()
    topo = build_topology(sub)
    be = topo.edges[topo.edge_elems[:, 1] < 0]
    elem_ins = np.r_[ins[sub.tris].any(1) if T else np.zeros(0, bool), ins[sub.quads].any(1)]
    ce = be[elem_ins[topo.edge_elems[topo.edge_elems[:, 1] < 0, 0]]]
    if not len(ce):
        return "the insert has no coast edges"
    cg = coo_matrix((np.ones(len(ce)), (ce[:, 0], ce[:, 1])), shape=(n, n))
    nc, lab = connected_components(cg, directed=False)
    on = np.zeros(n, bool)
    on[ce.ravel()] = True
    sizes = np.bincount(lab[on], minlength=nc)
    if (sizes > 0).sum() < 2:
        return "the insert's coast is one piece (no two banks)"
    a, b = np.argsort(sizes)[::-1][:2]
    return g, on & (lab == a), on & (lab == b)


def cap_star(sub: Mesh, depth: np.ndarray, chain: np.ndarray, cap: float) -> tuple:
    """every node of the chain and of every cell holding one capped at `cap`; returns (depth, capped mask, the mask of
    those nodes: the dam)."""
    T = sub.n_tris
    n = sub.n_nodes
    hit = np.zeros(n, bool)
    hit[chain] = True
    star = np.r_[hit[sub.tris].any(1) if T else np.zeros(0, bool), hit[sub.quads].any(1)]
    nb = np.zeros(n, bool)
    nb[sub.tris[star[:T]].ravel()] = True
    nb[sub.quads[star[T:]].ravel()] = True
    out = depth.copy()
    low = nb & (out > cap)
    out[low] = cap
    return out, low, nb


def grid_sill(sub: Mesh, depth: np.ndarray, ins: np.ndarray, line: Line, zbar: np.ndarray,
              sides: tuple | None = None, nearest: bool = False) -> tuple:
    """B215 (module docstring): the grid's saddle held at its level by a dam of the insert's nodes. sides: (graph,
    bank A mask, bank B mask) in place of `banks` (a group's route, meshom.insert.group.route_banks, B246). nearest
    (sill_level = "nearest", B249): L is the level surface nearest the saddle's grid depth (the level GEBCO's saddle
    falls on in FESOM3's levels), not the deepest one above it. Returns (depth, record; its `held`: (the dam's node
    ids, the cap), which depth_area keeps, B250)."""
    from scipy.sparse.csgraph import dijkstra

    from meshom.straits import locate

    sd, slon, slat = line.saddle
    if not np.isfinite(slon):
        return depth, {"skipped": "no saddle cell found on the grid"}
    p = lonlat_to_xyz(np.array([slon]), np.array([slat]))
    e = int(locate(sub, p)[0])
    T = sub.n_tris
    if e < 0:
        return depth, {"skipped": "the grid's saddle lies outside the mesh"}
    nd = sub.tris[e] if e < T else sub.quads[e - T]
    if not ins[nd].any():
        return depth, {"skipped": "the grid's saddle lies outside the insert", "saddle_m": sd}
    r = banks(sub, ins) if sides is None else sides
    if isinstance(r, str):
        return depth, {"skipped": r}
    g, src, dst = r
    via = int(nd[np.argmin(np.sum((sub.points[nd] - p) ** 2, 1))])
    w = g.astype(np.float64).multiply(np.maximum(depth, 1.0)[None, :]).tocsr()   # a step costs the depth entered
    dist, pred = dijkstra(w, directed=True, indices=via, return_predecessors=True)
    chain = [via]
    for bank in (src, dst):
        cand = np.flatnonzero(bank & np.isfinite(dist))
        if not len(cand):
            return depth, {"skipped": "the saddle's node reaches no bank"}
        t = int(cand[np.argmin(dist[cand])])
        q = [t]
        while q[-1] != via:
            q.append(int(pred[q[-1]]))
        chain += q[:-1]
    chain = np.array(chain)
    z = np.asarray(zbar, dtype=np.float64)
    k = int(np.searchsorted(z, sd, side="right")) - 1
    if nearest and k + 1 < len(z) and z[k + 1] - sd < sd - z[k]:       # B249: the nearest level surface
        k += 1
    L, Ln = float(z[max(k, 0)]), float(z[min(k + 1, len(z) - 1)])
    cap = 0.5 * (L + Ln) - SILL_MARGIN_M
    out, low, nb = cap_star(sub, depth, chain, cap)
    lon, lat = xyz_to_lonlat(sub.points[via][None])
    return out, {"held": (np.flatnonzero(nb), cap),
                 "rule": "nearest" if nearest else "grid", "saddle_m": sd,
                 "saddle_at": [round(slon, 4), round(slat, 4)], "level_m": L,
                 "next_level_m": Ln, "cap_m": cap, "via": [round(float(lon[0]), 4), round(float(lat[0]), 4)],
                 "chain_nodes": int(len(chain)), "capped": int(low.sum()), "capped_host_nodes": int((low & ~ins).sum()),
                 "cut_max_m": float((depth - out).max())}
