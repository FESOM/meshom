# SPDX-License-Identifier: GPL-3.0-or-later
"""FESOM3's rules for an ORCA mesh: how FESOM3 turns NEMO's ORCA025 into the quad mesh it runs on, rewritten for
meshom after FESOM3's NEMO-to-mesh conversion and its dead-end removal (FESOM3: github.com/FESOM/fesom3).
Decisions B87-B95.

Arrays are in NEMO's layout: rows x columns with the two wrap columns (the last two repeat the first two), T row 0 in
the south, the north fold row last (meshom.orca.generator.OrcaGrid has the same layout). `kb` is NEMO's bottom level
per T point (`mbathy`: water where > 0).

The rules, in their order:
1. **The southern closing row**: one F row added south of F row 0, its latitude F row 0's minus the T
   spacing of column 0 (T row 1 - T row 0), its longitude F row 0's. Node rows below are "extended" rows: extended
   row r = F row r - 1.
2. **The fold rows' F points**: on the two F rows next to the pivot T row (F rows j_p - 1 and j_p), the
   five F points centred on the geographic North Pole (two runs of columns, one on each side of the fold) are
   replaced by the straight line between the run's ends in the polar plane (x, y) = cos(lat) (cos lon, sin lon),
   latitude back by arccos(sqrt(x^2 + y^2)) (north). ORCA025: columns 379-383 and 1058-1062; by place
   (`pole_columns`): the column on F row j_p - 1 nearest the pole in each half, +-2.
3. **One-cell land removed** (`one_cell_land`): a land T cell whose four edge neighbours are water takes
   its east neighbour's level; one sequential sweep, columns 1 .. NI - 2 outer, rows 1 .. nj - 2 inner, the level
   changed in place (a cell made water counts as water for the cells after it); the wrap columns are not refreshed
   (column NI - 2, the copy of column 0, sees the old copy of column 1 as its east neighbour).
4. **The hand edits** (`ORCA025_EDITS`): 14 index edits that copy a neighbour's level into land cells,
   in order (an edit may read a cell an earlier one changed). Their places are in `ORCA025_EDITS`; `carry_edits`
   carries them to other grids by place.
5. **Node flags**: 1 at the four corners of every land T cell (rows 0 .. nj - 1, columns 1 .. NI - 1).
6. **Node levels**: the deepest of the four T cells around the F point (max of `kb`); the closing row
   takes the row above's, the fold row the row below's, the last column the second's.
7. **The numbering, the fold's node merge and the cyclic merge**: node id = extended row + rows x
   column (column-major); T pivot: the top extended row's columns 0 .. NI/2 - 1 take the ids of the row below,
   mirrored (column i -> NI - 1 - i, column 0 -> 1); then column NI - 2 takes column 0's ids (in this order: the fold
   copies column NI - 2's own id before the cyclic merge replaces it).
8. **The quads**: every water T cell of rows 0 .. nj - 3, columns 1 .. NI - 2 (column-major order), then
   the pivot row nj - 2 for columns 1 .. NI/2 - 1 only; corners SW, NW, NE, SE (clockwise seen from outside).
9. **Unused nodes dropped**: nodes kept in id order, renumbered.
10. **The levels**: layer bottoms from NEMO's T depths `nav_lev`: z_0 = 0, z_1 = 2 lev_1,
    z_k = 2 (lev_k - z_{k-1}) + z_{k-1}, in float32 (`levels_from_t`).
11. **The files**: nod2d.out `id lon lat flag`, elem2d.out 4 columns, depth.out = Nz + 1, the levels,
    the node depths z_k, the node layer counts k (fesom_ini `use_depthfile="nemo"`: node levels = k + 1).
12. **Dead ends removed** (`remove_deadends`): every element with fewer than
    two edge neighbours, all at once per pass, until none; the nodes of a removed element that stay get flag 1;
    nodes and elements keep their order; layer counts kept.
12b. **Only the ocean-connected water** (ours, C48, B170; `ocean_only`): every water component but the largest
    (most elements) is removed, as rule 12 removes an element (nodes and elements keep their order, the nodes of a
    removed element that stay get flag 1); passes until one component is left, rule 12 again after each. Elements
    join across a shared edge; with cavities only where their FESOM3 vertex columns (`element_columns`) share a layer
    (so a removal can leave a kept element with one edge neighbour: rule 12 takes it). FESOM3's rules have no
    such step (its ORCA025 is one component); B111's cavity cells made land leave the open water enclosed by ice
    shelves as closed seas (eORCA36: 13 Antarctic pockets), which have no initial data and whose level would rise
    without bound under runoff. `info["closed_water"]` names each removed component (`water_report`).
13. **Node levels closed** (ours, B119; `close_node_levels`): after rule 12 every node's layer count becomes the
    deepest of its elements' (an element's = the fewest of its nodes', as in fesom_ini's NEMO mode). FESOM3's model
    stops at setup unless every node's deepest layer has a wet adjacent element (FESOM3's bottom invariant), and
    fesom_ini keeps depth.out's node counts as given; rule 6 takes a node's level before rule 12, so a node whose
    deepest cell was a removed dead end keeps a level no element reaches. Only such nodes change (shallower); no
    element's level changes.

**Ice-shelf cavities (option, B118; default off: cavity cells are land as B111):** `convert(..., kt=...)` takes NEMO's
first wet level per T cell (`top_level`) and gives each node its top layer `ntop` (`node_tops`: the shallowest first
wet layer of the node's wet cells, the dual of rule 6, so every wet cell keeps its water; FESOM3's vertex column:
element top = max over its vertices), then FESOM2's fewest-layers fix at the vertices (`clamp_tops`: a top is raised
until every element keeps 3 layers); `write` adds `cavity_depth@node.out` (FESOM2's and fesom_ini's cavity file: one
value per node, metres, negative under ice, 0 in the open ocean; `cavity_depth`: the top of the node's first wet
layer, so fesom_ini's rule gives back `ntop` with which_depth_n2e='max').

FESOM3 states only a T pivot (ORCA025). An F pivot (ORCA1, ORCA05) is ours (`numbering`): the pole line is F row
j_p, F(j_p, i) = F(j_p, K - i) merged, every T row up to j_p a cell row (B90).
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numba import njit

NMQ = "github.com/FESOM/fesom3"


@dataclass(frozen=True)
class Edit:
    """One of FESOM3's hand edits: T cells rows[0]:rows[1], cols[0]:cols[1] (0-based, end exclusive) take the level of
    the cell (drow, dcol) away. `cells`: per target cell (lon, lat of the target T point, lon, lat of its source) on
    FESOM3's ORCA025 (mesh_hgr.nc glamt, gphit), the place by which `carry_edits` carries the edit to other grids.
    `line`: the edit's label in the records (116 .. 129, in FESOM3's order)."""
    strait: str
    rows: tuple
    cols: tuple
    src: tuple
    line: int
    cells: tuple = ()


# FESOM3's 14 hand edits in its order; places from FESOM3's mesh_hgr.nc (glamt, gphit)
ORCA025_EDITS = (
    Edit("Gibraltar", (664, 665), (1125, 1128), (-1, 0), 116,
         ((-5.888, 38.205, -5.893, 38.011), (-5.637, 38.201, -5.642, 38.007), (-5.386, 38.197, -5.391, 38.003))),
    Edit("Bosphorus", (687, 690), (1261, 1262), (0, 1), 117,
         ((28.412, 41.058, 28.661, 41.045), (28.426, 41.208, 28.675, 41.195), (28.44, 41.358, 28.689, 41.344))),
    Edit("Dardanelles", (684, 685), (1252, 1257), (-1, 0), 118,
         ((26.129, 40.711, 26.116, 40.556), (26.378, 40.699, 26.365, 40.544), (26.628, 40.687, 26.615, 40.532),
          (26.877, 40.675, 26.864, 40.521), (27.126, 40.663, 27.114, 40.509))),
    Edit("Bab-el-Mandeb", (550, 551), (1320, 1321), (0, 1), 119, ((42.75, 12.89, 43.0, 12.89),)),
    Edit("Bab-el-Mandeb", (549, 550), (1321, 1322), (0, 1), 120, ((43.0, 12.646, 43.25, 12.646),)),
    Edit("Nares Strait", (960, 961), (1000, 1005), (1, 0), 121,
         ((-64.809, 80.906, -65.3, 80.99), (-64.312, 80.978, -64.803, 81.062), (-63.806, 81.048, -64.298, 81.133),
          (-63.292, 81.119, -63.785, 81.204), (-62.771, 81.188, -63.263, 81.274))),
    Edit("Nares Strait", (962, 963), (1009, 1010), (0, -1), 122, ((-61.02, 81.706, -61.587, 81.638),)),
    Edit("Fury and Hecla Strait", (924, 925), (886, 887), (0, 1), 123, ((-84.822, 69.831, -84.641, 69.901),)),
    Edit("Fury and Hecla Strait", (923, 924), (887, 888), (0, 1), 124, ((-84.435, 69.838, -84.251, 69.908),)),
    Edit("Hell Gate", (975, 976), (939, 940), (0, -1), 125, ((-90.106, 76.681, -90.271, 76.592),)),
    Edit("Hell Gate", (972, 975), (938, 939), (0, -1), 126,
         ((-89.136, 76.474, -89.308, 76.387), (-89.513, 76.515, -89.682, 76.427), (-89.891, 76.554, -90.057, 76.466))),
    Edit("Hell Gate", (974, 975), (939, 940), (0, -1), 127, ((-89.723, 76.642, -89.891, 76.554),)),
    Edit("Nevelskoy Strait", (751, 752), (284, 285), (0, -1), 128, ((141.85, 52.659, 141.59, 52.634),)),
    Edit("Nevelskoy Strait", (745, 749), (283, 284), (0, -1), 129,
         ((141.815, 51.732, 141.556, 51.709), (141.779, 51.883, 141.52, 51.86), (141.742, 52.034, 141.483, 52.01),
          (141.705, 52.185, 141.446, 52.16))),
)
ORCA025_POLE_COLS = ((379, 384), (1058, 1063))        # FESOM3's columns on ORCA025
POLE_HALF = 2                                         # FESOM3's five points: the pole's column +-2


# ---- the rules on the T levels ------------------------------------------------------------------------------------

@njit(cache=True)
def _one_cell_land(b):
    nj, ni = b.shape
    n = 0
    for j in range(1, ni - 1):
        for i in range(1, nj - 1):
            if b[i, j] <= 0 and b[i, j + 1] > 0 and b[i, j - 1] > 0 and b[i + 1, j] > 0 and b[i - 1, j] > 0:
                b[i, j] = b[i, j + 1]
                n += 1
    return n


def one_cell_land(kb: np.ndarray) -> tuple[np.ndarray, int]:
    """Rule 3 on a copy of kb (nj, NI): (kb, cells made water)."""
    b = np.array(kb, copy=True)
    n = _one_cell_land(b)
    return b, int(n)


def apply_edits(kb: np.ndarray, edits) -> tuple[np.ndarray, list]:
    """Rule 4 on a copy of kb, in order: (kb, per edit {strait, line, cells, opened})."""
    b = np.array(kb, copy=True)
    rec = []
    for e in edits:
        r, c = slice(*e.rows), slice(*e.cols)
        before = b[r, c].copy()
        b[r, c] = b[e.rows[0] + e.src[0]:e.rows[1] + e.src[0], e.cols[0] + e.src[1]:e.cols[1] + e.src[1]]
        rec.append({"strait": e.strait, "line": e.line, "cells": int(before.size),
                    "opened": int(((before <= 0) & (b[r, c] > 0)).sum())})
    return b, rec


def levels_from_t(nav_lev: np.ndarray) -> np.ndarray:
    """Rule 10: the layer bottoms (Nz + 1, metres, 0 first) from NEMO's T depths, float32 arithmetic."""
    f32 = np.float32
    lev = np.asarray(nav_lev, dtype=f32)
    z = np.zeros(len(lev) + 1)
    z[1] = f32(2) * lev[0]
    for k in range(2, len(lev) + 1):
        zp = f32(z[k - 1])
        z[k] = f32(2) * (lev[k - 1] - zp) + zp
    return z


def level_of_depth(depth_m: np.ndarray, z: np.ndarray, k_min: int) -> np.ndarray:
    """NEMO's bottom level of a T cell of depth `depth_m` (m, positive down; <= 0: land, 0) on the layer bottoms z
    (Nz + 1, 0 first): the layer that holds the bottom (the first k with z_k >= depth, a partial step), at least
    k_min, at most Nz (B92)."""
    d = np.asarray(depth_m, dtype=np.float64)
    k = np.searchsorted(z, d, side="left")
    k = np.clip(k, k_min, len(z) - 1)
    return np.where(np.isfinite(d) & (d > 0), k, 0).astype(np.int16)


def top_level_of_draft(draft_m: np.ndarray, z: np.ndarray) -> np.ndarray:
    """The first wet layer (1-based) of a column whose ice base is at `draft_m` (m, positive down; 0: open ocean) on
    the layer bottoms z (Nz + 1, 0 first): the layer that holds the ice base (the last k with z_{k-1} <= draft, a
    partial step at the top, as `level_of_depth` at the bottom; B118). Open ocean: 1."""
    d = np.asarray(draft_m, dtype=np.float64)
    k = np.searchsorted(z, d, side="right")
    return np.clip(k, 1, len(z) - 1).astype(np.int16)


def one_cell_land_tops(kb: np.ndarray, kb1: np.ndarray, kt: np.ndarray) -> np.ndarray:
    """Rule 3's cells made water (kb <= 0 < kb1) take the first wet level of the east neighbour they took their
    level from (`one_cell_land`: the east neighbour is not yet swept when the cell is); a copy of kt (B118)."""
    t = np.array(kt, copy=True)
    r, c = np.nonzero((kb <= 0) & (kb1 > 0))
    t[r, c] = kt[r, c + 1]
    return t


def node_tops(elem: np.ndarray, elem_top: np.ndarray, n: int) -> np.ndarray:
    """B118: the top layer (1-based first wet layer) of every node, the shallowest of its elements' (cells') tops:
    the dual of rule 6 (a node's bottom is the deepest of its cells), so the node column holds every adjacent cell's
    water and FESOM3's element top (max over its vertices) is no deeper than the
    cell's own top."""
    t = np.full(n, np.iinfo(np.int16).max, np.int16)
    np.minimum.at(t, elem.ravel(), np.repeat(np.asarray(elem_top, np.int16), elem.shape[1]))
    return t


MIN_LAYERS_CAVITY = 3   # B118: FESOM2's fewest layers under a cavity (as fesom_ini checks them)


def clamp_tops(elem: np.ndarray, ntop: np.ndarray, nlev: np.ndarray,
               min_layers: int = MIN_LAYERS_CAVITY) -> tuple[np.ndarray, int]:
    """FESOM2's fix for too thin a water column under the ice (the cavity top is raised, the bottom kept) at the
    vertices (B118): every node's top is at most (its elements' shallowest bottom layer) -
    (min_layers - 1), so every element (top = max over its nodes, bottom = min over its nodes' `nlev`) keeps
    min_layers layers. One pass suffices (tops only rise). Returns (ntop, nodes changed)."""
    eb = nlev[elem].min(axis=1) - (min_layers - 1)
    cap = np.full(len(ntop), np.iinfo(np.int64).max, np.int64)
    np.minimum.at(cap, elem.ravel(), np.repeat(eb, elem.shape[1]))
    new = np.maximum(np.minimum(ntop, cap), 1).astype(ntop.dtype)
    return new, int((new != ntop).sum())


def cavity_depth(ntop: np.ndarray, z: np.ndarray) -> np.ndarray:
    """B118: the value of cavity_depth@node.out per node: minus the top of its first wet layer (m), 0 where the
    node's column starts at the surface. fesom_ini takes the first layer whose mid-depth
    lies below this value: the node's own top layer again."""
    return -np.asarray(z, float)[np.asarray(ntop, np.int64) - 1] + 0.0


# ---- the F points -------------------------------------------------------------------------------------------------

def pole_columns(lon_f: np.ndarray, lat_f: np.ndarray, row: int) -> tuple:
    """Rule 2 by place: on F row `row` (j_p - 1) the column nearest the geographic North Pole in each half of the
    unique columns, +-POLE_HALF: ((c0, c1), (c2, c3)), end exclusive."""
    nu = lat_f.shape[1] - 2
    la = np.asarray(lat_f[row, :nu], dtype=np.float64)
    a = int(np.argmax(la[:nu // 2]))
    b = nu // 2 + int(np.argmax(la[nu // 2:]))
    return tuple((c - POLE_HALF, c + POLE_HALF + 1) for c in (a, b))


def fold_fix(lon_f: np.ndarray, lat_f: np.ndarray, rows: tuple, cols: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Rule 2 on copies of the F arrays (in their dtype: float32 as NEMO's file and FESOM3's script):
    on each row of `rows`, each column run (c0, c1) is replaced by the straight line between its end points in the
    polar plane."""
    lo, la = np.array(lon_f, copy=True), np.array(lat_f, copy=True)
    t = lo.dtype.type
    pi = t(np.pi)
    for r in rows:
        x = np.cos(la[r] * pi / t(180)) * np.cos(lo[r] * pi / t(180))
        y = np.cos(la[r] * pi / t(180)) * np.sin(lo[r] * pi / t(180))
        for c0, c1 in cols:
            k = np.arange(c1 - c0, dtype=lo.dtype)
            n = t(c1 - c0 - 1)
            x[c0:c1] = x[c0] + k * (x[c1 - 1] - x[c0]) / n
            y[c0:c1] = y[c0] + k * (y[c1 - 1] - y[c0]) / n
        for c0, c1 in cols:
            lo[r, c0:c1] = np.arctan2(y[c0:c1], x[c0:c1]) * t(180) / pi
            la[r, c0:c1] = np.arccos(np.sqrt(y[c0:c1] ** 2 + x[c0:c1] ** 2)) * t(180) / pi
    return lo, la


def nemo_halo(a: np.ndarray, point: str) -> np.ndarray:
    """NEMO 4.2's halo-free array (Nj0, Ni0) of a T-pivot ORCA (jperio 4; domain_cfg and coordinates of eORCA36) in
    NEMO's layout with halos (Nj0 + 1, Ni0 + 2), the layout of these rules (B111): column 0 = unique column Ni0 - 1,
    columns 1 .. Ni0 the unique ones, column Ni0 + 1 = unique column 0; the halo row on top by NEMO's north fold
    (lbcnfd, 0-based on the halo layout): T point (nj - 1, i) = (nj - 3, NI - i), F point (nj - 1, i) = (nj - 4,
    NI - 1 - i) for i = 1 .. NI - 2, then the wrap columns of that row."""
    nj0, ni0 = a.shape
    out = np.empty((nj0 + 1, ni0 + 2), a.dtype)
    out[:nj0, 1:ni0 + 1] = a
    out[:nj0, 0] = a[:, ni0 - 1]
    out[:nj0, ni0 + 1] = a[:, 0]
    NI, nj = ni0 + 2, nj0 + 1
    i = np.arange(NI)
    if point == "t":
        out[nj - 1] = out[nj - 3, (NI - i) % NI]
    else:
        out[nj - 1] = out[nj - 4, NI - 1 - i]
    out[nj - 1, 0], out[nj - 1, NI - 1] = out[nj - 1, NI - 2], out[nj - 1, 1]      # then the cyclic columns
    return out


def south_row(lon_f: np.ndarray, lat_f: np.ndarray, lat_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rule 1: the F arrays with the closing row added in the south (extended rows)."""
    dy = lat_t[1, 0] - lat_t[0, 0]
    return np.vstack([lon_f[:1], lon_f]), np.vstack([lat_f[:1] - dy, lat_f])


# ---- the mesh -----------------------------------------------------------------------------------------------------

def node_flags(kb: np.ndarray) -> np.ndarray:
    """Rule 5 on the extended F rows (nj + 1, NI)."""
    nj, ni = kb.shape
    ind = np.zeros((nj + 1, ni), np.int8)
    land = kb[:, 1:] <= 0
    for di in (0, 1):
        for dj in (0, 1):
            ind[di:nj + di, dj:ni - 1 + dj][land] = 1
    return ind


def node_levels(kb: np.ndarray) -> np.ndarray:
    """Rule 6 on the extended F rows: the layer count per node."""
    nj, ni = kb.shape
    d = np.zeros((nj + 1, ni), np.int64)
    d[1:nj, :ni - 1] = np.maximum.reduce([kb[:-1, :-1], kb[1:, :-1], kb[:-1, 1:], kb[1:, 1:]])
    d[0] = d[1]
    d[-1] = d[-2]
    d[:, ni - 1] = d[:, 1]
    return d


def numbering(nj: int, ni: int, pivot: str, k_mirror: int | None = None) -> np.ndarray:
    """Rule 7: node ids (0-based, column-major) of the extended F rows (nj + 1, NI), the fold and the
    cyclic merge applied. T pivot as FESOM3; F pivot (ours, B90): the pole line is extended row nj - 1 (F row
    nj - 2) and F(j, i) = F(j, K - i) for the columns past the mirror, K = k_mirror (meshom.orca.mesh.mirror_k)."""
    ne = nj + 1
    num = (np.arange(ne)[:, None] + ne * np.arange(ni)[None, :]).astype(np.int64)
    h = (ni - 2) // 2
    if pivot == "T":
        num[ne - 2, 1:h + 1] = num[ne - 3, ni - 2:h:-1]        # the fold: columns 1 .. h take the row below's, mirrored
        num[ne - 2, 0] = num[ne - 3, 1]                        # column 0 takes the row below's column 1
    else:
        r = ne - 2
        i = np.arange(ni - 2)
        m = (k_mirror - i) % (ni - 2)
        src = i > m                                            # the later column of each pair takes the earlier's id
        num[r, i[src]] = num[r, m[src]]
    num[:, ni - 2] = num[:, 0]                                 # cyclicity
    return num


def quads(kb: np.ndarray, num: np.ndarray, pivot: str) -> tuple[np.ndarray, np.ndarray]:
    """Rule 8: ((e, 4) node ids SW, NW, NE, SE of the water T cells in FESOM3's order, (e, 2) their T
    (row, column))."""
    nj, ni = kb.shape
    last = nj - 3 if pivot == "T" else nj - 2                  # T pivot: the pivot row is added half below
    J, I = np.meshgrid(np.arange(1, ni - 1), np.arange(0, last + 1), indexing="ij")   # column-major order
    J, I = J.ravel(), I.ravel()
    sel = kb[I, J] > 0
    J, I = J[sel], I[sel]
    q = [np.stack([num[I, J - 1], num[I + 1, J - 1], num[I + 1, J], num[I, J]], axis=1)]
    if pivot == "T":
        i = nj - 2
        j = np.arange(1, (ni - 2) // 2 + 1)
        j = j[kb[i, j] > 0]
        q.append(np.stack([num[i, j - 1], num[i + 1, j - 1], num[i + 1, j], num[i, j]], axis=1))
    cell = [np.stack([I, J], axis=1)] + ([np.stack([np.full(len(j), nj - 2), j], axis=1)] if pivot == "T" else [])
    return np.vstack(q), np.vstack(cell)


@dataclass
class Fesom3Mesh:
    """A FESOM3 ORCA mesh as FESOM3's files hold it: lon, lat (degrees, the dtype of the F points: float32 for NEMO's
    file), elem (e, 4) 0-based SW-NW-NE-SE, flag (n,), nlev (n,) the layer counts, z (Nz + 1) the layer bottoms;
    cell (e, 2) the T (row, column) of each element; node_rc (n, 2) the extended F (row, column) of each node; ntop
    (n,) the first wet layer of each node with ice-shelf cavities (B118), None without."""
    lon: np.ndarray
    lat: np.ndarray
    elem: np.ndarray
    flag: np.ndarray
    nlev: np.ndarray
    z: np.ndarray
    cell: np.ndarray
    node_rc: np.ndarray
    info: dict = field(default_factory=dict)
    ntop: np.ndarray | None = None      # B118: the node's first wet layer (1 = the surface); None: no cavities

    @property
    def depth(self) -> np.ndarray:
        return self.z[self.nlev]


def convert(lon_f: np.ndarray, lat_f: np.ndarray, lat_t: np.ndarray, kb: np.ndarray, z: np.ndarray,
            pivot: str = "T", edits=ORCA025_EDITS, pole_cols=ORCA025_POLE_COLS, k_mirror: int | None = None,
            deadends: bool = True, one_cell: bool = True, kt: np.ndarray | None = None,
            ocean_only: bool = True) -> Fesom3Mesh:
    """FESOM3's conversion (rules 1-13) of the F points (nj, NI) and the bottom levels kb (nj, NI) of an ORCA grid in
    NEMO's layout. edits: `ORCA025_EDITS` (by index), a list from `carry_edits`, or (); pole_cols: rule 2's column
    runs (None: none); z: the layer bottoms (levels_from_t); one_cell False: kb has rule 3 already (carry_edits).
    kt (B118, default None = no cavities): NEMO's first wet level per T cell (nj, NI) of the water in kb (cavity
    cells are water in kb then; the edits copy kt with kb, rule 3 by `one_cell_land_tops`, a water cell without a
    top gets 1); the mesh then has `ntop` (`node_tops`, `clamp_tops`). ocean_only False: rule 12b skipped."""
    nj, ni = kb.shape
    info = {"pivot": pivot, "rows": int(nj), "cols": int(ni)}
    lon2, lat2 = south_row(lon_f, lat_f, lat_t)
    if pole_cols:
        lon2, lat2 = fold_fix(lon2, lat2, (nj - 2, nj - 1), pole_cols)    # F rows j_p - 1, j_p (extended rows)
        info["pole_cols"] = [list(c) for c in pole_cols]
    b, n1 = one_cell_land(kb) if one_cell else (np.array(kb, copy=True), 0)
    info["one_cell_land"] = n1
    if kt is not None:
        kt = one_cell_land_tops(kb, b, kt) if one_cell else kt
        kt, _ = apply_edits(kt, edits)
    b, info["edits"] = apply_edits(b, edits)
    flag = node_flags(b)
    lev = node_levels(b)
    num = numbering(nj, ni, pivot, k_mirror)
    q, cell = quads(b, num, pivot)
    used = np.zeros((nj + 1) * ni, bool)
    used[q.ravel()] = True
    ids = np.flatnonzero(used)                                   # column-major linear ids in order
    new = np.full(len(used), -1, np.int64)
    new[ids] = np.arange(len(ids))
    r, c = ids % (nj + 1), ids // (nj + 1)
    m = Fesom3Mesh(lon2[r, c], lat2[r, c], new[q], flag[r, c].astype(np.int64), lev[r, c], np.asarray(z, float),
                   cell, np.stack([r, c], axis=1), info)
    info.update(nodes_raw=int(len(ids)), quads_raw=int(len(q)), water_cells=int((b[:, 1:-1] > 0).sum()))
    if deadends:
        m = remove_deadends(m)
    if ocean_only:
        m = keep_ocean(m, kt, deadends)
    m.nlev, info["node_levels_closed"] = close_node_levels(m.elem, m.nlev)
    if kt is not None:
        m.ntop, info["tops_clamped"] = _tops(m, kt)
    return m


def _tops(m: Fesom3Mesh, kt: np.ndarray) -> tuple[np.ndarray, int]:
    """B118: (the node tops, nodes clamped) of the mesh's elements from NEMO's first wet level of their cells."""
    et = np.maximum(kt[m.cell[:, 0], m.cell[:, 1]], 1)
    return clamp_tops(m.elem, node_tops(m.elem, et, len(m.lon)), m.nlev)


def element_columns(elem: np.ndarray, nlev: np.ndarray, ntop: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """FESOM3's vertex column of every element: (top, bottom) layer, top =
    the deepest of its nodes' first wet layers (1 without cavities), bottom = the fewest of its nodes' layer counts."""
    eb = nlev[elem].min(axis=1)
    return (np.ones_like(eb) if ntop is None else ntop[elem].max(axis=1)), eb


def water_components(elem: np.ndarray, n: int, top: np.ndarray | None = None,
                     bottom: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The water components of the elements (e, k): (label per element, elements per component), component 0 the
    largest (most elements). Two elements join across a shared edge; with layer columns (top, bottom per element)
    only where the columns share a layer (B170: water passes a face only at a layer both sides hold)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from meshom.orca.straits import cell_links

    a, b = cell_links(elem, n)
    if top is not None:
        ok = np.maximum(top[a], top[b]) <= np.minimum(bottom[a], bottom[b])
        a, b = a[ok], b[ok]
    e = len(elem)
    _, lab = connected_components(coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(e, e)), directed=False)
    size = np.bincount(lab)
    order = np.argsort(-size, kind="stable")
    rank = np.empty(len(size), np.int64)
    rank[order] = np.arange(len(size))
    return rank[lab], size[order]


def water_report(m: Fesom3Mesh, lab: np.ndarray, size: np.ndarray) -> list[dict]:
    """Per component but the largest (labels of `water_components`): its elements, nodes, centre [lon, lat], box
    [west, east, south, north] (degrees, west > east across 180) and the range of its nodes' layer counts."""
    from meshom.geo import lonlat_to_xyz, xyz_to_lonlat
    from meshom.straits import _lon_box

    out = []
    for k in range(1, len(size)):
        nd = np.unique(m.elem[lab == k])
        lon, lat = m.lon[nd].astype(np.float64), m.lat[nd].astype(np.float64)
        c = lonlat_to_xyz(lon, lat).mean(axis=0)
        clon, clat = xyz_to_lonlat(c[None] / np.linalg.norm(c))
        w, e = _lon_box(lon)
        out.append({"n_elems": int(size[k]), "n_nodes": int(len(nd)),
                    "at": [round(float(clon[0]), 3), round(float(clat[0]), 3)],
                    "box": [round(w, 3), round(e, 3), round(float(lat.min()), 3), round(float(lat.max()), 3)],
                    "layers": [int(m.nlev[nd].min()), int(m.nlev[nd].max())]})
    return out


def keep_ocean(m: Fesom3Mesh, kt: np.ndarray | None = None, deadends: bool = False) -> Fesom3Mesh:
    """Rule 12b (C48, B170): passes until one water component is left, each removing every component but the
    largest (with kt, the cavities: joined only at a common layer of the vertex columns, the tops as convert gives
    them). deadends: rule 12 again after each pass (with cavities a removed piece can touch kept elements by edges
    without a common layer and leave them one edge neighbour); its passes add to `info["deadends"]`.
    `info["closed_water"]`: the removed components (`water_report`), elements and nodes removed."""
    removed, n_el, n_nd = [], 0, 0
    while True:
        top = bot = None
        if kt is not None:
            top, bot = element_columns(m.elem, m.nlev, _tops(m, kt)[0])
        lab, size = water_components(m.elem, len(m.lon), top, bot)
        if len(size) == 1:
            break
        removed += water_report(m, lab, size)
        n0 = len(m.lon)
        n_el += int(size[1:].sum())
        m = _drop(m, lab > 0)
        n_nd += n0 - len(m.lon)
        if deadends:
            d0 = m.info["deadends"]
            m = remove_deadends(m)
            d = m.info["deadends"]
            m.info["deadends"] = {"passes": d0["passes"] + d["passes"], "removed": d0["removed"] + d["removed"]}
    m.info["closed_water"] = {"elems": n_el, "nodes": n_nd, "removed": removed}
    return m


def neighbour_count(elem: np.ndarray, n: int) -> np.ndarray:
    """Edge neighbours per element (sides 1-2, 2-3, 3-4, 4-1 shared by two elements)."""
    a, b = elem, np.roll(elem, -1, axis=1)
    key = np.minimum(a, b) * np.int64(n) + np.maximum(a, b)
    real = a != b
    _, inv, cnt = np.unique(key[real], return_inverse=True, return_counts=True)
    shared = np.zeros(a.shape, bool)
    shared[real] = cnt[inv.ravel()] >= 2
    return shared.sum(axis=1)


def remove_deadends(m: Fesom3Mesh) -> Fesom3Mesh:
    """Rule 12: passes until no element has fewer than two edge neighbours."""
    passes, gone = [], 0
    while True:
        dead = neighbour_count(m.elem, len(m.lon)) < 2
        if not dead.any():
            break
        passes.append(int(dead.sum()))
        gone += int(dead.sum())
        m = _drop(m, dead)
    m.info["deadends"] = {"passes": passes, "removed": gone}
    return m


def _drop(m: Fesom3Mesh, dead: np.ndarray) -> Fesom3Mesh:
    """The mesh without the elements `dead` (as FESOM3 removes them): the nodes of a removed element that stay get flag
    1, unused nodes go; nodes and elements keep their order."""
    flag = m.flag.copy()
    flag[m.elem[dead].ravel()] = 1
    keep = ~dead
    el = m.elem[keep]
    used = np.zeros(len(m.lon), bool)
    used[el.ravel()] = True
    new = np.full(len(m.lon), -1, np.int64)
    new[used] = np.arange(used.sum())
    return Fesom3Mesh(m.lon[used], m.lat[used], new[el], flag[used], m.nlev[used], m.z, m.cell[keep],
                      m.node_rc[used], m.info, None if m.ntop is None else m.ntop[used])


def close_node_levels(elem: np.ndarray, nlev: np.ndarray) -> tuple[np.ndarray, int]:
    """Rule 13 (B119): every node's layer count = the deepest of its elements', an element's = the fewest of its
    nodes' (fesom_ini's NEMO mode): L'(n) = max over e containing n of min over m in e of L(m).
    L' <= L, and no element's level changes (each element holds each of its nodes), so after one pass FESOM3's bottom
    invariant holds. Every node lies in an element (rules 9, 12). Returns (L', nodes
    changed)."""
    eb = nlev[elem].min(axis=1)
    new = np.zeros(len(nlev), nlev.dtype)
    np.maximum.at(new, elem.ravel(), np.repeat(eb, elem.shape[1]))
    return new, int((new != nlev).sum())


# ---- files --------------------------------------------------------------------------------------------------------

WRITE_CHUNK = 1 << 22  # lines formatted at a time (eORCA36: ~1e8 nodes; the files are the same for any chunk)


def water_check(m: Fesom3Mesh) -> dict:
    """C52: a global FESOM3 mesh is one water component (`water_components`; with cavities, m.ntop, joined only at a
    common layer of the vertex columns). Returns {"n_components": 1, "by_layer"}; more components: ValueError naming
    each extra one (elements, nodes, centre, box, layer counts: `water_report`)."""
    top, bot = element_columns(m.elem, m.nlev, m.ntop) if m.ntop is not None else (None, None)
    lab, size = water_components(m.elem, len(m.lon), top, bot)
    if len(size) > 1:
        raise ValueError(f"{len(size)} water components (a global mesh holds only the ocean-connected water, C48): "
                         f"main {size[0]} elements; " + "; ".join(
                             f"#{k} {r['n_elems']} elements, {r['n_nodes']} nodes at {r['at']} box {r['box']} "
                             f"layers {r['layers']}" for k, r in enumerate(water_report(m, lab, size), 1)))
    return {"n_components": 1, "by_layer": m.ntop is not None}


def write(m: Fesom3Mesh, path: Path) -> None:
    """Rule 11: nod2d.out, elem2d.out, depth.out in FESOM3's formats (byte for byte); with cavities
    (m.ntop, B118) also cavity_depth@node.out (`write_cavity`). A mesh of more than one water component is refused
    before any file is written (`water_check`, C52); the check's record goes to m.info["water"]."""
    m.info["water"] = water_check(m)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    n = len(m.lon)
    with open(path / "nod2d.out", "w") as f:
        f.write("%9i \n" % n)
        for s in range(0, n, WRITE_CHUNK):
            e = min(n, s + WRITE_CHUNK)
            f.write("".join("%9i %8.4f %8.4f %8i\n" % t for t in zip(range(s + 1, e + 1), m.lon[s:e].tolist(),
                                                                      m.lat[s:e].tolist(), m.flag[s:e].tolist())))
    with open(path / "elem2d.out", "w") as f:
        f.write("%9i \n" % len(m.elem))
        for s in range(0, len(m.elem), WRITE_CHUNK):
            f.write("".join("%9i %9i %9i %9i\n" % tuple(q) for q in (m.elem[s:s + WRITE_CHUNK] + 1).tolist()))
    z = m.z
    with open(path / "depth.out", "w") as f:
        f.write("%9i \n" % len(z))
        f.write("".join("%8.4f\n" % v for v in z.tolist()))
        for s in range(0, n, WRITE_CHUNK):
            f.write("".join("%8.4f\n" % v for v in z[m.nlev[s:s + WRITE_CHUNK]].tolist()))
        for s in range(0, n, WRITE_CHUNK):
            f.write("".join("%9i \n" % v for v in m.nlev[s:s + WRITE_CHUNK].tolist()))
    if m.ntop is not None:
        write_cavity(path, cavity_depth(m.ntop, z))


def write_cavity(path: Path, cav: np.ndarray) -> None:
    """B118: cavity_depth@node.out (as FESOM2's test mesh pi_cavity): one value per node, no
    header, metres, negative under the ice, 0 in the open ocean."""
    with open(Path(path) / "cavity_depth@node.out", "w") as f:
        for s in range(0, len(cav), WRITE_CHUNK):
            f.write("".join("%10.4f\n" % v for v in cav[s:s + WRITE_CHUNK].tolist()))


# ---- every preset: NEMO's water and levels, the edits by place ----------------------------------------------------

# NEMO's ORCA025.L46 T depths (FESOM3's mesh_hgr.nc nav_lev, float32): the levels of a preset whose source holds
# depths in metres (B92)
L46_T = (3.0467727184295654, 9.454049110412598, 16.36396598815918, 23.898710250854492, 32.20928955078125,
         41.48185348510742, 51.94512939453125, 63.879051208496094, 77.62451171875, 93.59412384033203,
         112.28348541259766, 134.2822723388672, 160.28399658203125, 191.09251403808594, 227.62332153320312,
         270.8962097167969, 322.01690673828125, 382.1443786621094, 452.4429016113281, 534.0197143554688,
         627.8524780273438, 734.7150268554688, 855.1112060546875, 989.2288818359375, 1136.9219970703125,
         1297.724365234375, 1470.8929443359375, 1655.4716796875, 1850.365478515625, 2054.413818359375,
         2266.45361328125, 2485.370849609375, 2710.13330078125, 2939.811767578125, 3173.58837890625,
         3410.755859375, 3650.7119140625, 3892.94970703125, 4137.046875, 4382.654296875, 4629.48486328125,
         4877.30322265625, 5125.9189453125, 5375.1767578125, 5624.95166015625, 5875.14111328125)
K_MIN = 3               # B92: the fewest layers of a water cell from a depth (FESOM3's ORCA025 mbathy: 3 .. 45)


@dataclass(frozen=True)
class Source:
    """A preset's NEMO water: `npz` (scripts/grids/orca_levels_npz.py or nemo_inputs_npz.py) in the data folder, our
    T (j, i) = its (j + row_offset, i + col_offset); `kind` "level" (NEMO's bottom level kb with its own layers z) or
    "depth" (NEMO's model bathymetry in metres, levels L46)."""
    npz: str
    row_offset: int
    col_offset: int
    kind: str
    note: str = ""
    by_index: bool = False      # the source is this grid itself (FESOM3's ORCA025): taken by index, no position check


SOURCES = {
    "ORCA2": Source("orca2", 0, -1, "level", "NEMO 5.0 ORCA_R2 domain_cfg bottom_level, its 31 levels"),
    "ORCA1": Source("eorca1", 40, 0, "depth", "Met Office GO6 eORCA1 bathymetry (GEBCO 2021, edits July 2024)"),
    "ORCA05": Source("eorca05", 93, 0, "level", "MEOM eORCA05.L121 mesh_mask mbathy, its 121 levels"),
    "ORCA025": Source("nemo025q", 0, 0, "level", "FESOM3's ORCA025.L46 mesh_zgr mbathy (fesom3_orca025/nemo025q)",
                      True),
    "ORCA12": Source("eorca12", 547, 0, "depth", "Met Office GO6 eORCA12 bathymetry (GEBCO 2021)"),
    "ORCA36": Source("eorca36", 1670, -1, "depth", "MEOM eORCA36.L121 domain_cfg bathy_metry where bottom_level > 0 "
                     "and top_level = 1 (ice-shelf cavities land), NEMO 4.2 without halos"),
}


def _source_arrays(src: Source, data_dir: Path) -> dict:
    a = dict(np.load(Path(data_dir) / f"{src.npz}.npz"))
    if "mbathy" in a:                                         # nemo_inputs_npz.py
        a["kb"] = a["mbathy"]
        a["z"] = levels_from_t(a["nav_lev"])
    return a


def mirror_t(g) -> np.ndarray:
    """(ni,) the T column each column of the fold's halo row repeats (T: K_T - i, K_T = the F mirror's K + 1)."""
    from meshom.orca.mesh import mirror_k

    ni = g.lat["t"].shape[1] - 2
    return (mirror_k(g) + 1 - np.arange(ni)) % ni


def _halo(kb: np.ndarray, g) -> np.ndarray:
    """The wrap columns and the fold's halo row of kb (nj, NI) set from the unique cells (NEMO's halo rule)."""
    nj = kb.shape[0]
    ni = kb.shape[1] - 2
    m = mirror_t(g)
    src_row = nj - 3 if g.pivot == "T" else nj - 2
    kb[nj - 1, :ni] = kb[src_row, m]
    kb[:, ni:] = kb[:, :2]
    return kb


def bottom_levels(g, preset: str, data_dir: Path, gebco: str | None = None, extra_south: int = 0,
                  min_depth_m: float | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """NEMO's bottom level on every T point of g (NEMO's layout, the wrap columns and the fold's halo row set by NEMO's
    halo rule): (kb (nj, NI) int16, z the layer bottoms, info). From the preset's SOURCES entry cell for cell where its
    T point is ours (meshom.orca.mesh.transfer_values); elsewhere (eORCA's ice-shelf rows, ORCA2's hand patch, rows
    added in the south) the depth rule (B83: GEBCO mean over the cell, water from min_depth_m) on the levels
    (`level_of_depth`). A "depth" source: its metres on L46 (`level_of_depth`, K_MIN)."""
    from meshom.orca.mesh import MIN_DEPTH_M, depth_mask, transfer_values

    src = SOURCES[preset]
    a = _source_arrays(src, data_dir)
    if src.kind == "level":
        z = np.asarray(a["z"], float)
        vals, k_min = a["kb"].astype(np.int16), int(a["kb"][a["kb"] > 0].min())
    else:
        z = levels_from_t(np.asarray(L46_T, np.float32))
        k_min = K_MIN
        vals = level_of_depth(a["depth_m"], z, k_min)
    if src.by_index and extra_south == 0 and vals.shape == g.lat["t"].shape:
        v, mapped = vals[:, :-2], np.ones(vals[:, :-2].shape, bool)
    else:
        v, mapped = transfer_values(g, a["glamt"], a["gphit"], vals, src.row_offset - extra_south, src.col_offset)
    nj, ni = mapped.shape
    kb = np.zeros((nj, ni + 2), np.int16)
    kb[:, :ni] = v
    need = ~mapped
    need[[0, nj - 1]] = False                                   # row 0 (south edge) and the halo row: not cells
    info = {"source": src.npz, "note": src.note, "kind": src.kind, "levels": int(len(z) - 1), "k_min": k_min,
            "cells_mapped": int(mapped[1:nj - 1].sum()), "cells_depth_rule": int(need.sum())}
    if need.any():
        if not gebco:
            raise ValueError(f"{preset}: {int(need.sum())} cells need the depth rule (no NEMO level there): give gebco")
        rows = np.unique(np.nonzero(need)[0])
        _, dep = depth_mask(g, gebco, rows=rows)
        md = MIN_DEPTH_M if min_depth_m is None else min_depth_m
        dd = np.where(dep >= md, dep, 0.0)
        kk = level_of_depth(np.nan_to_num(dd), z, k_min)
        kb[:, :ni][need] = kk[need]
        info["depth_rule_rows"] = [int(rows.min()), int(rows.max())]
    return _halo(kb, g), z, info


def _cells_graph(g, kb: np.ndarray):
    """The water T cells of kb as FESOM3's quads (rule 8, without the pivot half's twins): (centres lon, lat, links)."""
    from meshom.orca.mesh import mirror_k
    from meshom.orca.straits import cell_links

    nj, ni = kb.shape
    num = numbering(nj, ni, g.pivot, mirror_k(g))
    q, cell = quads(kb, num, g.pivot)
    return g.lon["t"][cell[:, 0], cell[:, 1]], g.lat["t"][cell[:, 0], cell[:, 1]], cell_links(q, (nj + 1) * ni), cell


CARRY_NEAR_KM = 200.0   # B91: a carried edit's opening path is drawn to within this of FESOM3's edit cells


def _open_path(g, b: np.ndarray, strait: str, targets: np.ndarray) -> list:
    """The fewest land T cells (edge links, the strait's box) whose opening joins the strait's two ends; ties to the
    path nearest FESOM3's edit cells `targets` (xyz). Returns [(row, column)] of the land cells to open."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    from meshom.geo import R_EARTH_M, lonlat_to_xyz
    from meshom.orca.straits import STRAITS, _in_box, _nearest

    A, B, box = STRAITS[strait]
    nj, NI = b.shape
    ni = NI - 2
    J, I = np.meshgrid(np.arange(1, nj - 1), np.arange(ni), indexing="ij")
    lon, lat = g.lon["t"][J, I], g.lat["t"][J, I]
    inb = _in_box(lon, lat, box)
    J, I, lon, lat = J[inb], I[inb], lon[inb], lat[inb]
    n = len(J)
    if n == 0:
        return []
    key = {(int(j), int(i)): k for k, (j, i) in enumerate(zip(J, I))}
    wet = b[J, I] > 0
    P = lonlat_to_xyz(lon, lat)
    dk = np.arccos(np.clip((P @ targets.T).max(axis=1), -1, 1)) * R_EARTH_M / 1e3 if len(targets) else np.zeros(n)
    cost = np.where(wet, 1e-6, 1.0) + 1e-3 * np.minimum(dk, CARRY_NEAR_KM) / CARRY_NEAR_KM
    ra, rb = [], []
    for k, (j, i) in enumerate(zip(J.tolist(), I.tolist())):
        for dj, di in ((1, 0), (0, 1)):
            o = key.get((j + dj, (i + di) % ni))
            if o is not None:
                ra += [k, o]
                rb += [o, k]
    ra, rb = np.asarray(ra), np.asarray(rb)
    G = coo_matrix((cost[rb], (ra, rb)), shape=(n, n)).tocsr()

    w = np.flatnonzero(wet)
    sa, sb = _nearest(lon, lat, w, A), _nearest(lon, lat, w, B)        # within END_DEG (a basin with no water: none)
    if sa < 0 or sb < 0 or sa == sb:
        return []
    _, pred = dijkstra(G, indices=sa, return_predecessors=True)
    if pred[sb] < 0:
        return []
    path, v = [], sb
    while v != sa and v >= 0:
        path.append(v)
        v = pred[v]
    return [(int(J[k]), int(I[k])) for k in path[::-1] if not wet[k]]


def carry_edits(g, kb: np.ndarray) -> tuple[np.ndarray, list]:
    """FESOM3's hand edits by place (B91) on kb (after `one_cell_land`): for each strait of ORCA025_EDITS that is
    closed on this grid (meshom.orca.straits.measure, the strait's ends and box), the fewest land T cells that join its
    two ends within the box are opened (`_open_path`: edge links; among equal counts the path nearest FESOM3's edit
    cells), each taking the level of its deepest water edge neighbour (FESOM3's edits copy a neighbour's level), in
    the path's order. Returns (kb, per strait {strait, before, after, cells_opened, at}). ORCA025 with FESOM3's own
    levels takes the edits by index (`build_preset`)."""
    from meshom.geo import lonlat_to_xyz
    from meshom.orca.straits import STRAITS, measure

    b = np.array(kb, copy=True)
    nj, NI = b.shape
    ni = NI - 2
    out = []
    cl, cp, links, _ = _cells_graph(g, b)                     # rebuilt only after a strait was opened
    for strait in dict.fromkeys(e.strait for e in ORCA025_EDITS):
        if strait not in STRAITS:
            continue
        before = measure(strait, cl, cp, links)
        rec = {"strait": strait, "before": {k: before.get(k) for k in ("open", "cells_across")}, "cells_opened": 0,
               "at": []}
        if not before.get("open"):
            tg = np.array([c[:2] for e in ORCA025_EDITS if e.strait == strait for c in e.cells])
            path = _open_path(g, b, strait, lonlat_to_xyz(tg[:, 0], tg[:, 1]))
            for j, i in path:
                nbr = [b[j + dj, (i + di) % ni] for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= j + dj < nj]
                lev = int(max(nbr))
                if lev <= 0:                                  # the path's next cell: the level of the path so far
                    lev = int(max([b[p] for p in path if b[p] > 0] or [K_MIN]))
                b[j, i] = lev
                rec["at"].append([round(float(g.lon["t"][j, i]), 3), round(float(g.lat["t"][j, i]), 3), lev])
            rec["cells_opened"] = len(path)
            b = _halo(b, g)
            cl, cp, links, _ = _cells_graph(g, b)
            after = measure(strait, cl, cp, links)
        else:
            after = before
        rec["after"] = {k: after.get(k) for k in ("open", "cells_across")}
        out.append(rec)
    return b, out


def build_preset(preset: str, data_dir: Path, gebco: str | None = None, g=None, deadends: bool = True,
                 float32: bool = True) -> Fesom3Mesh:
    """A preset's whole grid (meshom.orca.presets.build) as FESOM3 makes ORCA025 (rules 1-12): the bottom levels of
    `bottom_levels`; ORCA025 with FESOM3's source takes the edits and the pole columns by index,
    every other preset the edits by place (`carry_edits`, only where the strait is closed) and the pole columns by
    place (`pole_columns`). Positions rounded to float32 as FESOM3's files (float32=True)."""
    from meshom.orca import build
    from meshom.orca.mesh import mirror_k

    g = g if g is not None else build(preset)
    kb, z, info = bottom_levels(g, preset, data_dir, gebco)
    lon_f, lat_f = g.lon["f"], g.lat["f"]
    nj = kb.shape[0]
    if preset == "ORCA025" and SOURCES[preset].npz == "nemo025q":
        m = convert(lon_f, lat_f, g.lat["t"], kb, z, g.pivot, ORCA025_EDITS, ORCA025_POLE_COLS, mirror_k(g),
                    deadends)
        m.info["pole_cols_by_place"] = [list(c) for c in pole_columns(lon_f, lat_f, nj - 3)]
    else:
        kb1, n1 = one_cell_land(kb)
        kb1 = _halo(kb1, g)
        kb2, carried = carry_edits(g, kb1)
        m = convert(lon_f, lat_f, g.lat["t"], kb2, z, g.pivot, (), pole_columns(lon_f, lat_f, nj - 3), mirror_k(g),
                    deadends, one_cell=False)
        m.info.update(one_cell_land=n1, edits_carried=carried)
    m.info["bottom_levels"] = info
    m.info["preset"] = preset
    if float32:
        m.lon, m.lat = m.lon.astype(np.float32), m.lat.astype(np.float32)
    return m


def run_fesom_ini_nemo(binary: str, mesh_dir: Path, work_dir: Path, n_part: int = 1) -> dict:
    """fesom_ini on a FESOM3 ORCA folder with FESOM3's NEMO depth file (`use_depthfile="nemo"`; fesom_ini keeps
    derived files of an earlier mesh: they are removed first). The record as
    meshom.io.fesom.run_fesom_ini's."""
    import resource
    import subprocess
    import time

    mesh_dir, work = Path(mesh_dir).resolve(), Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    for f in ("edgenum.out", "edges.out", "edge_tri.out", "nlvls.out", "elvls.out", "elvls_raw.out"):
        (mesh_dir / f).unlink(missing_ok=True)
    (work / "namelist.config").write_text(f'&paths\nMeshPath="{mesh_dir}/"\n/\n&geometry\nuse_depthfile="nemo"\n/\n'
                                          f'&run_config\n/\n&machine\nn_levels=1\nn_part={n_part}\n/\n')

    def big_stack() -> None:
        hard = resource.getrlimit(resource.RLIMIT_STACK)[1]
        resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))
    t0 = time.perf_counter()
    with open(work / "fesom_ini.out", "w") as f:
        rc = subprocess.run([str(binary)], cwd=work, stdout=f, stderr=subprocess.STDOUT, preexec_fn=big_stack).returncode
    log = (work / "fesom_ini.out").read_text(errors="replace").splitlines()
    stops = [ln.strip() for ln in log if any(k in ln for k in ("ERROR", "Recompile", "Aborting", "Insufficient"))]
    derived = all((mesh_dir / f).exists() for f in ("edgenum.out", "edges.out", "edge_tri.out", "nlvls.out"))
    return {"binary": str(binary), "n_part": n_part, "use_depthfile": "nemo", "exit_status": rc,
            "wall_s": round(time.perf_counter() - t0, 2), "stop_lines": stops[:10], "derived_files": derived,
            "total_time": next((ln.strip() for ln in log if "Total time" in ln), None),
            "log": str(work / "fesom_ini.out"),
            "ok": rc == 0 and derived and not any("ERROR" in ln or "Recompile" in ln for ln in stops)}


def main(argv: list[str] | None = None) -> None:
    """python -m meshom.orca.fesom3 PRESET DATA_DIR OUT_DIR [--gebco GEBCO] [--fesom-ini BIN]: a preset's whole grid
    by FESOM3's rules into OUT_DIR (nod2d.out, elem2d.out, depth.out, fesom3_orca.json)."""
    import argparse
    import json

    ap = argparse.ArgumentParser(prog="python -m meshom.orca.fesom3")
    ap.add_argument("preset")
    ap.add_argument("data_dir")
    ap.add_argument("out")
    ap.add_argument("--gebco")
    ap.add_argument("--fesom-ini")
    a = ap.parse_args(argv)
    m = build_preset(a.preset, Path(a.data_dir), a.gebco)
    out = Path(a.out)
    write(m, out)
    rec = {"nodes": int(len(m.lon)), "quads": int(len(m.elem)), **m.info}
    if a.fesom_ini:
        rec["fesom_ini"] = run_fesom_ini_nemo(a.fesom_ini, out, out.parent / f"{out.name}_fesom_ini")
    (out / "fesom3_orca.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: v for k, v in rec.items() if k != "edits_carried"}, default=str)[:2000])


if __name__ == "__main__":
    main()
