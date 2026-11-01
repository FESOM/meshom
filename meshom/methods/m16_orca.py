"""m16_orca: an ORCA grid built by NEMO's own generator (meshom.orca), whole or between two latitudes, with NEMO's
land-sea mask; outside the latitude range the sphere is meshed by a meshom method, joined conformingly (decisions B71,
B74-B76, B78-B86, B108, B109).

- "Build ORCA X" (`orca` = a preset of meshom.orca.presets, `lat_south` and `lat_north` unset): the preset's whole
  grid, its north fold closed (meshom.orca.mesh.cells), the cells NEMO's mask calls water (meshom.orca.mesh: cell for
  cell from NEMO's file where its T point is ours, else the depth rule), cleaned as FESOM3 needs (fesom_cleanup:
  spurs removed, pinch nodes split, closed seas removed unless `keep_seas` is true, B171).
- "ORCA X from A to B": the ORCA rows from the first F row south of `lat_south` (the grid is extended south when A
  lies south of the preset's own first row) to the F row whose mean latitude is nearest `lat_north`; a north cap on
  the cut row (when `lat_north` is set) and a south cap on the first row (when that row lies north of the preset's own
  first row) meshed by `cap_method` on the domain bounded by that row as one exact loop, so the joint is conforming
  (the caps' elements next to it may be triangles), merged by the row's node positions.

- A refined part around the north fold (`fine_orca`, `fine_lat`; B108): the same ORCA map on a lattice k times finer
  (ORCA12 = ORCA025 x 3: every third ORCA12 line is an ORCA025 line, meshom.orca.mesh.nesting), its cells per column
  from the fold down while the T point lies north of `fine_lat`, masked like the coarse part (its own NEMO source),
  kept unchanged as a second ORCA part. The cap is then the band between the coarse part's north edge and the fine
  part's edge (two exact loops); with m14's `auto_blocks` off and m02's ORCA `frame` (B109) its fill keeps ORCA's
  directions and only changes size.
- The ladder (`transition = "ladder"`, `ladder_lat`; B114-B116): the coarse part up to a plain F row, the fine part
  from a plain fine F row (every column), between them meshom.orca.ladder's structured strip in ORCA's index space
  (rows on j-lines graded as h, columns on i-lines, 2 Y-splits per coarse column, staggered) in place of cap_method.
- The nest (`nest`; B172-B176): the refined preset's cells inside a box of coarse F lines, the coarse part without
  the box widened by the frame's width, between the two boxes meshom.orca.nest's frame (graded rectangular rings,
  tensor corners, the ladder along the sides), merged node for node with both parts.
- A nest in the nest (`nest.inner`; B230): the nest's fine box holds a second frame of `inner.width` steps of a
  preset that refines the nest's fine preset by 3 (ORCA36 in ORCA12), and that preset's cells inside it; the two
  frames meet on the fine box's edge (ring n of the first is ring 0 of the second), so the box holds no cells of the
  middle preset and the step from the coarse grid to the finest is 9 in two graded frames of 3.

The ORCA part is never touched after: no smoothing or optimisation acts on it (keep `[mesh] optimize` unset and
`[run] smooth` false). meshom run's land cut then cuts the caps by the coast (meshom's rule there) and leaves the ORCA
cells as NEMO's mask has them: `cut_wet` (the worker's hook) marks them water and protected (B85).

Under `meshom run` (MESHOM_RUN_DIR set) the method writes <run dir>/orca.json (the grid, the rows, the mask and its
clean-up, the joints) and <run dir>/orca_belt.npz (the ORCA part's points and elements: the reference of
`belt_unchanged`).
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from meshom.domain import Domain
from meshom.mesh import Mesh, build_topology
from meshom.methods import METHODS, register
from meshom.size import SizeField

TAKES_LAND = True           # meshom.run hands this method the true coast as opts["land"] under a land cut

DEFAULTS = dict(
    orca="ORCA1",             # B78-B81: the preset
    lat_south=None,           # B74: None = the preset's own first row; else the first F row south of this
    lat_north=None,           # B74: None = to the north fold (the whole grid); else the F row nearest this
    region=None,              # B100: None; else [[lon, lat], ...]: the ORCA part per column up to this line (no lat_north)
    fine_orca=None,           # B108: None; else a preset that is `orca` refined k times on the same map (ORCA12 for
                              # ORCA025): its cells around the north fold kept as a second ORCA part
    fine_lat=None,            # B108: ... the fine cells per column from the fold down while the T point lies north of it
    transition="fill",        # B114: "fill" = cap_method between the two parts (B108-B112); "ladder" = the structured
                              # strip in ORCA's index space (meshom.orca.ladder) between two plain rows
    ladder_lat=None,          # B114: [lat_c, lat_f]: the coarse part up to its last F row south of lat_c, the fine
                              # part from its last F row south of lat_f (all north of lat_f); replaces region, fine_lat
    ladder_theta_deg=None,    # B115: an arm's tilt at a Y-split (None: meshom.orca.ladder.THETA_DEG)
    ladder_relax=None,        # B115: the relaxation's weights [widths, tilts, bases] (None: ladder.RELAX; false: off)
    nest=None,                # B176: None; else {fine_orca, box = [c0, c1, r0, r1] (coarse F lines) or lonlat = [lon0,
                              # lat0, lon1, lat1], width (fine steps, default nest.H_DEFAULT), theta_deg, relax}: the
                              # fine ORCA in the box, the coarse ORCA outside the widened box, meshom.orca.nest's frame;
                              # B230: inner = {fine_orca (refines nest.fine_orca by 3), width, theta_deg, relax}: a
                              # second frame inside the fine box and that preset's cells inside it
                              # B216: depth = "gebco" (default "": the run's): the nodes inside the coarse box take
                              # the box mean per node (no smoothing, no strait floor), with depth_channel ("near",
                              # "cells") and sill_level ("grid", "shallower") as an insert's (nest_depth)
    mask="nemo",              # B82: NEMO's mask ("nemo"), FESOM3's rules ("fesom3", B94), or "all" (every cell)
    ref_dir=None,             # the reference npz (scripts/grids/orca_refs_npz.py); None: $MESHOM_ORCA_REFS
    levels_dir=None,          # B91: NEMO's levels npz (orca_levels_npz.py, nemo_inputs_npz.py); $MESHOM_ORCA_LEVELS
    gebco=None,               # B83: the depth rule's grid (None: the rule cannot run)
    min_depth_m=None,         # B83: None = meshom.orca.mesh.MIN_DEPTH_M
    keep_seas=False,          # B84, B171: NEMO's closed seas removed (true: kept, in the ORCA part and the run's cut)
    cap_method="m14_blocks_rep",   # B75
    cap_opts={},
    size_tolerance=0.20,      # G38
)

JOINT_TOL = 1e-9            # rad: a cap boundary node within this of an ORCA row node is that node
_STATE: dict = {}           # the last run's ORCA part, for cut_wet (same process: meshom.run's worker)


def _record_dir() -> Path | None:
    d = os.environ.get("MESHOM_RUN_DIR")
    return Path(d) if d else None


def _levels_dir(o: dict) -> Path:
    d = o.get("levels_dir") or os.environ.get("MESHOM_ORCA_LEVELS")
    if not d:
        raise ValueError("m16_orca: FESOM3's rules need `levels_dir` (or $MESHOM_ORCA_LEVELS): NEMO's levels npz")
    return Path(d)


def fesom3_levels(g, name: str, o: dict, extra: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """FESOM3's rules 1-4 on the bottom levels of g (B94, meshom.orca.fesom3): NEMO's levels (`bottom_levels`), the
    pole columns' F points corrected in g (rule 2), one-cell land removed (rule 3), the hand edits (rule 4: by index
    on ORCA025 with FESOM3's own levels, else carried by place). Returns (kb (nj, NI) after the rules, z, info)."""
    from meshom.orca import fesom3 as F

    kb, z, info = F.bottom_levels(g, name, _levels_dir(o), o["gebco"], extra, o["min_depth_m"])
    nj = kb.shape[0]
    exact = name == "ORCA025" and F.SOURCES[name].by_index and extra == 0
    cols = F.ORCA025_POLE_COLS if exact else F.pole_columns(g.lon["f"], g.lat["f"], nj - 3)
    g.lon["f"], g.lat["f"] = F.fold_fix(g.lon["f"], g.lat["f"], (nj - 3, nj - 2), cols)
    kb, n1 = F.one_cell_land(kb)
    info.update(pole_cols=[list(c) for c in cols], one_cell_land=n1)
    if exact:
        kb, info["edits"] = F.apply_edits(kb, F.ORCA025_EDITS)
    else:
        kb, info["edits_carried"] = F.carry_edits(g, F._halo(kb, g))
    return kb, z, info


def _ref_dir(o: dict) -> Path:
    d = o.get("ref_dir") or os.environ.get("MESHOM_ORCA_REFS")
    if not d:
        raise ValueError("m16_orca: NEMO's mask needs `ref_dir` (or $MESHOM_ORCA_REFS): the reference npz")
    return Path(d)


def merge(belt: Mesh, row_nodes: np.ndarray, cap: Mesh) -> tuple[Mesh, dict]:
    """The belt plus a cap, the cap's boundary nodes identified with the belt's nodes on the joint row by position
    (row_nodes: those belt nodes; a cap boundary node above a land cell of the row has no belt node and stays the
    cap's). info: joint counts (cap boundary nodes, matched, unmatched, the largest match distance, joint edges with
    a belt and a cap element)."""
    ctop = build_topology(cap)
    bnd = np.flatnonzero(ctop.boundary_node)
    tree = cKDTree(belt.points[row_nodes])
    dist, k = tree.query(cap.points[bnd])
    ok = dist <= JOINT_TOL
    nb = belt.n_nodes
    new_id = np.full(cap.n_nodes, -1, dtype=np.int64)
    new_id[bnd[ok]] = row_nodes[k[ok]]
    rest = np.flatnonzero(new_id < 0)
    new_id[rest] = nb + np.arange(len(rest))
    pts = np.vstack([belt.points, cap.points[rest]])
    quads = np.vstack([belt.quads, new_id[cap.quads]]) if cap.n_quads else belt.quads
    tris = np.vstack([belt.tris, new_id[cap.tris]]) if cap.n_tris else belt.tris
    zone = None
    if cap.tri_zone is not None or belt.tri_zone is not None:
        bz = np.zeros(belt.n_tris, bool) if belt.tri_zone is None else np.asarray(belt.tri_zone, bool)
        cz = np.zeros(cap.n_tris, bool) if cap.tri_zone is None else np.asarray(cap.tri_zone, bool)
        zone = np.r_[bz, cz]
    out = Mesh(pts, tris.astype(np.int64), quads.astype(np.int64), np.zeros(len(pts), np.int8), zone)
    top = build_topology(out)
    shared = int(((top.edge_elems[:, 1] >= 0) & np.isin(top.edges, row_nodes).all(axis=1)).sum())
    info = {"cap_boundary_nodes": int(len(bnd)), "matched": int(ok.sum()), "unmatched": int((~ok).sum()),
            "max_match_rad": float(dist[ok].max()) if ok.any() else 0.0,
            "joint_edges_shared": shared,
            "unmatched_min_rad": float(dist[~ok].min()) if (~ok).any() else None,
            "boundary_edges_after": int((top.edge_elems[:, 1] < 0).sum())}
    # conforming: every cap boundary node is either a belt node or not near one (a gap of a node would be a T-junction)
    info["conforming"] = bool(info["unmatched_min_rad"] is None or info["unmatched_min_rad"] > 1e-6)
    return out, info


def _pos_ids(*point_sets: np.ndarray) -> list[np.ndarray]:
    """A common integer id per position (rounded to 1e-9 of the radius, 6 mm) for the points of each set."""
    allp = np.vstack(point_sets)
    _, inv = np.unique(np.round(allp * 1e9).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    out, k = [], 0
    for p in point_sets:
        out.append(inv[k:k + len(p)])
        k += len(p)
    return out


def _canon(ids: np.ndarray, elems: np.ndarray) -> np.ndarray:
    """Elements as rows of position ids, cyclic order kept, the smallest first, as one void value per row."""
    e = ids[elems]
    if not len(e):
        return np.zeros(0, dtype=np.dtype((np.void, 8 * max(elems.shape[1], 1))))
    n = e.shape[1]
    r = np.argmin(e, axis=1)
    e = np.ascontiguousarray(np.take_along_axis(e, (r[:, None] + np.arange(n)) % n, axis=1).astype(np.int64))
    return e.view(np.dtype((np.void, 8 * n))).ravel()


def _ruled(mesh: Mesh, belt_points: np.ndarray, belt_quads: np.ndarray, belt_tris: np.ndarray) -> np.ndarray:
    """(n_elems,) the elements of `mesh` (tris first) that are elements of the belt (same positions, same order)."""
    mi, bi = _pos_ids(mesh.points, belt_points)
    out = np.zeros(mesh.n_elems, bool)
    if mesh.n_tris and len(belt_tris):
        out[:mesh.n_tris] = np.isin(_canon(mi, mesh.tris), _canon(bi, belt_tris))
    if mesh.n_quads and len(belt_quads):
        out[mesh.n_tris:] = np.isin(_canon(mi, mesh.quads), _canon(bi, belt_quads))
    return out


def belt_unchanged(mesh: Mesh, belt_points: np.ndarray, belt_quads: np.ndarray,
                   belt_tris: np.ndarray | None = None) -> dict:
    """Whether `mesh` keeps the ORCA part as it was: every node of `mesh` within 1 m of a belt node and its distance
    (max displacement, m; 0 = untouched), no node within 1 km that is not a belt node, and every element of `mesh` made
    of belt nodes one of the belt's elements (same positions, same cyclic order)."""
    from meshom.geo import R_EARTH_M

    tree = cKDTree(belt_points)
    d, k = tree.query(mesh.points, distance_upper_bound=1.0 / R_EARTH_M)
    hit = np.isfinite(d)
    near, _ = tree.query(mesh.points)
    moved = (near * R_EARTH_M < 1000.0) & ~hit
    bt = np.zeros((0, 3), np.int64) if belt_tris is None else np.asarray(belt_tris)
    on = np.r_[hit[mesh.tris].all(axis=1) if mesh.n_tris else np.zeros(0, bool),
               hit[mesh.quads].all(axis=1) if mesh.n_quads else np.zeros(0, bool)]
    kept = _ruled(mesh, belt_points, belt_quads, bt)
    return {"belt_nodes_kept": int(hit.sum()), "belt_nodes_total": int(len(belt_points)),
            "max_displacement_m": float(d[hit].max() * R_EARTH_M) if hit.any() else 0.0,
            "nodes_within_1km_not_on_belt": int(moved.sum()),
            "belt_elems_kept": int(kept.sum()), "belt_elems_total": int(len(belt_quads) + len(bt)),
            "elems_on_belt_nodes": int(on.sum()), "belt_elems_altered": int((on & ~kept).sum())}


def cut_wet(mesh: Mesh) -> tuple[np.ndarray, np.ndarray] | None:
    """meshom.run's hook (B85): the elements of `mesh` that are this run's ORCA cells (same positions as the ORCA
    part's elements) and their water flag (all True: the ORCA part holds only NEMO's water cells). None when the
    method has not run in this process."""
    st = _STATE.get("belt")
    if st is None:
        return None
    ruled = _ruled(mesh, *st)
    return ruled, ruled.copy()


def node_depth(run_dir: Path, mesh: Mesh) -> tuple[np.ndarray, np.ndarray] | None:
    """FESOM3's node depth on the ORCA nodes of a run's final mesh (B94, rule 6: the deepest of the four T cells' NEMO
    levels, as the layer bottom z_k): (mesh node ids, depth m). None when the run's ORCA part has no levels (mask not
    "fesom3"). The run's bathymetry stage gives these nodes this depth in place of the GEBCO one (meshom.run
    mesh_depth's hook; the nodes match the ORCA part's by position, JOINT_TOL)."""
    f = Path(run_dir) / "orca_belt.npz"
    if not f.exists():
        return None
    b = np.load(f)
    if "node_level" not in b or not len(b["node_level"]):
        return None
    d, k = cKDTree(b["points"]).query(mesh.points, distance_upper_bound=JOINT_TOL)
    hit = np.flatnonzero(np.isfinite(d))
    return hit, np.asarray(b["z"], float)[b["node_level"][k[hit]]]


def nest_box_nodes(run_dir: Path, mesh: Mesh) -> np.ndarray | None:
    """B216: (N,) bool, the nodes of `mesh` strictly inside the nest's coarse box: the coarse grid's F points on the
    box's four lines (orca_belt.npz) as one loop, meshom.size.polygon_inside; the loop's own points (the frame's ring
    0, nodes of the coarse grid) are outside. None when the run has no nest."""
    from meshom.geo import lonlat_to_xyz
    from meshom.size import polygon_inside

    f = Path(run_dir) / "orca_belt.npz"
    if not f.exists():
        return None
    b = np.load(f)
    if "nest_coarse_box" not in b.files:
        return None
    c0, c1, r0, r1 = map(int, b["nest_coarse_box"])
    lon, lat = b["lon"], b["lat"]                             # F rows t_rows[0] - 1 .. t_rows[1], columns 0 .. ni - 1
    k0, ni = int(b["t_rows"][0]) - 1, lon.shape[1]
    rows = np.r_[np.full(c1 - c0, r0), np.arange(r0, r1), np.full(c1 - c0, r1), np.arange(r1, r0, -1)] - k0
    cols = np.r_[np.arange(c0, c1), np.full(r1 - r0, c1), np.arange(c1, c0, -1), np.full(r1 - r0, c0)] % ni
    loop = lonlat_to_xyz(lon[rows, cols], lat[rows, cols])
    d, _ = cKDTree(loop).query(mesh.points, distance_upper_bound=JOINT_TOL)
    return polygon_inside(mesh.points, loop) & ~np.isfinite(d)


def nest_depth(run_dir: Path, mesh: Mesh, depth: np.ndarray, cfg, fmt: str, log=print) -> tuple[np.ndarray, dict]:
    """B216, `nest.depth = "gebco"`: the nodes strictly inside the nest's coarse box (nest_box_nodes) take meshom's
    grid rule as an insert's own nodes do (meshom.insert.core.region_depth, B211): the box mean per node, no smoothing,
    no strait depth floor, the run's slope limiter, the depth floor and the regions; with `nest.depth_channel` the
    channel raise (B213; the line's ends: the deepest node of each of the two largest wet pieces of the box's edge;
    box means at the median spacing of the box's nodes) and with `nest.sill_level` the sill (B212 / B215). Computed on
    the box's cells and one cell ring round them, nodes ordered by position and cells by their nodes, so the depth is
    the same in every mesh that holds the box's cells (an insert elsewhere recomputes it exactly). Every other node
    keeps `depth`. Returns (depth, record)."""
    from meshom.insert.core import region_depth

    o = cfg.mesh.opts.get("nest") or {}
    inside = nest_box_nodes(run_dir, mesh)
    if inside is None or not inside.any():
        return depth, {"nodes": 0}
    out, rec = region_depth(mesh, inside, depth, cfg, fmt, str(o.get("depth_channel") or ""),
                            str(o.get("sill_level") or ""), "[nest]", log)
    log(f"[nest] depth of the {int(inside.sum())} nodes inside the coarse box: the box mean per node, no smoothing, no "
        f"strait floor{', channel ' + o['depth_channel'] if o.get('depth_channel') else ''}"
        f"{', sill ' + o['sill_level'] if o.get('sill_level') else ''}: {out[inside].min():.0f}-{out[inside].max():.0f} m "
        f"(before: {np.asarray(depth)[inside].min():.0f}-{np.asarray(depth)[inside].max():.0f} m)", flush=True)
    return out, rec


def _rows(g, lat_south, lat_north) -> tuple[int, int, bool, bool]:
    """(first T row, last T row, south cap, north cap) of the ORCA part: the first F row is the last one whose mean
    latitude is below lat_south (row 0 without it), the last F row the one whose mean latitude is nearest lat_north
    (the pole line without it)."""
    lat_f = g.lat["f"][:, :-2].mean(axis=1)
    f0 = 0
    if lat_south is not None:
        below = np.flatnonzero(lat_f[: g.j_pole] < lat_south)
        f0 = int(below[-1]) if len(below) else 0
    j1 = g.j_pole
    if lat_north is not None:
        fc = int(np.argmin(np.abs(lat_f[: g.j_pole] - lat_north)))
        if fc >= g.j_pole - 1:
            raise ValueError(f"lat_north {lat_north} reaches the fold")
        j1 = fc
    return f0 + 1, j1, f0 > 0, lat_north is not None


def _part_cells(g, j0: int, j1: int, last: np.ndarray | None):
    """The ORCA part's cells: T rows j0 .. j1, or with `last` (B100) per column i the rows j0 .. last[i]."""
    from meshom.orca.mesh import cells, region_cells

    oc = cells(g, j0, j1)
    return oc if last is None else region_cells(oc, last)


def cleanup_keeps_seas(o: dict, caps: bool) -> bool:
    """B171: whether an ORCA part's clean-up (fesom_cleanup) keeps its separate water bodies: with `keep_seas`, or
    when the part has a cap or a fill next to it (a body of the part may reach the ocean only through it; the run's
    land cut then keeps only the ocean-connected water, C48)."""
    return bool({**DEFAULTS, **o}["keep_seas"]) or caps


def cut_keeps_seas(opts: dict) -> bool:
    """meshom.run's hook (B85, B171): whether the run's land cut keeps the closed seas made only of ORCA cells."""
    return bool({**DEFAULTS, **opts}["keep_seas"])


def _masked(g, name: str, o: dict, extra: int, make_cells, caps: bool = False, context=None) -> dict:
    """An ORCA part with its mask (o["mask"]): make_cells() gives its cells (called again after FESOM3's rule 2 moved
    the pole columns' F points); caps: the part meets a cap or a fill (`cleanup_keeps_seas`); context(oc) -> (n,)
    bool: cells of make_cells() that are not the part's but its grid's ring across a box edge (the nest, B176): they
    take part in the clean-up with their own mask, so the cells at the edge are judged as in the whole grid, and are
    dropped after it. Returns the part: oc, belt (its kept cells as a mesh), wetmap / wet / mapped (nj, ni)
    or None, node_level (per belt node) and z or None, info (mask, cleanup, levels)."""
    from meshom.orca import PRESETS
    from meshom.orca.mesh import MIN_DEPTH_M, depth_mask, sub_cells, transfer_mask
    from meshom.orca.mesh import fesom_cleanup as _cleanup

    def fesom_cleanup(mesh, keep, keep_seas):
        if context is None:
            return _cleanup(mesh, keep, keep_seas)
        from meshom.methods.m14.cut import half_edges

        ctx = context(oc)
        _, _, e, tw = half_edges(mesh)                            # B176: the ring's cells on the crop edge are pinned:
        crop = np.bincount(e[tw < 0], minlength=len(ctx)) > 0     # the grid goes on beyond them, so the crop is never
        _, kept, cinfo = _cleanup(mesh, keep, keep_seas, ctx & crop)      # a dead end (the frame's cut decides later)
        kept = kept & ~ctx                                        # the ring across the box edge dropped now
        keep[ctx] = False
        cinfo.update(context_cells=int(ctx.sum()), pinned_cells=int((ctx & crop).sum()))
        return sub_cells(oc, kept)[0], kept, cinfo

    pr = PRESETS[name]
    oc = make_cells()
    nj, ni = g.lat["t"].shape[0], g.lat["t"].shape[1] - 2
    ci, cj = oc.cell_ij[:, 0], oc.cell_ij[:, 1]
    present = np.zeros((nj, ni), bool)                            # the cells of the ORCA mesh (one copy at a T pivot)
    present[ci, cj] = True
    info = {"cells": oc.info}
    node_level, z = None, None
    if o["mask"] == "fesom3":                                     # B94: FESOM3's rules on NEMO's levels
        from meshom.orca import fesom3 as F

        kb, z, finfo = fesom3_levels(g, name, o, extra)
        oc = make_cells()                                         # again: the pole columns' F points moved (rule 2)
        ci, cj = oc.cell_ij[:, 0], oc.cell_ij[:, 1]
        info["cells"] = oc.info
        keep = kb[ci, cj] > 0
        info["mask"] = {"source": "fesom3", **finfo, "cells": int(len(keep)), "water_cells": int(keep.sum())}
        belt, kept, cinfo = fesom_cleanup(oc.mesh, keep, cleanup_keeps_seas(o, caps))
        info["cleanup"] = cinfo
        lev = F.node_levels(kb)                                   # rule 6 on the extended F rows (F row r -> r + 1)
        T = oc.mesh.n_tris
        used = np.unique(np.r_[oc.mesh.tris[kept[:T]].ravel(), oc.mesh.quads[kept[T:]].ravel()])
        nij = oc.node_ij[used]
        node_level = lev[nij[:, 0] + 1, nij[:, 1]]
        wetmap = np.zeros((nj, ni), np.int8)
        wetmap[ci, cj] = keep.astype(np.int8) + kept.astype(np.int8)
        wet, mapped = kb[:, :ni] > 0, np.ones((nj, ni), bool)
        info["levels"] = {"z_m": [round(float(x), 4) for x in z], "node_level_min": int(node_level.min()),
                          "node_level_max": int(node_level.max())}
    elif o["mask"] == "nemo":
        wet = np.zeros((nj, ni), bool)
        mapped = np.zeros((nj, ni), bool)
        if pr.ref is not None:
            ref = np.load(_ref_dir(o) / f"{pr.ref}.npz")
            wet, mapped = transfer_mask(g, ref, pr.ref_row_offset - extra, pr.ref_col_offset, pr.ref_mask)
        need = ~mapped[ci, cj]
        rows_need = np.unique(ci[need])
        n_depth = int(need.sum())
        if n_depth:
            if not o["gebco"]:
                raise ValueError(f"m16_orca: {n_depth} cells need the depth rule (no NEMO mask there): set `gebco`")
            dw, _ = depth_mask(g, o["gebco"], min_depth_m=o["min_depth_m"] or MIN_DEPTH_M, rows=rows_need)
            wet = np.where(mapped, wet, dw)
        keep = wet[ci, cj]
        info["mask"] = {"source": pr.ref or "depth rule", "cells": int(len(keep)),
                        "cells_from_nemo": int((~need).sum()), "cells_by_depth_rule": n_depth,
                        "depth_rule_rows": [int(rows_need.min()), int(rows_need.max())] if n_depth else None,
                        "min_depth_m": float(o["min_depth_m"] or MIN_DEPTH_M), "water_cells": int(keep.sum())}
        belt, kept, cinfo = fesom_cleanup(oc.mesh, keep, cleanup_keeps_seas(o, caps))
        info["cleanup"] = cinfo
        wetmap = np.zeros((nj, ni), np.int8)                      # 1 water by the mask, 2 kept after the clean-up
        wetmap[ci, cj] = keep.astype(np.int8) + kept.astype(np.int8)
    else:
        belt, wetmap, mapped, wet = oc.mesh, None, None, None
        if context is not None:
            belt = sub_cells(oc, ~context(oc))[0]
    if context is not None:
        present[:] = False
        ctx = context(oc)
        info["context_cells"] = int(ctx.sum())
        present[oc.cell_ij[~ctx, 0], oc.cell_ij[~ctx, 1]] = True
    return {"oc": oc, "belt": belt, "wetmap": wetmap, "wet": wet, "mapped": mapped, "present": present,
            "node_level": node_level, "z": z, "info": info}


def _union(a: Mesh, b: Mesh) -> Mesh:
    """Two meshes without common nodes as one (b's nodes after a's)."""
    n = a.n_nodes
    return Mesh(np.vstack([a.points, b.points]), np.vstack([a.tris, b.tris + n]).astype(np.int64),
                np.vstack([a.quads, b.quads + n]).astype(np.int64), np.zeros(n + b.n_nodes, np.int8))


def _fine_part(gc, o: dict, gf=None, first_row: int | None = None, box=None, top: bool = True) -> dict:
    """B108: the refined ORCA part: preset o["fine_orca"] (the coarse grid refined k times on the same map, checked by
    meshom.orca.mesh.nesting). top: its cells around the north fold, per column from the top down while the T point
    lies north of o["fine_lat"] (top_rows), or (the ladder, B114) every column from T row first_row up, the whole
    cells' edge as one loop (the fill on its left). box (the nest, B176): also (or only, top false) its cells inside the
    box of coarse F lines (c0, c1, r0, r1). Masked as the coarse part (o["mask"]), one part for both."""
    from meshom.orca import PRESETS, build
    from meshom.orca.mesh import OrcaCells, cells, edge_loop, in_box, nesting, sub_cells, top_cells, top_rows

    name = str(o["fine_orca"])
    gf = build(name, 0) if gf is None else gf
    nest = nesting(gc, gf)
    if nest["p99_m"] > 1.0:
        raise ValueError(f"m16_orca: {name} is not {o['orca']} refined on the same map: {nest}")
    nf = gf.lon["f"].shape[1] - 2
    first, fbox = None, None
    if top:
        first = top_rows(gf, float(o["fine_lat"])) if first_row is None else np.full(nf, int(first_row), np.int64)
    if box is not None:
        k, b_i, b_j = nest["k"], nest["col_offset"], nest["row_offset"]
        c0, c1, r0, r1 = map(int, box)
        fbox = (k * c0 + b_i, k * c1 + b_i, k * r0 + b_j, k * r1 + b_j)
    j0 = min(([int(first.min())] if top else []) + ([fbox[2] + 1] if fbox else []))
    j1 = gf.j_pole if top else fbox[3]

    if fbox:                                                       # B176: the fine ring outside the box: context
        from meshom.orca.nest import CONTEXT_CELLS as r
        fwide = (fbox[0] - r, fbox[1] + r, fbox[2] - r, fbox[3] + r)
        j0 = min(j0, fwide[2] + 1)
        j1 = max(j1, fwide[3])

    def topk(oc):
        return oc.cell_ij[:, 0] >= first[oc.cell_ij[:, 1]] if top else np.zeros(len(oc.cell_ij), bool)

    def make():
        oc = cells(gf, j0, j1)
        keep = topk(oc)
        if fbox:
            keep |= in_box(oc.cell_ij, oc.ni, fwide)
        mesh, used = sub_cells(oc, keep)
        return OrcaCells(mesh, oc.cell_ij[keep], oc.node_ij[used], oc.rows, oc.ni, oc.folded, dict(oc.info))

    ctx = (lambda oc: ~topk(oc) & ~in_box(oc.cell_ij, oc.ni, fbox)) if fbox else None
    part = _masked(gf, name, o, 0, make, caps=True, context=ctx)
    loop, holes = (edge_loop(top_cells(part["oc"], first).mesh) if fbox else edge_loop(part["oc"].mesh)) \
        if top else (None, [])
    nj, ni = gf.lat["t"].shape[0], gf.lat["t"].shape[1] - 2
    part.update(g=gf, first=first, loop=loop, nest=nest, name=name, rows=(j0, j1), fbox=fbox,
                info={**part["info"], "orca": name, "verified": PRESETS[name].verified, "grid_rows": int(nj),
                      "cols": int(ni), "pivot": gf.pivot,
                      "lat_min": float(o["fine_lat"]) if top and first_row is None else None, "nesting": nest,
                      "first_t_rows": [int(first.min()), int(first.max())] if top else None,
                      "box_fine_f_lines": list(fbox) if fbox else None,
                      "empty_columns": int((first > gf.j_pole).sum()) if top else 0,
                      "edge_nodes": int(len(loop)) if top else 0, "holes": holes,
                      "orca_nodes": int(part["belt"].n_nodes), "orca_quads": int(part["belt"].n_quads),
                      "orca_tris": int(part["belt"].n_tris)})
    return part


def run(domain: Domain, h: SizeField, opts: dict | None = None) -> tuple[Mesh, dict]:
    from meshom.orca import PRESETS, build, extra_rows_for

    o = dict(DEFAULTS)
    o.update(opts or {})
    name = str(o["orca"])
    pr = PRESETS[name]
    extra = extra_rows_for(pr, o["lat_south"])
    g = build(name, extra)
    j0, j1, south_cap, north_cap = _rows(g, o["lat_south"], o["lat_north"])
    last = None
    ladder = o.get("transition", "fill") == "ladder"
    nest_o = dict(o["nest"]) if o.get("nest") else None
    gf_l = None
    if nest_o is not None:                                        # B176: one fine grid for the nest (and the top part)
        if o.get("fine_orca") and str(o["fine_orca"]) != str(nest_o["fine_orca"]):
            raise ValueError("m16_orca: `nest.fine_orca` and `fine_orca` must be one preset")
        gf_l = build(str(nest_o["fine_orca"]), 0)
    inner_o = dict(nest_o["inner"]) if nest_o is not None and nest_o.get("inner") else None
    gf_in = build(str(inner_o["fine_orca"]), 0) if inner_o is not None else None     # B230: the nest in the nest
    if ladder:                                                    # B114: two plain rows, the strip between them
        from meshom.orca.ladder import edge_rows

        gf_l = build(str(o["fine_orca"]), 0) if gf_l is None else gf_l
        r_c, r_f = edge_rows(g, gf_l, *map(float, o["ladder_lat"]))
        j1, north_cap = r_c, True
    if o.get("region") is not None:                               # B100: the ORCA part bounded by a line
        from meshom.orca.mesh import region_last_rows

        if o["lat_north"] is not None:
            raise ValueError("m16_orca: `region` and `lat_north` exclude each other")
        last = region_last_rows(g, j0, o["region"])
        j1, north_cap = int(last.max()), True
    nj, ni = g.lat["t"].shape[0], g.lat["t"].shape[1] - 2
    info = {"orca": name, "verified": pr.verified, "extra_south_rows": extra, "grid_rows": int(nj), "cols": int(ni),
            "pivot": g.pivot, "t_rows": [int(j0), int(j1)], "south_cap": south_cap, "north_cap": north_cap,
            "lat_first_f_row": float(g.lat["f"][j0 - 1, :-2].mean()),
            "lat_last_f_row": float(g.lat["f"][j1, :-2].mean()),
            "last_f_row_lat_min": float(g.lat["f"][j1, :-2].min()),
            "last_f_row_lat_max": float(g.lat["f"][j1, :-2].max()),
            "generator": {k: v for k, v in g.info.items() if not isinstance(v, dict)}}
    nbox = None
    if nest_o is not None:                                        # B176: the fine box (coarse F lines), the coarse box
        from meshom.orca import nest as NS
        from meshom.orca.mesh import box_cells, nesting

        nk = nesting(g, gf_l)
        H = int(nest_o.get("width") or NS.H_DEFAULT)
        if nest_o.get("box") is not None:
            c0, c1, r0, r1 = map(int, nest_o["box"])
            nbox = (c0, c1, r0 + extra, r1 + extra)               # the preset's own rows -> this grid's
        else:
            nbox = NS.box_from_lonlat(g, nest_o["lonlat"])
        cbox = NS.boxes(nbox, H, nk["k"], nk)["coarse_box"]
        info["nest_box"], info["nest_coarse_box"] = list(nbox), list(cbox)
        if inner_o is not None:                                   # B230: the inner fine box in the fine grid's F lines:
            H_in = int(inner_o.get("width") or NS.H_DEFAULT)      # the nest's fine box less the inner frame's width
            k, b_i, b_j = nk["k"], nk["col_offset"], nk["row_offset"]
            w = H_in // 3
            ibox_c = (k * nbox[0] + b_i, k * nbox[1] + b_i, k * nbox[2] + b_j, k * nbox[3] + b_j)
            ibox = (ibox_c[0] + w, ibox_c[1] - w, ibox_c[2] + w, ibox_c[3] - w)
            info["nest_inner_box"], info["nest_inner_coarse_box"] = list(ibox), list(ibox_c)
        from meshom.orca.mesh import in_box

        r = NS.CONTEXT_CELLS                                     # the ring inside the box: context of the clean-up
        inner = (cbox[0] + r, cbox[1] - r, cbox[2] + r, cbox[3] - r)
        part = _masked(g, name, o, extra, lambda: box_cells(_part_cells(g, j0, j1, last), inner, False), caps=True,
                       context=lambda oc: in_box(oc.cell_ij, oc.ni, cbox))
    else:
        part = _masked(g, name, o, extra, lambda: _part_cells(g, j0, j1, last), caps=south_cap or north_cap)
    info.update(part["info"])
    belt, wetmap, wet, mapped, present = part["belt"], part["wetmap"], part["wet"], part["mapped"], part["present"]
    node_level, z = part["node_level"], part["z"]
    fine = fine_in = None
    coarse_belt = belt

    def add_part(pt, key):                                        # one z for all parts (offset when they differ)
        nonlocal node_level, z, belt
        if node_level is not None:
            fz, fl = np.asarray(pt["z"], float), np.asarray(pt["node_level"])
            if not (len(fz) == len(z) and np.array_equal(fz, z)):
                fl = fl + len(z)
                z = np.r_[z, fz]
                info[key]["levels_offset"] = True
            node_level = np.r_[node_level, fl]
        n0 = belt.n_nodes
        belt = _union(belt, pt["belt"])
        return n0

    fbox_l = nbox if inner_o is None else None                    # B230: the inner frame and ORCA fill the fine box
    if o.get("fine_orca") or (nest_o is not None and inner_o is None):   # B108: the refined part (B176: the box)
        top = bool(o.get("fine_orca"))
        if top and not north_cap:
            raise ValueError("m16_orca: `fine_orca` needs a north edge of the coarse part (`region` or `lat_north`)")
        fo = o if top else {**o, "fine_orca": nest_o["fine_orca"]}
        fine = _fine_part(g, fo, gf_l, r_f + 1, fbox_l) if ladder else _fine_part(g, fo, gf_l, None, fbox_l, top)
        info["fine"] = fine["info"]
        add_part(fine, "fine")
    if inner_o is not None:                                       # B230: the innermost preset's cells in the inner box
        io = {**o, "orca": str(nest_o["fine_orca"]), "fine_orca": str(inner_o["fine_orca"])}
        fine_in = _fine_part(gf_l, io, gf_in, None, ibox, top=False)
        info["fine_inner"] = fine_in["info"]
        inner_n0 = add_part(fine_in, "fine_inner")
    mesh = belt
    joints = {}
    copts = dict(o["cap_opts"])
    if "size_tolerance" not in copts and o.get("size_tolerance") is not None:
        copts["size_tolerance"] = o["size_tolerance"]
    if o.get("land") is not None and "land" not in copts:
        from meshom.methods import own_optimize

        if own_optimize(o["cap_method"]) is not None:           # the m14 methods: water-aware placement (B70)
            copts["land"] = o["land"]
    for side, on, frow in (("north", north_cap, j1), ("south", south_cap, j0 - 1)):
        if not on:
            continue
        from meshom.geo import lonlat_to_xyz

        loop = lonlat_to_xyz(g.lon["f"][frow, :-2], g.lat["f"][frow, :-2])
        if side == "north" and last is not None:                  # B100: the staircase of the region's edge
            from meshom.orca.mesh import staircase

            loop = staircase(g, last)
        if side == "south":
            loop = loop[::-1]                                     # the cap (south) on the loop's left
        loops = [loop] if side == "south" or fine is None or fine["loop"] is None else [loop, fine["loop"]]   # B108
        if side == "north" and ladder:                            # B114-B116: the structured strip
            from meshom.orca.ladder import RELAX, THETA_DEG
            from meshom.orca.ladder import build as ladder_build

            th, rw = o.get("ladder_theta_deg"), o.get("ladder_relax")
            cap, _, linfo = ladder_build(g, fine["g"], r_c, r_f, fine["nest"], THETA_DEG if th is None else float(th),
                                         relax_w=RELAX if rw is None else (tuple(rw) if rw else None))
            info["ladder"] = linfo
        else:
            cap = METHODS[o["cap_method"]](Domain(loops, f"orca_cap_{side}", [True] * len(loops)), h, copts)
        tree = cKDTree(np.vstack(loops))
        dd, _ = tree.query(mesh.points, distance_upper_bound=JOINT_TOL)
        row_nodes = np.flatnonzero(np.isfinite(dd))
        mesh, joint = merge(mesh, row_nodes, cap)
        joint.update(cap_nodes=int(cap.n_nodes), cap_tris=int(cap.n_tris), cap_quads=int(cap.n_quads),
                     row_nodes=int(sum(len(lp) for lp in loops)), row_nodes_in_orca=int(len(row_nodes)))
        if len(loops) > 1:
            joint.update(coarse_edge_nodes=int(len(loop)), fine_edge_nodes=int(len(fine["loop"])))
        joints[side] = joint
        if not joint["conforming"]:
            raise RuntimeError(f"m16_orca: the {side} cap does not conform to the ORCA row: {joint}")
    if nest_o is not None:                                        # B172-B176: the frame between the two boxes
        from meshom.orca.nest import RELAX as NRELAX
        from meshom.orca.nest import build as nest_build

        th, rw = nest_o.get("theta_deg"), nest_o.get("relax")
        from meshom.orca.ladder import THETA_DEG

        nest_rec = {}
        frames = [("nest", g, gf_l, nbox, nk, H, th, rw)]
        if inner_o is not None:                                   # B230: the inner frame, merged after the first
            frames.append(("nest_inner", gf_l, gf_in, ibox, fine_in["nest"], H_in, inner_o.get("theta_deg"),
                           inner_o.get("relax")))             # (its ring 0 is the first frame's ring n)
        for key, gc_, gf_, box_, nk_, H_, th, rw in frames:
            cap, ft, ninfo = nest_build(gc_, gf_, box_, nk_, H_, THETA_DEG if th is None else float(th),
                                        NRELAX if rw is None else (tuple(rw) if rw else None))
            info[key] = ninfo
            nest_rec.update({f"{key}_points": cap.points, f"{key}_xy": np.c_[ft.x, ft.y], f"{key}_ring": ft.ring,
                             f"{key}_side": ft.side})
            edge = cap.points[(ft.ring == 0) | (ft.ring == ninfo["n"])]
            dd, _ = cKDTree(edge).query(mesh.points, distance_upper_bound=JOINT_TOL)
            row_nodes = np.flatnonzero(np.isfinite(dd))
            mesh, joint = merge(mesh, row_nodes, cap)
            joint.update(cap_nodes=int(cap.n_nodes), cap_tris=int(cap.n_tris), cap_quads=int(cap.n_quads),
                         row_nodes=int(len(edge)), row_nodes_in_orca=int(len(row_nodes)),
                         coarse_edge_nodes=int(ninfo["ring0_nodes"]), fine_edge_nodes=int(ninfo["ringn_nodes"]))
            joints[key] = joint
            if not joint["conforming"]:
                raise RuntimeError(f"m16_orca: the {key} frame does not conform to the ORCA parts: {joint}")
    info["joints"] = joints
    info["orca_nodes"] = int(belt.n_nodes)
    info["orca_quads"] = int(belt.n_quads)
    info["orca_tris"] = int(belt.n_tris)
    if fine is not None or fine_in is not None:
        info["coarse_nodes"] = int(coarse_belt.n_nodes)
    _STATE["belt"] = (belt.points, belt.quads, belt.tris)
    d = _record_dir()
    if d is not None:
        (d / "orca.json").write_text(json.dumps(info, indent=1))
        rs = slice(j0, j1 + 1)
        extra_arrays = {}
        if fine is not None:                                      # B108: the refined part (its own rows)
            gf, fr = fine["g"], slice(fine["rows"][0], fine["rows"][1] + 1)
            extra_arrays = dict(fine_orca=np.array(fine["name"]),
                                fine_first=fine["first"] if fine["first"] is not None else np.zeros(0, np.int64),
                                fine_tlon=gf.lon["t"][fr, :-2], fine_tlat=gf.lat["t"][fr, :-2],
                                fine_t_rows=np.array([fr.start, fr.stop - 1]),
                                fine_edge=fine["loop"] if fine["loop"] is not None else np.zeros((0, 3)),
                                fine_wetmap=fine["wetmap"][fr] if fine["wetmap"] is not None else np.zeros(0, np.int8),
                                fine_nodes=np.arange(coarse_belt.n_nodes, coarse_belt.n_nodes + fine["belt"].n_nodes),
                                fine_quads=fine["belt"].quads + coarse_belt.n_nodes,
                                fine_tris=fine["belt"].tris + coarse_belt.n_nodes)
        if fine_in is not None:                                   # B230: the innermost part (belt node ids)
            extra_arrays.update(inner_orca=np.array(fine_in["name"]),
                                inner_nodes=np.arange(inner_n0, inner_n0 + fine_in["belt"].n_nodes),
                                inner_quads=fine_in["belt"].quads + inner_n0,
                                inner_tris=fine_in["belt"].tris + inner_n0,
                                nest_inner_box=np.array(ibox), nest_inner_coarse_box=np.array(ibox_c))
        if nbox is not None:                                      # B176: the boxes (coarse F lines, this grid's rows)
            extra_arrays.update(nest_box=np.array(nbox), nest_coarse_box=np.array(cbox), **nest_rec)
        np.savez(d / "orca_belt.npz", points=belt.points, quads=belt.quads, tris=belt.tris,
                 lon=g.lon["f"][j0 - 1:j1 + 1, :-2], lat=g.lat["f"][j0 - 1:j1 + 1, :-2],
                 tlon=g.lon["t"][rs, :-2], tlat=g.lat["t"][rs, :-2], t_rows=np.array([j0, j1]),
                 wetmap=wetmap[rs] if wetmap is not None else np.zeros(0, np.int8),
                 nemo_wet=wet[rs] if wet is not None else np.zeros(0, bool),
                 nemo_mapped=mapped[rs] if mapped is not None else np.zeros(0, bool),
                 present=present[rs], pivot=np.array(g.pivot),
                 last_row=last if last is not None else np.zeros(0, np.int64),
                 node_level=node_level if node_level is not None else np.zeros(0, np.int64),
                 z=z if z is not None else np.zeros(0), **extra_arrays)
    return mesh, info


@register()
def generate(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    return run(domain, h, opts)[0]
