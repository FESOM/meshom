"""The insert pipeline: region -> joins -> block -> merge -> ring cleaning -> scores (insert_window, on a host
window in memory), and the run-folder wrapper (run: window cut, splice, the optional smoothing of the spliced insert,
depth, model files, record, maps).
"""
import dataclasses
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from meshom.geo import R_EARTH_M, arc_length, lonlat_to_xyz, xyz_to_lonlat
from meshom.insert import assemble, block, measure, region
from meshom.insert.geom import Axis, Chart, Polyline, smooth_bank
from meshom.insert.host import Window, coast_neighbours, cut_window, graph, splice
from meshom.insert.region import InsertError
from meshom.mesh import Mesh

WINDOW_PAD_H = 10.0          # B159: the window = the path, 1.5 x the reach and this many host cells around it
PROBE_KM = 60.0              # the probe window (host cell size near the path) reaches this far beyond the path


@dataclass
class Spec:
    name: str
    path: np.ndarray                         # (K,3) unit vectors, basin A -> basin B
    cells: int | None = None
    h_fine_m: float | None = None
    reach_m: float | None = None
    mouth_a: np.ndarray | None = None        # (K,3) hints
    mouth_b: np.ndarray | None = None
    force: bool = False
    stages: str = assemble.STAGES
    ring_rows: int = assemble.RING_ROWS
    bank_sigma_h: float = assemble.BANK_SIGMA_H
    bank_cap_h: float = assemble.COAST_CAP_H
    collar_rings: int | None = None          # B192 / B193: None = assemble.COLLAR_RINGS, 0 on a structured host
    clean_h: str = "insert"
    h_min_m: float | None = None             # B165: the size floor (None: the host mesh's smallest cell in run())
    coast: str = "real"                      # B164: banks on the "real" coast (the run's) or the "host" staircase
    coast_source: str = ""                   # B164: another coast shapefile for the banks ("" = the run's)
    bank_sigma_cells: float = 1.5            # B164: real-bank smoothing in local insert cells
    bays: bool | str = "auto"                # B166 / B190: bays as blocks of their own: "auto" (large), True, False
    kind: str = "strait"                     # B181 / B182: "strait", "strip" or "corridor"
    width_m: float | None = None             # strip: coast to offshore join; corridor: full width
    band_m: float | None = None              # strip / corridor: the fine band's width (None: what the room leaves)
    ljoins: bool = True                      # B183: L-joins round a cape when no straight join passes
    rows: str = "harmonic"                   # B195-B198: a strip's rows: "harmonic" (a harmonic map) | "offset" (B181)
    coast_rounds: int = 3                    # B198: rounds of extra coast smoothing round bad cells (harmonic strip)
    coast_shift_cells: float = 1.0           # B198: the extra smoothing moves the bank at most this many fine cells
    cells_a: int | None = None               # B200: the join count of side A / B (None: any, today's rules)
    cells_b: int | None = None
    growth_a: float | None = None            # B201: the size slope along the block on side A / B (None: 0.12)
    growth_b: float | None = None
    bank_land_cells: float = 0.0             # B202: the banks may move into land by this many local insert cells
    fill: list = field(default_factory=list)  # B204: fill lines, each (K,3) unit vectors: the bank follows them
    fill_rows: int | None = None             # B204: rows of the strip behind a fill line (None: by its depth)
    corner_steps: bool = False               # B206: a hinted join cuts every convex step of its staircase
    add_columns: bool = False                # B207: target lowered where a block cell > 1.1 x host; counts not capped
    corner_split: bool = False               # B208: corner triangles split at the diagonal's midpoint (a block node)
    join_margin: float = 1.0                 # B209: the size target's cap at the joins in host cell sizes
    fix_rings: int = 0                       # B210: stage a on the insert and this many element rings after the splice
    depth_at: str = ""                       # B211: the insert's own nodes' depth rule ("": the run's)
    depth_smooth: int | None = None          # B211: smoothing passes over the insert's nodes (None: the run's)
    strait_floor: bool = True                # B211: the run's strait depth floors on the insert's nodes
    sill_level: str = ""                     # B212: "shallower": the sill's level rounded up; B215: "grid"
    depth_channel: str = ""                  # B213: "cells" | "near": the grid's deepest channel raises the insert
    depth_quantile: float = 0.0              # B214: the node box's quantile instead of its mean (0: the mean)


@dataclass
class Result:
    ok: bool
    record: dict
    window: Mesh | None = None               # the cleaned window
    merged: Mesh | None = None               # the window right after the merge (before cleaning)
    host_window: Window | None = None
    chart: Chart | None = None
    profile: object = None
    joins: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    outline: np.ndarray | None = None
    h_ins: object = None
    ring_case: object = None
    real: object = None                      # the RealCoast (B164) or None
    ring_out: object = None                  # the cleaned case's mesh (B192)


def _ll(a) -> np.ndarray | None:
    if a is None or len(a) == 0:
        return None
    p = np.asarray(a, dtype=np.float64)
    return lonlat_to_xyz(p[:, 0], p[:, 1])


def spec_from(cfg_entry, cfg=None) -> Spec:
    """A Spec from an InsertCfg (an [[inserts]] entry or the CLI's inline fields); the strait's path from the run's
    [[straits]] or the built-in list (meshom.straits.BUILTIN), else the mouths' midpoints."""
    from meshom import straits as stm

    e = cfg_entry
    path = _ll(e.path)
    if e.kind != "strait":
        if path is None or e.width_km is None:
            raise InsertError(f"an insert of kind {e.kind!r} needs a path and width_km")
    if path is None and e.strait:
        own = [s for s in (cfg.straits if cfg is not None else []) if s.name == e.strait and s.path]
        lists = [cfg.coast.straits] if cfg is not None and cfg.coast.straits != "none" else []
        entries = [d for nm in lists + ["fesom2_mesh_gen"] for d in stm.builtin(nm) if d["name"] == e.strait]
        p = own[0].path if own else (entries[0]["path"] if entries else None)
        if p is None:
            raise InsertError(f"strait {e.strait!r}: not in the run's [[straits]] nor in the built-in list")
        path = _ll(p)
    ma, mb = _ll(e.mouth_a), _ll(e.mouth_b)
    if path is None:
        if ma is None or mb is None:
            raise InsertError("an insert needs a strait, a path or both mouths")
        path = np.array([ma.mean(0), mb.mean(0)])
        path /= np.linalg.norm(path, axis=1, keepdims=True)
    name = e.name or (e.strait.lower() if e.strait else "strait")
    return Spec(name, path, e.cells, e.h_fine_m, e.reach_km * 1e3 if e.reach_km else None, ma, mb, e.force,
                e.stages, e.ring_rows, e.bank_sigma_h, e.bank_cap_h, e.collar_rings, e.clean_h, e.h_min_m, e.coast,
                e.coast_source, e.bank_sigma_cells, e.bays, e.kind, e.width_km * 1e3 if e.width_km else None,
                e.band_km * 1e3 if e.band_km is not None else None, e.ljoins, e.rows, e.coast_rounds,
                e.coast_shift_cells, e.cells_a, e.cells_b, e.growth_a, e.growth_b, e.bank_land_cells,
                [_ll(f) for f in e.fill], e.fill_rows, e.corner_steps, e.add_columns, e.corner_split, e.join_margin,
                e.fix_rings, e.depth_at, e.depth_smooth, e.strait_floor, e.sill_level, e.depth_channel,
                e.depth_quantile)


def _path_frame(path: np.ndarray):
    c = path.mean(0)
    c /= np.linalg.norm(c)
    half = float(R_EARTH_M * np.max(arc_length(path, np.broadcast_to(c, path.shape))))
    return c, half


def host_size(mesh: Mesh, near_xyz: np.ndarray, radius_m: float) -> float:
    """Median great-circle edge length of the host elements within radius_m of the points near_xyz."""
    from scipy.spatial import cKDTree

    pts = mesh.points
    d, _ = cKDTree(near_xyz).query(pts, distance_upper_bound=2 * np.sin(0.5 * radius_m / R_EARTH_M))
    ok = np.isfinite(d)
    cells = [c for c in (mesh.tris, mesh.quads) if len(c)]
    L = []
    for c in cells:
        sel = c[ok[c].all(1)]
        if len(sel):
            L.append(R_EARTH_M * arc_length(pts[sel[:, 0]], pts[sel[:, 1]]))
    if not L:
        raise InsertError("no host elements near the strait's path")
    return float(np.median(np.concatenate(L)))


def insert_window(win: Window, spec: Spec, h_run, tol=0.20, gates=None, log=print, real=None,
                  h_min: float | None = None, orca: bool | None = None) -> Result:
    """The insert in a host window (module docstring). real: the run's coast loops ((K,3) unit vectors, water on the
    left) for real-coast banks (B164; None or spec.coast == "host": the host's coast as in step 1). h_min: the size
    floor (m; default spec.h_min_m, else none). Raises InsertError when the region is not a single disc or the neck
    holds fewer than 2 cells of the floor; refusals of the rules come back in the record."""
    from meshom import straits as stm
    from meshom.insert import network
    from meshom.insert import realcoast as rcm

    t0 = time.perf_counter()
    rec = {"name": spec.name, "timing": {}}
    mesh = win.mesh
    c, half = _path_frame(spec.path)
    chart = Chart(c)
    xy = chart.fwd(mesh.points)
    seg = xy[win.coast_edges]
    path_xy = chart.fwd(spec.path)
    h_host = host_size(mesh, stm.densify(spec.path, 1000.0), max(half, 3 * 1000.0))
    h_min = float(spec.h_min_m if spec.h_min_m is not None else (h_min or 0.0))
    reach = spec.reach_m or region.REACH_H * h_host
    axis = Axis(path_xy, reach + 4 * h_host, 0.25 * h_host)
    tree = stm._elem_tree(mesh)
    in_water = lambda o: stm.locate(mesh, chart.inv(o), tree) >= 0          # noqa: E731
    rc = None
    if real is not None and spec.coast == "real":
        win_r = float(np.max(np.linalg.norm(xy, axis=1))) / chart.scale(np.zeros((1, 2)))[0]
        rc = rcm.RealCoast.from_loops(real, chart, c, win_r + 2 * h_host, spec.coast_source or "run")
        if not rc.pieces:
            rc = None
    rules = {"A": None, "B": None}                     # B200 / B201: per-side join counts and size slopes
    for sd, nj, gr in (("A", spec.cells_a, spec.growth_a), ("B", spec.cells_b, spec.growth_b)):
        if nj is not None or gr is not None:
            rules[sd] = region.SideRule(nj, gr, spec.add_columns)
    g_side = tuple(r.growth if r is not None and r.growth is not None else block.G_LAYOUT
                   for r in (rules["A"], rules["B"]))
    if rc is not None:
        t = time.perf_counter()
        prof1 = region.width_profile(axis, rc.segments("raw"), in_water, h_host, spec.cells, spec.h_fine_m, h_min)
        ht1 = np.where(prof1.s < prof1.s0, region.side_target(prof1, prof1.n_cells, rules["A"]),
                       region.side_target(prof1, prof1.n_cells, rules["B"]))

        def h_loc(q):
            s = axis.project(q)
            k = np.clip(np.searchsorted(prof1.s, s), 0, len(prof1.s) - 1)
            lat = np.linalg.norm(q - prof1.xy[k], axis=1)
            W = np.where(np.isfinite(prof1.W[k]), prof1.W[k], 0.0)
            g = np.where(prof1.s[k] < prof1.s0, g_side[0], g_side[1])
            h = ht1[k] + g * np.maximum(0.0, lat - 0.5 * W)
            return np.clip(np.where(np.isfinite(h), h, h_host), prof1.h_fine, h_host)

        def w_loc(q):
            s = axis.project(q)
            k = np.clip(np.searchsorted(prof1.s, s), 0, len(prof1.s) - 1)
            lat = np.linalg.norm(q - prof1.xy[k], axis=1)
            W = prof1.W[k]
            return np.where(np.isfinite(W) & (lat <= W), W, np.inf)

        rc.prepare(h_loc, w_loc, prof1.h_fine, spec.bank_sigma_cells, spec.bays, land_cells=spec.bank_land_cells,
                   fill=[chart.fwd(f) for f in spec.fill])
        rec["real_coast"] = {"source": rc.source, "pieces": len(rc.pieces),
                             "raw_neck_km": float(prof1.W[prof1.i0] / 1e3),
                             "bays_found": [b.record() | {"mouth_lonlat": [round(float(v[0]), 4) for v in xyz_to_lonlat(
                                 chart.inv((0.5 * (b.a + b.b))[None]))]} for b in rc.bays],
                             "prepare_s": time.perf_counter() - t,
                             "land": rc.land or None}
        prof_seg = rc.segments("main")
    else:
        prof_seg = seg
    prof = region.width_profile(axis, prof_seg, in_water, h_host, spec.cells, spec.h_fine_m, h_min)
    rec["profile"] = {"neck_width_km": float(prof.W[prof.i0] / 1e3), "cells": prof.n_cells,
                      "h_fine_km": prof.h_fine / 1e3, "h_host_km": h_host / 1e3, "reach_km": reach / 1e3,
                      "h_min_km": h_min / 1e3, "cells_max": prof.n_max if h_min > 0 else None,
                      "floor": prof.floor_note or None, "coast": "real" if rc is not None else "host",
                      "neck_lonlat": [float(v[0]) for v in xyz_to_lonlat(chart.inv(prof.xy[prof.i0][None]))]}
    log(f"[insert {spec.name}] neck {prof.W[prof.i0] / 1e3:.2f} km ({rec['profile']['coast']} coast), "
        f"{prof.n_cells} cells -> fine {prof.h_fine / 1e3:.2f} km; host {h_host / 1e3:.2f} km; floor "
        f"{h_min / 1e3:.2f} km; reach {reach / 1e3:.0f} km", flush=True)
    if prof.floor_note:
        log(f"[insert {spec.name}] FLOOR: {prof.floor_note}", flush=True)
    res = Result(False, rec, host_window=win, chart=chart, profile=prof)
    res.real = rc
    if prof.h_fine * region.RATIO > h_host:
        fix = (f"more cells across (cells >= {int(np.ceil(region.RATIO * prof.W[prof.i0] / h_host))})")
        if prof.floor_note:
            fix = (f"none within the size floor h_min_m {h_min / 1e3:.2f} km (the neck holds at most {prof.n_max} "
                   f"cells of it); a smaller h_min_m")
        rec["refused"] = [f"no zoom: the fine size {prof.h_fine / 1e3:.2f} km (neck {prof.W[prof.i0] / 1e3:.1f} km / "
                          f"{prof.n_cells} cells) is not finer than the host's {h_host / 1e3:.2f} km by the ratio "
                          f"{region.RATIO}; fix: {fix}"]
        log(f"[insert {spec.name}] REFUSED {rec['refused'][0]}", flush=True)
        return res
    g = graph(mesh)
    ch = region.Chains(g, xy, coast_neighbours(win.coast_edges, mesh.n_nodes), mesh.quads)
    rings, host_fixed, rec["host"] = assemble.collar_of(spec.collar_rings, g, orca)
    eidx = {(int(a), int(b)): k for k, (a, b) in enumerate(g.edges)}

    def sphere_len(chain):
        return float(sum(g.length_m[eidx[(min(a, b), max(a, b))]] for a, b in zip(chain[:-1], chain[1:])))

    coast_nodes = np.unique(win.coast_edges)

    def bank_fn(bxy):                                  # the host bank, smoothed and capped (step 1, B161)
        hb = float(np.median(np.linalg.norm(np.diff(bxy, axis=0), axis=1)))
        sig = spec.bank_sigma_h * hb
        cap = lambda p: spec.bank_cap_h * np.asarray(h_run(chart.inv(p)))                      # noqa: E731
        return smooth_bank(bxy, sig, cap, max(min(0.25 * prof.h_fine, 0.25 * sig if sig > 0 else 1e9), 1.0))

    bank_dir = None
    if rc is not None:
        def bank_dir(p, length):                       # B164: the corner rule against the real (main) coast
            pts = rc.walk(p, length)
            if not pts:
                return None
            dn = [abs(float(axis.project(q[None])[0]) - prof.s0) for q in pts]
            return pts[int(np.argmin(dn))]

    square = None
    if rc is not None and spec.bank_land_cells > 0:
        def square(p, u, hj):                          # B202: the squared bank end's depth into land / allowance
            L = rcm.BLEND_H * hj
            pts = rc.walk(p, L)
            if not pts:
                return np.inf
            dn = [abs(float(axis.project(q[None])[0]) - prof.s0) for q in pts]
            q1 = pts[int(np.argmin(dn))]
            if np.dot(u, q1 - p) <= 0:                 # an obtuse end is not squared (it would go into the water)
                return np.inf
            pts2 = rc.walk(p, L + 0.25 * hj)
            q2 = pts2[int(np.argmin([abs(float(axis.project(q[None])[0]) - prof.s0) for q in pts2]))]
            curve = rcm.hermite_end(p, u, q1, q2 - q1)
            dep = rc.depth(curve, "base")
            return float(np.max(dep / np.maximum(spec.bank_land_cells * h_loc(curve), 1.0)))

    bay_hit = None
    if rc is not None and any(b.meshed for b in rc.bays):
        from meshom.insert.geom import point_in_polygon

        polys = [b.arc for b in rc.bays if b.meshed]

        def bay_hit(cxy):                              # B166: the chain crosses a bay's mouth or ends inside the bay
            for poly in polys:
                if point_in_polygon(cxy[[0, -1]], poly).any():
                    return True
                if region.first_crossing(cxy, np.stack([poly[:1], poly[-1:]], axis=1))[0] >= 0:
                    return True
            return False

    joins, cands, refuse, by_side = [], [], [], {}
    for side, hint in (("A", spec.mouth_a), ("B", spec.mouth_b)):
        if hint is not None:
            j = region.hinted(side, chart.fwd(hint), prof, axis, coast_nodes, ch, sphere_len, bank_fn, bank_dir,
                              bay_hit, rule=rules[side], square=square)
            cs = [j]
        else:
            cs = region.candidates(side, prof, axis, seg, win.coast_edges, ch, sphere_len, h_host, reach, bank_fn,
                                   bank_dir, bay_hit, rule=rules[side], square=square)
            j = region.choose(cs, h_host, False, rec["host"]["structured"])
            if j is None and spec.ljoins:                       # B183: L-joins when no plain join passes
                cl = region.candidates_l(side, prof, axis, seg, win.coast_edges, ch, sphere_len, h_host, reach,
                                         bank_fn, bank_dir, bay_hit, rule=rules[side], square=square)
                jl = region.choose(cl, h_host, False, rec["host"]["structured"])
                cs = cs + cl
                if jl is not None:
                    j = jl
            if j is None and spec.force:
                j = region.choose(cs, h_host, True)
        cands += cs
        by_side[side] = cs
        if j is None or (not j.ok and not spec.force):                 # a hint picks the join, never skips the rules
            refuse.append(region.refusal(side, cs, prof, reach, h_host))
        joins.append(j)
    rec["candidates"] = [c.record() for c in cands]
    rec["joins"] = [j.record() if j is not None else None for j in joins]
    rec["timing"]["region_s"] = time.perf_counter() - t0
    res = Result(False, rec, host_window=win, chart=chart, profile=prof, joins=joins, candidates=cands)
    res.real = rc
    if refuse:
        refuse.append("advice: " + region.advise(by_side, prof, reach))      # B184: both sides at once
        rec["refused"] = refuse
        for r in refuse:
            log(f"[insert {spec.name}] REFUSED {r}", flush=True)
        return res
    jA, jB = joins
    for j in joins:
        log(f"[insert {spec.name}] join {j.side}: {j.shape} {j.d_m / 1e3:.1f} km out, tier {j.tier} (zig {j.zig:.2f}), "
            f"{j.n} edges of {j.h_join_m / 1e3:.2f} km, corners {j.corner_deg[0]:.0f} / {j.corner_deg[1]:.0f} deg, "
            f"room {j.room:.2f}, funnel {j.funnel:.2f}{', FAILS ' + str(j.fails) if j.fails else ''}", flush=True)
    cA, cB = jA.nodes, jB.nodes
    seed = int(stm.locate(mesh, chart.inv(prof.xy[prof.i0][None]), tree)[0])
    reg = assemble.region_between(mesh, seed, [cA, cB], win.kind)
    bS_ids, bN_ids = assemble.banks(mesh, reg, cA, cB)
    banks_xy, bank_src, base_xy, squared = [], [], [], []
    inward = ((xy[cA[1]] - xy[cA[0]], xy[cB[1]] - xy[cB[0]]), (xy[cA[-2]] - xy[cA[-1]], xy[cB[-2]] - xy[cB[-1]]))
    for ids, (ua, ub) in zip((bS_ids, bN_ids), inward):
        b = None
        if rc is not None:
            arc, why = rc.arc(xy[ids[0]], xy[ids[-1]], guide=xy[ids])
            host_len = Polyline(xy[ids]).L
            if arc is not None and Polyline(arc).L > 3.0 * host_len + 4 * h_host:
                arc, why = None, "the real coast between the ends is far longer than the host's (around an island)"
            if arc is not None:
                la, lb = rcm.BLEND_H * jA.h_join_m, rcm.BLEND_H * jB.h_join_m
                if spec.bank_land_cells > 0:           # B202: square ends, today's bank kept for the measures
                    arc = trim_to_joins(arc, xy[cA], xy[cB])
                    b, sq = rcm.square_ends(arc, xy[ids[0]], xy[ids[-1]], ua, ub, la, lb)
                    squared.append(sq)
                    a0, _ = rc.arc(xy[ids[0]], xy[ids[-1]], guide=xy[ids], which="base")
                    base_xy.append(rcm.blend_ends(a0, xy[ids[0]], xy[ids[-1]], la, lb) if a0 is not None else None)
                else:
                    b = rcm.blend_ends(arc, xy[ids[0]], xy[ids[-1]], la, lb)
                bank_src.append("real")
            else:
                bank_src.append(f"host ({why})")
        if b is None:
            b = bank_fn(xy[ids])
            if rc is None:
                bank_src.append("host")
        banks_xy.append(b)
    rec["banks"] = bank_src
    if len(base_xy) == 2 and all(b is not None for b in base_xy):
        rec["bank_land"] = land_record(rc, chart, xy[cA], xy[cB], banks_xy, base_xy) | {"squared": squared}
        log(f"[insert {spec.name}] banks into land: max {rec['bank_land']['max_km']:.2f} km, water added "
            f"{rec['bank_land']['added_km2']:.1f} km^2 (removed {rec['bank_land']['removed_km2']:.1f} km^2)",
            flush=True)
    bS, bN = Polyline(banks_xy[0]), Polyline(banks_xy[1])
    t = time.perf_counter()
    blk = block.build(xy[cA], xy[cB], bS, bN, prof.n_cells, h_min=h_min, cell_floor=h_min, scale=chart.scale,
                      corners=(jA.shape in ("L", "hint"), jB.shape in ("L", "hint")),   # B180 / B203: at a bend
                      growth=g_side if any(r is not None for r in rules.values()) else block.G_LAYOUT,
                      monotone=(spec.cells_a is not None, spec.cells_b is not None),       # B200, B201
                      steps=(spec.corner_steps and jA.shape == "hint", spec.corner_steps and jB.shape == "hint"),
                      add_columns=spec.add_columns,                                        # B207
                      split=spec.corner_split, margin=spec.join_margin)                    # B208, B209
    rec["timing"]["block_s"] = time.perf_counter() - t
    rec["block"] = blk.info
    log(f"[insert {spec.name}] block: {blk.M} rows, counts {blk.N.min()}-{blk.N.max()}, {len(blk.tris)} Y-splits, "
        f"fine lattice {blk.info['fine']} ({rec['timing']['block_s']:.0f} s)", flush=True)
    if rc is not None and any(b.meshed for b in rc.bays):
        net = network.build(blk, bS, bN, rc.bays, chart.scale, h_min, spec.ring_rows, log, fill_rows=spec.fill_rows)
    else:
        net = network.from_block(blk, spec.ring_rows, chart.scale)
    rec["bays"] = net.bays
    rec["predict"] = {"fine_inverted": net.fine_inverted}
    if spec.stages:
        rec["released_cells"] = network.release_bad(net)
    new_xyz = chart.inv(net.xy)
    m = assemble.merge_net(mesh, reg, cA, cB, net, new_xyz)
    rec["merge"] = {"removed_host_elems": int(reg.sum()), "block_quads": int(m.block_q.sum()),
                    "block_tris": int(m.block_t.sum()), "inverted_block_cells": m.inverted}
    sp, sh = net.samples
    h_ins = assemble.InsertH(h_run, chart.inv(sp), sh, max(g_side))     # B201: the faster side's slope
    bsx = net.bank_s if net.bank_s is not None else banks_xy[0]
    bnx = net.bank_n if net.bank_n is not None else banks_xy[1]
    outline = np.vstack([xy[cA], bnx, xy[cB][::-1], bsx[::-1]])
    bend = []                                          # B169: how tightly the banks bend per local cell (predictor)
    for b in (banks_xy[0], banks_xy[1]):
        hb = np.asarray(h_ins.layer(chart.inv(b)))
        step = np.median(np.linalg.norm(np.diff(b, axis=0), axis=1))
        half = np.maximum(1, np.round(0.5 * hb / max(step, 1.0)).astype(np.int64))
        bend.append(rcm._curvature(b, False, half) * hb)
    bend = np.concatenate(bend)
    rec["predict"]["bank_bend_max"] = float(bend.max())
    rec["predict"]["bank_bend_p99"] = float(np.percentile(bend, 99))
    if rc is not None:
        isl = rcm.islands_in(rc, outline)
        rec["islands_inside"] = [{"area_km2": round(a, 3), "lonlat": [float(v[0]) for v in
                                                                      xyz_to_lonlat(chart.inv(p[None]))]}
                                 for a, p in isl]
        if isl:
            log(f"[insert {spec.name}] {len(isl)} island(s) of the real coast inside the insert are not kept (step 2 "
                f"builds no holes): {rec['islands_inside']}", flush=True)
    cut_xyz = mesh.points[win.kind == 2]
    keys, _ = assemble.coast_keys_of(m.mesh, cut_xyz)
    host_seg = np.stack([mesh.points[win.coast_edges[:, 0]], mesh.points[win.coast_edges[:, 1]]], axis=1)
    t = time.perf_counter()
    ref_host = rc is None and spec.bank_cap_h <= assemble.COAST_RULE_H
    out, crec, case, ring_out = assemble.clean(m, keys, h_ins, tol, gates, spec.stages, host_seg,
                                               f"insert_{spec.name}", opts={"d": {}, "b": {}, "e": {}, "a": {}},
                                               verbose=True, ref_host=ref_host,
                                               rings=rings, clean_h=spec.clean_h, **assemble.collar_opts(host_fixed))
    rec["timing"]["clean_s"] = time.perf_counter() - t
    rec["clean"] = crec
    rec["collar"] = collar_record(mesh, reg, m, case, ring_out, out, h_host, rings, host_fixed)
    try:
        from meshom.meshfix.score import score as ms

        rec["ring_score"] = {"before": _small(ms(case, case.mesh)), "after": _small(ms(case, ring_out))}
    except Exception as ex:                                  # the ring score is supplementary; say why it is missing
        rec["ring_score"] = {"error": f"{type(ex).__name__}: {ex}"}
    chains_xy = [xy[cA], xy[cB]]
    hj = [jA.h_join_m, jB.h_join_m]
    sc = {}
    real_seg = rc.segments("raw") if rc is not None else None
    ins_nodes = {nm: measure.insert_coast_nodes(mm, m.mesh, m.host_node, chart)          # B168: by the merged banks
                 for nm, mm in (("merged", m.mesh), ("final", out))}
    for nm, mm in (("host", mesh), ("merged", m.mesh), ("final", out)):
        z = measure.zones(mm, chart, outline, chains_xy, hj, h_host)
        sc[nm] = measure.metrics(mm, z, chart, seg, h_run, h_ins)
        if real_seg is not None:
            sc[nm]["coast_real"] = measure.coast_real(mm, z, chart, real_seg, seg, h_ins, h_run, outline,
                                                      np.vstack(chains_xy), insert_nodes=ins_nodes.get(nm))
    rec["scores"] = sc
    rec["validity_window"] = measure.validity(out)
    v = rec["validity_window"]
    why = []
    if not (v["inverted"] == 0 and v["nonconvex"] == 0 and v["nonmanifold_edges"] == 0
            and v["same_direction_edges"] == 0):
        why.append(f"the window has {v['inverted']} inverted / {v['nonconvex']} non-convex cells, "
                   f"{v['nonmanifold_edges']} edges with > 2 cells, {v['same_direction_edges']} shared edges "
                   f"traversed the same way by both cells (the rules passed but the block folded)")
    ends = np.array([cA[0], cA[-1], cB[0], cB[-1]])                       # B162: the joins' coast ends are held
    dj, _ = cKDTree(out.points).query(mesh.points[ends])
    rec["join_ends_moved"] = int((dj >= assemble.MATCH_CHORD).sum())
    if rec["join_ends_moved"]:
        why.append(f"{rec['join_ends_moved']} coast end node(s) of the joins left their host position")
    hc = measure.host_coast(out, m.mesh, m.host_node, ins_nodes["final"], chart, seg, h_run, cut_xyz)
    rec["host_coast"] = hc                                                  # B162 / B168: the outside's coast
    if hc["h_run_max"] > assemble.HOST_SHIFT_H:
        why.append(f"the host's coast outside the insert moved {hc['h_run_max']:.2f} h (> {assemble.HOST_SHIFT_H} h)")
    rec["valid"] = not why
    if why:
        rec["invalid"] = why
        log(f"[insert {spec.name}] INVALID result: {'; '.join(why)} (no model files)", flush=True)
    fl = measure.floor_check(mesh, out, h_min)                              # B165: the floor on the result
    fm = measure.floor_check(mesh, m.mesh, h_min)
    rc_xy = np.array([blk.xy[blk.off[i]:blk.off[i] + blk.N[i] + 1].mean(0) for i in range(blk.M + 1)])
    neck_row = int(np.argmin(np.linalg.norm(rc_xy - prof.xy[prof.i0], axis=1)))
    fl.update(merged_smallest_m=fm["smallest_m"], merged_below=fm["below"],
              layout_rounds=blk.info.get("floor_rounds", 0), layout_raised=blk.info.get("floor_raised", False),
              cells_across=int(blk.N[neck_row]), cells_requested=int(prof.n_cells))
    rec["floor"] = fl
    log(f"[insert {spec.name}] floor {h_min / 1e3:.2f} km: smallest cell {fl['smallest_m'] / 1e3:.3f} km, "
        f"{fl['below']} below; cells across the neck {fl['cells_across']} (asked {prof.n_cells}), "
        f"{fl['layout_rounds']} layout rounds", flush=True)
    rec["timing"]["total_s"] = time.perf_counter() - t0
    if fl["below"]:
        n_pass = max(2, int(np.floor(prof.n_cells * fl["smallest_m"] / h_min)))
        by_cells = (f" or cells <= {n_pass}" if prof.W[prof.i0] / n_pass * region.RATIO <= h_host else
                    " (fewer cells across would not be finer than the host's)")
        rec["refused"] = [f"size floor: {fl['below']} cell(s) the insert made are below h_min_m {h_min:.0f} m "
                          f"(smallest {fl['smallest_m']:.0f} m at {fl['at']}; the layout raised its target in "
                          f"{fl['layout_rounds']} rounds, up to the host's size); fix: h_min_m <= "
                          f"{np.floor(fl['smallest_m']):.0f} m{by_cells}"]
        log(f"[insert {spec.name}] REFUSED {rec['refused'][0]}", flush=True)
        res = Result(False, rec, out, m.mesh, win, chart, prof, joins, cands, outline, h_ins, case)
        res.real, res.ring_out = rc, ring_out
        return res
    res = Result(True, rec, out, m.mesh, win, chart, prof, joins, cands, outline, h_ins, case)
    res.real, res.ring_out = rc, ring_out
    return res


def trim_to_joins(arc: np.ndarray, xyA: np.ndarray, xyB: np.ndarray) -> np.ndarray:
    """B202: a moved coast can pass a join's coast end node on the outer side, so the coast point nearest to the node
    lies beyond the join: the bank arc is cut where it crosses join A in its first half (it starts there) and join B
    in its second half (it ends there)."""
    segA = np.stack([xyA[:-1], xyA[1:]], axis=1)
    segB = np.stack([xyB[:-1], xyB[1:]], axis=1)
    n = len(arc)
    k, p, _ = region.first_crossing(arc[::-1], segA)       # the last crossing with join A
    if k >= 0 and n - 2 - k < n // 2:
        arc = np.vstack([p[None], arc[n - 1 - k:]])
    n = len(arc)
    k, p, _ = region.first_crossing(arc, segB)
    if k >= 0 and k >= n // 2:
        arc = np.vstack([arc[:k + 1], p[None]])
    return arc


def land_record(rc, chart: Chart, xyA, xyB, banks, base) -> dict:
    """B202: how far the insert's banks moved into land from today's banks (the smoothed real coast, blended onto the
    join ends) and the water area this adds and removes between the joins (sphere km, chart lengths over the chart
    scale)."""
    from meshom.insert.geom import point_in_polygon

    dep = []
    for b, b0, right in ((banks[0], base[0], True), (banks[1], base[1], False)):     # bank S: land on its right
        pl = Polyline(b0)
        sp = pl.project(b)
        foot, t = pl.at(sp), pl.tangent(sp, max(1e-3 * pl.L, 1.0))
        cr = t[:, 0] * (b[:, 1] - foot[:, 1]) - t[:, 1] * (b[:, 0] - foot[:, 0])
        d = np.linalg.norm(b - foot, axis=1)
        dep.append(np.where((cr < 0) == right, d, -d))
    dep = np.concatenate(dep)
    sc = float(np.mean(chart.scale(np.vstack(banks))))
    new = np.vstack([xyA, banks[1], xyB[::-1], banks[0][::-1]])
    old = np.vstack([xyA, base[1], xyB[::-1], base[0][::-1]])
    lo, hi = np.minimum(new.min(0), old.min(0)), np.maximum(new.max(0), old.max(0))
    step = max(float(np.min(hi - lo)) / 400.0, 50.0)
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step))
    g = np.c_[gx.ravel(), gy.ravel()]
    a, b = point_in_polygon(g, new), point_in_polygon(g, old)
    cell = (step / sc) ** 2 / 1e6
    return {"max_km": float(max(dep.max(), 0.0) * 1e-3 / sc), "into_water_km": float(max(-dep.min(), 0.0) * 1e-3 / sc),
            "added_km2": float((a & ~b).sum() * cell), "removed_km2": float((b & ~a).sum() * cell),
            "coast_max_km": float(rc.land.get("max_m", 0.0) * 1e-3 / sc)}


def collar_record(host: Mesh, reg: np.ndarray, m, case, ring_out: Mesh, out: Mesh, h_host: float, rings: int,
                  host_fixed: bool) -> dict:
    """B192: the collar's measures: its cells before and after the cleaning (measure.cells), the host nodes moved
    (count, the largest move in host cells) and those beyond the collar (must be 0)."""
    T = host.n_tris
    kept = np.zeros(host.n_nodes, bool)
    kept[host.tris[~reg[:T]].ravel()] = True
    kept[host.quads[~reg[T:]].ravel()] = True
    free = case.mesh.points[case.kind != 2]
    collar = np.zeros(host.n_nodes, bool)
    if len(free):
        collar = cKDTree(free).query(host.points)[0] < assemble.MATCH_CHORD
    mv = measure.host_moved(host, out, h_host, kept, collar)
    return {"rings": int(rings), "host_fixed": bool(host_fixed), "before": measure.cells(case.mesh, case.kind),
            "after": measure.cells(ring_out), "host_moved": mv["moved"],
            "host_moved_max_cells": mv["largest_host_cells"], "beyond_collar_moved": mv["beyond_collar"]}


def _small(s: dict) -> dict:
    keep = ("quad_max_angle", "triangles", "irregular", "size", "coast", "validity", "gates")
    return {k: v for k, v in s.items() if k in keep}


# ------------------------------------------------------------------------------------------------- run folder

def host_min_cell(mesh: Mesh, src: Path, cache_dir: Path) -> float:
    """B165: the default size floor = the host mesh's smallest cell (the smallest mean great-circle edge length of an
    element), cached by the mesh file's size and time."""
    from meshom.quality.kernel import element_geometry

    st = src.stat()
    key = f"{src.resolve()}:{st.st_size}:{st.st_mtime_ns}"
    f = cache_dir / "h_min.json"
    if f.exists():
        try:
            d = json.loads(f.read_text())
            if d.get("key") == key:
                return float(d["h_min_m"])
        except Exception:
            pass
    g = element_geometry(mesh)
    parts = [x.mean(1) for x, n in ((g.tri_edges, mesh.n_tris), (g.quad_edges, mesh.n_quads)) if n]
    h = float(np.min(np.concatenate(parts)))
    cache_dir.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"key": key, "h_min_m": h}))
    return h


def real_loops(cfg, source: str = "") -> list | None:
    """The real coast's loops (unit vectors, water on the left): the run's coast (meshom.run.coast_of: its
    [coast] source, min_area_km2 and Antarctica rule), or another shapefile read with the run's area limit."""
    from meshom.coast import read_coast
    from meshom.run import coast_of, source_path

    if source:
        c = read_coast(str(source_path(source)), cfg.coast.min_area_km2, cfg.coast.antarctica)
    else:
        c = coast_of(cfg)
    return None if c is None else list(c.loops)


def run_inputs(run_dir: Path, cfg, cache_dir: Path):
    """(straits, h) of the run as meshom.run.write_inputs builds them, from <run>/cache/size.pkl when valid, else
    from (or into) cache_dir/size.pkl: nothing is written into the run folder."""
    import hashlib

    from meshom.run import SIZE_CACHE_VERSION, coast_of, run_straits, size_field

    key = f"{SIZE_CACHE_VERSION}:{hashlib.sha256((run_dir / 'config.resolved.toml').read_bytes()).hexdigest()}"
    for f in (run_dir / "cache" / "size.pkl", cache_dir / "size.pkl"):
        if f.exists():
            try:
                with open(f, "rb") as fh:
                    c = pickle.load(fh)
                if c["key"] == key:
                    return c["straits"], c["h"]
            except Exception:
                pass
    coast = coast_of(cfg)
    straits = run_straits(cfg, coast)
    h = size_field(cfg, coast, None, straits)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "size.pkl", "wb") as fh:
        pickle.dump({"key": key, "straits": straits, "h": h, "loops": None}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return straits, h


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 5)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x


def _part(full: Mesh, window: Mesh) -> np.ndarray:
    """(T+Q,) bool: the elements of the spliced mesh that came from the insert's window (host.splice appends them
    after the host's kept triangles and quads)."""
    return np.r_[np.arange(full.n_tris) >= full.n_tris - window.n_tris,
                 np.arange(full.n_quads) >= full.n_quads - window.n_quads]


def fix_spliced(mesh: Mesh, win: Window, res: Result, full: Mesh, sinfo: dict, spec: Spec, h_min: float, log=print):
    """B210: assemble.fix_rings on the spliced mesh (the case from the window's part, the window's cut held): (the
    mesh with the new positions, the record, the ids in it of the host nodes moved, the window mesh with the new
    positions for the maps)."""
    h2n = sinfo["host_to_new"]
    cut = np.zeros(full.n_nodes, bool)
    cut[h2n[win.old_id[win.kind == 2]]] = True
    d, gid = cKDTree(full.points).query(res.window.points)            # the window's nodes in the spliced mesh
    assert d.max() < assemble.MATCH_CHORD
    prof = res.record["profile"]
    ll = prof.get("neck_lonlat")                                       # a band has none: the path's centre
    neck = lonlat_to_xyz(np.array([ll[0]]), np.array([ll[1]]))[0] if ll else _path_frame(spec.path)[0]
    out, frec, moved = assemble.fix_rings(win.mesh, full, _part(full, res.window), cut, spec.fix_rings, h_min,
                                          prof["h_host_km"] * 1e3, neck, f"insert_{spec.name}_fix")
    w = res.window
    window = Mesh(out.points[gid], w.tris, w.quads, w.node_flag, w.tri_zone)
    b, a = frec["before"], frec["after"]
    log(f"[insert {spec.name}] fix_rings {spec.fix_rings}: {frec['case']['nodes']} nodes ({frec['case']['free']} "
        f"free, {frec['case']['sliding']} sliding), quads > 110 / > 120 {b['q110']} / {b['q120']} -> {a['q110']} / "
        f"{a['q120']}, worst {b['qmax']:.1f} -> {a['qmax']:.1f} deg; host nodes moved {frec['host_moved']} (max "
        f"{frec['host_moved_max_cells']:.2f} host cells); smallest cell {frec['floor']['smallest_m'] / 1e3:.3f} km "
        f"({frec['wall_s']:.0f} s)", flush=True)
    return out, frec, moved, window


SILL_MARGIN_M = 1.0          # B212: the sill's nodes are capped this far above the midpoint between the two levels


def insert_depth(depth: np.ndarray, mesh: Mesh, win: Window, res: Result, full: Mesh, sinfo: dict, h2n: np.ndarray,
                 moved: np.ndarray | None, spec: Spec, cfg, fmt: str, straits: list, run_dir: Path, log=print):
    """The model depth of the spliced mesh after meshom.run.model_depth (the run's rules on every node): with
    fix_rings the host nodes it moved keep their host depth (B210: the run's rules on the host window, where they
    stand at their host positions); with an own rule of the insert (B211: depth_at, depth_smooth, strait_floor) its
    own nodes (those the splice appended) take that rule on the window's part of the mesh: the grid rule, the
    smoothing (None: the run's passes over the whole part; a number: that many passes among the insert's nodes only),
    the run's slope limiter, the strait depth floors when strait_floor, the depth floor and the run's regions;
    depth_quantile (B214) replaces the grid rule's value by a quantile of the node box, depth_channel (B213) raises
    the nodes along the grid's deepest channel between the joins, and sill_level "shallower" (B212) then rounds the
    sill up in the model's levels (sill_shallower), "grid" (B215) holds the grid's saddle at its level
    (meshom.insert.channel). Returns (depth, record)."""
    from meshom import run as runmod
    from meshom.bathymetry import (apply_regions, cells_deepest_depth, depth_policy, levels, node_area_m2,
                                   node_depth, node_spacing_m, smooth_depth)

    b = cfg.bathymetry
    rec = {"host_moved_kept": 0}
    depth = np.asarray(depth, dtype=np.float64).copy()
    if moved is not None and len(moved):
        hd = runmod.model_depth(win.mesh, cfg, fmt, straits, run_dir)
        host_of = np.full(full.n_nodes, -1, np.int64)
        ok = h2n >= 0
        host_of[h2n[ok]] = np.flatnonzero(ok)
        wpos = np.full(mesh.n_nodes, -1, np.int64)
        wpos[win.old_id] = np.arange(len(win.old_id))
        k = wpos[host_of[moved]]
        assert (k >= 0).all()
        rec["host_moved_kept"] = int(len(moved))
        rec["host_moved_change_m"] = float(np.abs(depth[moved] - hd[k]).max())
        depth[moved] = hd[k]
    own = (bool(spec.depth_at) or spec.depth_smooth is not None or not spec.strait_floor or bool(spec.sill_level)
           or bool(spec.depth_channel) or spec.depth_quantile > 0)
    if b.source == "none" or not own:
        return depth, rec
    part = _part(full, res.window)
    T0 = full.n_tris
    wid = np.unique(np.r_[full.tris[part[:T0]].ravel(), full.quads[part[T0:]].ravel()])
    loc = np.full(full.n_nodes, -1, np.int64)
    loc[wid] = np.arange(len(wid))
    sub = Mesh(full.points[wid], loc[full.tris[part[:T0]]].reshape(-1, 3), loc[full.quads[part[T0:]]].reshape(-1, 4),
               np.zeros(len(wid), np.int8))
    ins = wid >= full.n_nodes - sinfo["appended_nodes"]               # the insert's own nodes (appended last)
    floor = runmod.depth_floor_m(cfg, fmt)
    at = spec.depth_at or b.depth_at
    d, info = (cells_deepest_depth if at == "cells_deepest" else node_depth)(sub, Path(b.source), b.variable, floor)
    grid = None
    if spec.depth_quantile > 0 or spec.depth_channel or spec.sill_level == "grid":
        from meshom.insert import channel as chan
        from meshom.mesh import signed_area

        sp, _, _ = node_spacing_m(sub)
        cells_ins = np.r_[ins[sub.tris].any(1) if sub.n_tris else np.zeros(0, bool), ins[sub.quads].any(1)]
        r_max = max(float(sp[ins].max()), float(np.sqrt(np.abs(signed_area(sub))[cells_ins].max())))
        nodes_ins = np.unique(np.r_[sub.tris[cells_ins[:sub.n_tris]].ravel(), sub.quads[cells_ins[sub.n_tris:]].ravel()])
        lo_, la_ = xyz_to_lonlat(sub.points[nodes_ins])
        grid = chan.read_window(Path(b.source), b.variable, lo_, la_, 2.0 * r_max)
    if spec.depth_quantile > 0:
        d, rec["quantile"] = chan.node_quantile(sub, d, ins, grid, spec.depth_quantile, floor)
    _, edges, _ = node_spacing_m(sub)
    area = node_area_m2(sub)
    passes = b.smooth_passes if spec.depth_smooth is None else int(spec.depth_smooth)
    d = smooth_depth(d, edges, area, passes, nodes=None if spec.depth_smooth is None else ins)
    d, pol = depth_policy(sub, d, floor, 0, b.max_slope, straits if spec.strait_floor else None, b.carve_taper)
    if b.regions:
        d, _ = apply_regions(sub, d, b.regions, floor)
    rec.update(rule={"depth_at": at, "smooth_passes": passes, "smooth_among": "run" if spec.depth_smooth is None
                     else "insert", "strait_floor": bool(spec.strait_floor), "sill_level": spec.sill_level or None,
                     "depth_channel": spec.depth_channel or None, "depth_quantile": spec.depth_quantile or None},
               insert_nodes=int(ins.sum()), grid=info.get("depth_m"),
               strait_depth_floor=pol.get("strait_depth_floor"),
               run_rule_m={"min": float(depth[wid[ins]].min()), "max": float(depth[wid[ins]].max())})
    line = None
    if spec.depth_channel or spec.sill_level == "grid":
        h_f = res.profile.h_fine if res.profile is not None else float(np.median(node_spacing_m(sub)[0][ins]))
        line = chan.channel_line(sub, ins, depth[wid], grid, h_f)
        rec["channel"] = line.record if line is not None else {"skipped": "the insert has no two joins"}
        if line is not None:
            log(f"[insert {spec.name}] channel line {line.record['km']:.0f} km (box means at {h_f / 1e3:.2f} km), "
                f"its deepest connection {line.record['deepest_connection_box_m']:.0f} m; the grid's saddle "
                f"{line.saddle[0]:.0f} m at {line.record['grid_saddle']['at']}", flush=True)
        if line is not None and spec.depth_channel:
            d, rec["channel"]["raise"] = chan.line_raise(sub, d, ins, grid, line, spec.depth_channel)
            r = rec["channel"]["raise"]
            log(f"[insert {spec.name}] depth_channel {spec.depth_channel}: {r['nodes_raised']} nodes raised, by up to "
                f"{r['raise_max_m']:.0f} m (mean {r['raise_mean_m']:.0f} m)", flush=True)
    if spec.sill_level == "shallower":
        d, rec["sill"] = sill_shallower(sub, d, ins, levels(b.levels))
        s = rec["sill"]
        if "barrier_m" in s:
            log(f"[insert {spec.name}] sill {s['barrier_m']:.1f} m at {s['at']}: level {s['level_m']:.0f} m, "
                f"{s['capped']} nodes capped at {s['cap_m']:.1f} m (largest cut {s['cut_max_m']:.1f} m)", flush=True)
    elif spec.sill_level == "grid":
        d, rec["sill"] = (chan.grid_sill(sub, d, ins, line, levels(b.levels)) if line is not None
                          else (d, {"skipped": "no channel line"}))
        s = rec["sill"]
        if "cap_m" in s:
            log(f"[insert {spec.name}] the grid's saddle {s['saddle_m']:.0f} m at {s['saddle_at']}: level "
                f"{s['level_m']:.0f} m, a dam of {s['chain_nodes']} nodes through {s['via']}, {s['capped']} nodes "
                f"capped at {s['cap_m']:.1f} m (largest cut {s['cut_max_m']:.0f} m)", flush=True)
        else:
            log(f"[insert {spec.name}] sill_level grid: {s}", flush=True)
    depth[wid[ins]] = d[ins]
    rec["insert_m"] = {"min": float(d[ins].min()), "max": float(d[ins].max())}
    log(f"[insert {spec.name}] depth of the insert's {int(ins.sum())} nodes: {at}, {passes} smoothing passes "
        f"({rec['rule']['smooth_among']}), strait floors {'on' if spec.strait_floor else 'off'}: "
        f"{d[ins].min():.0f}-{d[ins].max():.0f} m (the run's rules: {rec['run_rule_m']['min']:.0f}-"
        f"{rec['run_rule_m']['max']:.0f} m); host nodes kept at their host depth {rec['host_moved_kept']}", flush=True)
    return depth, rec


def sill_shallower(sub: Mesh, depth: np.ndarray, ins: np.ndarray, zbar: np.ndarray) -> tuple[np.ndarray, dict]:
    """B212: the insert's sill rounded up in the model's levels. FESOM3's fesom_ini gives an element the level whose
    layer midpoint lies below the mean of its nodes' depths (the nearest level surface), a node the deepest level of
    its elements, and the model's element bottom is the shallowest of its nodes' levels: a node at the sill takes the
    next level down as soon as one element round it reaches below the midpoint. Here the barrier is the bank-to-bank
    chain of the insert's nodes (neighbours: nodes of one insert cell) whose deepest node is shallowest (a minimax
    path from one bank's coast nodes to the other's; banks: the two largest components of the insert cells' coast
    edges); its depth B is the insert's sill. Every node of the chain and every node sharing a cell with one is capped
    at the midpoint between L (the deepest level surface not below B) and the next level, minus SILL_MARGIN_M: the
    chain's nodes then take level L, and every element on the chain has its bottom at L or above. Returns (depth,
    record)."""
    import heapq

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from meshom.geo import xyz_to_lonlat
    from meshom.mesh import build_topology

    T = sub.n_tris
    cells = [c for c in (sub.tris, sub.quads) if len(c)]
    in_ins = [c[ins[c].any(1)] for c in cells]                         # the insert's cells
    pairs = np.vstack([np.c_[c[:, i], c[:, j]] for c in in_ins for i in range(c.shape[1])
                       for j in range(i + 1, c.shape[1])])
    n = sub.n_nodes
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()
    g = ((g + g.T) > 0).tocsr()
    topo = build_topology(sub)
    be = topo.edges[topo.edge_elems[:, 1] < 0]
    node_in = np.zeros(n, bool)
    for c in in_ins:
        node_in[c.ravel()] = True
    elem_ins = np.r_[ins[sub.tris].any(1) if T else np.zeros(0, bool), ins[sub.quads].any(1)]
    ce = be[elem_ins[topo.edge_elems[topo.edge_elems[:, 1] < 0, 0]]]  # coast edges of the insert's cells
    if not len(ce):
        return depth, {"skipped": "the insert has no coast edges"}
    cg = coo_matrix((np.ones(len(ce)), (ce[:, 0], ce[:, 1])), shape=(n, n))
    nc, lab = connected_components(cg, directed=False)
    on = np.zeros(n, bool)
    on[ce.ravel()] = True
    sizes = np.bincount(lab[on], minlength=nc)
    if (sizes > 0).sum() < 2:
        return depth, {"skipped": "the insert's coast is one piece (no two banks)"}
    a, bnk = np.argsort(sizes)[::-1][:2]
    src, dst = on & (lab == a), on & (lab == bnk)
    cost = np.full(n, np.inf)
    prev = np.full(n, -1, np.int64)
    heap = [(float(depth[i]), int(i)) for i in np.flatnonzero(src)]
    for c, i in heap:
        cost[i] = c
    heapq.heapify(heap)
    done = np.zeros(n, bool)
    end = -1
    while heap:
        c, i = heapq.heappop(heap)
        if done[i]:
            continue
        done[i] = True
        if dst[i]:
            end = i
            break
        for j in g.indices[g.indptr[i]:g.indptr[i + 1]]:
            cj = max(c, float(depth[j]))
            if cj < cost[j]:
                cost[j], prev[j] = cj, i
                heapq.heappush(heap, (cj, int(j)))
    if end < 0:
        return depth, {"skipped": "no chain between the banks"}
    chain = [end]
    while prev[chain[-1]] >= 0:
        chain.append(int(prev[chain[-1]]))
    chain = np.array(chain[::-1])
    B = float(cost[end])
    z = np.asarray(zbar, dtype=np.float64)
    k = int(np.searchsorted(z, B, side="right")) - 1
    L, Ln = float(z[max(k, 0)]), float(z[min(k + 1, len(z) - 1)])
    cap = 0.5 * (L + Ln) - SILL_MARGIN_M
    hit = np.zeros(n, bool)
    hit[chain] = True
    star = np.r_[hit[sub.tris].any(1) if T else np.zeros(0, bool), hit[sub.quads].any(1)]
    nb = np.zeros(n, bool)
    nb[sub.tris[star[:T]].ravel()] = True
    nb[sub.quads[star[T:]].ravel()] = True
    out = depth.copy()
    low = nb & (out > cap)
    out[low] = cap
    top = chain[int(np.argmax(depth[chain]))]
    return out, {"barrier_m": B, "level_m": L, "next_level_m": Ln, "cap_m": cap, "chain_nodes": int(len(chain)),
                 "at": [round(float(v[0]), 4) for v in xyz_to_lonlat(sub.points[top][None])],
                 "capped": int(low.sum()), "capped_host_nodes": int((low & ~ins).sum()),
                 "cut_max_m": float((depth - out).max()), "nodes_star": int(nb.sum())}


def run(run_dir: Path, spec: Spec, out_dir: Path | None = None, mesh_name: str | None = None,
        write: bool = True, log=print) -> dict:
    """The insert in a finished run (module docstring): writes <out>/insert_<name>.npz, .json (the record),
    .score.json (zone scores and the ring's meshfix score), .map.png, .profile.png and, when the run wrote model
    files, fesom-insert_<name>/ in the run's format with depth by the run's bathymetry rules (insert_depth: the
    insert's own rule, B211-B215). With fix_rings the spliced mesh is smoothed first (fix_spliced, B210). Returns the
    record."""
    from meshom import compete
    from meshom import run as runmod
    from meshom.config import load as load_cfg
    from meshom.insert import figures

    run_dir = Path(run_dir).resolve()
    out_dir = Path(out_dir).resolve() if out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    cfg = load_cfg(run_dir / "config.resolved.toml")
    src = runmod.mesh_file(run_dir, mesh_name)
    mesh = compete._load(src)
    log(f"[insert {spec.name}] host {src} ({mesh.n_nodes} nodes)", flush=True)
    straits, h_run = run_inputs(run_dir, cfg, out_dir / "cache")
    h_min, h_min_src = spec.h_min_m, "spec"
    if h_min is None:
        h_min, h_min_src = host_min_cell(mesh, src, out_dir / "cache"), "the host mesh's smallest cell"
    log(f"[insert {spec.name}] size floor h_min {h_min / 1e3:.2f} km ({h_min_src})", flush=True)
    real = None
    if spec.coast == "real":
        real = real_loops(cfg, spec.coast_source)
    c, half = _path_frame(spec.path)
    from meshom import straits as stm

    h_host = host_size(mesh, stm.densify(spec.path, 1000.0), max(half, 3000.0) + PROBE_KM * 1e3)
    reach = spec.reach_m or region.REACH_H * h_host
    radius = half + 1.5 * reach + WINDOW_PAD_H * h_host
    win = cut_window(mesh, c, radius)
    log(f"[insert {spec.name}] window {radius / 1e3:.0f} km: {win.mesh.n_nodes} nodes, {win.mesh.n_elems} elems",
        flush=True)
    orca = "orca" in cfg.mesh.method                       # B193: an ORCA run (its windows may still be meshom's)
    if spec.kind == "strait":
        res = insert_window(win, spec, h_run, cfg.size_tolerance() or 0.20, cfg.gates_all(), log, real=real,
                            h_min=h_min, orca=orca)
    else:                                                    # B181 / B182: a coastal strip or a current corridor
        from meshom.insert.band import insert_band

        res = insert_band(win, spec, h_run, cfg.size_tolerance() or 0.20, cfg.gates_all(), log, real=real,
                          h_min=h_min, orca=orca)
    rec = {"run": str(run_dir), "mesh": str(src), "config": str(run_dir / "config.resolved.toml"),
           "spec": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in dataclasses.asdict(spec).items()
                    if k != "path"} | {"path_lonlat": np.c_[xyz_to_lonlat(spec.path)].T.tolist()},
           "window_km": radius / 1e3, "h_min_m": h_min, "h_min_source": h_min_src, **res.record}
    stem = out_dir / f"insert_{spec.name}"
    if res.profile is not None:
        figures.profile(stem.with_suffix(".profile.png"), res.profile, res.joins, res.candidates)
    if not res.ok and (res.window is None or res.profile is not None):
        rec["status"] = "refused"
        stem.with_suffix(".json").write_text(json.dumps(_jsonable(rec), indent=1))
        return rec
    full = sinfo = moved = None
    if res.ok:
        t = time.perf_counter()
        full, sinfo = splice(mesh, win, res.window)
        rec["timing"]["splice_s"] = time.perf_counter() - t
        if spec.fix_rings > 0 and res.record.get("valid"):           # B210: stage a on the insert and its rings
            full, rec["fix_rings"], moved, res.window = fix_spliced(mesh, win, res, full, sinfo, spec, h_min, log)
    seg = res.chart.fwd(win.mesh.points)[win.coast_edges]
    real_seg = res.real.segments("raw") if res.real is not None else None
    if res.profile is not None:
        neck = res.profile.xy[res.profile.i0]
        chains_xy = [res.chart.fwd(win.mesh.points[j.nodes]) for j in res.joins]
        h_f, zooms = res.profile.h_fine, []
    else:
        neck, chains_xy, h_f = res.band["mid"], res.band["chains_xy"], res.band["h_fine"]
        zooms = res.band.get("zooms", [])
    bays = [(b.arc, b.meshed) for b in res.real.bays] if res.real is not None else []
    figures.maps(stem.with_suffix(".map.png"), res.chart,
                 [("host", win.mesh), ("merged (before cleaning)", res.merged), ("insert (final)", res.window)],
                 res.outline, chains_xy, seg, neck, h_f, real_seg, zooms=zooms, bays=bays)
    if res.ring_case is not None and res.ring_out is not None:
        figures.collar(stem.with_suffix(".collar.png"), res.chart, res.ring_case.mesh, res.ring_out, chains_xy, seg,
                       f"collar ({res.record.get('collar', {}).get('rings', '?')} rings)")
    if not res.ok:                                           # a band refused after its build: the maps say where
        rec["status"] = "refused"
        stem.with_suffix(".json").write_text(json.dumps(_jsonable(rec), indent=1))
        return rec
    h2n = sinfo.pop("host_to_new")
    np.save(stem.parent / (stem.name + ".host_to_new.npy"), h2n)
    rec["splice"] = sinfo
    rec["validity"] = {"host": measure.validity(mesh), "final": measure.validity(full)}
    compete._save(full, stem.with_suffix(".npz"))
    rec["status"] = "ok" if res.record.get("valid") else "invalid"
    if write and rec["status"] == "ok" and (run_dir / "fesom").exists():
        t = time.perf_counter()
        fmt = cfg.mesh.format
        meta = {}
        depth = runmod.model_depth(full, cfg, fmt, straits, run_dir, meta)
        depth, rec["depth"] = insert_depth(depth, mesh, win, res, full, sinfo, h2n, moved, spec, cfg, fmt, straits,
                                           run_dir, log)
        fdir = out_dir / f"fesom-insert_{spec.name}"
        fdir.mkdir(exist_ok=True)
        np.save(fdir / "depth.npy", depth)
        checks = runmod.write_files(full, fmt, fdir, depth, cfg.bathymetry.levels)
        rec["write"] = {"format": fmt, "dir": str(fdir), "depth_min_m": float(depth.min()),
                        "depth_max_m": float(depth.max()), **checks, "wall_s": time.perf_counter() - t}
        log(f"[insert {spec.name}] wrote {fmt} files to {fdir}: {checks}", flush=True)
    rec["timing"]["run_s"] = time.perf_counter() - t0
    stem.with_suffix(".json").write_text(json.dumps(_jsonable(rec), indent=1, default=str))
    score = {"scores": rec.get("scores"), "ring_score": rec.get("ring_score"), "validity": rec.get("validity"),
             "joins": rec.get("joins")}
    stem.with_suffix(".score.json").write_text(json.dumps(_jsonable(score), indent=1, default=str))
    return rec
