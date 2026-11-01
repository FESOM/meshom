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
    kind: str = "strait"                     # B181 / B182: "strait", "strip" or "corridor"; B223: "depth"
    width_m: float | None = None             # strip: coast to offshore join; corridor, depth: full width
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
    sill_level: str = ""                     # B212: "shallower" (rounded up); B215: "grid"; B249: "nearest"
    depth_channel: str = ""                  # B213: "cells" | "near": the grid's deepest channel raises the insert
    depth_quantile: float = 0.0              # B214: the node box's quantile instead of its mean (0: the mean)
    depth_area: bool = False                 # B250: the sections' areas scaled to the grid's
    polygon: np.ndarray | None = None        # B223: kind "depth": the region (K,3) instead of width_m round the path
    channels: list = field(default_factory=list)  # B240-B242: a group's channels (meshom.insert.group.Channel)
    gap_km: float | None = None              # B241: a group's island gap (None: 5 x h_min)
    island_km2: float = 10.0                 # B241: a group's smallest island
    growth: float | None = None              # B243: a group's size slope (None: 0.12)
    gap_cells: int = 3                       # B243: a group's least fill cells across any passage
    split_deg: float | None = None           # B243: a group's patch quads with a corner above this: two triangles
    bank_smooth_cells: float = 0.0           # B245: a group's block banks smoothed over this many block cells
    routes: list = field(default_factory=list)    # B246: a group's routes: (name, a (3,), b (3,), via, corridor m)
    blend_cells: float | None = None         # B247: a group's coast blend onto the join ends (None: BLEND_H)
    growth_max: float | None = None          # B252: a group's steepest slope to the joins' host size (None: GROWTH_MAX)
    even_sweeps: int = 0                     # B252: a group's sweeps of the area pass on its patches and host rings
    even_blocks: str = ""                    # B252: the blocks' nodes in that pass ("", "cross", "block")


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
    collar_moved: np.ndarray | None = None   # B217: the host window's nodes the collar moved (window node ids)


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
    if e.kind == "group":                    # B240: the envelope is the path; the channels as Channel
        from meshom.insert.even import EVEN_SWEEPS
        from meshom.insert.group import Channel

        env = _ll(e.envelope)
        if env is None or len(env) < 3:
            raise InsertError("a group insert needs an envelope polygon of at least 3 points")
        chans = [Channel(str(c.get("name", f"channel{k}")), _ll(c["a"]), _ll(c["b"]), int(c.get("cells", 5)))
                 for k, c in enumerate(e.channels)]
        return Spec(e.name or "group", env, h_min_m=e.h_min_m, coast=e.coast, coast_source=e.coast_source,
                    bank_sigma_cells=e.bank_sigma_cells, kind="group", fix_rings=e.fix_rings, depth_at=e.depth_at,
                    depth_smooth=e.depth_smooth, strait_floor=e.strait_floor, sill_level=e.sill_level,
                    depth_channel=e.depth_channel, depth_quantile=e.depth_quantile, depth_area=e.depth_area,
                    channels=chans,
                    gap_km=e.gap_km, island_km2=e.island_km2, growth=e.growth, bank_land_cells=e.bank_land_cells,
                    gap_cells=e.gap_cells, bank_smooth_cells=e.bank_smooth_cells,
                    routes=[(str(r.get("name", f"route{k}")), _ll([r["a"]])[0], _ll([r["b"]])[0],
                             _ll(r.get("via")), float(r["corridor_km"]) * 1e3 if r.get("corridor_km") else None)
                            for k, r in enumerate(e.routes)], blend_cells=e.blend_cells, split_deg=e.split_deg,
                    growth_max=e.growth_max,
                    even_sweeps=EVEN_SWEEPS if e.even_sweeps is None else int(e.even_sweeps),
                    even_blocks=e.even_blocks)
    if e.kind not in ("strait", "depth"):
        if path is None or e.width_km is None:
            raise InsertError(f"an insert of kind {e.kind!r} needs a path and width_km")
    if e.kind == "depth" and e.width_km is None and not e.polygon:
        raise InsertError("an insert of kind 'depth' needs width_km (the region's full width round its path) or a "
                          "polygon")
    if e.kind == "depth" and path is None and not e.strait and e.polygon:
        path = _ll(e.polygon)                                # the record's path: the polygon's outline
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
                e.depth_quantile, e.depth_area, _ll(e.polygon))


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
    banks_xy, bank_src, base_xy, squared, fallback, joint = [], [], [], [], [], []
    inward = ((xy[cA[1]] - xy[cA[0]], xy[cB[1]] - xy[cB[0]]), (xy[cA[-2]] - xy[cA[-1]], xy[cB[-2]] - xy[cB[-1]]))
    for side, ids, (ua, ub) in zip("SN", (bS_ids, bN_ids), inward):
        b = None
        if rc is not None:
            arc, why = rc.arc(xy[ids[0]], xy[ids[-1]], guide=xy[ids])
            host_len = Polyline(xy[ids]).L
            if arc is not None and Polyline(arc).L > 3.0 * host_len + 4 * h_host:
                arc, why = None, "the real coast between the ends is far longer than the host's (around an island)"
            if arc is not None and why:                # B219: the piece chosen for both ends together
                joint.append({"bank": side, "note": why})
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
                lo_, la_ = xyz_to_lonlat(chart.inv(xy[ids[[0, -1]]]))
                fallback.append({"bank": side, "reason": why,
                                 "ends_lonlat": [[round(float(x), 4), round(float(y), 4)] for x, y in zip(lo_, la_)]})
                log(f"[insert {spec.name}] WARNING: bank {side} falls back to the host's coast (the real coast was "
                    f"asked): {why}; ends {fallback[-1]['ends_lonlat']}", flush=True)
        if b is None:
            b = bank_fn(xy[ids])
            if rc is None:
                bank_src.append("host")
        banks_xy.append(b)
    rec["banks"] = bank_src
    if rc is not None:                                 # B219: a real bank asked but the host's coast taken
        rec["bank_fallback"] = fallback
        rec["bank_joint"] = joint
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
    rec["collar"], collar_ids = collar_record(mesh, reg, m, case, ring_out, out, h_host, rings, host_fixed)
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
        res.real, res.ring_out, res.collar_moved = rc, ring_out, collar_ids
        return res
    res = Result(True, rec, out, m.mesh, win, chart, prof, joins, cands, outline, h_ins, case)
    res.real, res.ring_out, res.collar_moved = rc, ring_out, collar_ids
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
                  host_fixed: bool) -> tuple[dict, np.ndarray]:
    """B192: the collar's measures: its cells before and after the cleaning (measure.cells), the host nodes moved
    (count, the largest move in host cells) and those beyond the collar (must be 0); and the ids in `host` of the
    host nodes moved (B217). Returns (record, ids)."""
    T = host.n_tris
    kept = np.zeros(host.n_nodes, bool)
    kept[host.tris[~reg[:T]].ravel()] = True
    kept[host.quads[~reg[T:]].ravel()] = True
    free = case.mesh.points[case.kind != 2]
    collar = np.zeros(host.n_nodes, bool)
    if len(free):
        collar = cKDTree(free).query(host.points)[0] < assemble.MATCH_CHORD
    mv = measure.host_moved(host, out, h_host, kept, collar)
    return ({"rings": int(rings), "host_fixed": bool(host_fixed), "before": measure.cells(case.mesh, case.kind),
             "after": measure.cells(ring_out), "host_moved": mv["moved"],
             "host_moved_max_cells": mv["largest_host_cells"], "beyond_collar_moved": mv["beyond_collar"]},
            mv["ids"])


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
    b = frec["before"]
    f = frec["fixer_result"]
    log(f"[insert {spec.name}] fix_rings {spec.fix_rings}: {frec['case']['nodes']} nodes ({frec['case']['free']} "
        f"free, {frec['case']['sliding']} sliding), quads > 110 / > 120 {b['q110']} / {b['q120']} -> {f['q110']} / "
        f"{f['q120']}, worst {b['qmax']:.1f} -> {f['qmax']:.1f} deg; kept the {frec['kept']} ({frec['keep_rule']}); "
        f"host nodes moved {frec['host_moved']} (max {frec['host_moved_max_cells']:.2f} host cells); smallest cell "
        f"{frec['floor']['smallest_m'] / 1e3:.3f} km ({frec['wall_s']:.0f} s)", flush=True)
    if frec["kept"] == "before":
        log(f"[insert {spec.name}] WARNING: fix_rings made the insert worse; its result is not used", flush=True)
    return out, frec, moved, window


SILL_MARGIN_M = 1.0          # B212: the sill's nodes are capped this far above the midpoint between the two levels
COLLAR_MATCH_H = 1.5         # B217: a host node the collar moved is matched within this many of its spacings


def host_depth_of(src: Path, n_nodes: int) -> np.ndarray | None:
    """B217: the saved depth of a host mesh that is an insert's result (insert_<name>.npz: fesom-insert_<name>/depth.npy
    beside it, written after the mesh, one value per node), else None (the host's depth is then the run's rules)."""
    f = src.parent / f"fesom-{src.stem}" / "depth.npy"
    if not (src.stem.startswith("insert_") and f.exists() and f.stat().st_mtime >= src.stat().st_mtime):
        return None
    d = np.load(f)
    return d if len(d) == n_nodes else None


def collar_moved(win: Window, res: Result, full: Mesh, sinfo: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """B217: the host nodes the collar moved (res.collar_moved: the host window's nodes outside the replaced region
    that are no node of the cleaned window at their position, B192) and their nodes in the spliced mesh, which the
    splice appended: a one-to-one match of the moved nodes to appended nodes within COLLAR_MATCH_H of their host
    spacing (mean edge length) with the least sum of distances (the most matches first). Returns (ids in the spliced
    mesh, host ids, record: moved, matched, the largest move, the unmatched ones)."""
    from scipy.optimize import linear_sum_assignment

    from meshom.bathymetry import node_spacing_m

    wm = np.asarray(res.collar_moved if res.collar_moved is not None else [], np.int64)
    none = np.zeros(0, np.int64)
    if not len(wm):
        return none, none, {"moved": 0, "matched": 0}
    app = np.arange(full.n_nodes - sinfo["appended_nodes"], full.n_nodes)
    if not len(app):
        return none, none, {"moved": int(len(wm)), "matched": 0}
    sp, _, _ = node_spacing_m(win.mesh)
    chord = 2.0 * np.sin(0.5 * COLLAR_MATCH_H * sp[wm] / R_EARTH_M)
    k = min(8, len(app))
    dist, j = cKDTree(full.points[app]).query(win.mesh.points[wm], k=k)
    dist, j = dist.reshape(len(wm), k), j.reshape(len(wm), k)
    ok = dist <= chord[:, None]
    cand = np.unique(j[ok])
    big = 1e3                                               # no pair: far above any sum of real chords
    C = np.full((len(wm), max(len(cand), 1)), big)
    r = np.repeat(np.arange(len(wm)), k).reshape(len(wm), k)[ok]
    C[r, np.searchsorted(cand, j[ok])] = dist[ok]
    ri, ci = linear_sum_assignment(C)
    m = C[ri, ci] < big
    ri, ci = ri[m], ci[m]
    km = 2.0 * np.arcsin(0.5 * C[ri, ci]) * R_EARTH_M / 1e3 if len(ri) else np.zeros(1)
    left = np.setdiff1d(np.arange(len(wm)), ri)
    lo, la = xyz_to_lonlat(win.mesh.points[wm[left]])
    near = 2.0 * np.arcsin(0.5 * np.minimum(dist[left, 0], 2.0)) * R_EARTH_M / np.maximum(sp[wm[left]], 1.0)
    rec = {"moved": int(len(wm)), "matched": int(len(ri)), "move_max_km": float(km.max()),
           "unmatched": [{"lonlat": [round(float(x), 4), round(float(y), 4)], "nearest_spacings": round(float(q), 2)}
                         for x, y, q in zip(lo, la, near)]}
    return app[cand[ci]].astype(np.int64), win.old_id[wm[ri]].astype(np.int64), rec


def insert_depth(depth: np.ndarray, mesh: Mesh, win: Window, res: Result, full: Mesh, sinfo: dict, h2n: np.ndarray,
                 moved: np.ndarray | None, spec: Spec, cfg, fmt: str, straits: list, run_dir: Path, log=print,
                 host: np.ndarray | None = None, collar: tuple | None = None):
    """The model depth of the spliced mesh after meshom.run.model_depth (the run's rules on every node): `host` (the
    host mesh's own depth when it is an insert's result, host_depth_of) on every host node the splice kept (B217);
    the host nodes fix_rings moved (B210) and those the collar moved (`collar`: (ids in the spliced mesh, host ids),
    collar_moved, B217) keep their host depth (`host`, else the run's rules on the host window, where they stand at
    their host positions); with an own rule of the insert (B211-B215: depth_at, depth_smooth, strait_floor,
    depth_quantile, depth_channel, sill_level) its own nodes (those the splice appended, less the collar's moved host
    nodes) take that rule on the window's part of the mesh (rule_depth). Returns (depth, record)."""
    from meshom import run as runmod

    b = cfg.bathymetry
    rec = {"host_moved_kept": 0}
    depth = np.asarray(depth, dtype=np.float64).copy()
    ok = h2n >= 0
    if host is not None:                                   # B217: a chain: the host mesh's own depth on its kept nodes
        depth[h2n[ok]] = host[ok]
        rec["host_depth"] = {"source": "the host mesh's own (an insert's result)", "nodes": int(ok.sum())}
    host_of = np.full(full.n_nodes, -1, np.int64)
    host_of[h2n[ok]] = np.flatnonzero(ok)
    fid = np.asarray(moved if moved is not None else [], np.int64)
    hid = host_of[fid]
    cm = np.zeros(full.n_nodes, bool)
    if collar is not None and len(collar[0]):
        cm[collar[0]] = True
        fid, hid = np.r_[fid, collar[0]], np.r_[hid, collar[1]]
        rec["collar_moved_kept"] = int(len(collar[0]))
    if len(fid):
        if host is not None:
            hd = host[hid]
        else:
            hw = runmod.model_depth(win.mesh, cfg, fmt, straits, run_dir)
            wpos = np.full(mesh.n_nodes, -1, np.int64)
            wpos[win.old_id] = np.arange(len(win.old_id))
            k = wpos[hid]
            assert (k >= 0).all()
            hd = hw[k]
        rec["host_moved_kept"] = int(len(fid))
        rec["host_moved_change_m"] = float(np.abs(depth[fid] - hd).max())
        depth[fid] = hd
    own = (bool(spec.depth_at) or spec.depth_smooth is not None or not spec.strait_floor or bool(spec.sill_level)
           or bool(spec.depth_channel) or spec.depth_quantile > 0 or spec.depth_area)
    if b.source == "none" or not own:
        return depth, rec
    part = _part(full, res.window)
    T0 = full.n_tris
    wid = np.unique(np.r_[full.tris[part[:T0]].ravel(), full.quads[part[T0:]].ravel()])
    loc = np.full(full.n_nodes, -1, np.int64)
    loc[wid] = np.arange(len(wid))
    sub = Mesh(full.points[wid], loc[full.tris[part[:T0]]].reshape(-1, 3), loc[full.quads[part[T0:]]].reshape(-1, 4),
               np.zeros(len(wid), np.int8))
    ins = (wid >= full.n_nodes - sinfo["appended_nodes"]) & ~cm[wid]   # the insert's own nodes (appended last)
    d, r = rule_depth(sub, ins, depth[wid], cfg, fmt, straits, spec.depth_at, spec.depth_smooth, spec.strait_floor,
                      spec.sill_level, spec.depth_channel, spec.depth_quantile,
                      res.profile.h_fine if res.profile is not None else None, f"[insert {spec.name}]", log,
                      spec.depth_area, insert_axis(res) if spec.depth_area else None)
    r["insert_nodes"], r["insert_m"] = r.pop("nodes"), r.pop("nodes_m")
    rec.update(r)
    depth[wid[ins]] = d[ins]
    ru = rec["rule"]
    log(f"[insert {spec.name}] depth of the insert's {int(ins.sum())} nodes: {ru['depth_at']}, {ru['smooth_passes']} "
        f"smoothing passes ({ru['smooth_among']}), strait floors {'on' if spec.strait_floor else 'off'}: "
        f"{d[ins].min():.0f}-{d[ins].max():.0f} m (the run's rules: {rec['run_rule_m']['min']:.0f}-"
        f"{rec['run_rule_m']['max']:.0f} m); host nodes kept at their host depth {rec['host_moved_kept']}", flush=True)
    return depth, rec


def insert_axis(res: Result) -> np.ndarray | None:
    """B250: the strait's axis between its two joins on the sphere ((K,3): the profile's centreline points whose
    arc length lies between the joins'), None for an insert without a profile (a strip, a corridor)."""
    pr = res.profile
    if pr is None or res.chart is None:
        return None
    sj = [j.s_m for j in res.joins if j is not None]
    lo, hi = (min(sj), max(sj)) if len(sj) >= 2 else (float(pr.s[0]), float(pr.s[-1]))
    keep = (pr.s >= lo) & (pr.s <= hi)
    return res.chart.inv(pr.xy[keep]) if keep.sum() >= 2 else None


def rule_depth(sub: Mesh, ins: np.ndarray, base: np.ndarray, cfg, fmt: str, straits: list | None, depth_at: str = "",
               depth_smooth: int | None = None, strait_floor: bool = True, sill_level: str = "",
               depth_channel: str = "", depth_quantile: float = 0.0, h_fine: float | None = None, tag: str = "",
               log=print, depth_area: bool = False, axis: np.ndarray | None = None,
               path: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """An own depth rule (B211-B215) on the nodes `ins` of `sub`, a part of a mesh that holds every cell round them
    (insert_depth: the insert's own nodes; meshom.methods.m16_orca.nest_depth: the nodes inside a nest's coarse box,
    B216); `base`: the depth of `sub`'s nodes by the run's rules (for the record only: every calculation here takes
    its depths from the bathymetry grid, B218, never from the host's node depths). The grid rule `depth_at` ("": the
    run's), depth_quantile (B214: a quantile of the node box instead of the mean), `depth_smooth` passes among `ins`
    (None: the run's passes over the whole part), the run's slope limiter, the strait depth floors with
    `strait_floor`, the depth floor and the run's regions; then depth_channel (B213: the channel line between the two
    largest pieces of the other nodes of the cells of `ins` (the joins; its ends from the grid's box means along
    them, B218), box means at h_fine, None: the median spacing of `ins`) and sill_level ("shallower", B212:
    sill_shallower; "grid", B215: meshom.insert.channel.grid_sill), both on the grid rule's node depths of `sub`
    (host nodes included); then depth_area (B250: meshom.insert.area.area_depth along `axis` (K,3), the strait's
    centreline from one join to the other, the dam's cap kept; with the channel line the deepest connection between
    the joins within a level of the grid saddle's, B251, else the dam's capped nodes and the cells the channel line
    crosses never made shallower). `path` ((K,3), a depth region's path, B253): when the coast of the cells of `ins`
    gives no two banks (open water), the joins, the dam's banks and the area rule's sections come from the path and
    the grid rule's depths at the region's outer nodes (meshom.insert.channel.open_sides). `tag` heads the log
    lines. Returns (the
    depth on `sub`, the rule's values on `ins`; the record)."""
    from meshom import run as runmod
    from meshom.bathymetry import (apply_regions, cells_deepest_depth, depth_policy, levels, node_area_m2,
                                   node_depth, node_spacing_m, smooth_depth)

    b = cfg.bathymetry
    rec = {}
    floor = runmod.depth_floor_m(cfg, fmt)
    at = depth_at or b.depth_at
    if at == "cells_deepest":
        d, info = cells_deepest_depth(sub, Path(b.source), b.variable, floor)
    else:                                                     # the run's box factor (B222)
        d, info = node_depth(sub, Path(b.source), b.variable, floor, box_factor=b.box_factor)
    grid = None
    grid_like = sill_level in ("grid", "nearest")         # B249: "nearest", the dam at the nearest level
    if depth_quantile > 0 or depth_channel or grid_like:
        from meshom.insert import channel as chan
        from meshom.mesh import signed_area

        sp, _, _ = node_spacing_m(sub)
        cells_ins = np.r_[ins[sub.tris].any(1) if sub.n_tris else np.zeros(0, bool), ins[sub.quads].any(1)]
        r_max = max(float(sp[ins].max()), float(np.sqrt(np.abs(signed_area(sub))[cells_ins].max())))
        nodes_ins = np.unique(np.r_[sub.tris[cells_ins[:sub.n_tris]].ravel(), sub.quads[cells_ins[sub.n_tris:]].ravel()])
        lo_, la_ = xyz_to_lonlat(sub.points[nodes_ins])
        grid = chan.read_window(Path(b.source), b.variable, lo_, la_, 2.0 * r_max)
    if depth_quantile > 0:
        d, rec["quantile"] = chan.node_quantile(sub, d, ins, grid, depth_quantile, floor)
    _, edges, _ = node_spacing_m(sub)
    area = node_area_m2(sub)
    passes = b.smooth_passes if depth_smooth is None else int(depth_smooth)
    d = smooth_depth(d, edges, area, passes, nodes=None if depth_smooth is None else ins)
    d, pol = depth_policy(sub, d, floor, 0, b.max_slope, straits if strait_floor else None, b.carve_taper)
    if b.regions:
        d, _ = apply_regions(sub, d, b.regions, floor)
    rec.update(rule={"depth_at": at, "smooth_passes": passes, "smooth_among": "run" if depth_smooth is None
                     else "insert", "strait_floor": bool(strait_floor), "sill_level": sill_level or None,
                     "depth_channel": depth_channel or None, "depth_quantile": depth_quantile or None,
                     "depth_area": bool(depth_area)},
               nodes=int(ins.sum()), grid=info.get("depth_m"),
               strait_depth_floor=pol.get("strait_depth_floor"),
               run_rule_m={"min": float(base[ins].min()), "max": float(base[ins].max())})
    line = open_ = None
    if path is not None and (depth_channel or grid_like or depth_area):
        from meshom.insert import channel as chan

        if isinstance(chan.banks(sub, ins), str):            # B253: open water, no two banks on the coast
            open_ = chan.open_sides(sub, ins, d, path)
            if isinstance(open_, str):
                rec["open_water"] = {"skipped": open_}
                open_ = None
            else:
                rec["open_water"] = open_["record"]
            log(f"{tag} open water (B253): {rec['open_water']}", flush=True)
    if depth_channel or grid_like:
        h_f = h_fine if h_fine is not None else float(np.median(node_spacing_m(sub)[0][ins]))
        line = chan.channel_line(sub, ins, grid, h_f, joins=open_["joins"] if open_ else None)
        if line is not None and open_:                        # B253: no wet width in open water
            line.narrow, line.record["narrows"] = chan.open_narrows(sub, ins, line)
        rec["channel"] = line.record if line is not None else {"skipped": "the insert has no two joins"}
        if line is not None:
            lr = line.record
            nr = lr["narrows"]
            nar = (f"{nr['from']} - {nr['to']} (narrowest {nr['width_min_km']:.1f} km at {nr['at']})"
                   if "from" in nr else f"{nr['km']:.0f} km in the region's cells (open water)")
            log(f"{tag} channel line {lr['km']:.0f} km (box means at {h_f / 1e3:.2f} km) from {lr['ends'][0]} "
                f"({lr['end_from'][0]}, {lr['end_box_m'][0]:.0f} m) to {lr['ends'][1]} ({lr['end_from'][1]}, "
                f"{lr['end_box_m'][1]:.0f} m), its deepest connection {lr['deepest_connection_box_m']:.0f} m; the "
                f"grid's saddle {line.saddle[0]:.0f} m at {lr['grid_saddle']['at']}; narrows {nar}"
                f"{'' if lr['corridor'] else '; the corridor does not join the ends'}", flush=True)
            for s_, e_ in zip("AB", lr["end_limited"]):
                if e_:
                    log(f"{tag} WARNING: the channel line's search is end-limited at join {s_}: its deepest "
                        f"connection equals that end's depth, the basin beyond the join (in the grid's window) "
                        f"is no deeper", flush=True)
        if line is not None and depth_channel:
            d, rec["channel"]["raise"] = chan.line_raise(sub, d, ins, grid, line, depth_channel)
            r = rec["channel"]["raise"]
            log(f"{tag} depth_channel {depth_channel}: {r['nodes_raised']} nodes raised, by up to "
                f"{r['raise_max_m']:.0f} m (mean {r['raise_mean_m']:.0f} m)", flush=True)
    if sill_level == "shallower":
        d, rec["sill"] = sill_shallower(sub, d, ins, levels(b.levels))
        s = rec["sill"]
        if "barrier_m" in s:
            log(f"{tag} sill {s['barrier_m']:.1f} m at {s['at']}: level {s['level_m']:.0f} m, "
                f"{s['capped']} nodes capped at {s['cap_m']:.1f} m (largest cut {s['cut_max_m']:.1f} m)", flush=True)
    elif grid_like:
        d, rec["sill"] = (chan.grid_sill(sub, d, ins, line, levels(b.levels), nearest=sill_level == "nearest",
                                         sides=open_["sides"] if open_ else None)
                          if line is not None
                          else (d, {"skipped": "no channel line"}))
        s = rec["sill"]
        if "cap_m" in s:
            log(f"{tag} the grid's saddle {s['saddle_m']:.0f} m at {s['saddle_at']}: level "
                f"{s['level_m']:.0f} m, a dam of {s['chain_nodes']} nodes through {s['via']}, {s['capped']} nodes "
                f"capped at {s['cap_m']:.1f} m (largest cut {s['cut_max_m']:.0f} m)", flush=True)
        else:
            log(f"{tag} sill_level grid: {s}", flush=True)
    held = rec["sill"].pop("held", None) if "sill" in rec else None
    if depth_area:
        if axis is None:
            rec["area"] = {"applied": False, "skipped": "no axis (an insert without a profile)"}
            log(f"{tag} depth_area: skipped, no axis", flush=True)
        else:
            from meshom.insert.area import area_depth

            h_f = h_fine if h_fine is not None else float(np.median(node_spacing_m(sub)[0][ins]))
            route = None
            if line is not None:                           # the nodes of the cells the channel line crosses
                from meshom.straits import locate

                el = locate(sub, lonlat_to_xyz(line.lon, line.lat))
                el = np.unique(el[el >= 0])
                route = np.zeros(sub.n_nodes, bool)
                T_ = sub.n_tris
                route[sub.tris[el[el < T_]].ravel()] = True
                route[sub.quads[el[el >= T_] - T_].ravel()] = True
            d, rec["area"] = area_depth(sub, d, ins, axis, levels(b.levels), floor, Path(b.source), b.variable, h_f,
                                        held, route, tag, log,                           # B251: the band
                                        saddle_m=line.saddle[0] if line is not None else None,
                                        line=lonlat_to_xyz(line.lon, line.lat) if line is not None else None,
                                        open_ends=open_ is not None)
    rec["nodes_m"] = {"min": float(d[ins].min()), "max": float(d[ins].max())}
    return d, rec


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
    return out, {"held": (np.flatnonzero(nb), cap),
                 "barrier_m": B, "level_m": L, "next_level_m": Ln, "cap_m": cap, "chain_nodes": int(len(chain)),
                 "at": [round(float(v[0]), 4) for v in xyz_to_lonlat(sub.points[top][None])],
                 "capped": int(low.sum()), "capped_host_nodes": int((low & ~ins).sum()),
                 "cut_max_m": float((depth - out).max()), "nodes_star": int(nb.sum())}


REGION_STEP_M = 1000.0       # B223: a depth region's path is densified to this spacing for the distance to it
REGION_AREA_RINGS = 2        # B223 / B250: with depth_area a region's window holds this many more cell rings


def _canonical(e: np.ndarray) -> np.ndarray:
    """Elements in an order of their own: each turned to start at its smallest node id (orientation kept), the rows
    sorted."""
    if not len(e):
        return e
    k = e.shape[1]
    e = np.take_along_axis(e, (np.argmin(e, axis=1)[:, None] + np.arange(k)) % k, axis=1)
    return e[np.lexsort(e.T[::-1])]


def region_depth(mesh: Mesh, inside: np.ndarray, depth: np.ndarray, cfg, fmt: str, depth_channel: str = "",
                 sill_level: str = "", tag: str = "", log=print, depth_area: bool = False,
                 axis: np.ndarray | None = None, path: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """B216, B223: the nodes `inside` take an insert's own grid rule (rule_depth, B211): the box mean per node, no
    smoothing, no strait depth floor, the run's slope limiter, the depth floor and the regions; with `depth_channel`
    the channel raise (B213) and with `sill_level` the sill (B215 / B212), the channel line between the two largest
    pieces of the other nodes of their cells (B218; in open water, with `path`, from the path, B253); with
    `depth_area` the section areas (B250, the nodes `inside` playing the insert's own nodes, the sections across
    `axis` (K,3), the strait's path; None: skipped). Computed on their cells and one cell ring round them (with
    depth_area REGION_AREA_RINGS more, so that a section's mesh run is not cut at the window's edge), nodes ordered
    by position and cells by their nodes, so the depth is the same in every mesh that holds those cells. Every
    other node keeps `depth`. Returns (depth, record)."""
    T = mesh.n_tris

    def cells_of(nodes):
        return np.r_[nodes[mesh.tris].any(1) if T else np.zeros(0, bool), nodes[mesh.quads].any(1)]

    near = inside
    for _ in range(1 + (REGION_AREA_RINGS if depth_area else 0)):
        c = cells_of(near)
        near = np.zeros(mesh.n_nodes, bool)
        near[mesh.tris[c[:T]].ravel()] = True
        near[mesh.quads[c[T:]].ravel()] = True
    c = cells_of(near)                                        # the cells of `inside` and one ring (or more) round them
    wid = np.unique(np.r_[mesh.tris[c[:T]].ravel(), mesh.quads[c[T:]].ravel()])
    wid = wid[np.lexsort(mesh.points[wid].T[::-1])]
    loc = np.full(mesh.n_nodes, -1, np.int64)
    loc[wid] = np.arange(len(wid))
    sub = Mesh(mesh.points[wid], _canonical(loc[mesh.tris[c[:T]]].reshape(-1, 3)),
               _canonical(loc[mesh.quads[c[T:]]].reshape(-1, 4)), np.zeros(len(wid), np.int8))
    ins = inside[wid]
    d, rec = rule_depth(sub, ins, depth[wid], cfg, fmt, None, depth_at="node", depth_smooth=0, strait_floor=False,
                        sill_level=sill_level, depth_channel=depth_channel, tag=tag, log=log, depth_area=depth_area,
                        axis=axis, path=path)
    out = np.array(depth, dtype=np.float64)
    out[wid[ins]] = d[ins]
    rec["change_m"] = {"min": float((d - depth[wid])[ins].min()), "max": float((d - depth[wid])[ins].max())}
    return out, rec


def depth_region_nodes(mesh: Mesh, path: np.ndarray, width_m: float | None,
                       polygon: np.ndarray | None = None) -> np.ndarray:
    """B223: (N,) bool, the nodes of `mesh` inside the polygon (great-circle edges, meshom.size.polygon_inside), else
    those within width_m / 2 of the path (the great-circle distance to its points densified every REGION_STEP_M)."""
    from meshom import straits as stm

    if polygon is not None:
        from meshom.size import polygon_inside

        return polygon_inside(mesh.points, polygon)
    p = stm.densify(path, REGION_STEP_M)
    d, _ = cKDTree(p).query(mesh.points, distance_upper_bound=2.0 * np.sin(0.25 * width_m / R_EARTH_M) + 1e-12)
    return np.isfinite(d)


def depth_run(run_dir: Path, spec: Spec, cfg, mesh: Mesh, src: Path, straits: list, out_dir: Path, write: bool,
              log=print) -> dict:
    """B223, kind = "depth": no mesh change; the host nodes inside the polygon or within width_m / 2 of the path
    (depth_region_nodes) take the nest's grid rule (region_depth: B211's box mean per node, no smoothing, no strait
    floor; the entry's depth_channel, B213, sill_level, B215 / B212 / B249, and depth_area, B250, on sections across
    the entry's path, none for a polygon alone; in open water the joins and banks from the path, B253, unless the
    region is a polygon without a path), every other node keeps the host's depth (the run's rules,
    meshom.run.model_depth, or the host mesh's own saved depth when it is an insert's result, B217). Writes
    insert_<name>.npz (the host mesh), .host_to_new.npy (every node kept), .collar_moved.npy (none), .json (the
    record: `region`, `depth`) and fesom-insert_<name>/depth.npy, with model files when the run wrote them."""
    from meshom import compete
    from meshom import run as runmod

    fmt = cfg.mesh.format
    stem = out_dir / f"insert_{spec.name}"
    inside = depth_region_nodes(mesh, spec.path, spec.width_m, spec.polygon)
    where = "inside the polygon" if spec.polygon is not None else f"{spec.width_m / 1e3:.0f} km wide round the path"
    rec = {"run": str(run_dir), "mesh": str(src), "config": str(run_dir / "config.resolved.toml"), "kind": "depth",
           "spec": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in dataclasses.asdict(spec).items()
                    if k not in ("path", "polygon")} | {"path_lonlat": np.c_[xyz_to_lonlat(spec.path)].T.tolist()},
           "region": {"nodes": int(inside.sum()), "where": where,
                      "width_km": spec.width_m / 1e3 if spec.polygon is None else None,
                      "polygon": None if spec.polygon is None else np.c_[xyz_to_lonlat(spec.polygon)].round(4).tolist()}}
    if not inside.any():
        rec.update(status="refused", reason=f"no host node {where}")
        stem.with_suffix(".json").write_text(json.dumps(_jsonable(rec), indent=1))
        log(f"[insert {spec.name}] REFUSED: {rec['reason']}", flush=True)
        return rec
    depth = runmod.model_depth(mesh, cfg, fmt, straits, run_dir)
    host = host_depth_of(src, mesh.n_nodes)
    if host is not None:                                   # B217: a chain keeps the host mesh's own depth
        depth = np.array(host, dtype=np.float64)
        rec["host_depth"] = {"source": "the host mesh's own (an insert's result)", "nodes": int(mesh.n_nodes)}
    axis = None                                            # B250: the sections across the strait's path
    alone = (spec.polygon is not None and spec.path.shape == spec.polygon.shape   # a polygon alone: its outline
             and np.array_equal(spec.path, spec.polygon))
    if spec.depth_area and len(spec.path) >= 2 and not alone:
        axis = spec.path
    out, rec["depth"] = region_depth(mesh, inside, depth, cfg, fmt, spec.depth_channel, spec.sill_level,
                                     f"[insert {spec.name}]", log, spec.depth_area, axis,
                                     None if alone else spec.path)                # B253: open water from the path
    log(f"[insert {spec.name}] depth region {where}: {int(inside.sum())} "
        f"host nodes by the grid rule{', channel ' + spec.depth_channel if spec.depth_channel else ''}"
        f"{', sill ' + spec.sill_level if spec.sill_level else ''}{', section areas' if spec.depth_area else ''}: "
        f"{out[inside].min():.0f}-{out[inside].max():.0f} m "
        f"(before: {depth[inside].min():.0f}-{depth[inside].max():.0f} m); the mesh unchanged", flush=True)
    compete._save(mesh, stem.with_suffix(".npz"))
    np.save(stem.parent / (stem.name + ".host_to_new.npy"), np.arange(mesh.n_nodes, dtype=np.int64))
    np.save(stem.parent / (stem.name + ".collar_moved.npy"), np.zeros((2, 0), np.int64))
    fdir = out_dir / f"fesom-insert_{spec.name}"
    fdir.mkdir(exist_ok=True)
    np.save(fdir / "depth.npy", out)                       # after the mesh: B217's chain reads it
    rec["status"] = "ok"
    if write and (run_dir / "fesom").exists():
        checks = runmod.write_files(mesh, fmt, fdir, out, cfg.bathymetry.levels)
        rec["write"] = {"format": fmt, "dir": str(fdir), "depth_min_m": float(out.min()),
                        "depth_max_m": float(out.max()), **checks}
        log(f"[insert {spec.name}] wrote {fmt} files to {fdir}: {checks}", flush=True)
    stem.with_suffix(".json").write_text(json.dumps(_jsonable(rec), indent=1, default=str))
    return rec


def run(run_dir: Path, spec: Spec, out_dir: Path | None = None, mesh_name: str | None = None,
        write: bool = True, log=print) -> dict:
    """The insert in a finished run (module docstring): writes <out>/insert_<name>.npz, .json (the record),
    .score.json (zone scores and the ring's meshfix score), .map.png, .profile.png and, when the run wrote model
    files, fesom-insert_<name>/ in the run's format with depth by the run's bathymetry rules (insert_depth: the
    insert's own rule, B211-B215; on another insert's result its saved depth on the host nodes, B217). With fix_rings
    the spliced mesh is smoothed first (fix_spliced, B210). Also <out>/insert_<name>.host_to_new.npy (host node id ->
    id in the result, -1 dropped) and .collar_moved.npy (2, K): the ids in the result and the host ids of the host
    nodes the collar moved (B217). Returns the record."""
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
    if spec.kind == "depth":                                 # B223: a depth region, the mesh unchanged
        return depth_run(run_dir, spec, cfg, mesh, src, straits, out_dir, write, log)
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
    if spec.kind == "group":                                 # B240: the envelope and WINDOW_PAD_H host cells round it
        radius = half + WINDOW_PAD_H * h_host
    win = cut_window(mesh, c, radius)
    log(f"[insert {spec.name}] window {radius / 1e3:.0f} km: {win.mesh.n_nodes} nodes, {win.mesh.n_elems} elems",
        flush=True)
    orca = "orca" in cfg.mesh.method                       # B193: an ORCA run (its windows may still be meshom's)
    head = {"run": str(run_dir), "mesh": str(src), "config": str(run_dir / "config.resolved.toml"),
            "spec": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in dataclasses.asdict(spec).items()
                     if k != "path"} | {"path_lonlat": np.c_[xyz_to_lonlat(spec.path)].T.tolist()},
            "window_km": radius / 1e3, "h_min_m": h_min, "h_min_source": h_min_src}
    stem = out_dir / f"insert_{spec.name}"
    try:
        if spec.kind == "strait":
            res = insert_window(win, spec, h_run, cfg.size_tolerance() or 0.20, cfg.gates_all(), log, real=real,
                                h_min=h_min, orca=orca)
        elif spec.kind == "group":                           # B240: channels, patches and islands in one envelope
            from meshom.insert.group import insert_group

            res = insert_group(win, spec, log, real=real, h_min=h_min)
        else:                                                # B181 / B182: a coastal strip or a current corridor
            from meshom.insert.band import insert_band

            res = insert_band(win, spec, h_run, cfg.size_tolerance() or 0.20, cfg.gates_all(), log, real=real,
                              h_min=h_min, orca=orca)
    except InsertError as ex:                                # B254: a rule that fails after the joins (forced ones
        log(f"[insert {spec.name}] REFUSED {ex}", flush=True)   # too) is a refusal, not the end of the command
        rec = head | {"status": "refused", "refused": [str(ex)]}
        stem.with_suffix(".json").write_text(json.dumps(_jsonable(rec), indent=1))
        return rec
    rec = head | res.record
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
        cfid, chid, rec["collar_moved"] = collar_moved(win, res, full, sinfo)      # B217, before any fixer move
        if spec.kind == "group":                                     # B244: the blocks' nodes before any move
            from meshom.insert.group import ids_in

            res.group_ids = ids_in(full, res.group)
            np.savez(stem.parent / (stem.name + ".group.npz"),     # per channel: its block's and cross lines' ids
                     **{f"{w}{k}": v for k, (b, x) in enumerate(res.group_ids) for w, v in (("block", b), ("cross", x))})
        if spec.fix_rings > 0 and res.record.get("valid"):           # B210: stage a on the insert and its rings
            full, rec["fix_rings"], moved, res.window = fix_spliced(mesh, win, res, full, sinfo, spec, h_min, log)
        if spec.kind == "group" and res.record.get("valid") and (spec.even_sweeps or spec.split_deg):
            from meshom.insert.group import even_group, split_obtuse

            own = np.zeros(full.n_nodes, bool)
            own[full.n_nodes - sinfo["appended_nodes"]:] = True
            own[np.asarray(cfid, np.int64)] = False
            cut = np.zeros(full.n_nodes, bool)
            cut[sinfo["host_to_new"][win.old_id[win.kind == 2]]] = True
            tag = f"[insert {spec.name}]"
            if spec.even_sweeps:                                     # B252: even neighbour areas
                full, res.window, rec["even_areas"] = even_group(full, res.window, own, res.group_ids, cut, h_min,
                                                                 spec.even_sweeps, log=log, tag=tag,
                                                                 hold=spec.even_blocks)
            if spec.split_deg:                                       # B243: obtuse patch quads split
                full, res.window, rec["split_obtuse"] = split_obtuse(full, res.window, own, res.group_ids,
                                                                     spec.split_deg)
                log(f"{tag} split_deg {spec.split_deg:g}: {rec['split_obtuse']['split']} patch quads split into two "
                    f"triangles (worst corner {rec['split_obtuse']['worst_deg']:.1f} deg)", flush=True)
                if spec.even_sweeps and rec["split_obtuse"]["split"]:   # B252: the split triangles evened too
                    full, res.window, rec["even_areas_split"] = even_group(
                        full, res.window, own, res.group_ids, cut, h_min, spec.even_sweeps, log=log, tag=tag,
                        hold=spec.even_blocks)
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
    np.save(stem.parent / (stem.name + ".collar_moved.npy"), np.stack([cfid, chid]))
    rec["splice"] = sinfo
    rec["validity"] = {"host": measure.validity(mesh), "final": measure.validity(full)}
    compete._save(full, stem.with_suffix(".npz"))
    rec["status"] = "ok" if res.record.get("valid") else "invalid"
    if write and rec["status"] == "ok" and (run_dir / "fesom").exists():
        t = time.perf_counter()
        fmt = cfg.mesh.format
        meta = {}
        depth = runmod.model_depth(full, cfg, fmt, straits, run_dir, meta)
        spec_d = dataclasses.replace(spec, sill_level="", depth_channel="") if spec.kind == "group" else spec
        if spec.kind == "group" and spec.routes:                       # B246: the section areas along the routes
            spec_d = dataclasses.replace(spec_d, depth_area=False)
        depth, rec["depth"] = insert_depth(depth, mesh, win, res, full, sinfo, h2n, moved, spec_d, cfg, fmt, straits,
                                           run_dir, log, host=host_depth_of(src, mesh.n_nodes), collar=(cfid, chid))
        if (spec.kind == "group" and spec.routes and (spec.sill_level or spec.depth_channel or spec.depth_area)
                and cfg.bathymetry.source != "none"):
            from meshom.insert.group import route_depth                    # B246: per route, across the patches

            own = np.zeros(full.n_nodes, bool)
            own[full.n_nodes - sinfo["appended_nodes"]:] = True
            own[np.asarray(cfid, np.int64)] = False                        # the collar's moved host nodes
            depth, rec["depth"]["routes"] = route_depth(
                full, depth, own, spec.routes, cfg, spec.sill_level, spec.depth_channel, spec.depth_area,
                res.band["h_fine"], log, f"[insert {spec.name}]")
        elif spec.kind == "group" and (spec.sill_level or spec.depth_channel) and cfg.bathymetry.source != "none":
            from meshom.insert.group import branch_depth                   # B244: each channel's own sill

            depth, rec["depth"]["branches"] = branch_depth(
                full, depth, res.group_ids, cfg, spec.sill_level, spec.depth_channel,
                [c["cell_min_km"] * 1e3 for c in res.record["channels"]], log, f"[insert {spec.name}]")
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
             "joins": rec.get("joins"), "banks": rec.get("banks"), "bank_fallback": rec.get("bank_fallback")}
    stem.with_suffix(".score.json").write_text(json.dumps(_jsonable(score), indent=1, default=str))
    return rec
