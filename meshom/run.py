"""Batch runs from a config (`meshom run CONFIG`, `meshom runs [DIR]`): the pipeline the UI and scripts share.

run(cfg):
1. coast: [coast] source -> meshom.coast.Coast (loops with water on the left, merged across 180°; Antarctica from the
   GSHHS L5 / L6 file next to an L1 source, [coast] antarctica), or none;
2. size: the [[size.layers]] in file order (min; zone modes set / smooth act on the running h) -> clip to [size] h_min_m / h_max_m -> min with the strait zones ([coast] straits,
   [[straits]]: meshom.straits) and, with [coast] auto_straits, the automatic ones (meshom.lfs.auto_straits: h = W/n
   on the passages of the coast's medial axis that h_min keeps open) -> gradient limiter (grad_max) -> x scale;
3. domain: "none" -> the closed sphere; treatment "loops" -> the coast filtered to h (meshom.coast_filter: opening of
   the water at h, closing at h/2, strait corridors protected; stage coast_filter) and resampled at h; treatment
   "cut" -> the closed sphere, and after the method the elements whose centroid is on land are removed and the mask
   cleaned with FESOM2's rules (meshom.methods.m14.cut: spurs, pinched nodes, the largest basin), as fesom2_mesh_gen
   cuts JIGSAW; the strait corridors are water whatever the coast says and are protected in the clean-up
   (meshom.straits);
3b. hand-placed blocks ([[blocks]], meshom.methods.m14.hand.resolve): the corners snapped and the four sides laid
   on the sphere against the true coast (treatment "cut": the method sees no coast) or the filtered one (treatment
   "loops": the coast the method gets), h_m x scale, rows and cols / scale; given to the method as opts["blocks"]
   (the m14 methods build them as fixed blocks, B60); in the land cut the cells of a built block with a coast side
   are water and protected and the element across each of its fitted bank edges is land (as m14/cut.py), the other
   blocks are cut by the mask;
4. the method runs in a child process killed after [run] timeout (compete._generate / compete._spawn, shared with
   `meshom compete`), with [mesh] opts plus the run's size tolerance (Config.size_tolerance: 0.20 for m14 by
   default, 0 = off; decisions G38), then optionally the common smoothing;
5. scoring as in `meshom compete` (compete._score: quality summary, size fidelity, irregular counts against rho_c,
   gates; the size gate at the size tolerance, Config.gates_all), the standard figures when asked;
5b. the optimisation stage (meshom.optimize, decisions Z1): [mesh] optimize unset = the method's default
   (methods.own_optimize: on for the m14 methods), true / false forced. A method with its own stage (option
   "optimize": the m14 methods) runs it as its last stage, given [mesh] optimize and optimize_opts in its opts, so
   raw.npz is already optimised (record <run dir>/optimize.json) and no "optimized" record is made; otherwise (other
   methods, and the m14 methods under a land cut, which then run without it) the run applies it here on the final
   mesh (smoothed if smoothing ran) at the run's size tolerance with [mesh] optimize_opts, saved as optimized.npz,
   scored as the record "optimized" (its "optimize" block: per-step counts, times, before / after counts, checks)
   and from then on the final mesh (bathymetry, write, straits);
6. bathymetry ([bathymetry] source, meshom.bathymetry.node_depth): depth per node of the final mesh, positive down,
   at least the floor (min_depth_m, and for fesom2 / fesom3 the bottom of the min_layers-th level), then the depth
   rules of meshom.bathymetry.depth_policy (smoothing, slope limiter, the straits' tapered depth floors), the
   method's own node depths (m16_orca), then the [[bathymetry.regions]] rules in file order (model_depth, decisions
   O35, O36), saved as depth.npy;
7. the final mesh in [mesh] format: "fesom-c" (default; meshom.io.fesom.write_fesomc), "fesom2" (write_fesom2:
   nod2d.out, elem2d.out, aux3d.out with the [bathymetry] levels, then fesom2_checks on the written files) or
   "fesom3" (write_fesom3: 4-column elem2d.out and aux3d.out, then fesom3_checks, and fesom_ini with 1 partition
   when [mesh] fesom3_ini names the binary: run_fesom_ini, work folder fesom_ini/);
8. straits (when the config has any): meshom.straits.strait_report on the final mesh and depth, in stages.json (stage
   "straits") and in the final quality.json record ("straits"), with the water components of the final mesh
   ("water": count, nodes, area, box, the straits ending in each cut-off one, `connectivity_ok` = one component and
   every strait's two ends in it; a failed check is reported, the run goes on: decisions C44).

The run directory (`[mesh] out`, `--out`, default .tmp/runs/<config stem>) holds config.resolved.toml (every default
filled in, paths absolute), stages.json (one record per stage: coast, size, blocks, coast_filter, generate, post,
smooth, score, optimize, figures, bathymetry, write), quality.json ({"raw": record, "smooth": record,
"optimized": record}, records as `meshom compete` writes them),
raw.npz, smooth.npz, optimized.npz, depth.npy, fesom/ (the files of the final mesh) and figures/{raw,smooth}/.

write_mesh(run_dir, mesh_name, fmt, out, progress, fesom_ini) (`meshom write`): a saved mesh of a run folder (a hand
edit edited-NN.npz, raw, smooth, a repair) as model files with the run's config: depth recomputed (step 6; saved as
<out>/depth_base.npy, then the mesh's node edits <mesh stem>.depth_edits.json when there is one), files and
checks (step 7) into <run_dir>/fesom-<mesh stem>/, the mesh scored (step 5), the straits and passages checked (step
8, C51: check only), fesom_ini on request, all into its quality.json. Its caches in <run_dir>/cache/ (size.pkl: the
size field; rho_c.json: the rho_c integrals per domain; depth.npz: the GEBCO box sums) make a write after a small hand
edit take seconds.
"""
import datetime
import json
import pickle
import re
import shutil
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np

from meshom import bench, compete
from meshom.coast import Coast, read_coast
from meshom.config import (RUNS_DIR, CoastDistanceLayer, Config, ConfigError, ConstantLayer, LatitudeLayer,
                           OrcaLayer, ZoneLayer, save)
from meshom.domain import Domain, closed_sphere, in_water
from meshom.geo import R_EARTH_M, lonlat_to_xyz, xyz_to_lonlat
from meshom.mesh import Mesh, build_topology
from meshom.size import (Clipped, Latitude, Minimum, SetZone, SizeField, SmoothZone, Uniform, _fibonacci, blend_zone,
                         coast_distance, layered, limit_gradient, scaled, zone)
from meshom.straits import TRI_METHODS, StraitsField, straits_of

PLACEHOLDER_DEPTH_M = 4400.0      # depth of every node when [bathymetry] source = "none"


# ------------------------------------------------------------------------------------------------ inputs

REPO = Path(__file__).resolve().parents[1]
MAIN_CHECKOUT = REPO.parents[1] if REPO.parent.name == ".worktrees" else REPO


def source_path(src: str) -> Path:
    """[coast] source as an absolute path: relative to the working directory; a relative path not found there is
    taken relative to the main checkout's root (a run from a git worktree `.worktrees/<name>` finds
    `../coastlines/...` as the main checkout does)."""
    p = Path(src).expanduser()
    if p.is_absolute() or p.exists() or not (MAIN_CHECKOUT / p).exists():
        return p.resolve()
    return (MAIN_CHECKOUT / p).resolve()


def coast_of(cfg: Config) -> Coast | None:
    if cfg.coast.source == "none":
        return None
    return read_coast(str(source_path(cfg.coast.source)), cfg.coast.min_area_km2, cfg.coast.antarctica)


def layer_fields(cfg: Config, coast: Coast | None) -> list[tuple[str, SizeField]]:
    """(name, field) of every enabled [[size.layers]] entry, in order; meshom.size.layered combines them (min for a
    field; a zone of mode "set" or "smooth" is a step on the running h: SetZone, SmoothZone). A coast_distance layer
    measures the distance to its `coasts` class (Coast.select)."""
    base = cfg.base_h()
    out = []
    for i, ly in enumerate(cfg.size.layers):
        if not ly.enabled:
            continue
        if isinstance(ly, ConstantLayer):
            f = Uniform(ly.h_m)
        elif isinstance(ly, LatitudeLayer):
            f = Latitude(ly.h_m, ly.h_min_m)
        elif isinstance(ly, CoastDistanceLayer):
            f = coast_distance(coast.select(ly.coasts, ly.island_max_km2).points(0.25 * ly.h_m), ly.h_m,
                               ly.plateau_m, ly.grad, np.inf if base is None else base)
        elif isinstance(ly, OrcaLayer):                                 # B101: an ORCA grid's cell size
            from meshom.orca.sizes import orca_size

            f = orca_size(ly.orca, ly.h_min_m)
        elif ly.mode == "blend":                                        # S37: h_m to the running h between two lines
            f = blend_zone(_lonlat_xyz(ly.polygon), _lonlat_xyz(ly.outer), ly.h_m, ly.h_factor)
        elif ly.mode == "set":
            f = SetZone(_lonlat_xyz(ly.polygon), float(ly.h_m), float(ly.transition_m))
        elif ly.mode == "smooth":
            f = SmoothZone(_lonlat_xyz(ly.polygon), float(ly.smooth_m), float(ly.transition_m))
        else:
            h_out = ly.h_out_m if ly.h_out_m is not None else base
            if h_out is None:
                raise ConfigError(f"size.layers[{i}].h_out_m: needed without a constant layer or [size] h_max_m")
            f = zone(ly.polygon, ly.h_m, ly.transition_m, h_out)
        out.append((ly.name or ly.kind, f))
    return out


def _lonlat_xyz(polygon: list) -> np.ndarray:
    p = np.asarray(polygon, dtype=np.float64)
    return lonlat_to_xyz(p[:, 0], p[:, 1])


def h_floor(cfg: Config) -> float:
    """The smallest h the layers can take (before the limiter and the scale): sets the limiter's grid spacing."""
    lo = []
    for ly in cfg.layers():
        if isinstance(ly, ZoneLayer) and ly.mode == "smooth":      # an average of the running h: never below it
            continue
        if isinstance(ly, ZoneLayer) and ly.mode == "blend" and ly.h_factor is not None:   # B110: scales the running h
            if lo:
                lo.append(min(lo) * min(1.0, float(ly.h_factor)))
            continue
        lo.append(ly.h_min_m if isinstance(ly, (LatitudeLayer, OrcaLayer)) else ly.h_m)
        if isinstance(ly, ZoneLayer) and ly.mode == "min" and ly.h_out_m is not None:
            lo[-1] = min(lo[-1], ly.h_out_m)
    h = min(lo)
    return max(h, cfg.size.h_min_m) if cfg.size.h_min_m is not None else h


def background_h(cfg: Config, coast: Coast | None) -> SizeField:
    """The size field without strait zones, limiter and scale: the layers in file order (meshom.size.layered: min,
    zone modes set / smooth on the running h, which starts at the base h when the first layer is such a zone) ->
    clip. A widened strait (meshom.straits, profile "widen", decisions C45) takes its h_m from it."""
    base = cfg.base_h()
    h = layered([f for _, f in layer_fields(cfg, coast)], np.inf if base is None else base, 0.5 * h_floor(cfg))
    if cfg.size.h_min_m is not None or cfg.size.h_max_m is not None:
        h = Clipped(h, cfg.size.h_min_m or 0.0, np.inf if cfg.size.h_max_m is None else cfg.size.h_max_m)
    return h


def run_straits(cfg: Config, coast: Coast | None) -> list:
    """The run's straits (meshom.straits.straits_of) with their profiles, widened ones measured on background_h."""
    return straits_of(cfg, background_h(cfg, coast))


def size_field(cfg: Config, coast: Coast | None, info: dict | None = None,
               straits: list | None = None) -> SizeField:
    """min over the layers -> clip -> min with the strait zones (listed, then automatic) -> limit_gradient (unless
    grad_max is inf) -> x [mesh] scale. `straits`: the run's straits (run_straits; computed when not given); only
    the resolved ones get a zone (a widened strait has none, decisions C45).

    The strait zones (meshom.straits.StraitsField: h_m within half the corridor of the path, then linear to the base h
    with the slope grad_max) come after the clip, so [size] h_min_m does not undo them. The limiter's grid spacing is
    half the smallest h the layers and straits can take, times the scale (the mesh cannot resolve finer features of h
    anyway); `info` receives the limiter's grid record and the node estimate over the sphere."""
    h = background_h(cfg, coast)
    floor = h_floor(cfg)
    straits = straits_of(cfg, h) if straits is None else straits
    zoned = [s for s in straits if s.profile == "resolve"]
    if zoned:
        base = cfg.base_h()
        g = cfg.size.grad_max if np.isfinite(cfg.size.grad_max) else 0.1
        h = Minimum((h, StraitsField(tuple(zoned), np.inf if base is None else base, g)))
        floor = min(floor, min(s.h_m for s in zoned))
    if straits and info is not None:
        info["straits"] = len(straits)
        info["strait_profiles"] = {p: sum(s.profile == p for s in straits) for p in ("resolve", "widen")}
    if cfg.coast.auto_straits and coast is not None:                    # S22, S24: passages found, no hand marking
        from meshom.lfs import auto_straits

        base = cfg.base_h()
        g = cfg.size.grad_max if np.isfinite(cfg.size.grad_max) else 0.1
        zone, ainfo = auto_straits(coast.loops, h, cfg.coast.strait_n_min, g, cfg.size.h_min_m or 0.0,
                                   np.inf if base is None else base)
        if zone is not None:
            h = Minimum((h, zone))
            floor = min(floor, ainfo["h_zone_min_m"])
        if info is not None:
            info["auto_straits"] = ainfo
    if np.isfinite(cfg.size.grad_max):
        h = limit_gradient(h, None, cfg.size.grad_max, spacing_m=0.5 * floor * cfg.mesh.scale,
                           max_nodes=cfg.size.limiter_max_nodes)
        if info is not None:
            info["limiter"] = h.info
    if cfg.mesh.scale != 1.0:
        h = scaled(h, cfg.mesh.scale)
    if info is not None:
        lat = _fibonacci(200_000)
        info["n_nodes_est_sphere"] = float(4.0 * np.pi * R_EARTH_M**2 / len(lat) * np.sum(h(lat) ** -2.0))
    return h


def filtered_coast(cfg: Config, coast: Coast, h: SizeField, straits: list | None = None,
                   info: dict | None = None) -> Coast:
    """Treatment "loops": the coast filtered to h (meshom.coast_filter, decisions C16: opening of the water at h and
    closing at h/2, the strait corridors protected, the cut-off basins below [coast] keep_basins_km2 dropped; [coast]
    smoothing_m > 0 sets the closing radius in metres, 0 turns the filter off: the coast as read); `info` receives
    the filter's record."""
    if cfg.coast.smoothing_m == 0:
        if info is not None:
            info["filter"] = "off"
        return coast
    from meshom.coast_filter import filter_coast

    out, rec = filter_coast(coast, h, run_straits(cfg, coast) if straits is None else straits,
                            cfg.coast.smoothing_m or None,
                            cfg.coast.keep_basins_km2)
    if info is not None:
        info.update(rec)
    return out


def coast_domain(cfg: Config, coast: Coast, h: SizeField) -> Domain:
    """Treatment "loops": the filtered coast (filtered_coast) resampled at h (Coast.loops_at), as `run` builds it."""
    return Domain(filtered_coast(cfg, coast, h).loops_at(h), "coast")


# --------------------------------------------------------------------------------------------- land cut

def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def cut_land(mesh: Mesh, land: Domain, keep_basins_km2: float | None = None,
             straits: list | None = None, fixed: dict | None = None,
             passages: dict | None = None, wet_rule=None, keep_method_seas: bool = False) -> tuple[Mesh, dict]:
    """Remove the elements whose centroid is on land (in_water against the true coast loops), clean the element mask
    with FESOM2's rules and renumber (boundary loops first, flag 1): meshom.methods.m14.cut, rules cited there.
    Elements whose centroid lies in a strait corridor (meshom.straits.corridor_mask) are water and protected in the
    clean-up's main loop; so are the cells of the built fixed blocks with a coast side (`fixed`, the record
    m14_blocks.CUT_FIXED: "protect" element mask, "coast_nodes" bank nodes, "bank_edges" bank edges; B60), and the
    element across each of their bank edges is land, as m14's own cut (m14/cut.py fixed_banks).

    `passages` ({"cells": K, "h_bg": the background h, "quads", "tol"}; `[coast] passage_min_cells`, decisions C51):
    after the cut, every through-passage of the cut mesh with fewer than K cells across (meshom.passages) becomes a
    widened corridor (passage_straits) and the uncut mesh is cut again with them; up to passages.MAX_ROUNDS rounds, a
    passage found again inside an earlier corridor gets one cell more. info["passages"] records the rounds, the
    widened passages, the land cut away and the land components (islands) touched.

    `wet_rule(mesh) -> (ruled, wet)` or None (a method's own mask, m16_orca's `cut_wet`, B85): the ruled elements are
    water or land as `wet` says, whatever the coast; strait corridors do not act on them; the ruled water elements are
    protected in the clean-up's main loop as the corridors are (its final unprotected pass still applies); every
    water body follows C48 (the largest only by default; B171: the method's closed seas too); with keep_method_seas
    (m16_orca's `keep_seas`) a water body made only of ruled water elements (NEMO's Caspian) is kept whatever
    keep_basins_km2 says, unless it meets an unruled element (the joint: then it follows C48 as the rest, B95); the
    passage rule skips passages made only of ruled elements (B95: NEMO's straits stay as the method has them)."""
    out, info, land_wet, keep0 = _cut_once(mesh, land, keep_basins_km2, straits, fixed, wet_rule=wet_rule,
                                           keep_method_seas=keep_method_seas)
    if not passages or passages["cells"] < 2:
        return out, info
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from meshom.methods.m14.cut import element_areas, half_edges
    from meshom.passages import MAX_ROUNDS, cells_for, narrow_passages, passage_straits

    K = int(passages["cells"])
    extra, rounds, widened = [], [], []
    rec = None
    for rnd in range(MAX_ROUNDS + 1):
        skip = wet_rule(out)[0] if wet_rule is not None else None  # B95: passages only in the method's cells skipped
        rec = narrow_passages(out, K, straits=straits, skip=skip)
        found = rec["passages"]
        rounds.append({"n_found": len(found), "n_dead_end": rec["n_dead_end"], "n_inlet": rec["n_inlet"],
                       "n_pools": rec["n_pools"], "n_skipped": rec["n_skipped"]})
        if not found or rnd == MAX_ROUNDS:
            break
        cells = cells_for(found, extra, K)               # found again inside an earlier corridor: one cell more
        new = [s for p, c in zip(found, cells)
               for s in passage_straits([p], passages["h_bg"], c, passages["quads"], passages["tol"], len(extra))]
        for p, s in zip(found, new):
            widened.append({k: p[k] for k in ("kind", "at", "h_m", "cells_across", "basin_km2", "near")}
                           | {"round": rnd, "name": s.name, "corridor_m": round(s.corridor_m, 1),
                              "cells": s.cells, "h_bg_m": round(s.h_m, 1)})
            extra.append(s)
        out, info, _, keep = _cut_once(mesh, land, keep_basins_km2, (straits or []) + extra, fixed, land_wet,
                                       wet_rule, keep_method_seas)
    if not extra:
        keep = keep0
    areas = element_areas(mesh)
    cut = keep & ~keep0 & ~land_wet                      # real land made water by the passage corridors
    rejoined = keep & ~keep0 & land_wet                  # water the first clean-up had dropped, kept now
    lost = keep0 & ~keep                                 # water dropped now (a corridor's ragged edge)
    a, b, e, tw = half_edges(mesh)                       # land components (islands, continents) of the uncut mesh
    lnd = ~land_wet
    ok = (tw >= 0) & lnd[e] & lnd[e[np.maximum(tw, 0)]]
    g = coo_matrix((np.ones(int(ok.sum())), (e[ok], e[tw[ok]])), shape=(mesh.n_elems, mesh.n_elems))
    lab = connected_components(g, directed=False)[1]
    touched = np.unique(lab[cut])
    tot = np.bincount(lab[lnd], weights=areas[lnd], minlength=lab.max() + 1)
    gone = np.bincount(lab[cut], weights=areas[cut], minlength=lab.max() + 1)
    info["passages"] = {
        "cells": K, "rounds": rounds, "n_widened": len(widened), "n_left": len(rec["passages"]) if rec else 0,
        "left": [{k: p[k] for k in ("kind", "at", "h_m", "cells_across", "near")} for p in (rec or {}).get("passages", [])],
        "cut_km2": round(float(areas[cut].sum()) / 1e6, 1), "rejoined_km2": round(float(areas[rejoined].sum()) / 1e6, 1),
        "lost_km2": round(float(areas[lost].sum()) / 1e6, 1),
        "n_land_touched": int(len(touched)),
        "n_land_removed": int(sum(gone[c] >= 0.999 * tot[c] for c in touched)),
        "land_touched_km2": sorted((round(float(tot[c]) / 1e6, 1) for c in touched), reverse=True),
        "widened": widened}
    return out, info


def _cut_once(mesh: Mesh, land: Domain, keep_basins_km2: float | None = None,
              straits: list | None = None, fixed: dict | None = None,
              land_wet: np.ndarray | None = None, wet_rule=None,
              keep_method_seas: bool = False) -> tuple[Mesh, dict, np.ndarray, np.ndarray]:
    """One land cut (cut_land without the passage rounds): (mesh, info, the elements whose centroid is in the water
    of `land` (computed when not given; the ruled elements of `wet_rule` as it says), the elements kept)."""
    from meshom.methods.m14.cut import clean, element_areas, extract, fixed_banks, half_edges
    from meshom.straits import _components, corridor_mask

    P = mesh.points
    cent = _unit(np.r_[P[mesh.tris].sum(axis=1), P[mesh.quads].sum(axis=1)])
    rule = wet_rule(mesh) if wet_rule is not None else None
    if land_wet is None:
        land_wet = in_water(land, cent)
        if rule is not None:
            land_wet = np.where(rule[0], rule[1], land_wet)
    forced, per = corridor_mask(cent, straits) if straits else (None, {})
    if rule is not None and forced is not None:
        forced &= ~rule[0]                                # strait corridors act on the method's own cells only
    if rule is not None:
        rp = rule[0] & rule[1]
        forced = rp if forced is None else forced | rp
    wet = land_wet | forced if forced is not None else land_wet.copy()
    he = half_edges(mesh)
    if fixed is not None:
        inside = fixed["protect"]
        fixed_banks(he, wet, inside, fixed["bank_edges"], mesh.n_nodes)
        forced = inside if forced is None else forced | inside
    keep, cnt = clean(he, wet, mesh.n_nodes, element_areas(mesh), keep_basins_km2 * 1e6 if keep_basins_km2 else None,
                      forced)
    n_seas = 0
    if rule is not None and keep_method_seas:             # B85, B171: with keep_seas the method's own closed seas
        rw = rule[0] & rule[1]                            # (water bodies made only of its cells, NEMO's Caspian) stay
        lost = wet & ~keep & rw
        if lost.any():
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components

            a_, b_, e_, tw_ = he
            cand = keep | lost
            ok = cand[e_] & (tw_ >= 0) & cand[e_[np.maximum(tw_, 0)]]
            g = coo_matrix((np.ones(int(ok.sum())), (e_[ok], e_[tw_[ok]])), shape=(mesh.n_elems, mesh.n_elems))
            lab = connected_components(g, directed=False)[1]
            bad = np.unique(lab[cand & ~rw]) if (cand & ~rw).any() else np.zeros(0, np.int64)
            # B95: a sea of the method's cells that meets the rest of the mesh (the Caspian's ORCA half at the cap's
            # edge) follows the rest's rule (C48, the coast there): ours wins where they meet, no half seas
            hj = lost[e_] & (tw_ >= 0)
            tj = e_[np.maximum(tw_, 0)]
            joint = np.unique(lab[e_[hj & ~rule[0][tj]]])
            back = lost & ~np.isin(lab, bad) & ~np.isin(lab, np.unique(lab[keep])) & ~np.isin(lab, joint)
            cnt["method_seas_at_joint_dropped"] = int((lost & np.isin(lab, joint)).sum())
            keep = keep | back
            n_seas = int(len(np.unique(lab[back])))
            cnt["method_seas_kept"] = int(back.sum())
    out, _, loops = extract(mesh, keep)
    info = {"n_elems_before": mesh.n_elems, "n_removed_land": int((~wet).sum()),
            "n_removed_clean": int((wet & ~keep).sum()), "removed_clean_by": cnt, "n_loops": len(loops),
            "n_nodes": out.n_nodes, "n_components": int(_components(out).max()) + 1 if out.n_elems else 0}
    if fixed is not None:
        info["n_block_protected"] = int(inside.sum())
    if rule is not None:
        info.update(n_ruled=int(rule[0].sum()), n_ruled_wet=int((rule[0] & rule[1]).sum()),
                    n_ruled_removed=int((rule[0] & rule[1] & ~keep).sum()), n_method_seas_kept=n_seas)
    if straits:
        info.update(n_corridor=int(forced.sum()), n_forced_wet=int((forced & ~land_wet).sum()),
                    n_corridor_kept=int((forced & keep).sum()), corridor_per_strait=per)
    return out, info, land_wet, keep


def _boundary_domain(mesh: Mesh) -> Domain:
    from meshom.methods.m14.cut import boundary_loops

    loops = boundary_loops(mesh)
    return Domain([mesh.points[lp] for lp in loops], "cut", [True] * len(loops))


def hand_blocks(cfg: Config, coast: Coast | None, h: SizeField) -> list[dict]:
    """The config's [[blocks]] resolved on the sphere against the coast given (meshom.methods.m14.hand.resolve):
    h_m is multiplied by [mesh] scale and rows / cols are divided by it (rounded, >= 1), as h is."""
    from dataclasses import asdict

    from meshom.methods.m14.hand import resolve

    sc = cfg.mesh.scale
    out = []
    for b in cfg.blocks:
        d = asdict(b)
        if d["h_m"] is not None:
            d["h_m"] *= sc
        for k in ("rows", "cols"):
            if d[k] is not None and sc != 1.0:
                d[k] = max(1, int(round(d[k] / sc)))
        out.append(resolve(d, coast.loops if coast is not None else None, h))
    return out


def _worker(spec_json: str) -> None:
    """Child process: load (domain, h, land, straits, blocks, passages) from the run's inputs.pkl and run
    compete._generate (the land cut as its post step, with the passage rounds when `passages` is given). A method whose
    module sets TAKES_LAND gets the true coast as opts["land"] under a land cut (as the m14 methods, B70); one whose
    module defines passage_h(passages, opts) has the passage rule's h replaced by what it returns; one that defines
    cut_wet(mesh) sets the water of its own cells in the land cut (m16_orca, B85)."""
    from meshom.methods import METHODS, m14_blocks

    a = json.loads(spec_json)
    with open(a["inputs"], "rb") as f:
        domain, h, land, straits, blocks, passages = pickle.load(f)
    base = a["method"][:-4] if a["method"].endswith("_rep") and a["method"][:-4] in METHODS else a["method"]
    on_m14 = METHODS[base].__module__ == "meshom.methods.m14_blocks"     # m14 itself or composed on it (_rep)
    if not on_m14:
        blocks = []                                     # only the m14 methods (m14_blocks_rep included) build them
    mod = sys.modules[METHODS[base].__module__]
    if passages is not None and hasattr(mod, "passage_h"):  # a method's own h of the passage rule
        passages = dict(passages, h_bg=mod.passage_h(passages, a["opts"]))
    # the blocks protected in the land cut: those the method built (its record of this run), not every outline asked
    wet_rule = getattr(mod, "cut_wet", None)           # m16_orca: NEMO's mask on its ORCA cells (B85)
    seas = bool(getattr(mod, "cut_keeps_seas", lambda o: False)(a["opts"]))   # B171: m16_orca's keep_seas
    post = (lambda m: cut_land(m, land, a.get("keep_basins_km2"), straits, m14_blocks.CUT_FIXED if on_m14 else None,
                               passages, wet_rule, seas)) if land is not None else None
    opts = dict(a["opts"], blocks=blocks) if blocks else a["opts"]
    takes_land = on_m14 or getattr(mod, "TAKES_LAND", False)          # m16_orca: for its cap method
    if takes_land and land is not None:                 # B70: the placement counts the water of the cut that follows
        opts = dict(opts, land=land)
    compete._generate(a["method"], domain, h, opts, a["smooth"], Path(a["dir"]), post=post)


# ------------------------------------------------------------------------------------- depth, files, straits

def _method_node_depth(cfg: Config, run_dir: Path, mesh: Mesh, depth: np.ndarray) -> np.ndarray:
    """The depth with the method's own node depths where it has them: m16_orca with mask "fesom3" gives its ORCA
    nodes FESOM3's depth (the deepest of the four cells' NEMO levels, B94), in place of the GEBCO one; with
    `[mesh.opts] depth = "gebco"` (B221) it does not, and every node keeps the run's [bathymetry] depth."""
    if cfg.mesh.method != "m16_orca" or cfg.mesh.opts.get("depth", "nemo") == "gebco":
        return depth
    from meshom.methods.m16_orca import node_depth

    r = node_depth(run_dir, mesh)
    if r is None:
        return depth
    depth = np.array(depth, dtype=float)
    depth[r[0]] = r[1]
    return depth


def _nest_depth(cfg: Config, run_dir: Path, mesh: Mesh, depth: np.ndarray, fmt: str, meta: dict) -> np.ndarray:
    """m16_orca's `nest.depth = "gebco"` (B216): the nodes inside the nest's coarse box take meshom's grid rule
    (meshom.methods.m16_orca.nest_depth; record meta["nest_depth"]); every other node keeps `depth`."""
    if cfg.mesh.method != "m16_orca" or (cfg.mesh.opts.get("nest") or {}).get("depth", "") != "gebco":
        return depth
    from meshom.methods.m16_orca import nest_depth

    depth, meta["nest_depth"] = nest_depth(run_dir, mesh, depth, cfg, fmt)
    return depth


def depth_floor_m(cfg: Config, fmt: str) -> float:
    """The run's depth floor in format `fmt`: [bathymetry] min_depth_m, and for fesom2 / fesom3 at least the bottom of
    the min_layers-th level (bathymetry.min_depth_of, decisions O18)."""
    from meshom.bathymetry import levels, min_depth_of

    b = cfg.bathymetry
    zbar = levels(b.levels) if fmt in ("fesom2", "fesom3") else None
    return min_depth_of(b.min_depth_m, zbar, b.min_layers)


def mesh_depth(mesh: Mesh, cfg: Config, fmt: str, straits: list, meta: dict | None = None,
               cache: Path | None = None) -> np.ndarray:
    """Depth per node of `mesh` (m, positive down) with the config's [bathymetry] (the run's bathymetry stage):
    node_depth at the floor (depth_floor_m), then depth_policy; without a source every node gets PLACEHOLDER_DEPTH_M.
    `meta` receives the stage record. `cache`: node_depth's box-sum cache file (write_mesh: <run>/cache/depth.npz);
    the depth is the same with or without. model_depth adds the method's depths and the regions."""
    from meshom.bathymetry import cells_deepest_depth, depth_policy, node_depth

    b = cfg.bathymetry
    if b.source == "none":
        return np.full(mesh.n_nodes, PLACEHOLDER_DEPTH_M)
    meta = {} if meta is None else meta
    floor = depth_floor_m(cfg, fmt)
    if b.depth_at == "cells_deepest":                                                # B103
        depth, info = cells_deepest_depth(mesh, Path(b.source), b.variable, floor, cache=cache)
    else:                                                                             # O16; box_factor: B222
        depth, info = node_depth(mesh, Path(b.source), b.variable, floor, cache=cache, box_factor=b.box_factor)
    meta.update(info)
    depth, meta["policy"] = depth_policy(mesh, depth, floor, b.smooth_passes, b.max_slope, straits, b.carve_taper)
    return depth


def model_depth(mesh: Mesh, cfg: Config, fmt: str, straits: list, run_dir: Path, meta: dict | None = None,
                cache: Path | None = None) -> np.ndarray:
    """The node depth of the model files before any node edit (decisions O35): mesh_depth (box mean or
    cells_deepest, depth_policy), the method's own node depths (_method_node_depth, off with m16_orca's
    `depth = "gebco"`, B221; _nest_depth, B216; with a source), then the [[bathymetry.regions]] rules in file order
    at the run's floor (bathymetry.apply_regions; `meta["regions"]`: one record per region). The run's bathymetry
    stage and write_mesh both use it."""
    from meshom.bathymetry import apply_regions

    meta = {} if meta is None else meta
    depth = mesh_depth(mesh, cfg, fmt, straits, meta, cache)
    if cfg.bathymetry.source != "none":
        depth = _method_node_depth(cfg, run_dir, mesh, depth)
        depth = _nest_depth(cfg, run_dir, mesh, depth, fmt, meta)
    if cfg.bathymetry.regions:
        depth, meta["regions"] = apply_regions(mesh, depth, cfg.bathymetry.regions, depth_floor_m(cfg, fmt))
    return depth


def write_files(mesh: Mesh, fmt: str, path: Path, depth: np.ndarray, level_set) -> dict:
    """The mesh files in format `fmt` into `path` (the run's write stage without fesom_ini); returns the format's
    checks as {"fesom2_checks": ...} / {"fesom3_checks": ...} (none for fesom-c) and, for every format, the mesh's
    water record {"water": ...} (`water_record`, C52). More than one water component adds "water_components" to the
    checks' stop: FESOM's initial-state fill sweeps until every surface node has a value, so a cut-off
    piece without climatology keeps it sweeping forever."""
    from meshom.bathymetry import levels
    from meshom.io.fesom import fesom2_checks, fesom3_checks, write_fesom2, write_fesom3, write_fesomc

    path = Path(path)
    if fmt == "fesom2":
        write_fesom2(mesh, path, depth, levels(level_set))
        out = {"fesom2_checks": fesom2_checks(path)}
    elif fmt == "fesom3":
        write_fesom3(mesh, path, depth, levels(level_set))
        out = {"fesom3_checks": fesom3_checks(path)}
    else:
        flag = build_topology(mesh).boundary_node.astype(np.int8)
        write_fesomc(Mesh(mesh.points, mesh.tris, mesh.quads, flag), path, depth)
        out = {}
    water = water_record(mesh)[1]
    for c in out.values():
        c["water_components"] = water["n_components"]
        if not water["connectivity_ok"]:
            c["stop"].append("water_components")
    return {**out, "water": water}


def water_record(mesh: Mesh) -> tuple[np.ndarray, dict]:
    """(component of every element, the water record) of a mesh: its edge-connected water components
    (straits.water_components, C44); `connectivity_ok` = one component, the check of every written mesh (C52: a
    meshom mesh is global and keeps only the ocean-connected water, C48)."""
    from meshom.straits import water_components

    lab, comps = water_components(mesh)
    return lab, {"connectivity_ok": len(comps) == 1, "n_components": len(comps), "components": comps,
                 "not_connected": []}


def water_text(w: dict) -> str:
    """The failure line of a water record: each component cut off from the main water with its size and place."""
    return (f"WATER NOT CONNECTED: {w['n_components']} water components; "
            + "; ".join(f"#{c['id']} {c['n_elems']} elements, {c['n_nodes']} nodes, {c['area_km2']:.0f} km2, box "
                        f"{c['box']}" + (f", straits {c['straits']}" if c.get("straits") else "")
                        for c in w["components"][1:])
            + (f"; straits not connected: {w['not_connected']}" if w["not_connected"] else ""))


def straits_record(mesh: Mesh, straits: list, depth: np.ndarray | None) -> tuple[list, dict]:
    """(strait reports, water record) of the run's straits stage on `mesh` (depth None: no depth check): the water
    record of `water_record`, `connectivity_ok` also needs every strait connected."""
    from meshom.straits import strait_report

    lab, water = water_record(mesh)
    reps = strait_report(mesh, straits, depth, comp=lab)
    for r, s in zip(reps, straits):
        r["profile"] = s.profile
    for c in water["components"][1:]:      # the straits that end in a water cut off from the main one
        c["straits"] = [r["name"] for r in reps if c["id"] in r["ends_component"]]
    water["connectivity_ok"] = water["connectivity_ok"] and all(r["connected"] for r in reps)
    water["not_connected"] = [r["name"] for r in reps if not r["connected"]]
    return reps, water


# ------------------------------------------------------------------------------------------------------ run

OUTPUTS = ("raw.npz", "smooth.npz", "optimized.npz", "stages.json", "quality.json", "inputs.pkl", "depth.npy")


def resolved(cfg: Config, out: Path) -> Config:
    from meshom.bathymetry import LEVEL_SETS

    src = cfg.coast.source if cfg.coast.source == "none" else str(source_path(cfg.coast.source))
    b = cfg.bathymetry
    bsrc = b.source if b.source == "none" else str(source_path(b.source))
    lev = str(source_path(b.levels)) if isinstance(b.levels, str) and b.levels not in LEVEL_SETS else b.levels
    ini = str(Path(cfg.mesh.fesom3_ini).expanduser().resolve()) if cfg.mesh.fesom3_ini else ""
    return replace(cfg, mesh=replace(cfg.mesh, out=str(out), fesom3_ini=ini), coast=replace(cfg.coast, source=src),
                   bathymetry=replace(b, source=bsrc, levels=lev))


def run(cfg: Config, out: Path | None = None, _round: int = 0, _hist: list | None = None) -> dict:
    """Run the config's pipeline into its run directory; returns quality.json's content.

    [coast] passage_min_cells (C51), treatment "loops": when the final mesh has a through-passage below the cells, the
    run starts again with a widened `[[straits]]` entry along each (passage_<round>_<i>, `cells` = the key, one more
    where an earlier passage corridor did not suffice; the coast filter keeps the corridor water), at most
    passages.MAX_ROUNDS times (`_round`, `_hist`: the rebuilds so far). With "cut" the land cut repairs them itself
    (cut_land)."""
    out = (Path(out) if out else cfg.out_dir()).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for f in OUTPUTS:
        (out / f).unlink(missing_ok=True)
    for f in [*(out / "figures").glob("*/*"), *(out / "fesom").glob("*")]:
        shutil.rmtree(f) if f.is_dir() else f.unlink()           # fesom/dist_<N>/: the partitioner's folders
    cfg = resolved(cfg, out)
    save(cfg, out / "config.resolved.toml")
    smooth = cfg.run.smooth and compete.smooth_available()

    rec: list[dict] = []

    def flush() -> None:
        (out / "stages.json").write_text(json.dumps(rec))

    with bench.stage("coast", rec, source=cfg.coast.source, treatment=cfg.coast.treatment) as meta:
        coast = coast_of(cfg)
        meta.update(n_loops=len(coast.loops) if coast else 0,
                    n_vertices=int(sum(len(lp) for lp in coast.loops)) if coast else 0,
                    antarctica=coast.antarctica if coast else None)
    flush()
    with bench.stage("size", rec) as meta:
        straits = run_straits(cfg, coast)
        h = size_field(cfg, coast, meta, straits)
    flush()
    loops = None
    if coast is not None and cfg.coast.treatment == "loops":           # the coast the method and the blocks see
        with bench.stage("coast_filter", rec) as meta:
            coast_l = filtered_coast(cfg, coast, h, straits, meta)
            loops = coast_l.loops_at(h)
            meta.update(n_loops_at=len(loops), n_nodes_at=int(sum(len(lp) for lp in loops)))
        flush()
    blocks = []
    if cfg.blocks:
        with bench.stage("blocks", rec) as meta:
            blocks = hand_blocks(cfg, coast_l if loops is not None else coast, h)
            meta["blocks"] = [{"name": b["name"], "sides": b["kinds"], "snapped": b["snapped"],
                               "corners_lonlat": np.round(np.column_stack(xyz_to_lonlat(b["corners"])), 4).tolist()}
                              for b in blocks]
        flush()
    land = None
    if coast is None:
        domain = closed_sphere()
    elif loops is not None:
        domain = Domain(loops, "coast")
    else:
        domain, land = closed_sphere(), coast.domain()
    K = cfg.coast.passage_min_cells
    passages = ({"cells": K, "h_bg": background_h(cfg, coast), "quads": cfg.mesh.method not in TRI_METHODS,
                 "tol": float(cfg.size_tolerance() or 0.0)} if land is not None and K >= 2 else None)
    with open(out / "inputs.pkl", "wb") as f:
        pickle.dump((domain, h, land, straits, blocks, passages), f, protocol=pickle.HIGHEST_PROTOCOL)
    mopts = dict(cfg.mesh.opts)
    tol = cfg.size_tolerance()                                          # G38: the method's last stage gets it
    if tol is not None or cfg.mesh.size_tolerance is not None:          # (0 = off, passed on so m14 skips its 0.20)
        mopts["size_tolerance"] = tol or 0.0
    from meshom.methods import own_optimize

    own = own_optimize(cfg.mesh.method)                                 # Z1: the method's own last stage (m14)
    want = bool(cfg.mesh.optimize) if cfg.mesh.optimize is not None else bool(mopts.get("optimize", own))
    run_stage = want and (own is None or land is not None)              # else the method runs it (no double run)
    if own is not None:                                                 # a land cut follows: the stage after it
        mopts["optimize"] = want and land is None
        if cfg.mesh.optimize_opts:
            mopts["optimize_opts"] = dict(cfg.mesh.optimize_opts)
    spec = json.dumps({"method": cfg.mesh.method, "opts": mopts, "smooth": smooth, "dir": str(out),
                       "inputs": str(out / "inputs.pkl"), "keep_basins_km2": cfg.coast.keep_basins_km2})
    status, err = compete._spawn("import sys; from meshom.run import _worker; _worker(sys.argv[1])", spec,
                                 cfg.run.timeout)
    (out / "inputs.pkl").unlink(missing_ok=True)
    child = [x for x in json.loads((out / "stages.json").read_text()) if x["stage"] in ("generate", "post", "smooth")]
    stages = {x["stage"]: x for x in child}
    rec += child
    flush()

    result, final = {}, None
    raw_stage = "post" if "post" in stages else "generate"
    for key, stage, f in [("raw", raw_stage, "raw.npz")] + ([("smooth", "smooth", "smooth.npz")] if smooth else []):
        r = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "config": cfg.name,
             "method": cfg.mesh.method, "scale": cfg.mesh.scale, "smoothed": key == "smooth", "opts": mopts,
             **bench.machine_info()}
        if stage in stages and (out / f).exists():
            mesh = compete._load(out / f)
            gen = stages["generate"]["wall_s"] + (stages["post"]["wall_s"] if "post" in stages else 0.0)
            with bench.stage("score", rec, mesh=key):
                dom = _boundary_domain(mesh) if land is not None else domain
                rho = compete.rho_c_integrals(dom, h)
                r.update(status="ok", wall_s=gen + (stages["smooth"]["wall_s"] if key == "smooth" else 0.0),
                         generate_s=stages["generate"]["wall_s"], peak_rss_mb=stages[stage]["peak_rss_mb"],
                         **compete._score(mesh, dom, h, rho, cfg.gates_all()))
            final = (mesh, dom, key)
            if cfg.run.figures:
                from meshom.figures import render_run

                with bench.stage("figures", rec, mesh=key):
                    render_run(mesh, dom, h, out / "figures" / key,
                               title=f"{cfg.name} | {cfg.mesh.method} | scale {cfg.mesh.scale:g} | {key}")
        else:
            r.update(status=status if status != "ok" else "error", error=err or f"{f} missing")
        result[key] = compete._round(r)
        flush()
    if run_stage and final is not None:                                 # Z1: the optimisation stage
        from meshom.optimize import optimize, record

        mesh, _, key = final
        with bench.stage("optimize", rec, mesh=key) as meta:
            mesh, oinfo = optimize(mesh, h, tol, cfg.mesh.optimize_opts)
            meta.update(record(oinfo))
        compete._save(mesh, out / "optimized.npz")
        r = dict(result[key], date=datetime.datetime.now().isoformat(timespec="seconds"), input=key,
                 wall_s=result[key]["wall_s"] + rec[-1]["wall_s"], optimize_s=rec[-1]["wall_s"],
                 peak_rss_mb=rec[-1]["peak_rss_mb"])
        with bench.stage("score", rec, mesh="optimized"):
            dom = _boundary_domain(mesh) if land is not None else domain
            r.update(compete._score(mesh, dom, h, compete.rho_c_integrals(dom, h), cfg.gates_all()))
        r["optimize"] = record(oinfo)
        result["optimized"] = compete._round(r)
        final = (mesh, dom, "optimized")
        if cfg.run.figures:
            from meshom.figures import render_run

            with bench.stage("figures", rec, mesh="optimized"):
                render_run(mesh, dom, h, out / "figures" / "optimized",
                           title=f"{cfg.name} | {cfg.mesh.method} | scale {cfg.mesh.scale:g} | optimized")
        flush()
    (out / "quality.json").write_text(json.dumps(result, indent=1))
    if final is not None:
        mesh, _, key = final
        b = cfg.bathymetry
        if b.source != "none":
            with bench.stage("bathymetry", rec, mesh=key, depth_at=b.depth_at) as meta:
                depth = model_depth(mesh, cfg, cfg.mesh.format, straits, out, meta)
            np.save(out / "depth.npy", depth)
            flush()
        else:
            depth = model_depth(mesh, cfg, cfg.mesh.format, straits, out)
        with bench.stage("write", rec, mesh=key, format=cfg.mesh.format) as meta:
            meta.update(write_files(mesh, cfg.mesh.format, out / "fesom", depth, b.levels))
            if cfg.mesh.format == "fesom3" and cfg.mesh.fesom3_ini:
                from meshom.io.fesom import run_fesom_ini

                meta["fesom3_ini"] = run_fesom_ini(cfg.mesh.fesom3_ini, out / "fesom", out / "fesom_ini")
        flush()
        result[key]["water"] = meta["water"]                            # C52; the straits stage adds the straits
        (out / "quality.json").write_text(json.dumps(result, indent=1))
        if straits:
            with bench.stage("straits", rec, mesh=key) as meta:
                reps, water = straits_record(mesh, straits, depth if b.source != "none" else None)
                meta.update(straits=reps, n_ok=sum(r["ok"] for r in reps),
                            n_connected=sum(r["connected"] for r in reps), water=water)
            result[key]["straits"] = reps
            result[key]["water"] = water
            (out / "quality.json").write_text(json.dumps(result, indent=1))
        if coast is not None and cfg.coast.passage_min_cells >= 2:      # C51: judged on the final mesh
            from meshom.passages import MAX_ROUNDS, narrow_passages

            with bench.stage("passages", rec, mesh=key) as meta:
                pr = narrow_passages(mesh, cfg.coast.passage_min_cells, straits=straits)
                narrow = [{k: v for k, v in p.items() if k != "path"} for p in pr["passages"]]
                post = stages.get("post", {}).get("passages", {})
                meta.update(cells=pr["cells"], n_narrow=len(narrow), n_dead_end=pr["n_dead_end"],
                            n_inlet=pr["n_inlet"], n_pools=pr["n_pools"], narrow=narrow)
                meta["rebuilds"] = _hist or []
            result[key]["passages"] = {"cells": pr["cells"], "n_narrow": len(narrow), "narrow": narrow,
                                       "n_widened": post.get("n_widened", 0), "cut_km2": post.get("cut_km2", 0.0),
                                       "rebuilds": _hist or []}
            (out / "quality.json").write_text(json.dumps(result, indent=1))
            if narrow and cfg.coast.treatment == "loops" and _round < MAX_ROUNDS:
                from meshom.config import StraitCfg
                from meshom.passages import cells_for

                prev = [s for s in straits if s.name.startswith("passage_")]
                extra = []
                for i, (p, c) in enumerate(zip(pr["passages"], cells_for(pr["passages"], prev, pr["cells"]))):
                    lon, lat = xyz_to_lonlat(p["path"])
                    extra.append(StraitCfg(name=f"passage_{_round}_{i}", cells=c, profile="widen",
                                           path=np.round(np.column_stack([lon, lat]), 5).tolist()))
                flush()
                hist = (_hist or []) + [{"round": _round, "n_narrow": len(narrow), "nodes": mesh.n_nodes,
                                         "narrow": narrow}]
                return run(replace(cfg, straits=list(cfg.straits) + extra), out, _round + 1, hist)
    flush()
    return result


# ---------------------------------------------------------------------------------------- write a saved mesh

EDITED = re.compile(r"edited-(\d+)\.npz")


def mesh_file(run_dir: Path, mesh_name: str | None = None) -> Path:
    """The mesh file of `meshom write`: None = the newest edited-NN.npz of the run (largest NN), else the run's final
    mesh (optimized.npz, else smooth.npz, else raw.npz: the order of run()'s final mesh, the one in its fesom/); a name
    (raw, smooth, optimized, edited-NN, repaired, with or without .npz) = that file in the run directory; any other
    path ending in .npz is taken as given."""
    run_dir = Path(run_dir)
    if mesh_name is None:
        nn = [(int(m.group(1)), p) for p in run_dir.glob("edited-*.npz") if (m := EDITED.fullmatch(p.name))]
        if nn:
            return max(nn)[1]
        return next((run_dir / f for f in ("optimized.npz", "smooth.npz") if (run_dir / f).exists()),
                    run_dir / "raw.npz")
    p = Path(mesh_name).expanduser()
    if p.suffix == ".npz" and (p.is_absolute() or p.exists()):
        return p.resolve()
    return run_dir / (mesh_name if mesh_name.endswith(".npz") else f"{mesh_name}.npz")


SIZE_CACHE_VERSION = 1      # bump when what write_inputs builds changes: older caches are then rebuilt


def write_inputs(run_dir: Path, cfg: Config, meta: dict | None = None) -> tuple[list, SizeField, list | None, str]:
    """(straits, h, domain loops, key) of a run folder for write_mesh, as the run builds them: run_straits,
    size_field, and for treatment "loops" the coast filtered to h (filtered_coast(...).loops_at(h); None for "cut"
    and without a coast). Taken from <run_dir>/cache/size.pkl when its key (SIZE_CACHE_VERSION and the sha256 of
    config.resolved.toml) is the run's, else built and saved there (decisions O33). `meta` gets "cache": "hit" or
    "built"."""
    import hashlib
    import os

    key = f"{SIZE_CACHE_VERSION}:{hashlib.sha256((run_dir / 'config.resolved.toml').read_bytes()).hexdigest()}"
    f = run_dir / "cache" / "size.pkl"
    meta = {} if meta is None else meta
    if f.exists():
        try:
            with open(f, "rb") as fh:
                c = pickle.load(fh)
            if c["key"] == key:
                meta["cache"] = "hit"
                return c["straits"], c["h"], c["loops"], key
        except Exception:                   # a cache of older code that no longer unpickles: rebuilt below
            pass
    coast = coast_of(cfg)
    straits = run_straits(cfg, coast)
    h = size_field(cfg, coast, None, straits)
    loops = None
    if coast is not None and cfg.coast.treatment != "cut":
        loops = filtered_coast(cfg, coast, h, straits).loops_at(h)
    f.parent.mkdir(exist_ok=True)
    tmp = f.with_name(f.name + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump({"key": key, "straits": straits, "h": h, "loops": loops}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, f)
    meta["cache"] = "built"
    return straits, h, loops, key


def write_rho_c(run_dir: Path, key: str, dom: Domain, h: SizeField, meta: dict | None = None) -> dict:
    """compete.rho_c_integrals(dom, h) (~20 s on a 281 k-node Arctic mesh) through <run_dir>/cache/rho_c.json, one
    entry per (size key of write_inputs, the domain's loops as bytes): a hand edit that leaves the boundary as it
    is gets the stored value, which is the same as a new integration (it is a deterministic function of the two).
    `meta` gets "rho_c_cache": "hit" or "built"."""
    import hashlib
    import os

    hs = hashlib.sha256(key.encode())
    for lp in dom.loops:
        hs.update(np.ascontiguousarray(lp, dtype=np.float64).tobytes() + b"|")
    k = hs.hexdigest()
    f = run_dir / "cache" / "rho_c.json"
    c = json.loads(f.read_text()) if f.exists() else {}
    meta = {} if meta is None else meta
    if k in c:
        meta["rho_c_cache"] = "hit"
        return c[k]
    c[k] = compete.rho_c_integrals(dom, h)
    f.parent.mkdir(exist_ok=True)
    tmp = f.with_name(f.name + ".tmp")
    tmp.write_text(json.dumps(c))
    os.replace(tmp, f)
    meta["rho_c_cache"] = "built"
    return c[k]


def write_mesh(run_dir: Path, mesh_name: str | None = None, fmt: str | None = None, out: Path | None = None,
               progress: Callable[[str], None] | None = None, fesom_ini: bool = False) -> dict:
    """`meshom write`: a saved mesh of a run folder (a hand edit edited-NN.npz, raw, smooth, a repair; mesh_file)
    written as model files with the run's config (config.resolved.toml): the format `fmt` (default the run's [mesh]
    format), the depth recomputed on the mesh's nodes with the run's [bathymetry] (mesh_depth, as the bathymetry
    stage, model_depth: the regions included; source "none" = 4400 m) and saved as <out>/depth_base.npy (C3,
    decisions O38), then the node edits of the mesh file, <mesh stem>.depth_edits.json next to it, when there is one
    (bathymetry.apply_node_edits, C2, O37; record "depth_edits"), the files written into `out` (default
    <run_dir>/fesom-<mesh stem>/, never the run's own fesom/) with the format's checks, the mesh scored as the run
    scores its final mesh (compete._score with the run's gates and size tolerance, Config.gates_all; domain: the
    closed sphere, the filtered coast for treatment "loops", the mesh's own boundary for "cut"), plus the straits
    report when the config has straits and the passages check when the run has a coast and [coast]
    passage_min_cells >= 2 (record "passages": {cells, n_narrow, narrow} as the run's, check only: the mesh is
    written as it is, never widened).
    Writes <out>/quality.json = {<mesh stem>: record}; returns the record.

    Caches in <run_dir>/cache/ (decisions O33): size.pkl (the straits, h and domain loops; write_inputs), rho_c.json
    (the rho_c integrals per domain; write_rho_c) and depth.npz (the GEBCO box sums per node box;
    bathymetry.cached_box_means), so a write after a small edit reads only the boxes of the moved nodes and their
    neighbours; the depth equals a full recomputation bit for bit.
    `progress(stage)` is called as each stage starts (size, score, bathymetry, write, straits, passages, fesom_ini).
    `fesom_ini`: with format fesom3 and [mesh] fesom3_ini set, run fesom_ini on the written folder as the run does
    (io.fesom.run_fesom_ini, 1 partition), work folder <run_dir>/fesom_ini-<mesh stem>/, its record "fesom_ini"."""
    run_dir = Path(run_dir).expanduser().resolve()
    src = mesh_file(run_dir, mesh_name)
    name = src.stem
    out = Path(out).expanduser().resolve() if out else run_dir / f"fesom-{name}"
    if out == run_dir / "fesom":
        raise ValueError(f"{out}: the run's own fesom/ is never overwritten")
    from meshom.config import load

    cfg = load(run_dir / "config.resolved.toml")
    fmt = fmt or cfg.mesh.format
    mesh = compete._load(src)
    st: list[dict] = []

    def stage(stage_name: str, **kw):
        if progress is not None:
            progress(stage_name)
        return bench.stage(stage_name, st, **kw)

    with stage("size") as meta:
        straits, h, loops, key = write_inputs(run_dir, cfg, meta)
    if cfg.coast.source == "none":
        dom = closed_sphere()
    elif loops is None:
        dom = _boundary_domain(mesh)
    else:
        dom = Domain(loops, "coast")
    rec = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "config": cfg.name,
           "method": cfg.mesh.method, "scale": cfg.mesh.scale, "mesh": str(src), "format": fmt, "out": str(out),
           "status": "ok", **bench.machine_info()}
    with stage("score") as meta:
        rec.update(compete._score(mesh, dom, h, write_rho_c(run_dir, key, dom, h, meta), cfg.gates_all()))
    b = cfg.bathymetry
    with stage("bathymetry", source=b.source) as meta:
        depth = model_depth(mesh, cfg, fmt, straits, run_dir, meta, cache=run_dir / "cache" / "depth.npz")
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "depth_base.npy", depth)                          # C3, decisions O38
        edits = src.with_name(f"{name}.depth_edits.json")
        if edits.exists():                                              # C2, decisions O37
            from meshom.bathymetry import apply_node_edits

            depth, rec["depth_edits"] = apply_node_edits(depth, edits, depth_floor_m(cfg, fmt))
            meta["depth_edits"] = rec["depth_edits"]
    with stage("write", format=fmt):
        checks = write_files(mesh, fmt, out, depth, b.levels)
    rec.update(checks)
    if straits:
        with stage("straits"):
            rec["straits"], rec["water"] = straits_record(mesh, straits, depth if b.source != "none" else None)
    K = cfg.coast.passage_min_cells
    if cfg.coast.source != "none" and K >= 2:                           # C51, as the run's passages stage
        from meshom.passages import narrow_passages

        with stage("passages") as meta:
            pr = narrow_passages(mesh, K, straits=straits)
            narrow = [{k: v for k, v in p.items() if k != "path"} for p in pr["passages"]]
            meta.update(n_narrow=len(narrow), n_dead_end=pr["n_dead_end"], n_inlet=pr["n_inlet"],
                        n_pools=pr["n_pools"])
        rec["passages"] = {"cells": pr["cells"], "n_narrow": len(narrow), "narrow": narrow}
    if fesom_ini and fmt == "fesom3" and cfg.mesh.fesom3_ini:
        from meshom.io.fesom import run_fesom_ini

        with stage("fesom_ini"):
            rec["fesom_ini"] = run_fesom_ini(cfg.mesh.fesom3_ini, out, run_dir / f"fesom_ini-{name}")
    rec["stages"] = st
    rec = compete._round(rec)
    (out / "quality.json").write_text(json.dumps({name: rec}, indent=1))
    return rec


def write_line(rec: dict) -> str:
    """One summary line of a write_mesh record."""
    q, ic = rec["quality"], rec["irregular"]
    c = rec.get("fesom2_checks") or rec.get("fesom3_checks")
    chk = "no checks" if c is None else ("checks ok" if not c["stop"] else f"checks STOP {c['stop']}")
    return (f"{Path(rec['mesh']).name} -> {rec['out']} ({rec['format']}): {q['n_nodes']} nodes, {q['n_tris']} tris, "
            f"{q['n_quads']} quads; Euler {ic['euler_sum']} ({'ok' if ic['euler_ok'] else 'FAIL'}); {chk}; gates "
            f"{rec['gates_passed']}/{rec['gates_total']} {sorted(rec['gates_failed'])}"
            + (f"; {rec['passages']['n_narrow']} passages below {rec['passages']['cells']} cells"
               if "passages" in rec else "")
            + (f"; fesom_ini {'ok' if rec['fesom_ini']['ok'] else 'FAIL'}" if "fesom_ini" in rec else "")
            + (f"; {water_text(rec['water'])}" if not rec.get("water", {}).get("connectivity_ok", True) else ""))


# -------------------------------------------------------------------------------------------------- listing

def run_line(d: Path, root: Path) -> str:
    """One line per run directory: status, method, nodes, elements, gates, time, date (quality.json, stages.json)."""
    name = str(d.relative_to(root)) if d != root else d.name
    q = json.loads((d / "quality.json").read_text()) if (d / "quality.json").exists() else None
    if q is None:
        st = json.loads((d / "stages.json").read_text()) if (d / "stages.json").exists() else []
        last = st[-1]["stage"] if st else "-"
        return f"{name:<32} {'running?':<8} (no quality.json; last stage: {last})"
    r = q.get("optimized", q.get("smooth", q["raw"]))
    head = f"{name:<32} {r['status']:<8} {r['method']:<24}"
    if r["status"] != "ok":
        return head + f" {r['error'].splitlines()[-1][:80] if r.get('error') else ''}"
    s = r["quality"]
    return head + (f" {s['n_nodes']:>9} nodes {s['n_quads']:>9} quads {s['n_tris']:>8} tris  gates "
                   f"{r['gates_passed']}/{r['gates_total']}  {r['wall_s']:8.1f} s  "
                   f"{'optimized' if 'optimize' in r else 'smoothed' if r['smoothed'] else 'raw':<8} {r['date']}")


def list_runs(root: Path = RUNS_DIR) -> list[str]:
    """Run directories (those holding config.resolved.toml) at or below root, one line each."""
    root = Path(root)
    dirs = sorted({p.parent for p in root.rglob("config.resolved.toml")}) if root.is_dir() else []
    return [run_line(d, root) for d in dirs]
