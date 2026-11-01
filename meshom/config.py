"""The meshom config: one TOML file that drives a batch run (`meshom run CONFIG`) and that the UI reads and writes.

docs/config.md has the full description and an example. Tables (units: metres, km^2 for areas, seconds):

    [mesh]    method (a meshom.methods.METHODS name), opts (table, given to the method as is), out (run directory,
              default .tmp/runs/<config stem>), scale (default 1: the final h is multiplied by it, as in
              `meshom compete --scale`), format ("fesom-c" default: the FESOM-C mixed files; "fesom2": the FESOM2
              triangle files nod2d.out, elem2d.out, aux3d.out; "fesom3": FESOM3 = fesom_F, 4-column elem2d.out and
              aux3d.out, meshom.io.fesom), fesom3_ini (path of FESOM3's fesom_ini, run on the fesom3 folder with 1
              partition in the write stage; "" = off, the default), size_tolerance (none = the method's
              default: 0.20 for the m14 methods, else off; 0 = off: the method fits the cell size to h; a number t =
              the allowed |ln(s/h)| per edge: the method's last stage makes the size change between neighbours as
              gentle as possible within it, meshom.size_fit.smooth_fit, given to the method as
              opts["size_tolerance"], and the size_fidelity_p99 gate becomes t unless [gates] sets it;
              decisions G38, M38), optimize (the optimisation stage meshom.optimize, decisions Z1; unset = the
              method's default: on for the m14 methods, which run it as their own last stage, off for the others;
              true / false (1 / 0) force it on / off; a method without its own stage, and every method under a land
              cut of [coast] treatment "cut", gets it from the run after the method and the smoothing, scored as
              "optimized"), optimize_opts (its options, meshom.optimize.DEFAULTS)
    [coast]   source ("none" = the closed sphere, or a land-polygon shapefile such as GSHHS level 1), min_area_km2
              (smaller polygons are dropped), treatment ("cut": mesh the closed sphere, then remove the elements on
              land with FESOM2's clean-up rules; "loops": the coast filtered to h, meshom.coast_filter, then
              resampled at h bounds the domain), smoothing_m (loops: the filter's closing radius in metres, the
              opening's is twice it; none = h/2 and h from the local h; 0 = no filter), antarctica
              ("ice_front" default, "grounding", "none": GSHHS L1 has no Antarctica, it is added from the L5 / L6
              file next to the L1 source, meshom.coast), straits ("none" default, or the name of a built-in strait
              list, "fesom2_mesh_gen": meshom.straits.BUILTIN), strait_h_min_m (every strait's h_m is raised to it;
              default 0), auto_straits (default false: the passages of the coast's medial axis get the size zone
              h = W/n, meshom.lfs.auto_straits), strait_n_min (5: cells across an automatic strait),
              strait_profile ("resolve" default: the size zone h_m along each listed strait; "widen": no zone, the
              corridor at >= strait_widen_cells (2) cells of the background h; meshom.straits, decisions C45),
              passage_min_cells (2: every through-passage of the final mesh, listed or not, at least this many
              cells across, widened into the land by the cut; 0 = off; meshom.passages, decisions C51)
    [[straits]]  channels kept open (meshom.straits): name, path = [[lon, lat], ...] (the centreline), width_m,
              h_m, cells (default 2), depth_m (a depth floor), profile ("resolve" | "widen": wins over the regions
              and [coast] strait_profile), enabled; an entry whose name is in the built-in list overrides the
              fields it gives (enabled = false drops that strait), other entries are added
    [[strait_regions]]  which straits are opened and how (C46): profile ("resolve" | "widen" | "off"), name, box =
              [west, east, south, north] or polygon = [[lon, lat], ...]; the first region holding a strait's middle
              point wins; with any region, a strait in none is not opened
    [[inserts]]  curvilinear inserts fitted into the finished run by `python -m meshom.insert` (not by
              `meshom run`; decisions B158-B169, B180-B185): kind ("strait" | "strip" | "corridor" | "group"), name, strait or
              path, mouth_a / mouth_b (join hints), cells or h_fine_m, reach_km, force, stages, ring_rows,
              bank_sigma_h, width_km and band_km (strip, corridor), ljoins, cells_a / cells_b, growth_a / growth_b,
              bank_land_cells (B200-B202), fill, fill_rows (B204), corner_steps (B206), add_columns (B207),
              corner_split (B208), join_margin (B209), fix_rings (B210), depth_at, depth_smooth, strait_floor
              (B211), sill_level (B212, B215), depth_channel (B213), depth_quantile (B214), depth_area
              (B250); kind "depth": a depth region, the mesh unchanged (width_km round the path, or
              polygon; B223); group: envelope,
              channels (B240, B242), gap_km, island_km2 (B241), growth, gap_cells (B243), bank_smooth_cells (B245),
              routes (B246), blend_cells (B247)
              (meshom.config.InsertCfg)
    [size]    grad_max (|grad h| limit of the gradient limiter, default 0.1; inf = no limiting), h_min_m / h_max_m
              (optional global clip), preview_deg (UI preview grid), limiter_max_nodes (grid size cap), from (an h
              variant file, relative to the config's folder: its [size] keys and layers first, this table's keys
              over them, its layers after them; `load` inlines it, decisions S35; loads_size / dumps_size)
    [[size.layers]]  applied in file order to a running h (min, or a zone's mode; decisions S34); `kind` selects:
              constant        h_m
              latitude        h_m (at the equator), h_min_m                 h = max(h_m cos(lat), h_min_m)
              coast_distance  h_m (at the coast), plateau_m, grad            h = h_m + grad max(d - plateau_m, 0)
                              coasts ("all" | "continents" | "islands": d to that class), island_max_km2 (1e5:
                              a land polygon smaller on the sphere is an island; Antarctica a continent; S33)
              zone            h_m, transition_m, polygon = [[lon, lat], ...], h_out_m (default: the base h), mode:
                              "min" (default) h = h_m inside (great-circle edges), linear to h_out_m over transition_m;
                              "set" h_m inside whatever the layers before give, linear to the running h over
                              transition_m; "smooth" (smooth_m) the running h with ln h averaged over the radius
                              smooth_m inside, blended to the unsmoothed h over transition_m (S36); "blend" (outer
                              = [[lon, lat], ...]) h_m inside, the running h outside `outer`, between them
                              ln h linear in t = d_in / (d_in + d_out) (S37; transition_m not used); with
                              h_factor in place of h_m: h_factor x the running h inside, ln h = ln h_run +
                              (1 - t) ln h_factor between (B110: an ORCA grid refined k times inside, 1 / k)
              orca            orca (a preset), h_min_m: h = the preset's cell size, sqrt(area) (B101)
              every layer takes an optional name and enabled (default true; false: kept in the file, skipped)
    [[blocks]]  hand-placed curvilinear blocks, fixed blocks of the m14 methods (meshom.methods.m14.hand, B60):
              name, corners = [[lon, lat] x 4] (counter-clockwise, the block on the left of each side), rows (cells
              along side 1-2), cols (along side 2-3; both default: from h), h_m (the cell size inside; default h),
              sides (4 of "coast", "cut", "lat", "offset"; default per side: coast when both corners lie within 1 h
              of one coast loop and the coast path between them is shorter than 3 x the chord, else cut), snap_m
              (corners of coast sides within it move onto the coast; default 1 h), filter_m (the coast sides' filter;
              default 2 h)
    [gates]   overrides of DEFAULT_GATES (quality gates)
    [bathymetry]  source ("none" = no bathymetry stage, the FESOM files get a placeholder depth; else a netCDF grid
              of elevation, positive up, such as GEBCO 2024), variable ("elevation"), min_depth_m (10: shallower
              wet nodes are set to it), depth_at ("node": depth per node, the mean of the grid's ocean cells in the
              node's control cell, meshom.bathymetry; "cells_deepest": per element the mean of its ocean cells, a
              node the deepest of its elements, FESOM3's ORCA rule, B103), levels (the FESOM2 level surfaces zbar, metres: a list, a named
              set, default "core2" = the 48 CORE2 levels, or a file with nl then nl values such as an aux3d.out),
              min_layers (4: fesom2 / fesom3 depth >= zbar[4] = 30 m), smooth_passes (0 = off), max_slope (0 = off),
              carve_taper (1: the strait floors' cosine taper in strait h_m; meshom.bathymetry.depth_policy)
    [[bathymetry.regions]]  depth rules by area (C1, decisions O35, O36), after every other depth rule, in file
              order: name, box = [lon_min, lat_min, lon_max, lat_max] (lon_min > lon_max: across 180) or polygon =
              [[lon, lat], ...]; depth_m (set), min_depth_m, max_depth_m (in that order); then the floor
              (meshom.bathymetry.apply_regions)
    [run]     timeout (s, the method's child process is killed after it), smooth (also run the common smoothing),
              figures (render the standard picture set)

The base h (`Config.base_h`) is the largest enabled constant layer, else h_max_m, else none: coast_distance layers stop
growing there and zones grow to it. `load` returns a Config of typed dataclasses; a key that is not in the schema is
an error naming its path (e.g. "size.layers[2].transiton_m: unknown key"). Values are not checked further (input
scope: the user fixes the config). `dumps(load(p))` loads back equal; `enabled = true` and empty names are not written.

Each gate is evaluated on the whole mesh and, where the metric has them, on the `_coast`, `_interior`, `_coast3` and
`_strait` subsets (gate names with that suffix); see meshom.quality.metrics.check_gates. `boundary_rule` is an
option, not a gate: which elements the `boundary_edges_max` gate applies to.
"""
import tomllib
from dataclasses import MISSING, dataclass, field, fields, replace
from pathlib import Path
from typing import ClassVar

DEFAULT_GATES: dict = {
    "quad_max_angle_hard_deg": 150.0,     # every quad
    "quad_max_angle_target_deg": 135.0,   # p99.9
    "tri_max_angle_hard_deg": 120.0,      # every triangle
    "tri_max_angle_target_deg": 90.0,     # p99, only when the mesh has no quads
    "tri_min_angle_hard_deg": 20.0,       # every triangle (non-degeneracy)
    "quad_sj_hard": 0.3,                  # every quad: min corner sin(angle)
    "elem_size_ratio_p99": 1.15,          # element mean edge across each edge (the node ratio is reported only)
    "elem_size_ratio_max": 2.0,           # no factor-2 jump
    "size_fidelity_p99": 0.15,            # |ln(s/h)| per edge; only when the summary has size_fidelity (needs h)
    "elem_neighbours_min": 2,
    "elems_per_node_max": 32,
    "tri_cluster_max": 2,                 # no edge-connected cluster of >= 3 triangles outside triangle zones
                                          # (Mesh.tri_zone); not applicable to meshes without quads
    "boundary_edges_max": 1,              # no element with >= 2 boundary edges, applied per boundary_rule
    "boundary_rule": "tri",               # option: "tri" = triangles only (FESOM2: a triangle
                                          # with 2 boundary edges has one neighbour), "all" = every element (the
                                          # strict form), "off" = no boundary_edges_max gate
    "pinch_nodes_max": 0,                 # manifold boundary: every boundary node has exactly 2 boundary edges
    "tri_share_max": 0.35,                # a large triangle zone is fine for now, better than bad quads;
                                          # the allowed level is this parameter (triangles / elements of a mixed mesh;
                                          # N/A for triangle-only meshes). m14_blocks_tri has 26-32 %.
    "euler_ok": True,                     # sum(4 - deg) + T = 4 chi of the domain (compete.irregular_counts)
}

RUNS_DIR = Path(".tmp/runs")


def tolerance_gates(gates: dict, tol: float | None, explicit: dict | None = None) -> dict:
    """gates with the size-fidelity gate set to the size tolerance the mesh was built for (decisions G38): a copy
    with size_fidelity_p99 = tol when tol is given and `explicit` (a config's own [gates] table) does not name that
    gate; else gates as they are. The element-ratio gates never change."""
    if tol is None or (explicit is not None and "size_fidelity_p99" in explicit):
        return gates
    return {**gates, "size_fidelity_p99": float(tol)}


def load_gates(path: Path | None = None) -> dict:
    """DEFAULT_GATES updated with the [gates] table of a TOML file (any file; the other tables are ignored)."""
    gates = dict(DEFAULT_GATES)
    if path is not None:
        with open(path, "rb") as f:
            gates.update(tomllib.load(f).get("gates", {}))
    return gates


class ConfigError(ValueError):
    pass


# ----------------------------------------------------------------------------------------------------- schema

@dataclass
class MeshCfg:
    method: str
    opts: dict = field(default_factory=dict)
    out: str | None = None
    scale: float = 1.0
    format: str = "fesom-c"                            # "fesom-c" (mixed, 4 columns) | "fesom2" (triangles only)
    #                                                    | "fesom3" (fesom_F: 4 columns + aux3d.out)
    fesom3_ini: str = ""                               # fesom_ini binary run on a fesom3 folder (1 partition); "" = off
    size_tolerance: float | None = None                # allowed |ln(s/h)|: smoothness first within it (G38); None = fit h
    optimize: bool | None = None                       # the optimisation stage (meshom.optimize, Z1); None = the
    #                                                    method's default (methods.own_optimize: on for m14)
    optimize_opts: dict = field(default_factory=dict)  # its options (meshom.optimize.DEFAULTS)


@dataclass
class CoastCfg:
    source: str = "none"
    min_area_km2: float = 0.0
    treatment: str = "cut"
    smoothing_m: float | None = None                  # loops: coast filter closing radius (m); None = h/2, 0 = off
    antarctica: str = "ice_front"                      # GSHHS L1 source: + L5 ice front | "grounding" (L6) | "none"
    keep_basins_km2: float | None = None               # cut and the loops filter: keep every basin of at least
                                                       # this area (None = FESOM2: the largest basin only)
    straits: str = "none"                              # built-in strait list (meshom.straits.BUILTIN) or "none"
    strait_h_min_m: float = 0.0                        # every strait's h_m is raised to at least this
    auto_straits: bool = False                         # size zones h = W/n on the medial axis' passages (S22, S24)
    strait_n_min: int = 5                              # cells across an automatic strait (S22 / S23)
    strait_profile: str = "resolve"                    # every listed strait: "resolve" (size zone) | "widen" (C45)
    strait_widen_cells: int = 2                        # widen: cells across at the background h (C45)
    passage_min_cells: int = 2                         # every through-passage >= this many cells across (C51); 0 = off


@dataclass
class StraitCfg:
    name: str
    path: list = field(default_factory=list)          # [[lon, lat], ...] centreline; empty = the built-in entry's
    width_m: float | None = None                       # channel kept open (corridor = max(width, cells h 1.15))
    h_m: float | None = None                           # target size along the path
    cells: int | None = None                           # cells across wanted (default 2)
    depth_m: float | None = None                       # depth floor on the path's elements
    profile: str | None = None                         # "resolve" | "widen": wins over regions and the default (C46)
    enabled: bool = True


@dataclass
class StraitRegionCfg:
    profile: str                                       # "resolve" | "widen" | "off" for the straits inside (C46)
    name: str = ""
    box: list | None = None                            # [west, east, south, north] degrees (west > east: across 180)
    polygon: list = field(default_factory=list)        # [[lon, lat], ...] (great-circle edges) when no box


@dataclass
class InsertCfg:
    """An [[inserts]] entry: a curvilinear insert fitted into a finished run (meshom.insert, B158-B169, B180-B189):
    a strait, a coastal strip or a current corridor."""
    name: str = ""
    strait: str = ""                                   # a strait of the run's [[straits]] or the built-in list
    path: list = field(default_factory=list)          # [[lon, lat], ...] centreline (basin A -> basin B) instead
    mouth_a: list = field(default_factory=list)       # join hint on side A: [[lon, lat], ...] from bank to bank
    mouth_b: list = field(default_factory=list)       # join hint on side B
    cells: int | None = None                           # cells across the neck (default 10, B159)
    h_fine_m: float | None = None                      # or the fine size at the neck: cells = round(W / h_fine_m)
    reach_km: float | None = None                      # growth limit per side (default 12 host cells, B159)
    force: bool = False                                # build with the least-violating joins when no join passes
    stages: str = "d,b,e,a"                            # meshfix stages on the join ring ("" = no cleaning, B162)
    ring_rows: int = 2                                 # block rows at each join that the cleaning may change
    bank_sigma_h: float = 1.5                          # bank smoothing in host coast cells (B161)
    bank_cap_h: float = 0.2                            # the smoothed bank stays within this x the run's h of the
                                                       # host coast (B161; the coast rule is 0.25)
    collar_rings: int | None = None                    # the collar cleaned round the joins, in host element rings
                                                       # (None: the default on meshom's and triangle hosts, 0 on a
                                                       # structured host such as ORCA: its nodes stay; B192, B193)
    clean_h: str = "insert"                            # the cleaning's h: "insert" (run h + insert layer) | "mesh"
    h_min_m: float | None = None                       # size floor: never finer (default: the host mesh's smallest
                                                       # cell, B165)
    coast: str = "real"                                # banks on the "real" coast (the run's [coast] source) or the
                                                       # "host" mesh's coast (step 1's capped staircase, B164)
    coast_source: str = ""                             # another land shapefile for the real banks ("" = the run's)
    bank_sigma_cells: float = 1.5                      # real-bank smoothing, in local insert cells (B164)
    bays: bool | str = "auto"                          # bays of the real coast as blocks of their own: "auto" (a
                                                       # mouth of 3 cells, no narrower inside, B190) | true (every
                                                       # bay, B166) | false (none: all bridged)
    kind: str = "strait"                               # "strait" | "strip" (a coastal strip: path along the coast)
                                                       # | "corridor" (open water: path = the axis) (B181, B182;
                                                       # both experimental) | "depth" (a depth region: the host nodes
                                                       # take the GEBCO rule, the mesh unchanged; B223) | "group"
                                                       # (channels, patches and islands in one envelope, B240;
                                                       # experimental)
    width_km: float | None = None                      # strip: coast to the offshore join; corridor, depth: full width
    band_km: float | None = None                       # strip / corridor: width of the fine band (default: what the
                                                       # room left after the growth to the host size, B181)
    ljoins: bool = True                                # L-shaped joins round a cape when no straight join passes
                                                       # (B183)
    rows: str = "harmonic"                             # a strip's rows: "harmonic" (from a harmonic map of the
                                                       # band, B195-B198) | "offset" (offset curves, B181)
    coast_rounds: int = 3                              # harmonic strip: rounds of extra coast smoothing round the
                                                       # bad cells next to the coast (B198)
    coast_shift_cells: float = 1.0                     # ... which moves the bank at most this many fine cells from
                                                       # where the base smoothing (bank_sigma_cells) put it
    cells_a: int | None = None                         # strait: the join of side A / B has exactly this many
    cells_b: int | None = None                         # edges, the counts run monotonically from the neck to it
                                                       # (None: any count; B200)
    growth_a: float | None = None                      # strait: the size slope along the block on side A / B
    growth_b: float | None = None                      # (dh/ds; None: 0.12); the room and funnel rules scale with it
                                                       # (B201)
    bank_land_cells: float = 0.0                       # strait: the real banks may move into land by this many local
                                                       # insert cells (smoother, squared at the joins; B202)
    fill: list = field(default_factory=list)          # strait: fill lines [[[lon, lat], ...], ...]: each replaces the
                                                       # real coast between its ends (snapped onto it) where it lies
                                                       # seaward; the water cut off is meshed as a strip of its own
                                                       # hanging off that bank (B204)
    fill_rows: int | None = None                       # rows of that strip (None: its depth / the local cell, >= 1)
    corner_steps: bool = False                         # strait: a join from a hint cuts every convex step of its
                                                       # host staircase by a corner triangle, not only its bends (B206)
    add_columns: bool = False                          # strait: the size target is lowered where a block cell is
                                                       # above 1.1 x the host cell (more columns and rows); with
                                                       # cells_a / cells_b the counts may rise above the neck's and the
                                                       # join's and come back to the join's count before the join (B207)
    corner_split: bool = False                         # strait: each corner triangle of a join is split at the
                                                       # midpoint of its diagonal, a node of the block's first row:
                                                       # two triangles of ~0.8 x the host cell, no long diagonal (B208)
    join_margin: float = 1.0                           # strait: the block's size target at the joins is capped at this
                                                       # times the host cell size (B209)
    fix_rings: int = 0                                 # after the splice: meshfix stage a on the insert's cells and
                                                       # this many element rings round them, the host's coast and
                                                       # every other node held (0 = off); moved host nodes keep
                                                       # their host depth in the model files (B210)
    depth_at: str = ""                                 # the insert's own nodes: "" (the run's [bathymetry] depth_at)
                                                       # | "node" | "cells_deepest" (B211)
    depth_smooth: int | None = None                    # ... None: the run's smooth_passes; a number: that many passes
                                                       # over the insert's own nodes only (B211)
    strait_floor: bool = True                          # ... the run's strait depth floors also on them (B211)
    sill_level: str = ""                               # "" | "shallower": the insert's sill gets the shallower of the
                                                       # two levels round its depth in the model's levels (B212) |
                                                       # "grid": the grid's saddle between the joins held at the
                                                       # deepest level not below its depth by a dam of nodes (B215)
                                                       # | "nearest": the same dam at the level nearest its depth
                                                       # (the level GEBCO's saddle falls on, B249)
    depth_channel: str = ""                            # "" | "cells" | "near": the grid's deepest channel between
                                                       # the joins (box means at the fine size) raises the insert's
                                                       # nodes of the cells it crosses / within one spacing of it to
                                                       # its box-mean depth there (B213)
    depth_quantile: float = 0.0                        # the insert's nodes take this quantile of the grid's ocean
                                                       # cells in their node box instead of the mean (0: off, B214)
    depth_area: bool = False                           # the insert's nodes scaled so that each section across
                                                       # the strait has the grid's wet area in FESOM3's element
                                                       # bottoms (after the other own rules; the dam's cap kept,
                                                       # B250; the deepest connection within a level of the grid
                                                       # saddle's, B251; kind "depth": the region's nodes, the
                                                       # path's sections)
    polygon: list = field(default_factory=list)       # kind "depth": the region [[lon, lat], ...] (great-circle
                                                       # edges) instead of width_km round the path (B223)
    envelope: list = field(default_factory=list)      # group: [[lon, lat], ...] polygon: the host cells whose
                                                       # centroid lies inside are replaced as a whole (B240)
    channels: list = field(default_factory=list)      # group: [{name, a, b, cells}, ...]: a channel block between the
                                                       # cross lines a and b ([[lon, lat], [lon, lat]], bank to bank),
                                                       # `cells` across from a to b (B242)
    gap_km: float | None = None                        # group: an island closer than this to other land joins it
                                                       # (default 5 x h_min_m, B241)
    island_km2: float = 10.0                           # group: an island smaller than this is dropped (water, B241)
    growth: float | None = None                        # group: the size slope of the patches and blocks (0.12, B243)
    gap_cells: int = 3                                 # group: the least fill cells across any passage (B243)
    split_deg: float | None = None                     # group: after fix_rings, each patch quad (an own cell, not a
                                                       # block's) whose largest corner exceeds this (deg) becomes
                                                       # two triangles along the diagonal from that corner (B243)
    bank_smooth_cells: float = 0.0                     # group: each block's banks smoothed by a Gaussian of this many
                                                       # of its cells, ends held: capes cut, pockets filled (B245)
    routes: list = field(default_factory=list)        # group: [{name, a = [lon, lat], b = [lon, lat], via =
                                                       # [[lon, lat], ...], corridor_km}, ...]: the straits through
                                                       # the group, a and b in the basins at their ends; with
                                                       # corridor_km the line stays within corridor_km / 2 of the
                                                       # polyline a, via..., b (a strait beside a deeper one); the
                                                       # channel raise, the dam and the section areas act along each
                                                       # route's line across blocks and patches (B246)
    blend_cells: float | None = None                   # group: the real coast blends onto a join's end node over this
                                                       # many join cells (None: 2, as a strait's banks, B164; B247)


@dataclass
class BlockCfg:
    name: str = ""
    corners: list = field(default_factory=list)       # 4 x [lon, lat], counter-clockwise (the block on the left)
    rows: int | None = None                            # cells along side 1-2 (and 3-4); None = from h
    cols: int | None = None                            # cells along side 2-3 (and 4-1); None = from h
    h_m: float | None = None                           # cell size inside; None = the run's h
    sides: list | None = None                          # 4 of "coast" | "cut" | "lat" | "offset"; None = the rule
    snap_m: float | None = None                        # corners within it move onto the coast; None = 1 h
    filter_m: float | None = None                      # Gaussian filter of the coast sides; None = 2 h


@dataclass
class ConstantLayer:
    kind: ClassVar[str] = "constant"
    h_m: float
    name: str = ""
    enabled: bool = True


@dataclass
class LatitudeLayer:
    kind: ClassVar[str] = "latitude"
    h_m: float
    h_min_m: float
    name: str = ""
    enabled: bool = True


@dataclass
class CoastDistanceLayer:
    kind: ClassVar[str] = "coast_distance"
    h_m: float
    grad: float
    plateau_m: float = 0.0
    coasts: str = "all"                                # "all" | "continents" | "islands" (S33)
    island_max_km2: float = 100_000.0                  # a land polygon smaller than this is an island (S33)
    name: str = ""
    enabled: bool = True


@dataclass
class ZoneLayer:
    kind: ClassVar[str] = "zone"
    transition_m: float
    h_m: float | None = None                           # min / set: h inside; smooth: not used
    polygon: list = field(default_factory=list)       # [[lon, lat], ...] degrees, great-circle edges
    h_out_m: float | None = None                       # min only: h at transition_m and beyond (default: the base h)
    mode: str = "min"                                  # "min" | "set" | "smooth" (S34) | "blend" (S37)
    smooth_m: float | None = None                      # smooth: radius of the ln h average (m)
    outer: list = field(default_factory=list)         # blend: the running h outside this polygon ([[lon, lat], ...])
    h_factor: float | None = None                      # blend (B110): h_factor x the running h inside, in place of h_m
    name: str = ""
    enabled: bool = True


@dataclass
class OrcaLayer:
    kind: ClassVar[str] = "orca"
    orca: str                                          # a preset of meshom.orca.presets
    h_min_m: float
    name: str = ""
    enabled: bool = True


LAYERS = {c.kind: c for c in (ConstantLayer, LatitudeLayer, CoastDistanceLayer, ZoneLayer, OrcaLayer)}


@dataclass
class SizeCfg:
    layers: list = field(default_factory=list)
    grad_max: float = 0.1
    h_min_m: float | None = None
    h_max_m: float | None = None
    preview_deg: float = 0.5
    limiter_max_nodes: int = 12_000_000


@dataclass
class DepthRegionCfg:
    name: str = ""
    box: list | None = None                            # [lon_min, lat_min, lon_max, lat_max] degrees (lon_min >
    #                                                    lon_max: across 180)
    polygon: list = field(default_factory=list)        # [[lon, lat], ...] (great-circle edges) when no box
    min_depth_m: float | None = None                   # nodes inside at least this deep
    max_depth_m: float | None = None                   # nodes inside at most this deep
    depth_m: float | None = None                       # nodes inside set to this (before min / max)


@dataclass
class BathymetryCfg:
    source: str = "none"                               # netCDF grid of elevation (positive up), or "none"
    variable: str = "elevation"                        # GEBCO's name; ocean = value < 0
    min_depth_m: float = 10.0                          # wet nodes shallower than this get it (fesom2_mesh_gen: 10 m)
    depth_at: str = "node"                             # "node" (box mean per node) | "cells_deepest" (B103)
    levels: list | str = "core2"                       # FESOM2 level surfaces (m, positive down): a list, a named set
                                                       # (meshom.bathymetry.LEVEL_SETS) or a file (nl, then nl values)
    min_layers: int = 4                                # fesom2 / fesom3: depth >= zbar[min_layers] (CORE2: 30 m), the
                                                       # depth FESOM raises shallower nodes to
    smooth_passes: int = 0                             # volume-conserving diffusion passes over the whole mesh (0 = off)
    max_slope: float = 0.0                             # slope limiter |d_i - d_j| / L_ij <= max_slope; 0 = off
    carve_taper: float = 1.0                           # strait depth floors: cosine taper over this x the strait h_m
    regions: list = field(default_factory=list)        # [[bathymetry.regions]] (DepthRegionCfg): depth rules by area,
    #                                                    after every other depth rule, in file order (O35, O36)


@dataclass
class RunCfg:
    timeout: float = 1800.0
    smooth: bool = False
    figures: bool = False


@dataclass
class Config:
    mesh: MeshCfg
    coast: CoastCfg = field(default_factory=CoastCfg)
    size: SizeCfg = field(default_factory=SizeCfg)
    blocks: list = field(default_factory=list)          # [[blocks]] entries (BlockCfg)
    gates: dict = field(default_factory=dict)           # overrides of DEFAULT_GATES
    run: RunCfg = field(default_factory=RunCfg)
    bathymetry: BathymetryCfg = field(default_factory=BathymetryCfg)
    straits: list = field(default_factory=list)         # [[straits]] entries (StraitCfg)
    strait_regions: list = field(default_factory=list)  # [[strait_regions]] entries (StraitRegionCfg), first match
    inserts: list = field(default_factory=list)         # [[inserts]] entries (InsertCfg): meshom.insert, not the run
    name: str = "config"                                # the config file's stem (not a TOML key)
    size_from: str = field(default="", compare=False)   # the [size] from file, resolved ("" = none; not a TOML key)

    def gates_all(self) -> dict:
        """DEFAULT_GATES, the [gates] table on top, and the size gate at [mesh] size_tolerance (tolerance_gates)."""
        return tolerance_gates({**DEFAULT_GATES, **self.gates}, self.size_tolerance(), self.gates)

    def size_tolerance(self) -> float | None:
        """The run's size tolerance (methods.effective_tolerance): [mesh] size_tolerance, else an
        opts["size_tolerance"] given to the method directly, else the method's default (0.20 for the m14 methods);
        None = off (0 given)."""
        from meshom.methods import effective_tolerance

        o = dict(self.mesh.opts)
        if self.mesh.size_tolerance is not None:
            o["size_tolerance"] = self.mesh.size_tolerance
        return effective_tolerance(self.mesh.method, o)

    def layers(self) -> list:
        """The enabled [[size.layers]] (a layer with enabled = false is kept in the file and skipped)."""
        return [ly for ly in self.size.layers if ly.enabled]

    def base_h(self) -> float | None:
        """The base h: the largest constant layer, else [size] h_max_m, else the largest h_m (or h_out_m) of the
        enabled layers (so a config made of zones alone, as the UI's h builder allows, still has a base)."""
        hs = [ly.h_m for ly in self.layers() if isinstance(ly, ConstantLayer)]
        if hs:
            return max(hs)
        if self.size.h_max_m is not None:
            return self.size.h_max_m
        hs = [v for ly in self.layers() if (v := getattr(ly, "h_out_m", None) or getattr(ly, "h_m", None)) is not None]
        return max(hs) if hs else None

    def out_dir(self) -> Path:
        return Path(self.mesh.out) if self.mesh.out else RUNS_DIR / self.name


# ------------------------------------------------------------------------------------------------ load, dump

def _value(v, tp, path: str):
    if tp in (float, float | None) and isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if tp in (int, int | None) and isinstance(v, float) and v.is_integer():
        return int(v)
    if path.endswith((".polygon", ".path", ".corners", ".outer", ".mouth_a", ".mouth_b")):
        return [[float(a) for a in p] for p in v]
    if path.endswith(".box"):
        return [float(a) for a in v]
    if path.endswith(".levels") and isinstance(v, list):
        return [float(a) for a in v]
    return v


def _build(cls, d: dict, path: str, skip: tuple = ()):
    known = {f.name for f in fields(cls)}
    for k in d:
        if k not in known and k not in skip:
            raise ConfigError(f"{path}.{k}: unknown key (known: {', '.join(sorted(known))})")
    kw = {}
    for f in fields(cls):
        if f.name in d:
            kw[f.name] = _value(d[f.name], f.type, f"{path}.{f.name}")
        elif f.default is MISSING and f.default_factory is MISSING:
            raise ConfigError(f"{path}.{f.name}: missing")
    return cls(**kw)


def _layer(d: dict, path: str):
    if "kind" not in d:
        raise ConfigError(f"{path}.kind: missing (one of {', '.join(LAYERS)})")
    if d["kind"] not in LAYERS:
        raise ConfigError(f"{path}.kind: unknown layer kind {d['kind']!r} (one of {', '.join(LAYERS)})")
    return _build(LAYERS[d["kind"]], d, path, skip=("kind",))


def _size_tables(size: dict, base_dir: Path | None, where: str = "size") -> tuple[dict, list, str]:
    """([size] keys without layers and from, the layers, the resolved from file or ""): with `from`, that file's
    [size] keys and layers first (its own `from` too), this table's keys over them, its layers after them (S35)."""
    size = dict(size)
    src = size.pop("from", None)
    own = [_layer(ly, f"{where}.layers[{i}]") for i, ly in enumerate(size.pop("layers", []))]
    if src is None:
        return size, own, ""
    f = Path(src).expanduser()
    f = (f if f.is_absolute() else Path(base_dir or ".") / f).resolve()
    with open(f, "rb") as fh:
        d = tomllib.load(fh)
    _size_only(d, f.name)
    keys, layers, _ = _size_tables(d.get("size", {}), f.parent, f"{f.name}: size")
    return {**keys, **size}, layers + own, str(f)


def _size_only(d: dict, where: str) -> None:
    for k in d:
        if k != "size":
            raise ConfigError(f"{where}: {k}: a size variant holds only [size] and [[size.layers]]")


def _bathymetry(b: dict) -> BathymetryCfg:
    b = dict(b)
    regions = [_build(DepthRegionCfg, r, f"bathymetry.regions[{i}]") for i, r in enumerate(b.pop("regions", []))]
    return replace(_build(BathymetryCfg, b, "bathymetry"), regions=regions)


def from_dict(d: dict, name: str = "config", base_dir: Path | None = None) -> Config:
    """The Config of the TOML tables d; [size] from is resolved against base_dir (default: the working directory)."""
    tables = {"mesh", "coast", "size", "blocks", "gates", "run", "bathymetry", "straits", "strait_regions",
              "inserts"}
    for k in d:
        if k not in tables:
            raise ConfigError(f"{k}: unknown key (known: {', '.join(sorted(tables))})")
    if "mesh" not in d:
        raise ConfigError("mesh: missing")
    size, layers, size_from = _size_tables(d.get("size", {}), base_dir)
    blocks = d.get("blocks", [])
    if isinstance(blocks, dict):                    # the old reserved [blocks] table: empty, or not a block list
        if blocks:
            raise ConfigError("blocks: [[blocks]] (an array of tables) expected, got a [blocks] table")
        blocks = []
    gates = d.get("gates", {})
    for k in gates:
        if k not in DEFAULT_GATES:
            raise ConfigError(f"gates.{k}: unknown gate (known: {', '.join(DEFAULT_GATES)})")
    return Config(mesh=_build(MeshCfg, d["mesh"], "mesh"), coast=_build(CoastCfg, d.get("coast", {}), "coast"),
                  size=replace(_build(SizeCfg, size, "size"), layers=layers),
                  blocks=[_build(BlockCfg, b, f"blocks[{i}]") for i, b in enumerate(blocks)],
                  gates=dict(gates), run=_build(RunCfg, d.get("run", {}), "run"),
                  bathymetry=_bathymetry(d.get("bathymetry", {})),
                  straits=[_build(StraitCfg, st, f"straits[{i}]") for i, st in enumerate(d.get("straits", []))],
                  strait_regions=[_build(StraitRegionCfg, r, f"strait_regions[{i}]")
                                  for i, r in enumerate(d.get("strait_regions", []))],
                  inserts=[_build(InsertCfg, x, f"inserts[{i}]") for i, x in enumerate(d.get("inserts", []))],
                  name=name, size_from=size_from)


def loads(text: str, name: str = "config", base_dir: Path | None = None) -> Config:
    return from_dict(tomllib.loads(text), name, base_dir)


def load(path: Path) -> Config:
    """The config file at path ([size] from relative to its folder)."""
    path = Path(path)
    with open(path, "rb") as f:
        return from_dict(tomllib.load(f), path.stem, path.parent)


def loads_size(text: str, base_dir: Path | None = None) -> SizeCfg:
    """An h variant: a TOML text of [size] and [[size.layers]] only (its own [size] from resolved against
    base_dir) as a SizeCfg; a config refers to one with [size] from = "<file>.toml" (S35)."""
    d = tomllib.loads(text)
    _size_only(d, "size variant")
    keys, layers, _ = _size_tables(d.get("size", {}), base_dir)
    return replace(_build(SizeCfg, keys, "size"), layers=layers)


def dumps_size(size: SizeCfg) -> str:
    """The h variant file of a SizeCfg ([size] and [[size.layers]], as dumps writes them); loads_size reads it back
    equal."""
    import tomli_w

    return tomli_w.dumps({"size": _size_dict(size)})


def _plain(obj) -> dict:
    out = {}
    for f in fields(obj):
        v = getattr(obj, f.name)
        if v is None or (f.name == "name" and v == "") or (f.name == "enabled" and v is True):
            continue
        out[f.name] = v
    return out


def _size_dict(size: SizeCfg) -> dict:
    d = _plain(size)
    d["layers"] = [{"kind": ly.kind, **_plain(ly)} for ly in size.layers]
    return d


def to_dict(cfg: Config) -> dict:
    """The TOML tables of cfg (None, empty names and enabled = true omitted; empty [[blocks]] / [gates] omitted)."""
    d = {"mesh": _plain(cfg.mesh), "coast": _plain(cfg.coast), "size": _size_dict(cfg.size), "run": _plain(cfg.run),
         "bathymetry": _plain(cfg.bathymetry)}
    d["bathymetry"].pop("regions")
    if cfg.bathymetry.regions:
        d["bathymetry"]["regions"] = [{k: v for k, v in _plain(r).items() if v != []} for r in cfg.bathymetry.regions]
    if cfg.blocks:
        d["blocks"] = [_plain(b) for b in cfg.blocks]
    if cfg.gates:
        d["gates"] = cfg.gates
    if cfg.straits:
        d["straits"] = [_plain(st) for st in cfg.straits]
    if cfg.strait_regions:
        d["strait_regions"] = [{k: v for k, v in _plain(r).items() if v != []} for r in cfg.strait_regions]
    if cfg.inserts:
        d["inserts"] = [{k: v for k, v in _plain(x).items() if v != [] and v != ""} for x in cfg.inserts]
    return d


def dumps(cfg: Config) -> str:
    import tomli_w

    return tomli_w.dumps(to_dict(cfg))


def save(cfg: Config, path: Path) -> None:
    Path(path).write_text(dumps(cfg))
