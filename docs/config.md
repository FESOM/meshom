# meshom config

One TOML file describes a mesh: the coast, the size function `h`, the fill method and the run settings. A batch run
reads it (`meshom run CONFIG`), and the web UI (`meshom ui`, `ui/web/README.md`) reads and writes the same file
through the same code: the schema lives in `meshom/config.py` (`load`, `loads`, `dumps`, `save`; typed dataclasses),
and the UI has no schema of its own (`ui/common/scenario.py` converts the config to and from the page's JSON only).

- Units: metres for lengths and sizes, km² for areas, seconds for times, degrees for longitude and latitude.
- `h` is the target edge length.
- A key that is not in the schema is an error that names its path, e.g.
  `size.layers[2].transiton_m: unknown key (known: h_m, h_out_m, name, polygon, transition_m)`.
  Values are not checked further: meshom is a tool for professionals.
- `meshom.config.dumps(load(path))` loads back to an equal config.
- Relative paths (`coast.source`, `mesh.out`) are relative to the working directory of the run (a `coast.source` not
  found there, in a git worktree: relative to the main checkout's root, `[coast]` below).
  `config.resolved.toml` in the run directory has every default filled in and both paths absolute.

## Example

```toml
[mesh]
method = "m02_linf_frontal"        # a name from meshom.methods.METHODS (`meshom compete --methods` lists them)
out = ".tmp/runs/global_400km"     # run directory (default .tmp/runs/<config file stem>)
scale = 1.0                        # the final h is multiplied by this (quick runs: 2, 4)
format = "fesom-c"                 # output files: "fesom-c" (mixed, default), "fesom2" (triangle meshes), "fesom3"
fesom3_ini = ""                    # fesom3 only: FESOM3's fesom_ini binary, run with 1 partition; "" = off
# size_tolerance = 0.2             # allowed |ln(s/h)| per edge: smoothness first within it (default: 0.20 for the
                                   # m14 methods, fit h for the others; 0 = off: fit h)

[mesh.opts]                        # given to the method as is (its own options; not checked by the schema)

[coast]
source = "../coastlines/GSHHS_shp/i/GSHHS_i_L1.shp"   # land polygons (GSHHG 2.3.7); "none" = the closed sphere
antarctica = "ice_front"           # GSHHS L1 has no Antarctica: add the L5 ice front next to it (or "grounding", "none")
min_area_km2 = 5000.0              # smaller polygons (islands) are dropped
treatment = "cut"                  # "cut" (default) or "loops", see below
# keep_basins_km2 = 100000.0       # special cases only: keep, besides the largest, every water basin of at least this
                                   # area. Default (unset): the largest basin only (by area), the water connected to
                                   # the world ocean, which is the rule
                                   # for global meshes (decisions C48: no lakes, no cut-off basins). At 150 km without a
                                   # strait list Gibraltar does not connect, so the Mediterranean is dropped: open it
                                   # with `straits`, not with this key.
straits = "none"                   # straits kept open: "none" (default) or a built-in list, "fesom2_mesh_gen" (90 straits)
                                   # Gibraltar is always opened with a real coast (C49), even with "none"
strait_h_min_m = 0.0               # every resolved strait's h_m is raised to at least this (e.g. 10000 for a CORE2-like step)
strait_profile = "resolve"         # every listed strait: "resolve" (size zone h_m) or "widen" (CORE2: background h, >= 2 cells)
strait_widen_cells = 2             # widen: cells across at the background h
passage_min_cells = 2              # every through-passage of the final mesh >= this many cells across (cut: widened); 0 = off
auto_straits = false               # true: passages of the coast's medial axis get the size zone h = W/n (see "Straits")
strait_n_min = 5                   # cells across an automatic strait

[size]
grad_max = 0.1                     # gradient limiter: |grad h| <= grad_max; inf = no limiting
h_min_m = 10000.0                  # optional global clip of the min over the layers
h_max_m = 400000.0
preview_deg = 0.5                  # UI preview grid (lon-lat); not used by a batch run
limiter_max_nodes = 12000000       # cap of the limiter grid (memory), see "Gradient limiter"

[[size.layers]]
kind = "constant"
h_m = 400000.0

[[size.layers]]
kind = "latitude"
h_m = 400000.0                     # at the equator
h_min_m = 150000.0                 # h = max(h_m cos(lat), h_min_m)

[[size.layers]]
kind = "coast_distance"
name = "coast"
h_m = 150000.0                     # at the coast
plateau_m = 100000.0               # h stays h_m up to this distance from the coast
grad = 0.1                         # then grows with this slope: h = h_m + grad max(d - plateau_m, 0)

[[size.layers]]
kind = "zone"
name = "Bering Strait"
h_m = 50000.0                      # inside the polygon
transition_m = 1000000.0           # linear from h_m at the edge to h_out_m at this distance
h_out_m = 400000.0                 # optional; default: the base h (largest constant layer, else h_max_m, else the largest layer h_m)
polygon = [[172.0, 62.0], [-172.0, 60.0], [-160.0, 64.0], [-163.0, 69.0], [-175.0, 70.0], [176.0, 66.0]]

[[blocks]]                         # a hand-placed block (the m14 methods build it first; see "[[blocks]]")
name = "Atlantic"
corners = [[-42.0, -20.0], [15.0, -20.0], [-7.0, 40.0], [-76.0, 40.0]]   # [lon, lat], counter-clockwise
sides = ["lat", "cut", "lat", "cut"]   # coast / cut / lat / offset (default: coast along a coast, else cut)
rows = 64                          # cells along side 1-2 (default: from h)
cols = 70                          # cells along side 2-3 (default: from h)

[gates]                            # overrides of meshom.config.DEFAULT_GATES
size_fidelity_p99 = 0.15

[run]
timeout = 1800.0                   # the method's child process is killed after this many seconds
smooth = false                     # also run the common smoothing (meshom.smooth, boundary nodes fixed)
figures = false                    # render the standard picture set (meshom.figures.render_run)

[bathymetry]
source = "none"                    # elevation grid (netCDF, positive up), e.g. GEBCO_2024.nc; "none" = no depth stage
variable = "elevation"
min_depth_m = 10.0                 # wet nodes shallower than this get it
depth_at = "node"
levels = "core2"                   # FESOM2 level surfaces: "core2" (48 levels), a list of metres, or a file
min_layers = 4                     # fesom2 / fesom3: depth >= zbar[min_layers] (30 m with core2)
smooth_passes = 0                  # volume-conserving depth smoothing passes over the whole mesh (0 = off)
max_slope = 0.0                    # slope limiter |dd| / L <= max_slope (0 = off)
carve_taper = 1.0                  # strait depth floors: cosine taper over carve_taper x the strait h_m

[[bathymetry.regions]]             # a depth rule by area, after every other depth rule (file order; "Depth regions")
name = "Gibraltar"
box = [-6.2, 35.6, -5.0, 36.3]     # [lon_min, lat_min, lon_max, lat_max]; or polygon = [[lon, lat], ...]
min_depth_m = 50.0                 # and / or max_depth_m, depth_m (set)

[[straits]]                        # a channel kept open (see "Straits"); one table per strait, added to [coast] straits
name = "Bosphorus"                 # an entry named as one of the built-in list overrides the fields it gives
path = [[29.147, 41.247], [29.065, 41.146], [28.99, 40.979]]   # centreline [lon, lat], from one basin into the other
width_m = 2050.0                   # channel width kept open; the corridor is max(width_m, cells h_m 1.15 + h_m / sqrt 3)
h_m = 10000.0                      # target size along the path
cells = 2                          # cells across wanted (default 2)
depth_m = 35.0                     # depth floor near the path (carving, optional; see [bathymetry] carve_taper)
profile = "resolve"                # optional: "resolve" | "widen", wins over [[strait_regions]] and [coast] strait_profile

[[strait_regions]]                 # optional: which straits are opened and how (see "Strait profiles"); first match wins
name = "Baltic"
profile = "off"                    # "resolve" | "widen" | "off" (not opened) for the straits whose middle lies inside
box = [9.0, 31.0, 53.0, 66.0]      # [west, east, south, north] degrees (west > east: across 180), or polygon = [[lon, lat], ...]
```

## Tables

### `[mesh]`

| key | type | default | meaning |
|---|---|---|---|
| `method` | string | required | fill method, a key of `meshom.methods.METHODS` |
| `opts` | table | `{}` | passed to the method as `opts` |
| `out` | path | `.tmp/runs/<config stem>` | run directory; `meshom run --out DIR` overrides it |
| `scale` | float | 1 | the final h (after the limiter) is multiplied by it, as `meshom compete --scale`: the node count changes by 1/scale², ρ_c does not, and \|∇h\| is multiplied by it too |
| `format` | `"fesom-c"`, `"fesom2"` or `"fesom3"` | `"fesom-c"` | the files of `fesom/`: `fesom-c` = FESOM-C mixed format (`write_fesomc`: 4 columns, `depth.out`, placeholder `aux3d.out`); `fesom2` = FESOM2 (`write_fesom2`: `nod2d.out` with flag 1 on boundary nodes, `elem2d.out` with 3 columns, `aux3d.out` with the `[bathymetry] levels` and the node depths written negative down, as CORE2's file) for **triangle meshes only** (quads are not written; `fesom2_checks` in the `write` stage then shows unused nodes); `fesom3` = FESOM3 (`write_fesom3`): `nod2d.out` and `aux3d.out` as `fesom2` (flag 1 on boundary nodes; the `[bathymetry] levels` and node depths negative down), `elem2d.out` with 4 columns as `fesom-c` (a triangle `n1 n2 n3 n1`, clockwise), no `depth.out` (fesom_ini's level finder makes the node levels; files fesom_ini derives and the partition folders `dist_<N>`, left by an earlier run, are removed: run fesom_ini again for a multi-rank run); `fesom3_checks` in the `write` stage counts what fesom_ini or the model stops on (decisions O25) |
| `size_tolerance` | float or none | the method's: 0.20 for the m14 methods (`m14_blocks` and every variant, `m14_blocks_rep` included), none for the others (m15_tri too) | how far the cell size may deviate from h, as \|ln(s/h)\| per edge (0.2 = ±22 %; decisions G38, M38). None or 0: the methods fit s = h as before (`size_tolerance = 0` switches an m14 method's 0.20 off, and the size gate stays at G9's 0.15). A value t: the method's last stage (m14 and `_mix`: after the seam-band smoothing of the stitch, the whole fill moves with block and coast nodes fixed; `m14_blocks_rep`: after the repair, whose bad edge is then max(t, 0.15); m15_tri: instead of the size fit) is `meshom.size_fit.smooth_fit`, which makes the element size change between neighbours (the `elem_size_ratio` of the gates) as gentle as possible and the angles as regular as possible while every edge stays within \|ln(s/h)\| ≤ t (a soft wall from t − 0.01). Measured (docs/methods/m14_blocks.md "Smoothness within a size tolerance"): t = 0.20 takes the CORE2-like `m14_blocks_rep` from 31 to 39 of 39 gates (ratio p99 1.21 → 1.11). It reaches the method as `opts["size_tolerance"]` (`meshom compete --opts '{"size_tolerance": 0.2}'` does the same), and the `size_fidelity_p99` gate becomes t, unless `[gates]` sets that gate: a mesh is judged against the tolerance it was built for. The element-ratio gates do not change |
| `fesom3_ini` | path | `""` (off) | `fesom3` only: FESOM3's partitioner binary `fesom_ini` (e.g. `/work/ab0995/a270126/projects/mesh_generation/fesom_F_build/build_intel_dp/bin/fesom_ini`), run in the `write` stage on `fesom/` with 1 partition (namelist in `fesom_ini/`, log `fesom_ini/fesom_ini.out`): it writes the edge and node-level files a 1-rank run needs and stops at the partitioning; the stage records `exit_status`, the stop lines of the log and `ok` (`run_fesom_ini`); never part of a default run |
| `optimize` | bool or unset | unset = the method's default: on for the m14 methods (`m14_blocks` and every variant, `m14_blocks_rep` included), off for the others | the method-independent optimisation stage (`meshom.optimize`, decisions Z1–Z13); `true` / `false` (or 1 / 0) force it on / off. The m14 methods run it as their own last stage (method option `optimize`, `optimize_opts`: `raw.npz` is then already optimised, its record is `optimize.json` in the run directory and there is no `optimized` record; `meshom compete` and the seam lakes run the same stage); for the other methods, and for an m14 method under a land cut (`[coast] treatment = "cut"`, the method then runs without it), the run applies it after the method (and the smoothing) — never twice. What it does: edge flips by angle in triangle areas, triangle pairs merged into quads ≤ 135°, triangle clusters ≥ 3 broken (polygon re-splits, node collapse), cancelling valence pairs removed, then `size_fit.smooth_fit` at the run's size tolerance (else 0.15) on the nodes near the changes; boundary and coast nodes fixed, the Euler sum and the boundary kept; the run's stage writes `optimized.npz`, scored as the record `optimized` of `quality.json`, the mesh the later stages (bathymetry, write, straits) use. `meshom optimize DIR` does the same on a saved mesh |
| `optimize_opts` | table | `{}` | options of the stage, also handed to an m14 method's own stage (`meshom.optimize.DEFAULTS`: `steps`, `passes`, `quad_max`, `relax_rings`, `fit`, `fit_rings`, …) |

### `[coast]`

| key | type | default | meaning |
|---|---|---|---|
| `source` | path or `"none"` | `"none"` | polygon shapefile of land (GSHHS level 1, Natural Earth land); `"none"` meshes the closed sphere |
| `min_area_km2` | float | 0 | land polygons below this area are dropped (the shapefile's `area` field in km² when it has one, as GSHHS; else the ring's spherical area) |
| `treatment` | `"cut"` or `"loops"` | `"cut"` | how the coast enters the mesh, below |
| `smoothing_m` | float | none | `loops` only: the coast filter's closing radius r_c in metres, the same everywhere (the opening's is 2 r_c); none = r_c = h/2 and r_o = h from the local h (the default); 0 = no filter (the raw loops resampled at h, as before 2026-10-23); see "The coast filter" below |
| `keep_basins_km2` | float | none | special cases only (a regional mesh, a lake wanted on purpose): water basins of at least this area are kept besides the largest one: in the cut (a basin the cut separates) and in the `loops` coast filter (a basin the opening cuts off); none = the largest basin only (FESOM2), the rule for global meshes: only the water connected to the world ocean, no lakes, no cut-off basins (decisions C48) |
| `straits` | `"none"` or a built-in list name | `"none"` | straits kept open, see "Straits" below; `"fesom2_mesh_gen"` = meshom's list derived from fesom2_mesh_gen (`meshom/data/straits_fesom2_mesh_gen.json`, 90 straits) |
| `strait_h_min_m` | float | 0 | every resolved strait's `h_m` (built-in or `[[straits]]`) is raised to at least this |
| `strait_profile` | `"resolve"` or `"widen"` | `"resolve"` | the profile of every listed strait without its own `profile` when there are no `[[strait_regions]]`; see "Strait profiles" below (decisions C45) |
| `strait_widen_cells` | int | 2 | `widen`: the cells across a widened strait at the background h (a `[[straits]]` `cells` overrides it) |
| `passage_min_cells` | int | 2 | every through-passage of the water, listed or not, at least this many cells across on the final mesh; the cut widens a narrower one into the land (3 for meshes where straits matter; 0 or 1 = off); see "Narrow passages" below (decisions C51) |
| `auto_straits` | bool | false | the automatic strait zones (decisions S22, S24, S31): see "Automatic straits" below |
| `strait_n_min` | int | 5 | N_min, the cells across an automatic strait (S22 / S23) |
| `antarctica` | `"ice_front"`, `"grounding"` or `"none"` | `"ice_front"` | GSHHS L1 has no Antarctica (GSHHG ≥ 2.3 keeps it in L5 = ice front, L6 = grounding line): for a source `GSHHS_<res>_L1.shp` the polygons of `GSHHS_<res>_L5.shp` (`ice_front`, FESOM2's practice: no ice-shelf cavities) or `_L6.shp` (`grounding`) next to it are added; `none`, a source that is not GSHHS L1, or a missing L5 / L6 file add nothing (the run's `coast` stage record says which) |

The coast is read by `meshom.coast.read_coast`: every ring keeps the shapefile's orientation (outer rings clockwise,
so water is on the left, the domain convention); polygons cut at the antimeridian (GSHHS: Eurasia-Africa, Wrangel
Island, Fiji) are joined back into one loop by removing their edges along ±180°. Antarctica (GSHHS L5 / L6) comes as
two halves cut along 0° and ±180° and closed through the pole row (lat −90°): the pole vertices and the meridian
edges to them are dropped and the two coastal chains joined into one loop around the South Pole, so the land test and
the cut see the pole as land (ice front, `i` resolution: 13.9 M km², one loop of 5 478 vertices).
A relative `source` is taken relative to the working directory of the run; inside a git worktree
(`.worktrees/<name>`) a relative path that does not exist there is taken relative to the main checkout's root, so
`../coastlines/...` works from both (`meshom.run.source_path`).

- **`cut`** (default, the way fesom2_mesh_gen cuts its JIGSAW mesh): the method meshes the closed sphere with h; then
  every element whose centroid is on land (tested against the true coast polygons) is removed, and the element mask
  is cleaned with FESOM2's rules after D. Sein (AWI), fesom2_mesh_gen (spurs, pinched nodes, only the largest
  basin; `meshom.methods.m14.cut`). The coast is a staircase of whole elements. Works on any real coast.
- **`loops`**: the coast filtered to h (below), resampled at spacing h (`meshom.coast.Coast.loops_at`; islands with
  fewer than 3 steps dropped), is the domain boundary, and the method meshes the water only (the analytic `continent`
  case of `meshom compete` works this way, without the filter). Stage `coast_filter` in `stages.json` records the
  filter (points, loops before and after, the pieces the closing kept, the basins kept and dropped, times) and the
  loops at h (`n_loops_at`, `n_nodes_at`).

### The coast filter (`[coast] treatment = "loops"`)

`meshom.coast_filter.filter_coast` (decisions C16, C40, C41) smooths the coast to the local h
before the resampling, on the sphere (no charts: the antimeridian and the poles are ordinary points):

1. **opening** of the water at r_o = h: water that no disc of radius h inside the water covers becomes land, so bays,
   straits and gaps between islands narrower than 2 h close (unless listed, S24) and every passage left open is at
   least two cells wide; water corners are rounded. The strait corridors (`[coast] straits`, `[[straits]]`: within
   W/2 of the path) are water after it whatever the coast says, as in the cut. A passage whose h is refined to W/n
   (`auto_straits`) is wider than 2 h there and stays open by itself.
2. **closing** at r_c = h/2: land that no disc of radius h/2 inside the land covers becomes water, so capes, spits
   and islands narrower than h go; land corners are rounded. A piece of land whose removal would split a land mass
   (an isthmus: Panama or Kra at h = 100 km, or the plug the opening put into a short strait) or join two waters the
   opening left apart (a spit in front of a closed lagoon) is kept: the closing never opens a passage. A neck thinner
   than the point spacing (a 5 km neck at h = 100 km, spacing about h/8) is first bridged by points along it, so it
   counts too.
3. **basins**: water the opening cut off (the Mediterranean behind Gibraltar at h = 100 km without the strait list)
   becomes land unless it is the largest water body or at least `keep_basins_km2`.

Representation: signed distances on an adaptive icosahedral point set (spacing about h/8 within 1.9 h of the coast
and the corridor edges, coarse elsewhere) triangulated by its convex hull; the result is traced by marching
triangles. Measured (16 CPUs): GSHHS i L1 ≥ 3000 km² + ice front at h = 100 km: 0.63 M points, 17 s,
0.65 GB, 160 loops → 17 (14 basins dropped: the Mediterranean, Hudson Bay, the Black, Red and Baltic Seas, the Persian
Gulf, …); West Africa (a 25 km strip in 200 km): 0.28 M points, 8 s; the CORE2-like strait example in `loops` mode
(30 km coasts, 10 km straits): 6.2 M points, 209 s, 4.9 GB, all 89 straits connected. At a coarse h
every passage narrower than 2 h closes unless it is listed or h is refined there. `smoothing_m` sets r_c in metres
everywhere instead (r_o = 2 r_c); `smoothing_m = 0` turns the filter off.

### Straits (`[coast] straits`, `[[straits]]`)

`meshom.straits` keeps narrow channels open that the cut would close at the mesh's h (at 25–35 km Gibraltar and the
Bosphorus are land). The straits of a run are the built-in list named by `[coast]
straits` (none by default) merged by name with the `[[straits]]` tables: an entry with a built-in name overrides the
fields it gives (`enabled = false` drops that strait), other entries are added; then `h_m` is raised to `[coast]
strait_h_min_m`. Every strait acts in four places:

| where | what |
|---|---|
| size (`meshom.run.size_field`) | profile `resolve` only (a widened strait has none, see "Strait profiles"): a zone along the path: h_m within W/2 of the path, then linear in the distance to the base h with the slope `grad_max`; applied after `[size] h_min_m` / `h_max_m` (the global clip does not undo it) and before the limiter, whose grid spacing follows the smallest strait h_m too |
| cut (`meshom.run.cut_land`) | every element whose centroid is within W/2 of the path is water whatever the coast says, and the clean-up's spur and pinch rules do not remove it; one unprotected pass follows (FESOM2 rejects elements with one neighbour), which may trim the corridor's ragged edge |
| bathymetry | carving: the nodes within `h_m` of the path get at least `depth_m`, with a cosine taper over `[bathymetry] carve_taper` × `h_m` beyond |
| report | stage `straits` in `stages.json` and `"straits"` in the final `quality.json` record (`meshom.straits.strait_report`): per strait the cells across the narrowest of 40 sections of the inner 80 % of the path (wet width of the transect through the path point / mean edge of the elements on it, as fesom2_mesh_gen's strait check), the sections whose path point is dry, the water component of each path end (`ends_component`) and `connected` = both in the main water (component 0, the largest), the smallest element depth (mean of its 3 nodes) on the inner path, and `ok` = connected, no dry section, cells ≥ `cells`. With it `"water"` (stage `straits`, quality.json): the final mesh's water components (area order: elements, nodes, area km², box [west, east, south, north], the straits ending in each cut-off one) and `connectivity_ok` = one component and every strait connected; when false `meshom run` prints a `WATER NOT CONNECTED` line, the run goes on (decisions C44). Without straits the `write` stage gives the same record (C52, every written mesh: one water component) |

W, the corridor, is max(`width_m`, `cells` · h_m · 1.15 + h_m / √3): fesom2_mesh_gen's band of n = cells + 1 nodes at
size h with its 15 % margin, plus the width the centroid test loses (a third of a triangle row on each side). On a
mesh with quads (every method but the triangle-only ones, `meshom.straits.TRI_METHODS`) W is max(`width_m`,
(`cells` · 1.15 + √2) · e^t · h_m), t the run's size tolerance (0.20 for m14): edges up to e^t h_m and a quad's centroid
up to half its diagonal from its far corner, so `cells` cells stay across at any lattice angle (decisions C45).

| key | type | default | meaning |
|---|---|---|---|
| `name` | string | required | the strait's name (matches the built-in list) |
| `path` | `[[lon, lat], ...]` | the built-in entry's | the centreline, from basin to basin, great-circle segments |
| `width_m` | float | the built-in entry's, else 0 | channel width kept open (the corridor is at least this wide) |
| `h_m` | float | the built-in entry's | target size along the path (raised to `[coast] strait_h_min_m`) |
| `cells` | int | 2 | cells across wanted |
| `depth_m` | float | the built-in entry's, else none | depth floor near the path (carving, `[bathymetry] carve_taper`); 0 = none |
| `profile` | `"resolve"` or `"widen"` | none | the strait's own profile: wins over `[[strait_regions]]` and `[coast] strait_profile`, and opens it outside every region |
| `enabled` | bool | true | false drops the strait |

The built-in list `fesom2_mesh_gen` (`meshom/data/straits_fesom2_mesh_gen.json`): 90 strait centrelines after
D. Sein (AWI), fesom2_mesh_gen: its 88 curated straits (Rae Strait left out: closed there) plus the Great Belt and
Marmara_West (the 13–21 km wide channel of western Marmara from the Dardanelles' end at Gelibolu, a part of
fesom2_mesh_gen's automatic basin connection for Marmara: without it the cut closes that channel at strait h 20 km and
the Black Sea is cut off, decisions C27). Per strait: `width_m` = the narrowest channel fesom2_mesh_gen opens along it
(the real width, or its forced band where it widens the channel); `h_m` = the resolution fesom2_mesh_gen gives the
strait (0.35–30 km: a 2 M-node mesh); `depth_m` = its carve depth (Bosphorus 35 m, Dardanelles 55 m), else the sill
on its transects (the shallowest of the per-section deepest GEBCO points), none below 10 m.

### Strait profiles (`[coast] strait_profile`, `[[strait_regions]]`)

A mesh's purpose chooses its strait treatment: CORE2 has no small cells (a long time step) and
widens its straits, NG5 resolves the main straits at 5 km, an Arctic mesh resolves the Arctic and Canadian straits and
need not open the Baltic. Each listed strait gets one of two profiles (`meshom.straits`, decisions C45–C47):

| profile | size | corridor (forced water in the cut) |
|---|---|---|
| `resolve` (default) | the zone above: `h_m` (raised to `strait_h_min_m`) within W/2 of the path, graded to the base h | W = max(`width_m`, (`cells` · 1.15 + 1/√3) · `h_m`), `cells` default 2 |
| `widen` | no zone: the strait is meshed at the background h (the layers after the `[size]` clip, before the limiter); its `h_m` is that h at its largest over the inner 80 % of the path (sampled every 1 km), a given `h_m` is not used | the same formula with that h and `cells` = `strait_widen_cells` (2): at least that many cells of the background h across, the channel widened into the land on both sides of the path, as CORE2 opens Gibraltar ×8 and NG5 the Bosphorus ×16 |

Both profiles get the depth floor (within `h_m` of the path: for a widened strait the background h) and the report
(stage `straits`: each record has its `profile`; `cells_target` is the profile's `cells`).

Which straits are opened, and how: without `[[strait_regions]]` every strait takes `[coast] strait_profile`. With
regions, the first region (in file order) holding a strait's middle point (half its arc length) gives its profile,
`"off"` = not opened; a strait in no region is not opened either. Not opened means left as the coast and the cut
give it: closed where the coast closes it at h. A `[[straits]]` entry's own `profile` wins over both and opens the
strait wherever it lies (`enabled = false` still drops it). A region is a `box` [west, east, south, north] in degrees
(west > east crosses 180) or a `polygon` [[lon, lat], ...] with great-circle edges (inside an open hemisphere). Size
stage record: `strait_profiles` = the count per profile.

Examples: `docs/examples/fesom2_core2like_straits.toml` and `fesom2_core2like_straits45.toml` (resolve at 10 and
20 km), `fesom2_core2like_widen.toml` (widened; the Black Sea, Marmara and Azov left out as in CORE2: widened to 86 km the Bosphorus blows FESOM2 up, decisions C45), `fesom2_arctic_straits.toml` (regions: the Baltic off, north
of 63N resolved at 10 km plus Bering and Fram entries, the rest widened).

### Narrow passages (`[coast] passage_min_cells`)

A passage one cell wide is a dead end for the model: every node across it is on the coast, so nothing flows through
it. With `passage_min_cells = K` (default 2; 3 for meshes where straits matter; 0 = off) every
through-passage of the water, listed strait or not, gets at least K cells across at its narrowest section, the cell
size being the local background h (as `widen`). It is judged on the mesh as built, because the staircase of the land
cut narrows passages after any coast-based widening (`meshom.passages`, decisions C51):

- **measure** (topological, no metres): ring distance of every node from the coast (nodes sharing an element are one
  ring apart); quads split into two triangles along the diagonal with fewer interior corners (a quad touching both
  coasts is one cell); with r = K // 2 the core nodes are r rings or more from the coast, and two triangles sharing an
  edge are core-linked when the edge has one core end (K even) or two (K odd). Across a channel of k cells the core
  links run along it iff k ≥ K. **Pools** = components of the core links holding a core node (K even) or a triangle of
  three (K odd): water at least K cells across. Every other triangle is rim; a **neck** = rim triangles joined by
  their links, or one link between two pool triangles.
- **through-passage**: a neck reaching two pools, or one pool from two ends or more with the coast along it on two
  boundary loops or more (water around an island). Left alone: a neck with one end (a one-cell fjord, a staircase
  corner) and a **dead-end inlet**: a neck that is the only link to pools of less than 8 h² (`BASIN_MIN_H2`; h = the
  neck's mean edge; the side with the largest pool is never one).
- **fix, `cut`**: each passage becomes a `widen` corridor (path: from a pool triangle through the neck's rim triangles
  to the other pool, two 3-point smoothings, extended 1 h into both pools; h_m = the background h at its largest on
  the path; W = (K · 1.15 + 1/√3) h on triangles, (K · 1.15 + √2) e^t h with quads, as `widen`) and the uncut mesh is
  cut again with them; up to 3 rounds (`MAX_ROUNDS`), a passage found again inside an earlier corridor gets one cell
  more. **`loops`**: the run starts again with a `[[straits]]` entry `passage_<round>_<i>` (`widen`, `cells`) along
  each passage (the coast filter keeps corridors wet), at most 3 times; the entries are in `config.resolved.toml`.
- **records**: stage `post` → `passages` (rounds with found / dead ends / inlets, `n_widened`, `n_left`, `cut_km2` =
  real land made water, `rejoined_km2` = water the first clean-up had dropped, `lost_km2`, the land components
  touched and removed, every widened passage with place, kind, nearest listed strait, corridor); stage `passages` and
  quality.json `passages` on the final mesh (`n_narrow`, each one left: kind, place, cells across, nearest listed
  strait); `meshom run` prints a `NARROW PASSAGES` line when one is left.

Measured: the CORE2-like mesh A2 (m14 quads, cut, 90 widened
straits): 49 narrow passages (25 inlets left) → 0 in two rounds (51 corridors), 289 458 km² of land cut away (the
listed straits' corridors already hold 827 556 km²: +35 %; +0.09 % water), 24 islands removed (37 024 km², the
largest 7 890 km² Buton–Muna at h = 31 km), 144 439 → 144 691 nodes, 39/39 gates both; its triangle twin (m15_tri,
same h and straits): 29 → 0 in one round, 87 673 km² (+27 % on the straits' 324 194), 3 islands removed
(1 095 km²), 163 263 → 163 386 nodes, 29/29 both; the conforming global 100 km (loops): 0 at K = 2, 39/39; at K = 3
5 → 2 → 1 → 2 over its 3 rebuilds (each rebuild re-meshes the whole domain and new passages appear elsewhere; 39/39).

### Automatic straits (`[coast] auto_straits`)

With `auto_straits = true` the run finds the passages of the coast itself (`meshom.lfs.auto_straits`, no hand marking,
decision S22): the medial axis of the water (the Voronoi vertices of the coast sampled at h/4, pruned to half-widths
≥ ½ min(h, `strait_n_min` · `h_min_m`) and without the short side branches of coast bumps, S32), then every run of it
with W ≤ 20 h whose width changes by at most n · `grad_max` per unit length and whose two ends open into wider water
(a bay or a fjord ends in a dead end and is no passage), n = max(`strait_n_min`, round(median W/h)) cells across (S31).
A passage whose W/n stays ≥ `h_min_m` is open and gets the size zone h = W/n within W/2 of its centreline, linear with
the slope `grad_max` to the base h beyond, where that is finer than h (after the clip, before the limiter, as the listed
straits); a narrower one stays closed (S24): only a `[coast] straits` / `[[straits]]` entry opens it. The `size` stage
of `stages.json` lists every passage (`auto_straits`: centre and narrowest point, the centreline at ≤ 16 stations,
W min / median / max, n, open, refines). The pruning keeps half-widths ≥ ½ min(h, `strait_n_min` · `h_min_m`), so with
a coarse h and a large `h_min_m` (the CORE2-like example: h ≥ 25 km, `h_min_m` = 20 km) only passages wider than about
h are found and only those wider than `strait_n_min` · `h_min_m` (100 km) are open: Gibraltar (14 km) and the Bosphorus
(2 km) stay with the strait list. The m14 methods build their strait blocks from the passages of the final h
(`meshom/methods/m14/strait.py`): a block only where h ≈ W/n there (what this zone makes) and only when the method gets
the coast loops (treatment `loops`; with `cut` the method meshes the closed sphere and the runner cuts: there a strait
block is placed by hand, `[[blocks]]` with `coast` sides).

### `[size]` and `[[size.layers]]`

`h = scale × limit(clip(the layers in file order, h_min_m, h_max_m), grad_max)`.

| key | type | default | meaning |
|---|---|---|---|
| `from` | string | none | an **h variant** file (below): its `[size]` keys and layers come first |
| `layers` | array of tables | required | the criteria, applied in file order to a running h (below) |
| `grad_max` | float | 0.1 | gradient limit g: neighbouring cells differ by ≤ 1 + g; `inf` switches the limiter off |
| `h_min_m`, `h_max_m` | float | none | global clip |
| `preview_deg` | float | 0.5 | step of the UI's lon-lat preview grid |
| `limiter_max_nodes` | int | 12 000 000 | node cap of the limiter grid |

The **base h** is the largest `constant` layer, else `h_max_m`, else the largest `h_m` / `h_out_m` of the enabled layers. A `coast_distance` layer stops growing at the base
(beyond that it cannot win the min, and the KD-tree search stops there), and a zone grows to the base unless it has
its own `h_out_m`.

| kind | keys | h |
|---|---|---|
| `constant` | `h_m` | h_m |
| `latitude` | `h_m`, `h_min_m` | max(h_m cos(lat), h_min_m) |
| `coast_distance` | `h_m`, `grad`, `plateau_m` (0), `coasts` ("all"), `island_max_km2` (100 000) | h_m + grad · max(d − plateau_m, 0), at most the base; d = geodesic distance to the nearest coast point of the class `coasts` (the coast vertices, segments split to ≤ h_m/4, in a 3-D KD-tree built once) |
| `zone` | `h_m`, `transition_m`, `polygon`, `h_out_m` (base), `mode` ("min"), `smooth_m`, `outer` | mode `min`: h_m inside the polygon (great-circle edges; it must fit in a hemisphere), then linear in the geodesic distance to h_out_m at `transition_m`; `transition_m = 0` is a step that the limiter grades. Modes `set`, `smooth` and `blend`: below |
| `orca` | `orca` (a preset), `h_min_m` | the preset's cell size, max(√(cell area), h_min_m), from the 4 nearest T points (B101; it builds the grid: ~2-4 min for ORCA025) |

**Layer order and zone modes** (decisions S34). The layers are applied in file order to a running h; the first layer
starts it (a `set` or `smooth` zone as the first layer acts on the base h). Every kind but `zone` takes the minimum,
h = min(h, layer); so does a zone of `mode = "min"` (the default). Min commutes: a config of min layers alone gives
the same h in any order, bit for bit as before the modes. With a `set` or `smooth` zone in the config a min layer acts
only where it is below the base h: beyond its reach (a zone outside its transition, a coast layer far from the coast,
a `constant` at the base) it leaves the running h alone, so a later layer does not undo a set zone that raises h above
the base outside its own area.

- `mode = "set"` (keys `h_m`, `transition_m`, `polygon`; `h_out_m` is not used): inside the polygon h = h_m whatever
  the layers before it give (finer or coarser); outside it h blends linearly in the geodesic distance from h_m to the
  running h over `transition_m` (0: a step). The layers after it still apply on top.
- `mode = "smooth"` (keys `smooth_m`, `transition_m`, `polygon`; `h_m` and `h_out_m` are not used): inside the
  polygon the running h is replaced by its smoothed value s, exp of the mean of ln h over the geodesic disc of radius
  `smooth_m` (biweight weights (1 − (r/smooth_m)²)², so s is smooth); outside it h = (1 − w) h + w s with w falling
  linearly from 1 to 0 over `transition_m`. s is computed once per run on a cubed-sphere grid (S36: s held at
  smooth_m/12, ln h sampled at half the smallest layer h, as the limiter's grid).
- `mode = "blend"` (keys `h_m`, `polygon`, `outer` = [[lon, lat], ...]; `transition_m` is not used, give 0): h = h_m
  inside the polygon, the running h outside `outer`, and between the two edges ln h = (1 − t) ln h_m + t ln h_e with
  t = d_in / (d_in + d_out), the geodesic distances to the polygon's and to the outer polygon's edge; h_e is the
  running h at the nearest point of the outer edge (ln h averaged over ±50 km along it). So h rises from h_m to the
  running h of the outer edge over whatever distance lies between the edges (S37). Both polygons must fit in a
  hemisphere, the polygon inside the outer one. Used for an ORCA part joined to a fine region (the `orca` layer
  before it gives the running h: ORCA's own cells at ORCA's edge).
  With `h_factor` in place of `h_m` (B110): h = h_factor × the running h inside the polygon, and between the edges
  ln h = ln h_run + (1 − t) ln h_factor (the running h at the point itself). For an ORCA grid refined k times inside
  (`fine_orca`): `h_factor = 1/k` after the coarse grid's `orca` layer (`docs/examples/orca025_arctic_v4.toml`).

The global clip (`h_min_m`, `h_max_m`), the strait zones and the gradient limiter come after all layers, as before.

**Coast classes** (`coasts`, decisions S33): `"all"` (default) measures d to every coast loop; `"continents"` and
`"islands"` to one class. An island is a land polygon of the coast source (a loop after `[coast] min_area_km2`)
whose area on the sphere is below `island_max_km2` (default 100 000 km²); Antarctica (the loop around the South Pole)
is a continent whatever its area. On GSHHS i with `min_area_km2 = 3000`: 136 islands, 24 continents (Iceland
102 000 km² and Great Britain 219 000 km² are continents; Ireland 83 000 and Hokkaido 78 000 are islands). Two layers
refine differently: e.g. continents at 30 km and islands at 60 km, or no island layer at all.

**h variants** (`[size] from`, decisions S35). `from = "<file>.toml"` (relative to the config file's folder) loads that
file's `[size]` keys and `[[size.layers]]` first; the config's own `[size]` keys override them and its own layers are
appended after the file's. A variant file holds only `[size]` and `[[size.layers]]` (and may itself have a `from`);
`meshom.config.dumps_size(cfg.size)` writes one and `loads_size(text, base_dir)` reads it. `load` inlines the
variant: `dumps` and the run's `config.resolved.toml` carry the merged layers and no `from`, so a run stays
reproducible when the variant file changes (`Config.size_from` records the file's path).

Every layer takes an optional `name` (shown by the UI) and `enabled` (default `true`; `false` keeps the layer in the
file and out of the run: it is not a field, does not set the base h and does not set the limiter's grid spacing; the
UI's layer checkbox). `enabled = true` and empty names are not written by `dumps`. The classes are in
`meshom/size.py` (`Uniform`, `Latitude`, `CoastDistance`, `Zone`, `SetZone`, `SmoothZone`, `Minimum`, `Clipped`);
`meshom.run.layer_fields(cfg, coast)` builds them from a config (enabled layers only) and `meshom.size.layered(fields,
base_h, spacing_m)` combines them in order (`meshom.run.background_h`: layers and clip; `size_field`: the run's h).

**Gradient limiter** (`meshom.size.limit_gradient`). The largest field ≤ h with
h_j ≤ h_i + g d_ij on every edge of a grid, by Dijkstra-style relaxation (Numba, deterministic). The grid is an
equiangular cubed sphere (no pole or antimeridian; cell edges between 0.71 and 1 times the nominal spacing; O(1)
point location) with a nominal spacing of half the smallest h the layers can take (× scale), capped at
`limiter_max_nodes` (12 M nodes ≈ 7.1 km; the stage record says when the cap was hit). Its 16-neighbour stencil makes
the graph distance at most 2.7 % longer than the geodesic, so the result may be up to that much less graded off the
stencil directions. h is returned as a bilinear interpolant on the grid: equal to the input at nodes where the input
was already graded. Limiting runs across land (as fesom2_mesh_gen); that only ever gives smaller h, never steeper.

### `[bathymetry]`

The `bathymetry` stage of `meshom run` (`meshom.bathymetry.node_depth`) gives every node of the final mesh a depth,
metres positive down; `depth.npy` holds it, the stage record in `stages.json` its statistics, and both output formats
write it (`depth.out` of FESOM-C, `aux3d.out` of FESOM2). Without a source every node gets the placeholder 4400 m.

| key | type | default | meaning |
|---|---|---|---|
| `source` | path or `"none"` | `"none"` | a regular lon-lat netCDF grid of elevation (positive up, ocean < 0) such as GEBCO 2024 (`/work/ab0995/a270126/projects/mesh_generation/bathymetry/GEBCO_2024.nc`, `SOURCES.md` there); read in bands of rows, never whole |
| `variable` | string | `"elevation"` | the grid variable; its two dimensions are latitude and longitude (cell centres) |
| `min_depth_m` | float | 10 | every node shallower than the floor gets it; the floor is `min_depth_m` (fesom2_mesh_gen's 10 m), for `format = "fesom2"` / `"fesom3"` at least the bottom of the `min_layers`-th level |
| `min_layers` | int | 4 | `fesom2` / `fesom3` only: the floor is max(`min_depth_m`, zbar[`min_layers`]) (zbar[0] = 0 the surface): 30 m with the CORE2 levels. FESOM2 and FESOM3 (fesom_ini, `use_depthfile = 'aux3d'`) raise every node above zbar(`thers_zbar_lev` = 5) = zbar[4] and give every element ≥ 4 layers, so 4 writes the depth the model uses; 0 = `min_depth_m` only |
| `depth_at` | `"node"` or `"cells_deepest"` | `"node"` | `"node"`: depth per node: the mean of the grid's **ocean** cells in a lon-lat box of half-width = half the node's mean edge length (the node's control cell); nodes whose box has no ocean cell take the 1/length-weighted mean of their neighbours, ring by ring; then the floor. `"cells_deepest"` (B103, FESOM3's ORCA rule on our cells): per element the mean of the ocean cells in a box of the element's area, a node the deepest of its elements (so no coastal node makes an element shallower than its own water: narrow straits keep their depth); then the same fill and floor |
| `smooth_passes` | int | 0 (off) | passes of volume-conserving diffusion of the node depths over the whole mesh (`meshom.bathymetry.smooth_depth`): per edge the flux min(A_i/deg_i, A_j/deg_j)/2 · (d_j − d_i), A = the node's area share (area/3 of each triangle, /4 of each quad); Σ A d is kept to round-off, no new minimum or maximum, each pass moves a node about half way to its neighbours' mean. 4 is what lets the CORE2-like strait mesh run 10 days at 45 min (decisions O29) |
| `max_slope` | float | 0 (off) | slope limiter (`limit_slope`): wherever \|d_i − d_j\| > `max_slope` · L_ij water moves from the deeper node to the shallower (volume kept, range kept) until no edge exceeds it; measured to make Gibraltar worse on the CORE2-like mesh, so off |
| `carve_taper` | float | 1 | carving of the straits' `depth_m` floors (`meshom.straits.depth_floor`): depth ≥ `depth_m` · w, w = 1 within the strait's `h_m` of its path, then (1 + cos(π x / T)) / 2 over T = `carve_taper` · `h_m`, 0 beyond |
| `levels` | list, `"core2"` or path | `"core2"` | the FESOM2 level surfaces zbar (m): `"core2"` = the 48 levels (47 layers) of the CORE2 mesh, a list of depths, or a file holding nl and then nl values (`aux3d.out`, `depth_zlev.out`); written into `aux3d.out` |

The depth rules run in this order (`meshom.bathymetry.depth_policy`): box mean and fill, floor, smoothing, slope
limiter, carving, floor. The stage record (`stages.json`, `bathymetry` → `policy`) counts the nodes moved by more than
1 m and by more than 10 %, the largest moves up and down, inside the strait corridors and elsewhere, the relative
volume change (carving adds water; smoothing and the limiter keep it) and the edge slope p99 before and after.

#### Depth regions (`[[bathymetry.regions]]`) and node edits

Two ways to change the depth the model gets, both with the last word (decisions O35-O38):

1. **`[[bathymetry.regions]]`, rules by area in the config** (C1). They act after every other depth step (the box
   mean or `cells_deepest`, the depth rules above, m16_orca's own NEMO node depths), in file order, on the nodes
   inside the area, in `meshom run`'s bathymetry stage (`depth.npy`, `fesom/`) and in every `meshom write` of that
   run (`meshom.run.model_depth`, `meshom.bathymetry.apply_regions`). A run keeps the regions of its
   `config.resolved.toml`: a region added to the config later reaches the next run, not the old run's writes.
2. **Node edits of one mesh file** (C2): `<run>/<mesh stem>.depth_edits.json` (next to the mesh file), written by
   the UI's node tool, applied by `meshom write` of that mesh after the regions (`meshom.bathymetry.apply_node_edits`).
   `meshom run` never reads them.

| key of a region | type | default | meaning |
|---|---|---|---|
| `name` | string | `""` | shown in the record |
| `box` | `[lon_min, lat_min, lon_max, lat_max]` | none | degrees; sides along meridians and parallels, edges included; `lon_min > lon_max` (or `lon_max > 180`) crosses 180°. Note the order differs from `[[strait_regions]]` `box` ([west, east, south, north]) |
| `polygon` | `[[lon, lat], ...]` | `[]` | used when there is no `box`; great-circle edges, within a hemisphere, any orientation, may cross 180° |
| `depth_m` | float | none | the nodes inside are set to it (applied first) |
| `min_depth_m` | float | none | then raised to at least it |
| `max_depth_m` | float | none | then lowered to at most it |

After its rule a region's nodes are kept at the run's floor (`min_depth_m`, for `fesom2` / `fesom3` at least the
bottom of the `min_layers`-th level: 30 m with CORE2): a `max_depth_m` or `depth_m` below it gives the floor, the
depth FESOM would raise the node to anyway (O36). A later region overrides an earlier one where they overlap. The
bathymetry stage record has `regions`: per region `name`, `n_nodes` (inside), `n_changed`, `n_clamped_to_floor`.
With `source = "none"` the regions act on the 4400 m placeholder.

The edits file (O37; node ids are the 0-based node indices of the `.npz` mesh file, as the hand editor's):

```json
{"mesh": "edited-01.npz", "n_nodes": 144410,
 "edits": [{"op": "set", "depth_m": 60.0, "nodes": [101, 102], "outline": [[-5.9, 35.8], [-5.4, 35.8], [-5.4, 36.1]],
            "date": "2026-10-27T15:00:00"}]}
```

`op`: `"set"` d = `depth_m`, `"min"` d = max(d, `depth_m`), `"max"` d = min(d, `depth_m`); the edits apply in list
order, then the edited nodes are kept at the floor as above. `outline` and `date` are the page's (not read). An
`n_nodes` other than the mesh's stops the write (the file belongs to another mesh: a hand editor that copies the
file to a new `edited-NN` sets `mesh` and `n_nodes` to the new file's). The write's record gets
`"depth_edits": {"file", "n_edits", "n_nodes_changed", "n_clamped_to_floor", "floor_m"}` (also in its
`bathymetry` stage), and `<out>/depth_base.npy` holds the depth after the regions and before the node edits (C3,
O38), so a preview can apply the current edits to an edit-free base.

### `[[blocks]]`

Hand-placed curvilinear blocks (decisions U6, B60), one table per block. The m14
methods (`m14_blocks`, `_cut`, `_tri`, `m14_blocks_rep`; not `_igm`, `_mix`) build them as fixed blocks before their strait blocks and
their automatic placement (`meshom/methods/m14/hand.py`); the other methods ignore them. Coast-to-coast blocks are
built only this way, in cut mode. `m14_blocks_rep` keeps a built block as drawn: its repair,
coast stage, coast fix, fit and optimisation stage never change the block's cells nor move its nodes (decisions R24,
method option `freeze_blocks`, default `["hand"]`); before that, under `[coast] treatment = "cut"` the repair
remeshed the continent channel by hand, 3 cells across (its banks are not yet coast nodes on the sphere), and the
block lost its protection in the land cut (B60).

| key | type | default | meaning |
|---|---|---|---|
| `name` | string | `""` | shown in the records |
| `corners` | `[[lon, lat] × 4]` | required | counter-clockwise: the block lies on the left of each side (side k runs from corner k to corner k + 1, side 4 back to corner 1) |
| `rows` | int | from h | cells along sides 1-2 and 3-4; default: the rounded mean ∫ds/h of the two sides |
| `cols` | int | from h | cells along sides 2-3 and 4-1; default as `rows` |
| `h_m` | float | the run's h | the cell size inside (constant) |
| `sides` | 4 of `"coast"`, `"cut"`, `"lat"`, `"offset"` | per side: `coast` when both corners lie within 1 h of one coast loop and the coast path between them is shorter than 3 × the chord, else `cut` | `coast` = the coast from the one corner forward along their loop to the other (water on the left), filtered by `filter_m`; `cut` = the great-circle arc; `lat` = the parallel (the latitude linear in longitude between the corners); `offset` = an offshore curve parallel to the coast: the opposite side when that is a `coast` side (a coastal band), else the coast between the corners' nearest coast points, extruded along the normals of its smoothing at 8 h by the corners' distances and cut where it passes nearest to the corners |
| `snap_m` | float | 1 h | a corner within this distance of the coast moves onto its nearest coast point (0 = never); not a corner whose two sides are given and neither is `coast` (a cut-mode corner on land stays) |
| `filter_m` | float | 2 h | Gaussian filter of the coast sides along their arc length (C35) |

h is the run's h at the corners (or `h_m`); with `[mesh] scale` ≠ 1, `h_m` is multiplied by the scale and `rows` /
`cols` are divided by it (rounded, at least 1), as h is. The sides are laid on the sphere by `meshom run` against the
true coast with `cut` and against the filtered coast with `loops` (the coast the method gets; `stages.json` stage
`blocks`: the sides' kinds and the corners after the snap). The block is the harmonic
Dirichlet map of its four sides (as the strait blocks, decision S31) with 30 "size" sweeps; it is built as given, not
tested or split: its B22 numbers (|ln s/h| per edge, quad angles, inverted cells) are in `m14.json` `fixed_blocks`
(kind `"hand"`, with `b22` = every edge within 0.10 and every angle ≤ 135°), and `row_cells_h` / `col_cells_h`, the
lengths of its grid lines in h, say what `rows` / `cols` would fit. The default counts follow the sides only: a block
whose middle is longer than its sides needs them by hand (the Atlantic between 20° S and 40° N is 59 h long on its
parallels and 72 h at the equator; `rows = 64` passes B22, the default 59 does not, B60).

How a block enters the mesh:
- **cut mode** (`treatment = "cut"`, or `m14_blocks_cut`): the method meshes the closed sphere, so corners may lie on
  land. A block without a `coast` side is cut by the land mask with everything else (ORCA style: the coast is the
  staircase of its cells). A block with a `coast` side is water: the elements whose centroid is inside its outline are
  kept and protected in the land cut (as a strait corridor), so its fitted coast sides stay.
- **conforming** (`treatment = "loops"`, or the compete cases with loops): a block with a `coast` side joins the coast
  loops through its other sides (as a strait block, B59); a block without one is one more exact outline of the fill
  and its sides must lie in water.

### NEMO's ORCA grids (`method = "m16_orca"`)

NEMO's ORCA grids built by NEMO's own generator, whole or between two latitudes with meshom elsewhere (decisions
B71-B86). `meshom.orca` is a port of G. Madec's tripolar generator with A. Sellar's tropical stretch (B78; see
`NOTICE`); the family is a set
of presets (B79-B81, B231): `ORCA2`, `ORCA1`, `ORCA05`, `ORCA025`, `ORCA12`, `ORCA36`, each checked against a NEMO
grid (ORCA1, ORCA05, ORCA025, ORCA12, ORCA36 to below 1 m away from the poles; ORCA2 to a few km, its older generator's
functions being fitted, and without its hand-made 1-degree Mediterranean patch). `ORCA36` = `ORCA12` × 3 is meant for
a box (`nest.inner`): its whole grid has 119 M cells.

- **Whole** (`lat_south`, `lat_north` unset): the preset's grid, its north fold closed, as quads (a few pole cells
  dropped, on land); the land as NEMO's own file of that grid has it, cell for cell (B82; the depth rule where it does
  not map, B83), cleaned as FESOM3 needs (B84). The run's land cut leaves the ORCA cells as they are (B85).
- **A part bounded by a line** (`region`, B100): as a latitude range, but the ORCA part ends per column at a line
  (the cells whose T point lies inside the polygon, contiguous from the south); the edge is a staircase of ORCA cell
  edges, the north cap is meshed against it. `docs/examples/orca025_arctic45.toml` (ORCA025 joined to a 4.5 km quad
  Arctic; its h: an `orca` layer and a `blend` zone).
- **A refined part around the fold** (`fine_orca`, `fine_lat`, B108): with `region` or `lat_north`, a second ORCA
  part from a preset that is `orca` refined k times on the same map (`ORCA12` = `ORCA025` × 3: every third ORCA12
  line is an ORCA025 line; checked, `meshom.orca.mesh.nesting`): per column the cells from the fold down while the T
  point lies north of `fine_lat`, masked like the coarse part (its own NEMO source, B92), kept unchanged. The cap is
  then the band between the two parts' edges; give it `cap_opts` `auto_blocks = false`, `strait_blocks = false` and
  `[mesh.opts.cap_opts.fill] frame = "<coarse preset>"` (B109: m02's field is ORCA's own i/j directions, so the
  fill only changes size) and `seed_loops = [1]` (B112: one front, grown from the fine part's edge, the cap's second
  loop; the coarse edge's nodes only stop it), and h a `blend` zone with `h_factor = 1/k` (B110). `docs/examples/orca025_arctic_v4.toml`.
- **The ladder between the two parts** (`transition = "ladder"`, `ladder_lat = [lat_c, lat_f]`, B114-B116; with
  `fine_orca`, without `region`, `lat_north`, `fine_lat`): the coarse part ends at a plain F row (the last one lying
  wholly south of `lat_c`), the fine part starts at a plain fine F row (the last lying wholly south of `lat_f`; every
  column up to the fold), and between them `meshom.orca.ladder` builds a structured strip in ORCA's own index space
  in place of `cap_method`: rows on fractional j-lines whose height grades from the coarse cells to the fine ones as
  the `blend` zone's h (B110), columns on i-lines, each coarse column turned into three by two Y-splits (one triangle
  each), staggered so that no row or column carries a seam. Both edges are the grids' own F points (node for node).
  Its h: the `blend` zone with `polygon` = the fine edge row, `outer` = the coarse edge row
  (`scripts/orca025_arctic_v5.py`, `docs/examples/orca025_arctic_v5.toml`).
- **A finer ORCA nested in a box** (`[mesh.opts.nest]`, B172-B176): a box of four coarse F lines (columns c0..c1,
  rows r0..r1; or a lon-lat box converted to the enclosing lines) holds the refined preset (`ORCA12` in `ORCA025`:
  its cells inside the box, its own NEMO source, kept unchanged); the coarse part keeps every cell outside the box
  widened by `width / 3` coarse cells; between the two boxes `meshom.orca.nest` builds a frame in ORCA's index space:
  n + 1 rectangular rings graded from the coarse cells to the fine ones (as the ladder's rows), tensor corners (one
  square cell per band on each diagonal, no irregular node there), the ladder's staggered Y-splits along the sides
  and the band rule near the corners. Both box edges are the grids' own F points (node for node). Combines with the
  other forms (the coarse part loses the box either way; with `fine_orca` both fine parts use one preset). Keep the
  box below the fold and away from NEMO's grid poles; a box across the east-west wrap is fine. Its h: a `blend` zone
  with `h_factor = 1/3`, `polygon` = the fine box's edge, `outer` = the coarse box's edge
  (`scripts/orca025_gs_nest.py`, `docs/examples/orca025_gs_nest.toml`).
- **A nest in the nest** (`[mesh.opts.nest.inner]`, B230): a third preset that refines the nest's fine preset by 3
  (`ORCA36` in `ORCA12`, so 9 times `ORCA025`) fills the nest's fine box: a second frame of `inner.width` steps of
  the third preset runs inside the box's edge (its outer ring is the first frame's inner ring: the box's edge, every
  ORCA12 F point) and the third preset's cells fill the rest, with their own NEMO source. The box then holds no cells
  of the middle preset; the step from the coarse cells to the finest is two graded frames of 3. Size the box so that
  the third preset's part (the box less `inner.width / 3` middle cells on every side) covers the region. Its h: a
  second `blend` zone with `h_factor = 1/3`, `polygon` = the inner box's edge, `outer` = the nest's fine box's edge.
- **The nest's depth** (`[mesh.opts.nest] depth = "gebco"`, B216): FESOM3's ORCA depth (the deepest of the four
  NEMO cells' levels) on the finer cells digs a strait's sills a level or two too deep. With `depth = "gebco"` every
  node strictly inside the coarse box (the box's own edge nodes keep the coarse grid's depth) takes meshom's grid
  rule instead: the box mean of the grid's ocean cells per node, no smoothing, no strait depth floor, the slope
  limiter, floor and regions of `[bathymetry]`; `depth_channel = "near"` and `sill_level = "grid"` add an insert's
  channel raise and the grid's saddle held at its level (the ends: the two largest wet stretches of the coarse box's
  edge). The rest of the mesh keeps the run's depth. `meshom write` (or the run's bathymetry stage) recomputes it on
  the saved mesh; the mesh does not change.
- **A latitude range** (`lat_south` and / or `lat_north`): the ORCA rows from the first F row south of `lat_south` to
  the F row whose mean latitude is nearest `lat_north`, NEMO's mask on them; a north cap (when `lat_north` is set) and
  a south cap (when `lat_south` lies north of the preset's own first row) meshed by `cap_method` on the domain bounded
  by that row as one exact loop: the joint is conforming, the caps' elements next to it may be triangles. The run cuts
  the caps by the coast (meshom's rule there), the ORCA part keeps NEMO's mask; where a strait crosses the joint the
  cap's cut decides.

Options (`[mesh.opts]`):

| key | type | default | meaning |
|---|---|---|---|
| `orca` | string | `"ORCA1"` | the preset: `ORCA2`, `ORCA1`, `ORCA05`, `ORCA025`, `ORCA12`, `ORCA36` (`meshom.orca.presets.PRESETS`) |
| `lat_south` | float or unset | unset | unset: the preset's own first row (NEMO's standard rows: ORCA1 78.3 S); a latitude: the first F row is the last one south of it; the grid is extended south when the preset ends north of it (B74) |
| `lat_north` | float or unset | unset | unset: to the north fold (the whole grid); a latitude: the last F row is the one whose mean latitude is nearest it (B74) |
| `region` | [[lon, lat], ...] or unset | unset | the ORCA part bounded by a line (B100): a polygon of the lon-lat plane (straight edges, holding the whole south: corners at lon ±181, lat −91); per ORCA column the cells from the first row up to the first T point outside it (ORCA's land cells count, so seas behind land inside it stay ORCA); the north cap is meshed against that staircase. Excludes `lat_north` |
| `fine_orca` | string or unset | unset | B108: the refined preset of the part around the fold (`ORCA12` with `ORCA025`); needs `region` or `lat_north` and `fine_lat` |
| `fine_lat` | float | unset | B108: the refined part's cells per column from the fold down while their T point lies north of it |
| `transition` | string | `"fill"` | B114: between a coarse and a `fine_orca` part, `"fill"` = `cap_method` (B108-B112), `"ladder"` = the structured strip of `meshom.orca.ladder` between two plain rows |
| `ladder_lat` | [float, float] | unset | B114: `transition = "ladder"`: the coarse part up to its last F row lying wholly south of the first latitude, the fine part from its last F row lying wholly south of the second (it covers everything north of it) |
| `ladder_theta_deg` | float | 18 | B115: an arm's tilt on the sphere at a Y-split (the triangle's apex 2θ, the quad corner at the arm 90° + θ) |
| `nest` | table or unset | unset | B176: `fine_orca` (the refined preset, k = 3), `box` = [c0, c1, r0, r1] (the fine box as coarse F lines of the preset's own rows; columns unwrapped, c0 < c1) or `lonlat` = [lon0, lat0, lon1, lat1] (the smallest box of F lines holding every cell whose T point lies inside; lon0 > lon1 crosses the antimeridian), `width` (the frame's width in fine steps, a multiple of 3, default 39: 24 rings, B172), `theta_deg` (an arm's tilt at a ladder Y-split, default 18, B115), `relax` (the frame's relaxation weights [widths, tilts, bases], default [1, 3, 3], false: off, B175), `inner` (a table, unset by default: a nest in the nest, B230: `fine_orca` = a preset refining `fine_orca` by 3, `width` in its steps (default 39), `theta_deg`, `relax` as above), `depth` (B216: `""` by default = the run's depth rules: the ORCA nodes FESOM3's depth from NEMO's levels with `mask = "fesom3"`, the frames' nodes `[bathymetry]`; `"gebco"`: every node inside the coarse box takes the box mean of the grid's ocean cells per node, without smoothing and without the strait depth floors, as an insert's own nodes with `depth_at = "node"`, `depth_smooth = 0`, `strait_floor = false`), `depth_channel` and `sill_level` (with `depth = "gebco"`: an insert's `depth_channel` (B213) and `sill_level` (B212, B215) on those nodes, the channel's ends the deepest box means along the two largest wet stretches of the coarse box's edge, B218) |
| `mask` | string | `"nemo"` | `"nemo"`: NEMO's mask on the ORCA cells (B82-B84); `"fesom3"`: FESOM3's rules (below, B94); `"all"`: every cell, the land as the run's cut has it for the caps (the ORCA cells are still protected, B85; for tests) |
| `ref_dir` | path | `$MESHOM_ORCA_REFS` | the folder of NEMO's reference grids as npz (`scripts/grids/orca_refs_npz.py OUT_DIR`; eORCA05: `... OUT_DIR eorca05=PATH`) |
| `levels_dir` | path | `$MESHOM_ORCA_LEVELS` | `mask = "fesom3"`: the folder of NEMO's bottom levels / model bathymetry as npz (`scripts/grids/orca_levels_npz.py`, `scripts/grids/nemo_inputs_npz.py`; the names of `meshom.orca.fesom3.SOURCES`, B92) |
| `gebco` | path | unset | the depth rule's grid (B83); required when a cell has no NEMO mask (eORCA's ice-shelf rows, ORCA2's Mediterranean patch) |
| `min_depth_m` | float | 10 | the depth rule's minimum mean cell depth (B83) |
| `keep_seas` | bool | false | NEMO's closed seas (the Caspian, ORCA025's Great Lakes, ...) removed, as every water body cut off from the ocean (B171, C48); true keeps them as separate water bodies (B84) and the run's cut keeps them too (a water body made only of ORCA cells, B85; one that meets a cap follows the cap's rule, B95); the write stage then reports more than one water body (C52) |
| `cap_method` | string | `"m14_blocks_rep"` | the caps' method (any key of `METHODS` that keeps an exact loop's nodes: the m14 methods, m15_tri; B75) |
| `cap_opts` | table | `{}` | the cap method's options; the run's size tolerance and, for the m14 methods under a land cut, the true coast (`land`, B70) are added unless given here. One block over a uniform fine region and none in the transition (B106): `place_region` = the fine zone's polygon, `max_blocks = 1`, `b_cap = 100` (the B pre-screen off) (`[mesh.opts.cap_opts]`, as `docs/examples/orca025_arctic45.toml`) |
| `size_tolerance` | float | 0.20 | the cap method's tolerance and the run's size gate (m14's default; `[mesh] size_tolerance` overrides it) |

With `m16_orca` keep `[coast] treatment = "cut"` (it gives the ORCA cells their mask and cuts the caps), `[coast]
passage_min_cells` 0 for a whole grid, 2 for a range mesh (C51 acts on the caps only: passages made only of ORCA cells
are skipped, B95), `[coast] min_area_km2 = 0`,
`[mesh] optimize` unset and `[run] smooth` false (both would move ORCA nodes). h does not shape the ORCA part; give it
the grid's zonal spacing (`latitude` layer: `h_m` = 222 390 for ORCA2, 111 195 ORCA1, 55 597.5 ORCA05, 27 798.75
ORCA025, 9 266.25 ORCA12, 3 088.75 ORCA36): the size gate then fails on ORCA's anisotropic cells by design (report it,
do not fix it).
The method writes `orca.json` (the grid, the rows, the mask and its clean-up with places, the joints) and
`orca_belt.npz` (the ORCA part's points and elements, its T points and NEMO's flags) into the run folder;
`scripts/orca_report.py` (the ORCA part unchanged, quality per part, the joint), `scripts/orca_mask_report.py` (the
mask against NEMO's: cells and km2 that differ, the Red Sea, the Caspian, Gibraltar, the Indonesian passages, ...) and
`scripts/orca_figures.py` read them. Configs: `docs/examples/orca1_whole.toml`, `orca2_whole.toml`,
`orca025_whole.toml`, `orca1_40n_arctic10.toml` (results: B86).

#### FESOM3's rules (`mask = "fesom3"`, `python -m meshom.orca.fesom3`)

FESOM3 runs on NEMO's ORCA025 turned into a quad mesh by FESOM3's own conversion (github.com/FESOM/fesom3).
`meshom.orca.fesom3` has its rules rewritten (decisions B87-B95): water where NEMO's bottom
level is > 0; one-cell land removed; 14 strait edits by index (Gibraltar, the Bosphorus, the Dardanelles, Bab-el-Mandeb,
the Nares Strait, Fury and Hecla, Hell Gate, the Nevelskoy Strait); the node depth the deepest of the four cells,
in NEMO's levels; the F points around the geographic North Pole straightened; the fold's nodes merged; a closing row in
the south; dead ends removed; then (ours) only the water connected to the world ocean kept (rule 12b, B170: the
largest water component; with cavities joined only at a common layer), and `write` refuses a mesh of more than
one water component (C52). Two ways:

- **The whole grid, FESOM3's files** (the reproduction): `python -m meshom.orca.fesom3 PRESET LEVELS_DIR OUT_DIR
  [--gebco GEBCO] [--fesom-ini BIN]` writes `nod2d.out`, `elem2d.out`, FESOM3's NEMO `depth.out` (the levels, the
  node depths, the node layer counts) and `fesom3_orca.json`; `--fesom-ini` runs fesom_ini with
  `use_depthfile="nemo"`. For ORCA025 with FESOM3's own inputs (`scripts/grids/nemo_inputs_npz.py
  ../fesom3_orca025/nemo025q LEVELS_DIR/nemo025q.npz`) it gives FESOM3's mesh (B88).
- **In a meshom run** (`[mesh.opts] mask = "fesom3"`, `levels_dir`): the ORCA cells take FESOM3's rules 1-4 (the
  water); B84's clean-up follows (its spurs are FESOM3's dead ends; its pinch fans are meshom's, FESOM3 keeps 17
  pinch nodes in ORCA025); `orca_belt.npz` holds the ORCA nodes' layer counts and levels (`node_level`, `z`) and
  `m16_orca.node_depth(run_dir, mesh)` gives the ORCA nodes their FESOM3 depth, which the run's bathymetry stage puts
  in place of the GEBCO depth on those nodes (B94).

Each preset's water (`meshom.orca.fesom3.SOURCES`, B92): ORCA025 FESOM3's own ORCA025.L46 `mbathy` (by index);
ORCA2 NEMO 5.0's `bottom_level` (31 levels); ORCA05 MEOM's eORCA05.L121 `mbathy` (121 levels); ORCA1 and ORCA12 the
Met Office GO6 model bathymetry (Zenodo 22010172, 15495870) on L46 (`level_of_depth`); ORCA36 MEOM's eORCA36.L121
`bathy_metry` where `bottom_level` > 0 and `top_level` = 1 (an ice-shelf cavity is land) on L46 (B231); cells without
a NEMO value (eORCA's ice-shelf rows, ORCA2's Mediterranean patch, rows added in the south) by GEBCO (B83's rule) on
the same levels. Every preset but ORCA025 takes the strait edits by place where the strait is closed (`carry_edits`,
B91).

### `[[inserts]]` (`python -m meshom.insert`)

Curvilinear inserts are fitted into a finished run afterwards (`python -m meshom.insert --run RUN_DIR --config
FILE.toml`), not by `meshom run` (decisions B158-B169, B180-B186, B190-B223). `kind = "strait"` (default): a strait
(`strait` or `path`, `cells` or `h_fine_m`, `reach_km`, join hints `mouth_a` / `mouth_b`, `ljoins`); refused inserts say
which rule fails and which cell counts both sides accept. The banks follow the real coast (`coast = "real"`, default);
when a bank's two ends lie nearest to different pieces of it (an island near one end), the piece is chosen for both
ends together by the host's bank, and a bank that still takes the host's coast is listed in the record's and the
score file's `bank_fallback` (which bank, why, its ends) with a warning in the log (B219). A strait can be fitted to its host side by side (all off by
default): `cells_a` / `cells_b` ask for a join with exactly that many host edges on side A / B, the block's counts
running monotonically from the neck to it (equal counts: the rows only widen, no Y-split), or with `add_columns = true`
rising above both and coming back to the join's count before the join: `add_columns` lowers the size target where a block cell is
above 1.1 x the host cell (more columns and rows, by Y-splits spread over the rows), which helps where the block fans
out at a bank's turn, not at a staircase join's diagonal steps; `growth_a` / `growth_b` set the size slope along the
block on that side (default 0.12; the room and funnel rules scale with it), so a short, fast
transition is allowed where a strait opens fast; `bank_land_cells` lets the real banks move into land by up to that
many local insert cells where a smoother line lies there (capes cut, never into the water beyond the smoothed coast)
and squares a bank's end where it meets its join at an acute angle (the record's `bank_land`: how far the banks moved
and the water added; the added water gets its depth by the run's bathymetry rules, from the grid's ocean cells round
it and at least the depth floor). `fill = [[[lon, lat], ...], ...]` (strait; none by default) draws lines that cut
water off a bank: each line's ends move onto the nearest real coast, the line is smoothed as the banks are, and where
it lies at least half a local insert cell off the coast the main block's bank follows it; the water behind it (bays
found on that stretch included) is meshed as a strip of its own hanging off that bank, `fill_rows` rows deep (default
its depth / the local cell, at least 1), its columns fewer by Y-splits where the coast behind is shorter; where the line
comes closer to the coast than half a cell the bank stays on the coast. `corner_steps = true` cuts every convex step
of a hinted join's host staircase by a corner triangle, not only its bends over two edges; `corner_split = true`
splits each corner triangle of a join at the midpoint of its diagonal, the midpoint a node of the block's first row (two
triangles of ~0.8 x the host cell and no long diagonal edge next to the block); `join_margin` (default 1.0) caps the
block's size target at the joins at that many host cell sizes, so the outermost rows are a little finer than the host
(with both, 0.85 and `add_columns`, a staircase join at Gibraltar keeps the insert's cells within 1 % of the host cell next to
them, at the cost of a size step of up to ~1.4 across the join). `fix_rings = N` (default 0 = off) smooths the
spliced mesh: meshom's mesh fixer stage a (free-node geometry, `meshom.meshfix.geom`) on the cells the insert created
or changed and N element rings round them, every other node held, the host's coast fixed and the insert's banks
sliding, sizes held at the smaller of the spliced mesh's and the host's node sizes; host nodes in the rings move (less
than a host cell) and keep their host depth in the model files; the smoothing's result is kept only when it is not
worse than the spliced mesh by the insert's objectives, in this order: no more invalid cells and no more cells below
the floor, then a lower angle excess above 120° (summed over the quads), then a worst corner lower by more than 0.5°,
then (angles tied) the sizes vote
(the neighbour size ratio p99, the largest cell against the host cell, the smallest cell); else no node moves (B220);
the record's `fix_rings` has the case, which mesh was kept and why (`kept`, `keep_rule`, `objectives` of both), the
cells before and after, the host nodes moved, the size floor and the validity. The host nodes the collar moved keep their
host depth too (B217: the splice appends them at their new positions; each is matched to the nearest new node within
its host spacing), and on another insert's result (`--mesh insert_<name>.npz`) every host node keeps that result's
saved depth (`fesom-insert_<name>/depth.npy`), so a chain of inserts keeps each one's own depth rule (B217; the
record's `depth.host_depth`, `depth.collar_moved_kept`, `collar_moved`; `insert_<name>.collar_moved.npy`: the ids
in the result and the host ids of those nodes). The depth of the insert's own nodes follows
the run's `[bathymetry]` unless the insert sets its own rule: `depth_at` (`"node"`: the box mean per node, keeps
sills and channels; `"cells_deepest"`; default the run's), `depth_smooth` (smoothing passes among the insert's own
nodes only; default: the run's passes over the whole mesh), `strait_floor` (default true: the run's strait depth
floors carve them too; false: the insert's own grid depth sets its sill), and `sill_level = "shallower"`: the
insert's sill (the bank-to-bank chain of its nodes whose deepest node is shallowest) gets the shallower of the two
model levels round its depth, by capping the chain's nodes and their cell neighbours just above the midpoint between
the levels (FESOM3's level finder rounds an element's mean depth to the nearest level and gives a node the deepest
of its elements). Three more rules act on the insert's own nodes along the strait's deep channel, all on the bathymetry grid at the mesh's scale (every depth these rules compute with comes from the grid, never from the host's node depths, B218): the channel line is the deepest path between the insert's two joins on the grid's box means at the insert's fine size, inside the insert's cells widened by one fine cell; its ends are the deepest box means sampled along each join, and when the connection between them reaches a join's own depth (the search is end-limited there) that end moves into the deepest water of the basin beyond the join (the record's `depth.channel`: `end_from`, `join_limited`, `end_limited`; the log warns when the basin beyond is no deeper). `depth_channel = "cells"` gives every insert node of a cell the line crosses at least the cell's channel depth (the box mean, box of the cell's area, centred on the line inside the cell); `depth_channel = "near"` gives every insert node within one of its own spacings of the line at least the box mean of its own box centred on the line's nearest point (the node's box moved onto the channel), so the channel's elements reach its depth in the model's levels (FESOM3 averages an element's nodes). Both act only on the line's narrows: the stretch round its narrowest point where the water across the line (the shortest crossing within 45° of its normal) is at most 6 times as wide as there, so the line's legs in the open basins are not dug. `depth_quantile = q` (0-1; default 0 = the mean) gives the insert's nodes the q-quantile of the ocean cells in their node box. `sill_level = "grid"` holds the grid's own saddle between the joins (the deepest connection on the grid's cells; with several barriers of that depth in series, the one nearest the line's shallowest point, B218) at its level: the bank-to-bank chain of the insert's nodes through the node nearest the saddle that follows the shallowest nodes, and every node of a cell holding one of its nodes, are capped just above the midpoint between the deepest level surface not below the saddle's depth and the next level, so no element path between the basins is deeper than that level. The record's `depth` says which rule each node set took (`channel`: the line, its ends and the grid's saddle; `sill`: the dam). `depth_area = true` (B250) then brings each section across the strait to the bathymetry grid's area: sections across the insert's centreline (between its joins) every half fine cell; each one's area in FESOM3's element bottoms (fesom_ini's levels: an element's level the level surface nearest the mean of its nodes' depths, a node the deepest of its elements, the bottom the shallowest of its nodes' levels; `meshom.io.fesom.model_levels`) against the grid's raw wet area along the same line within the mesh's banks widened by one local cell; sections that run into open water or a bay past a bank or over the host's cells are skipped, and only the strait's narrows count (the run of sections round the narrowest wet crossing up to the first one 6 times as wide, as the channel raise's narrows), not the open basins at its ends. The insert's own nodes are scaled below the depth floor (the shape across the section and the floor kept) by a factor per section (0.25-4), smoothed along the axis over one fine cell and tapered to 1 over two fine cells past the outer sections, in rounds until every section is within 5 % (the best of 20 rounds is kept; the level steps leave some sections off, and a section cannot be smaller than the depth floor times the mesh's width); the dam of `sill_level` keeps its cap (its nodes never go deeper); with the channel line (`depth_channel` or `sill_level`) the dam's nodes and the channel's cells are scaled as the others, but the strait's deepest connection in FESOM3's element bottoms, between the two ends of the insert's axis and through the cells near the channel line, stays within one level of the level nearest the grid's saddle (B251: not shallower than the level above it, nor deeper than the level below it; a chain of cells from end to end along the line keeps that level, a round that leaves the band is not kept); when the input's connection is already shallower than that band, or without the line, the nodes the dam capped keep the cap and the cells the channel line crosses are never made shallower (B250). Where the channel's cells fill most of a section (3-4 cells across) the area cannot come down far. The record's `depth.area`: the sections counted and skipped, the deviation before and after (mean, max, within 5 %), the rounds, the factor and, per section, the grid's and the mesh's areas (`curve`). The cell count across the
narrowest row is the requested `cells` (the count smoothing never lifts it). `bays = "auto"` (default) gives a bay of the real coast a
block of its own when its mouth holds 3 insert cells and the bay reaches at least its mouth's width inland; smaller
pockets and lagoons are bridged (`true`: every bay with a mouth of 2 cells; `false`: none). `collar_rings`: on meshom's
own meshes and triangle hosts the host cells within this many element rings of the joins (default 10) are cleaned
with the insert's edge rows; on a structured host (ORCA) the default 0 keeps every host node and cell as it is and the
joins follow its grid lines (a number asks for a collar there too). `kind = "strip"` (`path` along the coast, `width_km` from the coast to
the offshore host chain, `h_fine_m`, `band_km`, `bays`) and `kind = "corridor"` (`path` = the axis in open water,
`width_km` the full width, `h_fine_m`, `band_km`) are **experimental**. A strip's rows come from a harmonic map of
the band (`rows = "harmonic"`, default): its rows cannot cross, so capes, bays and bridged estuaries do not fold
them; `rows = "offset"` keeps the earlier offset curves for comparison (they can fold at sharp coast features, and
such a result is invalid: no model files). The strip's coast is the real coast smoothed at `bank_sigma_cells` fine
cells; where quads next to it keep a corner above 125 deg the coast is smoothed more there and the band rebuilt, at
most `coast_rounds` times (default 3), moving the coast at most `coast_shift_cells` fine cells (default 1). Corridors
use the offset rows. `kind = "depth"` changes no cell (B223): the host's nodes within `width_km` / 2 of the path
(`strait` or `path`), or inside `polygon = [[lon, lat], ...]` (great-circle edges), take the grid rule of an insert's
own nodes as the nest's depth does (B216: the box mean per node, no smoothing, no strait depth floor, with
`depth_channel` and `sill_level` as above, the channel line between the two largest pieces of host nodes just outside
the region on its cells; with `depth_area` the section areas as above, the region's nodes scaled on sections across the
entry's path, none for a polygon alone); every other node keeps the host's depth (the run's rules, or an insert's saved
depth, B217); written: `insert_<name>.npz` (the host mesh), `insert_<name>.json` (`region`, `depth`) and
`fesom-insert_<name>/depth.npy`, with model files when the run wrote them. The general recipe with examples of each
kind: `docs/examples/inserts.toml`; bands: `docs/examples/inserts_gulf_stream.toml`.
use the offset rows. `kind = "group"` (**experimental**, B240-B245) replaces every host cell whose centroid lies inside
`envelope` (`[[lon, lat], ...]`) by a mesh of the real water there: the joins are the host edges round the envelope's
cells, the host's coast between them becomes the real coast, the real islands inside are holes (an island closer than
`gap_km`, default 5 x `h_min_m`, joins the land; one smaller than `island_km2`, default 10 km^2, is dropped); each
`[[inserts.channels]]` table (`name`, cross lines `a` and `b` bank to bank, `cells` default 5) is a quad block with that
count from `a` to `b`, and the rest is a quad-dominant fill (at least `gap_cells`, default 3, cells across any passage,
size slope `growth`, default 0.12); `bank_smooth_cells` (default 0) smooths each block bank over that many block cells
(capes cut); `sill_level` and `depth_channel` act per channel. The general recipe with examples of each kind:
`docs/examples/inserts.toml`; bands: `docs/examples/inserts_gulf_stream.toml`.

### `[gates]`

Overrides of the quality gates `meshom.config.DEFAULT_GATES`; an unknown gate name is an error. The same
table is read by `meshom quality --gates`. `size_fidelity_p99` is the run's size tolerance
(`[mesh] size_tolerance`, else the method's default: 0.20 for the m14 methods; `methods.effective_tolerance`) unless
this table names it (`config.tolerance_gates`).

### `[run]`

| key | type | default | meaning |
|---|---|---|---|
| `timeout` | float | 1800 | seconds; the method's child process is killed after it (status `timeout`) |
| `smooth` | bool | false | also apply the common smoothing (`meshom.smooth`, boundary nodes fixed) and score that mesh |
| `figures` | bool | false | render the standard picture set into `figures/raw/` and `figures/smooth/` |

## What the UI writes

The web UI saves the config with `meshom.config.dumps` (every default written out, comments dropped) and starts runs
with it, so every key it writes is one of the tables above:

| panel | keys |
|---|---|
| 1 h builder | `[[size.layers]]`: every kind with its keys, `name`, `enabled = false` for a switched-off layer, zone `polygon` (drawn on the map; zoom windows become zones too), zone `mode` / `smooth_m` (set h, smooth h), `coast_distance` `coasts` / `island_max_km2` (h at continents, h at islands); `[size] grad_max` (the field next to "gradient-limited h"); h variants with `meshom.config.dumps_size` / `loads_size` and `[size] from` |
| 2 Coastline | `[coast] source` (a shapefile found on disk, or `"none"`), `min_area_km2`, `smoothing_m` (the coast filter's closing radius, `loops` only) |
| 2 Coastline, "Bathymetry" | `[bathymetry] source` (empty: `"none"`), `variable`, `min_depth_m`, `depth_at`, `levels` (`core2`, a file, or a JSON list) |
| 4 Blocks | `[[blocks]]` (TOML text: the keys of the table above) |
| 5 Method | `[mesh] method`, `[mesh] opts` (only the options that differ from the method module's `DEFAULTS`; cleared when the method changes) |
| 7 Runs | `[mesh] out` = the new run folder and `[coast] source` made absolute, in that folder's `config.toml` only |

`[gates]`, `[run]`, `[mesh] scale`, `[mesh] format` and the other `[size]` keys are kept as loaded. In
the page's JSON a non-finite float travels as the string `"inf"` and integral floats come back as integers; the server
converts them back (`ui.common.scenario.from_json`: `meshom.config.from_dict` for the typed tables, the method's
`DEFAULTS` and `DEFAULT_GATES` for `[mesh] opts` and `[gates]`), so load → edit → save → load is exact
(tests/test_ui_web.py).

## Running

```
meshom run CONFIG [--out DIR]      # on Levante: scripts/on_node.sh -c 8 -m 7G -- python -m meshom.cli run CONFIG
meshom runs [DIR]                  # one line per run directory under DIR (default .tmp/runs)
meshom repair DIR [--polygon '[[lon, lat], ...]'] [--mesh raw.npz] [--out repaired] [--opts JSON]
                                   # local repair of a run's mesh (docs/methods/repair.md "CLI")
meshom write DIR [--mesh raw|smooth|optimized|edited-NN|PATH.npz] [--format fesom-c|fesom2|fesom3] [--out DIR2] [--fesom-ini]
                                   # a saved mesh of the run (a hand edit, a repair) as model files
meshom optimize DIR [--case C --scale S --method M] [--tol T] [--mesh raw.npz] [--out optimized] [--opts JSON]
                                   # the optimisation stage on a saved mesh (meshom.optimize; decisions Z1)
meshom ui [--config CONFIG] [--runs DIR] [--refs DIR] [--port N] [--allow-submit]    # the web UI (ui/web/README.md)
```

The run directory holds:

| file | content |
|---|---|
| `config.resolved.toml` | the config with every default filled in, `mesh.out`, `coast.source`, `bathymetry.source` (and a levels file) absolute |
| `stages.json` | one record per stage (`coast`, `size` with the limiter's grid, `coast_filter` (treatment `loops`), `generate`, `post` = the land cut, `smooth`, `score`, `figures`, `bathymetry`, `write` with `fesom2_checks` for the FESOM2 format, `fesom3_checks` (and `fesom3_ini` when set) for FESOM3, `straits` with the strait report): wall time, peak RSS, machine, commit |
| `quality.json` | `{"raw": record, "smooth": record, "optimized": record}` (smooth with `[run] smooth`, optimized when the run applies the stage itself (not for the m14 methods, whose stage is part of the method); the optimized record adds `input`, `optimize_s` and `optimize`: the stage's counts, times and checks), records as `meshom compete` writes them (`meshom.compete._score`: quality summary, size fidelity, irregular counts, ρ_c of the meshed water, gates) plus `config`; a failed run has `status` `error` or `timeout` and the error text |
| `raw.npz`, `smooth.npz`, `optimized.npz` | the meshes (`points`, `tris`, `quads`, `node_flag`); `optimized.npz` when the run applies the optimisation stage itself (`[mesh] optimize`) |
| `depth.npy` | the node depths of the final mesh (m, positive down) when `[bathymetry] source` is set (after the `[[bathymetry.regions]]`) |
| `fesom/` | the final mesh (`optimized.npz` when the run applied the optimisation stage, else smoothed if smoothing ran) in `[mesh] format`: FESOM-C (`meshom.io.fesom.write_fesomc`: `nod2d.out` with flag 1 on boundary nodes, `elem2d.out` with 4 columns, a triangle written `n1 n2 n3 n1`, clockwise, `depth.out` positive down, a placeholder `aux3d.out`) or FESOM2 (`write_fesom2`: `nod2d.out`, `elem2d.out` 3 columns clockwise, `aux3d.out`) or FESOM3 (`write_fesom3`: `nod2d.out`, `elem2d.out` 4 columns clockwise, `aux3d.out`; `fesom_ini/` holds the fesom_ini log when `fesom3_ini` is set); the depth is the bathymetry stage's, else 4400 m; coordinates are geographic lon-lat (FESOM2 rotates them on reading with `force_rotation = .true.`) |
| `figures/raw/`, `figures/smooth/` | the standard picture set and `index.json` when `[run] figures` |
| `<NAME>.npz`, `<NAME>.json` | written by `meshom repair DIR [--out NAME]` (default `repaired`): the repaired mesh (plus `is_block` for m14 meshes) and its record (docs/methods/repair.md "CLI") |
| `edited-NN.npz`, `edited-NN.json` | hand edits saved by the UI's mesh editor (ui/web/edit.py: moved nodes, split elements; the npz layout of `raw.npz`) |
| `fesom-<NAME>/` | written by `meshom write DIR --mesh NAME` (`meshom.run.write_mesh`, decisions O32): the mesh `<NAME>.npz` (default: the newest `edited-NN`, else the run's final mesh: `optimized`, else `smooth`, else `raw`) as model files in the run's `[mesh] format` (or `--format`), the depth recomputed on its nodes with the run's `[bathymetry]` (as the bathymetry stage; source `"none"`: 4400 m; the `[[bathymetry.regions]]` included), saved as `depth_base.npy`, then the node edits of `<NAME>.depth_edits.json` when it exists (record `depth_edits`; "Depth regions" under `[bathymetry]`), and `quality.json` = `{"<NAME>": record}`: the run's scoring (`compete._score` with the run's gates and size tolerance; domain as the run's), the format's checks (`fesom2_checks` / `fesom3_checks`; FESOM-C has none), `straits` / `water` when the config has straits, `passages` ({cells, n_narrow, narrow} as the run's record, when the run has a coast and `[coast] passage_min_cells` >= 2; a check only, nothing is widened), `fesom_ini` (with `--fesom-ini`, format fesom3 and `[mesh] fesom3_ini`: run_fesom_ini's record, work folder `fesom_ini-<NAME>/`), `stages` (size, score, bathymetry, write, then straits, passages, fesom_ini when they run). The run's own `fesom/` is never written. The UI calls `meshom.run.write_mesh(run_dir, mesh_name, fmt, out, progress, fesom_ini)` (None = the defaults above; `progress(stage)` is called as each stage starts; returns the record; `write_line(record)` is the summary line the command prints) |
| `cache/` | `meshom write`'s caches (decisions O33), never read by `meshom run`; delete them any time (the next write rebuilds them): `size.pkl` (the straits, the size field h and the domain loops of treatment `loops`, keyed by the sha256 of `config.resolved.toml`; a changed config rebuilds it; ~290 MB for the 281 k-node Arctic mesh C, the limiter grid), `rho_c.json` (the rho_c integrals per domain: the mesh boundary for treatment `cut`), `depth.npz` (the `[bathymetry]` box sums per node box, tagged with the grid file's path, size and modification time; it keeps the boxes of every mesh written, ~13 MB per 281 k nodes). A write after a hand edit reads only the boxes of the moved nodes and their neighbours; the depth is the same bit for bit as a full recomputation. A changed coast file with the same config is not noticed: delete `cache/` |
| `optimized.npz` | the optimised mesh when the run applies the stage (`[mesh] optimize` on a method without its own stage, or an m14 method under a land cut) (record `optimized` in `quality.json`, stage `optimize` in `stages.json`); `meshom optimize DIR [--out NAME]` writes `<NAME>.npz` and `<NAME>.json` (before / after scores and the stage's record) and, in a run directory, the record `<NAME>` of `quality.json` |

`meshom runs` reads the status from `quality.json` (the optimized record if present, else the smoothed one); a directory with
`config.resolved.toml` but no `quality.json` is listed as `running?` with its last stage.
