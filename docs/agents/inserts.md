---
name: meshom-inserts
description: How to fit a finer structured quad patch into a finished mesh with python -m meshom.insert - straits, bays, coastal strips and current corridors - its rules and refusals, its outputs and how to judge them. Read it when a request wants more cells in one strait, bay, coast or current of a mesh that already exists.
---

# Inserts: zoom patches in a finished mesh

## What an insert is

An insert replaces the cells of a finished mesh in one place by a finer structured quad block whose rows follow the
banks, joined to the host mesh along chains of host edges. It is made only on demand, by its own command, on any
finished run folder: meshom's mixed or triangle meshes and ORCA grids. **`meshom run` never makes an insert**, and
none is made by default.

- `kind = "strait"` (default): a strait, from the narrowest section out to joins on both sides where the size can
  grow back to the host's. Bays of the real coast inside the region get blocks of their own when large enough.
- `kind = "strip"` (experimental): a band along a coast, from the coast out to an offshore host chain.
- `kind = "corridor"` (experimental): a band along an axis in open water (a current).
- `kind = "depth"`: no mesh change; the host's nodes in a region round a strait take the insert's GEBCO depth rule
  ([Depth only](#depth-only-a-depth-region-on-the-plain-host)).
- `kind = "group"` (experimental): several channels, the basins and junctions between them and their islands in one
  envelope of host cells (a strait system: parallel channels round islands, passages the host closes).

## Run one

```
python -m meshom.insert --run RUN_DIR --config FILE.toml [--out DIR] [--mesh NAME] [--no-write]
python -m meshom.insert --run RUN_DIR --insert 'name=gib; strait=Gibraltar; cells=10'
```

Run it on a compute node. `--config` takes every `[[inserts]]` entry of a TOML file (a file of `[[inserts]]` alone,
or a run config); `--insert` one entry inline as `key=value; ...` with TOML values. `--mesh` picks another mesh of
the run folder (or a path to an `.npz`), so inserts chain: give the next one the previous `insert_<name>.npz`; its
saved depth (`fesom-insert_<name>/depth.npy`) is kept on every host node, so the first insert's own depth rule
survives (B217). The
general recipe with one example of each kind: `docs/examples/inserts.toml`; corridors and strips:
`docs/examples/inserts_gulf_stream.toml`.

```toml
[[inserts]]
name = "gibraltar"
strait = "Gibraltar"               # a name of the built-in list or of the run's [[straits]]
cells = 10                         # cells across the narrowest section (or h_fine_m)

[[inserts]]
name = "denmark"
path = [[-31.0, 64.5], [-22.0, 67.5]]   # or a centreline from one basin to the other
cells = 12

[[inserts]]
name = "portugal"
kind = "strip"
path = [[-8.95, 37.3], [-8.9, 37.95], [-9.0, 38.45], [-9.55, 38.75], [-9.45, 39.4], [-8.95, 40.2]]
width_km = 50.0                    # from the coast out to the host chain
h_fine_m = 2500.0                  # the cell size at the coast
h_min_m = 1000.0                   # the size floor (default: the host mesh's smallest cell)
```

## Keys

| key | default | meaning |
|---|---|---|
| `kind` | `"strait"` | `"strait"`, `"strip"`, `"corridor"`, `"depth"` (B223) |
| `strait` / `path` | | a listed strait, or the centreline (strait), coast line (strip), axis (corridor) |
| `cells` / `h_fine_m` | 10 cells | cells across the neck, or the fine size (cells = round(W / h_fine_m)) (B159) |
| `h_min_m` | the host's smallest cell | size floor: the insert never makes cells below it (B165) |
| `reach_km` | 12 host cells | growth limit per side (B159) |
| `mouth_a`, `mouth_b` | | join hints: a line from bank to bank on side A / B |
| `ljoins` | true | L-shaped joins round a cape when no straight join passes (B183) |
| `bays` | `"auto"` | a bay with a mouth of at least 3 insert cells, reaching inland at least its mouth's width, gets its own block; smaller ones are bridged; `true`: every bay with a 2-cell mouth; `false`: none (B166, B190) |
| `collar_rings` | 10 on meshom's meshes and triangle hosts, 0 on ORCA | host cells within this many element rings of the joins are cleaned with the insert's edge rows; 0 keeps every host node and cell (B192, B193) |
| `coast`, `coast_source`, `bank_sigma_cells` | `"real"`, the run's coast, 1.5 | the banks follow the real coastline (never the host's staircase), smoothed over 1.5 fine cells (B164); a bank whose ends lie nearest to different coast pieces takes the piece that fits both ends and the host's bank; a bank still on the host's coast is listed in `bank_fallback` (B219) |
| `width_km`, `band_km` | | strip: coast to the host chain; corridor: the full width; the fine band's width; depth: the region's full width round the path |
| `polygon` | | depth: the region `[[lon, lat], ...]` (great-circle edges) instead of `width_km` round the path (B223) |
| `rows` | `"harmonic"` | strip rows from a harmonic map of the band: they cannot cross (B195-B197); `"offset"` can fold and is then invalid |
| `coast_rounds`, `coast_shift_cells` | 3, 1 | harmonic strips: extra coast smoothing round quads above 125°, moving the coast at most 1 fine cell (B198) |
| `cells_a`, `cells_b` | any | strait: the join of side A / B has exactly this many host edges, the counts run monotonically from the neck to it (B200) |
| `growth_a`, `growth_b` | 0.12 | strait: the size slope along the block on side A / B; the room and funnel rules scale with it (B201) |
| `bank_land_cells` | 0 | strait: the real banks may move up to this many local cells into land (smoother, ends squared at acute joins) (B202) |
| `fill`, `fill_rows` | none, by depth | strait: lines `[[[lon, lat], ...], ...]` that cut water off a bank: the bank follows a line where it lies at least half a local cell off the coast, the water behind it becomes a strip of its own, `fill_rows` rows (default its depth / the local cell, at least 1) (B204) |
| `corner_steps` | false | strait: a join from a hint cuts every convex step of its host staircase by a corner triangle, not only its bends (B206) |
| `corner_split` | false | strait: each corner triangle of a join is split at the midpoint of its diagonal, the midpoint a node of the block's first row: two triangles of ~0.8 x the host cell instead of one of 1.14 x and a diagonal edge of 1.41 host edges (B208) |
| `join_margin` | 1.0 | strait: the block's size target at the joins is capped at this many host cell sizes (0.85: the outermost rows a little finer than the host) (B209) |
| `add_columns` | false | strait: where a block cell is above 1.1 x the host cell the size target is lowered (more columns and rows); on a side with `cells_a` / `cells_b` the counts may then rise above the neck's and the join's and come back to the join's count before the join (B207); it helps where the block fans out at a bank's turn, not at a staircase join's diagonal steps |
| `fix_rings` | 0 | after the splice, meshom's mesh fixer stage a on the insert's cells and this many element rings round them (the host's coast and every other node held, the banks sliding); host nodes in the rings move less than a host cell and keep their host depth (B210); its result is kept only when it is not worse by the insert's objectives (validity and floor, the angle excess above 120° summed over the quads, the worst corner, then the sizes), else nothing moves (`fix_rings.kept`, B220) |
| `depth_at`, `depth_smooth`, `strait_floor` | the run's | the insert's own nodes: `"node"` (box mean per node) or `"cells_deepest"`; smoothing passes among the insert's nodes only; whether the run's strait depth floors carve them (B211) |
| `sill_level` | `""` | `"shallower"`: the insert's sill gets the shallower of the two model levels round its depth (B212); `"grid"`: the grid's own saddle between the joins is held at the deepest level not below its depth by a dam of the insert's nodes (B215); `"nearest"`: the same dam at the level nearest the saddle's depth (B249) |
| `depth_channel` | `""` | the grid's deepest channel between the joins (box means at the fine size; its ends the deepest box means along the joins, or the deeper basin beyond a join too shallow for the strait, kept inside the insert, B218): `"cells"`: the insert's nodes of every cell it crosses take at least its box-mean depth there; `"near"`: the insert's nodes within one spacing of it take at least the box mean of their own box centred on it; both only on the strait's narrows (B213) |
| `depth_quantile` | 0 | the insert's nodes take this quantile of the grid's ocean cells in their node box instead of the mean (B214) |
| `depth_area` | false | after the other depth rules, the insert's nodes are scaled so that each section across the strait has the grid's wet area in FESOM3's element bottoms (the dam's cap kept, the strait's deepest connection within a level of the grid saddle's; B250, B251) |
| `stages` | `"d,b,e,a"` | the clean-up stages on the join ring ([`cleanup.md`](cleanup.md); B162) |
| `envelope` | | group: `[[lon, lat], ...]`, the host cells whose centroid lies inside are replaced as a whole (B240) |
| `channels` | | group: `[[inserts.channels]]` tables `name`, `a`, `b` (cross lines bank to bank), `cells` (default 5, held from `a` to `b`) (B242) |
| `gap_km`, `island_km2` | 5 x `h_min_m`, 10 | group: an island closer than `gap_km` to other land joins it; one smaller than `island_km2` is dropped (B241) |
| `gap_cells`, `growth` | 3, 0.12 | group: the least fill cells across any passage; the size slope of the fill (B243) |
| `bank_smooth_cells` | 0 | group: each block bank smoothed over this many block cells, ends held: capes cut (B245) |
| `routes` | | group: `[[inserts.routes]]` tables `name`, `a`, `b` (points in the basins at the strait's ends), optional `via`, `corridor_km`: the channel raise, the dam (across islands, every passage) and the section areas act along each route's line over blocks and patches (B246) |
| `blend_cells` | 2 | group: the real coast blends onto a join's end node over this many join cells; 1 keeps the real coast on short coast runs (B247) |
| `split_deg` | | group: after fix_rings, patch quads with a corner above this become two triangles (B243) |
| `growth_max` | 0.5 | group: the steepest slope with which the fill's size reaches the host's at the joins; = `growth`: one slope (B252) |
| `even_sweeps` | 40 | group: sweeps of the pass that evens the neighbour areas of the patches, blocks and two host rings; 0: off (B252) |
| `even_blocks` | `""` | group: the blocks' nodes in that pass: `""` move, `"cross"` the cross lines held, `"block"` every block node held (B252) |
| `force` | false | build with the least-violating joins when none passes |

## Rules and refusals

A join must meet each bank at 70-110° (corner), the width must not open faster than the rows can grow (funnel), and
the size must be able to grow from the fine size back to the host's by at most 15 % per cell before the join (room).
A fine size not finer than the host's by the factor 1.15 is refused as "no zoom" (B158). A refused entry says
which rule fails and what passes, for example:

- `no zoom: the fine size ... is not finer than the host's ...; fix: more cells across (cells >= N)`;
- `size floor: N cell(s) the insert made are below h_min_m ...; fix: h_min_m <= M m or cells <= K`;
- `side A: none of N candidate joins ... passes the rules`, then the failing rule and an `advice:` line with the
  cell counts both sides accept (B184), or join hints, a larger `reach_km`, a smaller `h_min_m`.

Apply the fix it names and run again. Do not raise `force` to hide a refusal.

## Outputs and judging them

Per entry in `--out` (default the run folder): `insert_<name>.npz` (the whole mesh with the insert),
`insert_<name>.json` (status, the reason and fix of a refusal, quality of the insert and of the cells round it, the
coast shift against the real coast), `insert_<name>.score.json`, `insert_<name>.map.png` (the host, the result, zooms),
`insert_<name>.profile.png`, `insert_<name>.collar.png`, and `fesom-insert_<name>/` (model files of the whole mesh,
depth by the run's rules) when the run wrote model files.

- Exit code 0 when every entry was built, 2 when one was refused (also when a rule fails after the joins were
  chosen, forced ones included: B254).
- `status`: `ok` (valid), `refused` (a rule failed: no mesh), `invalid` (built but failed the validity gate, e.g.
  overlapping cells or a flipped edge: no model files; B186).
- Open the map figure. Check the insert's quad angles and size ratios in the record and the cells of the collar
  (B194).

## Recipe: a strait insert on a structured host (ORCA)

On an ORCA host every host node and cell stays where it is (`collar_rings` 0, B193), so the insert meets the host's
grid lines as they are. Choose in this order; the worked examples are the `gibraltar` and `dover` entries of
`docs/examples/inserts.toml` (ORCA025 host, floor 1.98 km).

1. **Joins.** Join each side along host grid lines where the strait has opened back to the host's size. Where the
   strait opens slowly, a straight host-line join passes the rules: Gibraltar's Mediterranean side (a grid line 147 km
   out, 8 edges), Dover's Channel (a grid line 251 km out, 6 edges) and North Sea (11 edges). Where it opens into a
   wide gulf no straight line passes the funnel rule: draw an arc hint `mouth_a` from coast to coast across the gulf
   (Gibraltar's Atlantic side, across the Gulf of Cadiz: the join follows the host's staircase, 21 edges 203 km out)
   and set `force = true` (the arc still fails the funnel rule's straight-line measure).
2. **Neck count and floor.** `cells` across the narrowest section (5 in both examples). The floor `h_min_m` (default:
   the host's smallest cell) must stay below the neck width / `cells`; a refusal names the count that passes.
3. **Per-side counts and slopes.** `cells_a` / `cells_b` = the join's host edge count, so the join meets the host one
   to one; a `growth_a` / `growth_b` below 0.12 on a side with room gives a gentler size change (Gibraltar's
   Mediterranean side: 0.10).
4. **Banks.** Keep the banks on the real coast (`bank_land_cells = 0`, the default) where the strait's sections
   matter: banks moved into land (B202) are smoother but widen the strait (Gibraltar at 5.75 W: 41 km wide with 1
   cell, 35 km with 0.5, 29 km on the real coast; GEBCO 2024: 30 km). `fill` lines cut wide bays off a bank, so the
   block's rows are not stretched over them (Dover: the Bay of Seine and the Thames, with banks 1 cell into land).
5. **A staircase join.** `corner_steps = true` cuts every step of the hinted staircase by a corner triangle,
   `corner_split = true` splits each at its diagonal's midpoint, `join_margin = 0.85` keeps the outermost rows a little
   finer than the host, so no new cell exceeds the host cell next to it (B206, B208, B209).
6. **`add_columns = true`** where the block fans out at a bank's turn (Dover at the Somme, Gibraltar's Atlantic end):
   more columns and rows there, back to the join's count at the join (B207).
7. **`fix_rings = 10`**: stage a on the insert and ten host rings round it after the splice (B210). Gibraltar: worst
   quad 136 -> 106 deg, none above 110 deg.
8. **Depth.** For a strait the insert resolves, give its nodes their own rule: `depth_at = "node"`,
   `depth_smooth = 0`, `strait_floor = false` (B211). The run's rules (the deepest element per node, the smoothing, the
   strait list's depth floor along the path) dig the sills: Gibraltar's deepest connection between the basins is 340 m
   in FESOM3's levels with the run's rules and 280 m with the insert's (GEBCO 2024: 300 m at Camarinal).
   `sill_level = "shallower"` (B212) rounds the insert's own sill to the shallower level; use it only when the level
   bottoms leave the sill a level too deep (at Gibraltar they do not: 280 m with and without it).
9. **The deep channel.** Where the strait's deep channel is about as wide as the insert's cells, the box mean per node
   puts its flank nodes on the shelf and the model's element bottoms (the level nearest the mean of the element's nodes)
   come out levels too shallow. `depth_channel = "near"` moves the box of every insert node within one cell of the
   grid's deepest channel onto the channel, on the strait's narrows only (where the water across the channel is at most
   6 times as wide as at its narrowest; never in the open basins, B213), and `sill_level = "grid"` then holds the grid's
   own saddle between the joins at its level with a dam of nodes across the strait (B215); use the two together, since
   deepening the channel alone also deepens the sill. Gibraltar (FESOM3 element bottoms, GEBCO 2024 in brackets): 5.60 W
   410 -> 580 m (682), 5.45 W 790 -> 910 m (976), 6.10-5.85 W 280 -> 340 m (400-600), the deepest connection 280 m
   before and after (300 m at Camarinal); the section areas grow (5.75 W +22 -> +34 %, 5.60 W -14 -> +12 % against
   GEBCO). `depth_channel = "cells"` raises every node of the cells the channel crosses: deeper at the narrows (5.60 W
   680 m) with larger sections (+48 % at 5.75 W, +34 % at 5.60 W). `depth_quantile` (a quantile of the node box, B214)
   deepens the shelves as much as the channel; it is not part of the recipe.
10. **The section areas.** A climate model needs each strait's cross-section about the real one's; the box means on
   cells about as wide as the strait, banks moved into land (step 4) and the channel raise leave the sections larger
   or smaller than the bathymetry's. `depth_area = true` scales the insert's own nodes after steps 8-9 so that every
   section across the strait (every half fine cell along its centreline, on the strait's narrows, not in the open
   basins) has the grid's raw wet area in FESOM3's element bottoms (fesom_ini's levels); the dam of
   `sill_level = "grid"` keeps its cap and the channel line names the grid's saddle, so use it with steps 8-9: the
   dam's nodes and the channel's cells are scaled as the others, but the strait's deepest connection in the element
   bottoms stays within one level of the level nearest the saddle (at most one level shallower, never deeper than one
   level below it; B251); a strait whose sill is already more than a level shallow keeps it: nothing on its channel
   is made shallower (Bab el Mandeb, whose Hanish sill the insert holds at 100 m against GEBCO's 143 m). It changes depth only: the mesh is the same. It cannot make a section smaller than the depth floor
   allows (FESOM3's minimum, 30 m with 4 CORE2 layers, times the mesh's width), nor than the channel's cells hold: a
   shallow strait whose banks moved into land (Dover) and a strait with 3-4 cells across and its capes cut (Bab el
   Mandeb: the channel's cells fill most of the section) stay larger than the bathymetry's (B250). Gibraltar: the
   sections on its narrows off by 13 % on average -> 6.5 %, the section at 5.53 W 1.13 -> 0.95 x GEBCO, the deepest
   connection 280 m before and after.

Gibraltar with this recipe (the `gibraltar` entry): neck 14.7 km, smallest cell 2.51 km, worst quad 106 deg, 0 above
110 deg, the insert's cells at most 0.93 x the host cell next to them; at 5.75 W 29 km wide (GEBCO 30 km), 5.0 km^2 in
FESOM3's element bottoms (GEBCO 3.7 km^2), the deepest element bottom there 280 m; the deepest connection between the
basins 280 m (GEBCO 300 m), the element bottom 580 m at 5.60 W and 910 m at 5.45 W (GEBCO 682 and 976 m). Dover (the
`dover` entry, without steps 7-9): worst quad 120 deg, none above 120 deg (B207).

**Judging a variant** (record `insert_<name>.json`, then the model files):

- angles: the worst quad corner and the quads above 110 / 120 deg of the insert and its rings (`fix_rings.after`:
  `qmax`, `q110`, `q120`; without `fix_rings`: `collar.after`); `fix_rings.kept` = `"before"` means the smoothing
  made the insert worse and was not used (`keep_rule` says why, B220);
- banks: `banks` and `bank_fallback` (B219): a bank on the host's staircase where the real coast was asked is a
  geometry defect to fix (join hints, or another `coast_source`);
- sizes: the smallest new cell (`fix_rings.floor.smallest_m`, never below `h_min_m`); the largest new cell against the
  host cell next to it, by mean great-circle edge length, new cells = `meshom.insert.measure.changed_cells` (aim: at
  most 1.0 x for the insert's own cells); the neighbour size ratio (`ratio_p99`, `ratio_max`);
- depth: sections of the written depth against the bathymetry grid: the crest depths of the sills, the level bottom
  the model uses (after `fesom_ini`: an element's bottom is the shallowest of its nodes' levels), the deepest
  connection between the two basins, the wet width and the area of a section across the strait; in the record,
  `depth.channel`: where the line's narrows lie (they must be the strait's), `end_limited` (true: the deepest
  connection is only a lower bound, the basin beyond that join is no deeper in the grid's window) and
  `depth.sill` (a `skipped` dam means the grid's saddle was not held). Compare variants on the same sections
  (the same centre, direction, length and integration limits for every variant, the whole wet crossing).

**Known limits:**

- the size step across the join next to split corner triangles is ~1.23 (their mean edge is ~0.8 x the host's), up to
  1.45 at the end of a staircase join;
- the host cells of the rings that `fix_rings` moves grow up to 1.06 x their size (1.10 with the banks on the real
  coast at Gibraltar);
- banks moved into land (`bank_land_cells`) widen the strait and its sections;
- a sill narrower than the insert's cells comes out deeper in the box mean than the grid's crest (Camarinal: 335-390 m
  by linear interpolation at the crest against GEBCO's 300 m); the level bottoms hold the passage at 280 m;
- a sill holds its level in FESOM3's element bottoms only over two cells along the channel: the dam of
  `sill_level = "grid"` caps the cells within one cell of it (Camarinal, cells of 4-5 km: 230-340 m from 5.80 to
  5.70 W, GEBCO 520-540 m); only the grid's deepest connection between the joins is held, not the other saddles;
- a node's depth is shared by every cell round it, so a deepened channel one or two cells wide deepens its flank cells
  too: the section areas grow (B213).

## Depth only: a depth region on the plain host

Where the host already has enough cells across a strait but its depth is wrong there (sills a level too deep, section
areas too large: an ORCA host's nodes take the deepest of their four NEMO cells' levels), `kind = "depth"` changes the
depth only (B223): the host's nodes in a region round the strait take the grid rule of an insert's own nodes, as the
nest's depth does inside its box (B216): the box mean per node, no smoothing, no strait depth floor, the run's slope
limiter, the depth floor and the run's regions; with `depth_channel` the channel raise on the narrows (B213) and with
`sill_level` the sill (B215, B212), and with `depth_area` the section areas (B250). Every other node keeps the host's
depth (the run's rules, or on another insert's result its saved depth, B217). The mesh is not changed.

```toml
[[inserts]]
name = "lancaster_depth"
kind = "depth"
strait = "Lancaster_Barrow"        # or path = [[lon, lat], ...]
width_km = 160.0                   # the region: the nodes within width_km / 2 of the path

[[inserts]]
name = "nares_depth"
kind = "depth"
strait = "Nares"
polygon = [[-77.4, 79.05], [-73.6, 80.1], [-68.0, 82.4], [-57.0, 81.35], [-62.5, 80.2], [-63.5, 79.3], [-69.3, 77.85]]
depth_channel = "near"             # optional: the channel raise on the narrows
```

- **The region.** `width_km` round the path (ends included: round caps), or `polygon` (great-circle edges). The
  channel line of `depth_channel` / `sill_level` runs between the two largest pieces of host nodes just outside the
  region on its cells, so the region must cover the strait from coast to coast and leave it across the channel at
  both ends: a polygon whose long sides lie on land and whose two short sides cross the channel. A wide buffer's caps
  reach other water (Nares with 220 km: the north cap ended in a Greenland fjord, 827 m, whose 15 km mouth became the
  narrows); check `depth.channel.ends` and `depth.channel.narrows` in the record. The box mean alone (no
  `depth_channel`, no `sill_level`) does not depend on the region's shape.
- **Open water.** A channel between submerged banks (no coast on two sides of the region: the Faroe Bank Channel,
  the Wyville Thomson Ridge) needs a path from basin A to basin B through the region (`path` or `strait`; a
  `polygon` alone has none): the region's outer nodes split at their shallowest point on each side into the two
  joins (round the deepest outer node near each end of the path), the rest are the dam's banks, left and right of
  the path; the raise acts along the line inside the region, and the area rule's sections end at the region's edge
  (B253). The region's sides should lie on the banks (shallower than the sill), its ends in the basins; a region
  along a ridge (a band, the path across it) holds the whole crest from bank to bank, so the dam caps the crest
  where the host crosses it elsewhere. Check `depth.open_water` (ends, col, join and bank nodes) in the record.
- **The sill.** `sill_level = "grid"` holds the grid's saddle at the deepest level not below its depth: at Nares
  (saddle 224 m, levels 190 and 230 m) that is 190 m, a level shallower than the nearest level (230 m) the box mean
  gives without it.
- **Section areas.** `depth_area = true` scales the region's nodes as an insert's own (B250), on sections across the
  entry's path (`strait` or `path`; a polygon alone has no axis: the rule is skipped and recorded). A section counts
  only when at least 90 % of the mesh's run across it lies in the region's cells, so the region must cover the whole
  crossing the mesh has there (Tsushima's east channel on ORCA025 v5: with 100 km round the path 2 of 21 sections
  counted; with 250 km, Korea to Kyushu, 15). Without the dam and the channel line the scaling may make the strait's
  deepest connection shallower (Tsushima: 135 -> 90 m, two levels): use it with `sill_level = "grid"` and
  `depth_channel = "near"`, and check `depth.channel.ends` (the line takes the deepest route of the whole region,
  which may be a parallel channel).
- **Outputs.** `insert_<name>.npz` (the host mesh), `.host_to_new.npy` (every node kept), `.collar_moved.npy` (none),
  `insert_<name>.json` (`region`: the nodes and the width or the polygon; `depth`: the rule's record) and
  `fesom-insert_<name>/depth.npy`, with model files when the run wrote them.

Nares and Lancaster Sound on ORCA025 v5 (its ORCA12 part, 6 and 10 cells across the narrowest sections; areas in
FESOM3's element bottoms against GEBCO 2024 in FESOM3's levels on the same sections): Nares (the box mean, a 220 km
buffer or a polygon like the one above: its two short sides cross Smith Sound and Robeson Channel) at Kennedy Channel 10.91 -> 7.79 km^2 (GEBCO 7.08), at the Kane Basin sill section
29.56 -> 24.76 km^2 (25.52), the deepest connection 230 m before and after (GEBCO 225 m, level 230 m); Lancaster
(the box mean, 160 km round the list's path) at the narrowest section 4.28 -> 3.27 km^2 (2.97), at the sill section
41.84 -> 39.14 km^2 (35.59), the deepest connection 135 -> 115 m (GEBCO 120 m, level 115 m).

## Config keys used

- `[[inserts]]`: `name`, `kind`, `strait`, `path`, `cells`, `h_fine_m`, `h_min_m`, `reach_km`, `mouth_a`, `mouth_b`, `ljoins`, `bays`, `collar_rings`, `coast`, `coast_source`, `bank_sigma_cells`, `width_km`, `band_km`, `rows`, `coast_rounds`, `coast_shift_cells`, `cells_a`, `cells_b`, `growth_a`, `growth_b`, `bank_land_cells`, `fill`, `fill_rows`, `corner_steps`, `corner_split`, `join_margin`, `add_columns`, `fix_rings`, `depth_at`, `depth_smooth`, `strait_floor`, `sill_level`, `depth_channel`, `depth_quantile`, `depth_area`, `stages`, `force`
- `[[inserts]]`: `name`, `kind`, `strait`, `path`, `cells`, `h_fine_m`, `h_min_m`, `reach_km`, `mouth_a`, `mouth_b`, `ljoins`, `bays`, `collar_rings`, `coast`, `coast_source`, `bank_sigma_cells`, `width_km`, `band_km`, `rows`, `coast_rounds`, `coast_shift_cells`, `cells_a`, `cells_b`, `growth_a`, `growth_b`, `bank_land_cells`, `fill`, `fill_rows`, `corner_steps`, `corner_split`, `join_margin`, `add_columns`, `fix_rings`, `depth_at`, `depth_smooth`, `strait_floor`, `sill_level`, `depth_channel`, `depth_quantile`, `polygon`, `stages`, `force`
## Group inserts (`kind = "group"`, experimental)

A group replaces every host cell whose centroid lies inside `envelope` by a new mesh of the real water there (B240):
the joins are the host edges where the envelope's cells meet the host cells outside it (as many as there are, node
for node); the host's coast between two joins is replaced by the real coast; the real islands inside are kept as
holes, joined to nearby land (`gap_km`) or dropped (`island_km2`) (B241). Each `[[inserts.channels]]` entry is a
structured quad block between its cross lines `a` and `b`, `cells` across from end to end, no triangle inside (B242);
the rest of the water (basins, junctions, the passages round islands) is a quad-dominant fill whose size grows from
the cross lines and the narrow gaps (at least `gap_cells` cells across) to the host's at the joins (B243), each fine
sample's slope steep enough to reach the host's size at every join node (at most `growth_max`, B252). After
fix_rings a pass moves the nodes of the patches, the blocks and two host rings round them to even the neighbour areas
(a triangle counted as half a quad), no corner above 120 deg, the coast held (B252). With `sill_level = "grid"`
each channel holds its own saddle (B244).

```toml
[[inserts]]
name = "lombok_alas"
kind = "group"
envelope = [[114.63, -7.38], [117.87, -7.38], [117.87, -9.87], [114.63, -9.87]]
island_km2 = 5.0
bank_sigma_cells = 0.5
bank_smooth_cells = 1.0
fix_rings = 3
depth_at = "node"
depth_smooth = 0
strait_floor = false
sill_level = "grid"

[[inserts.channels]]
name = "lombok_n"                  # Bali - Lombok
a = [[115.66, -8.36], [116.10, -8.36]]
b = [[115.55, -8.52], [116.10, -8.56]]
cells = 4

[[inserts.channels]]
name = "lombok_s"                  # Nusa Penida - Lombok
a = [[115.60, -8.68], [115.92, -8.72]]
b = [[115.62, -8.84], [115.85, -8.86]]
cells = 4

[[inserts.channels]]
name = "alas"                      # Lombok - Sumbawa (closed in the host)
a = [[116.68, -8.42], [116.80, -8.42]]
b = [[116.64, -8.62], [116.76, -8.62]]
cells = 3
```

Choose in this order:

1. **Envelope.** Wide enough that the size can grow from the finest channel back to the host's before the joins
   (about (host cell - channel cell) / 0.12 on every side) and with its edges in open water: a join across real land
   or an island crossing the edge is refused ("an island of the real coast crosses the group's boundary"); move the
   edge off it.
2. **Channels.** A cross line at each end of a channel, both ends on the same two pieces of land (a channel whose
   bank changes island, as Lombok Strait at Nusa Penida, is two channels meeting in a junction patch). `cells` is
   the count at the narrowest; 3 is enough where 5 would force cells near the floor. Put the controlling sill inside
   the block, not in a patch, so its dam holds.
3. **Islands.** `gap_km` closes passages narrower than about 5 floor cells; lower it to keep an island inside a
   passage open; `island_km2` drops the small ones. The two banks of a channel are never joined, however narrow
   the channel (B255): a channel keeps a passage open that `gap_km` would close (the Oresund between Zealand and
   Sweden, 4 km), while the other islands still join the land. An island's narrow inlets (fjords) are bridged by
   the bay search as on the mainland coast, and no pocket chord crosses another coast piece (an island, a
   channel's other bank; B256). To drop the islands a host does not have, set `island_km2` above their area (the
   Danish straits on ORCA025 v5: 125 km^2 keeps only the islands that join the land or Zealand and Funen).
4. **Capes.** A block wrapping round a thin cape folds; `bank_smooth_cells = 1-2` cuts the cape (the block covers
   it). `bank_sigma_cells = 0.5` keeps islands near a coast from overlapping it after the smoothing.
5. **`fix_rings = 3`** relaxes the angles of the patches and of the host rings round them (B210, B220);
   `split_deg = 120` then turns the patch quads still above 120 deg into triangle pairs (B243); the area pass
   (`even_sweeps`, on by default) runs before the split and again after it (B252).
6. **Routes.** One `[[inserts.routes]]` per strait through the group (`a`, `b` in its two basins, in or next to the
   envelope's cells; `corridor_km` for a strait beside a deeper one, as Alas beside Lombok) with `depth_channel =
   "near"`, `sill_level = "grid"` or `"nearest"` and `depth_area = true`: the raise, the dam and the section areas
   then act along the strait's own line across blocks and patches (B246, B249). `blend_cells = 1` keeps the real coast
   where the host's coast at a join's end is far off it (B247).

Judge it as a strait (above), per channel: cells across at its narrowest and sill sections, the section areas against
the bathymetry in the model's levels, the sills (records `depth.branches`, `depth.routes`), and over the group: the worst quad and the
quads above 120 deg, the smallest cell, the largest cell against the host, the neighbour size ratio and the neighbour
area ratio (a triangle as half a quad; aim p99 <= 1.5). Known limits: an envelope that leaves less than about
(host cell - channel cell) / 0.2 between a channel or an island and a join cannot grade the size at an area ratio of
1.5 per cell (Ombai on ORCA025 v5: Atauro 25 km from the envelope's corner, area ratio p99 2.5; one host cell more
on four sides: 1.76); the node moves of the area pass shift the section areas by up to about 0.1-0.15 of GEBCO's at
the dam-held narrows (the area rule acts after them but holds the dam); without routes a sill in a patch gets no dam; on three or four cells across, FESOM3's element bottom (the shallowest
of its nodes' levels) leaves a channel one or two levels shallower than its dam.

## Config keys used

- `[[inserts]]` (group): `envelope`, `channels`, `gap_km`, `island_km2`, `gap_cells`, `growth`, `bank_smooth_cells`, `routes`, `blend_cells`, `split_deg`, `growth_max`, `even_sweeps`, `even_blocks`
- `[[inserts]]`: `name`, `kind`, `strait`, `path`, `cells`, `h_fine_m`, `h_min_m`, `reach_km`, `mouth_a`, `mouth_b`, `ljoins`, `bays`, `collar_rings`, `coast`, `coast_source`, `bank_sigma_cells`, `width_km`, `band_km`, `rows`, `coast_rounds`, `coast_shift_cells`, `cells_a`, `cells_b`, `growth_a`, `growth_b`, `bank_land_cells`, `fill`, `fill_rows`, `corner_steps`, `corner_split`, `join_margin`, `add_columns`, `fix_rings`, `depth_at`, `depth_smooth`, `strait_floor`, `sill_level`, `depth_channel`, `depth_quantile`, `stages`, `force`

## Sources

- Code: `meshom/insert/` (`__main__.py` the command, `core.py`, `group.py` the group inserts, `region.py` the rules, `block.py`, `band.py`,
  `harmonic.py`, `realcoast.py`, `network.py` the bays, `assemble.py` the collar, `measure.py`, `channel.py` the
  channel depth rules, `area.py` the section areas, `even.py` the area pass), `meshom/io/fesom.py` (`model_levels`: FESOM3's levels),
  `meshom/config.py` (`InsertCfg`).
- Docs: [`docs/config.md` "`[[inserts]]`"](../config.md#inserts-python--m-meshominsert), `docs/examples/inserts.toml`,
  `docs/examples/inserts_gulf_stream.toml`.
- Decisions: B158-B169, B180-B186, B190, B192-B220, B250.
- Decisions: B158-B169, B180-B186, B190, B192-B223.
- Decisions: B158-B169, B180-B186, B190, B192-B212, B240-B249, B252.
- Decisions: B254 (a refusal after the joins is a record, exit code 2), B255, B256 (group inserts).
