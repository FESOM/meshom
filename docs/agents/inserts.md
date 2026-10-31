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

## Run one

```
python -m meshom.insert --run RUN_DIR --config FILE.toml [--out DIR] [--mesh NAME] [--no-write]
python -m meshom.insert --run RUN_DIR --insert 'name=gib; strait=Gibraltar; cells=10'
```

Run it on a compute node. `--config` takes every `[[inserts]]` entry of a TOML file (a file of `[[inserts]]` alone,
or a run config); `--insert` one entry inline as `key=value; ...` with TOML values. `--mesh` picks another mesh of
the run folder (or a path to an `.npz`), so inserts chain: give the next one the previous `insert_<name>.npz`. The
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
| `kind` | `"strait"` | `"strait"`, `"strip"`, `"corridor"` |
| `strait` / `path` | | a listed strait, or the centreline (strait), coast line (strip), axis (corridor) |
| `cells` / `h_fine_m` | 10 cells | cells across the neck, or the fine size (cells = round(W / h_fine_m)) (B159) |
| `h_min_m` | the host's smallest cell | size floor: the insert never makes cells below it (B165) |
| `reach_km` | 12 host cells | growth limit per side (B159) |
| `mouth_a`, `mouth_b` | | join hints: a line from bank to bank on side A / B |
| `ljoins` | true | L-shaped joins round a cape when no straight join passes (B183) |
| `bays` | `"auto"` | a bay with a mouth of at least 3 insert cells, reaching inland at least its mouth's width, gets its own block; smaller ones are bridged; `true`: every bay with a 2-cell mouth; `false`: none (B166, B190) |
| `collar_rings` | 10 on meshom's meshes and triangle hosts, 0 on ORCA | host cells within this many element rings of the joins are cleaned with the insert's edge rows; 0 keeps every host node and cell (B192, B193) |
| `coast`, `coast_source`, `bank_sigma_cells` | `"real"`, the run's coast, 1.5 | the banks follow the real coastline (never the host's staircase), smoothed over 1.5 fine cells (B164) |
| `width_km`, `band_km` | | strip: coast to the host chain; corridor: the full width; the fine band's width |
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
| `fix_rings` | 0 | after the splice, meshom's mesh fixer stage a on the insert's cells and this many element rings round them (the host's coast and every other node held, the banks sliding); host nodes in the rings move less than a host cell and keep their host depth (B210) |
| `depth_at`, `depth_smooth`, `strait_floor` | the run's | the insert's own nodes: `"node"` (box mean per node) or `"cells_deepest"`; smoothing passes among the insert's nodes only; whether the run's strait depth floors carve them (B211) |
| `sill_level` | `""` | `"shallower"`: the insert's sill gets the shallower of the two model levels round its depth (B212); `"grid"`: the grid's own saddle between the joins is held at the deepest level not below its depth by a dam of the insert's nodes (B215) |
| `depth_channel` | `""` | the grid's deepest channel between the joins (box means at the fine size): `"cells"`: the insert's nodes of every cell it crosses take at least its box-mean depth there; `"near"`: the insert's nodes within one spacing of it take at least the box mean of their own box centred on it (B213) |
| `depth_quantile` | 0 | the insert's nodes take this quantile of the grid's ocean cells in their node box instead of the mean (B214) |
| `stages` | `"d,b,e,a"` | the clean-up stages on the join ring ([`cleanup.md`](cleanup.md); B162) |
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

- Exit code 0 when every entry was built, 2 when one was refused.
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
9. **The deep channel.** Where the strait's deep channel is about as wide as the insert's cells, the box mean per
   node puts its flank nodes on the shelf and the model's element bottoms (the level nearest the mean of the
   element's nodes) come out levels too shallow. `depth_channel = "near"` moves the box of every insert node within
   one cell of the grid's deepest channel onto the channel (B213), and `sill_level = "grid"` then holds the grid's
   own saddle between the joins at its level with a dam of nodes across the strait (B215); use the two together,
   since deepening the channel alone also deepens the sill. Gibraltar (FESOM3 element bottoms, GEBCO 2024 in
   brackets): 5.60 W 410 -> 580 m (682), 5.45 W 790 -> 910 m (976), 6.30-5.85 W 280 -> 340 m (400-600), the deepest
   connection 280 m before and after (300 m at Camarinal); the section areas grow (5.75 W +22 -> +34 %, 5.60 W
   -14 -> +12 % against GEBCO). `depth_channel = "cells"` raises every node of the cells the channel crosses: deeper
   at the narrows (5.60 W 680 m) with larger sections (+48 % at 5.75 W, +34 % at 5.60 W). `depth_quantile` (a
   quantile of the node box, B214) deepens the shelves as much as the channel; it is not part of the recipe.

Gibraltar with this recipe (the `gibraltar` entry): neck 14.7 km, smallest cell 2.51 km, worst quad 106 deg, 0 above
110 deg, the insert's cells at most 0.93 x the host cell next to them; at 5.75 W 29 km wide (GEBCO 30 km), 5.0 km^2 in
FESOM3's element bottoms (GEBCO 3.7 km^2), the deepest element bottom there 280 m; the deepest connection between the
basins 280 m (GEBCO 300 m), the element bottom 580 m at 5.60 W and 910 m at 5.45 W (GEBCO 682 and 976 m). Dover (the
`dover` entry, without steps 7-9): worst quad 120 deg, none above 120 deg (B207).

**Judging a variant** (record `insert_<name>.json`, then the model files):

- angles: the worst quad corner and the quads above 110 / 120 deg of the insert and its rings (`fix_rings.after`:
  `qmax`, `q110`, `q120`; without `fix_rings`: `collar.after`);
- sizes: the smallest new cell (`fix_rings.floor.smallest_m`, never below `h_min_m`); the largest new cell against the
  host cell next to it, by mean great-circle edge length, new cells = `meshom.insert.measure.changed_cells` (aim: at
  most 1.0 x for the insert's own cells); the neighbour size ratio (`ratio_p99`, `ratio_max`);
- depth: sections of the written depth against the bathymetry grid: the crest depths of the sills, the level bottom
  the model uses (after `fesom_ini`: an element's bottom is the shallowest of its nodes' levels), the deepest
  connection between the two basins, the wet width and the area of a section across the strait.

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

## Config keys used

- `[[inserts]]`: `name`, `kind`, `strait`, `path`, `cells`, `h_fine_m`, `h_min_m`, `reach_km`, `mouth_a`, `mouth_b`, `ljoins`, `bays`, `collar_rings`, `coast`, `coast_source`, `bank_sigma_cells`, `width_km`, `band_km`, `rows`, `coast_rounds`, `coast_shift_cells`, `cells_a`, `cells_b`, `growth_a`, `growth_b`, `bank_land_cells`, `fill`, `fill_rows`, `corner_steps`, `corner_split`, `join_margin`, `add_columns`, `fix_rings`, `depth_at`, `depth_smooth`, `strait_floor`, `sill_level`, `depth_channel`, `depth_quantile`, `stages`, `force`

## Sources

- Code: `meshom/insert/` (`__main__.py` the command, `core.py`, `region.py` the rules, `block.py`, `band.py`,
  `harmonic.py`, `realcoast.py`, `network.py` the bays, `assemble.py` the collar, `measure.py`, `channel.py` the
  channel depth rules), `meshom/config.py` (`InsertCfg`).
- Docs: [`docs/config.md` "`[[inserts]]`"](../config.md#inserts-python--m-meshominsert), `docs/examples/inserts.toml`,
  `docs/examples/inserts_gulf_stream.toml`.
- Decisions: B158-B169, B180-B186, B190, B192-B212.
