---
name: meshom-orca
description: How to build NEMO's ORCA grids as quad meshes with m16_orca - whole ORCA2/1/05/025/12, ORCA up to a latitude or a line with a meshom mesh beyond, a finer ORCA around the fold or nested in a box, FESOM3's ORCA rules - the NEMO files they read and how to check the result. Read it when a request names ORCA or NEMO.
---

# ORCA grids (`m16_orca`)

## What it builds

`m16_orca` (`meshom/methods/m16_orca.py`, the package `meshom.orca`) builds NEMO's tripolar ORCA grids with NEMO's
own grid generator, ported from G. Madec's ORCA generator (NEMO; Madec and Imbard, 1996; see `NOTICE`), with
A. Sellar's tropical stretch. The presets (`meshom.orca.presets.PRESETS`): `ORCA2`, `ORCA1`, `ORCA05`, `ORCA025`,
`ORCA12`, `ORCA36`. ORCA1, ORCA05, ORCA025, ORCA12 and ORCA36 match NEMO's grids to below 1 m away from the poles;
ORCA2 to a few km, without NEMO's hand-made 1-degree Mediterranean patch (B78-B81, B231). ORCA36 (119 M cells) is
for a box inside a nest, not for a whole mesh.

The ORCA cells take the land-sea mask of NEMO's own file of that grid, cell for cell (B82), cleaned as FESOM3 needs
(B84); the run's land cut leaves them alone (B85). NEMO's closed seas (the Caspian, ORCA025's Great Lakes) are
removed, as every water body cut off from the ocean (B171, C48); `keep_seas = true` keeps them for a special case.

## Data, once

1. NEMO's reference grids: `scripts/grids/fetch_grids.sh [DATA_DIR]` downloads the public grids (ORCA2 domain_cfg,
   eORCA1 / eORCA025 / eORCA12 GO6 grid_mask) into `curvilinear_grids/` next to the repository folder. ORCA05 needs
   the DRAKKAR eORCA05.L121 mesh_mask by hand (the server is named in the docstring of
   `scripts/grids/orca_refs_npz.py`).
2. Convert them once to npz, on a compute node: `python scripts/grids/orca_refs_npz.py OUT_DIR` (and
   `... OUT_DIR eorca05=/path/to/eORCA05_mesh_mask.nc4`). Point `[mesh.opts] ref_dir` (or the environment variable
   `MESHOM_ORCA_REFS`) at `OUT_DIR`.
3. For FESOM3's rules (below) also NEMO's levels: `scripts/grids/orca_levels_npz.py LEVELS NAME=PATH ...` (the
   Met Office GO6 bathymetries, Zenodo records 22010172, 15494369, 15495870; NEMO's ORCA2 domain_cfg; MEOM's
   eORCA05; MEOM's eORCA36.L121 domain_cfg as `eorca36=PATH`) and `scripts/grids/nemo_inputs_npz.py NEMO_DIR
   LEVELS/nemo025q.npz` for FESOM3's own ORCA025 inputs.
   `[mesh.opts] levels_dir` (or `MESHOM_ORCA_LEVELS`) points at `LEVELS`.

## Build a whole grid

Copy `docs/examples/orca2_whole.toml`, `orca1_whole.toml` or `orca025_whole.toml`; set `[mesh.opts] orca`,
`ref_dir`, `gebco` (a GEBCO-like grid for the cells NEMO's mask does not cover: eORCA's ice-shelf rows) and the
`latitude` layer's `h_m` to the grid's zonal spacing (it is only the score's reference: the grid does not follow h):
222 390 m for ORCA2, 111 195 ORCA1, 55 597.5 ORCA05, 27 798.75 ORCA025, 9 266.25 ORCA12, 3 088.75 ORCA36.

Keep: `[coast] treatment = "cut"`, `min_area_km2 = 0`, `passage_min_cells = 0` (NEMO's straits are its own),
`[mesh] optimize` unset and `[run] smooth = false` (both would move ORCA nodes).

| grid | measured |
|---|---|
| ORCA2 whole | 16 236 quads, one water body (the Caspian removed; the Red Sea cut off and removed, as NEMO's hand patch is not built), 3 min |
| ORCA1 whole | 65 160 quads, quad angles ≤ 91.1°, one water body, half a minute |
| ORCA025 whole | 903 398 quads, one water body (10 closed seas removed), 3 min on 32 CPUs, 29 GB |

The size gate fails on the ORCA part by design (ORCA's cells are anisotropic: 1 x 1/3 degree at ORCA1's equator).
Report it; do not try to fix it.

## ORCA up to a latitude or a line, a meshom mesh beyond

- **Latitude range**: `[mesh.opts] lat_south` / `lat_north` (`docs/examples/orca1_40n_arctic10.toml`). The ORCA rows
  end at the F row nearest `lat_north`; the north cap (and a south cap if `lat_south` is north of the preset's first
  row) is meshed by `cap_method` (default `m14_blocks_rep`) against that row as one exact loop, so the joint is
  conforming. The caps are cut by the coast with meshom's rules; give their h in `[size]` as for any mesh (at the
  joint about the ORCA row's spacing). Measured: ORCA1 80° S - 40° N + 10 km Arctic, 208 215 quads, 247 triangles,
  one water body, 35/39 gates (the size gate on ORCA), 10 min on 32 CPUs.
- **A line** (`region`, B100): a lon-lat polygon holding the whole south (corners at lon ±181, lat −91); per ORCA
  column the cells up to the first T point outside it. Keep the line off NEMO's grid poles (ORCA025: near 73° E 50° N
  and 107° W 66° N). h: an `orca` layer (ORCA's own cell size, B101), then a zone of `mode = "blend"` with `polygon`
  = the fine region and `outer` = ORCA's edge (S37). One block over a uniform fine region: `[mesh.opts.cap_opts]`
  `place_region` = the zone's polygon, `max_blocks = 1`, `b_cap = 100` (B106). Examples:
  `docs/examples/orca025_arctic45.toml` (written by `scripts/orca025_arctic_lines.py`) and
  `docs/examples/orca025_arctic45_v3.toml` (latitude edges, `scripts/orca025_arctic_lat.py`): 1.5-1.65 M nodes,
  39/39 gates, about an hour on 128 CPUs, 74-82 GB.
- In a range mesh a strait across the joint follows the cap's cut; `passage_min_cells = 2` acts on the caps only
  (B95). Meshom's strait list widened in the caps (`straits = "fesom2_mesh_gen"`, `strait_profile = "widen"`)
  keeps the Baltic and the Black Sea connected.

## A finer ORCA inside a coarser one

- **Around the fold** (`fine_orca`, `fine_lat`, B108): a second part from a preset that refines `orca` k times on
  the same map (`ORCA12` = `ORCA025` x 3), from the fold down while its T points lie north of `fine_lat`; the band
  between the two parts is filled along ORCA's own directions (`cap_opts` `auto_blocks = false`,
  `strait_blocks = false`, `fill.frame = "ORCA025"`, `fill.seed_loops = [1]`; B109, B112) and h is a `blend` zone
  with `h_factor = 1/3` (B110). `docs/examples/orca025_arctic_v4.toml`.
- **The ladder** (`transition = "ladder"`, `ladder_lat`, B114-B116): a structured strip in ORCA's index space
  between a plain coarse row and a plain fine row, each coarse column split into three. `docs/examples/orca025_arctic_v5.toml`.
- **A nest in a box** (`[mesh.opts.nest]`, B172-B176): `fine_orca`, `box = [c0, c1, r0, r1]` (coarse F lines) or
  `lonlat = [lon0, lat0, lon1, lat1]`, `width` (default 39 fine steps): the fine preset inside the box, a graded
  frame of rings between the boxes, square corners. Keep the box below the fold and away from the grid poles.
  `docs/examples/orca025_gs_nest.toml` (ORCA12 in a Gulf Stream box, written by `scripts/orca025_gs_nest.py`).
- **A nest in the nest** (`[mesh.opts.nest.inner]`, B230): `fine_orca` = a preset refining the nest's by 3
  (`ORCA36` in `ORCA12`), `width` (default 39 of its steps): a second frame inside the nest's box and that preset's
  cells inside it, so the box goes from ORCA025 to ORCA36 in two frames and holds no ORCA12 cells. h: a second
  `blend` zone (`h_factor = 1/3`, `polygon` the inner box's edge, `outer` the nest's box's edge). For a strait this
  gives the cells NEMO's finer grid has there (Gibraltar: 6 at the narrowest section in ORCA36, 3 in ORCA12).
- **The nest's depth** (`[mesh.opts.nest] depth = "gebco"`, B216): the nodes inside the coarse box take the box
  mean per node (no smoothing, no strait floor) instead of FESOM3's ORCA depth, which digs a strait's sills too deep;
  for a strait add `depth_channel = "near"` and `sill_level = "grid"` (as the Gibraltar insert). Only the depth
  changes: write the saved mesh again with `meshom write` instead of a new run.

## FESOM3's ORCA rules

FESOM3 runs on NEMO's ORCA025 turned into a quad mesh by FESOM3's own conversion (FESOM3, github.com/FESOM/fesom3).
`meshom.orca.fesom3` follows its rules (B87-B95): water where NEMO's bottom level is > 0, one-cell land removed,
14 strait edits, node depth the deepest of the four cells in NEMO's levels, the fold's nodes merged, only the
ocean-connected water kept (B170).

- The whole grid in FESOM3's files: `python -m meshom.orca.fesom3 PRESET LEVELS_DIR OUT_DIR [--gebco GEBCO]
  [--fesom-ini BIN]` (ORCA025: 4 min, 16 CPUs). It refuses a mesh of more than one water component (C52).
- In a meshom run: `[mesh.opts] mask = "fesom3"`, `levels_dir`; the ORCA nodes then get FESOM3's depths in the
  run's bathymetry stage (B94). For the meshom part `[bathymetry] depth_at = "cells_deepest"` keeps narrow straits at
  their cells' depth (B103).

## Check the result

- `python scripts/orca_report.py RUN_DIR`: the ORCA part unchanged (max node displacement 0 m), quality per part, the
  joint.
- `python scripts/orca_mask_report.py RUN_DIR`: the mask against NEMO's (cells and km² that differ, named straits).
- `python scripts/orca_figures.py RUN_DIR OUT_DIR`: the whole mesh, the joint, the fold.
- `python scripts/orca_fesom3_compare.py OUT_DIR GRID_NPZ a=DIR b=DIR`: two FESOM3 ORCA meshes strait by strait.

## Config keys used

- `[mesh]`: `method`, `format`, `fesom3_ini`, `optimize`
- `[mesh.opts]` `m16_orca`: `orca`, `lat_south`, `lat_north`, `region`, `fine_orca`, `fine_lat`, `transition`, `ladder_lat`, `nest`, `mask`, `ref_dir`, `levels_dir`, `gebco`, `keep_seas`, `cap_method`, `cap_opts`, `size_tolerance`
- `[mesh.opts]` `m14_blocks_rep`: `place_region`, `max_blocks`, `b_cap`, `auto_blocks`, `strait_blocks`
- `[coast]`: `treatment`, `min_area_km2`, `passage_min_cells`, `straits`, `strait_profile`
- `[[size.layers]]` `latitude`: `h_m`, `h_min_m`
- `[[size.layers]]` `orca`: `orca`, `h_min_m`
- `[[size.layers]]` `zone`: `mode`, `h_m`, `h_factor`, `polygon`, `outer`, `transition_m`
- `[bathymetry]`: `depth_at`
- `[run]`: `smooth`

## Sources

- Code: `meshom/methods/m16_orca.py`, `meshom/orca/` (`generator.py`, `presets.py`, `mesh.py`, `ladder.py`,
  `nest.py`, `fesom3.py`), `scripts/grids/`, `scripts/orca_report.py`.
- Docs: [`docs/config.md` "NEMO's ORCA grids"](../config.md#nemos-orca-grids-method--m16_orca),
  [`docs/config.md` "FESOM3's rules"](../config.md#fesom3s-rules-mask--fesom3-python--m-meshomorcafesom3), `NOTICE`.
- Decisions: B78-B95, B100, B101, B103, B106, B108-B110, B112, B114-B116, B170-B176, C48, C52, S37.
