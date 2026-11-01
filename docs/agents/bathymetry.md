---
name: meshom-bathymetry
description: How meshom gives the nodes their depth - the elevation grid, the box mean or deepest-cell rule, the floor, smoothing, slope limiter, strait sills, depth regions by area and node edits of one mesh - and what each model needs. Read it when the request mentions depth, sills or vertical levels, or before a model run.
---

# Bathymetry: node depths

## The depth stage

With `[bathymetry] source` set, the `bathymetry` stage (`meshom.bathymetry.node_depth`) gives every node of the
final mesh a depth in metres, positive down, saved as `depth.npy` and written into the model files. Without a source
every node gets the placeholder 4400 m: always give a grid for a model run.

- `source`: a regular lon-lat netCDF grid of elevation (positive up, ocean < 0), such as GEBCO 2024 (15″). It is
  read in bands of rows, never whole; the read takes about 1.5-2 minutes whatever the mesh size.
- `variable` (`"elevation"`): the grid variable.
- `depth_at`:
  - `"node"` (default): the mean of the grid's **ocean** cells in a lon-lat box of half-width = half the node's mean
    edge (its control cell). Nodes without an ocean cell take the length-weighted mean of their neighbours, ring by
    ring (O16, O17). `box_factor` (1) scales the box's side: 2 gives a box of 4 times the area, a smoother depth
    (B222; also in an insert's and the nest's own rule).
  - `"cells_deepest"` (B103): per element the mean of the ocean cells in a box of the element's area, and each node
    the deepest of its elements. Narrow straits keep their depth (the box means of `"node"` make coastal nodes
    shallow). FESOM3's ORCA rule applied to meshom's cells.

## Depth rules, in this order (O27)

1. box mean (or `cells_deepest`) and fill;
2. **floor**: nodes shallower than the floor get it. Floor = `min_depth_m` (10 m); for `format = "fesom2"` /
   `"fesom3"` at least the bottom of the `min_layers`-th level: max(`min_depth_m`, zbar[`min_layers`]) = 30 m with
   the CORE2 levels and `min_layers = 4` (O18). FESOM2 and FESOM3 raise shallower nodes to that anyway, so 4 writes
   the depth the model uses;
3. **smoothing**: `smooth_passes` passes of volume-conserving diffusion (0 = off; O28, O29). 4 passes let the
   CORE2-like strait mesh run FESOM2 at 45 min;
4. **slope limiter**: `max_slope` (0 = off). Measured to make Gibraltar worse: keep it off (O30);
5. **strait carving**: nodes within a strait's `h_m` of its path get at least its `depth_m`, with a cosine taper
   over `carve_taper` × `h_m` beyond (O31). The built-in list carries the sill depths (Bosphorus 35 m,
   Dardanelles 55 m);
6. floor again.

The `bathymetry` record in `stages.json` counts the nodes raised to the floor, the nodes without an ocean cell, the
moves of each rule (`policy`: nodes moved by more than 1 m and 10 %, the largest moves, the volume change, the slope
p99 before and after).

## Vertical levels

`levels` (`"core2"`: the 48 level surfaces, 47 layers, of the FESOM2 CORE2 mesh; a list of depths in metres; or a
file with nl then nl values, such as an `aux3d.out`). Written into `aux3d.out` for FESOM2 and FESOM3 (O19).

## Depth regions (`[[bathymetry.regions]]`, O35, O36)

Rules by area, applied after every other depth step (including `m16_orca`'s NEMO depths; `[mesh.opts] depth =
"gebco"` switches those off, B221), in file order, in
`meshom run` and in every `meshom write` of that run:

```toml
[[bathymetry.regions]]
name = "Gibraltar"
box = [-6.2, 35.6, -5.0, 36.3]     # [lon_min, lat_min, lon_max, lat_max]; lon_min > lon_max crosses 180
min_depth_m = 50.0                 # at least this deep; also depth_m (set, first) and max_depth_m (at most, last)
```

- Note the box order differs from `[[strait_regions]]` (`[west, east, south, north]`).
- `polygon = [[lon, lat], ...]` instead of `box`: great-circle edges, within a hemisphere.
- After its rule a region's nodes are kept at the floor (a `max_depth_m` below 30 m gives 30 m for FESOM2 / FESOM3).
  A later region overrides an earlier one where they overlap.
- The record `regions` gives per region `n_nodes`, `n_changed`, `n_clamped_to_floor`.
- A run keeps the regions of its `config.resolved.toml`: a region added to the config later reaches the next run,
  not the old run's writes.

## Node edits of one mesh (O37, O38)

`RUN_DIR/<mesh stem>.depth_edits.json` (written by the web UI's node tool, or by you) holds edits by node index
(0-based, the `.npz` mesh's order):

```json
{"mesh": "raw.npz", "n_nodes": 144410,
 "edits": [{"op": "set", "depth_m": 60.0, "nodes": [101, 102]}]}
```

`op`: `"set"`, `"min"` (at least), `"max"` (at most), in list order, then the floor. Only `meshom write RUN_DIR
--mesh <stem>` applies them (never `meshom run`); an `n_nodes` other than the mesh's stops the write. The write saves
the depth before the edits as `fesom-<mesh>/depth_base.npy`. The first write of a run builds `RUN_DIR/cache/` (the
grid's box sums); a write after a few edits then takes seconds and gives the same depth bit for bit (O33).

## What the models need

- FESOM2 / FESOM3: `aux3d.out` with the levels and the node depths (negative down); every node at least zbar(5)
  deep in the model; give `min_layers = 4`. FESOM3's `fesom_ini` makes the node levels from it.
- FESOM-C: `depth.out` (positive down) is what the model uses; `aux3d.out` is a placeholder.
- `fesom2_checks` / `fesom3_checks` flag |depth| > 11 000 m and depths above zbar(5).

## Config keys used

- `[bathymetry]`: `source`, `variable`, `depth_at`, `box_factor`, `min_depth_m`, `min_layers`, `smooth_passes`, `max_slope`, `carve_taper`, `levels`, `regions`
- `[[bathymetry.regions]]`: `name`, `box`, `polygon`, `depth_m`, `min_depth_m`, `max_depth_m`
- `[[straits]]`: `depth_m`, `h_m`
- `[mesh]`: `format`

## Sources

- Code: `meshom/bathymetry.py` (`node_depth`, `depth_policy`, `smooth_depth`, `apply_regions`, `apply_node_edits`),
  `meshom/straits.py` (`depth_floor`), `meshom/run.py` (`model_depth`, `write_mesh`).
- Docs: [`docs/config.md` "`[bathymetry]`"](../config.md#bathymetry),
  [`docs/config.md` "Depth regions"](../config.md#depth-regions-bathymetryregions-and-node-edits).
- Decisions: B94, B103, B221, B222, O16-O19, O27-O31, O33, O35-O38.
