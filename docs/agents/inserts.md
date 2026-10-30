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

## Config keys used

- `[[inserts]]`: `name`, `kind`, `strait`, `path`, `cells`, `h_fine_m`, `h_min_m`, `reach_km`, `mouth_a`, `mouth_b`, `ljoins`, `bays`, `collar_rings`, `coast`, `coast_source`, `bank_sigma_cells`, `width_km`, `band_km`, `rows`, `coast_rounds`, `coast_shift_cells`, `cells_a`, `cells_b`, `growth_a`, `growth_b`, `bank_land_cells`, `stages`, `force`

## Sources

- Code: `meshom/insert/` (`__main__.py` the command, `core.py`, `region.py` the rules, `block.py`, `band.py`,
  `harmonic.py`, `realcoast.py`, `network.py` the bays, `assemble.py` the collar, `measure.py`),
  `meshom/config.py` (`InsertCfg`).
- Docs: [`docs/config.md` "`[[inserts]]`"](../config.md#inserts-python--m-meshominsert), `docs/examples/inserts.toml`,
  `docs/examples/inserts_gulf_stream.toml`.
- Decisions: B158-B169, B180-B186, B190, B192-B199.
