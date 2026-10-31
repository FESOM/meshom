"""meshom.orca (NEMO's ORCA generator rewritten, its presets, the grid as a mesh with NEMO's mask) and m16_orca
(decisions B71-B86): the generator against NEMO's reference grids (each test skipped when its npz is absent:
scripts/grids/orca_refs_npz.py), the code's layout equal to the lattice, the fold closed, the FESOM3 clean-up, the
ORCA nodes unchanged in a latitude-range mesh."""
import os
from pathlib import Path

import numpy as np
import pytest

from meshom.domain import Domain
from meshom.mesh import Mesh, build_topology, signed_area
from meshom.size import Uniform
from tests.helpers import land_loop

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT.parents[1] if ROOT.parent.name == ".worktrees" else ROOT


def _ref_dir() -> Path | None:
    for d in (os.environ.get("MESHOM_ORCA_REFS"), ROOT / ".tmp" / "orca_builder" / "ref",
              MAIN / ".tmp" / "orca_builder" / "ref"):
        if d and (Path(d) / "eorca1.npz").exists():
            return Path(d)
    return None


REFS = _ref_dir()
needs_refs = pytest.mark.skipif(REFS is None, reason="NEMO reference npz not found (scripts/grids/orca_refs_npz.py)")


def _dist_km(l1, p1, l2, p2):
    p1, p2, dl = np.radians(p1), np.radians(p2), np.radians(l2 - l1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.229 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def test_set_coefficients_orca1():
    """set_coefficients and hgr_ctl for ORCA1: the grid equator is the Mercator row 20 rows north of
    the equator (19.6058 N), the ellipses end on the segment between the poles 50 N / 66 N."""
    from meshom.orca import PRESETS
    from meshom.orca.generator import _fsa, _fsb, _fsy0, _fsy0p, hgr_ctl, set_coefficients

    p = PRESETS["ORCA1"].params
    c = set_coefficients(p)
    assert abs(c.rphieq - 19.605793951) < 1e-8
    assert abs(_fsy0p(c.rjpnth + c.rjpeq - 1.0, c.cf) - _fsy0p(c.rjpj, c.cf)) < 1e-12
    assert abs(_fsb(c.rjpnth, c.cf)) < 1e-14 and abs(_fsa(1.0, c.cf) - c.req) < 1e-14
    ya = _fsy0(c.rjpnth, c.cf) + _fsa(c.rjpnth, c.cf)
    yb = _fsy0(c.rjpnth, c.cf) - _fsa(c.rjpnth, c.cf)
    lat = sorted(90.0 - 2.0 * np.degrees(np.arctan(abs(y))) for y in (ya, yb))
    assert np.allclose(lat, [50.0, 66.0])
    assert hgr_ctl(p, c)["no_overlap"]


def test_code_layout_is_the_lattice():
    """The code's own grid (the Fortran's layout) is the q = 1 lattice with T at I = i + 1/2, J = j + 1."""
    from meshom.orca.generator import GenParams, grid

    p = GenParams(jpi=91, jpeqt=40, jpeqn=5, jpnord=28)
    a, b = grid(p), grid(p, q=1, i_offset=0.5, j_offset=1.0)
    for k in "tuvf":
        assert a.lat[k].shape == b.lat[k].shape
        assert np.abs(((a.lon[k] - b.lon[k]) + 180.0) % 360.0 - 180.0).max() < 1e-9
        assert np.abs(a.lat[k] - b.lat[k]).max() < 1e-9


@needs_refs
def test_orca1_is_eorca1():
    """ORCA1 (the Fortran's parameters, its tropical stretch on) is eORCA1 to its float32 rounding north of eORCA's
    ice-shelf rows (eORCA1 row = ORCA1 row + 40, rows from 80 on), the pole line included; the fold's halo row (291)
    but its point on the wrap column (where NEMO's halo exchange and the code's mirror differ by 0.78 km, on land)."""
    from meshom.orca import build

    g = build("ORCA1")
    ref = np.load(REFS / "eorca1.npz")
    assert g.lat["t"].shape == (292, 362) and g.pivot == "F"
    for k in "tf":
        rows = np.arange(40, 292)
        d = _dist_km(g.lon[k][rows], g.lat[k][rows], ref["glam" + k][rows + 40], ref["gphi" + k][rows + 40])
        assert d[:-1, :-2].max() < 1e-3 and d[-1, 1:-2].max() < 1e-3, (k, d[:, :-2].max())   # 1 m (float32)


@needs_refs
@pytest.mark.parametrize("name,ref,cols,rows", [("ORCA025", "eorca025", np.arange(0, 1440, 61), (330, 700, 1100)),
                                                ("ORCA12", "eorca12", np.arange(0, 4320, 397), (1000, 2100, 3500))])
def test_quadrature_presets(name, ref, cols, rows):
    """ORCA025 and ORCA12 are ORCA05's map on the 1/2 and 1/6 lattices (the user guide's quadrature): F points of a few
    columns on whole row bands (south, tropics, the north part) within 0.05 km of eORCA025 / eORCA12."""
    from meshom.orca import PRESETS
    from meshom.orca.generator import lattice, row_map, set_coefficients

    pr = PRESETS[name]
    c = set_coefficients(pr.params)
    rm = row_map(pr.params, c)
    joff = (c.rjpj + rm.shift) - pr.pole_row / pr.q
    rf = np.load(REFS / f"{ref}.npz")
    rr = np.r_[[np.arange(r, r + 40) for r in rows]].ravel()
    our = rr - pr.ref_row_offset
    h = 0.5 / pr.q
    lo, la = lattice(pr.params, c, rm, {"f": pr.i_offset + cols / pr.q + h}, {"f": joff + our / pr.q + h})
    d = _dist_km(lo["f"], la["f"], rf["glamf"][np.ix_(rr, cols)], rf["gphif"][np.ix_(rr, cols)])
    assert np.median(d) < 0.005 and d.max() < 0.05, (np.median(d), d.max())


EORCA36 = MAIN / "curvilinear_grids" / "nemo_eorca36" / "eORCA36.L121-coordinates_4.2.nc"


def _lattice_points(name, kind, rows, cols):
    """lon, lat of a lattice preset's T or F points at our rows x cols (generator.lattice, NEMO's standard rows)."""
    from meshom.orca import PRESETS
    from meshom.orca.generator import lattice, row_map, set_coefficients

    pr = PRESETS[name]
    c = set_coefficients(pr.params)
    rm = row_map(pr.params, c)
    joff = (c.rjpj + rm.shift) - pr.pole_row / pr.q
    h = 0.5 / pr.q if kind == "f" else 0.0
    lo, la = lattice(pr.params, c, rm, {kind: pr.i_offset + np.asarray(cols) / pr.q + h},
                     {kind: joff + np.asarray(rows) / pr.q + h})
    return lo[kind], la[kind]


def test_orca36_preset_nests_in_orca12_and_orca025():
    """B231: ORCA36 is ORCA05's map on the 1/18 lattice: 12960 columns, its pole row 9171 of 9173 rows, and every
    third F line is ORCA12's with b_i = -1, b_j = +1 (F_36(3 r + 1, 3 c - 1) = F_12(r, c)), every ninth ORCA025's
    (b_i = -4, b_j = +4); its NEMO water source is read with the reference's own offsets."""
    from meshom.orca import PRESETS
    from meshom.orca.fesom3 import SOURCES

    p36, p12 = PRESETS["ORCA36"], PRESETS["ORCA12"]
    assert p36.params == p12.params == PRESETS["ORCA025"].params
    assert (p36.q, p36.pole_row, p36.rows, (p36.params.jpi - 1) * p36.q) == (18, 9171, 9173, 12960)
    assert (SOURCES["ORCA36"].row_offset, SOURCES["ORCA36"].col_offset) == (p36.ref_row_offset, p36.ref_col_offset)
    r = np.arange(200, 3000, 397)
    c = np.arange(1, 4320, 611)
    R, C = np.meshgrid(r, c, indexing="ij")
    for coarse, k, b in (("ORCA12", 3, (-1, 1)), ("ORCA025", 9, (-4, 4))):
        lo_c, la_c = _lattice_points(coarse, "f", r, c)
        lo_f, la_f = _lattice_points("ORCA36", "f", k * r + b[1], k * c + b[0])
        d = _dist_km(lo_c, la_c, lo_f, la_f) * 1e3
        assert d.max() < 1e-6, (coarse, d.max())
        lo_s, la_s = _lattice_points("ORCA36", "f", k * r + b[1] + 1, k * c + b[0])
        assert _dist_km(lo_c, la_c, lo_s, la_s).min() > 0.01                    # one fine row off: a fine step away


@pytest.mark.skipif(not EORCA36.exists(), reason="eORCA36 coordinates (../curvilinear_grids/nemo_eorca36) not found")
def test_orca36_is_eorca36():
    """B231: the preset's T and F points against eORCA36.L121's coordinates (the reference's row of our row 0 is
    1670, its column of ours -1) on rows from the first exact parallel to 43 N: within 2 m (B231: south of 20 N at
    most 1.8 m, 30-42 N 3 mm)."""
    netCDF4 = pytest.importorskip("netCDF4")
    from meshom.orca import PRESETS

    pr = PRESETS["ORCA36"]
    rows = np.array([1300, 2400, 3000, 4200, 5500, 6500])
    cols = np.array([1, 2000, 4321, 6481, 9000, 12959])
    with netCDF4.Dataset(EORCA36) as ds:
        for k in "tf":
            lo, la = _lattice_points("ORCA36", k, rows, cols)
            ref = [np.asarray(ds[v + k][rows + pr.ref_row_offset, cols + pr.ref_col_offset]) for v in ("glam", "gphi")]
            d = _dist_km(lo, la, ref[0], ref[1]) * 1e3
            assert d.max() < 2.0 and np.median(d) < 0.01, (k, d.max())

def test_fold_closed():
    """The whole grid as a mesh: the fold (F pivot here) closes, the only open edges are the southern row and the
    small holes left at the two poles (the clamped pole line's slivers, on land); every element CCW."""
    from meshom.compete import irregular_counts
    from meshom.geo import xyz_to_lonlat
    from meshom.orca.generator import GenParams, grid
    from meshom.orca.mesh import cells

    g = grid(GenParams(jpi=91, jpeqt=40, jpeqn=5, jpnord=28))
    g.info["i_offset"] = 0.5
    oc = cells(g, 1, g.j_pole)
    m = oc.mesh
    assert oc.info["fold_nodes"] > 0 and oc.info["fold_max_gap_m"] < 1e-3
    top = build_topology(m)
    open_edges = top.edges[top.edge_elems[:, 1] < 0]
    _, lat = xyz_to_lonlat(m.points[np.unique(open_edges)])
    south = lat < -80.0
    assert south.sum() == 90                                         # the southern row
    assert (lat[~south] > 60.0).all() and (~south).sum() < 12          # pole holes only
    assert (signed_area(m) > 0).all() and irregular_counts(m)["euler_ok"]


def test_fesom_cleanup():
    """B84 on a quad grid: a spur (one neighbour) and a chain behind it removed, a pinch resolved by dropping the
    smaller fan, a closed sea kept or dropped by keep_seas."""
    from meshom.orca.mesh import fesom_cleanup

    n = 8
    lon, lat = np.meshgrid(np.arange(n + 1) * 1.0, np.arange(n + 1) * 1.0)
    from meshom.geo import lonlat_to_xyz

    pts = lonlat_to_xyz(lon.ravel(), lat.ravel())
    q = np.array([[j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i]
                  for j in range(n) for i in range(n)])
    mesh = Mesh(pts, np.zeros((0, 3), np.int64), q, np.zeros(len(pts), np.int8))
    wet = np.zeros((n, n), bool)
    wet[0:3, 0:3] = True                    # a 3x3 sea
    wet[3, 1] = True                        # a spur on it
    wet[3:5, 3:5] = True                    # a 2x2 block touching the sea only at a corner: a pinch
    wet[6:8, 6:8] = True                    # a closed 2x2 sea
    out, keep, info = fesom_cleanup(mesh, wet.ravel(), keep_seas=True)
    k = keep.reshape(n, n)
    assert info["spurs_removed"] >= 1 and not k[3, 1]
    assert info["pinch_cells_removed"] >= 1 and not (k[2, 2] and k[3, 3])
    assert k[6:8, 6:8].all() and k[0:2, 0:2].all()
    from meshom.methods.m14.cut import _pinch_drop, _spurs, half_edges

    he = half_edges(mesh)
    assert not _pinch_drop(he, keep, mesh.n_nodes).any() and not _spurs(he, keep).any()
    assert out.n_quads == int(keep.sum())
    _, keep2, info2 = fesom_cleanup(mesh, wet.ravel(), keep_seas=False)
    assert not keep2.reshape(n, n)[6:8, 6:8].any() and info2["water_bodies"] >= 2


def _sea_case():
    """A 10 x 10 quad grid: a ruled ocean, a ruled closed sea, unruled coast water cut off from the ocean by land."""
    from meshom.geo import lonlat_to_xyz

    n = 10
    lon, lat = np.meshgrid(np.arange(n + 1) * 1.0, np.arange(n + 1) * 1.0)
    pts = lonlat_to_xyz(lon.ravel(), lat.ravel())
    q = np.array([[j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i]
                  for j in range(n) for i in range(n)])
    mesh = Mesh(pts, np.zeros((0, 3), np.int64), q, np.zeros(len(pts), np.int8))
    cells = np.zeros((n, n), bool)
    cells[0:4, 0:4] = True                     # the ocean (ruled)
    cells[7:9, 0:2] = True                     # a ruled closed sea
    ruled = np.zeros((n, n), bool)
    ruled[:, :5] = True                        # the method's cells: columns 0-4; columns 5-9 are coast-cut cells
    # land under columns 5-9 up to 7.8 N: their rows 8-9 are coast water (10 cells) cut off from the ocean
    land = Domain([land_loop(7.5, 3.0, [[-280, -400], [280, -400], [280, 530], [-280, 530]])], "land")
    return mesh, land, lambda m: (ruled.ravel(), (cells & ruled).ravel())


def test_cut_keeps_the_methods_closed_seas():
    """B85 with `keep_seas` (B171): in the run's land cut a water body made only of a method's ruled water cells
    (NEMO's Caspian) stays although the default keeps the largest water body only (C48); the others follow C48."""
    from meshom.run import cut_land
    from meshom.straits import _components

    mesh, land, rule = _sea_case()
    cut, info = cut_land(mesh, land, wet_rule=rule, keep_method_seas=True)
    assert info["n_method_seas_kept"] == 1                  # the ruled sea back, the 10 coast cells dropped (C48)
    assert cut.n_quads == 20 and int(_components(cut).max()) + 1 == 2, (cut.n_quads, info)


def test_cut_drops_the_methods_closed_seas_by_default():
    """B171: by default the method's closed seas follow C48 as every other water body: only the
    ocean stays (FESOM3 has no initial data for them, and runoff would raise their level without bound)."""
    from meshom.methods.m16_orca import cleanup_keeps_seas, cut_keeps_seas
    from meshom.run import cut_land
    from meshom.straits import _components

    mesh, land, rule = _sea_case()
    cut, info = cut_land(mesh, land, wet_rule=rule)
    assert info["n_method_seas_kept"] == 0
    assert cut.n_quads == 16 and int(_components(cut).max()) + 1 == 1, (cut.n_quads, info)
    assert not cut_keeps_seas({}) and cut_keeps_seas({"keep_seas": True})
    # the ORCA part's clean-up: the whole grid drops its seas; a part with a cap keeps its bodies for the run's cut
    # (one may reach the ocean only through the cap), and keep_seas keeps them always
    assert not cleanup_keeps_seas({}, caps=False)
    assert cleanup_keeps_seas({}, caps=True) and cleanup_keeps_seas({"keep_seas": True}, caps=False)


@pytest.fixture(scope="module")
def range_mesh():
    from meshom.methods.m16_orca import run

    mesh, info = run(Domain([], "sphere"), Uniform(250e3),
                     {"orca": "ORCA2", "lat_north": 40.0, "mask": "all", "cap_method": "m15_tri",
                      "size_tolerance": None})
    return mesh, info


def test_range_joint_conforming(range_mesh):
    from meshom.compete import irregular_counts

    mesh, info = range_mesh
    j = info["joints"]["north"]
    assert j["conforming"] and j["matched"] == j["row_nodes"] == 180 and j["joint_edges_shared"] == 180
    assert (signed_area(mesh) > 0).all() and irregular_counts(mesh)["euler_ok"]


def test_range_orca_unchanged_after_cut(range_mesh):
    """B85: in the run's land cut the ORCA cells keep their own mask (cut_wet), the cap is cut by the coast; the ORCA
    nodes do not move and its elements stay as they were."""
    from meshom.methods.m16_orca import _STATE, belt_unchanged, cut_wet
    from meshom.run import cut_land

    mesh, info = range_mesh
    pts, quads, tris = _STATE["belt"]
    land = Domain([land_loop(10.0, 20.0, [[-1500, -1500], [1500, -1500], [1500, 2500], [-1500, 2500]]),
                   land_loop(10.0, 60.0, [[-800, -800], [800, -800], [800, 800], [-800, 800]])], "land")
    cut, cinfo = cut_land(mesh, land, wet_rule=cut_wet)
    assert cinfo["n_removed_land"] > 0 and cinfo["n_ruled"] == len(quads) + len(tris)
    assert cinfo["n_ruled_removed"] == 0
    u = belt_unchanged(cut, pts, quads, tris)
    assert u["max_displacement_m"] == 0.0 and u["nodes_within_1km_not_on_belt"] == 0
    assert u["belt_elems_altered"] == 0 and u["belt_elems_kept"] == u["belt_elems_total"]
    assert cut.n_elems < mesh.n_elems                                   # the cap lost its land
