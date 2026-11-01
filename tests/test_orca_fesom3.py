"""FESOM3's ORCA rules (meshom.orca.fesom3, decisions B87-B95) on small synthetic grids, the strait measure
(meshom.orca.straits), and ORCA025 against FESOM3's own mesh (`../fesom3_orca025`, skipped when absent)."""
from pathlib import Path

import numpy as np
import pytest

from meshom.orca import fesom3 as F

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT.parents[1] if ROOT.parent.name == ".worktrees" else ROOT
FESOM3_ORCA025 = MAIN.parent / "fesom3_orca025"
needs_fesom3 = pytest.mark.skipif(not (FESOM3_ORCA025 / "nemo025q_fesom" / "elem2d.out").exists(),
                                  reason="FESOM3's ORCA025 (../fesom3_orca025) not found")


def _wrap(kb):
    """NEMO's layout: the two wrap columns repeat the first two."""
    return np.concatenate([kb, kb[:, :2]], axis=1)


def test_one_cell_land_sweep():
    """Rule 3: a land cell with four water edge neighbours takes its east neighbour's level;
    two land cells side by side stay; the wrap column (NI - 2, the copy of column 0) sees the old copy of column 1."""
    kb = np.full((6, 8), 5, np.int16)
    kb[2, 3] = 0                                  # one cell: filled from (2, 4)
    kb[2, 4] = 7
    kb[4, 5] = kb[4, 6] = 0                       # two cells: stay
    k = _wrap(kb)
    k[3, 0] = k[3, 8] = 0                         # column 0's cell (its copy at column 8 is swept)
    k[3, 1] = k[3, 9] = 0                         # column 1 one cell (filled), but its old copy (col 9) stays land
    k[3, 2] = 6
    out, n = F.one_cell_land(k)
    assert out[2, 3] == 7 and out[4, 5] == 0 and out[4, 6] == 0
    assert out[3, 1] == 0                         # column 1's west neighbour (column 0) is land: not one cell
    k2 = k.copy()
    k2[3, 0] = k2[3, 8] = 5                       # column 0 water: column 1 is one cell, filled from column 2
    out2, n2 = F.one_cell_land(k2)
    assert out2[3, 1] == 6 and n2 == n + 1


def test_edits_in_order():
    """Rule 4: edits copy a neighbour's level in order; a later edit reads an earlier one's change (as two of FESOM3's
    Hell Gate edits do)."""
    kb = np.zeros((5, 6), np.int16)
    kb[:, 0] = 9
    e1 = F.Edit("a", (1, 4), (1, 2), (0, -1), 1)        # column 1 rows 1-3 <- column 0
    e2 = F.Edit("b", (2, 3), (2, 3), (0, -1), 2)        # (2, 2) <- (2, 1), set by e1
    out, rec = F.apply_edits(kb, (e1, e2))
    assert (out[1:4, 1] == 9).all() and out[2, 2] == 9 and [r["opened"] for r in rec] == [3, 1]


def test_levels_and_node_levels():
    """Rules 10 and 6: FESOM3's layer bottoms from NEMO's T depths (float32), the node level the deepest of its four
    T cells; the closing row and the fold row copy their neighbours, the last column the second."""
    z = F.levels_from_t(np.asarray(F.L46_T, np.float32))
    assert ["%8.4f" % v for v in z[:3]] == ["  0.0000", "  6.0935", " 12.8146"] and len(z) == 47
    kb = np.array([[0, 1, 2, 3], [4, 0, 6, 0], [0, 0, 0, 9]], np.int16)
    lev = F.node_levels(kb)
    assert lev.shape == (4, 4)
    assert lev[1, 0] == 4 and lev[1, 1] == 6 and lev[2, 2] == 9 and lev[2, 0] == 4
    assert (lev[0] == lev[1]).all() and (lev[3] == lev[2]).all() and (lev[:, 3] == lev[:, 1]).all()
    assert list(F.level_of_depth(np.array([0.0, 5.0, 12.8146, 13.0, 9000.0]), z, 3)) == [0, 3, 3, 3, 46]


def _tgrid(nj=8, nu=8):
    """A small T-pivot ORCA-like layout: nj T rows, nu unique columns + 2 wrap columns, all water."""
    return _wrap(np.full((nj, nu), 4, np.int16))


def test_numbering_fold_and_cycle():
    """Rule 7: ids column-major; the top extended row's first half takes the mirrored ids of
    the row below (column i -> NI - 1 - i, 0 -> 1); column NI - 2 is column 0."""
    nj, NI = 8, 10
    num = F.numbering(nj, NI, "T")
    ne = nj + 1
    assert num[3, 2] == 3 + ne * 2
    assert (num[:, NI - 2] == num[:, 0]).all()
    assert num[ne - 2, 1] == (ne - 3) + ne * (NI - 2) and num[ne - 2, 0] == num[ne - 3, 1]
    assert num[ne - 2, 4] == num[ne - 3, NI - 1 - 4]


def test_convert_small_grid():
    """Rules 8, 9, 12 on a small T-pivot grid: the quads (column-major, the pivot row's first half), SW-NW-NE-SE,
    unused nodes dropped, a spur chain removed pass by pass and its nodes flagged."""
    kb = _tgrid()
    kb[0] = 0                                     # south row land (as NEMO's)
    kb[1:3, 2:5] = 0
    kb[1, 3] = 4                                  # a spur: (1, 3) has one edge neighbour (2, 3) is land
    kb = _wrap(kb[:, :8])
    nj, NI = kb.shape
    lon = np.tile(np.arange(NI, dtype=float) * 10.0, (nj, 1))
    lat = np.tile(np.linspace(-60, 60, nj)[:, None], (1, NI))
    m0 = F.convert(lon, lat, lat, kb, np.arange(6.0), "T", (), None, deadends=False, ocean_only=False)
    half = (NI - 2) // 2
    n_q = int((kb[0:nj - 2, 1:NI - 1] > 0).sum()) + int((kb[nj - 2, 1:half + 1] > 0).sum())
    assert len(m0.elem) == n_q
    first = m0.cell[0]
    assert tuple(first) == (1, 1)                 # column 1, row 1 first (column-major)
    m = F.convert(lon, lat, lat, kb, np.arange(6.0), "T", (), None)
    assert m.info["deadends"]["removed"] >= 1 and len(m.elem) < len(m0.elem)
    assert (F.neighbour_count(m.elem, len(m.lon)) >= 2).all()
    assert not any((c == (1, 3)).all() for c in m.cell)
    # SW, NW, NE, SE: the second corner is north of the first, the fourth east of it
    e = m.elem[0]
    assert m.lat[e[1]] > m.lat[e[0]] and m.lon[e[3]] > m.lon[e[0]]


def _deepest_element_level(elem, nlev):
    """Per node: the deepest level of its elements, an element's = the fewest of its nodes' (fesom_ini NEMO mode)."""
    mb = np.zeros(len(nlev), np.int64)
    np.maximum.at(mb, elem.ravel(), np.repeat(nlev[elem].min(axis=1), elem.shape[1]))
    return mb


def test_node_levels_closed_after_deadends(tmp_path):
    """Rule 13 (B119): a dead end (rule 12) that held its corner nodes' deepest cell leaves them a level no element
    reaches, which FESOM3's bottom invariant rejects; closing the node levels makes every
    node's level its deepest element's, changes only those nodes (shallower) and no element's level."""
    kb = _tgrid()
    kb[0] = 0                                     # south row land
    kb[2:5, 1:6] = 0                              # a land block north of row 1
    kb[2, 3] = 6                                  # a deep spur: one edge neighbour (1, 3), a dead end
    kb = _wrap(kb[:, :8])
    nj, NI = kb.shape
    lon = np.tile(np.arange(NI, dtype=float) * 10.0, (nj, 1))
    lat = np.tile(np.linspace(-60, 60, nj)[:, None], (1, NI))
    z = np.arange(8.0)
    m0 = F.convert(lon, lat, lat, kb, z, "T", (), None, deadends=False)
    assert m0.info["node_levels_closed"] == 0     # without rule 12 the invariant holds already
    raw = F.remove_deadends(m0)                   # rule 12 alone: the old order
    assert not any((c == (2, 3)).all() for c in raw.cell)
    bad = _deepest_element_level(raw.elem, raw.nlev) != raw.nlev
    assert bad.sum() == 2 and (raw.nlev[bad] == 6).all()       # the spur's two south corners
    lev, n = F.close_node_levels(raw.elem, raw.nlev)
    assert n == 2 and (lev[bad] == 4).all() and (lev[~bad] == raw.nlev[~bad]).all()
    assert (_deepest_element_level(raw.elem, lev) == lev).all()
    assert (lev[raw.elem].min(axis=1) == raw.nlev[raw.elem].min(axis=1)).all()     # element levels unchanged
    m = F.convert(lon, lat, lat, kb, z, "T", (), None)
    assert m.info["node_levels_closed"] == 2 and (m.nlev == lev).all() and (m.elem == raw.elem).all()
    F.write(m, tmp_path)
    d = np.loadtxt(tmp_path / "depth.out", skiprows=1)
    nz, n_nod = len(z), len(m.lon)
    assert (d[nz:nz + n_nod] == z[lev]).all() and (d[nz + n_nod:] == lev).all()


def test_fold_fix_line():
    """Rule 2: a point off the straight line (polar plane) between its run's ends is put on it;
    points on the line stay."""
    lon = np.zeros((4, 12))
    lat = np.full((4, 12), 80.0)
    x = np.linspace(-0.1, 0.1, 5)                 # a straight line y = 0.05 in the polar plane
    y = np.full(5, 0.05)
    lon[2, 3:8] = np.degrees(np.arctan2(y, x))
    lat[2, 3:8] = np.degrees(np.arccos(np.hypot(x, y)))
    lo2, la2 = F.fold_fix(lon, lat, (2,), ((3, 8),))
    assert np.allclose(lo2[2, 3:8], lon[2, 3:8]) and np.allclose(la2[2, 3:8], lat[2, 3:8])
    lon[2, 5] += 3.0
    lo3, la3 = F.fold_fix(lon, lat, (2,), ((3, 8),))
    assert np.allclose(lo3[2, 3:8], lo2[2, 3:8]) and np.allclose(la3[2, 3:8], la2[2, 3:8])


def test_write_formats(tmp_path):
    """Rule 11: FESOM3's file formats."""
    m = F.Fesom3Mesh(np.array([1.5, 2.25], np.float32), np.array([-3.0, 4.0], np.float32),
                     np.array([[0, 1, 1, 0]]), np.array([1, 0]), np.array([2, 3]), np.array([0.0, 6.0, 12.0, 20.0]),
                     np.zeros((1, 2)), np.zeros((2, 2)))
    F.write(m, tmp_path)
    assert (tmp_path / "nod2d.out").read_text() == "        2 \n        1   1.5000  -3.0000        1\n" \
                                                     "        2   2.2500   4.0000        0\n"
    assert (tmp_path / "elem2d.out").read_text().splitlines()[1] == "        1         2         2         1"
    d = (tmp_path / "depth.out").read_text().splitlines()
    assert d[0] == "        4 " and d[5:7] == [" 12.0000", " 20.0000"] and d[7:] == ["        2 ", "        3 "]


def test_strait_measure():
    """B93: on a grid of cells, a channel 2 cells wide between two basins is open, 2 across, its sill the shallowest
    cell along the deepest path; blocked, it is closed."""
    from meshom.orca.straits import STRAITS, cell_links, measure

    STRAITS["_test"] = ((0.5, 0.5), (0.5, 8.5), (-1.0, 10.0, -1.0, 10.0))
    try:
        n = 9
        lon, lat = np.meshgrid(np.arange(n + 1) * 1.0, np.arange(n + 1) * 1.0)
        nid = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
        water = np.ones((n, n), bool)
        water[4, :] = False
        water[4, 3:5] = True                       # the channel: columns 3, 4 of row 4
        depth_c = np.full((n, n), 100.0)
        depth_c[4, 3], depth_c[4, 4] = 30.0, 60.0
        J, I = np.nonzero(water)
        el = np.stack([nid[J, I], nid[J + 1, I], nid[J + 1, I + 1], nid[J, I + 1]], axis=1)
        links = cell_links(el, nid.size)
        r = measure("_test", I + 0.5, J + 0.5, links, depth_c[J, I])
        assert r["open"] and r["cells_across"] == 2 and r["sill_m"] == 60.0
        keep = ~((J == 4) & (I == 4))
        el2 = el[keep]
        r2 = measure("_test", (I + 0.5)[keep], (J + 0.5)[keep], cell_links(el2, nid.size), depth_c[J, I][keep])
        assert r2["open"] and r2["cells_across"] == 1 and r2["sill_m"] == 30.0
        keep3 = ~((J == 4) & (I >= 3) & (I <= 4))
        el3 = el[keep3]
        r3 = measure("_test", (I + 0.5)[keep3], (J + 0.5)[keep3], cell_links(el3, nid.size))
        assert not r3["open"]
    finally:
        del STRAITS["_test"]


def _nemo025q() -> dict:
    import netCDF4

    out = {}
    src = FESOM3_ORCA025 / "nemo025q"
    for f, names in (("mesh_hgr.nc", ("glamf", "gphif", "nav_lat", "nav_lev")), ("mesh_zgr.nc", ("mbathy",))):
        with netCDF4.Dataset(src / f) as d:
            for k in names:
                v = d.variables[k]
                v.set_auto_maskandscale(False)
                a = np.array(v[:])
                out[k] = a[0] if a.ndim == 3 and a.shape[0] == 1 else a
    return out


@needs_fesom3
def test_orca025_inputs_give_fesom3_mesh(tmp_path):
    """The rules on FESOM3's own inputs give nemo025q_fesom: elem2d.out and depth.out byte for byte, nod2d.out but
    the one fold point FESOM3's MATLAB output has 1 float32 ulp off."""
    d = _nemo025q()
    m = F.convert(d["glamf"], d["gphif"], d["nav_lat"], d["mbathy"], F.levels_from_t(d["nav_lev"]), "T")
    assert m.info["deadends"]["removed"] == 46 and m.info["one_cell_land"] == 287
    F.write(m, tmp_path)
    ref = FESOM3_ORCA025 / "nemo025q_fesom"
    for f in ("elem2d.out", "depth.out"):
        assert (tmp_path / f).read_bytes() == (ref / f).read_bytes(), f
    a = (tmp_path / "nod2d.out").read_text().splitlines()
    b = (ref / "nod2d.out").read_text().splitlines()
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(a) == len(b) and len(diff) == 1, diff[:3]
    assert abs(float(diff[0][0].split()[1]) - float(diff[0][1].split()[1])) <= 1.01e-4


@needs_fesom3
def test_orca025_preset_gives_fesom3_mesh():
    """meshom's ORCA025 (the generator, preset ORCA025) with FESOM3's levels gives FESOM3's elements, flags and node
    levels exactly and its node positions to float32 for 98.7 % of the nodes, within 1 m but next to the geographic
    North Pole (B88)."""
    from meshom.geo import R_EARTH_M, lonlat_to_xyz
    from meshom.orca import build

    d = _nemo025q()
    z = F.levels_from_t(d["nav_lev"])
    g = build("ORCA025")
    m = F.convert(g.lon["f"], g.lat["f"], g.lat["t"], d["mbathy"], z, "T")
    n = F.convert(d["glamf"], d["gphif"], d["nav_lat"], d["mbathy"], z, "T")
    assert np.array_equal(m.elem, n.elem) and np.array_equal(m.flag, n.flag) and np.array_equal(m.nlev, n.nlev)
    a, b = lonlat_to_xyz(m.lon, m.lat), lonlat_to_xyz(n.lon.astype(float), n.lat.astype(float))
    dist = 2 * np.arcsin(np.clip(np.linalg.norm(a - b, axis=1) / 2, 0, 1)) * R_EARTH_M
    same = (m.lon.astype(np.float32) == n.lon) & (m.lat.astype(np.float32) == n.lat)
    assert same.mean() > 0.985
    pole_km = (90.0 - m.lat) * 111.2
    assert dist[pole_km > 100.0].max() < 5.0 and dist.max() < 700.0


def _grid_mesh(n):
    from meshom.geo import lonlat_to_xyz
    from meshom.mesh import Mesh

    lon, lat = np.meshgrid(np.arange(n + 1) * 1.0, np.arange(n + 1) * 1.0)
    pts = lonlat_to_xyz(lon.ravel(), lat.ravel())
    q = np.array([[j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i]
                  for j in range(n) for i in range(n)])
    return Mesh(pts, np.zeros((0, 3), np.int64), q, np.zeros(len(pts), np.int8))


def test_cut_method_sea_at_the_joint_follows_the_rest():
    """B95: a water body of the method's own cells that meets an unruled element (the Caspian's ORCA half at the
    cap's edge) follows the rest's rule (C48: dropped), one that does not stays (B85)."""
    from meshom.domain import Domain
    from meshom.run import cut_land
    from tests.helpers import land_loop

    n = 10
    mesh = _grid_mesh(n)
    cells = np.zeros((n, n), bool)
    cells[0:4, 0:4] = True                     # the ocean (ruled)
    cells[7:9, 0:2] = True                     # a ruled closed sea away from the joint
    cells[7:9, 3:5] = True                     # a ruled sea at the joint (column 5 is unruled)
    ruled = np.zeros((n, n), bool)
    ruled[:, :5] = True
    land = Domain([land_loop(7.5, 3.0, [[-280, -400], [280, -400], [280, 530], [-280, 530]])], "land")
    cut, info = cut_land(mesh, land, wet_rule=lambda m: (ruled.ravel(), (cells & ruled).ravel()), keep_method_seas=True)
    assert info["n_method_seas_kept"] == 1 and info["removed_clean_by"]["method_seas_at_joint_dropped"] == 4
    assert cut.n_quads == 20


def test_passage_rule_skips_the_methods_cells():
    """B95: a one-cell passage made only of skipped (ruled) elements is not reported, otherwise it is."""
    from meshom.methods.m14.cut import extract
    from meshom.passages import narrow_passages

    n = 9
    mesh = _grid_mesh(n)
    w = np.ones((n, n), bool)
    w[4, :] = False
    w[4, 4] = True                              # two basins joined by one cell
    sub, _, _ = extract(mesh, w.ravel())
    r = narrow_passages(sub, 2)
    assert len(r["passages"]) == 1 and r["n_skipped"] == 0
    r2 = narrow_passages(sub, 2, skip=np.ones(sub.n_elems, bool))
    assert len(r2["passages"]) == 0 and r2["n_skipped"] == 1


def test_run_puts_fesom3_node_depths_on_the_orca_nodes(tmp_path):
    """B94: the run's bathymetry stage gives the ORCA nodes their FESOM3 depth (orca_belt.npz) and leaves the rest."""
    from types import SimpleNamespace

    from meshom.mesh import Mesh
    from meshom.run import _method_node_depth

    pts = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-1.0, 0, 0]])
    mesh = Mesh(pts, np.array([[0, 1, 2], [1, 3, 2]]), np.zeros((0, 4), np.int64), np.zeros(4, np.int8))
    np.savez(tmp_path / "orca_belt.npz", points=pts[[2, 0]], node_level=np.array([3, 1]),
             z=np.array([0.0, 10.0, 20.0, 35.0]))
    depth = np.array([100.0, 200.0, 300.0, 400.0])
    cfg = SimpleNamespace(mesh=SimpleNamespace(method="m16_orca", opts={}))
    got = _method_node_depth(cfg, tmp_path, mesh, depth)
    assert got.tolist() == [10.0, 200.0, 35.0, 400.0]
    other = SimpleNamespace(mesh=SimpleNamespace(method="m14_blocks_rep"))
    assert _method_node_depth(other, tmp_path, mesh, depth) is depth


def test_nemo_halo_t_pivot():
    """B111: NEMO 4.2's halo-free T-pivot arrays back in NEMO's layout: the wrap columns, and the halo row by the north
    fold (T (nj-1, i) = (nj-3, NI-i), F (nj-1, i) = (nj-4, NI-1-i)); the pivot row of a T-pivot grid is self-mirrored
    (T (nj-2, i) = (nj-2, NI-i)) and the F row above it mirrors the one below (nj-2, i) = (nj-3, NI-1-i)."""
    ni, nj0 = 8, 6
    a = np.arange(nj0 * ni, dtype=float).reshape(nj0, ni)
    t = F.nemo_halo(a, "t")
    assert t.shape == (nj0 + 1, ni + 2)
    assert (t[:nj0, 1:ni + 1] == a).all() and (t[:, 0] == t[:, ni]).all() and (t[:, 1] == t[:, ni + 1]).all()
    NI, nj = ni + 2, nj0 + 1
    assert all(t[nj - 1, i] == t[nj - 3, NI - i] for i in range(1, NI - 1)) and t[nj - 1, 0] == t[nj - 1, ni]
    f = F.nemo_halo(a, "f")
    assert all(f[nj - 1, i] == f[nj - 4, NI - 1 - i] for i in range(1, NI - 1)) and f[nj - 1, 1] == f[nj - 1, ni + 1]


def test_cavity_tops(tmp_path):
    """B118: cavities as water (option kt): the node top is the shallowest top of its cells, so every cell keeps its
    water (FESOM3's element top = max over its nodes); FESOM2's fewest-layers fix raises tops until every element has
    3 layers; rule 3's filled cell takes its east neighbour's top; cavity_depth@node.out holds minus the top of the
    first wet layer, which fesom_ini's rule (which_depth_n2e='max') turns back into the
    element top. Without kt nothing changes."""
    z = np.arange(0.0, 70.0, 10.0)                       # 6 layers
    kb = _tgrid(8, 8)
    kb[:] = 6
    kb[0] = 0
    kt = np.ones_like(kb)
    kt[4:6, 4:7] = 3                                     # an ice shelf over three cells by two
    kt[1:4, 1:4] = 5                                     # too thin (2 layers): the middle cell's nodes see only it
    kb[2, 6], kt[2, 7] = 0, 2                            # one-cell land, filled from its east neighbour (top 2)
    kb, kt = _wrap(kb[:, :8]), _wrap(kt[:, :8])
    nj, NI = kb.shape
    lon = np.tile(np.arange(NI, dtype=float) * 10.0, (nj, 1))
    lat = np.tile(np.linspace(-60, 60, nj)[:, None], (1, NI))
    m0 = F.convert(lon, lat, lat, kb, z, "T", (), None)
    m = F.convert(lon, lat, lat, kb, z, "T", (), None, kt=kt)
    assert m0.ntop is None and (m.elem == m0.elem).all() and (m.nlev == m0.nlev).all()
    et = m.ntop[m.elem].max(axis=1)                       # FESOM3's element top
    own = kt[m.cell[:, 0], m.cell[:, 1]]
    assert (et <= own).all()                              # every cell keeps its water
    assert (m.nlev[m.elem].min(axis=1) - et + 1 >= 3).all() and m.info["tops_clamped"] > 0
    assert et[(m.cell[:, 0] == 2) & (m.cell[:, 1] == 2)] == [4]      # raised from 5 to keep 3 layers
    inner = (m.cell[:, 0] == 4) & (m.cell[:, 1] == 5)     # the shelf's middle cell: its own top
    assert et[inner] == [3]
    kb1, _ = F.one_cell_land(kb)
    assert kb1[2, 6] == 6 and F.one_cell_land_tops(kb, kb1, kt)[2, 6] == 2
    F.write(m0, tmp_path / "a")
    F.write(m, tmp_path / "b")
    assert not (tmp_path / "a" / "cavity_depth@node.out").exists()
    assert (tmp_path / "a" / "depth.out").read_bytes() == (tmp_path / "b" / "depth.out").read_bytes()
    cav = np.loadtxt(tmp_path / "b" / "cavity_depth@node.out")
    assert len(cav) == len(m.lon) and (cav <= 0).all() and ((cav == 0) == (m.ntop == 1)).all()
    zf = -z
    Z = 0.5 * (zf[:-1] + zf[1:])
    for e in range(len(m.elem)):                          # fesom_ini's raw ulevels with which_depth_n2e='max'
        dmean, nlv = cav[m.elem[e]].min(), m.nlev[m.elem[e]].min() + 1
        u = next(nz for nz in range(1, nlv) if Z[nz - 1] < dmean or nlv - nz <= 3)
        assert u == et[e]
    assert list(F.top_level_of_draft(np.array([0.0, 5.0, 10.0, 10.5, 65.0]), z)) == [1, 1, 2, 2, 6]


def _lonlat(kb):
    nj, NI = kb.shape
    return (np.tile(np.arange(NI, dtype=float) * 10.0, (nj, 1)),
            np.tile(np.linspace(-60, 60, nj)[:, None], (1, NI)))


def _in(cell, rows, cols):
    return (cell[:, 0] >= rows[0]) & (cell[:, 0] < rows[1]) & (cell[:, 1] >= cols[0]) & (cell[:, 1] < cols[1])


def _boundary_nodes(elem, n):
    a, b = elem, np.roll(elem, -1, axis=1)
    key = (np.minimum(a, b) * n + np.maximum(a, b)).ravel()
    u, c = np.unique(key, return_counts=True)
    bnd = np.zeros(n, bool)
    bnd[u[c == 1] // n] = bnd[u[c == 1] % n] = True
    return bnd


def test_water_components_join_at_a_common_layer():
    """F.water_components: elements join across a shared edge; with layer columns (top, bottom) only where the two
    columns share a layer. Component 0 is the largest (most elements)."""
    elem = np.array([[0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3]])          # three quads in a row: A | B | C
    lab, size = F.water_components(elem, 8)
    assert size.tolist() == [3] and lab.tolist() == [0, 0, 0]
    lab, size = F.water_components(elem, 8, np.array([1, 5, 1]), np.array([3, 8, 8]))
    assert size.tolist() == [2, 1] and lab.tolist() == [1, 0, 0]           # A ends above B's top: cut off
    lab, size = F.water_components(elem, 8, np.array([1, 3, 1]), np.array([3, 8, 8]))
    assert size.tolist() == [3]                                           # layer 3 shared


def test_closed_water_removed(tmp_path):
    """Rule 12b (C48, B170): water cut off from the world ocean (a 2 x 2 pocket inside a ring of land) is removed
    after the dead ends; the record names it (elements, nodes, place); the rest is unchanged. The writer refuses a
    mesh with more than one water component and names the extra one (C52)."""
    kb = _tgrid(12, 12)
    kb[0] = 0
    kb[3:9, 3:9] = 0                              # a land ring two cells wide ...
    kb[5:7, 5:7] = 4                              # ... around a 2 x 2 pocket of water
    kb = _wrap(kb[:, :12])
    lon, lat = _lonlat(kb)
    z = np.arange(6.0)
    m0 = F.convert(lon, lat, lat, kb, z, "T", (), None, ocean_only=False)
    m = F.convert(lon, lat, lat, kb, z, "T", (), None)
    pocket = _in(m0.cell, (5, 7), (5, 7))
    assert pocket.sum() == 4 and F.water_components(m0.elem, len(m0.lon))[1].tolist() == [len(m0.elem) - 4, 4]
    assert F.water_components(m.elem, len(m.lon))[1].tolist() == [len(m.elem)]
    assert np.array_equal(m.cell, m0.cell[~pocket]) and not _in(m.cell, (5, 7), (5, 7)).any()
    keep = np.unique(m0.elem[~pocket])
    assert np.array_equal(m.lon, m0.lon[keep]) and np.array_equal(m.nlev, m0.nlev[keep])
    assert np.array_equal(m.flag, m0.flag[keep])
    cw = m.info["closed_water"]
    assert cw["elems"] == 4 and cw["nodes"] == 9 and len(cw["removed"]) == 1
    r = cw["removed"][0]
    assert r["n_elems"] == 4 and r["n_nodes"] == 9 and r["layers"] == [4, 4]
    pl = m0.lon[np.unique(m0.elem[pocket])]
    assert r["box"][:2] == [float(pl.min()), float(pl.max())] and pl.min() <= r["at"][0] <= pl.max()
    F.write(m, tmp_path / "ok")
    with pytest.raises(ValueError, match=r"2 water components.*4 elements, 9 nodes"):
        F.write(m0, tmp_path / "bad")
    assert not (tmp_path / "bad" / "elem2d.out").exists()


def test_closed_water_with_cavities():
    """Rule 12b with cavities (B118): elements join only across a shared edge at a common layer of FESOM3's vertex
    columns. An open pocket inside an ice shelf (top layer 5) whose water ends at layer 3 meets the cavity water
    nowhere: a closed pool, removed; its border nodes stay with the shelf and become coast (flag 1). A pocket down to
    layer 8 meets the cavity water under the shelf and stays."""
    z = np.arange(0.0, 90.0, 10.0)                # 8 layers
    kb = _tgrid(12, 12)
    kb[:] = 8
    kb[0] = 0
    kt = np.ones_like(kb)
    kt[3:9, 3:9] = 5                              # an ice shelf two cells wide ...
    kt[5:7, 5:7] = 1                              # ... around an open 2 x 2 pocket
    shallow = kb.copy()
    shallow[5:7, 5:7] = 3
    kt = _wrap(kt[:, :12])
    lon, lat = _lonlat(kt)
    deep = F.convert(lon, lat, lat, _wrap(kb[:, :12]), z, "T", (), None, kt=kt)
    assert _in(deep.cell, (5, 7), (5, 7)).sum() == 4 and deep.info["closed_water"]["elems"] == 0
    sh = _wrap(shallow[:, :12])
    m0 = F.convert(lon, lat, lat, sh, z, "T", (), None, kt=kt, ocean_only=False)
    et, eb = m0.ntop[m0.elem].max(axis=1), m0.nlev[m0.elem].min(axis=1)
    assert F.water_components(m0.elem, len(m0.lon))[1].tolist() == [len(m0.elem)]       # one piece in 2-D
    assert F.water_components(m0.elem, len(m0.lon), et, eb)[1].tolist() == [len(m0.elem) - 4, 4]
    m = F.convert(lon, lat, lat, sh, z, "T", (), None, kt=kt)
    assert not _in(m.cell, (5, 7), (5, 7)).any() and len(m.elem) == len(m0.elem) - 4
    cw = m.info["closed_water"]
    assert cw["elems"] == 4 and cw["nodes"] == 1 and cw["removed"][0]["n_nodes"] == 9
    rc0 = {tuple(r) for r in m0.node_rc[_boundary_nodes(m0.elem, len(m0.lon))].tolist()}
    coast = _boundary_nodes(m.elem, len(m.lon)) & np.array([tuple(r) not in rc0 for r in m.node_rc.tolist()])
    assert coast.sum() == 8 and (m.flag[coast] == 1).all()          # the pocket's border: new coast, flagged
    et, eb = m.ntop[m.elem].max(axis=1), m.nlev[m.elem].min(axis=1)
    assert F.water_components(m.elem, len(m.lon), et, eb)[1].tolist() == [len(m.elem)]


def test_closed_water_removal_leaves_no_dead_end(tmp_path):
    """Rule 12 again after rule 12b (B170): an open cell in an ice shelf that ends above the shelf's water touches two
    shelf cells by edges without a common layer; once it is removed they have one edge neighbour each, on which
    FESOM3's partitioner stops, so they go as dead ends."""
    b = np.full((16, 16), 20, np.int16)
    b[0] = 0
    b[-3:] = 0
    t = np.ones_like(b)
    t[3:11, 3:11] = 15                            # an ice shelf ...
    for r, c in [(5, 5), (5, 6), (5, 7), (6, 5), (7, 5), (7, 7), (7, 8)]:
        b[r, c] = 0                               # ... with land in it ...
    b[6, 6], t[6, 6] = 3, 1                       # ... next to an open cell whose water ends at layer 3
    b, t = _wrap(b), _wrap(t)
    lon, lat = _lonlat(b)
    m = F.convert(lon, lat, lat, b, np.arange(22.0), "T", (), None, kt=t)
    cells = {tuple(c) for c in m.cell.tolist()}
    assert m.info["closed_water"]["elems"] == 1 and (6, 6) not in cells
    assert m.info["deadends"]["removed"] == 2 and not {(7, 6), (6, 7)} & cells
    assert F.neighbour_count(m.elem, len(m.lon)).min() >= 2
    assert ((m.flag == 1) == _boundary_nodes(m.elem, len(m.lon))).all()
    F.write(m, tmp_path)
    assert m.info["water"]["n_components"] == 1


def test_closed_water_goes_before_node_levels_are_closed():
    """Rule 12b before rule 13 (B170): a deeper closed pocket (6 layers) touches the ocean (4 layers) at one node, which
    stays with the ocean; once the pocket is gone, rule 13 gives that node the ocean's 4 layers, as FESOM3's bottom
    invariant asks (B119). Rule 13 first would leave it 6 layers with no element that deep."""
    kb = _tgrid(12, 12)
    kb[0] = 0
    kb[3:9, 3:9] = 0                              # a land ring ...
    kb[4:6, 4:6] = 6                              # ... around a deeper pocket at its inner corner ...
    kb[3, 3] = 4                                  # ... whose corner cell is water: one node shared with the ocean
    kb = _wrap(kb[:, :12])
    lon, lat = _lonlat(kb)
    m0 = F.convert(lon, lat, lat, kb, np.arange(8.0), "T", (), None, ocean_only=False)
    pocket = {tuple(r) for r in m0.node_rc[np.unique(m0.elem[_in(m0.cell, (4, 6), (4, 6))])].tolist()}
    m = F.convert(lon, lat, lat, kb, np.arange(8.0), "T", (), None)
    assert m.info["closed_water"]["elems"] == 4 and not _in(m.cell, (4, 6), (4, 6)).any()
    assert (m.nlev == _deepest_element_level(m.elem, m.nlev)).all()
    shared = np.flatnonzero([tuple(r) in pocket for r in m.node_rc.tolist()])
    assert len(shared) == 1 and m.nlev[shared[0]] == 4 and m.flag[shared[0]] == 1


def test_closed_water_uses_the_clamped_tops():
    """Rule 12b joins elements at the tops convert ends with (B118's 3-layer fix included, B170): a shelf of one raw
    layer (top 5, bottom 5) is clamped to layers 3-5 and so meets an open pocket of layers 1-3 at layer 3; the pocket
    is kept. With the raw tops it would be removed although FESOM3 sees it joined."""
    z = np.arange(0.0, 90.0, 10.0)
    kb = _tgrid(12, 12)
    kb[:] = 8
    kb[0] = 0
    kt = np.ones_like(kb)
    kt[3:9, 3:9] = kb[3:9, 3:9] = 5               # the shelf: one raw layer
    kt[5:7, 5:7], kb[5:7, 5:7] = 1, 3             # the open pocket: layers 1-3
    kb, kt = _wrap(kb[:, :12]), _wrap(kt[:, :12])
    lon, lat = _lonlat(kb)
    m = F.convert(lon, lat, lat, kb, z, "T", (), None, kt=kt)
    assert m.info["tops_clamped"] > 0 and m.info["closed_water"]["elems"] == 0
    assert _in(m.cell, (5, 7), (5, 7)).sum() == 4
