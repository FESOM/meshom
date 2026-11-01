"""The depth of an insert's own nodes and of the host nodes round it (decisions B210-B217), on a synthetic strait: a
bathymetry grid of two deep basins (with a trench along the axis) joined by a strait of 13 km with a deep channel of
3.7 km and a 300 m sill, and a structured mesh along it whose middle part plays the insert."""
import dataclasses
import json

import numpy as np
import pytest

from meshom.geo import R_EARTH_M, lonlat_to_xyz, xyz_to_lonlat
from tests.helpers import structured_patch

netCDF4 = pytest.importorskip("netCDF4")

D = 1.0 / 120.0                              # grid spacing (deg), ~0.93 km
DX = 2800.0                                  # the mesh's spacing (m)
ZBAR = [0.0, 10.0, 20.0, 30.0, 50.0, 100.0, 150.0, 200.0, 250.0, 280.0, 340.0, 400.0, 500.0, 700.0, 1000.0,
        2000.0, 4000.0]
QUIET = lambda *a, **k: None                 # noqa: E731


def _write_grid(path, lon, lat, z):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        ds.createVariable("lat", "f8", ("lat",))[:] = lat
        ds.createVariable("lon", "f8", ("lon",))[:] = lon
        ds.createVariable("elevation", "f4", ("lat", "lon"))[:] = z
    return path


@pytest.fixture(scope="module")
def strait(tmp_path_factory):
    """(grid file, grid lon, grid lat, mesh, insert nodes): basins |lon| >= 0.45 of 2000 m with a 3500 m trench
    |lat| < 0.005; the strait |lat| < 0.06 of 150 m with a channel |lat| < 0.013 of 600 m; a sill of 300 m on two
    grid columns at 0.10-0.12 E with two 450 m cells on a diagonal across it (8-connected only); land elsewhere. The
    mesh: 64 x 3 quads of 2.8 km from 0.81 W to 0.81 E, its rows at +-1.4 and +-4.2 km off the axis; the insert's own
    nodes |lon| < 0.65."""
    lon = np.arange(-1.0, 1.0, D) + D / 2
    lat = np.arange(-0.4, 0.4, D) + D / 2
    LO, LA = np.meshgrid(lon, lat)
    z = np.full(LO.shape, 50.0)
    basin = np.abs(LO) >= 0.45
    z[basin] = -2000.0
    z[basin & (np.abs(LA) < 0.005)] = -3500.0
    st = (np.abs(LO) < 0.45) & (np.abs(LA) < 0.06)
    z[st] = -150.0
    z[st & (np.abs(LA) < 0.013)] = -600.0
    jj = np.flatnonzero((lon > 0.10) & (lon < 0.12))
    rr = np.flatnonzero(np.abs(lat) < 0.013)
    z[np.ix_(rr, jj)] = -300.0
    z[rr[1], jj[0]] = z[rr[2], jj[1]] = -450.0
    nc = _write_grid(tmp_path_factory.mktemp("grid") / "strait.nc", lon, lat, z)
    m = structured_patch(64, 3, DX)
    lo, _ = xyz_to_lonlat(m.points)
    return nc, lon, lat, m, np.abs(lo) < 0.65


def _cfg(nc, **bathy):
    from meshom.config import loads

    extra = "".join(f"{k} = {v!r}\n".replace("'", '"') for k, v in bathy.items())
    return loads(f'[mesh]\nmethod = "m00_background"\nformat = "fesom2"\n[bathymetry]\nsource = "{nc}"\n'
                 f'levels = {ZBAR}\nmin_layers = 4\n{extra}')


@pytest.fixture(scope="module")
def line(strait):
    from meshom.bathymetry import node_depth
    from meshom.insert import channel as chan

    nc, _, _, m, ins = strait
    base, _ = node_depth(m, nc, "elevation", 30.0)
    lo, la = xyz_to_lonlat(m.points)
    grid = chan.read_window(nc, "elevation", lo, la, 8000.0)
    return base, grid, chan.channel_line(m, ins, grid, DX)


# ------------------------------------------------------------------------------------------------ B213, B215

def test_channel_line_follows_the_channel_and_its_narrows_stay_in_the_strait(strait, line):
    """B213: the channel line runs between the two joins (the host pieces at both ends of the insert) along the
    deep channel; its narrows (wet width at most 6 x the narrowest, 13 km here) cover the strait and none of the open
    basins; the record names the narrowest width and the limit."""
    _, lon, lat, m, ins = strait
    base, grid, ln = line
    rec = ln.record["narrows"]
    assert abs(abs(ln.lon[0]) - 0.65) < 0.03 and abs(abs(ln.lon[-1]) - 0.65) < 0.03  # the joins' ends, in the basins
    assert ln.lon[0] * ln.lon[-1] < 0
    assert 12.0 < rec["width_min_km"] < 13.5 and rec["limit_km"] == pytest.approx(6.0 * rec["width_min_km"])
    nar = ln.narrow
    assert np.abs(ln.lon[nar]).max() < 0.45 + 2 * D and np.abs(ln.lon[~nar]).min() > 0.45 - D
    assert (np.abs(ln.lon) < 0.40)[~nar].sum() == 0                                  # the whole strait
    assert np.abs(ln.lat[nar]).max() < 0.013                                         # on the channel
    assert ln.record["deepest_connection_box_m"] > 300.0


@pytest.mark.parametrize("mode", ["near", "cells"])
def test_channel_raise_acts_on_the_narrows_only(strait, line, mode):
    """B213: the raise deepens the insert's nodes along the channel in the strait ("near": nodes within one spacing
    of the line take at least the box mean of their own size centred on the line; "cells": the nodes of every cell
    the line crosses take at least the cell's channel depth), never lowers a node, never touches a host node, and
    leaves the basins alone although their trench along the line is deeper than the nodes there (which the same
    raise on the whole line deepens)."""
    from scipy.spatial import cKDTree

    from meshom.bathymetry import node_spacing_m
    from meshom.insert import channel as chan
    from meshom.mesh import signed_area
    from meshom.straits import locate

    _, _, _, m, ins = strait
    base, grid, ln = line
    d, rec = chan.line_raise(m, base, ins, grid, ln, mode)
    lo, _ = xyz_to_lonlat(m.points)
    up = d - base
    assert (up >= 0).all() and (up[~ins] == 0).all()
    raised = up > 1e-6
    assert raised.sum() == rec["nodes_raised"] > 10 and rec["raise_max_m"] == pytest.approx(up.max())
    assert np.abs(lo[raised]).max() < 0.45
    pts = lonlat_to_xyz(ln.lon[ln.narrow], ln.lat[ln.narrow])
    if mode == "near":
        sp, _, _ = node_spacing_m(m)
        dist, j = cKDTree(pts).query(m.points)
        near = ins & (2 * np.arcsin(dist / 2) * R_EARTH_M <= chan.CHANNEL_NEAR * sp)
        assert not (raised & ~near).any()
        bm = grid.box_mean(ln.lon[ln.narrow][j[near]], ln.lat[ln.narrow][j[near]], 0.5 * sp[near])
        assert np.allclose(d[near], np.where(np.isfinite(bm), np.maximum(base[near], bm), base[near]))
    else:
        el = locate(m, pts)
        crossed = np.zeros(m.n_nodes, bool)
        area = np.abs(signed_area(m))
        for e in np.unique(el[el >= 0]):
            nd = m.quads[e - m.n_tris]
            k = el == e
            want = np.nanmean(grid.box_mean(ln.lon[ln.narrow][k], ln.lat[ln.narrow][k],
                                            np.full(k.sum(), 0.5 * np.sqrt(area[e]))))
            assert (d[nd[ins[nd]]] >= want - 1e-9).all()
            crossed[nd[ins[nd]]] = True
        assert not (raised & ~crossed).any()
    whole, _ = chan.line_raise(m, base, ins, grid, dataclasses.replace(ln, narrow=None), mode)
    assert ((whole - base > 1e-6) & (np.abs(lo) > 0.5)).sum() > 0                  # the trench would dig the basins


def test_grid_saddle_is_the_four_connected_deepest_connection_of_the_grid(line):
    """B215: the saddle is the grid's own deepest connection between the line's ends over 4-connected cells and the
    grid's depth (300 m on the sill), not the box means' (deeper: the box holds channel water) and not through the
    two 450 m cells that touch only at a corner."""
    _, _, ln = line
    sd, slon, slat = ln.saddle
    assert sd == pytest.approx(300.0) and 0.10 < slon < 0.12 and abs(slat) < 0.013
    assert ln.record["grid_saddle"]["m"] == pytest.approx(300.0)
    assert ln.record["deepest_connection_box_m"] > 400.0


def test_grid_sill_caps_a_dam_through_the_saddle_from_bank_to_bank(strait, line):
    """B215: the chain of the insert's nodes through the node nearest the saddle to both banks (the least sum of
    depths) and every node of a cell holding one are capped at the midpoint between L = 280 m (the deepest level
    surface not below 300 m) and the next level 340 m, minus 1 m; after the cap every cell round a chain node has its
    mean above the midpoint, so the dam keeps level L; nothing else changes."""
    from meshom.insert import channel as chan

    _, _, _, m, ins = strait
    base, grid, ln = line
    lo, la = xyz_to_lonlat(m.points)
    dep = np.where(ins, 600.0, base)
    p = lonlat_to_xyz(np.array([ln.saddle[1]]), np.array([ln.saddle[2]]))
    via = int(np.argmin(np.linalg.norm(m.points - p, axis=1)))
    out, rec = chan.grid_sill(m, dep, ins, ln, np.array(ZBAR))
    assert (rec["saddle_m"], rec["level_m"], rec["next_level_m"]) == (pytest.approx(300.0), 280.0, 340.0)
    assert rec["cap_m"] == 0.5 * (280.0 + 340.0) - chan.SILL_MARGIN_M == 309.0
    capped = out < dep
    assert capped.sum() == rec["capped"] and rec["capped_host_nodes"] == 0
    assert (out[capped] == 309.0).all() and capped[via]
    dam = capped[m.quads].all(1)                                 # the cells round the chain: every node capped
    assert dam.sum() >= 3 and (out[m.quads[dam]].mean(1) < 310.0).all()
    lat_rows = np.unique(np.round(la[capped], 4))
    assert len(lat_rows) == 4                                    # bank to bank
    assert np.abs(lo[capped] - lo[via]).max() < 3.5 * DX / 111e3
    assert np.array_equal(out[~capped], dep[~capped])


# ------------------------------------------------------------------------------------------------ B218

def _variant(tmp_path_factory, name, edit):
    """the strait fixture's grid with `edit(z, LO, LA)` applied, written to its own file."""
    lon = np.arange(-1.0, 1.0, D) + D / 2
    lat = np.arange(-0.4, 0.4, D) + D / 2
    LO, LA = np.meshgrid(lon, lat)
    z = np.full(LO.shape, 50.0)
    basin = np.abs(LO) >= 0.45
    z[basin] = -2000.0
    z[basin & (np.abs(LA) < 0.005)] = -3500.0
    st = (np.abs(LO) < 0.45) & (np.abs(LA) < 0.06)
    z[st] = -150.0
    z[st & (np.abs(LA) < 0.013)] = -600.0
    jj = np.flatnonzero((lon > 0.10) & (lon < 0.12))
    rr = np.flatnonzero(np.abs(lat) < 0.013)
    z[np.ix_(rr, jj)] = -300.0
    z[rr[1], jj[0]] = z[rr[2], jj[1]] = -450.0
    edit(z, LO, LA)
    return _write_grid(tmp_path_factory.mktemp(name) / f"{name}.nc", lon, lat, z)


def _own(nc, m, ins, base):
    from meshom.insert.core import rule_depth

    return rule_depth(m, ins, base, _cfg(nc), "fesom2", None, depth_at="node", depth_smooth=0, strait_floor=False,
                      sill_level="grid", depth_channel="near", h_fine=DX, log=QUIET)


def test_channel_ends_come_from_the_grid_not_from_the_host_depths(strait, tmp_path_factory):
    """B218: the west basin has a 40 m shelf north of 0.025 N, where the join's northern host node stands; the host's
    depth (an ORCA node: the deepest of four ocean model cells) puts 3000 m there. The line's ends are the joins'
    deepest box means (the trench), so the deepest connection and the grid's saddle are the strait's (300 m), the dam
    is built and nothing is end-limited; the host's depths change nothing."""
    from meshom.bathymetry import node_depth

    nc = _variant(tmp_path_factory, "shelf", lambda z, LO, LA: z.__setitem__((LO <= -0.45) & (LA > 0.025), -40.0))
    _, _, _, m, ins = strait
    lo, la = xyz_to_lonlat(m.points)
    base, _ = node_depth(m, nc, "elevation", 30.0)
    shelf = int(np.flatnonzero(~ins & (lo < 0))[np.argmax(la[~ins & (lo < 0)])])
    host = base.copy()
    host[shelf] = 3000.0
    d, rec = _own(nc, m, ins, host)
    ch = rec["channel"]
    assert ch["grid_saddle"]["m"] == pytest.approx(300.0) and 0.10 < ch["grid_saddle"]["at"][0] < 0.12
    assert ch["deepest_connection_box_m"] > 300.0 and ch["end_limited"] == [False, False]
    assert ch["end_from"] == ["join", "join"] and min(ch["end_box_m"]) > 2000.0
    assert all(abs(e[1]) < 0.013 for e in ch["ends"])                              # on the trench, not the shelf
    assert rec["sill"]["saddle_m"] == pytest.approx(300.0) and rec["sill"]["capped"] > 0
    d2, rec2 = _own(nc, m, ins, base)
    assert np.array_equal(d, d2) and rec2["channel"]["ends"] == ch["ends"]


def test_channel_line_stays_in_the_insert(strait, tmp_path_factory):
    """B218: an 800 m channel outside the insert (8.3-9.4 km north of the axis, past the mesh's outer row and its one
    fine cell of margin) joins the two basins deeper than the strait's 300 m sill. The line, its deepest connection
    and the saddle stay in the strait; the narrows lie on the strait's channel."""
    def edit(z, LO, LA):
        z[(np.abs(LO) < 0.45) & (LA > 0.075) & (LA < 0.085)] = -800.0

    nc = _variant(tmp_path_factory, "side", edit)
    _, _, _, m, ins = strait
    from meshom.bathymetry import node_depth

    base, _ = node_depth(m, nc, "elevation", 30.0)
    _, rec = _own(nc, m, ins, base)
    ch = rec["channel"]
    assert abs(ch["narrows"]["at"][1]) < 0.013 and abs(ch["shallowest_on_line"]["at"][1]) < 0.06
    assert 300.0 < ch["deepest_connection_box_m"] < 800.0 and ch["grid_saddle"]["m"] == pytest.approx(300.0)
    assert ch["corridor"] and rec["sill"]["capped"] > 0


def test_the_saddle_is_found_when_an_end_cell_is_shallow(strait, tmp_path_factory):
    """B218: the grid column under the west join is a 250 m ridge one cell wide between deep water (its box mean,
    over three columns, is ~1400 m, so the line's end sits on it); from that end cell the grid's own search is
    end-limited and finds no saddle cell, so the saddle is sought from the deepest grid cell of each leg of the line:
    the strait's 300 m sill, and the dam is built."""
    from meshom.bathymetry import node_depth

    def ridge(z, LO, LA):
        lon = LO[0]
        col = lon[np.argmin(np.abs(lon + 0.65473))]
        z[np.isclose(LO, col)] = -250.0

    nc = _variant(tmp_path_factory, "ridge1", ridge)
    _, _, _, m, ins = strait
    base, _ = node_depth(m, nc, "elevation", 30.0)
    _, rec = _own(nc, m, ins, base)
    ch = rec["channel"]
    assert ch["grid_saddle"]["m"] == pytest.approx(300.0) and 0.10 < ch["grid_saddle"]["at"][0] < 0.12
    assert ch["grid_saddle"]["from"] == "the deepest grid cell of each leg of the line"
    assert ch["end_limited"] == [False, False] and rec["sill"]["capped"] > 0


def test_the_saddle_of_two_sills_of_one_depth_in_series(strait, tmp_path_factory):
    """B218: a second 300 m sill across the channel (0.12-0.10 W) in series with the first: no single band of grid
    cells at the critical depth touches both sides, so the saddle is the band on the route between them nearest the
    line's shallowest point (one of the two sills, 300 m), and the dam is built."""
    from meshom.bathymetry import node_depth

    def two(z, LO, LA):
        z[(LO > -0.12) & (LO < -0.10) & (np.abs(LA) < 0.013)] = -300.0

    nc = _variant(tmp_path_factory, "two_sills", two)
    _, _, _, m, ins = strait
    base, _ = node_depth(m, nc, "elevation", 30.0)
    _, rec = _own(nc, m, ins, base)
    ch = rec["channel"]
    at = ch["grid_saddle"]["at"][0]
    assert ch["grid_saddle"]["m"] == pytest.approx(300.0) and (0.10 < at < 0.12 or -0.12 < at < -0.10)
    assert abs(at - ch["shallowest_on_line"]["at"][0]) < 0.03 and rec["sill"]["capped"] > 0


def test_a_too_shallow_join_takes_the_basin_pool_beyond_it(strait, tmp_path_factory):
    """B218: a 100 m ridge along the west join (0.675-0.635 W, the whole basin across) with the deep basin beyond and
    behind it: the join's deepest box mean (100 m) is the line's deepest connection (the search is end-limited at
    that join), so its end moves to the deepest cell of the pool beyond the join (outside the insert's cells, nearer
    that join, joined to it through cells at least as deep as the join's end); the connection stays the ridge's,
    now not end-limited, and the grid's saddle is found on the ridge. With the whole basin beyond the join a shelf
    as shallow as the join, the end stays and the record says the search is end-limited there."""
    from meshom.bathymetry import node_depth

    _, _, _, m, ins = strait

    def ridge(z, LO, LA):
        z[(LO > -0.675) & (LO < -0.635) & (np.abs(LO) >= 0.45)] = -100.0

    nc = _variant(tmp_path_factory, "ridge", ridge)
    base, _ = node_depth(m, nc, "elevation", 30.0)
    _, rec = _own(nc, m, ins, base)
    ch = rec["channel"]
    assert ch["grid_saddle"]["m"] == pytest.approx(100.0) and -0.675 < ch["grid_saddle"]["at"][0] < -0.635
    west = int(np.argmin([e[0] for e in ch["ends"]]))
    assert ch["end_from"][west] == "pool" and ch["end_from"][1 - west] == "join"
    assert ch["join_deepest_box_m"][west] == pytest.approx(100.0, abs=1.0)
    assert ch["join_limited"][west] and not ch["join_limited"][1 - west]
    assert ch["ends"][west][0] < -0.6547 and ch["end_box_m"][west] > 1000.0 and ch["pool_cells"][west] > 0
    assert ch["deepest_connection_box_m"] == pytest.approx(100.0, abs=1.0) and ch["end_limited"] == [False, False]

    def shelf(z, LO, LA):
        z[(LO <= -0.635) & (np.abs(LO) >= 0.45)] = -100.0

    nc2 = _variant(tmp_path_factory, "flat", shelf)
    base2, _ = node_depth(m, nc2, "elevation", 30.0)
    _, rec2 = _own(nc2, m, ins, base2)
    ch2 = rec2["channel"]
    west = int(np.argmin([e[0] for e in ch2["ends"]]))
    assert ch2["end_from"][west] == "join" and ch2["end_limited"][west] and not ch2["end_limited"][1 - west]
    assert ch2["join_limited"][west]


# ------------------------------------------------------------------------------------------------ B212, B214

def test_sill_rounded_to_the_shallower_level():
    """B212: the barrier is the bank-to-bank chain of the insert's nodes whose deepest node is shallowest (260 m
    here, not the 250 m column with one 420 m node); the chain and every node sharing a cell with it are capped at
    the midpoint between L = 250 m and 280 m minus 1 m, so every cell round the chain has its mean above the
    midpoint; host nodes and nodes farther away keep their depth."""
    from meshom.insert.core import SILL_MARGIN_M, sill_shallower

    m = structured_patch(20, 3, DX)
    n = 21
    col = np.arange(m.n_nodes) % n
    ins = (col >= 2) & (col <= 18)
    dep = np.full(m.n_nodes, 500.0)
    dep[col == 6] = [250.0, 250.0, 420.0, 250.0]
    dep[col == 14] = 260.0
    out, rec = sill_shallower(m, dep, ins, np.array(ZBAR))
    assert rec["barrier_m"] == 260.0 and rec["level_m"] == 250.0 and rec["next_level_m"] == 280.0
    assert rec["cap_m"] == 265.0 - SILL_MARGIN_M == 264.0
    capped = out < dep
    assert np.array_equal(np.flatnonzero(capped), np.flatnonzero(np.isin(col, [13, 15])))
    assert rec["capped"] == 8 and rec["capped_host_nodes"] == 0 and rec["chain_nodes"] == 4
    near = np.isin(col, [13, 14, 15])
    cells = near[m.quads].all(1)
    assert (out[m.quads[cells]].mean(1) < 265.0).all()
    assert np.array_equal(out[~capped], dep[~capped])


def test_node_quantile_of_the_box(strait, tmp_path):
    """B214: an insert node takes the q-quantile of the grid's ocean cells in its node box (half its mean edge
    length), at least the floor; host nodes and nodes whose box holds no ocean cell keep their depth."""
    from meshom.bathymetry import node_spacing_m
    from meshom.insert import channel as chan

    nc, _, _, m, ins = strait
    lo, la = xyz_to_lonlat(m.points)
    grid = chan.read_window(nc, "elevation", lo, la, 8000.0)
    dep = np.full(m.n_nodes, 77.0)
    d, rec = chan.node_quantile(m, dep, ins, grid, 0.75, 30.0)
    sp, _, _ = node_spacing_m(m)
    ci, cj, hi, hj = grid.boxes(lo, la, 0.5 * sp)
    for i in np.flatnonzero(ins):
        z = grid.elev[ci[i] - hi[i]:ci[i] + hi[i] + 1, cj[i] - hj[i]:cj[i] + hj[i] + 1]
        assert d[i] == max(float(np.quantile(-z[z < 0].astype(float), 0.75)), 30.0)
    assert (d[~ins] == 77.0).all() and rec["nodes"] == ins.sum() and rec["kept_without_ocean_cell"] == 0
    assert (d[ins] > 350.0).any()
    land = structured_patch(2, 2, DX, lat0=0.25)                 # every box on land
    lo2, la2 = xyz_to_lonlat(land.points)
    g2 = chan.read_window(nc, "elevation", lo2, la2, 8000.0)
    d2, r2 = chan.node_quantile(land, np.full(9, 55.0), np.ones(9, bool), g2, 0.75, 30.0)
    assert (d2 == 55.0).all() and r2["kept_without_ocean_cell"] == 9


# ------------------------------------------------------------------------------------------------ B211

def test_own_rule_box_mean_without_smoothing_or_strait_floor(strait):
    """B211: depth_at "node", depth_smooth 0, strait_floor false: the insert's nodes take the box mean per node (at
    least the floor), not the run's smoothed value nor the run's strait depth floor; depth_smooth > 0 smooths among
    the insert's nodes only; None takes the run's passes over the whole part."""
    from meshom.bathymetry import node_depth
    from meshom.insert.core import rule_depth
    from meshom.run import depth_floor_m
    from meshom.straits import Strait

    nc, _, _, m, ins = strait
    cfg = _cfg(nc, smooth_passes=4)
    floor = depth_floor_m(cfg, "fesom2")
    assert floor == 50.0
    box, _ = node_depth(m, nc, "elevation", floor)
    path = lonlat_to_xyz(np.linspace(-0.3, 0.3, 7), np.zeros(7))
    straits = [Strait("s", path, 4000.0, 3000.0, depth_m=1500.0)]
    d0, r0 = rule_depth(m, ins, box, cfg, "fesom2", straits, depth_at="node", depth_smooth=0, strait_floor=False,
                        log=QUIET)
    assert np.array_equal(d0[ins], np.maximum(box[ins], floor))
    assert r0["rule"]["smooth_passes"] == 0 and r0["rule"]["strait_floor"] is False and r0["nodes"] == ins.sum()
    d1, r1 = rule_depth(m, ins, box, cfg, "fesom2", straits, depth_at="node", depth_smooth=0, strait_floor=True,
                        log=QUIET)
    lo, la = xyz_to_lonlat(m.points)
    on = (np.abs(lo) < 0.3) & (np.abs(la) < 0.02)                       # within h_m of the strait's path
    assert (d1[ins] >= d0[ins]).all() and (d0[on] < 1500.0).all() and (d1[on] == 1500.0).all()
    d2, r2 = rule_depth(m, ins, box, cfg, "fesom2", straits, depth_smooth=None, strait_floor=False, log=QUIET)
    assert r2["rule"]["smooth_passes"] == 4 and r2["rule"]["smooth_among"] == "run"
    assert not np.allclose(d2[ins], d0[ins]) and not np.allclose(d2[~ins], box[~ins])
    d3, r3 = rule_depth(m, ins, box, cfg, "fesom2", straits, depth_smooth=2, strait_floor=False, log=QUIET)
    assert r3["rule"]["smooth_among"] == "insert" and np.array_equal(d3[~ins], np.maximum(box[~ins], floor))
    assert not np.allclose(d3[ins], d0[ins])


# --------------------------------------------------------------------------------------------- B210, B211, B217

@pytest.fixture(scope="module")
def spliced(strait):
    """A host window round the strait's middle and an 'insert' in it: its own nodes (|x| < 40 km, off the coast)
    moved by 300 m, two host nodes of the middle row at 50 and 53 km moved by the collar 0.45 and 0.6 spacings west
    (so the first one lies nearer to the second's new place than to its own), spliced back."""
    from meshom.insert.core import Result
    from meshom.insert.host import cut_window, splice
    from meshom.mesh import Mesh

    _, _, _, host, _ = strait
    win = cut_window(host, lonlat_to_xyz(np.array([0.0]), np.array([0.0]))[0], 60e3)
    w = win.mesh
    lo, la = xyz_to_lonlat(w.points)
    x = lo * np.pi / 180 * 6.371e6
    p = w.points.copy()
    own = (np.abs(x) < 40e3) & (win.kind == 0)
    east = np.array([0.0, 1.0, 0.0])                         # at (0, 0) east is +y
    p[own] += 300.0 / 6.371e6 * east
    mid = np.isclose(la, la[np.argmin(np.abs(la))])
    a = int(np.flatnonzero(mid & (win.kind == 0))[np.argmin(np.abs(x[mid & (win.kind == 0)] - 50.4e3))])
    b = int(np.flatnonzero(mid)[np.argmin(np.abs(x[mid] - x[a] - DX))])
    p[a] -= 0.45 * DX / 6.371e6 * east
    p[b] -= 0.60 * DX / 6.371e6 * east
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    new = Mesh(p, w.tris, w.quads, w.node_flag)
    res = Result(True, {}, window=new)
    res.collar_moved = np.array([a, b])
    full, sinfo = splice(host, win, new)
    h2n = sinfo.pop("host_to_new")
    return host, win, res, full, sinfo, h2n, own, (a, b)


def test_collar_moved_host_nodes_are_matched_one_to_one(spliced):
    """B217: the host nodes the collar moved are matched to the appended nodes within 1.5 of their spacings with the
    least sum of distances (the nearest appended node of the first is the second's), and a node moved beyond 1.5
    spacings is left unmatched and named."""
    from meshom.insert.core import collar_moved

    host, win, res, full, sinfo, h2n, own, (a, b) = spliced
    fid, hid, rec = collar_moved(win, res, full, sinfo)
    assert rec["moved"] == rec["matched"] == 2 and not rec["unmatched"]
    want = {int(win.old_id[a]): res.window.points[a], int(win.old_id[b]): res.window.points[b]}
    for f, h in zip(fid, hid):
        assert np.allclose(full.points[f], want[int(h)], atol=1e-15)
    assert 0.55 * DX / 1e3 < rec["move_max_km"] < 0.65 * DX / 1e3
    far = dataclasses.replace(res)
    far.collar_moved = np.array([a, b])
    p = res.window.points.copy()
    p[b] += 3.0 * DX / R_EARTH_M * np.array([0.0, 0.0, 1.0])         # 3 spacings north
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    far.window = dataclasses.replace(res.window, points=p)
    from meshom.insert.host import splice

    full2, sinfo2 = splice(host, win, far.window)
    _, hid2, rec2 = collar_moved(win, far, full2, sinfo2)
    assert rec2["matched"] == 1 and list(hid2) == [int(win.old_id[a])] and len(rec2["unmatched"]) == 1


def _depth_of(full, spliced, cfg, spec, **kw):
    from meshom.insert.core import insert_depth
    from meshom.run import model_depth

    host, win, res, _, sinfo, h2n, _, _ = spliced
    d = model_depth(full, cfg, "fesom2", [], None)
    return d, insert_depth(d, host, win, res, full, sinfo, h2n, kw.pop("moved", None), spec, cfg, "fesom2", [],
                           None, QUIET, **kw)


def test_insert_depth_own_rule_and_host_depth_round_it(strait, spliced):
    """B211 / B217 / B210 in insert_depth: the insert's own nodes (those the splice appended, less the host nodes
    the collar moved) take the own rule; the collar's moved host nodes and the host nodes the fixer moved keep the
    run's depth at their host positions; every other node keeps the run's depth."""
    from meshom.bathymetry import node_depth
    from meshom.insert.core import Spec, collar_moved
    from meshom.run import model_depth

    nc = strait[0]
    host, win, res, full, sinfo, h2n, own, (a, b) = spliced
    cfg = _cfg(nc, smooth_passes=4)
    spec = Spec("t", np.zeros((2, 3)), depth_at="node", depth_smooth=0, strait_floor=False)
    cfid, chid, _ = collar_moved(win, res, full, sinfo)
    run_d, (d, rec) = _depth_of(full, spliced, cfg, spec, collar=(cfid, chid))
    app = np.arange(full.n_nodes - sinfo["appended_nodes"], full.n_nodes)
    mine = np.setdiff1d(app, cfid)
    assert len(mine) == own.sum() and rec["insert_nodes"] == len(mine) and rec["collar_moved_kept"] == 2
    box, _ = node_depth(full, nc, "elevation", 50.0)
    assert np.allclose(d[mine], np.maximum(box[mine], 50.0), rtol=0, atol=1e-9)
    assert not np.allclose(run_d[mine], d[mine])                             # the run's rule smooths
    hw = model_depth(win.mesh, cfg, "fesom2", [], None)
    wpos = {int(g): k for k, g in enumerate(win.old_id)}
    assert np.array_equal(d[cfid], hw[[wpos[int(h)] for h in chid]])
    assert not np.allclose(run_d[cfid], d[cfid])                            # their new places' depth differs
    rest = np.setdiff1d(np.arange(full.n_nodes), app)
    assert np.array_equal(d[rest], run_d[rest])
    # the fixer moves two kept host nodes (B210): they keep the depth at their host positions
    lo, _ = xyz_to_lonlat(full.points)
    kept = np.flatnonzero(h2n >= 0)
    fx = h2n[kept[np.argsort(np.abs(lo[h2n[kept]] - 0.35))[:2]]]
    moved_full = dataclasses.replace(full, points=full.points.copy())
    moved_full.points[fx] = full.points[fx] + 900.0 / 6.371e6 * np.array([0.0, 0.0, 1.0])
    moved_full.points[fx] /= np.linalg.norm(moved_full.points[fx], axis=1, keepdims=True)
    run2, (d2, rec2) = _depth_of(moved_full, spliced, cfg, Spec("t", np.zeros((2, 3))), moved=fx)
    host_of = {int(h2n[i]): int(i) for i in kept}
    assert np.array_equal(d2[fx], hw[[wpos[host_of[int(f)]] for f in fx]])
    assert not np.allclose(run2[fx], d2[fx]) and rec2["host_moved_kept"] == 2


def test_chain_keeps_the_saved_host_depth(strait, spliced, tmp_path):
    """B217: on an insert's result (insert_<name>.npz with fesom-insert_<name>/depth.npy beside it, written after
    the mesh, one value per node) every host node the splice keeps and the collar's moved host nodes take the saved
    depth, not the run's rules; another file name, an older depth file or another node count give None."""
    import os

    from meshom.insert.core import Spec, collar_moved, host_depth_of

    nc = strait[0]
    host, win, res, full, sinfo, h2n, own, _ = spliced
    src = tmp_path / "insert_gib.npz"
    src.write_bytes(b"mesh")
    saved = np.linspace(1000.0, 1100.0, host.n_nodes)
    (tmp_path / "fesom-insert_gib").mkdir()
    f = tmp_path / "fesom-insert_gib" / "depth.npy"
    np.save(f, saved)
    st = os.stat(src)
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert np.array_equal(host_depth_of(src, host.n_nodes), saved)
    assert host_depth_of(src, host.n_nodes + 1) is None
    other = tmp_path / "run_mesh.npz"
    other.write_bytes(b"mesh")
    assert host_depth_of(other, host.n_nodes) is None
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns - 10**9))
    assert host_depth_of(src, host.n_nodes) is None
    cfg = _cfg(nc, smooth_passes=4)
    cfid, chid, _ = collar_moved(win, res, full, sinfo)
    run_d, (d, rec) = _depth_of(full, spliced, cfg, Spec("t", np.zeros((2, 3))), host=saved, collar=(cfid, chid))
    ok = h2n >= 0
    assert np.array_equal(d[h2n[ok]], saved[ok]) and rec["host_depth"]["nodes"] == ok.sum()
    assert np.array_equal(d[cfid], saved[chid])
    app = np.arange(full.n_nodes - sinfo["appended_nodes"], full.n_nodes)
    mine = np.setdiff1d(app, cfid)
    assert np.array_equal(d[mine], run_d[mine])                               # no own rule: the run's


# ------------------------------------------------------------------------------------------------ B216

def _lattice_mesh(lon, lat, wet=None):
    """Quads on the lon-lat lattice (rows of lat, columns of lon) whose centre is wet (wet(lon, lat), default all),
    the unused nodes dropped."""
    from meshom.mesh import Mesh

    LO, LA = np.meshgrid(lon, lat)
    nr, nc = LA.shape
    idx = np.arange(nr * nc).reshape(nr, nc)
    q = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    if wet is not None:
        q = q[wet(LO.ravel()[q].mean(1), LA.ravel()[q].mean(1))]
    used = np.unique(q)
    loc = np.full(nr * nc, -1)
    loc[used] = np.arange(len(used))
    return (Mesh(lonlat_to_xyz(LO.ravel()[used], LA.ravel()[used]), np.zeros((0, 3), np.int64), loc[q],
                 np.zeros(len(used), np.int8)), used)


@pytest.fixture(scope="module")
def nest_run(tmp_path_factory):
    """A run folder whose orca_belt.npz holds a lattice of F points (0.025 deg x 0.02 deg round the strait) and a
    nest's coarse box on it (0.4 W to 0.4 E, 0.08 S to 0.08 N: its north and south lines on land), and a mesh of that
    lattice's wet cells."""
    run_dir = tmp_path_factory.mktemp("nest_run")
    lon, lat = -0.8 + 0.025 * np.arange(65), -0.16 + 0.02 * np.arange(17)
    LO, LA = np.meshgrid(lon, lat)
    np.savez(run_dir / "orca_belt.npz", lon=LO, lat=LA, t_rows=np.array([1, 16]),
             nest_coarse_box=np.array([16, 48, 4, 12]))
    m, used = _lattice_mesh(lon, lat, lambda x, y: (np.abs(x) >= 0.45) | (np.abs(y) < 0.06))
    return run_dir, m, used % len(lon), used // len(lon)


def test_nest_depth_by_the_grid_rule_inside_the_coarse_box(strait, nest_run):
    """B216: the nodes strictly inside the nest's coarse box (the coarse F points on its four lines as one loop: those
    are outside) take the box mean per node (no smoothing, no strait depth floor, at least the floor); every other
    node keeps its depth; model_depth does it only for m16_orca with nest.depth = "gebco"."""
    from meshom.bathymetry import node_depth
    from meshom.methods.m16_orca import nest_box_nodes, nest_depth
    from meshom.run import model_depth

    nc = strait[0]
    run_dir, m, col, row = nest_run
    inside = nest_box_nodes(run_dir, m)
    assert np.array_equal(inside, (col > 16) & (col < 48) & (row > 4) & (row < 12)) and inside.sum() > 20
    cfg = _cfg(nc, smooth_passes=4)
    cfg.mesh.opts["nest"] = {"depth": "gebco"}
    before = np.full(m.n_nodes, 4321.0)
    d, rec = nest_depth(run_dir, m, before, cfg, "fesom2", log=QUIET)
    box, _ = node_depth(m, nc, "elevation", 50.0)
    assert np.array_equal(d[inside], np.maximum(box[inside], 50.0)) and (d[~inside] == 4321.0).all()
    assert rec["rule"]["smooth_passes"] == 0 and rec["rule"]["strait_floor"] is False and rec["nodes"] == inside.sum()
    run_d = model_depth(m, cfg, "fesom2", [], run_dir)                 # method m00: no nest rule
    cfg.mesh.method = "m16_orca"
    nest_d = model_depth(m, cfg, "fesom2", [], run_dir)
    assert np.array_equal(nest_d[~inside], run_d[~inside]) and np.array_equal(nest_d[inside], d[inside])
    assert not np.allclose(run_d[inside], d[inside])
    cfg.mesh.opts["nest"] = {"depth": ""}
    assert np.array_equal(model_depth(m, cfg, "fesom2", [], run_dir), run_d)


def test_nest_depth_is_the_same_in_every_mesh_holding_the_box(strait, nest_run):
    """B216: the box's nodes ordered by position and its cells by their nodes: another node numbering, another cell
    order and cells added outside the box (an insert elsewhere) give the box's nodes the same depth to the bit, also
    with the channel raise between the box edge's two wet stretches, the sill and the slope limiter."""
    from meshom.methods.m16_orca import nest_box_nodes, nest_depth
    from meshom.mesh import Mesh
    from meshom.run import model_depth

    nc = strait[0]
    run_dir, m, _, _ = nest_run
    cfg = _cfg(nc, max_slope=0.02)
    cfg.mesh.opts["nest"] = {"depth": "gebco", "depth_channel": "near", "sill_level": "grid"}
    run_d = model_depth(m, _cfg(nc), "fesom2", [], run_dir)            # the run's depth (the record only)
    d0, rec0 = nest_depth(run_dir, m, run_d, cfg, "fesom2", log=QUIET)
    assert rec0["channel"]["raise"]["nodes_raised"] > 0 and rec0["sill"]["capped"] > 0
    extra, _ = _lattice_mesh(np.linspace(0.85, 0.95, 4), np.linspace(-0.1, 0.1, 5))   # cells outside the box
    pts = np.vstack([m.points, extra.points])
    quads = np.vstack([m.quads, extra.quads + m.n_nodes])
    perm = np.random.default_rng(7).permutation(len(pts))
    inv = np.argsort(perm)
    other = Mesh(pts[perm], np.zeros((0, 3), np.int64), np.roll(inv[quads], 1, axis=1)[::-1],
                 np.zeros(len(pts), np.int8))
    inside0, inside1 = nest_box_nodes(run_dir, m), nest_box_nodes(run_dir, other)
    assert np.array_equal(inside1[inv[:m.n_nodes]], inside0)
    d1, _ = nest_depth(run_dir, other, model_depth(other, _cfg(nc), "fesom2", [], run_dir), cfg, "fesom2", log=QUIET)
    assert np.array_equal(d1[inv[:m.n_nodes]][inside0], d0[inside0])


# ------------------------------------------------------------------------------------------------ B250

ZFINE = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0] + [float(v) for v in range(75, 1001, 25)] + [1500.0, 2000.0, 3000.0,
                                                                                        4000.0]


def _levels_reference(el, depth, zbar, thers=5):
    """fesom_ini's level loop element by element, in its own order (an independent copy for the test): elements
    el (E,4) with a triangle's first node repeated; depths and zbar positive down."""
    zb = -np.abs(np.asarray(zbar, float))
    nl = len(zb)
    Z = 0.5 * (zb[:-1] + zb[1:])
    x = np.minimum(-np.abs(np.asarray(depth, float)), zb[thers - 1])
    E = len(el)
    nv = np.where(el[:, 3] == el[:, 0], 3, 4)
    side = {}
    for e in range(E):
        for k in range(nv[e]):
            key = tuple(sorted((int(el[e, k]), int(el[e, (k + 1) % nv[e]]))))
            side.setdefault(key, []).append(e)
    nb = [[] for _ in range(E)]
    for es in side.values():
        if len(es) == 2:
            nb[es[0]].append(es[1])
            nb[es[1]].append(es[0])
    lev = np.zeros(E, int)
    for e in range(E):
        dmean = x[el[e, :nv[e]]].mean()
        lev[e] = next((nz + 1 for nz in range(nl - 1) if Z[nz] < dmean), nl)
        lev[e] = max(lev[e], thers)
    for nz in range(thers + 1, nl + 1):
        again = True
        while again:
            again = False
            for e in range(E):
                if lev[e] >= nz and sum(lev[j] >= nz for j in nb[e]) < 2:
                    lev[e] = nz - 1
                    again = True
    nlv = np.zeros(int(el.max()) + 1, int)
    for e in range(E):
        for k in range(nv[e]):
            nlv[el[e, k]] = max(nlv[el[e, k]], lev[e])
    return lev, nlv, np.abs(zb)[np.array([nlv[el[e, :nv[e]]].min() for e in range(E)]) - 1]


def test_model_levels_follow_fesom_ini():
    """B250: meshom.io.fesom.model_levels is FESOM3's level finder (fesom_ini, no cavity, the element's depth the
    mean of its nodes'): a node shallower than zbar[4] counts at zbar[4]; an element's level is the level surface
    nearest the mean of its nodes' depths; a deep element with fewer than two edge neighbours as deep drops to their
    level; a node takes the deepest level of its elements and an element's bottom is the shallowest of its nodes'
    levels. Hand cases, and random depths on a mixed mesh against fesom_ini's loop element by element."""
    from meshom.io.fesom import model_levels
    from meshom.mesh import Mesh

    zb = np.array(ZFINE)
    m = structured_patch(6, 3, DX)
    n = 7
    col, row = np.arange(m.n_nodes) % n, np.arange(m.n_nodes) // n
    d = np.full(m.n_nodes, 33.0)                         # raised to zbar[4] = 40 m (its nearest surface was 30 m)
    lev, nlv, bot = model_levels(m, d, zb)
    assert (lev == 5).all() and (nlv == 5).all() and (bot == 40.0).all()
    d[:] = 46.0                                          # nearest surface 50 m
    _, _, bot = model_levels(m, d, zb)
    assert (bot == 50.0).all()
    d[(col == 3) & (row == 1)] = 900.0                   # one deep node: its four cells reach 259 m mean (250 m)
    lev, nlv, bot = model_levels(m, d, zb)
    four = (np.arange(18) % 6 >= 2) & (np.arange(18) % 6 <= 3) & (np.arange(18) // 6 <= 1)
    assert (zb[lev[four] - 1] == 250.0).all() and (zb[lev[~four] - 1] == 50.0).all()
    assert (zb[nlv[(col >= 2) & (col <= 4) & (row <= 2)] - 1] == 250.0).all()     # every node of the four cells
    assert (bot[four] == 250.0).all() and (bot[~four] == 50.0).all()             # the cells round them: 50 m
    d[:] = 46.0
    d[(col == 3) & (row == 3)] = 900.0                   # on the coast row: two cells, each with one neighbour as deep
    d[(col == 1) & (row == 0)] = 900.0
    lev, _, _ = model_levels(m, d, zb)
    assert (zb[lev - 1] <= 50.0 + 1e-9).all()            # the 2-core of each deeper layer is empty
    rng = np.random.default_rng(3)
    for k in range(4):
        tris = np.array([[0, 1, 8], [0, 8, 7], [5, 6, 13]])
        quads = np.array([q for q in m.quads if not ({0, 1, 8, 7} >= set(q) or {5, 6, 13, 12} >= set(q))])
        mm = Mesh(m.points, tris if k % 2 else tris[:0], quads, m.node_flag)
        used = np.unique(np.r_[mm.tris.ravel(), mm.quads.ravel()])
        dd = np.full(m.n_nodes, 40.0)
        dd[used] = rng.choice([25.0, 60.0, 140.0, 420.0, 610.0, 980.0, 2600.0], len(used), p=[.2, .2, .2, .15, .1,
                                                                                             .1, .05])
        el = np.r_[np.c_[mm.tris, mm.tris[:, :1]] if mm.n_tris else np.zeros((0, 4), int), mm.quads]
        want = _levels_reference(el, dd, zb)
        got = model_levels(mm, dd, zb)
        assert np.array_equal(got[0], want[0]) and np.array_equal(got[1][used], want[1][used])
        assert np.array_equal(got[2], want[2])


def _wide_strait(nc, rows):
    """the strait fixture's grid with a mesh of `rows` rows of 2.8 km cells across (3: 8.4 km, inside the 13.3 km
    strait; 5: 14 km, its outer nodes on land), the insert's own nodes |lon| < 0.65, the rule of the gibraltar
    example (box means, the channel raise, the grid's dam) and the fine levels ZFINE."""
    from meshom.bathymetry import node_depth
    from meshom.config import loads

    m = structured_patch(64, rows, DX)
    lo, _ = xyz_to_lonlat(m.points)
    cfg = loads(f'[mesh]\nmethod = "m00_background"\nformat = "fesom2"\n[bathymetry]\nsource = "{nc}"\n'
                f'levels = {ZFINE}\nmin_layers = 4\n')
    base, _ = node_depth(m, nc, "elevation", 40.0)
    return m, np.abs(lo) < 0.65, cfg, base


def _section_areas(m, d, nc, lon_at):
    """(mesh area in FESOM3's element bottoms, the grid's wet area) on the meridian at lon_at across the strait, km^2,
    computed here without meshom.insert.area: samples every 100 m from 9 km S to 9 km N."""
    import netCDF4

    from meshom.io.fesom import model_levels
    from meshom.straits import locate

    y = np.arange(-9000.0, 9000.1, 100.0)
    q = lonlat_to_xyz(np.full(len(y), lon_at), np.degrees(y / R_EARTH_M))
    e = locate(m, q)
    bot = model_levels(m, d, ZFINE)[2]
    mesh = float(np.where(e >= 0, bot[np.maximum(e, 0)], 0.0).sum() * 100.0)
    with netCDF4.Dataset(nc) as ds:
        glon, glat, z = ds["lon"][:], ds["lat"][:], ds["elevation"][:]
    j = int(np.argmin(np.abs(glon - lon_at)))
    i = np.clip(np.rint((np.degrees(y / R_EARTH_M) - glat[0]) / D).astype(int), 0, len(glat) - 1)
    grid = float(np.maximum(-z[i, j], 0.0).sum() * 100.0)
    return mesh / 1e6, grid / 1e6


def _deepest_connection(m, bot):
    """the deepest connection (m) in element bottoms between the westernmost and the easternmost cells of `m`:
    cells added deepest first (edge neighbours joined) until the two are joined."""
    from meshom.mesh import build_topology

    ee = build_topology(m).edge_elems
    ee = ee[ee[:, 1] >= 0]
    lo = xyz_to_lonlat(np.r_[m.points[m.tris].mean(1) if m.n_tris else np.zeros((0, 3)),
                             m.points[m.quads].mean(1)])[0]
    w, e = int(np.argmin(lo)), int(np.argmax(lo))
    par = np.arange(len(bot))

    def root(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    on = np.zeros(len(bot), bool)
    nb = [[] for _ in range(len(bot))]
    for a, b in ee:
        nb[a].append(b)
        nb[b].append(a)
    for i in np.argsort(-bot, kind="stable"):
        on[i] = True
        for j in nb[i]:
            if on[j]:
                par[root(i)] = root(j)
        if on[w] and on[e] and root(w) == root(e):
            return float(bot[i])
    return 0.0


@pytest.mark.parametrize("rows", [3, 5])
def test_depth_area_brings_the_sections_to_the_grids_area(strait, rows):
    """B250: with depth_area the insert's nodes are scaled so that each section across the strait has the grid's
    wet area in FESOM3's element bottoms: the 3-row mesh (8.4 km in the 13.3 km strait) is deepened, the 5-row mesh
    (14 km, its outer nodes on land) made shallower; sections checked here on their own (the meridians 0.30 W, 0.20 W,
    0.30 E: mesh / grid within 10 % and nearer than before). Only the strait's sections count (the basins beyond
    0.45 deg are open water past the mesh's banks). The dam of sill_level = "grid" holds its cap: none of its nodes
    goes deeper, its cells' bottoms stay at the saddle's level or above; the deepest connection between the basins in
    the element bottoms stays in the sill band of B251 (275-325 m round the saddle's 300 m level), here at 300 m; no
    host node changes, no node rises above the floor; the factor changes smoothly along the axis; the record names
    the counted and skipped sections and the deviation before and after."""
    nc = strait[0]
    m, ins, cfg, base = _wide_strait(nc, rows)
    from meshom.insert.core import rule_depth
    from meshom.io.fesom import model_levels

    axis = lonlat_to_xyz(np.linspace(-0.8, 0.8, 33), np.zeros(33))
    kw = dict(depth_at="node", depth_smooth=0, strait_floor=False, sill_level="grid", depth_channel="near", h_fine=DX,
              log=QUIET)
    d0, r0 = rule_depth(m, ins, base, cfg, "fesom2", None, **kw)
    d1, r1 = rule_depth(m, ins, base, cfg, "fesom2", None, depth_area=True, axis=axis, **kw)
    a = r1["area"]
    assert a["applied"] and r1["rule"]["depth_area"] and "area" not in r0 and "held" not in r1["sill"]
    assert a["counted"] >= 50 and a["skipped"]["the grid's water runs on past a mesh bank"] > 0
    ok = [c for c in a["curve"] if c["ok"]]
    assert all(abs(c["lonlat"][0]) <= 0.4501 for c in ok)                     # the strait's sections only
    assert 12.0 < a["narrows"]["width_min_km"] < 14.0
    assert a["after"]["mean"] < 0.10 < a["before"]["mean"] and a["after"]["within"] > a["before"]["within"] + 30
    for lon_at in (-0.30, -0.20, 0.30):
        mb, g = _section_areas(m, d0, nc, lon_at)
        ma, _ = _section_areas(m, d1, nc, lon_at)
        assert abs(ma / g - 1.0) < min(0.10, abs(mb / g - 1.0) - 0.03), (lon_at, mb, ma, g)
        assert (ma > mb) == (rows == 3)
    assert np.array_equal(d1[~ins], d0[~ins]) and (d1[ins] >= 40.0 - 1e-9).all()
    cap = r1["sill"]["cap_m"]
    assert a["dam"]["cap_m"] == cap and a["dam"]["nodes"] > a["dam"]["kept_at_cap"] > 0
    capped = ins & (np.abs(d0 - cap) < 1e-9)
    assert capped.sum() == a["dam"]["kept_at_cap"] and (d1[capped] <= cap).all()
    assert a["sill"]["band_m"] == [275.0, 325.0] and a["sill"]["connection_m"] == 300.0
    lo = xyz_to_lonlat(m.points)[0]
    lo_c = xyz_to_lonlat(m.points[m.quads].mean(1))[0]
    dam = np.abs(lo_c - r1["sill"]["via"][0]) < 0.5 * DX / 111e3 + 1e-3        # the cells on both sides of the chain
    near = ins & (np.abs(lo - r1["sill"]["via"][0]) < 1.5 * DX / 111e3)
    assert (d1[near] <= cap + 1e-9).all()
    bot0, bot1 = model_levels(m, d0, ZFINE)[2], model_levels(m, d1, ZFINE)[2]
    assert dam.sum() >= rows and (bot1[dam] <= r1["sill"]["level_m"]).all()
    assert a["route_nodes"] > 0 and _deepest_connection(m, bot1) == _deepest_connection(m, bot0) == 300.0
    assert a["factor"]["max_step"] < 0.35 and a["nodes_changed"] > 50


def _narrow_strait(tmp_path_factory):
    """a strait of 7.8 km (150 m, its channel 350 m, the 300 m sill at 0.10-0.12 E) under the 5-row mesh of 14 km:
    the sections are about twice the grid's."""
    def narrow(z, LO, LA):
        st = (np.abs(LO) < 0.45) & (np.abs(LA) < 0.06)
        z[st] = 50.0
        z[(np.abs(LO) < 0.45) & (np.abs(LA) < 0.035)] = -150.0
        z[(np.abs(LO) < 0.45) & (np.abs(LA) < 0.013)] = -350.0
        z[(LO > 0.10) & (LO < 0.12) & (np.abs(LA) < 0.013)] = -300.0

    nc = _variant(tmp_path_factory, "narrow", narrow)
    return (nc,) + _wide_strait(nc, 5)


def test_depth_area_lets_the_sill_rise_one_level(tmp_path_factory):
    """B251: on the narrow strait (sections about twice the grid's) the dam's nodes and the channel line's cells are
    scaled as the others, so the sections come nearer the grid's (the meridians 0.30 W, 0.20 W, 0.30 E within 20 %;
    holding them, as before, left 26 %), while the deepest connection in FESOM3's element bottoms rises by one level
    and no more (the saddle's 300 m level -> 275 m, the band 275-325 m): a chain of cells from join to join keeps its
    cells' mean depth at the midpoint of 250 and 275 m plus 1 m. The dam's nodes are made shallower, none goes deeper
    than the cap; the record names the band, the connection before and after and the chain."""
    from meshom.insert.core import rule_depth
    from meshom.io.fesom import model_levels

    nc, m, ins, cfg, base = _narrow_strait(tmp_path_factory)
    axis = lonlat_to_xyz(np.linspace(-0.8, 0.8, 33), np.zeros(33))
    kw = dict(depth_at="node", depth_smooth=0, strait_floor=False, sill_level="grid", depth_channel="near", h_fine=DX,
              log=QUIET)
    d0, r0 = rule_depth(m, ins, base, cfg, "fesom2", None, **kw)
    d1, r1 = rule_depth(m, ins, base, cfg, "fesom2", None, depth_area=True, axis=axis, **kw)
    for lon_at in (-0.30, -0.20, 0.30):
        mb, g = _section_areas(m, d0, nc, lon_at)
        ma, _ = _section_areas(m, d1, nc, lon_at)
        assert mb / g > 1.5 and abs(ma / g - 1.0) < 0.20, (lon_at, mb, ma, g)
    bot0, bot1 = model_levels(m, d0, ZFINE)[2], model_levels(m, d1, ZFINE)[2]
    assert _deepest_connection(m, bot0) == 300.0 and _deepest_connection(m, bot1) == 275.0
    a, s = r1["area"], r1["area"]["sill"]
    assert r1["sill"]["saddle_m"] == pytest.approx(300.0) and a["factor"]["min"] < 0.5
    assert s["level_m"] == 300.0 and s["band_m"] == [275.0, 325.0] and s["connection_in_m"] == 300.0
    assert s["connection_m"] == 275.0 and s["chain_cells"] >= 46 and s["chain_mean_m"] == 263.5
    cap = r1["sill"]["cap_m"]
    capped = ins & (np.abs(d0 - cap) < 1e-9)
    near = ins & (np.abs(xyz_to_lonlat(m.points)[0] - r1["sill"]["via"][0]) < 1.5 * DX / 111e3)   # the dam's nodes
    assert s["dam_nodes_shallower"] > 0 and (d1[capped] < cap - 1.0).any() and (d1[near] <= cap + 1e-9).all()
    assert np.array_equal(d1[~ins], d0[~ins])


def test_depth_area_sill_band_refuses_rounds_that_leave_it(tmp_path_factory, monkeypatch):
    """B251: without the chain (none found) every round is checked: a round whose deepest connection leaves the band
    is not kept, so the kept depth still has its connection at 275 m or deeper, never two levels up (250 m)."""
    from meshom.insert import area
    from meshom.insert.core import rule_depth
    from meshom.io.fesom import model_levels

    nc, m, ins, cfg, base = _narrow_strait(tmp_path_factory)
    axis = lonlat_to_xyz(np.linspace(-0.8, 0.8, 33), np.zeros(33))
    monkeypatch.setattr(area, "_chain", lambda *a, **k: None)
    d1, r1 = rule_depth(m, ins, base, cfg, "fesom2", None, depth_area=True, axis=axis, depth_at="node",
                        depth_smooth=0, strait_floor=False, sill_level="grid", depth_channel="near", h_fine=DX,
                        log=QUIET)
    s = r1["area"]["sill"]
    assert s["chain_cells"] == 0 and s["rounds_out_of_band"] > 0
    assert any(not h["in_band"] and h["connection_m"] < 275.0 for h in r1["area"]["rounds"])
    assert s["connection_m"] >= 275.0 and _deepest_connection(m, model_levels(m, d1, ZFINE)[2]) >= 275.0


def test_depth_area_sill_band_ties_and_a_sill_already_shallow(tmp_path_factory):
    """B251: the band's lower edge is one level above the level nearest the grid's saddle, a tie taking the deeper
    level (a saddle of 287.5 m between 275 and 300 m: 275 m, not 250 m), its upper edge one level below the nearest
    with a tie taking the shallower (300 m). When the input's deepest connection already lies above the band (a
    saddle of 400 m: the band 375-425 m, the input's connection 300 m) B250's holds stay: the depth equals the rule's
    without a saddle, node for node."""
    from meshom.insert.area import area_depth
    from meshom.insert.core import rule_depth
    from meshom.straits import locate

    nc, m, ins, cfg, base = _narrow_strait(tmp_path_factory)
    axis = lonlat_to_xyz(np.linspace(-0.8, 0.8, 33), np.zeros(33))
    d0, _ = rule_depth(m, ins, base, cfg, "fesom2", None, depth_at="node", depth_smooth=0, strait_floor=False,
                       sill_level="grid", depth_channel="near", h_fine=DX, log=QUIET)
    route = np.zeros(m.n_nodes, bool)
    el = locate(m, lonlat_to_xyz(np.linspace(-0.44, 0.44, 200), np.zeros(200)))
    route[m.quads[el[el >= 0] - m.n_tris].ravel()] = True
    kw = dict(held=None, route=route, log=QUIET)
    z = np.array(ZFINE)
    _, tie = area_depth(m, d0, ins, axis, z, 40.0, nc, "elevation", DX, saddle_m=287.5, **kw)
    assert tie["sill"]["band_m"] == [275.0, 300.0] and tie["sill"]["connection_in_m"] == 300.0
    d_old, r_old = area_depth(m, d0, ins, axis, z, 40.0, nc, "elevation", DX, **kw)
    d_deep, r_deep = area_depth(m, d0, ins, axis, z, 40.0, nc, "elevation", DX, saddle_m=400.0, **kw)
    assert "sill" not in r_old and "B250's holds" in r_deep["sill"]["skipped"] and "375-425" in r_deep["sill"]["skipped"]
    assert np.array_equal(d_deep, d_old) and (d_old[route & ins] >= d0[route & ins] - 1e-9).all()


def test_depth_area_connection_runs_along_the_line():
    """B251: the strait's connection is sought only through the cells within two fine cells of the channel line:
    on a 20 x 9 patch whose row 0 is 500 m deep and whose middle row (the line) is 100 m, the connection between the
    two end columns is 500 m through every cell and 100 m through the line's cells (no stored zero counts as an
    edge); the cheapest chain there follows the line."""
    from meshom.insert import area

    m = structured_patch(20, 9, DX)
    lo, la = xyz_to_lonlat(m.points[m.quads].mean(1))
    row = np.rint((la - la.min()) / (DX / 111.2e3)).astype(int)
    col = np.rint((lo - lo.min()) / (DX / 111.2e3)).astype(int)
    bot = np.where(row == 0, 500.0, np.where(row == 4, 100.0, 40.0))
    ea, eb = np.flatnonzero(col == 0), np.flatnonzero(col == col.max())
    line = lonlat_to_xyz(np.linspace(lo.min(), lo.max(), 40), np.full(40, la[row == 4].mean()))
    adj = area._adjacency(m)
    ladj = area.line_adjacency(m, adj, [line], DX, np.r_[ea[row[ea] == 4], eb[row[eb] == 4]])
    assert area.connection(adj, bot, ea, eb) == 500.0
    assert area.connection(ladj, bot, ea[row[ea] == 4], eb[row[eb] == 4]) == 100.0
    ch = area._chain(ladj, bot >= 100.0, ea, eb, np.where(row == 4, 1.0, area.AREA_CHAIN_COST))
    assert ch is not None and (row[ch] == 4).all() and len(ch) == col.max() + 1


def test_depth_area_counts_only_the_strait_narrows(tmp_path_factory):
    """B250: a section counts only on the strait's narrows (its wet crossing at most 6 times the narrowest one):
    with basins 82 km wide (more than 6 x the 13.3 km strait) closed by land within the mesh's 84 km, the basins'
    sections are complete crossings but skipped as outside the narrows, and the strait's count."""
    from meshom.bathymetry import node_depth
    from meshom.insert.area import area_depth

    lon = np.arange(-1.0, 1.0, D) + D / 2
    lat = np.arange(-0.6, 0.6, D) + D / 2
    LO, LA = np.meshgrid(lon, lat)
    z = np.full(LO.shape, 50.0)
    z[(np.abs(LO) >= 0.45) & (np.abs(LA) < 0.37)] = -400.0
    z[(np.abs(LO) < 0.45) & (np.abs(LA) < 0.06)] = -150.0
    nc = _write_grid(tmp_path_factory.mktemp("bays") / "bays.nc", lon, lat, z)
    m = structured_patch(64, 30, DX)
    lo, _ = xyz_to_lonlat(m.points)
    ins = np.abs(lo) < 0.7
    d0, _ = node_depth(m, nc, "elevation", 40.0)
    axis = lonlat_to_xyz(np.linspace(-0.8, 0.8, 33), np.zeros(33))
    d1, a = area_depth(m, d0, ins, axis, np.array(ZFINE), 40.0, nc, "elevation", DX, log=QUIET)
    cur = a["curve"]
    assert a["skipped"]["outside the strait's narrows"] > 5 and a["counted"] > 20
    assert all(c["why"] == "outside the strait's narrows" for c in cur if 0.47 < abs(c["lonlat"][0]) < 0.68)
    assert all(c["ok"] for c in cur if abs(c["lonlat"][0]) < 0.4)


def test_depth_area_skipped_without_an_axis_and_read_from_the_config(strait):
    """B250: an insert without an axis (no profile: a strip or a corridor) keeps the other rules' depth and says so;
    `depth_area` is read from an [[inserts]] entry into the insert's Spec; the axis is the profile's centreline
    between the two joins."""
    from types import SimpleNamespace

    from meshom.config import InsertCfg, _build
    from meshom.insert.core import insert_axis, rule_depth, spec_from
    from meshom.insert.geom import Chart

    nc = strait[0]
    m, ins, cfg, base = _wide_strait(nc, 3)
    kw = dict(depth_at="node", depth_smooth=0, strait_floor=False, h_fine=DX, log=QUIET)
    d0, _ = rule_depth(m, ins, base, cfg, "fesom2", None, **kw)
    d1, r1 = rule_depth(m, ins, base, cfg, "fesom2", None, depth_area=True, axis=None, **kw)
    assert np.array_equal(d0, d1) and r1["area"]["applied"] is False and "no axis" in r1["area"]["skipped"]
    e = _build(InsertCfg, {"name": "t", "path": [[0.0, 0.0], [1.0, 0.0]], "depth_area": True}, "inserts[0]")
    assert spec_from(e).depth_area is True and spec_from(_build(InsertCfg, {"path": [[0.0, 0.0], [1.0, 0.0]]},
                                                                     "x")).depth_area is False
    ch = Chart(lonlat_to_xyz(np.array([0.0]), np.array([0.0]))[0])
    xy = np.c_[np.linspace(-50e3, 50e3, 11), np.zeros(11)]
    res = SimpleNamespace(profile=SimpleNamespace(s=np.linspace(0, 100e3, 11), xy=xy), chart=ch,
                          joins=[SimpleNamespace(s_m=72e3), SimpleNamespace(s_m=18e3)])
    ax = insert_axis(res)
    assert np.allclose(ax, ch.inv(xy[2:8]))
    assert insert_axis(SimpleNamespace(profile=None, chart=ch, joins=[])) is None
# ------------------------------------------------------------------------------------------------ B223

@pytest.fixture(scope="module")
def depth_run_dir(strait, nest_run, tmp_path_factory):
    """A run folder of the nest_run's lattice mesh (raw.npz; config.resolved.toml on the strait's grid, 4 smoothing
    passes, a constant size layer, no model files)."""
    from meshom import compete, config

    d = tmp_path_factory.mktemp("depth_run")
    m = nest_run[1]
    compete._save(m, d / "raw.npz")
    cfg = _cfg(strait[0], smooth_passes=4)
    cfg = dataclasses.replace(cfg, size=dataclasses.replace(cfg.size, layers=[config.ConstantLayer(h_m=2800.0)]))
    config.save(cfg, d / "config.resolved.toml")
    return d, m


def _depth_entry(**kw):
    from meshom.config import InsertCfg, _build

    return _build(InsertCfg, {"name": "dep", "kind": "depth", "path": [[-0.6, 0.0], [0.6, 0.0]], "width_km": 20.0}
                  | kw, "insert")


def test_a_rule_failing_after_the_joins_is_a_refusal_record(depth_run_dir, tmp_path, monkeypatch):
    """B254: an InsertError raised while an insert is built (the joins do not close the strait: a second channel
    round a forced join) is written as a refused record and the command exits with 2, not with the exception."""
    from meshom.config import InsertCfg, _build, load
    from meshom.insert import core
    from meshom.insert.__main__ import main
    from meshom.insert.region import InsertError

    d, m = depth_run_dir

    def fails(*a, **k):
        raise InsertError("the joins do not close the strait: the host cells between them reach the window's edge "
                          "(a coast gap or a second channel around the joins)")
    monkeypatch.setattr(core, "insert_window", fails)
    e = _build(InsertCfg, {"name": "two", "path": [[-0.6, 0.0], [0.6, 0.0]], "cells": 5, "force": True,
                           "coast": "host"}, "insert")
    rec = core.run(d, core.spec_from(e, load(d / "config.resolved.toml")), tmp_path / "a", log=QUIET)
    assert rec["status"] == "refused" and "do not close" in rec["refused"][0]
    assert json.loads((tmp_path / "a" / "insert_two.json").read_text())["status"] == "refused"
    assert not (tmp_path / "a" / "insert_two.npz").exists()
    code = main(["--run", str(d), "--insert", "name=two; path=[[-0.6, 0.0], [0.6, 0.0]]; cells=5; force=true; "
                 "coast='host'", "--out", str(tmp_path / "b")])
    assert code == 2


def test_depth_region_on_the_plain_host(strait, depth_run_dir, tmp_path):
    """B223: kind = "depth" leaves the mesh as it is; the host nodes within width_km / 2 of the path (or inside the
    polygon) take the nest's grid rule (the box mean per node, no smoothing, no strait floor, at least the floor; with
    depth_channel the raise on the narrows and with sill_level the dam at the grid's saddle), every other node keeps
    the run's depth (on an insert's result its saved depth); insert_<name>.npz is the host mesh,
    fesom-insert_<name>/depth.npy the depth."""
    from meshom import compete
    from meshom.bathymetry import node_depth
    from meshom.config import load
    from meshom.insert.core import region_depth, run, spec_from
    from meshom.run import model_depth

    nc = strait[0]
    d, m = depth_run_dir
    cfg = load(d / "config.resolved.toml")
    lo, la = xyz_to_lonlat(m.points)
    off = np.maximum(np.abs(lo) - 0.6, 0.0)                     # the distance to the path (on the equator), deg
    inside = np.hypot(off, la) * np.radians(1.0) * R_EARTH_M <= 10e3
    assert 0 < inside.sum() < m.n_nodes
    run_d = model_depth(m, cfg, "fesom2", [], d)
    box, _ = node_depth(m, nc, "elevation", 50.0)

    rec = run(d, spec_from(_depth_entry(), cfg), tmp_path / "plain", log=QUIET)
    assert rec["status"] == "ok" and rec["region"]["nodes"] == inside.sum()
    out = compete._load(tmp_path / "plain" / "insert_dep.npz")
    assert np.array_equal(out.points, m.points) and np.array_equal(out.quads, m.quads)
    assert np.array_equal(np.load(tmp_path / "plain" / "insert_dep.host_to_new.npy"), np.arange(m.n_nodes))
    dep = np.load(tmp_path / "plain" / "fesom-insert_dep" / "depth.npy")
    assert np.array_equal(dep[~inside], run_d[~inside])
    assert np.array_equal(dep[inside], np.maximum(box[inside], 50.0)) and not np.allclose(dep[inside], run_d[inside])
    assert rec["depth"]["rule"]["smooth_passes"] == 0 and rec["depth"]["rule"]["strait_floor"] is False

    rec = run(d, spec_from(_depth_entry(depth_channel="near", sill_level="grid"), cfg), tmp_path / "chan", log=QUIET)
    dep = np.load(tmp_path / "chan" / "fesom-insert_dep" / "depth.npy")
    exp, _ = region_depth(m, inside, run_d, cfg, "fesom2", "near", "grid", log=QUIET)
    assert np.array_equal(dep, exp) and np.array_equal(dep[~inside], run_d[~inside])
    assert rec["depth"]["channel"]["raise"]["nodes_raised"] > 0 and rec["depth"]["sill"]["capped"] > 0

    poly = [[-0.71, -0.09], [0.71, -0.09], [0.71, 0.09], [-0.71, 0.09]]      # a polygon instead of the path's width
    rec = run(d, spec_from(_depth_entry(path=[], width_km=None, polygon=poly), cfg), tmp_path / "poly", log=QUIET)
    dep = np.load(tmp_path / "poly" / "fesom-insert_dep" / "depth.npy")
    inp = (np.abs(lo) < 0.71) & (np.abs(la) < 0.09)
    assert rec["status"] == "ok" and rec["region"]["nodes"] == inp.sum() and rec["region"]["width_km"] is None
    assert np.array_equal(dep[~inp], run_d[~inp]) and np.array_equal(dep[inp], np.maximum(box[inp], 50.0))

    host = tmp_path / "chain"                                   # B217: on an insert's result its saved depth stays
    (host / "fesom-insert_prev").mkdir(parents=True)
    compete._save(m, host / "insert_prev.npz")
    np.save(host / "fesom-insert_prev" / "depth.npy", np.full(m.n_nodes, 777.0))
    rec = run(d, spec_from(_depth_entry(), cfg), host, str(host / "insert_prev.npz"), log=QUIET)
    dep = np.load(host / "fesom-insert_dep" / "depth.npy")
    assert rec["host_depth"]["nodes"] == m.n_nodes
    assert (dep[~inside] == 777.0).all() and np.array_equal(dep[inside], np.maximum(box[inside], 50.0))


# ------------------------------------------------------------------------------------------------ B248, B249

def test_depth_area_crossing_leaves_out_water_beyond_land_past_the_bank(tmp_path_factory):
    """B248: the strait (13.3 km) with a land strip 1.1 km wide on its north bank and open water (100 m) beyond it;
    the 5-row mesh (14 km) reaches past the north bank onto the strip. The grid's crossing takes only the wet runs
    the mesh's run itself meets: the water beyond the strip, met only in the widening by one cell, is other water.
    The strait's sections count (before: skipped, the grid's water running on past the bank), with the strait's
    own area (150 m over the strait and 600 m over its channel on the grid's cells: 3.61 km^2, off the sill)."""
    from meshom.insert.area import area_depth

    def strip(z, LO, LA):
        z[(LA >= 0.07) & (np.abs(LO) < 0.45)] = -100.0

    nc = _variant(tmp_path_factory, "strip", strip)
    m, ins, cfg, base = _wide_strait(nc, 5)
    axis = lonlat_to_xyz(np.linspace(-0.8, 0.8, 33), np.zeros(33))
    _, a = area_depth(m, base, ins, axis, np.array(ZFINE), 40.0, nc, "elevation", DX, log=QUIET)
    ok = [c for c in a["curve"] if c["ok"]]
    mid = [c for c in a["curve"] if abs(c["lonlat"][0]) < 0.35]
    assert a["counted"] >= 40 and all(c["ok"] for c in mid)
    far = [c["gebco_km2"] for c in ok if abs(c["lonlat"][0]) < 0.35 and not 0.09 < c["lonlat"][0] < 0.13]
    assert len(far) > 40 and all(3.4 < g < 3.7 for g in far)       # the water beyond the strip: + 0.24 km^2
    assert a["skipped"].get("the grid's water runs on past a mesh bank", 0) < 0.5 * a["counted"]


def test_grid_sill_nearest_takes_the_level_nearest_the_saddle(strait, line):
    """B249: sill_level = "nearest": the dam holds the level surface nearest the grid's saddle (a saddle of 330 m
    lies nearer 340 m than 280 m: L = 340 m, the cap the midpoint to 400 m less 1 m); "grid" keeps the deepest level
    not below it (280 m); a saddle of 300 m takes 280 m either way; rule_depth reads "nearest" as a dam of its own."""
    import dataclasses as dc

    from meshom.insert import channel as chan
    from meshom.insert.core import rule_depth

    _, _, _, m, ins = strait
    base, grid, ln = line
    dep = np.where(ins, 600.0, base)
    l330 = dc.replace(ln, saddle=(330.0, ln.saddle[1], ln.saddle[2]))
    _, g = chan.grid_sill(m, dep, ins, l330, np.array(ZBAR))
    _, n = chan.grid_sill(m, dep, ins, l330, np.array(ZBAR), nearest=True)
    _, n300 = chan.grid_sill(m, dep, ins, ln, np.array(ZBAR), nearest=True)
    assert (g["level_m"], g["rule"]) == (280.0, "grid") and (n["level_m"], n["rule"]) == (340.0, "nearest")
    assert n["cap_m"] == 0.5 * (340.0 + 400.0) - chan.SILL_MARGIN_M and n300["level_m"] == 280.0
    nc = strait[0]
    _, r = rule_depth(m, ins, base, _cfg(nc), "fesom2", None, depth_at="node", depth_smooth=0, strait_floor=False,
                      sill_level="nearest", h_fine=DX, log=QUIET)
    assert r["sill"]["rule"] == "nearest" and r["sill"]["level_m"] == 280.0 and r["sill"]["capped"] > 0


# ------------------------------------------------------------------------------------------------ B223, B250

def test_depth_region_with_depth_area(strait, depth_run_dir, tmp_path):
    """B223, B250: kind = "depth" with depth_area: the region's nodes take the section-area rule along the region's
    path (its axis) after the grid rule and the dam; region_depth gives the same depth; host nodes outside the
    region keep the run's depth."""
    from meshom.config import load
    from meshom.insert.core import region_depth, run, spec_from
    from meshom.run import model_depth

    d, m = depth_run_dir
    cfg = load(d / "config.resolved.toml")
    lo, la = xyz_to_lonlat(m.points)
    off = np.maximum(np.abs(lo) - 0.6, 0.0)
    inside = np.hypot(off, la) * np.radians(1.0) * R_EARTH_M <= 10e3
    run_d = model_depth(m, cfg, "fesom2", [], d)
    spec = spec_from(_depth_entry(depth_area=True, sill_level="grid"), cfg)
    rec = run(d, spec, tmp_path / "area", log=QUIET)
    a = rec["depth"]["area"]
    assert rec["status"] == "ok" and a["applied"] and a["counted"] > 0
    dep = np.load(tmp_path / "area" / "fesom-insert_dep" / "depth.npy")
    exp, _ = region_depth(m, inside, run_d, cfg, "fesom2", "", "grid", log=QUIET, depth_area=True, axis=spec.path)
    assert np.array_equal(dep, exp) and np.array_equal(dep[~inside], run_d[~inside])


def test_depth_area_on_a_depth_region(strait, tmp_path):
    """B223 / B250: `depth_area` acts on a depth region: its nodes play the insert's own nodes and the sections run
    across the entry's path. On the 3-row mesh (8.4 km inside the 13.3 km strait, the region |lon| < 0.65) the region's
    sections come to the grid's area (checked on three meridians on their own), the dam of sill_level = "grid" holds
    (the nodes it capped keep the cap) and the deepest connection in the element bottoms stays the saddle's (300 m);
    every node off the region keeps its depth; without depth_area nothing changes. Through `run` (kind = "depth"):
    the record's `depth.area` is applied with the path as the axis, the depth equals region_depth's; a polygon alone
    has no axis (skipped, the depth as without the rule)."""
    from meshom import compete, config
    from meshom.config import load
    from meshom.insert.core import depth_region_nodes, region_depth, run, spec_from
    from meshom.io.fesom import model_levels
    from meshom.run import model_depth

    nc = strait[0]
    m, ins, cfg, base = _wide_strait(nc, 3)
    axis = lonlat_to_xyz(np.linspace(-0.8, 0.8, 33), np.zeros(33))
    d0, r0 = region_depth(m, ins, base, cfg, "fesom2", "near", "grid", log=QUIET)
    d1, r1 = region_depth(m, ins, base, cfg, "fesom2", "near", "grid", log=QUIET, depth_area=True, axis=axis)
    assert "area" not in r0 and r1["rule"]["depth_area"] and "held" not in r1["sill"]
    a = r1["area"]
    assert a["applied"] and a["counted"] >= 50 and a["after"]["mean"] < 0.10 < a["before"]["mean"]
    assert all(abs(c["lonlat"][0]) <= 0.4501 for c in a["curve"] if c["ok"])
    for lon_at in (-0.30, -0.20, 0.30):
        mb, g = _section_areas(m, d0, nc, lon_at)
        ma, _ = _section_areas(m, d1, nc, lon_at)
        assert abs(ma / g - 1.0) < min(0.10, abs(mb / g - 1.0) - 0.03) and ma > mb, (lon_at, mb, ma, g)
    assert np.array_equal(d1[~ins], base[~ins]) and np.array_equal(d0[~ins], base[~ins])
    cap = r1["sill"]["cap_m"]
    capped = ins & (np.abs(d0 - cap) < 1e-9)
    assert a["dam"]["cap_m"] == cap and capped.sum() == a["dam"]["kept_at_cap"] > 0 and (d1[capped] == cap).all()
    bot0, bot1 = model_levels(m, d0, ZFINE)[2], model_levels(m, d1, ZFINE)[2]
    assert _deepest_connection(m, bot1) == _deepest_connection(m, bot0) == 300.0

    d = tmp_path / "run"                                       # the same mesh as a run's host
    d.mkdir()
    compete._save(m, d / "raw.npz")
    cfg = dataclasses.replace(cfg, size=dataclasses.replace(cfg.size, layers=[config.ConstantLayer(h_m=2800.0)]))
    config.save(cfg, d / "config.resolved.toml")
    cfg = load(d / "config.resolved.toml")
    entry = _depth_entry(path=[[-0.6, 0.0], [0.6, 0.0]], width_km=10.0, depth_channel="near", sill_level="grid",
                         depth_area=True)
    spec = spec_from(entry, cfg)
    rec = run(d, spec, tmp_path / "area", log=QUIET)
    dep = np.load(tmp_path / "area" / "fesom-insert_dep" / "depth.npy")
    inside = depth_region_nodes(m, spec.path, spec.width_m)
    run_d = model_depth(m, cfg, "fesom2", [], d)
    exp, _ = region_depth(m, inside, run_d, cfg, "fesom2", "near", "grid", log=QUIET, depth_area=True,
                          axis=spec.path)
    assert rec["status"] == "ok" and rec["depth"]["area"]["applied"] and rec["depth"]["area"]["counted"] > 20
    assert np.array_equal(dep, exp) and np.array_equal(dep[~inside], run_d[~inside])
    poly = [[-0.71, -0.09], [0.71, -0.09], [0.71, 0.09], [-0.71, 0.09]]
    rec = run(d, spec_from(_depth_entry(path=[], width_km=None, polygon=poly, depth_area=True), cfg),
              tmp_path / "poly", log=QUIET)
    assert rec["depth"]["area"]["applied"] is False and "no axis" in rec["depth"]["area"]["skipped"]
    rec0 = run(d, spec_from(_depth_entry(path=[], width_km=None, polygon=poly), cfg), tmp_path / "poly0", log=QUIET)
    assert np.array_equal(np.load(tmp_path / "poly" / "fesom-insert_dep" / "depth.npy"),
                          np.load(tmp_path / "poly0" / "fesom-insert_dep" / "depth.npy")) and "area" not in rec0["depth"]
# ------------------------------------------------------------------------------------------------ B253

@pytest.fixture(scope="module")
def open_run_dir(tmp_path_factory):
    """A channel between submerged banks: the strait fixture's grid with its land at 150 m of water (the strait's
    channel of 600 m and its 300 m sill between banks of 150 m, the basins of 2000 m), a run folder of a 64 x 12
    patch of 2.8 km quads over it (33.6 km across: a region round the axis touches no coast), the fine levels."""
    from meshom import compete, config

    nc = _variant(tmp_path_factory, "open", lambda z, LO, LA: np.copyto(z, -150.0, where=z > 0))
    d = tmp_path_factory.mktemp("open_run")
    m = structured_patch(64, 12, DX)
    compete._save(m, d / "raw.npz")
    cfg = config.loads(f'[mesh]\nmethod = "m00_background"\nformat = "fesom2"\n[bathymetry]\nsource = "{nc}"\n'
                       f'levels = {ZFINE}\nmin_layers = 4\n')
    cfg = dataclasses.replace(cfg, size=dataclasses.replace(cfg.size, layers=[config.ConstantLayer(h_m=2800.0)]))
    config.save(cfg, d / "config.resolved.toml")
    return nc, d, m


def test_open_water_region_takes_its_joins_and_banks_from_the_path(open_run_dir, tmp_path):
    """B253: a depth region 14 km wide round the path in open water (its cells hold no coast edge): the ring of host
    nodes round it splits at its col (the shallowest ring node of the deeper way round, 150 m on the banks) into the
    joins round the deepest ring node at each end of the path (the basins), the rest of the ring is the two banks,
    left and right of the path. The channel line runs from basin to basin through the channel, the grid's saddle is
    the 300 m sill, the raise acts along the line's stretch in the region (the wet width runs on past it: B213's
    narrows would be one point), the dam reaches both banks (a chain of 7 nodes across 5 rows), and the area rule's
    sections, limited to the region, count and come nearer the grid's area. A polygon alone (no path) keeps the
    coast rule: no two joins, no line."""
    from meshom.insert.core import run, spec_from

    nc, d, m = open_run_dir
    lo, la = xyz_to_lonlat(m.points)
    off = np.maximum(np.abs(lo) - 0.6, 0.0)
    inside = np.hypot(off, la) * np.radians(1.0) * R_EARTH_M <= 7e3
    entry = _depth_entry(width_km=14.0, depth_channel="near", sill_level="grid", depth_area=True)
    rec = run(d, spec_from(entry), tmp_path / "open", log=QUIET)
    dr = rec["depth"]
    ow = dr["open_water"]
    assert rec["status"] == "ok" and rec["region"]["nodes"] == inside.sum()
    assert ow["ends"][0][0] < -0.6 and ow["ends"][1][0] > 0.6 and ow["col_m"] == pytest.approx(150.0, abs=5.0)
    assert ow["coast_nodes"] == 0 and min(ow["bank_nodes"]) > 10 and min(ow["join_nodes"]) > 3
    ch = dr["channel"]
    assert ch["ends"][0][0] < -0.45 and ch["ends"][1][0] > 0.45
    assert ch["grid_saddle"]["m"] == pytest.approx(300.0) and 0.10 < ch["grid_saddle"]["at"][0] < 0.12
    assert "open_water" in ch["narrows"] and ch["narrows"]["km"] > 100.0     # no wet width: the line in the region
    assert ch["raise"]["nodes_raised"] > 40                    # the channel's flank rows along the whole strait
    s = dr["sill"]
    assert s["saddle_m"] == pytest.approx(300.0) and s["capped"] > 0 and s["chain_nodes"] >= 7   # bank to bank
    a = dr["area"]
    assert a["applied"] and a["open_ends"] and a["counted"] > 20 and a["after"]["mean"] <= a["before"]["mean"]
    poly = [[-0.66, -0.063], [0.66, -0.063], [0.66, 0.063], [-0.66, 0.063]]
    rec = run(d, spec_from(_depth_entry(path=[], width_km=None, polygon=poly, sill_level="grid")), tmp_path / "pg",
              log=QUIET)
    assert "open_water" not in rec["depth"] and "skipped" in rec["depth"]["channel"]


def test_open_water_path_is_clipped_to_the_region(open_run_dir):
    """B253: a polygon region with a path running on 0.3 deg past both its ends: the path is clipped to its stretch
    through the region, so the joins are the ring's ends in the two basins as with the path inside, and the coast
    rule (a patch whose region reaches the patch's edge: its boundary is coast) is kept when there are two banks."""
    from meshom.bathymetry import node_depth
    from meshom.insert import channel as chan
    from meshom.size import polygon_inside

    nc, _, m = open_run_dir
    poly = lonlat_to_xyz(np.array([-0.66, 0.66, 0.66, -0.66]), np.array([-0.063, -0.063, 0.063, 0.063]))
    ins = polygon_inside(m.points, poly)
    base, _ = node_depth(m, nc, "elevation", 40.0)
    long = lonlat_to_xyz(np.array([-0.95, 0.95]), np.array([0.0, 0.0]))
    r = chan.open_sides(m, ins, base, long)
    assert not isinstance(r, str)
    (ja, _), (jb, _) = r["joins"]
    lo, la = xyz_to_lonlat(m.points)
    assert (lo[ja] < -0.4).all() and (lo[jb] > 0.4).all()
    g, ba, bb = r["sides"]
    assert (la[ba] > 0).all() and (la[bb] < 0).all() and (np.abs(lo[ba | bb]) < 0.66).all()
    grid = chan.read_window(nc, "elevation", lo, la, 8000.0)
    ln = chan.channel_line(m, ins, grid, DX, joins=r["joins"])
    deep = np.where(ins, 600.0, base)                           # every region node below the cap: the dam shows
    out, s = chan.grid_sill(m, deep, ins, ln, np.array(ZFINE), sides=r["sides"])
    capped = out < deep
    assert s["saddle_m"] == pytest.approx(300.0) and capped.sum() == s["capped"] > 0
    assert la[capped].min() < -0.05 and la[capped].max() > 0.05          # from bank to bank across the region
    assert np.abs(lo[capped] - s["via"][0]).max() < 6.0 * DX / 111e3   # a chain across, not along the channel
    wide = np.abs(la) < 0.2                                     # the whole patch's width: its edges are coast
    assert not isinstance(chan.banks(m, wide & (np.abs(lo) < 0.66)), str)
