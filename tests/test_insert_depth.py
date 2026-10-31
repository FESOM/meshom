"""The depth of an insert's own nodes and of the host nodes round it (decisions B210-B217), on a synthetic strait: a
bathymetry grid of two deep basins (with a trench along the axis) joined by a strait of 13 km with a deep channel of
3.7 km and a 300 m sill, and a structured mesh along it whose middle part plays the insert."""
import dataclasses

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
    return base, grid, chan.channel_line(m, ins, base, grid, DX)


# ------------------------------------------------------------------------------------------------ B213, B215

def test_channel_line_follows_the_channel_and_its_narrows_stay_in_the_strait(strait, line):
    """B213: the channel line runs between the two joins (the host pieces at both ends of the insert) along the
    deep channel; its narrows (wet width at most 6 x the narrowest, 13 km here) cover the strait and none of the open
    basins; the record names the narrowest width and the limit."""
    _, lon, lat, m, ins = strait
    base, grid, ln = line
    rec = ln.record["narrows"]
    assert abs(ln.lon[0] + 0.65) < 0.03 and abs(ln.lon[-1] - 0.65) < 0.03          # the joins' ends, in the basins
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
    run_d = model_depth(m, _cfg(nc), "fesom2", [], run_dir)            # the run's depth: the line's ends
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
