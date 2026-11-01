"""meshom.insert.group: several channels, patches and islands in one envelope (decisions B240-B245)."""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from meshom.geo import lonlat_to_xyz
from meshom.insert import group, synth
from meshom.insert.geom import point_in_polygon
from meshom.insert.measure import validity
from meshom.mesh import build_topology

H = 20e3
ISLAND = (105e3, 27e3)                 # the separating island: an ellipse, semi-axes (m)
SMALL = (-115e3, 100e3, 8e3)           # a small island 7 km off the north land's west side: centre, radius (m)


def _ellipse(cx, cy, a, b, n=200, cw=True):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = np.c_[cx + a * np.cos(t), cy + b * np.sin(t)]
    return p[::-1] if cw else p                          # clockwise: the water on the left


def _two_channels():
    """Two basins (x < -100 km, x > 100 km) joined by two channels round an elliptic island; the north and south
    land have bumps between the channels; a small island off the north land's west end."""
    a, b = ISLAND

    def water(p):
        x, y = p[:, 0], p[:, 1]
        isl = (x / a) ** 2 + (y / b) ** 2 <= 1.0
        bump = (np.abs(x) <= 100e3) & (np.abs(y) >= 80e3)
        small = np.hypot(x - SMALL[0], y - SMALL[1]) <= SMALL[2]
        return (np.abs(y) <= 150e3) & ~isl & ~bump & ~small
    win, ch = synth.lattice_host(water, (-400e3, 400e3, -170e3, 170e3), H)
    far = 2000e3
    north = np.array([[-100e3, 80e3], [-100e3, far], [100e3, far], [100e3, 80e3]])     # clockwise
    south = np.array([[100e3, -80e3], [100e3, -far], [-100e3, -far], [-100e3, -80e3]])
    loops = [ch.inv(group._dense(np.vstack([north, north[:1]]), 1e3)[:-1]),
             ch.inv(group._dense(np.vstack([south, south[:1]]), 1e3)[:-1]), ch.inv(_ellipse(0.0, 0.0, a, b)),
             ch.inv(_ellipse(SMALL[0], SMALL[1], SMALL[2], SMALL[2], 60))]
    return win, ch, loops


@pytest.fixture(scope="module")
def grp():
    win, ch, loops = _two_channels()
    env = np.array([[-330e3, -110e3], [330e3, -110e3], [330e3, 110e3], [-330e3, 110e3]])
    chans = [group.Channel("north", ch.inv(np.array([[-80e3, 15e3], [-80e3, 95e3]])),
                           ch.inv(np.array([[80e3, 15e3], [80e3, 95e3]])), 5),
             group.Channel("south", ch.inv(np.array([[-80e3, -95e3], [-80e3, -15e3]])),
                           ch.inv(np.array([[80e3, -95e3], [80e3, -15e3]])), 5)]
    res = group.build(win, ch, env, chans, loops, h_min=2e3, gap_m=9.9e3, island_km2=10.0, growth=0.12,
                      log=lambda *a, **k: None, fill_opts={"verbose": False})
    reg = group.envelope_region(win, ch, env)
    return win, ch, env, reg, res


def test_group_is_valid_and_keeps_the_host_outside_the_envelope(grp):
    win, ch, env, reg, res = grp
    assert res.ok, res.record.get("refused") or res.record.get("invalid")
    v = validity(res.window)
    assert v["inverted"] == 0 and v["nonconvex"] == 0 and v["nonmanifold_edges"] == 0
    assert v["same_direction_edges"] == 0 and v["water_components"] == 1
    m, out = win.mesh, res.window
    T = m.n_tris
    keep = ~reg[T:]
    d, j = cKDTree(out.points).query(m.points)
    assert (d[np.unique(m.quads[keep])] < 1e-12).all()   # every node of a cell outside the envelope stays
    hq = {tuple(sorted(r)) for r in j[m.quads[keep]].tolist()}
    oq = {tuple(sorted(r)) for r in out.quads.tolist()}
    assert hq <= oq                                      # ... and every such cell
    assert res.record["region"]["joins"] >= 1 and res.record["region"]["host_islands_inside"] == 1


def test_channels_are_blocks_of_fixed_count_without_triangles(grp):
    *_, res = grp
    for c in res.record["channels"]:
        assert c["triangles"] == 0 and set(c["info"]["counts"]) == {5}
        assert c["cell_min_km"] >= 2.0


def test_the_island_is_a_hole_and_the_small_one_joins_the_land(grp):
    win, ch, env, reg, res = grp
    isl = res.record["islands"]
    assert len(isl["holes"]) == 1 and abs(isl["holes"][0]["area_km2"] - np.pi * 105 * 27) < 0.06 * np.pi * 105 * 27   # smoothed at the cell size
    assert len(isl["joined"]) == 1 and isl["joined"][0]["to"] == "coast" and isl["joined"][0]["gap_km"] < 9.9
    out = res.window
    xy = ch.fwd(out.points)
    cen = np.r_[xy[out.tris].mean(1) if out.n_tris else np.zeros((0, 2)), xy[out.quads].mean(1)]
    hole = res.holes[0]
    assert not point_in_polygon(cen, hole).any()                              # no cell on the island
    topo = build_topology(out)
    bn = np.unique(topo.edges[topo.edge_elems[:, 1] < 0])
    from meshom.insert.geom import Polyline
    assert (Polyline(np.vstack([hole, hole[:1]])).distance(xy[bn]) < 1.0).sum() >= 20   # its coast: mesh boundary
    gap = (cen[:, 0] > SMALL[0] + SMALL[2]) & (cen[:, 0] < -100e3) & (np.abs(cen[:, 1] - SMALL[1]) < 4e3)
    assert not gap.any()                                 # the water between the small island and the land closed


def test_at_least_the_asked_cells_across_each_channel(grp):
    win, ch, env, reg, res = grp
    out = res.window
    xy = ch.fwd(out.points)
    for y0, y1 in ((27e3, 80e3), (-80e3, -27e3)):
        line = np.c_[np.zeros(400), np.linspace(y0 + 1.0, y1 - 1.0, 400)]
        cells = set()
        for c in out.quads:
            if point_in_polygon(line, xy[c]).any():
                cells.add(tuple(c))
        assert len(cells) >= 5


def test_join_island_closes_the_water_between(grp):
    P = np.c_[np.linspace(0, 100e3, 101), np.zeros(101)]                     # a coast, water on its left (north)
    B = _ellipse(50e3, 12e3, 8e3, 8e3, 80)                                   # an island 4 km off it
    new = group.join_island(P, B, 15e3, closed=False)
    assert new[0].tolist() == P[0].tolist() and new[-1].tolist() == P[-1].tolist()
    assert new[:, 1].max() > 19e3                                            # round the island's far side
    land = np.vstack([new, [[100e3, -50e3], [0, -50e3]]])                   # the land below the new coast
    assert point_in_polygon(np.array([[50e3, 2e3]]), land)[0]                # the gap is land now
    assert not point_in_polygon(np.array([[50e3, 25e3]]), land)[0]           # beyond the island: water


def test_join_island_to_a_closed_loop_touching_at_one_point():
    A = _ellipse(0.0, 0.0, 50e3, 30e3, 12)                                    # a coarse island
    B = _ellipse(70e3, 0.0, 8e3, 8e3, 5)                                      # a small one ~14 km off its tip
    new = group.join_island(A, B, 16e3, closed=True)                          # only the tip point is near
    assert abs(group._shoelace(new)) > abs(group._shoelace(A))               # both kept, not the small one alone


# ------------------------------------------------------------------------------------------------ B246, B247

LEVELS = [0.0, 10.0, 20.0, 30.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 500.0, 600.0, 800.0,
          1000.0, 1500.0, 2000.0, 3000.0]


def _route_grid(path, ch):
    """A grid (1/120 deg) on the two-channel geometry: basins 2000 m; the north passage 600 m with a 250 m sill
    across it at x = 20-24 km; the south passage 400 m with a 350 m sill at x = -24 to -20 km; land +50 m. The grid's
    saddle between the basins is the south sill (350 m); the north passage is deeper on most of its length."""
    netCDF4 = pytest.importorskip("netCDF4")
    from meshom.geo import lonlat_to_xyz

    d = 1.0 / 120.0
    lon = np.arange(-4.2, 4.2, d) + d / 2
    lat = np.arange(-1.8, 1.8, d) + d / 2
    LO, LA = np.meshgrid(lon, lat)
    xy = ch.fwd(lonlat_to_xyz(LO.ravel(), LA.ravel()))
    x, y = xy[:, 0], xy[:, 1]
    a, b = ISLAND
    isl = (x / a) ** 2 + (y / b) ** 2 <= 1.0
    bump = (np.abs(x) <= 100e3) & (np.abs(y) >= 80e3)
    wet = (np.abs(y) <= 150e3) & ~isl & ~bump
    z = np.full(len(x), 50.0)
    z[wet] = -2000.0
    north, south = wet & (np.abs(x) <= 100e3) & (y > 0), wet & (np.abs(x) <= 100e3) & (y < 0)
    z[north] = -600.0
    z[north & (x >= 20e3) & (x <= 24e3)] = -250.0
    z[south] = -400.0
    z[south & (x >= -24e3) & (x <= -20e3)] = -350.0
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        ds.createVariable("lat", "f8", ("lat",))[:] = lat
        ds.createVariable("lon", "f8", ("lon",))[:] = lon
        ds.createVariable("elevation", "f4", ("lat", "lon"))[:] = z.reshape(LO.shape)
    return path


@pytest.fixture(scope="module")
def south_only(tmp_path_factory):
    """The two-channel geometry with a block on the south channel only: the north passage is a patch."""
    from meshom.config import loads

    win, ch, loops = _two_channels()
    env = np.array([[-330e3, -110e3], [330e3, -110e3], [330e3, 110e3], [-330e3, 110e3]])
    chans = [group.Channel("south", ch.inv(np.array([[-80e3, -95e3], [-80e3, -15e3]])),
                           ch.inv(np.array([[80e3, -95e3], [80e3, -15e3]])), 5)]
    res = group.build(win, ch, env, chans, loops, h_min=2e3, gap_m=9.9e3, island_km2=10.0, growth=0.12,
                      log=lambda *a, **k: None, fill_opts={"verbose": False})
    assert res.ok
    nc = _route_grid(tmp_path_factory.mktemp("route") / "route.nc", ch)
    cfg = loads(f'[mesh]\nmethod = "m00_background"\nformat = "fesom2"\n[bathymetry]\nsource = "{nc}"\n'
                f'levels = {LEVELS}\nmin_layers = 4\n')
    full = res.window
    d, _ = cKDTree(win.mesh.points).query(full.points)
    own = d > 1e-9                                        # the group's own nodes (not at a host position)
    return win, ch, res, cfg, nc, full, own


def _connection(m, bot, ch):
    """the deepest connection (m) in element bottoms between the cells west of x = -200 km and east of 200 km."""
    from meshom.mesh import build_topology as bt

    ee = bt(m).edge_elems
    ee = ee[ee[:, 1] >= 0]
    xy = ch.fwd(np.r_[m.points[m.tris].mean(1) if m.n_tris else np.zeros((0, 3)), m.points[m.quads].mean(1)])
    west, east = np.flatnonzero(xy[:, 0] < -200e3), np.flatnonzero(xy[:, 0] > 200e3)
    par = np.arange(len(bot))

    def root(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    nb = [[] for _ in range(len(bot))]
    for a, b in ee:
        nb[a].append(b)
        nb[b].append(a)
    on = np.zeros(len(bot), bool)
    for i in np.argsort(-bot, kind="stable"):
        on[i] = True
        for j in nb[i]:
            if on[j]:
                par[root(i)] = root(j)
        if on[west].any() and on[east].any():
            rw = {root(k) for k in west[on[west]]}
            if any(root(k) in rw for k in east[on[east]]):
                return float(bot[i])
    return 0.0


def test_route_dam_cuts_every_passage_beside_the_island(south_only):
    """B246: a route from basin to basin: the line, the grid's saddle between its ends (the south sill, 350 m), and
    the dam from bank to bank through it, its banks the open coast on each side of the line, so that the chain
    crosses the island and cuts the north passage (a patch, no block) too: the deepest connection between the basins
    in FESOM3's element bottoms is at the saddle's level (350 m) or above. Each channel's own dam (B244, the block
    only) leaves the north passage open below that level. Only the group's own nodes change."""
    from meshom.bathymetry import levels, node_depth
    from meshom.io.fesom import model_levels

    win, ch, res, cfg, nc, full, own = south_only
    zb = levels(cfg.bathymetry.levels)
    d0, _ = node_depth(full, nc, "elevation", 30.0)
    routes = [("strait", ch.inv(np.array([[-260e3, 0.0]]))[0], ch.inv(np.array([[260e3, 0.0]]))[0], None, None)]
    d1, recs = group.route_depth(full, d0, own, routes, cfg, "grid", "", False, 6e3, log=lambda *a, **k: None)
    r = recs[0]
    assert r["sill"]["saddle_m"] == pytest.approx(350.0) and r["sill"]["level_m"] == 350.0
    sx, sy = ch.fwd(lonlat_to_xyz(*np.array(r["sill"]["saddle_at"])[:, None]))[0]
    assert -26e3 < sx < -18e3 and sy < 0                                     # the saddle: the south sill
    assert np.array_equal(d1[~own], d0[~own])
    c0 = _connection(full, model_levels(full, d0, zb)[2], ch)
    c1 = _connection(full, model_levels(full, d1, zb)[2], ch)
    assert c0 > 350.0 and c1 <= 350.0, (c0, c1)
    ids = group.ids_in(full, res)                                            # B244: the block's dam only
    d2, _ = group.branch_depth(full, d0, ids, cfg, "grid", "", [6e3], log=lambda *a, **k: None)
    assert _connection(full, model_levels(full, d2, zb)[2], ch) > 350.0


def test_route_corridor_keeps_the_line_in_its_passage(south_only):
    """B246: the deepest line between the basins takes the deep north passage (600 m); a route with corridor_km
    round its polyline keeps its line, its saddle and its narrows in the south passage: the saddle is the south sill
    (350 m), every line point lies south of the island."""
    win, ch, res, cfg, nc, full, own = south_only
    from meshom.bathymetry import node_depth

    d0, _ = node_depth(full, nc, "elevation", 30.0)
    a, b = ch.inv(np.array([[-260e3, -50e3]]))[0], ch.inv(np.array([[260e3, -50e3]]))[0]
    q = dict(log=lambda *a, **k: None)
    _, free = group.route_depth(full, d0, own, [("free", a, b, None, None)], cfg, "", "", False, 6e3, **q)
    _, kept = group.route_depth(full, d0, own, [("kept", a, b, None, 60e3)], cfg, "", "", False, 6e3, **q)
    yf = ch.fwd(lonlat_to_xyz(*np.array(free[0]["line"]["shallowest_on_line"]["at"])[:, None]))[0][1]
    assert yf > 0                                                            # the free line: the north passage
    for r in (kept[0],):
        sx, sy = ch.fwd(lonlat_to_xyz(*np.array(r["line"]["grid_saddle"]["at"])[:, None]))[0]
        assert r["line"]["grid_saddle"]["m"] == pytest.approx(350.0) and sy < 0
        yk = ch.fwd(lonlat_to_xyz(*np.array(r["line"]["shallowest_on_line"]["at"])[:, None]))[0][1]
        assert yk < 0


def test_blend_cells_sets_the_coast_blend_at_the_joins(monkeypatch):
    """B247: the real coast blends onto a join's host end node over blend_cells join cells (default BLEND_H = 2, as a
    strait's banks): every coast run's blend lengths are blend_cells x the mean edge of the join at that end."""
    from meshom.insert import realcoast as rcm

    win, ch, loops = _two_channels()
    env = np.array([[-330e3, -110e3], [330e3, -110e3], [330e3, 110e3], [-330e3, 110e3]])
    chans = [group.Channel("south", ch.inv(np.array([[-80e3, -95e3], [-80e3, -15e3]])),
                           ch.inv(np.array([[80e3, -95e3], [80e3, -15e3]])), 5)]
    seen = {}
    orig = rcm.blend_ends
    for bl in (None, 0.5):
        got = seen.setdefault(bl, [])
        monkeypatch.setattr(rcm, "blend_ends", lambda arc, xa, xb, la, lb, got=got: got.append((la, lb)) or
                            orig(arc, xa, xb, la, lb))
        res = group.build(win, ch, env, chans, loops, h_min=2e3, gap_m=9.9e3, island_km2=10.0, growth=0.12,
                          log=lambda *a, **k: None, fill_opts={"verbose": False}, blend_cells=bl)
        assert res.ok
    assert len(seen[None]) == len(seen[0.5]) >= 2
    for (a0, b0), (a1, b1) in zip(seen[None], seen[0.5]):
        assert a1 == pytest.approx(a0 * 0.5 / rcm.BLEND_H) and b1 == pytest.approx(b0 * 0.5 / rcm.BLEND_H)
        assert 15e3 < a0 / rcm.BLEND_H < 25e3                                 # the join's edge: the host's 20 km


def test_split_obtuse_splits_patch_quads_above_the_limit():
    """B243 (split_deg): a patch quad with a corner above the limit becomes two triangles along the diagonal from
    that corner; a patch quad within the limit, a block's quad and a host quad (no own node) stay; the window part
    stays the tail of both lists; no node changes."""
    from meshom.mesh import Mesh, orient_ccw
    from meshom.quality.kernel import element_geometry

    from meshom.insert.geom import Chart
    ch = Chart(lonlat_to_xyz(np.array(0.0), np.array(0.0)))
    k = 10e3
    xy = np.array([[0, 0], [k, 0], [2 * k, 0], [3 * k, 0], [4 * k, 0],                  # 0-4
                   [0, k], [0.2 * k, k], [2 * k, k], [3 * k, k], [4 * k, k]])        # 5-9: node 6 pulled west
    quads = np.array([[0, 1, 6, 5], [1, 2, 7, 6], [2, 3, 8, 7], [3, 4, 9, 8]])
    full = orient_ccw(Mesh(ch.inv(xy), np.zeros((0, 3), np.int64), quads, np.zeros(10, np.int8)))
    window = Mesh(full.points, full.tris, full.quads[1:], full.node_flag)          # quads 1-3: the window part
    own = np.zeros(10, bool)
    own[[6, 7, 8, 9]] = True                                                        # quads 0-3 have own nodes
    blocks = [(np.array([3, 4, 8, 9]), np.array([], np.int64))]                     # quad 3: a block's cell
    qa = np.degrees(element_geometry(full).quad_angles.max(1))
    assert qa[1] > 125.0 and qa[2] < 120.0
    out, win, rec = group.split_obtuse(full, window, own, blocks, 120.0)
    assert rec["split"] == 1 and out.n_tris == 2 and out.n_quads == 3
    assert np.array_equal(out.points, full.points)
    kept = {tuple(sorted(r)) for r in out.quads.tolist()}
    assert kept == {tuple(sorted(r)) for r in full.quads[[0, 2, 3]].tolist()}
    obt = full.quads[1][np.argmax(element_geometry(Mesh(full.points, np.zeros((0, 3), np.int64), full.quads[1:2],
                                                        full.node_flag)).quad_angles[0])]
    assert all(obt in t for t in out.tris.tolist())                                 # the diagonal from that corner
    assert win.n_tris == 2 and win.n_quads == 2 and np.array_equal(win.quads, out.quads[-2:])
    assert (element_geometry(out).tri_angles.max() < np.radians(120.0))
