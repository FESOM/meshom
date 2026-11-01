"""meshom.insert.group: several channels, patches and islands in one envelope (decisions B240-B245)."""
import numpy as np
import pytest
from scipy.spatial import cKDTree

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
