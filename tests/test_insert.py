"""meshom.insert: a curvilinear strait insert fitted into a host mesh (decisions B158-B163)."""
import numpy as np
import pytest

from meshom.config import loads, dumps
from meshom.insert import block, synth
from meshom.insert.__main__ import parse_inline
from meshom.insert.core import Spec, insert_window
from meshom.insert.geom import Polyline, smooth_bank
from meshom.insert.host import cut_window, splice
from meshom.insert.measure import validity
from meshom.mesh import Mesh, build_topology
from meshom.quality.kernel import element_geometry
from meshom.size import Uniform

H = 10e3


@pytest.fixture(scope="module")
def banana():
    win, ch, path = synth.banana(H=H, L=400e3)
    res = insert_window(win, Spec("banana", path, cells=10, stages="", bank_cap_h=10.0), Uniform(H),
                        log=lambda *a, **k: None)
    return win, res


def test_block_of_a_rectangle_is_a_regular_grid():
    xa = np.c_[np.zeros(9), np.linspace(0, 80e3, 9)]
    xb = np.c_[np.full(9, 200e3), np.linspace(0, 80e3, 9)]
    bS = Polyline(np.c_[np.linspace(0, 200e3, 50), np.zeros(50)])
    bN = Polyline(np.c_[np.linspace(0, 200e3, 50), np.full(50, 80e3)])
    b = block.build(xa, xb, bS, bN, 8, fixed_count=True)
    assert len(b.tris) == 0 and (b.N == 8).all()
    assert np.allclose(b.xy[b.off[0]:b.off[0] + 9], xa) and np.allclose(b.xy[b.off[b.M]:b.off[b.M] + 9], xb)
    m = Mesh(np.c_[b.xy / 6.371e6, np.ones(len(b.xy))], np.zeros((0, 3), np.int64), b.quads, np.zeros(len(b.xy)))
    m.points = m.points / np.linalg.norm(m.points, axis=1, keepdims=True)
    a = np.degrees(element_geometry(m).quad_angles)
    assert 88.0 < a.min() and a.max() < 92.0                  # square cells: rows by the aspect iteration


def test_y_splits_change_the_count_and_stagger():
    ids, quads, tris, vert = block.topology(np.array([10, 10, 11, 12, 12, 13, 14, 14]))
    assert len(tris) == 4
    n = ids[-1][-1] + 1
    used = np.bincount(np.r_[quads.ravel(), tris.ravel()], minlength=n)
    assert (used > 0).all()


def test_smoothed_staircase_bank_stays_within_the_cap():
    k = np.arange(12)
    stair = np.c_[np.repeat(k, 2)[1:] * H, np.repeat(k, 2)[:-1] * H]   # a 45 deg staircase of H steps
    out = smooth_bank(stair, 1.5 * H, 0.2 * H, 0.1 * H)
    fwd = Polyline(stair).distance(out)
    back = Polyline(out).distance(stair)
    assert fwd.max() <= 0.2 * H + 1e-6 and back.max() <= 0.2 * H * 1.001
    raw_turn = np.abs(np.diff(np.arctan2(*np.diff(stair, axis=0).T[::-1])))
    turn = np.abs(np.diff(np.unwrap(np.arctan2(*np.diff(out, axis=0).T[::-1]))))
    assert turn.max() < raw_turn.max()                          # smoother than the staircase


def test_banana_insert_conforms_and_keeps_the_coast(banana):
    win, res = banana
    assert res.ok, res.record.get("refused")
    jA, jB = res.joins
    assert not jA.fails and not jB.fails
    m = res.window
    for j in (jA, jB):                                          # join nodes are the host's, unmoved
        d = np.min(np.linalg.norm(m.points[None, :, :] - win.mesh.points[j.nodes][:, None, :], axis=2), axis=1)
        assert d.max() < 1e-12
    v = validity(m)
    assert v["inverted"] == 0 and v["nonconvex"] == 0 and v["nonmanifold_edges"] == 0
    assert v["water_components"] == 1
    sc = res.record["scores"]["final"]
    assert sc["coast"]["shift_h_max"] <= 0.5                    # smooth banks: 1.5 cells of Gaussian, no cap
    assert sc["core"]["invalid"] == 0 and sc["core"]["quads"] > 300
    assert sc["core"]["qmax"] < 120.0                           # the rules passed: a clean core
    assert res.record["block"]["y_splits"] == sc["core"]["tris"] + sc["ring"]["tris"]


def test_banana_zoom_follows_the_neck(banana):
    _, res = banana
    b = res.record["block"]
    assert b["counts"][0] == res.joins[0].n and b["counts"][-1] == res.joins[1].n
    assert min(b["counts"]) == 10                               # the neck carries the requested cells
    assert b["h_min_m"] < 0.3 * H                               # zoom about x5 from the host


def test_coarse_host_is_refused_with_the_rule_and_a_fix():
    win, ch, path = synth.banana(H=40e3, L=400e3)
    res = insert_window(win, Spec("coarse", path, cells=10, stages="", reach_m=150e3), Uniform(40e3),
                        log=lambda *a, **k: None)
    assert not res.ok
    msg = " ".join(res.record["refused"])
    assert "room" in msg and "fix:" in msg


def test_coast_rule_against_smooth_banks_trade_off(banana):
    """On a staircase host the bank held within 0.2 h of the coast (default) keeps the coast rule but leaves the
    staircase in the bank (worse core angles) than the smooth bank (no cap)."""
    win, ch, path = synth.banana(H=H, L=400e3)
    keep = insert_window(win, Spec("keep", path, cells=10, stages=""), Uniform(H), log=lambda *a, **k: None)
    assert keep.ok
    k, s = keep.record["scores"]["final"], banana[1].record["scores"]["final"]
    assert k["coast"]["shift_h_max"] <= 0.25 < s["coast"]["shift_h_max"]
    assert k["core"]["qmax"] > s["core"]["qmax"]


def test_forced_build_with_failing_rules_records_them():
    win, ch, path = synth.butterfly(H=H)
    res = insert_window(win, Spec("bf", path, cells=16, stages="", force=True), Uniform(H), log=lambda *a, **k: None)
    assert res.ok
    assert any(j.fails for j in res.joins)                      # the rules' verdict is kept in the record
    assert any(j["fails"] for j in res.record["joins"])


def test_ring_cleaning_keeps_the_core_and_improves_the_ring():
    win, ch, path = synth.banana(H=H, L=400e3)
    res = insert_window(win, Spec("banana", path, cells=10, stages="d,b,e,a", force=True, collar_rings=3),
                        Uniform(H), log=lambda *a, **k: None)                 # B193: a collar asked on a lattice
    sc = res.record["scores"]
    assert sc["final"]["ring"]["q120"] < sc["merged"]["ring"]["q120"]
    if res.record.get("released_cells", 0) == 0:                # the core is cut out of the case: untouched
        assert sc["final"]["core"] == sc["merged"]["core"]
    else:                                                        # B166: its worst cells were released to the cleaning
        assert sc["final"]["core"]["q120"] < sc["merged"]["core"]["q120"]
    assert sc["final"]["coast"]["shift_h_max"] <= 0.25          # ref = the host coast (B162): no stacked shift
    assert validity(res.window)["inverted"] == 0


def test_window_splice_round_trip_keeps_ids():
    win, ch, path = synth.banana(H=H, L=400e3)
    host = win.mesh
    w = cut_window(host, host.points.mean(0), 80e3)
    assert (w.kind == 2).any() and (w.kind == 1).any()
    out, info = splice(host, w, w.mesh)
    assert info["appended_nodes"] == 0 and out.n_nodes == host.n_nodes
    assert np.array_equal(out.points, host.points)
    assert out.n_elems == host.n_elems


def test_inserts_table_and_inline_spec():
    cfg = loads('[mesh]\nmethod = "m00_background"\n[[inserts]]\nname = "gib"\nstrait = "Gibraltar"\ncells = 12\n'
                'mouth_a = [[-6.0, 35.8], [-6.0, 36.2]]\n')
    assert cfg.inserts[0].cells == 12 and cfg.inserts[0].mouth_a[1] == [-6.0, 36.2]
    assert loads(dumps(cfg)).inserts == cfg.inserts
    d = parse_inline("strait=Gibraltar; cells=10; path=[[-31.0, 64.5], [-22.0, 67.5]]; force=true")
    assert d == {"strait": "Gibraltar", "cells": 10, "path": [[-31.0, 64.5], [-22.0, 67.5]], "force": True}


def test_insert_h_layer_is_the_exact_cone_minimum():
    # the 16 nearest samples alone missed a farther, smaller one (h jumped, overestimated by 33 %)
    from meshom.insert.assemble import InsertH

    rng = np.random.default_rng(3)

    def patch(n):                                # points within ~100 km of (1, 0, 0), like a block's samples
        p = np.c_[np.ones(n), rng.uniform(-0.008, 0.008, (n, 2))]
        return p / np.linalg.norm(p, axis=1)[:, None]

    pts = patch(400)
    hk = rng.uniform(3e3, 5e3, 400)
    hk[:3] = 10.0                                # a few small samples: they win beyond the 16 nearest
    q = patch(300)
    lay = InsertH(lambda x: np.full(len(x), 1e9), pts, hk, 0.12)
    from meshom.geo import R_EARTH_M

    G = 0.12 * R_EARTH_M
    full = np.min(hk[None, :] + G * np.linalg.norm(q[:, None, :] - pts[None, :, :], axis=2), axis=1)
    assert np.allclose(lay(q), full, rtol=0, atol=1e-6)
    capped = InsertH(lambda x: np.full(len(x), 2e5), pts, hk, 0.12)
    assert np.allclose(capped(q), np.minimum(full, 2e5), rtol=0, atol=1e-6)


def test_mouth_hints_do_not_skip_the_rules():
    # hinted joins bypassed every refusal rule even with force off
    from meshom.insert import synth
    from meshom.size import Uniform

    win, ch = synth.lattice_host(lambda p: np.abs(p[:, 1]) <= 20000, (-150000, 150000, -70000, 70000), 10000)
    path = ch.inv(np.array([[-30000.0, 0.0], [30000.0, 0.0]]))
    kw = dict(cells=10, stages="", reach_m=80000, bank_sigma_h=0.0)
    quiet = lambda *a, **k: None                 # noqa: E731
    auto = insert_window(win, Spec("rect", path, **kw), Uniform(10000), log=quiet)
    assert not auto.ok
    cs = [min((j for j in auto.candidates if j.side == s and j.fails == ["room"]), key=lambda j: abs(j.d_m - 50000))
          for s in "AB"]
    hinted = Spec("rect_hinted", path, mouth_a=win.mesh.points[cs[0].nodes], mouth_b=win.mesh.points[cs[1].nodes],
                  **kw)
    r = insert_window(win, hinted, Uniform(10000), log=quiet)
    assert not r.ok and r.record.get("refused")


# ------------------------------------------------------------------ step 2: real coast, size floor, bays, joins

@pytest.fixture(scope="module")
def banana_real():
    win, ch, path, loops = synth.strait_real(H=H, L=400e3)
    quiet = lambda *a, **k: None                 # noqa: E731
    real = insert_window(win, Spec("real", path, cells=10, stages=""), Uniform(H), real=loops, log=quiet)
    host = insert_window(win, Spec("host", path, cells=10, stages="", coast="host"), Uniform(H), real=loops,
                         log=quiet)
    return win, real, host


def test_real_coast_banks_follow_the_real_coast_and_clean_the_core(banana_real):
    """B164: banks on the real coast (smoothed at the insert's cells), not on the host's staircase: the core is
    clean where the staircase bank held at 0.2 h leaves bad cells, and the coast shift is measured against the real
    coast."""
    win, real, host = banana_real
    assert real.ok and host.ok
    assert real.record["banks"] == ["real", "real"]
    r, h = real.record["scores"]["final"], host.record["scores"]["final"]
    assert r["coast_real"]["shift"]["h_ins_max"] <= 0.35     # the banks blend from the host join ends (B164)
    assert r["core"]["qmax"] < 110.0 < h["core"]["qmax"]
    assert r["core"]["invalid"] == 0
    # the join ends are the host's nodes, unmoved; the banks start on them exactly
    for j in real.joins:
        d = np.min(np.linalg.norm(real.window.points[None] - win.mesh.points[j.nodes][:, None], axis=2), axis=1)
        assert d.max() < 1e-12


def test_size_floor_caps_the_cells_across_and_says_so():
    win, ch, path, loops = synth.strait_real(H=H, L=400e3)
    quiet = lambda *a, **k: None                 # noqa: E731
    r = insert_window(win, Spec("floor", path, cells=10, stages="", h_min_m=3300.0), Uniform(H), real=loops,
                      log=quiet)
    p = r.record["profile"]
    W = p["neck_width_km"] * 1e3
    assert p["cells"] == int(W // 3300.0) and p["cells"] < 10
    assert "size floor" in p["floor"]
    if r.ok:
        assert r.record["block"]["h_min_m"] >= 3300.0 * 0.999
    from meshom.insert.region import InsertError

    with pytest.raises(InsertError, match="fewer than 2 cells"):
        insert_window(win, Spec("floor2", path, cells=10, stages="", h_min_m=0.6 * W), Uniform(H), real=loops,
                      log=quiet)


def test_a_bay_becomes_a_block_of_its_own():
    """B166: a deep bay on a bank is bridged by the main block and meshed as its own block (mouth = the main
    block's bank nodes, -1 corners at the mouth ends)."""
    win, ch, path, loops = synth.strait_real(H=H, L=400e3, bend=0.0, w_neck=40e3, bay=(10e3, 45e3, 50e3))
    r = insert_window(win, Spec("bay", path, cells=10, stages=""), Uniform(H), real=loops, log=lambda *a, **k: None)
    assert r.ok
    bays = [b for b in r.record["bays"] if b["status"] == "meshed"]
    assert len(bays) == 1 and bays[0]["side"] == "N" and bays[0]["cells"] >= 2
    sc = r.record["scores"]["final"]
    assert sc["core"]["invalid"] == 0
    m = r.window
    topo = build_topology(m)
    nq = np.bincount(m.quads.ravel(), minlength=m.n_nodes)
    assert ((nq == 3) & topo.boundary_node).sum() >= 2           # the two mouth ends: 3 quads at a coast node


def test_joins_prefer_host_grid_lines():
    """B167: the chain finder pays for turns, so a slightly oblique shape still gives the host column (tier 1)."""
    from meshom.insert.geom import Polyline
    from meshom.insert.host import coast_neighbours, graph
    from meshom.insert.region import Chains

    win, ch = synth.lattice_host(lambda p: np.abs(p[:, 1]) <= 40000, (-100000, 100000, -60000, 60000), 10000)
    xy = ch.fwd(win.mesh.points)
    g = graph(win.mesh)
    c = Chains(g, xy, coast_neighbours(win.coast_edges, win.mesh.n_nodes), win.mesh.quads)
    cn = np.unique(win.coast_edges)
    a = int(cn[np.argmin(np.linalg.norm(xy[cn] - [0, -40000], axis=1))])
    b = int(cn[np.argmin(np.linalg.norm(xy[cn] - [0, 40000], axis=1))])
    shape = Polyline(np.array([[-4000.0, -40000.0], [4000.0, 40000.0]]))
    p = c.path(a, b, shape)
    assert c.zig(p) == 0.0 and c.tier(p) == 1 and len(p) == 9


# ------------------------------------------------------------------ the floor and the held coast

def _made_cells_sizes(host, mesh):
    """Mean great-circle edge length of every cell of mesh that is not a host cell at the host's exact positions (the
    measure of core.host_min_cell), computed here without the module's own check."""
    from scipy.spatial import cKDTree

    d, j = cKDTree(host.points).query(mesh.points)
    hid = np.where(d < 1e-12, j, -1)
    keys = {tuple(r) for c in (host.tris, host.quads) for r in np.sort(c, axis=1).tolist()}
    g = element_geometry(mesh)
    me = np.r_[g.tri_edges.mean(1) if mesh.n_tris else np.zeros(0), g.quad_edges.mean(1)]
    made = np.array([min(r) < 0 or tuple(r) not in keys for c in (mesh.tris, mesh.quads) if len(c)
                     for r in np.sort(hid[c], axis=1).tolist()])
    return me[made]


def test_size_floor_holds_on_every_cell_the_insert_makes():
    """B165: h_min_m bounds the cells, not only the size target: the harmonic map squeezes
    the cells at the banana's inner bank to ~0.8 of a target held at the floor; the layout raises the target there."""
    win, ch, path, loops = synth.strait_real(H=H, L=400e3)
    r = insert_window(win, Spec("floor_cells", path, cells=10, stages="", h_min_m=2500.0), Uniform(H), real=loops,
                      log=lambda *a, **k: None)
    assert r.ok, r.record.get("refused")
    sizes = _made_cells_sizes(win.mesh, r.window)
    assert sizes.min() >= 2500.0                                 # no tolerance
    f = r.record["floor"]
    assert f["below"] == 0 and abs(f["smallest_m"] - sizes.min()) < 1e-6


def test_size_floor_refuses_a_build_it_cannot_lay_out(monkeypatch):
    """B165: a cell still below the floor after the layout rounds refuses the build (nothing written) and says which
    h_min_m or cells across would pass."""
    monkeypatch.setattr(block, "FLOOR_ROUNDS", 1, raising=False)  # no round may raise the target
    win, ch, path, loops = synth.strait_real(H=H, L=400e3)
    r = insert_window(win, Spec("floor_refused", path, cells=10, stages="", h_min_m=2500.0), Uniform(H), real=loops,
                      log=lambda *a, **k: None)
    assert not r.ok
    msg = " ".join(r.record["refused"])
    assert "size floor" in msg and "fix: h_min_m <=" in msg and "cells <=" in msg
    assert r.record["floor"]["below"] > 0


def test_cleaning_holds_the_joins_coast_ends_and_the_host_coast():
    """B162: with real-coast banks the cleaning slid the joins' coast end nodes (180-918 m)
    and 16 host coast nodes outside the replaced region (the old coast up to 1.3 km from the new). The coast end nodes
    and the host's coast outside the insert stay exactly in place; the inner join nodes may move (the cleaning repairs
    the join across them) and the insert's banks slide."""
    from scipy.spatial import cKDTree

    from meshom import straits as stm
    from meshom.insert import assemble

    win, ch, path, loops = synth.strait_real(H=H, L=400e3)
    r = insert_window(win, Spec("held", path, cells=10, stages="a"), Uniform(H), real=loops, log=lambda *a, **k: None)
    assert r.ok and r.record["valid"]
    tree = cKDTree(r.window.points)
    ends = [r.joins[0].nodes[0], r.joins[0].nodes[-1], r.joins[1].nodes[0], r.joins[1].nodes[-1]]
    assert tree.query(win.mesh.points[ends])[0].max() < 1e-12             # the joins' coast end nodes, exactly
    seed = int(stm.locate(win.mesh, r.chart.inv(r.profile.xy[r.profile.i0][None]), stm._elem_tree(win.mesh))[0])
    reg = assemble.region_between(win.mesh, seed, [j.nodes for j in r.joins], win.kind)
    T = win.mesh.n_tris
    inreg = np.unique(np.r_[win.mesh.tris[reg[:T]].ravel(), win.mesh.quads[reg[T:]].ravel()])
    coast = np.unique(win.coast_edges)
    outside = coast[~np.isin(coast, inreg)]
    assert tree.query(win.mesh.points[outside])[0].max() < 1e-12          # the host's coast outside, exactly
    assert r.record["clean"]["ring_sliding_nodes"] > 0                     # the insert's own banks slide
    assert r.record["host_coast"]["h_run_max"] == 0.0 and r.record["join_ends_moved"] == 0


def test_floor_rounds_never_make_the_block_coarser_than_the_host():
    """B165: the floor rounds raise the target at most to the host's size at the joins (step 2's cap);
    uncapped, Denmark's default floor (0.86 of the host's cell) raised it to 57 km, 2.2 host cells, the layout gave 3
    rows for 17 -> 14 -> 17 -> 21 cells and a quad folded. The layout keeps the rows its birth rule needs."""
    xa = np.c_[np.zeros(9), np.linspace(0, 80e3, 9)]             # host cells of 10 km on both joins
    xb = np.c_[np.full(9, 200e3), np.linspace(0, 80e3, 9)]
    bS = Polyline(np.c_[np.linspace(0, 200e3, 50), np.zeros(50)])
    bN = Polyline(np.c_[np.linspace(0, 200e3, 50), np.full(50, 80e3)])
    b = block.build(xa, xb, bS, bN, 8, h_min=0.0, cell_floor=15e3)  # a floor above the host's cell
    assert b.fine_h.max() <= 10e3 * (1 + 1e-9)
    assert b.info["cell_min_m"] < 15e3                           # the caller refuses it (core.insert_window)
    assert block.birth_rows(17, 14) + block.birth_rows(14, 21) == 4


# ------------------------------------------------------------------------------------------------- step 3

def _cell_angles(xy, cells):
    p = xy[cells]
    a = np.roll(p, -1, 1) - p
    b = np.roll(p, 1, 1) - p
    return np.degrees(np.mod(np.arctan2(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0], (a * b).sum(-1)), 2 * np.pi))


def test_an_l_bend_of_a_fixed_row_gets_a_corner_triangle():
    """B180: a join that turns a corner (an L) is cut at its bend by a triangle of the bend node and its two
    neighbours; the row next to it is zipped without the bend node, so no quad has its 90 deg corner split in two."""
    from meshom.insert.network import ccw

    leg1 = np.c_[np.zeros(11), np.linspace(100e3, 0, 11)]
    leg2 = np.c_[np.linspace(0, 100e3, 11)[1:], np.zeros(10)]
    xyA = np.vstack([leg1, leg2])
    xyB = np.c_[np.linspace(60e3, 100e3, 9), np.linspace(100e3, 60e3, 9)]
    bS = Polyline(np.c_[np.linspace(0, 60e3, 50), np.full(50, 100e3)])
    bN = Polyline(np.c_[np.full(50, 100e3), np.linspace(0, 60e3, 50)])
    assert block.bend_corners(xyA, xyB.mean(0)) == [10]
    assert block.bend_corners(np.c_[np.linspace(0, 1e5, 11), np.zeros(11)], np.array([5e4, 5e4])) == []
    b = block.build(xyA, xyB, bS, bN, 8, corners=True)
    assert b.info["corners"] == [[10], []]
    corner = [t for t in b.tris if b.off[0] + 10 in t]
    assert len(corner) == 1 and sorted(corner[0]) == [b.off[0] + 9, b.off[0] + 10, b.off[0] + 11]
    assert not any(b.off[0] + 10 in q for q in b.quads)            # the bend node is in no quad
    ang = _cell_angles(b.xy, ccw(b.xy, b.quads))
    assert ang.max() < 180.0


def test_zip_rows_gives_quads_where_rows_align_and_a_triangle_per_extra_node():
    xa = np.c_[np.linspace(0, 10, 11), np.zeros(11)]
    xb = np.c_[np.linspace(0, 10, 11), np.ones(11)]
    xy = np.vstack([xa, xb])
    q, t = block.zip_rows(xa[:, 0], xb[:, 0], np.arange(11), np.arange(11, 22), xy)
    assert len(q) == 10 and len(t) == 0
    xb2 = np.delete(xb, 4, axis=0)                                   # one column ends: a Y-split
    xy = np.vstack([xa, xb2])
    q, t = block.zip_rows(xa[:, 0], xb2[:, 0], np.arange(11), np.arange(11, 21), xy)
    assert len(q) == 9 and len(t) == 1


@pytest.fixture(scope="module")
def strip():
    from meshom.insert.band import insert_band

    win, ch, path, loops = synth.coast_strip(H=H)
    r = insert_band(win, Spec("strip", path, h_fine_m=3e3, kind="strip", width_m=120e3, stages=""), Uniform(H),
                    real=loops, log=lambda *a, **k: None)
    return win, r


def test_coastal_strip_follows_the_coast_with_fine_rows_and_joins_the_host(strip):
    """B181: a band of fine rows on the real coast (its spine), the size growing to the host's at the offshore host
    chain; the chain's nodes are the host's, unmoved; the result is valid and keeps the coast."""
    win, r = strip
    rec = r.record
    assert r.ok and rec["valid"], rec.get("invalid")
    assert rec["water_bodies"] == rec["water_bodies_host"] == 1 and rec["join_ends_moved"] == 0
    p = r.chart.inv(r.band["chains_xy"][0])                             # the host chain's nodes stay
    assert np.min(np.linalg.norm(r.window.points[None] - p[:, None], axis=2), axis=1).max() < 1e-9
    sc = rec["scores"]["final"]
    assert sc["core"]["invalid"] == 0
    assert sc["core"]["qmax"] < 150.0
    assert sc["coast_real"]["shift"]["h_ins_p99"] < 0.1               # the coast nodes lie on the real coast
    counts = rec["block"]["counts"]
    assert counts[0] == rec["block"]["spine_cells"]                     # row 0: the spine at the fine size
    sp = r.band["spine"]
    step = np.linalg.norm(np.diff(sp, axis=0), axis=1)
    assert abs(np.median(step) - 3e3) < 0.1 * 3e3
    assert sc["core"]["ratio_p99"] < 1.35


@pytest.fixture(scope="module")
def corridor():
    from meshom.insert.band import insert_band

    win, ch, path = synth.open_water(H=20e3)
    r = insert_band(win, Spec("corr", path, h_fine_m=6e3, kind="corridor", width_m=300e3, stages=""),
                    Uniform(20e3), log=lambda *a, **k: None)
    return win, r


def test_corridor_halves_share_their_axis_and_close_on_the_host_loop(corridor):
    """B182: two band blocks on either side of the axis share its nodes and their row ends; the insert is closed by a
    loop of host edges (both sides, both ends) whose nodes stay; one water body, no hanging node (every edge in at
    most two cells), the axis band at the fine size."""
    win, r = corridor
    rec = r.record
    assert r.ok, rec.get("refused")
    v = rec["validity_window"]
    assert v["inverted"] == 0 and v["nonmanifold_edges"] == 0
    assert rec["water_bodies"] == rec["water_bodies_host"] == 1 and rec["join_ends_moved"] == 0
    assert rec["halves"]["left"][0] == rec["halves"]["right"][0] == rec["block"]["spine_cells"]
    for cxy in r.band["chains_xy"]:                                    # the loop's host nodes are in the result
        p = r.chart.inv(cxy)
        d = np.min(np.linalg.norm(r.window.points[None] - p[:, None], axis=2), axis=1)
        assert d.max() < 1e-9
    sc = rec["scores"]["final"]
    assert sc["core"]["qmax"] < 150.0 and sc["core"]["ratio_p99"] < 1.4


def test_refusal_advice_checks_every_cell_count_on_both_sides():
    """B184: the advice lists the cell counts that pass both sides, none the no-zoom rule refuses; a build with one
    of them passes the rules."""
    import re

    win, ch, path = synth.banana(H=H, L=400e3)
    quiet = lambda *a, **k: None                 # noqa: E731
    res = insert_window(win, Spec("many", path, cells=40, stages="", reach_m=120e3), Uniform(H), log=quiet)
    adv = [x for x in res.record["refused"] if x.startswith("advice:")]
    assert len(adv) == 1
    m = re.search(r"pass both sides: ([0-9, -]+) \(", adv[0])
    assert m, adv[0]
    counts = [int(x) for x in re.findall(r"[0-9]+", m.group(1))]
    assert 2 not in counts                                           # 20 km / 2 cells: no zoom on a 10 km host
    n = max(counts)
    assert n < 40
    ok = insert_window(win, Spec("advised", path, cells=n, stages="", reach_m=120e3), Uniform(H), log=quiet)
    assert ok.ok and all(j.ok for j in ok.joins)


def test_l_join_round_a_cape(monkeypatch):
    """B183 / B180: with no straight join on the side where the mouth opens onto a coast, an L-join (a transect, then
    a leg back to the coast beyond the cape) closes the region; its bend gets a corner triangle and the block is
    valid."""
    from meshom.insert import region

    win, ch, path, loops = synth.cape_mouth(L=450e3)
    orig = region.candidates
    monkeypatch.setattr(region, "candidates", lambda side, *a, **k: orig(side, *a, **k) if side == "A" else [])
    r = insert_window(win, Spec("cape", path, cells=4, stages="", reach_m=200e3), Uniform(H), real=loops,
                      log=lambda *a, **k: None)
    assert r.ok and r.record["valid"]
    jb = r.joins[1]
    assert jb.shape == "L" and jb.ok
    assert r.record["block"]["corners"][1]                           # the bend of join B is cut by a triangle
    assert r.record["scores"]["final"]["core"]["invalid"] == 0


def test_validity_counts_overlapping_cells():
    """B186: two counter-clockwise cells on the same side of their shared edge overlap: no cell is inverted, but
    the shared edge is traversed the same way by both; the validity gate counts it (a fold hidden by reversing a
    cell passed the step-3 gate)."""
    from meshom.insert.geom import Chart
    from meshom.geo import lonlat_to_xyz
    from meshom.insert.measure import validity

    ch = Chart(lonlat_to_xyz(np.array(10.0), np.array(40.0)))
    xy = 1e4 * np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1], [0.5, 0.8], [0.5, 0.2]], float)
    good = Mesh(ch.inv(xy), np.zeros((0, 3), np.int64), np.array([[0, 1, 2, 3], [1, 4, 5, 2]]), np.zeros(8, np.int8))
    fold = Mesh(ch.inv(xy), np.zeros((0, 3), np.int64), np.array([[0, 1, 2, 3], [1, 2, 6, 7]]), np.zeros(8, np.int8))
    vg, vf = validity(good), validity(fold)
    assert vg["same_direction_edges"] == 0 and vg["inverted"] == 0
    assert vf["inverted"] == 0 and vf["nonconvex"] == 0             # each cell alone is fine ...
    assert vf["same_direction_edges"] == 1                          # ... but they overlap


def test_band_cells_keep_one_orientation():
    """B186: a band's cells are never reversed one by one; the synthetic strip and corridor have no shared edge
    traversed the same way by both cells."""
    from meshom.insert.band import insert_band

    win, ch, path, loops = synth.coast_strip(H=H)
    r = insert_band(win, Spec("strip", path, h_fine_m=3e3, kind="strip", width_m=120e3, stages=""), Uniform(H),
                    real=loops, log=lambda *a, **k: None)
    assert r.record["validity_window"]["same_direction_edges"] == 0
    assert measure_validity(r.merged)["same_direction_edges"] == 0


def measure_validity(m):
    from meshom.insert.measure import validity

    return validity(m)


# ------------------------------------------------------------------ step 4: bays by size, the collar, ORCA hosts

def test_bay_rule_by_size_and_shape():
    """B190: bays = "auto" meshes a pocket with a mouth of 3 local cells that reaches its mouth's width inland;
    True keeps B166 (a mouth of 2 cells), False none. bay_widths: the water across, parallel to the mouth."""
    from meshom.insert.realcoast import bay_meshed, bay_widths

    assert bay_widths(np.array([[0, 0], [0, -10], [10, -10], [10, 0]], float), (0.25, 0.5)) == (10.0, 10.0)
    assert bay_widths(np.array([[0, 0], [5, -10], [10, 0]], float), (0.25, 0.5)) == (7.5, 5.0)
    h = 1.0
    assert bay_meshed("auto", 4.0, h, 5.0) and not bay_meshed("auto", 4.0, h, 3.0)     # deep / shallow
    assert not bay_meshed("auto", 2.5, h, 5.0) and bay_meshed(True, 2.5, h, 5.0)       # 2.5 cells: only with True
    assert not bay_meshed(False, 10.0, h, 20.0)


def test_a_shallow_pocket_is_bridged_by_default():
    win, ch, path, loops = synth.strait_real(H=H, L=400e3, bend=0.0, w_neck=40e3, bay=(10e3, 45e3, 25e3))
    quiet = lambda *a, **k: None                                               # noqa: E731
    auto = insert_window(win, Spec("pocket", path, cells=10, stages=""), Uniform(H), real=loops, log=quiet)
    every = insert_window(win, Spec("pocket_all", path, cells=10, stages="", bays=True), Uniform(H), real=loops,
                          log=quiet)
    assert auto.ok and every.ok
    assert not [b for b in auto.record["bays"] if b["status"] == "meshed"]
    assert [b for b in every.record["bays"] if b["status"] == "meshed"]


def test_structured_host_keeps_every_node_and_a_collar_moves_none_beyond_it():
    """B193: on a lattice (structured, as ORCA) the cleaning keeps every host node by default; B192: a collar asked
    for moves host nodes only inside it."""
    win, ch, path = synth.banana(H=H, L=400e3)
    quiet = lambda *a, **k: None                                               # noqa: E731
    fixed = insert_window(win, Spec("orca_like", path, cells=10, stages="d,b,e,a", force=True), Uniform(H), log=quiet)
    assert fixed.record["host"]["structured"] and fixed.record["collar"]["host_fixed"]
    assert fixed.record["collar"]["host_moved"] == 0
    collar = insert_window(win, Spec("collar", path, cells=10, stages="d,b,e,a", force=True, collar_rings=4),
                           Uniform(H), log=quiet)
    c = collar.record["collar"]
    assert not c["host_fixed"] and c["host_moved"] > 0 and c["beyond_collar_moved"] == 0
    assert validity(collar.window)["inverted"] == 0


# ------------------------------------------------------------------ step 5: band rows from a harmonic map

def _band_case(shape):
    """A band region in a chart: a spine of 2 km cells (a straight coast with a bay or a notch), banks of 50 km at
    both ends, the host chain 50 km out (8 km edges); water north of the spine."""
    from meshom.geo import lonlat_to_xyz
    from meshom.insert.bandblock import band_levels
    from meshom.insert.geom import Chart, resample

    km = 1e3
    shapes = {"bay": [[0, 0], [90, 0], [95, -15], [105, -15], [110, 0], [200, 0]],
              "notch": [[0, 0], [95, 0], [100, -12], [105, 0], [200, 0]]}
    ch = Chart(lonlat_to_xyz(np.array([0.0]), np.array([45.0]))[0])
    hf = 2 * km
    sp = Polyline(np.array(shapes[shape], float) * km)
    spine = sp.at(np.linspace(0, sp.L, int(round(sp.L / hf)) + 1))
    bank_p = Polyline(np.array([[-50, 0], [0, 0]]) * km)
    bank_q = Polyline(np.array([[200, 0], [250, 0]]) * km)
    chain = resample(np.array([[-50, 0], [-50, 50], [250, 50], [250, 0]]) * km, 8 * km)
    d, h = band_levels(50 * km, hf, 20 * km, 8 * km)
    return ch, hf, spine, bank_p, bank_q, chain, d, h


@pytest.mark.parametrize("shape", ["bay", "notch"])
def test_harmonic_band_rows_cannot_fold_where_offset_rows_do(shape):
    """B195-B197: the harmonic map of the band is one-to-one (no flipped triangle in the square), its rows increase
    at every column, and the block built on it has no inverted or non-convex cell before any repair; the offset
    rows of B181 fold in the same bay and notch."""
    from meshom.insert import bandblock, harmonic

    ch, hf, spine, bank_p, bank_q, chain, d, h = _band_case(shape)
    hm = harmonic.harmonic_map(spine, bank_p, bank_q, chain, hf, ch)
    assert hm.info["flipped"] == 0 and hm.info["flat"] == 0
    S, T = harmonic.rows_t(hm, d, 20e3, len(spine) - 1)
    assert np.all(np.diff(T, axis=0) > 0)
    b = harmonic.build_harmonic(hm, spine, chain, d, h, 20e3)
    q, t = bandblock.orient_all(b.xy, b.quads, b.tris)
    sa = np.r_[bandblock._signed(b.xy, q), bandblock._signed(b.xy, t)]
    assert (sa > 0).all() and bandblock._convex(b.xy, q).all()
    K = b.K
    assert np.allclose(b.xy[b.off[K]:b.off[K + 1]], chain)                 # row K: the host chain, exact
    assert np.allclose(b.xy[b.off[0]:b.off[1]], spine[1:-1])             # row 0: the spine (P, Q merged)
    old = bandblock.build_side(bandblock.BandMap(spine, 1.0, 50e3, bank_p, bank_q, 0.25 * hf), spine, chain, d, h)
    q2, t2 = bandblock.orient_all(old.xy, old.quads, old.tris)
    assert (np.r_[bandblock._signed(old.xy, q2), bandblock._signed(old.xy, t2)] <= 0).sum() > 0


def test_harmonic_map_has_no_flat_triangle_at_a_boundary_ear():
    """B195: a sharp notch in the coast (4 km wide, 6 km deep, 0.8 km map spacing) leaves boundary ears (three
    consecutive spine points, no interior point) and edges across the notch with both ends on the spine; they are
    split, so every triangle of the map has a positive area in the square and none has its three corners on one
    side of it."""
    from meshom.geo import lonlat_to_xyz
    from meshom.insert import harmonic
    from meshom.insert.geom import Chart, resample

    km = 1e3
    ch = Chart(lonlat_to_xyz(np.array([0.0]), np.array([45.0]))[0])
    spine = np.vstack([np.c_[np.linspace(0, 28, 15), np.zeros(15)], [[30, -6]],
                       np.c_[np.linspace(32, 60, 15), np.zeros(15)]]) * km
    bank_p = Polyline(np.array([[-20, 0], [0, 0]]) * km)
    bank_q = Polyline(np.array([[60, 0], [80, 0]]) * km)
    chain = resample(np.array([[-20, 0], [-20, 20], [80, 20], [80, 0]]) * km, 8 * km)
    hm = harmonic.harmonic_map(spine, bank_p, bank_q, chain, 2 * km, ch)
    assert hm.info["flat"] == 0 and hm.info["flipped"] == 0 and hm.info["chords_split"] > 0
    p = hm.st[hm.tri]
    ar = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])
    assert (ar > 0).all() or (ar < 0).all()
    for k, v in ((0, 0.0), (0, 1.0), (1, 0.0), (1, 1.0)):
        assert not (p[:, :, k] == v).all(1).any()


def test_rejected_coast_rebuild_leaves_no_diagnostics(monkeypatch):
    """B198 / B199: a coast-smoothing rebuild with more bad cells is rejected; the record keeps the kept build's
    polish (each build's diagnostics travel with it)."""
    from meshom.insert import band, harmonic

    calls = {"bad": 0, "polish": 0}
    polish0 = harmonic.polish

    def bad(bb, K_ring=2):
        calls["bad"] += 1
        return np.repeat(bb["spine"][len(bb["spine"]) // 2][None], 1 if calls["bad"] == 1 else 3, axis=0)

    def polish(*a, **k):
        xy, r = polish0(*a, **k)
        calls["polish"] += 1
        return xy, dict(r, build=calls["polish"])

    monkeypatch.setattr(band, "coast_bad", bad)
    monkeypatch.setattr(harmonic, "polish", polish)
    win, ch, path, loops = synth.coast_strip(H=H)
    r = band.insert_band(win, Spec("strip_rej", path, h_fine_m=3e3, kind="strip", width_m=120e3, stages="",
                                   coast_rounds=1), Uniform(H), real=loops, log=lambda *a, **k: None)
    cs = r.record["coast_smooth"]
    assert calls["polish"] == 2 and cs["rounds"] == 1 and cs["bad"] == [1, 3] and cs["windows"] == 0
    assert r.record["polish"]["build"] == 1


def test_coast_shift_measures_the_whole_kept_coast(monkeypatch):
    """B198 / B199: the extra smoothing's coast shift is the largest distance of the whole kept coast (the end banks
    and the spine, sampled densely) from the base build's coast, here with a window on the bank beyond Q (the flank
    of the cape)."""
    from meshom.insert import band, harmonic
    from meshom.insert.geom import resample, seg_distance

    calls = {"bad": 0}
    coasts = []
    hmap0 = harmonic.harmonic_map

    def bad(bb, K_ring=2):                              # one bad cell on the bank beyond Q, 40-60 km from the spine
        calls["bad"] += 1
        a, sp = bb["arc"], bb["spine"]
        d = np.linalg.norm(a[:, None] - sp[None], axis=2).min(1)
        far = np.flatnonzero((d > 40e3) & (d < 60e3) & (np.linalg.norm(a - sp[-1], axis=1) < np.linalg.norm(a - sp[0],
                                                                                                          axis=1)))
        return a[far[:1]] if calls["bad"] == 1 else np.zeros((0, 2))

    def hmap(spine, bank_p, bank_q, *a, **k):
        coasts.append(np.vstack([bank_p.xy[:-1], spine, bank_q.xy[1:]]))
        return hmap0(spine, bank_p, bank_q, *a, **k)

    monkeypatch.setattr(band, "coast_bad", bad)
    monkeypatch.setattr(harmonic, "harmonic_map", hmap)
    win, ch, path, loops = synth.coast_strip(H=H)
    r = band.insert_band(win, Spec("strip_shift", path, h_fine_m=3e3, kind="strip", width_m=120e3, stages="",
                                   coast_rounds=1, coast_shift_cells=5.0), Uniform(H), real=loops,
                         log=lambda *a, **k: None)
    cs = r.record["coast_smooth"]
    assert cs["windows"] == 1 and len(coasts) == 2
    ref = coasts[0]
    d = seg_distance(resample(coasts[1], 100.0), np.stack([ref[:-1], ref[1:]], axis=1))
    assert d.max() > 5.0
    assert abs(cs["shift_km_max"] * 1e3 - d.max()) < 0.05 * d.max() + 1.0


def test_harmonic_strip_needs_no_repair(strip):
    """B195-B197 on the synthetic coast with a cape: nothing to untangle or unfold, the map one-to-one."""
    win, r = strip
    rec = r.record
    assert rec["rows"] == "harmonic"
    assert rec["block"]["map"]["flipped"] == 0
    assert rec["untangle"]["bad_before"] == 0 and rec["unfold"]["folded_before"] == 0


def test_offset_rows_stay_reachable():
    """B181's offset rows are kept as rows = "offset" (the comparison)."""
    from meshom.insert.band import insert_band

    win, ch, path, loops = synth.coast_strip(H=H)
    r = insert_band(win, Spec("strip_offset", path, h_fine_m=3e3, kind="strip", width_m=120e3, stages="",
                              rows="offset"), Uniform(H), real=loops, log=lambda *a, **k: None)
    assert r.ok and r.record["rows"] == "offset" and r.record["valid"]
    assert "map" not in r.record["block"]


def test_nested_pockets_bridge_an_estuarys_outer_channel(monkeypatch):
    """B195: a strip's coast repeats the pocket search on the bridged coast, so the channel outside an estuary's
    narrows (here 4 km wide and 12 km long; its smoothed mouth below 3 cells of 2.5 km) is bridged too."""
    from meshom.insert import realcoast as rcm

    km = 1e3
    p = np.array([(-50, 0), (-2, 0), (-2, -12), (-0.4, -12), (-0.4, -13), (-12, -13), (-12, -35), (12, -35),
                  (12, -13), (0.4, -13), (0.4, -12), (2, -12), (2, 0), (50, 0)], float) * km
    deepest = {}
    for n in (1, rcm.NEST_PASSES):
        monkeypatch.setattr(rcm, "NEST_PASSES", n)
        rc = rcm.RealCoast([p], [False])
        rc.prepare(lambda q: np.full(len(q), 2.5 * km), lambda q: np.full(len(q), np.inf), 2.5 * km, 1.5, "auto",
                   bridge="narrow")
        deepest[n] = -rc.main[0][:, 1].min()
    assert deepest[1] > 3 * km and deepest[rcm.NEST_PASSES] < 1 * km


def test_polish_lowers_the_largest_corner_and_keeps_cells_convex():
    """B197: a node pushed sideways in a regular grid is moved back by the polish; fixed nodes stay."""
    from meshom.insert import harmonic

    n = 6
    gx, gy = np.meshgrid(np.arange(n + 1, dtype=float), np.arange(n + 1, dtype=float))
    xy = np.c_[gx.ravel(), gy.ravel()]
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    quads = np.array([[idx[j, i], idx[j, i + 1], idx[j + 1, i + 1], idx[j + 1, i]] for j in range(n) for i in range(n)])
    c = idx[3, 3]
    xy[c] += [0.45, 0.3]
    fixed = np.zeros(len(xy), bool)
    fixed[np.r_[idx[0], idx[-1], idx[:, 0], idx[:, -1]]] = True
    a0 = max(harmonic._max_angles(xy, q) for q in quads)
    out, rec = harmonic.polish(xy, quads, np.zeros((0, 3), np.int64), fixed, 100.0)
    a1 = max(harmonic._max_angles(out, q) for q in quads)
    assert a1 < a0 - 10 and np.isfinite(a1)
    assert np.allclose(out[fixed], xy[fixed])


def test_polish_reports_what_the_returned_positions_leave():
    """B197 / B199: polish's bad_after counts the returned positions, also when the round limit ends it."""
    from meshom.insert import harmonic

    n = 6
    gx, gy = np.meshgrid(np.arange(n + 1, dtype=float), np.arange(n + 1, dtype=float))
    xy = np.c_[gx.ravel(), gy.ravel()]
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    quads = np.array([[idx[j, i], idx[j, i + 1], idx[j + 1, i + 1], idx[j + 1, i]] for j in range(n) for i in range(n)])
    xy[idx[3, 3]] += [0.45, 0.3]
    fixed = np.zeros(len(xy), bool)
    fixed[np.r_[idx[0], idx[-1], idx[:, 0], idx[:, -1]]] = True
    out, rec = harmonic.polish(xy, quads, np.zeros((0, 3), np.int64), fixed, 100.0, rounds=1)
    count = lambda x: sum(harmonic._max_angles(x, q) > 100.0 for q in quads)     # noqa: E731
    assert rec["rounds"] == 1 and rec["bad_before"] == count(xy) > 0 and rec["bad_after"] == count(out)


def test_staircase_chain_corners():
    """B197: a host chain's reflex step takes two columns, its convex step none."""
    from meshom.insert import harmonic

    chain = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [1, 3], [2, 3], [3, 3], [3, 4], [3, 5], [3, 6], [4, 6], [5, 6],
                      [6, 6]], float)
    inside = np.array([3.0, 1.0])                                    # the band lies to the right of the chain
    assert harmonic.reflex_corners(chain, inside) == [6]
    assert harmonic.reflex_corners(chain, inside, convex=True) == [3, 9]


def test_strip_fields_parse():
    """B195-B198: the strip's rows, coast rounds and coast shift cap are [[inserts]] fields."""
    from meshom.config import InsertCfg, _build
    from meshom.insert.core import spec_from

    e = _build(InsertCfg, parse_inline('name=s; kind="strip"; width_km=50; path=[[0, 0], [0, 1]]; rows="offset"; '
                                       'coast_rounds=1; coast_shift_cells=0.5'), "insert")
    s = spec_from(e)
    assert (s.rows, s.coast_rounds, s.coast_shift_cells) == ("offset", 1, 0.5)
    d = spec_from(_build(InsertCfg, parse_inline('name=s; kind="strip"; width_km=50; path=[[0, 0], [0, 1]]'),
                         "insert"))
    assert (d.rows, d.coast_rounds, d.coast_shift_cells) == ("harmonic", 3, 1.0)


def test_monotone_counts_only_widen_rows_with_equal_counts():
    """B200: with a set join count a side's count target runs monotonically from the neck to it: equal counts give
    the neck's count throughout (no columns born and killed again), a larger join count a non-decreasing run."""
    raw = np.array([9.0, 8, 7, 6.5, 6, 6.3, 7, 8, 7.5, 7, 6])
    assert (block.monotone_counts(raw, 6, 6, (True, True), 4) == 6).all()
    up = block.monotone_counts(raw, 9, 9, (True, True), 4)
    assert (np.diff(up[4:]) >= 0).all() and (np.diff(up[:5]) <= 0).all() and up.max() <= 9
    assert np.array_equal(block.monotone_counts(raw, 6, 6, (False, True), 4)[:4], raw[:4])   # side A untouched


def test_side_rule_scales_the_room_and_counts_the_funnel(banana):
    """B200 / B201: a faster side slope reaches the host size nearer the neck (the room rule's target, its slope
    scaled by growth / 0.12), so more joins near the neck have the room; with a set join count the funnel takes the counts between
    the neck's and the join's."""
    from meshom.insert import region

    win, res = banana
    prof = res.profile
    fast = region.side_target(prof, prof.n_cells, region.SideRule(None, 0.36))
    assert np.allclose(region.side_target(prof, prof.n_cells, None), prof.h_t)
    assert (fast >= prof.h_t - 1e-6).all() and (fast > prof.h_t * 1.5).any()
    Ns = np.array([4.0, 8.0, 20.0])
    assert np.array_equal(region.funnel_counts(Ns, 10, region.SideRule(10)), [10, 10, 10])
    assert np.array_equal(region.funnel_counts(Ns, 10, region.SideRule(14)), [10, 10, 14])
    assert np.array_equal(region.funnel_counts(Ns, 10, None), Ns)
    win, ch, path = synth.butterfly(H=H, open_deg=120.0)              # opens fast: W / n grows faster than 0.14
    quiet = lambda *a, **k: None                 # noqa: E731
    slow = insert_window(win, Spec("g0", path, cells=6, stages=""), Uniform(H), log=quiet)
    quick = insert_window(win, Spec("g3", path, cells=6, stages="", growth_a=0.3, growth_b=0.3), Uniform(H),
                          log=quiet)
    assert quick.ok and quick.record["valid"]
    n_room = [sum("room" in c.fails for c in r.candidates) for r in (slow, quick)]
    assert n_room[1] < n_room[0]                                       # more joins near the neck have the room
    if slow.ok:
        assert all(q.d_m <= s.d_m for q, s in zip(quick.joins, slow.joins))
    assert quick.record["block"]["growth"] == [0.3, 0.3]


def test_per_side_join_counts_are_met_one_to_one():
    """B200: cells_a / cells_b: each join has exactly that many edges and the block's counts run monotonically from
    the neck to them (equal counts: one count throughout, no Y-split); an unreachable count is refused with the
    counts the passing candidates have."""
    win, ch, path, loops = synth.strait_real(H=H, L=400e3)
    quiet = lambda *a, **k: None                 # noqa: E731
    r = insert_window(win, Spec("ab", path, cells=10, stages="", cells_a=10, cells_b=10, bank_land_cells=1.0),
                      Uniform(H), real=loops, log=quiet)
    assert r.ok and r.record["valid"]
    assert [j.n for j in r.joins] == [10, 10]
    assert r.record["block"]["counts"] == [10] * (r.record["block"]["rows"] + 1)
    assert r.record["block"]["monotone"] == [True, True]
    bad = insert_window(win, Spec("ab3", path, cells=10, stages="", cells_a=3), Uniform(H), real=loops, log=quiet)
    assert not bad.ok
    side_a = [x for x in bad.record["refused"] if x.startswith("side A")][0]
    assert "the side's 3 edges" in side_a and "cells_a = one of them" in side_a


def test_banks_into_land_cut_capes_and_never_enter_the_water():
    """B202: the into-land move cuts a cape by about the allowance and leaves a pocket (and straight coast) where it
    was; the moved line never lies on the water side."""
    from meshom.insert import realcoast as rcm

    x = np.linspace(0.0, 30.0, 601)
    cape = np.where(np.abs(x - 3) < 1, 1 - np.abs(x - 3), 0.0)          # into the water (left of +x: y > 0)
    pocket = -np.where(np.abs(x - 10) < 1, 1 - np.abs(x - 10), 0.0)     # into the land
    q = np.c_[x, cape + pocket]
    m, d = rcm.into_land(q, False, np.full(len(q), 0.5), np.full(len(q), 0.5))
    assert (d >= 0).all() and d.max() - 0.5 < 0.5                   # the last rounding passes D by a fraction of a cell
    assert np.max(m[:, 1][np.abs(m[:, 0] - 3) < 0.3]) < 0.75             # the cape (tip at 1) is cut
    near = lambda x0: np.abs(m[:, 0] - x0) < 0.2                         # noqa: E731
    assert np.min(m[near(10), 1]) < -0.8                                # the pocket keeps its line
    assert np.abs(m[(m[:, 0] > 20) & (m[:, 0] < 26), 1]).max() < 0.02    # so does a straight coast


def test_squared_bank_end_leaves_the_join_at_a_right_angle():
    """B202: an acute end (the coast heads back along the chain) is squared: the bank starts on the join end node at
    a right angle to the chain and reaches the coast la along; an obtuse end keeps the blend."""
    from meshom.insert import realcoast as rcm

    t = np.linspace(0.0, 1.0, 400)
    arc = np.c_[100e3 * t, 30e3 * t]                                    # the coast, 17 deg off the x axis
    xa, xb = np.array([0.0, 0.0]), arc[-1].copy()
    ua = np.array([1.0, 0.2])                                           # the chain from xa into the join: acute
    ub = np.array([0.0, 1.0])                                           # obtuse at xb
    b, sq = rcm.square_ends(arc, xa, xb, ua, ub, 20e3, 20e3)
    assert sq == [True, False]
    assert np.allclose(b[0], xa) and np.allclose(b[-1], xb)
    d0 = (b[1] - b[0]) / np.linalg.norm(b[1] - b[0])
    assert abs(np.dot(d0, ua / np.linalg.norm(ua))) < 0.05


def test_side_fields_parse():
    """B200-B202: the per-side counts and slopes and the into-land allowance are [[inserts]] fields."""
    from meshom.config import InsertCfg, _build
    from meshom.insert.core import spec_from

    s = spec_from(_build(InsertCfg, parse_inline('name=g; path=[[0, 0], [0, 1]]; cells_a=6; cells_b=9; '
                                                 'growth_a=0.3; bank_land_cells=1'), "insert"))
    assert (s.cells_a, s.cells_b, s.growth_a, s.growth_b, s.bank_land_cells) == (6, 9, 0.3, None, 1.0)
    d = spec_from(_build(InsertCfg, parse_inline('name=g; path=[[0, 0], [0, 1]]'), "insert"))
    assert (d.cells_a, d.cells_b, d.growth_a, d.growth_b, d.bank_land_cells) == (None, None, None, None, 0.0)
