"""m16_orca's nest (B172-B176): a refined ORCA in a box of the coarse grid, joined by meshom.orca.nest's frame, on
ORCA2 and its own map refined 3 times (a test preset, as ORCA12 is ORCA025 x 3): the frame is conforming, its rings 0
and n are the two grids' own F points, both ORCA parts stay unchanged, its triangles are the count the tensor corners
fix (each with one valence-5 node, no valence 3), its quads are near rectangular in the corners too, and a box across
the east-west wrap works as any other."""
import numpy as np
import pytest

from meshom.domain import Domain


@pytest.fixture(scope="module")
def presets():
    from dataclasses import replace

    from meshom.orca.presets import PRESETS

    PRESETS["ORCA2X3"] = replace(PRESETS["ORCA2"], name="ORCA2X3", q=3, pole_row=3 * 147, rows=0, ref=None,
                                 verified=False)
    yield PRESETS
    PRESETS.pop("ORCA2X3", None)


def _run(box, width):
    from meshom.methods.m16_orca import _STATE, run
    from meshom.size import Uniform

    mesh, info = run(Domain([], "sphere"), Uniform(1e5),
                     {"orca": "ORCA2", "mask": "all", "size_tolerance": None,
                      "nest": {"fine_orca": "ORCA2X3", "box": box, "width": width}})
    return mesh, info, _STATE["belt"]


@pytest.fixture(scope="module", params=[([60, 120, 40, 90], 39), ([165, 205, 60, 85], 21)], ids=["open", "wrap"])
def nest_mesh(presets, request):
    box, width = request.param
    return (box, width) + _run(box, width)


def test_frame_counts_and_quality_in_index_space():
    """B172-B175: triangles = N_n - N_0 + 8 n, every interior node of valence 4 or 5 (one 5 per triangle), the
    rings' nodes on the two lattices; quads near rectangular in the corner squares as on the sides, the size ratio
    across inner edges as the ladder's."""
    from meshom.orca.nest import index_quality, topology

    ft = topology(564, 489, 39)
    n, H, Lx, Ly = ft.info["n"], 39, 564, 489
    r0, rn = ft.ring == 0, ft.ring == n
    assert ft.info["ring_nodes_first_last"] == [int(r0.sum()), int(rn.sum())] == [2 * (Lx + Ly) // 3,
                                                                                    2 * (Lx + Ly - 4 * H)]
    assert np.all(np.mod(ft.x[r0], 3) == 0) and np.all(np.mod(ft.y[r0], 3) == 0)
    assert np.all(ft.x[rn] == np.round(ft.x[rn])) and np.all(ft.y[rn] == np.round(ft.y[rn]))
    assert len(ft.tris) == ft.info["tris_expected"] == rn.sum() - r0.sum() + 8 * n
    q = index_quality(ft)
    assert q["valence_interior"] == {4: q["valence_interior"][4], 5: len(ft.tris)}
    assert q["reflex_quads"] == 0 and q["quad_max"] < 112.0 and q["quad_max_p99_9"] < 111.0
    assert q["quads_over_110_pct"] < 0.2 and q["ratio_p99"] < 1.16 and q["tri_min"] > 30.0
    qc = index_quality(ft, corner=True)
    assert qc["quad_max"] < 112.0 and qc["quads_over_110_pct"] < 1.0 and qc["ratio_p99"] < 1.16


def test_nest_conforming_and_parts_unchanged(nest_mesh):
    """B176: the frame joins both ORCA parts node for node (every boundary node of the frame matched at 0 m), the
    coarse part keeps every cell outside the coarse box and the fine part every cell inside the fine box, both
    unchanged, the mesh valid and closed but for ORCA2's south row and the grid poles' holes."""
    from scipy.spatial import cKDTree

    from meshom.compete import irregular_counts
    from meshom.geo import lonlat_to_xyz
    from meshom.mesh import signed_area
    from meshom.methods.m16_orca import belt_unchanged
    from meshom.orca import build

    box, width, mesh, info, (pts, quads, tris) = nest_mesh
    j, ni = info["joints"]["nest"], info["nest"]
    assert j["conforming"] and j["unmatched"] == 0 and j["max_match_rad"] == 0.0
    assert j["matched"] == j["cap_boundary_nodes"] == ni["ring0_nodes"] + ni["ringn_nodes"]
    gc, gf = build("ORCA2"), build("ORCA2X3")
    nc, nf = 180, 540
    Ca, Cb, Ra, Rb = info["nest_coarse_box"]
    c0, c1, r0, r1 = box
    cc = np.r_[np.arange(Ca, Cb + 1), np.arange(Ca, Cb + 1), np.full(Rb - Ra + 1, Ca), np.full(Rb - Ra + 1, Cb)] % nc
    rr = np.r_[np.full(Cb - Ca + 1, Ra), np.full(Cb - Ca + 1, Rb), np.arange(Ra, Rb + 1), np.arange(Ra, Rb + 1)]
    bi, bj = info["fine"]["nesting"]["col_offset"], info["fine"]["nesting"]["row_offset"]
    X = np.r_[np.arange(3 * c0 + bi, 3 * c1 + bi + 1), np.arange(3 * c0 + bi, 3 * c1 + bi + 1)]
    fx = np.r_[X, np.full(3 * (r1 - r0) + 1, 3 * c0 + bi), np.full(3 * (r1 - r0) + 1, 3 * c1 + bi)] % nf
    fy = np.r_[np.full(len(X) // 2, 3 * r0 + bj), np.full(len(X) // 2, 3 * r1 + bj),
               np.arange(3 * r0 + bj, 3 * r1 + bj + 1), np.arange(3 * r0 + bj, 3 * r1 + bj + 1)]
    edge = np.vstack([lonlat_to_xyz(gc.lon["f"][rr, cc], gc.lat["f"][rr, cc]),
                      lonlat_to_xyz(gf.lon["f"][fy, fx], gf.lat["f"][fy, fx])])
    d, _ = cKDTree(mesh.points).query(edge)
    assert d.max() < 1e-12                                   # every box edge F point is a node of the mesh
    u = belt_unchanged(mesh, pts, quads, tris)
    assert u["max_displacement_m"] == 0.0 and u["belt_elems_kept"] == u["belt_elems_total"]
    assert u["belt_elems_altered"] == 0
    assert info["fine"]["orca_quads"] == 9 * (c1 - c0) * (r1 - r0)
    coarse = info["orca_quads"] + info["orca_tris"] - info["fine"]["orca_quads"] - info["fine"]["orca_tris"]
    r = 2                                                     # the clean-up's context ring (nest.CONTEXT_CELLS)
    ring_cells = (Cb - Ca) * (Rb - Ra) - (Cb - Ca - 2 * r) * (Rb - Ra - 2 * r)
    assert info["context_cells"] == ring_cells and coarse == info["cells"]["box_cells"] - ring_cells
    assert info["fine"]["context_cells"] == \
        (3 * (c1 - c0) + 2 * r) * (3 * (r1 - r0) + 2 * r) - 9 * (c1 - c0) * (r1 - r0)
    assert (signed_area(mesh) > 0).all() and irregular_counts(mesh)["euler_ok"]


def test_nest_triangles_and_valences(nest_mesh):
    """B173-B174: the mesh's triangles are the frame's, as many as N_n - N_0 + 8 n, each with one valence-5 node; no
    valence-3 node anywhere."""
    from meshom.compete import irregular_counts
    from meshom.methods.m16_orca import _ruled

    _, _, mesh, info, (pts, quads, tris) = nest_mesh
    ni = info["nest"]
    assert mesh.n_tris - len(tris) == ni["tris"] == ni["tris_expected"]
    assert not _ruled(mesh, pts, quads, tris)[:mesh.n_tris][-ni["tris"]:].any()
    ic = irregular_counts(mesh)
    assert ic["n3"] == 0 and ic["n_other"] == 0 and ic["n5"] == ni["tris"]


def test_nest_quality_on_the_sphere(nest_mesh):
    """B172-B175 on the sphere: the frame's quads stay near rectangular (corners included) and its elements' size
    changes slowly, also across the two box edges."""
    from meshom.mesh import build_topology
    from meshom.methods.m16_orca import _ruled
    from meshom.quality.kernel import element_geometry

    box, width, mesh, info, (pts, quads, tris) = nest_mesh
    orca = _ruled(mesh, pts, quads, tris)
    g = element_geometry(mesh)
    T = mesh.n_tris
    qmax = np.degrees(g.quad_angles.max(axis=1))[~orca[T:]]
    me = np.r_[g.tri_edges.mean(axis=1), g.quad_edges.mean(axis=1)]
    ee = build_topology(mesh).edge_elems
    ee = ee[ee[:, 1] >= 0]
    fr = ~orca[ee[:, 0]] | ~orca[ee[:, 1]]                    # edges with a frame element on a side
    ratio = np.maximum(me[ee[fr, 0]], me[ee[fr, 1]]) / np.minimum(me[ee[fr, 0]], me[ee[fr, 1]])
    if width == 39:
        assert qmax.max() < 115.0 and np.percentile(qmax, 99.9) < 112.0 and np.percentile(ratio, 99) < 1.16
    else:                                                     # the narrow frame (13 rings): steeper, still bounded
        assert qmax.max() < 120.0 and np.percentile(ratio, 99) < 1.25


def test_box_from_lonlat(presets):
    """B176: a lon-lat box becomes the smallest box of coarse F lines holding every cell whose T point lies in it,
    also across the antimeridian (columns unwrapped)."""
    from meshom.orca import build
    from meshom.orca.mesh import in_box
    from meshom.orca.nest import box_from_lonlat

    g = build("ORCA2")
    lo, la = g.lon["t"][1:g.j_pole - 1, :-2], g.lat["t"][1:g.j_pole - 1, :-2]
    ni = lo.shape[1]
    for bb in ([-60.0, 10.0, -20.0, 40.0], [170.0, -20.0, -170.0, 10.0], [74.0, -20.0, 80.0, 20.0]):
        c0, c1, r0, r1 = box_from_lonlat(g, bb)
        assert c0 < c1 < c0 + ni and r0 < r1
        dl = np.mod(lo - bb[0], 360.0)
        inside = (dl <= np.mod(bb[2] - bb[0], 360.0)) & (la >= bb[1]) & (la <= bb[3])
        rr, cc = np.nonzero(inside)
        ij = np.c_[rr + 1, cc]
        assert in_box(ij, ni, (c0, c1, r0, r1)).all()
        # the box is the smallest: each of its four edge lines has a cell of the lon-lat box next to it
        assert (ij[:, 0] == r0 + 1).any() and (ij[:, 0] == r1).any()
        cu = c0 + 1 + np.mod(ij[:, 1] - c0 - 1, ni)
        assert cu.min() == c0 + 1 and cu.max() == c1        # also across the grid's own column seam (ORCA2 near 74 E)


def test_cleanup_context_keeps_a_channel_across_the_box_edge():
    """B176: a one-cell channel joining two water bodies across a box edge and on beyond the clean-up's context ring
    keeps every cell inside the box (the ring's crop edge is pinned: the grid goes on there); the whole grid's clean-up
    keeps them all too. ORCA2's cells, a set mask (NEMO's file not read)."""
    from unittest.mock import patch

    from meshom.methods.m16_orca import DEFAULTS, _masked
    from meshom.orca import build
    from meshom.orca.mesh import box_cells, cells, fesom_cleanup, in_box

    g = build("ORCA2")
    nj, ni = g.lat["t"].shape[0], g.lat["t"].shape[1] - 2
    wet = np.zeros((nj, ni), bool)
    wet[45:56, 65:76] = True                                  # two basins joined by a channel across row 60
    wet[56:69, 70] = True
    wet[68:79, 65:76] = True
    oc = cells(g, 35, 85)
    ij = oc.cell_ij
    _, kept, _ = fesom_cleanup(oc.mesh, wet[ij[:, 0], ij[:, 1]], True)
    box, wide = (60, 80, 40, 60), (58, 82, 38, 62)
    with patch("numpy.load", return_value={}), \
            patch("meshom.orca.mesh.transfer_mask", return_value=(wet, np.ones_like(wet))):
        part = _masked(g, "ORCA2", dict(DEFAULTS, mask="nemo", ref_dir="unused"), 0,
                       lambda: box_cells(cells(g, 39, 62), wide, True), caps=True,
                       context=lambda o: ~in_box(o.cell_ij, o.ni, box))
    assert part["belt"].n_elems == int((kept & in_box(ij, ni, box)).sum()) == 126
    assert all(part["wetmap"][r, 70] == 2 for r in range(56, 61))


# ------------------------------------------------------------------------------- B230: a nest in the nest

@pytest.fixture(scope="module")
def tiny_family():
    """A coarse map of 6 deg (NEMO's generator with 60 columns, T pivot) and its lattices refined 3 and 9 times: the
    nest in the nest's three presets at a size a test can build (ORCA2 refined 9 times takes minutes)."""
    from meshom.orca.generator import GenParams, row_map, set_coefficients
    from meshom.orca.presets import PRESETS, Preset

    p = GenParams(jpi=61, jpeqt=27, jpeqn=3, jpnord=19, pivot="T")
    c = set_coefficients(p)
    pole = int(round(c.rjpj + row_map(p, c).shift - 1.0))                  # T rows at J = 1 + j / q
    names = {1: "TINY", 3: "TINYX3", 9: "TINYX9"}
    for q, nm in names.items():
        PRESETS[nm] = Preset(nm, p, layout="lattice", q=q, i_offset=0.5, pole_row=q * pole, verified=False)
    yield PRESETS
    for nm in names.values():
        PRESETS.pop(nm, None)


def _f_points(name, rows, cols):
    """Unit vectors of a lattice preset's F points (r, c) for the pairs of rows and cols (generator.lattice)."""
    from meshom.geo import lonlat_to_xyz
    from meshom.orca.generator import lattice, row_map, set_coefficients
    from meshom.orca.presets import PRESETS

    pr = PRESETS[name]
    c = set_coefficients(pr.params)
    rm = row_map(pr.params, c)
    joff = (c.rjpj + rm.shift) - pr.pole_row / pr.q
    h = 0.5 / pr.q
    out = []
    for r, cc in zip(rows, cols):
        lo, la = lattice(pr.params, c, rm, {"f": [pr.i_offset + cc / pr.q + h]}, {"f": [joff + r / pr.q + h]})
        out.append(lonlat_to_xyz(lo["f"][:, 0], la["f"][:, 0]))
    return np.vstack(out)


def _box_edge(box):
    c0, c1, r0, r1 = box
    cols = np.r_[np.arange(c0, c1 + 1), np.arange(c0, c1 + 1), np.full(r1 - r0 + 1, c0), np.full(r1 - r0 + 1, c1)]
    rows = np.r_[np.full(c1 - c0 + 1, r0), np.full(c1 - c0 + 1, r1), np.arange(r0, r1 + 1), np.arange(r0, r1 + 1)]
    return rows, cols


def test_nest_in_the_nest(tiny_family):
    """B230: `nest.inner` with a preset that refines the nest's fine preset by 3: its box is the fine box in the fine
    preset's F lines less inner.width / 3 fine cells on every side; a second frame joins it, its ring 0 the first
    frame's ring n (every fine F point of the fine box's edge), its ring n the inner box's edge; the box holds no cell
    of the middle preset, the innermost preset's cells fill the inner box; both frames conform, every ORCA part stays
    unchanged, the frames' cells are near rectangular and their size changes slowly."""
    from scipy.spatial import cKDTree

    from meshom.compete import irregular_counts
    from meshom.mesh import build_topology, signed_area
    from meshom.methods.m16_orca import _STATE, _ruled, belt_unchanged, run
    from meshom.quality.kernel import element_geometry
    from meshom.size import Uniform

    box, width = [23, 35, 14, 26], 21
    mesh, info = run(Domain([], "sphere"), Uniform(1e5),
                     {"orca": "TINY", "mask": "all", "size_tolerance": None,
                      "nest": {"fine_orca": "TINYX3", "box": box, "width": width,
                               "inner": {"fine_orca": "TINYX9", "width": width}}})
    pts, quads, tris = _STATE["belt"]
    w = width // 3
    fine_box = [3 * box[0] + 1, 3 * box[1] + 1, 3 * box[2] + 1, 3 * box[3] + 1]      # b_i = b_j = 1
    inner_box = [fine_box[0] + w, fine_box[1] - w, fine_box[2] + w, fine_box[3] - w]
    assert info["fine_inner"]["nesting"]["col_offset"] == info["fine_inner"]["nesting"]["row_offset"] == 1
    assert info["nest_inner_coarse_box"] == fine_box and info["nest_inner_box"] == inner_box
    assert "fine" not in info                                             # no cell of the middle preset
    assert info["fine_inner"]["orca_quads"] == 9 * (inner_box[1] - inner_box[0]) * (inner_box[3] - inner_box[2])
    for key in ("nest", "nest_inner"):
        assert info["joints"][key]["conforming"] and info["joints"][key]["max_match_rad"] == 0.0
    ji, ni = info["joints"]["nest_inner"], info["nest_inner"]
    assert ji["unmatched"] == 0 and ji["matched"] == ni["ring0_nodes"] + ni["ringn_nodes"]
    assert ni["tris"] == ni["tris_expected"] and mesh.n_tris == len(tris) + info["nest"]["tris"] + ni["tris"]
    r3, c3 = _box_edge(fine_box)
    r9, c9 = _box_edge([3 * inner_box[0] + 1, 3 * inner_box[1] + 1, 3 * inner_box[2] + 1, 3 * inner_box[3] + 1])
    edge = np.vstack([_f_points("TINYX3", r3, c3), _f_points("TINYX9", r9, c9)])
    assert cKDTree(mesh.points).query(edge)[0].max() < 1e-12                # both rings are nodes of the mesh
    assert ni["ring0_nodes"] == len(np.unique(np.c_[r3, c3], axis=0))
    u = belt_unchanged(mesh, pts, quads, tris)
    assert u["max_displacement_m"] == 0.0 and u["belt_elems_kept"] == u["belt_elems_total"]
    assert u["belt_elems_altered"] == 0
    assert (signed_area(mesh) > 0).all() and irregular_counts(mesh)["euler_ok"]
    orca = _ruled(mesh, pts, quads, tris)
    g = element_geometry(mesh)
    T = mesh.n_tris
    qmax = np.degrees(g.quad_angles.max(axis=1))[~orca[T:]]
    me = np.r_[g.tri_edges.mean(axis=1), g.quad_edges.mean(axis=1)]
    ee = build_topology(mesh).edge_elems
    ee = ee[ee[:, 1] >= 0]
    fr = ~orca[ee[:, 0]] | ~orca[ee[:, 1]]
    ratio = np.maximum(me[ee[fr, 0]], me[ee[fr, 1]]) / np.minimum(me[ee[fr, 0]], me[ee[fr, 1]])
    assert qmax.max() < 120.0 and np.percentile(ratio, 99) < 1.25
