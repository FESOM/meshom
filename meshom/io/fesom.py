"""FESOM2, FESOM-C and FESOM3 mesh files: nod2d.out, elem2d.out, depth.out, aux3d.out.

Rules (as FESOM2, FESOM-C and FESOM3 read them: github.com/FESOM/fesom2, /fesom-c, /fesom3): nod2d.out = N, then
`id lon lat flag`, ids 1..N; elem2d.out = M, then 3 columns (FESOM2) or 4 columns (FESOM-C, FESOM3; a triangle is
`n1 n2 n3 n1`); files are written clockwise (east = x, north = y), read in any orientation. FESOM-C depth.out is
positive down, one value per line; aux3d.out is `nl`, nl level depths, N node depths (FESOM2 and FESOM3: negative
down; FESOM-C: a partitioner-only placeholder). A FESOM3 folder never holds the FESOM-C depth files.
"""
import re
import shutil
from pathlib import Path

import numpy as np

from meshom.geo import lonlat_to_xyz, xyz_to_lonlat
from meshom.mesh import Mesh, build_topology, orient_ccw

NOD_FMT = "%i %15.10f %15.10f %i"


def _count(path: Path) -> int:
    with open(path) as f:
        return int(f.readline().split()[0])


def read_fesom(path: Path) -> Mesh:
    """nod2d.out + elem2d.out (3 or 4 columns, any orientation) -> CCW Mesh; flags kept in node_flag."""
    path = Path(path)
    n = _count(path / "nod2d.out")
    nod = np.loadtxt(path / "nod2d.out", skiprows=1, max_rows=n, ndmin=2)
    m = _count(path / "elem2d.out")
    el = np.loadtxt(path / "elem2d.out", skiprows=1, max_rows=m, dtype=np.int64, ndmin=2) - 1
    if el.shape[1] == 4:
        tri = el[:, 3] == el[:, 0]
        tris, quads = el[tri, :3], el[~tri]
    else:
        tris, quads = el, np.zeros((0, 4), dtype=np.int64)
    mesh = Mesh(lonlat_to_xyz(nod[:, 1], nod[:, 2]), np.ascontiguousarray(tris), np.ascontiguousarray(quads),
                nod[:, 3].astype(np.int8))
    return orient_ccw(mesh)


def _write_nod2d(path: Path, mesh: Mesh, flag: np.ndarray) -> None:
    lon, lat = xyz_to_lonlat(mesh.points)
    ids = np.arange(1, mesh.n_nodes + 1)
    np.savetxt(path / "nod2d.out", np.column_stack([ids, lon, lat, flag]), fmt=NOD_FMT,
               header=str(mesh.n_nodes), comments="")


def _write_elem4(path: Path, mesh: Mesh) -> None:
    """elem2d.out with 4 columns: triangles first as `n1 n2 n3 n1`, then quads, clockwise."""
    el = np.vstack([mesh.tris[:, [0, 2, 1, 0]], mesh.quads[:, [0, 3, 2, 1]]]) + 1      # CCW -> clockwise
    np.savetxt(path / "elem2d.out", el, fmt="%i", header=str(len(el)), comments="")


def _write_aux3d(path: Path, depth_m: np.ndarray, levels_m: np.ndarray) -> None:
    """aux3d.out: nl, the nl levels and the N node depths, all written negative down (as CORE2's file)."""
    levels = np.asarray(levels_m, dtype=np.float64)                                   # positive down
    with open(path / "aux3d.out", "w") as f:
        f.write(f"{len(levels)}\n")
        np.savetxt(f, 0.0 - levels, fmt="%.4f")
        np.savetxt(f, 0.0 - np.asarray(depth_m, dtype=np.float64), fmt="%15.10f")


# files the partitioners derive in the mesh folder (neither FESOM3's fesom_ini nor FESOM2's fesom_meshpart overwrites
# them) and the FESOM-C depth.out: every writer removes the derived files of an earlier mesh, the FESOM2 and FESOM3
# writers also a depth.out (they write none)
DERIVED = ("edgenum.out", "edges.out", "edge_tri.out", "nlvls.out", "elvls.out", "elvls_raw.out")
# partition folders dist_<npes> of an earlier mesh: FESOM3 reads dist_<npes>/rpart.out for a multi-rank run and a
# 1-partition fesom_ini stops before rewriting them, so they are removed with the derived files
STALE_DIRS = re.compile(r"dist_\d+")


def clear_stale(path: Path, files: tuple = DERIVED) -> None:
    """Remove `files` and the dist_<N> partition folders of an earlier mesh from the mesh folder `path`."""
    for f in files:
        (path / f).unlink(missing_ok=True)
    for d in path.iterdir():
        if d.is_dir() and STALE_DIRS.fullmatch(d.name):
            shutil.rmtree(d)


def write_fesomc(mesh: Mesh, path: Path, depth_m: np.ndarray) -> None:
    """FESOM-C files: nod2d.out (flag = node_flag), elem2d.out (4 columns, triangles first, clockwise),
    depth.out (positive down), placeholder aux3d.out (5 levels 0..4, then -4400 per node). Derived files and
    dist_<N> partition folders of an earlier mesh in `path` are removed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    clear_stale(path)
    _write_nod2d(path, mesh, mesh.node_flag)
    _write_elem4(path, mesh)
    np.savetxt(path / "depth.out", np.asarray(depth_m, dtype=np.float64), fmt="%15.10f")
    with open(path / "aux3d.out", "w") as f:
        f.write("5\n0\n1\n2\n3\n4\n")
        np.savetxt(f, np.full(mesh.n_nodes, -4400.0), fmt="%15.10f")


def write_fesom2(mesh: Mesh, path: Path, depth_m: np.ndarray, levels_m: np.ndarray) -> None:
    """FESOM2 files for a triangle-only mesh: nod2d.out (flag 1 on boundary nodes, 0 elsewhere), elem2d.out
    (3 columns, clockwise), aux3d.out (nl, levels, node depths; written negative down). Derived files, a depth.out
    and dist_<N> partition folders of an earlier mesh in `path` are removed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    clear_stale(path, DERIVED + ("depth.out",))
    flag = build_topology(mesh).boundary_node.astype(np.int64)
    _write_nod2d(path, mesh, flag)
    el = mesh.tris[:, [0, 2, 1]] + 1                                                  # CCW -> clockwise
    np.savetxt(path / "elem2d.out", el, fmt="%i", header=str(len(el)), comments="")
    _write_aux3d(path, depth_m, levels_m)


def write_fesom3(mesh: Mesh, path: Path, depth_m: np.ndarray, levels_m: np.ndarray) -> None:
    """FESOM3 files: nod2d.out as write_fesom2 (flag 1 on
    boundary nodes), elem2d.out with 4 columns as write_fesomc (triangle `n1 n2 n3 n1`, clockwise), aux3d.out as
    write_fesom2 (the real levels and node depths, negative down); no depth.out. The vertical levels per node are
    left to fesom_ini's level finder (aux3d path). Derived files and dist_<N> partition folders of an earlier run
    in `path` are removed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    clear_stale(path, DERIVED + ("depth.out",))
    _write_nod2d(path, mesh, build_topology(mesh).boundary_node.astype(np.int64))
    _write_elem4(path, mesh)
    _write_aux3d(path, depth_m, levels_m)


def read_aux3d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """aux3d.out -> (zbar (nl,), node depths (N,)), both metres positive down (FESOM2 reads either sign)."""
    v = np.asarray(Path(path).read_text().split(), dtype=np.float64)
    nl = int(v[0])
    return np.abs(v[1:1 + nl]), np.abs(v[1 + nl:])


def model_levels(mesh: Mesh, depth_m: np.ndarray, zbar_m: np.ndarray, thers_zbar_lev: int = 5):
    """FESOM3's levels of a mesh with node depths, as its partitioner fesom_ini finds them from aux3d.out (no cavity,
    the element's depth the mean of its nodes', FESOM3: github.com/koldunovn/fesom_F), and the element bottom the model
    uses. Every node's depth is raised to zbar[thers_zbar_lev - 1] at least; an element's level is the first layer
    whose mid-depth lies below the mean of its nodes' depths (the level surface nearest the mean), nl when none, at
    least thers_zbar_lev; then for each deeper level, the elements at that level or deeper with fewer than two edge
    neighbours there drop to the level above, until none (the 2-core of that layer, whatever the order); a node's
    level is the deepest of its elements'; an element's bottom is zbar at the shallowest of its nodes' levels.
    depth_m, zbar_m: metres, positive down. Returns (element levels (E,), node levels (N,), element bottom (E,) m),
    levels 1-based, elements in the mesh's order (triangles, then quads)."""
    zb = np.abs(np.asarray(zbar_m, dtype=np.float64))
    nl = len(zb)
    mid = 0.5 * (zb[:-1] + zb[1:])
    x = np.maximum(np.abs(np.asarray(depth_m, dtype=np.float64)), zb[thers_zbar_lev - 1])
    dm = np.r_[x[mesh.tris].mean(1) if mesh.n_tris else np.zeros(0), x[mesh.quads].mean(1)]
    k = np.searchsorted(mid, dm, side="right")           # mid-depths not below the mean: the first below is k (0-based)
    lev = np.maximum(np.where(k < nl - 1, k + 1, nl), thers_zbar_lev)
    ee = build_topology(mesh).edge_elems
    ee = ee[ee[:, 1] >= 0]
    E = len(dm)
    for nz in range(thers_zbar_lev + 1, nl + 1):
        while True:
            on = lev >= nz
            if not on.any():
                break
            both = on[ee[:, 0]] & on[ee[:, 1]]
            cnt = np.bincount(ee[both, 0], minlength=E) + np.bincount(ee[both, 1], minlength=E)
            iso = on & (cnt < 2)
            if not iso.any():
                break
            lev[iso] = nz - 1
    nlv = np.zeros(mesh.n_nodes, np.int64)
    T = mesh.n_tris
    for c, le in ((mesh.tris, lev[:T]), (mesh.quads, lev[T:])):
        for j in range(c.shape[1]):
            np.maximum.at(nlv, c[:, j], le)
    low = np.r_[nlv[mesh.tris].min(1) if T else np.zeros(0, np.int64), nlv[mesh.quads].min(1)]
    return lev, nlv, zb[low - 1]


def fesom2_checks(path: Path, euler_deg: tuple = (50.0, 15.0, -90.0), thers_zbar_lev: int = 5) -> dict:
    """The checks FESOM2's partitioner and model perform on a mesh directory (nod2d.out, elem2d.out, aux3d.out),
    recomputed from the files. `stop` lists the checks that make FESOM2 stop with the default namelist
    (force_rotation = .true., Euler angles 50/15/-90, thers_zbar_lev = 5): an element with < 2 neighbours, > 32
    elements or > 32 neighbour nodes at a node, nl < 3 or |depth| > 11000 m, a node inside the +-2.5 deg lon-lat box
    around the rotated North Pole (the "already rotated" abort), and an element containing a rotated pole (the metric
    is singular there). The rest is reported."""
    from meshom.geo import euler_matrix, rot_to_geo

    path = Path(path)
    mesh = read_fesom(path)
    topo = build_topology(mesh)
    n = mesh.n_nodes
    ncol = len(np.loadtxt(path / "elem2d.out", skiprows=1, max_rows=1, dtype=np.int64, ndmin=1))
    pairs = np.sort(np.vstack([topo.edges, mesh.quads[:, [0, 2]], mesh.quads[:, [1, 3]]]), axis=1)
    key = np.unique(pairs[:, 0] * n + pairs[:, 1])
    nb = np.bincount(np.r_[key // n, key % n], minlength=n)                # neighbour nodes, quad diagonals included
    bedge = topo.edges[topo.edge_elems[:, 1] < 0]
    bdeg = np.bincount(bedge.ravel(), minlength=n)
    zbar, depth = read_aux3d(path / "aux3d.out")
    m = euler_matrix(*euler_deg)
    poles = rot_to_geo(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]), m)  # rotated N / S pole in geographic xyz
    plon, plat = xyz_to_lonlat(poles)
    lon, lat = xyz_to_lonlat(mesh.points)
    dlon = (lon - plon[0] + 180.0) % 360.0 - 180.0
    in_box = (np.abs(dlon) <= 2.5) & (np.abs(lat - plat[0]) <= 2.5)
    P = mesh.points[np.vstack([mesh.tris, mesh.quads[:, [0, 1, 2]], mesh.quads[:, [0, 2, 3]]])]   # CCW triangles
    eid = np.r_[np.arange(mesh.n_tris), np.tile(np.arange(mesh.n_quads) + mesh.n_tris, 2)]       # their element

    def contains(p: np.ndarray) -> int:
        a, b, c = P[:, 0], P[:, 1], P[:, 2]
        s = [np.einsum("ij,ij->i", np.cross(u, v), np.broadcast_to(p, u.shape)) for u, v in ((a, b), (b, c), (c, a))]
        inside = (s[0] >= 0) & (s[1] >= 0) & (s[2] >= 0) & ((a + b + c) @ p > 0)
        return int(np.unique(eid[inside]).size)

    pole_deg = [float(np.degrees(np.arccos(np.clip((mesh.points @ p).max(), -1.0, 1.0)))) for p in poles]
    rec = {"n_nodes": n, "n_elems": mesh.n_elems, "elem2d_columns": ncol, "n_quads": mesh.n_quads,
           "unused_nodes": int((topo.elems_per_node == 0).sum()),
           "elems_lt2_neighbours": int((topo.elem_neighbours < 2).sum()),
           "elems_per_node_max": int(topo.elems_per_node.max()), "node_neighbours_max": int(nb.max()),
           "nodes_gt11_elems": int((topo.elems_per_node > 11).sum()),
           "boundary_nodes": int(topo.boundary_node.sum()),
           "flag1_nodes": int((mesh.node_flag == 1).sum()),
           "flag_equals_boundary": bool(np.array_equal(mesh.node_flag == 1, topo.boundary_node)),
           "tris_3_boundary_nodes": int(topo.boundary_node[mesh.tris].all(axis=1).sum()),
           "boundary_nodes_gt2_boundary_edges": int((bdeg > 2).sum()),
           "nl": len(zbar), "zbar_increasing": bool(np.all(np.diff(zbar) > 0)), "zbar_max_m": float(zbar.max()),
           "aux3d_depths": len(depth), "depth_min_m": float(depth.min()), "depth_max_m": float(depth.max()),
           f"nodes_shallower_than_zbar{thers_zbar_lev}": int((depth < zbar[thers_zbar_lev - 1]).sum()),
           "zbar_threshold_m": float(zbar[thers_zbar_lev - 1]),
           "nodes_deeper_than_last_level": int((depth > zbar[-1]).sum()),
           "rotated_north_pole_lonlat": [round(float(plon[0]), 4), round(float(plat[0]), 4)],
           "rotated_south_pole_lonlat": [round(float(plon[1]), 4), round(float(plat[1]), 4)],
           "nodes_in_rotated_pole_box": int(in_box.sum()),
           "elems_containing_rotated_north_pole": contains(poles[0]),
           "elems_containing_rotated_south_pole": contains(poles[1]),
           "nearest_node_to_rotated_poles_deg": [round(d, 3) for d in pole_deg],
           "nodes_north_of_88": int((lat > 88.0).sum())}
    stop = [k for k, bad in [("elems_lt2_neighbours", rec["elems_lt2_neighbours"] > 0),
                             ("elems_per_node_max", rec["elems_per_node_max"] > 32),
                             ("node_neighbours_max", rec["node_neighbours_max"] > 32),
                             ("nl", rec["nl"] < 3), ("depth_max_m", rec["depth_max_m"] > 11000.0),
                             ("aux3d_depths", rec["aux3d_depths"] != n),
                             ("nodes_in_rotated_pole_box", rec["nodes_in_rotated_pole_box"] > 0),
                             ("elems_containing_rotated_north_pole", rec["elems_containing_rotated_north_pole"] > 0),
                             ("elems_containing_rotated_south_pole", rec["elems_containing_rotated_south_pole"] > 0)]
            if bad]
    rec["stop"] = stop
    return rec


def fesom3_checks(path: Path, euler_deg: tuple = (50.0, 15.0, -90.0), thers_zbar_lev: int = 5) -> dict:
    """The checks FESOM3 (fesom_ini and the model's mesh reader) performs on a mesh directory, recomputed from the
    files; counts only, nothing is repaired. The record is fesom2_checks' (FESOM2-only fields such as the pole box
    stay as information) plus the element order: on the sphere from the signed turn of every corner, as FESOM3's mesh
    check does (`quads_nonconvex_sphere`: a quad whose four turns differ in sign; `elems_not_clockwise`: any other
    element not turning clockwise at every corner, i.e. reversed or degenerate; both are errors there, and a quad can
    be non-convex on the sphere while convex in rotated lon-lat) and fesom_ini's element test in rotated lon-lat
    (anticlockwise elements are flipped, a quad whose sub-triangles (1,2,3), (1,3,4) differ in sense is a bow-tie, one
    whose (1,2,4) or (2,3,4) differs from (1,2,3) is non-convex; both STOP). `stop`: an element with < 2
    neighbours, > 32 elements at a node, > 32 neighbour nodes with quad diagonals, a bow-tie or non-convex quad in
    rotated lon-lat, a quad non-convex or an element not clockwise on the sphere, nl < thers_zbar_lev (fesom_ini
    reads zbar(thers_zbar_lev)), a node count != N in aux3d.out, an element containing a rotated pole."""
    from meshom.geo import euler_matrix, geo_to_rot

    path = Path(path)
    rec = fesom2_checks(path, euler_deg, thers_zbar_lev)
    nod = np.loadtxt(path / "nod2d.out", skiprows=1, max_rows=rec["n_nodes"], ndmin=2)
    raw = np.loadtxt(path / "elem2d.out", skiprows=1, max_rows=rec["n_elems"], dtype=np.int64, ndmin=2) - 1
    if raw.shape[1] == 3:
        raw = np.column_stack([raw, raw[:, 0]])
    quad = raw[:, 3] != raw[:, 0]
    p = lonlat_to_xyz(nod[:, 1], nod[:, 2])
    P = p[raw]
    turn = np.empty((len(raw), 4))                                       # signed corner turns, < 0 = clockwise
    for sel, k in ((quad, 4), (~quad, 3)):
        q = P[sel, :k]
        e_in, e_out = q - np.roll(q, 1, axis=1), np.roll(q, -1, axis=1) - q
        turn[sel, :k] = np.einsum("ijk,ijk->ij", np.cross(e_in, e_out), q)
        turn[sel, k:] = turn[sel, :1]                                    # a triangle's 4th slot = its 1st corner
    cw, ccw = (turn < 0).all(axis=1), (turn > 0).all(axis=1)
    nonconvex_sphere = quad & ~cw & ~ccw
    not_cw = ~cw & ~nonconvex_sphere
    rlon, rlat = xyz_to_lonlat(geo_to_rot(p, euler_matrix(*euler_deg)))
    X = np.stack([rlon, rlat], axis=1)[raw]
    b, c, d = (X[:, k] - X[:, 0] for k in (1, 2, 3))
    for v in (b, c, d):
        v[:, 0] = (v[:, 0] + 180.0) % 360.0 - 180.0                       # cyclic_length / 2

    def cr(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    r, r1, r2, r3 = cr(b, c), cr(c, d), cr(b, d), cr(c - b, d - b)
    bowtie = quad & (r * r1 <= 0)
    nonconvex = quad & ~bowtie & ((r * r2 <= 0) | (r * r3 <= 0))
    rec.update(elems_not_clockwise=int(not_cw.sum()), quads_nonconvex_sphere=int(nonconvex_sphere.sum()),
               elems_anticlockwise_rotated=int(((r > 0) & ~bowtie & ~nonconvex).sum()),
               quads_bowtie_rotated=int(bowtie.sum()), quads_nonconvex_rotated=int(nonconvex.sum()))
    rec["stop"] = [k for k, bad in [("elems_lt2_neighbours", rec["elems_lt2_neighbours"] > 0),
                                    ("elems_per_node_max", rec["elems_per_node_max"] > 32),
                                    ("node_neighbours_max", rec["node_neighbours_max"] > 32),
                                    ("quads_bowtie_rotated", rec["quads_bowtie_rotated"] > 0),
                                    ("quads_nonconvex_rotated", rec["quads_nonconvex_rotated"] > 0),
                                    ("quads_nonconvex_sphere", rec["quads_nonconvex_sphere"] > 0),
                                    ("elems_not_clockwise", rec["elems_not_clockwise"] > 0),
                                    ("nl", rec["nl"] < thers_zbar_lev),
                                    ("aux3d_depths", rec["aux3d_depths"] != rec["n_nodes"]),
                                    ("elems_containing_rotated_north_pole",
                                     rec["elems_containing_rotated_north_pole"] > 0),
                                    ("elems_containing_rotated_south_pole",
                                     rec["elems_containing_rotated_south_pole"] > 0)] if bad]
    return rec


def run_fesom_ini(binary: str, mesh_dir: Path, work_dir: Path, n_part: int = 1) -> dict:
    """Run FESOM3's partitioner fesom_ini on a written FESOM3 folder: namelist.config with MeshPath and n_part only,
    the log in work_dir/fesom_ini.out, the stack limit raised to the hard limit. fesom_ini exits 0 also when it stops
    on an error, so the record keeps the stop lines of the log and whether the derived files (edges, node levels)
    exist. With 1 partition it writes them, then stops at the partitioning ("less than one"): all a 1-rank run
    needs. `ok` = exit 0, derived files present, no ERROR line."""
    import resource
    import subprocess
    import time

    mesh_dir, work = Path(mesh_dir).resolve(), Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    (work / "namelist.config").write_text(f'&paths\nMeshPath="{mesh_dir}/"\n/\n&geometry\n/\n&run_config\n/\n'
                                          f'&machine\nn_levels=1\nn_part={n_part}\n/\n')

    def big_stack() -> None:
        hard = resource.getrlimit(resource.RLIMIT_STACK)[1]
        resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))

    t0 = time.perf_counter()
    with open(work / "fesom_ini.out", "w") as f:
        rc = subprocess.run([str(binary)], cwd=work, stdout=f, stderr=subprocess.STDOUT,
                            preexec_fn=big_stack).returncode
    log = (work / "fesom_ini.out").read_text(errors="replace").splitlines()
    stops = [ln.strip() for ln in log if any(k in ln for k in ("ERROR", "Recompile", "Aborting", "Insufficient"))]
    derived = all((mesh_dir / f).exists() for f in ("edgenum.out", "edges.out", "edge_tri.out", "nlvls.out"))
    return {"binary": str(binary), "n_part": n_part, "exit_status": rc, "wall_s": round(time.perf_counter() - t0, 2),
            "stop_lines": stops[:10], "derived_files": derived,
            "total_time": next((ln.strip() for ln in log if "Total time" in ln), None),
            "log": str(work / "fesom_ini.out"),
            "ok": rc == 0 and derived and not any("ERROR" in ln or "Recompile" in ln for ln in stops)}
