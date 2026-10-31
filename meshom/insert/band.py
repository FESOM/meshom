"""Band inserts: a coastal strip and a current corridor (decisions B181, B182).

Both are built from band blocks (meshom.insert.bandblock): row 0 is a "spine" with nodes of its own at the fine size
(the real coast, or the corridor's axis) between P and Q, row K a host edge chain (fixed host nodes); the rows
between are offset from the spine at graded distances and wrap round P and Q by a quarter turn down to the spine's
continuation (the bank: the coast or the axis beyond P and Q), so in the middle they run along the spine (a band of
fine rows) and the size grows to the host's at the 15 % rule (h_fine over `band`, then 0.12 per unit distance).

- strip: the coast between the host coast nodes A0 and B0 nearest to the path's ends; the host chain follows the
  coast offset by `width` out to sea (an end join out from A0, the offshore join, an end join back to B0), found as
  the joins of a strait are (meshom.insert.region.Chains.path); the spine = the real coast smoothed at the fine size
  (meshom.insert.realcoast; pockets narrower than two cells bridged, wider bays kept, or all bridged with
  bays = false) between P and Q, placed about `width` along the coast from A0 and B0 where the coast turns least.
- corridor: an open-water region along the path (the axis), closed by a loop of host edges: the left and the right
  chain from the host node A at the path's start to B at its end, each following the axis offset by width / 2; two
  band blocks share the axis between P and Q as their spine and their row ends beyond.

Folds of the explicit rows (a rough coast) are untangled, the free nodes smoothed (meshom.smooth, kind "size"); the
ring cleaning, the size floor, the gates and the scores are those of a strait (meshom.insert.core).

A strip's rows come by default from a harmonic map of the band (meshom.insert.harmonic, B195-B198: rows that cannot
cross, columns along its field lines, the corners above the target polished); where quads next to the coast keep a
corner above COAST_DEG the coast is smoothed more there and the band rebuilt (`_resmooth`, B198). rows = "offset"
keeps the offset rows above (B181); corridors use them.
"""
import time

import numpy as np
from scipy.spatial import cKDTree

from meshom.insert import assemble, bandblock, block, measure, network, region
from meshom.insert.core import Result, collar_record, host_size, _path_frame, _small
from meshom.insert.geom import Chart, Polyline, resample
from meshom.insert.host import Window, coast_neighbours, graph
from meshom.insert.region import RATIO, G_RULE, InsertError
from meshom.mesh import build_topology

END_W = 1.0                  # B181: the banks (spine's continuation from row 0's ends to P / Q) are this x width long
                             # (strip; corridor: x width / 2): the rows turn round P and Q like a half O-grid
END_SPAN = (0.9, 1.3)        # B181: P (Q) is placed in this range x the bank length where the coast turns least
SMOOTH_ITERS = 20            # B181: sweeps of meshom.smooth (kind "size", smart guard) on the band's free nodes
BAND_MIN_H = 2.0             # B181: a band narrower than this many fine cells is refused (no fine band left)
COAST_DEG = 125.0            # B198: a quad next to the coast with a larger corner asks for more coast smoothing there
COAST_ROWS = 2               # B198: next to the coast = all nodes within this many rows of the spine (or on a bank)
RESMOOTH_W = 4.0             # B198: the extra smoothing's window (Gaussian), in fine cells
RESMOOTH_STEP = 1.0          # B198: each round adds this x the base width to the smoothing width in a window
SHIFT_STEP = 0.25           # B198: the coast's move from the extra smoothing sampled every this x the fine size
OFFSET_SIGMA = 0.5           # B181: the coast (axis) is smoothed with a Gaussian of this x the offset before the
                             # offset curve that row 0 follows is drawn


def _smooth_open(xy: np.ndarray, sigma: float, step: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    q = resample(xy, step)
    if sigma <= 0 or len(q) < 4:
        return q
    return gaussian_filter1d(q, sigma / step, axis=0, mode="nearest")


def _normals(q: np.ndarray) -> np.ndarray:
    t = np.gradient(q, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    return np.c_[-t[:, 1], t[:, 0]]


def offset_shape(line: np.ndarray, d: float, side: float, step: float) -> np.ndarray:
    """The polyline `line` smoothed (OFFSET_SIGMA x d), offset by d to its left (side +1) or right (-1), with the
    two end legs back to the line's end points: the shape a band's row 0 follows."""
    q = _smooth_open(line, OFFSET_SIGMA * d, step)
    off = q + side * d * _normals(q)
    k = max(int(np.ceil(d / step)), 2)
    t = np.linspace(0, 1, k + 1)[:, None]
    leg_a = line[0] + t[:-1] * (off[0] - line[0])
    leg_b = off[-1] + t[1:] * (line[-1] - off[-1])
    return np.vstack([leg_a, off, leg_b])


def calm(pl: Polyline, e: float, W: float) -> float:
    """B181: where the spine starts: the arc length in [END_SPAN] x e from the polyline's start where the coast turns
    least over +-W / 4 (the rows turn round P there: a cape or a bay at P folds them)."""
    best, s_best = np.inf, e
    for s in np.linspace(END_SPAN[0] * e, min(END_SPAN[1] * e, 0.45 * pl.L), 13):
        a, b, c = pl.at(np.clip([s - 0.25 * W, s, s + 0.25 * W], 0, pl.L))
        u, v = b - a, c - b
        t = abs(float(np.arctan2(u[0] * v[1] - u[1] * v[0], u @ v)))
        if t < best - 1e-9:
            best, s_best = t, float(s)
    return s_best


def coast_between(mesh, reg: np.ndarray, chain: np.ndarray) -> np.ndarray:
    """The host coast node path of the region's boundary from chain[0] to chain[-1] (the boundary minus the chain)."""
    from meshom.mesh import Mesh

    T = mesh.n_tris
    sub = Mesh(mesh.points, mesh.tris[reg[:T]], mesh.quads[reg[T:]], mesh.node_flag)
    st = build_topology(sub)
    be = st.edges[st.edge_elems[:, 1] < 0]
    n = mesh.n_nodes
    walls = assemble._key(np.c_[chain[:-1], chain[1:]], n)
    be = be[~np.isin(assemble._key(be, n), walls)]
    nb = {}
    for x, y in be:
        nb.setdefault(int(x), []).append(int(y))
        nb.setdefault(int(y), []).append(int(x))
    path, prev = [int(chain[0])], -1
    while path[-1] != int(chain[-1]):
        nxt = [v for v in nb.get(path[-1], []) if v != prev]
        if not nxt or len(path) > len(be) + 2:
            raise InsertError("the region's coast does not connect the strip's join ends (a pinch node or an island)")
        prev = path[-1]
        path.append(nxt[0])
    return np.array(path)


def _fine(spec, h_host, h_min, rec, log):
    hf = spec.h_fine_m if spec.h_fine_m else max(h_min, h_host / 3.0)
    note = ""
    if h_min > 0 and hf < h_min:
        note = f"h_fine_m {hf / 1e3:.2f} km is below the size floor h_min_m {h_min / 1e3:.2f} km: using the floor"
        hf = h_min
        log(f"[insert {spec.name}] FLOOR: {note}", flush=True)
    rec["profile"]["h_fine_km"] = hf / 1e3
    rec["profile"]["floor"] = note or None
    return hf


def _room(spec, h_f, h_host, depth, rec, what):
    """B181: the band's room rule: RATIO (h_f + ln 1.15 (depth - band)) >= h_host, depth = the distance from the
    spine to row 0 (strip: width; corridor: width / 2). The default band: what the growth at the layout slope
    leaves. Returns (band, refusal or None)."""
    grow = (h_host - h_f) / block.G_LAYOUT
    band = spec.band_m if spec.band_m is not None else max(depth - grow, 0.0)
    if spec.kind == "corridor" and spec.band_m is not None:
        band = 0.5 * spec.band_m
    room = RATIO * (h_f + G_RULE * max(depth - band, 0.0)) / h_host
    rec["profile"].update(band_km=band / 1e3, room=room, grow_km=grow / 1e3, depth_km=depth / 1e3)
    if room < 1.0:
        need = band + (h_host / RATIO - h_f) / G_RULE
        hf_ok = h_host / RATIO - G_RULE * (depth - band)
        return band, (f"room: the size grows from {h_f / 1e3:.2f} km only to {room * h_host / RATIO / 1e3:.2f} km at "
                      f"the host chain (host {h_host / 1e3:.2f} km, {what} {depth / 1e3:.0f} km, band "
                      f"{band / 1e3:.0f} km); fix: width_km >= {(need if spec.kind == 'strip' else 2 * need) / 1e3:.0f}"
                      f", a narrower band_km, or h_fine_m >= {hf_ok:.0f} m")
    if band < BAND_MIN_H * h_f:
        return band, (f"room: the band of fine rows is {band / 1e3:.1f} km, below {BAND_MIN_H:.0f} fine cells "
                      f"({h_f / 1e3:.2f} km); fix: a wider width_km or a coarser h_fine_m")
    return band, None


def _corner_angles(xy_chain: np.ndarray, coast_xy: np.ndarray, h_join: float) -> list:
    """The angle at which the chain's first chord (over CORNER_CHORD_H join cells) meets the coast (the chord over
    BANK_CHORD_H join cells along coast_xy, which starts at the chain's end node)."""
    from meshom.insert.region import BANK_CHORD_H, CORNER_CHORD_H, _chain_point

    u = _chain_point(xy_chain, CORNER_CHORD_H * h_join) - xy_chain[0]
    v = _chain_point(coast_xy, BANK_CHORD_H * h_join) - coast_xy[0]
    c = float(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def insert_band(win: Window, spec, h_run, tol=0.20, gates=None, log=print, real=None,
                h_min: float | None = None, orca: bool | None = None) -> Result:
    """A coastal strip or a current corridor in a host window (module docstring); the record as insert_window's."""
    from meshom import straits as stm
    from meshom.insert import realcoast as rcm

    t0 = time.perf_counter()
    rec = {"name": spec.name, "kind": spec.kind, "timing": {}, "profile": {}}
    mesh = win.mesh
    c, half = _path_frame(spec.path)
    chart = Chart(c)
    xy = chart.fwd(mesh.points)
    seg = xy[win.coast_edges]
    path_xy = chart.fwd(spec.path)
    h_host = host_size(mesh, stm.densify(spec.path, 1000.0), max(float(spec.width_m), 3000.0))   # near the band
    h_min = float(spec.h_min_m if spec.h_min_m is not None else (h_min or 0.0))
    rec["profile"].update(h_host_km=h_host / 1e3, h_min_km=h_min / 1e3, width_km=spec.width_m / 1e3)
    hf = _fine(spec, h_host, h_min, rec, log)
    res = Result(False, rec, host_window=win, chart=chart)
    W = float(spec.width_m)
    depth = W if spec.kind == "strip" else 0.5 * W
    band, why = _room(spec, hf, h_host, depth, rec, "width" if spec.kind == "strip" else "half width")
    log(f"[insert {spec.name}] {spec.kind}: fine {hf / 1e3:.2f} km, host {h_host / 1e3:.2f} km, floor "
        f"{h_min / 1e3:.2f} km, width {W / 1e3:.0f} km, band {band / 1e3:.1f} km, room "
        f"{rec['profile']['room']:.2f}", flush=True)
    if hf * RATIO > h_host:
        rec["refused"] = [f"no zoom: the fine size {hf / 1e3:.2f} km is not finer than the host's {h_host / 1e3:.2f} "
                          f"km by the ratio {RATIO}; fix: a smaller h_fine_m (and h_min_m)"]
    elif why:
        rec["refused"] = [why]
    if rec.get("refused"):
        log(f"[insert {spec.name}] REFUSED {rec['refused'][0]}", flush=True)
        return res
    g = graph(mesh)
    ch = region.Chains(g, xy, coast_neighbours(win.coast_edges, mesh.n_nodes), mesh.quads)
    rings, host_fixed, rec["host"] = assemble.collar_of(spec.collar_rings, g, orca)
    if rec["host"]["structured"]:
        ch.turn_w = region.TURN_W_GRID                  # B193: chains along long grid-line runs on ORCA
    eidx = {(int(a), int(b)): k for k, (a, b) in enumerate(g.edges)}

    def sphere_len(chain):
        return float(sum(g.length_m[eidx[(min(a, b), max(a, b))]] for a, b in zip(chain[:-1], chain[1:])))

    tree = stm._elem_tree(mesh)
    in_water = lambda o: stm.locate(mesh, chart.inv(o), tree) >= 0          # noqa: E731
    step = 0.25 * min(hf, h_host)
    rc = None
    if spec.kind == "strip":
        if real is None:
            raise InsertError("a coastal strip needs the real coast (coast = \"real\" and the run's coast)")
        win_r = float(np.max(np.linalg.norm(xy, axis=1))) / chart.scale(np.zeros((1, 2)))[0]
        rc = rcm.RealCoast.from_loops(real, chart, c, win_r + 2 * h_host, spec.coast_source or "run")
        rc.prepare(lambda q: np.full(len(q), hf), lambda q: np.full(len(q), np.inf), hf, spec.bank_sigma_cells,
                   spec.bays, bridge="narrow")   # B181 / B190: the pockets not meshed as bays bridged (bays =
                                                 # false: every pocket, its water left out)
        rec["real_coast"] = {"source": rc.source, "pieces": len(rc.pieces),
                             "bridged": [b.record() for b in rc.bays if not b.meshed]}
        coast_nodes = np.unique(win.coast_edges)
        kd = cKDTree(xy[coast_nodes])
        A0 = int(coast_nodes[kd.query(path_xy[0])[1]])
        B0 = int(coast_nodes[kd.query(path_xy[-1])[1]])
        arc0, why = rc.arc(xy[A0], xy[B0], guide=path_xy)
        if arc0 is None:
            rec["refused"] = [f"the real coast between the strip's ends: {why}"]
            log(f"[insert {spec.name}] REFUSED {rec['refused'][0]}", flush=True)
            return res
        q = _smooth_open(arc0, OFFSET_SIGMA * W, step)
        nrm = _normals(q)
        k = np.linspace(0, len(q) - 1, 40).astype(int)
        side = 1.0 if in_water(q[k] + 0.5 * W * nrm[k]).sum() >= in_water(q[k] - 0.5 * W * nrm[k]).sum() else -1.0
        shape = offset_shape(np.vstack([xy[A0], arc0[1:-1], xy[B0]]), W, side, step)
        chain = ch.path(A0, B0, Polyline(shape))
        if chain is None:
            rec["refused"] = ["no host edge chain follows the strip's offshore outline (it leaves the window or "
                              "meets the coast); fix: a narrower width_km or a shorter path"]
            log(f"[insert {spec.name}] REFUSED {rec['refused'][0]}", flush=True)
            return res
        chains = [chain]
        mid = Polyline(q).at(0.5 * Polyline(q).L)[0] + side * 0.5 * W * nrm[len(q) // 2]
    else:
        ax = Polyline(resample(path_xy, step))
        nd = np.flatnonzero(~g.boundary)
        kd = cKDTree(xy[nd])
        A0 = int(nd[kd.query(ax.xy[0])[1]])
        B0 = int(nd[kd.query(ax.xy[-1])[1]])
        axis_xy = np.vstack([xy[A0], ax.xy[1:-1], xy[B0]])
        chains = []
        for sd in (1.0, -1.0):
            shape = offset_shape(axis_xy, 0.5 * W, sd, step)
            used = set(int(v) for cc in chains for v in cc[1:-1])
            cc = ch.path(A0, B0, Polyline(shape), exclude=used)
            if cc is None:
                rec["refused"] = [f"no host edge chain follows the corridor's {'left' if sd > 0 else 'right'} side "
                                  "in open water (it meets the coast or leaves the window); fix: a narrower width_km "
                                  "or a path farther from the coast"]
                log(f"[insert {spec.name}] REFUSED {rec['refused'][0]}", flush=True)
                return res
            chains.append(cc)
        mid = Polyline(axis_xy).at(0.5 * Polyline(axis_xy).L)[0]
    hj = [sphere_len(cc) / (len(cc) - 1) for cc in chains]
    rec["joins"] = [{"n": int(len(cc) - 1), "h_join_km": round(h / 1e3, 3), "tier": int(ch.tier(cc)),
                     "zig": round(ch.zig(cc), 3)} for cc, h in zip(chains, hj)]
    seed = int(stm.locate(mesh, chart.inv(mid[None]), tree)[0])
    if spec.kind == "strip":                                # the cell on the coast side of the chain's middle edge
        k = len(chain) // 2
        a_, b_ = int(chain[k - 1]), int(chain[k])
        T_ = mesh.n_tris
        cells = [(i, t) for i, t in enumerate(mesh.tris) if a_ in t and b_ in t] + \
                [(T_ + i, q) for i, q in enumerate(mesh.quads) if a_ in q and b_ in q]
        if cells:
            cpl = Polyline(arc0)
            seed = min(cells, key=lambda it: float(cpl.distance(xy[it[1]].mean(0)[None])[0]))[0]
    if seed < 0:
        raise InsertError("the band's middle is not in the host's water")
    reg = assemble.region_between(mesh, seed, chains, win.kind)
    rec["timing"]["region_s"] = time.perf_counter() - t0
    t = time.perf_counter()
    host_bank = coast_between(mesh, reg, chain) if spec.kind == "strip" else None

    def publish(diag):
        """A build's diagnostics into the record (only the kept build's stay there)."""
        diag = dict(diag)
        if "corner_deg" in diag:
            rec["joins"][0]["corner_deg"] = diag.pop("corner_deg")
        rec.update(diag)

    def strip_coast(rc, diag):
        """The strip's coast from rc: (arc, spine polyline, bank before P, bank beyond Q), None when refused (the
        reason in rec["refused"])."""
        arc, why = rc.arc(xy[A0], xy[B0], guide=xy[host_bank])
        if arc is None or Polyline(arc).L > 3.0 * Polyline(xy[host_bank]).L + 4 * h_host:
            rec["refused"] = [f"the real coast of the strip does not follow the host's ({why or 'far longer'})"]
            return None
        arc = rcm.blend_ends(arc, xy[A0], xy[B0], rcm.BLEND_H * hj[0], rcm.BLEND_H * hj[0])
        pl = Polyline(arc)
        e = min(END_W * W, 0.3 * pl.L)
        sP, sQ = calm(pl, e, W), pl.L - calm(Polyline(pl.xy[::-1]), e, W)
        ang = [_corner_angles(xy[chain], pl.at(np.linspace(0, e, 20)), hj[0]),
               _corner_angles(xy[chain][::-1], pl.at(np.linspace(pl.L, sQ, 20)), hj[0])]
        diag["corner_deg"] = [round(a, 1) for a in ang]
        bad = [a for a in ang if not region.CORNER_DEG[0] <= a <= region.CORNER_DEG[1]]
        if bad and not spec.force:
            rec["refused"] = [f"corner: the end joins meet the coast at {ang[0]:.0f} / {ang[1]:.0f} deg (need "
                              f"{region.CORNER_DEG[0]:.0f}-{region.CORNER_DEG[1]:.0f}); fix: move the path's ends "
                              "along the coast, or force"]
            return None
        spine_pl = Polyline(np.vstack([pl.at(sP), pl.xy[(pl.s > sP) & (pl.s < sQ)], pl.at(sQ)]))
        bank_p = Polyline(np.vstack([pl.xy[pl.s < sP], pl.at(sP)]))
        bank_q = Polyline(np.vstack([pl.at(sQ), pl.xy[pl.s > sQ]]))
        return arc, spine_pl, bank_p, bank_q

    def spine_of(spine_pl):
        n_sp = max(2, int(round(spine_pl.L / hf)))
        return n_sp, spine_pl.at(np.linspace(0, spine_pl.L, n_sp + 1))

    def coast_of(rc):
        """B198: the strip's whole coast as the band takes it (bank before P, spine, bank beyond Q), None when
        refused."""
        parts = strip_coast(rc, {})
        if parts is None:
            rec.pop("refused", None)
            return None
        _, spine_pl, bank_p, bank_q = parts
        return np.vstack([bank_p.xy[:-1], spine_of(spine_pl)[1], bank_q.xy[1:]])

    def build_band(rc):
        """The band block from the real coast rc (strip) or the axis (corridor): None when refused (the reason in
        rec["refused"], the build's diagnostics published). The build's diagnostics travel with it (diag)."""
        diag = {}
        if spec.kind == "strip":
            parts = strip_coast(rc, diag)
            if parts is None:
                publish(diag)
                log(f"[insert {spec.name}] REFUSED {rec['refused'][0]}", flush=True)
                return None
            arc, spine_pl, bank_p, bank_q = parts
            sides = [(side, chain)]
        else:
            L_ax = Polyline(axis_xy).L
            e = min(END_W * 0.5 * W, 0.3 * L_ax)
            pl = Polyline(axis_xy)
            spine_pl = Polyline(np.vstack([pl.at(e), pl.xy[(pl.s > e) & (pl.s < L_ax - e)], pl.at(L_ax - e)]))
            bank_p = Polyline(np.vstack([pl.xy[pl.s < e], pl.at(e)]))
            bank_q = Polyline(np.vstack([pl.at(L_ax - e), pl.xy[pl.s > L_ax - e]]))
            sides = [(1.0, chains[0]), (-1.0, chains[1])]
        n_sp, spine = spine_of(spine_pl)
        d_lv, h_lv = bandblock.band_levels(depth, hf, band, h_host)
        rows_kind = spec.rows if spec.kind == "strip" else "offset"
        diag["rows"] = rows_kind
        if rows_kind == "harmonic":                             # B195-B198: rows and columns of a harmonic map
            from meshom.insert import harmonic

            try:
                hm = harmonic.harmonic_map(spine, bank_p, bank_q, xy[chain], hf, chart)
            except harmonic.MapError as ex:                     # B195: no inverse without a one-to-one map
                rec["refused"] = [f"{ex}; fix: a smaller h_fine_m or a smoother coast (bank_sigma_cells)"]
                publish(diag)
                log(f"[insert {spec.name}] REFUSED {rec['refused'][0]}", flush=True)
                return None
            blocks = [harmonic.build_harmonic(hm, spine, xy[chain], d_lv, h_lv, band)]
        else:
            blocks = [bandblock.build_side(bandblock.BandMap(spine, sd, depth, bank_p, bank_q, 0.25 * hf), spine,
                                           xy[cc], d_lv, h_lv, stair=host_fixed) for sd, cc in sides]
        b0 = blocks[0]
        K = b0.K
        xyn, rows_of, hn = b0.xy, b0.row, b0.h[b0.row]
        qd, tr = b0.quads, b0.tris
        ids_a = np.arange(b0.off[K], b0.off[K + 1])
        ids_b = np.zeros(0, np.int64)
        if len(blocks) > 1:                                     # the corridor's right half: the spine and the row ends
            b1 = blocks[1]                                      # are the left half's nodes (B182)
            loc = np.full(len(b1.xy), -1, np.int64)
            loc[b1.off[0]:b1.off[1]] = np.arange(b0.off[0], b0.off[1])
            for k in range(1, K + 1):
                loc[b1.off[k]] = b0.off[k]
                loc[b1.off[k + 1] - 1] = b0.off[k + 1] - 1
            fresh = loc < 0
            loc[fresh] = len(xyn) + np.arange(fresh.sum())
            xyn = np.vstack([xyn, b1.xy[fresh]])
            rows_of = np.r_[rows_of, b1.row[fresh]]
            hn = np.r_[hn, b1.h[b1.row[fresh]]]
            qd = np.vstack([qd, loc[b1.quads][:, ::-1]])           # the right half is the mirror: same orientation
            tr = np.vstack([tr, loc[b1.tris][:, ::-1]])
            ids_b = loc[np.arange(b1.off[K], b1.off[K + 1])]
            diag["halves"] = {"left": b0.info["counts"], "right": b1.info["counts"]}
        fixed = (rows_of == 0) | (rows_of == K)                 # B181: untangle folds of the explicit rows
        for b_ in blocks[:1]:
            ends = np.r_[b_.off[1:K], b_.off[2:K + 1] - 1]
            fixed[ends] = True
        xyn, diag["untangle"] = bandblock.untangle(xyn, qd, tr, fixed)
        qd, tr = bandblock.orient_all(xyn, qd, tr)          # B186: one orientation for the whole band, never per cell
        if SMOOTH_ITERS:                                        # B181: a size-aware smoothing of the band's free nodes
            from meshom.mesh import Mesh
            from meshom.smooth import smooth

            h_bl = assemble.InsertH(h_run, chart.inv(xyn), hn / chart.scale(xyn), block.G_LAYOUT)
            bm_ = Mesh(chart.inv(xyn), tr, qd, np.zeros(len(xyn), np.int8))
            xyn = chart.fwd(smooth(bm_, h_bl, fixed, iters=SMOOTH_ITERS, kind="size").points)
        xyn, diag["unfold"] = bandblock.unfold(xyn, qd, tr, fixed)   # B186: what still folds, repaired in place
        if rows_kind == "harmonic" and harmonic.POLISH_ROUNDS:   # B197: the corners above the target polished
            judge = rows_of[qd].max(1) <= K - spec.ring_rows
            xyn, diag["polish"] = harmonic.polish(xyn, qd, tr, fixed, harmonic.POLISH_DEG,
                                                 np.r_[judge, np.zeros(len(tr), bool)])
        return dict(arc=arc if spec.kind == "strip" else None, spine=spine, n_sp=n_sp, blocks=blocks, K=K,
                    xyn=xyn, rows_of=rows_of, hn=hn, qd=qd, tr=tr, ids_a=ids_a, ids_b=ids_b, fixed=fixed,
                    diag=diag, coast=(np.vstack([bank_p.xy[:-1], spine, bank_q.xy[1:]]) if spec.kind == "strip"
                                      else None))

    bb = build_band(rc)
    if bb is None:
        return res
    if spec.kind == "strip" and bb["diag"]["rows"] == "harmonic" and spec.coast_rounds:
        bb = _resmooth(bb, rc, spec, hf, build_band, coast_of, rec, log, K_ring=spec.ring_rows)
    publish(bb["diag"])                                     # the kept build's diagnostics only
    arc, spine, n_sp, blocks, K = bb["arc"], bb["spine"], bb["n_sp"], bb["blocks"], bb["K"]
    xyn, rows_of, hn, qd, tr = bb["xyn"], bb["rows_of"], bb["hn"], bb["qd"], bb["tr"]
    ids_a, ids_b, fixed = bb["ids_a"], bb["ids_b"], bb["fixed"]
    b0 = blocks[0]
    rq = rows_of[qd].max(1) if len(qd) else np.zeros(0, int)
    rt = rows_of[tr].max(1) if len(tr) else np.zeros(0, int)
    core_q, core_t = rq <= K - spec.ring_rows, rt <= K - spec.ring_rows
    net = network.Net(xyn, qd, tr, ids_a, ids_b, core_q, core_t, None, [],
                      (xyn, hn / chart.scale(xyn)), None, None, 0)
    outline = np.vstack([xy[chains[0]], (arc[::-1][1:-1] if spec.kind == "strip" else xy[chains[1]][::-1][1:-1])])
    bank_out = arc if spec.kind == "strip" else None
    rec["timing"]["block_s"] = time.perf_counter() - t
    rec["block"] = dict(b0.info, blocks=len(blocks), spine_cells=n_sp)
    rec["predict"] = {}
    log(f"[insert {spec.name}] {len(blocks)} band block(s): {K} rows, counts {b0.N.min()}-{b0.N.max()}, spine "
        f"{n_sp} cells of {hf / 1e3:.2f} km, {len(qd)} quads, {len(tr)} triangles ({rec['timing']['block_s']:.0f} s)",
        flush=True)
    if spec.stages:
        rec["released_cells"] = network.release_bad(net)
    new_xyz = chart.inv(net.xy)
    cA = chains[0]
    cB = chains[1] if len(chains) > 1 else np.zeros(0, np.int64)
    m = assemble.merge_net(mesh, reg, cA, cB, net, new_xyz)
    if spec.kind == "corridor":                             # B182: the loop has no coast ends to hold
        m.join_ends = None
    rec["merge"] = {"removed_host_elems": int(reg.sum()), "block_quads": int(m.block_q.sum()),
                    "block_tris": int(m.block_t.sum()), "inverted_block_cells": m.inverted}
    sp, sh = net.samples
    h_ins = assemble.InsertH(h_run, chart.inv(sp), sh, block.G_LAYOUT)
    cut_xyz = mesh.points[win.kind == 2]
    keys, _ = assemble.coast_keys_of(m.mesh, cut_xyz)
    host_seg = np.stack([mesh.points[win.coast_edges[:, 0]], mesh.points[win.coast_edges[:, 1]]], axis=1)
    t = time.perf_counter()
    stages = spec.stages
    if spec.kind == "corridor":                             # B182: no coast in the ring, no coast stage
        stages = ",".join(x for x in stages.split(",") if x.strip() and x.strip() != "e")
    out, crec, case, ring_out = assemble.clean(m, keys, h_ins, tol, gates, stages, host_seg,
                                               f"insert_{spec.name}", opts={"d": {}, "b": {}, "e": {}, "a": {}},
                                               verbose=True, ref_host=False, rings=rings,
                                               clean_h=spec.clean_h, **assemble.collar_opts(host_fixed))
    rec["timing"]["clean_s"] = time.perf_counter() - t
    rec["clean"] = crec
    rec["collar"], collar_ids = collar_record(mesh, reg, m, case, ring_out, out, h_host, rings, host_fixed)
    try:
        from meshom.meshfix.score import score as ms

        rec["ring_score"] = {"before": _small(ms(case, case.mesh)), "after": _small(ms(case, ring_out))}
    except Exception as ex:
        rec["ring_score"] = {"error": f"{type(ex).__name__}: {ex}"}
    chains_xy = [xy[cc] for cc in chains]
    sc = {}
    real_seg = rc.segments("raw") if rc is not None else None
    ins_nodes = {nm: measure.insert_coast_nodes(mm, m.mesh, m.host_node, chart)
                 for nm, mm in (("merged", m.mesh), ("final", out))}
    for nm, mm in (("host", mesh), ("merged", m.mesh), ("final", out)):
        z = measure.zones(mm, chart, outline, chains_xy, hj, h_host)
        sc[nm] = measure.metrics(mm, z, chart, seg, h_run, h_ins)
        if real_seg is not None:
            sc[nm]["coast_real"] = measure.coast_real(mm, z, chart, real_seg, seg, h_ins, h_run, outline,
                                                      np.vstack(chains_xy), insert_nodes=ins_nodes.get(nm))
    rec["scores"] = sc
    rec["validity_window"] = measure.validity(out)
    v = rec["validity_window"]
    why = []
    if not (v["inverted"] == 0 and v["nonconvex"] == 0 and v["nonmanifold_edges"] == 0
            and v["same_direction_edges"] == 0):
        why.append(f"the window has {v['inverted']} inverted / {v['nonconvex']} non-convex cells, "
                   f"{v['nonmanifold_edges']} edges with > 2 cells, {v['same_direction_edges']} shared edges "
                   f"traversed the same way by both cells (overlapping cells)")
    ends = np.unique([cc[e] for cc in chains for e in (0, -1)]) if spec.kind == "strip" else np.zeros(0, int)
    dj = cKDTree(out.points).query(mesh.points[ends])[0] if len(ends) else np.zeros(0)
    rec["join_ends_moved"] = int((dj >= assemble.MATCH_CHORD).sum())
    if rec["join_ends_moved"]:
        why.append(f"{rec['join_ends_moved']} end node(s) of the joins left their host position")
    hc = measure.host_coast(out, m.mesh, m.host_node, ins_nodes["final"], chart, seg, h_run, cut_xyz)
    rec["host_coast"] = hc
    if hc["h_run_max"] > assemble.HOST_SHIFT_H:
        why.append(f"the host's coast outside the insert moved {hc['h_run_max']:.2f} h (> {assemble.HOST_SHIFT_H} h)")
    rec["water_bodies"] = water_bodies(out)
    rec["water_bodies_host"] = water_bodies(mesh)
    if rec["water_bodies"] != rec["water_bodies_host"]:
        why.append(f"the window has {rec['water_bodies']} water bodies (the host's window {rec['water_bodies_host']})")
    rec["valid"] = not why
    if why:
        rec["invalid"] = why
        log(f"[insert {spec.name}] INVALID result: {'; '.join(why)} (no model files)", flush=True)
    fl = measure.floor_check(mesh, out, h_min)
    fm = measure.floor_check(mesh, m.mesh, h_min)
    fl.update(merged_smallest_m=fm["smallest_m"], merged_below=fm["below"])
    rec["floor"] = fl
    log(f"[insert {spec.name}] floor {h_min / 1e3:.2f} km: smallest cell {fl['smallest_m'] / 1e3:.3f} km, "
        f"{fl['below']} below", flush=True)
    rec["timing"]["total_s"] = time.perf_counter() - t0
    res = Result(False, rec, out, m.mesh, win, chart, None, [], [], outline, h_ins, case)
    res.real, res.ring_out, res.collar_moved = rc, ring_out, collar_ids
    r_z = 0.75 * depth
    zooms = [("spine end P", spine[0], r_z), ("spine end Q", spine[-1], r_z),
             ("host chain start", xy[chains[0][0]], r_z), ("host chain end", xy[chains[0][-1]], r_z),
             ("band (middle)", spine[len(spine) // 2], 12 * hf)]
    zooms += [(f"worst cell {k + 1}", p, 15 * hf) for k, p in enumerate(_worst(out, chart, outline, 2, 20 * hf))]
    res.band = {"chains_xy": chains_xy, "mid": spine[len(spine) // 2], "spine": spine, "h_fine": hf,
                "bank": bank_out, "zooms": zooms}
    if fl["below"]:
        rec["refused"] = [f"size floor: {fl['below']} cell(s) the insert made are below h_min_m {h_min:.0f} m "
                          f"(smallest {fl['smallest_m']:.0f} m at {fl['at']}); fix: h_min_m <= "
                          f"{np.floor(fl['smallest_m']):.0f} m or a coarser h_fine_m"]
        log(f"[insert {spec.name}] REFUSED {rec['refused'][0]}", flush=True)
        return res
    res.ok = True
    return res


def _quad_max_deg(xy, quads):
    """Largest corner (deg) of each quad in the chart (conformal: the sphere's angles)."""
    if not len(quads):
        return np.zeros(0)
    p = xy[quads]
    a = np.roll(p, -1, 1) - p
    b = np.roll(p, 1, 1) - p
    return np.degrees(np.arctan2(np.abs(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]), (a * b).sum(-1))).max(1)


def coast_bad(bb, K_ring: int = 2) -> np.ndarray:
    """B198: chart centroids of the band's quads next to the coast (all nodes within COAST_ROWS rows of the spine,
    or a node on a bank: a row end; not the host chain's ring rows) whose largest corner exceeds COAST_DEG."""
    b0 = bb["blocks"][0]
    K, rows_of, qd, xyn = bb["K"], bb["rows_of"], bb["qd"], bb["xyn"]
    if not len(qd):
        return np.zeros((0, 2))
    bank = np.zeros(len(xyn), bool)
    bank[np.r_[b0.off[1:K], b0.off[2:K + 1] - 1]] = True
    near = (rows_of[qd].max(1) <= COAST_ROWS) | (bank[qd].any(1) & (rows_of[qd].max(1) <= K - K_ring))
    bad = near & (_quad_max_deg(xyn, qd) > COAST_DEG)
    return xyn[qd[bad]].mean(1)


def _resmooth(bb, rc, spec, hf, build_band, coast_of, rec, log, K_ring: int = 2):
    """B198: the coast smoothed more round the bad cells next to it: each round adds RESMOOTH_STEP x the base width
    (bank_sigma_cells fine cells) to the smoothing width in a Gaussian window of RESMOOTH_W fine cells round each
    bad cell (one window per cluster), rebuilds the band and keeps it when it has fewer bad cells; a window whose
    smoothing would move the bank more than coast_shift_cells fine cells from where the base smoothing put it keeps
    its last width. At most coast_rounds rounds. The bank's move: the whole coast as the band takes it (coast_of:
    the end banks and the spine, sampled every SHIFT_STEP fine cells) against the base build's coast."""
    from meshom.insert.geom import seg_distance

    def prep(bumps):
        def mult(q):
            m = np.ones(len(q))
            for c, a in bumps:
                if a > 0:
                    m += a * np.exp(-0.5 * ((q - c) ** 2).sum(1) / (RESMOOTH_W * hf) ** 2)
            return m
        rc.prepare(lambda q: np.full(len(q), hf), lambda q: np.full(len(q), np.inf), hf, spec.bank_sigma_cells,
                   spec.bays, bridge="narrow", sigma_mult=mult if bumps else None)

    base = rc.segments("main")
    cap = spec.coast_shift_cells * hf
    ref = bb["coast"]
    ref_seg = np.stack([ref[:-1], ref[1:]], axis=1)

    def shift(coast, at=None):                             # how far the coast lies from the base build's
        q = resample(coast, SHIFT_STEP * hf)                # (near `at` only: within 3 windows)
        if at is not None:
            q = q[np.linalg.norm(q - at, axis=1) < 3 * RESMOOTH_W * hf]
        return float(seg_distance(q, ref_seg).max()) if len(q) else 0.0
    bumps, out = [], {"rounds": 0, "bad": [], "windows": 0, "capped": 0, "shift_km_max": 0.0}
    cur = coast_bad(bb, K_ring)
    out["bad"].append(int(len(cur)))
    for _ in range(spec.coast_rounds):
        if not len(cur):
            break
        trial = [[c.copy(), a] for c, a in bumps]
        raised = set()
        for c in cur:
            near = [i for i, (cc, _) in enumerate(trial) if np.linalg.norm(cc - c) < 2 * RESMOOTH_W * hf]
            if not near:
                trial.append([c.copy(), 0.0])
                near = [len(trial) - 1]
            raised.add(near[0])
        for i in raised:
            trial[i][1] += RESMOOTH_STEP
        for i in sorted(raised):                           # the cap: how far the extra smoothing moves the bank
            prep(trial)
            co = coast_of(rc)
            sh = shift(co, trial[i][0]) if co is not None else np.inf
            if sh > cap:
                trial[i][1] -= RESMOOTH_STEP
                out["capped"] += 1
        if all(abs(a - b) < 1e-12 for (_, a), (_, b) in zip(trial, bumps)) and len(trial) == len(bumps):
            break
        prep(trial)
        new = build_band(rc)
        out["rounds"] += 1
        if new is None:                                    # the rebuild refused: the band before it kept
            out["rejected"] = rec.pop("refused", ["?"])[0]
            break
        nb = coast_bad(new, K_ring)
        out["bad"].append(int(len(nb)))
        if len(nb) <= len(cur):
            bb, cur, bumps = new, nb, trial
        else:
            break
    prep(bumps)                                            # rc as the band kept was built
    if bumps:
        out["shift_km_max"] = round(shift(bb["coast"]) / 1e3, 3)
    out["left_xy_km"] = [[round(float(v) / 1e3, 2) for v in c] for c in cur]
    out["windows_xy_km"] = [[round(float(v) / 1e3, 2) for v in c] for c, a in bumps if a > 0]
    rc.base_main = base                                    # the bank before the extra smoothing (maps)
    out["windows"] = int(sum(1 for _, a in bumps if a > 0))
    out["widths_cells"] = [round(spec.bank_sigma_cells * (1 + a), 2) for _, a in bumps if a > 0]
    rec["coast_smooth"] = out
    log(f"[insert {spec.name}] coast smoothing: bad cells next to the coast {out['bad']}, {out['windows']} "
        f"window(s), bank moved <= {out['shift_km_max']:.2f} km", flush=True)
    return bb


def _worst(mesh, chart, outline, k: int, sep: float) -> list:
    """Chart positions of the k worst quads (largest corner) inside the outline, at least sep apart (map zooms)."""
    from meshom.insert.geom import point_in_polygon
    from meshom.quality.kernel import element_geometry

    if not mesh.n_quads:
        return []
    g = element_geometry(mesh)
    c = chart.fwd(g.centroid[mesh.n_tris:])
    qa = np.degrees(g.quad_angles).max(1)
    qa[~point_in_polygon(c, outline)] = -1
    out = []
    for i in np.argsort(-qa):
        if qa[i] < 0 or len(out) >= k:
            break
        if all(np.linalg.norm(c[i] - p) >= sep for p in out):
            out.append(c[i])
    return out


def water_bodies(mesh) -> int:
    """Edge-connected pieces of the mesh (one water body expected)."""
    from meshom.meshfix.cases import components

    lab = components(mesh)
    return int(len(np.unique(lab)))
