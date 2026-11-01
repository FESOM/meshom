"""The insert's section-area depth rule (decision B250), on the bathymetry grid's raw cells.

A climate model needs each strait's cross-section about the real one's. FESOM3 takes an element's bottom at a model
level (meshom.io.fesom.model_levels), so the area across a strait follows the insert's node depths only in steps, and
the box mean per node (B211) on cells about as wide as the strait gives sections that are too large or too small.
With `depth_area = true` the insert's own nodes are scaled after the grid rule, the channel raise and the sill
(B211-B215), so that the area of each section across the insert in FESOM3's element bottoms matches the grid's wet
area across the same line:

- sections: great circles across the strait's axis (the insert's centreline between its two joins), every
  AREA_STEP_H fine cells along it, normal to the axis's chord over one fine cell on each side; sampled every half
  grid cell;
- the mesh's run: the samples in the mesh's cells, contiguous through the axis point; its area: the sum over the
  samples of their element's bottom times the spacing;
- the grid's crossing: the grid's wet samples (the raw cell under the sample, elevation < 0; land counts 0) within
  the mesh's run widened by AREA_BANK_H local cells on each side (the real banks lie within a cell of the mesh's);
  its area: the sum of their depths times the spacing;
- a section counts when its axis point lies in the mesh's water, at least AREA_SHARE of its run lies in the
  insert's cells, the run ends on the mesh's coast within the samples, the grid has water there (at least the depth
  floor over one sample), the grid's water across it ends within AREA_OPEN_H local cells beyond the mesh's banks
  (else the line runs out into open water or into a bay the banks cut off), and it lies on the strait's narrows (the
  run of sections round the one with the narrowest wet crossing, up to the first wider than CHANNEL_WIDTH_K times it,
  as the channel raise's narrows, B213: the strait, not the open basins at its ends);
- the factor: a log factor per counted section, linear across the sections that do not count, tapered to 0 over
  AREA_TAPER_H fine cells beyond the first and the last, smoothed along the axis by a Gaussian of AREA_SMOOTH_H fine
  cells; an insert node takes the factor at its nearest axis point and its depth below the depth floor is scaled by
  it (floor + (depth - floor) x factor: the shape across the section kept, no node above the floor), the factor
  within AREA_FACTOR; the dam (B212, B215) holds: its nodes never go deeper than its cap and the nodes it capped keep
  the cap; and the nodes of the cells the channel line (B213) crosses are never made shallower, so the deepest route
  between the basins keeps its levels (a cell whose nodes were lowered towards the dam's cap still lost a level where
  its other nodes were shallower): the sections shrink on their flanks, not on the strait's deepest path;
- rounds: each counted section's log factor moves by AREA_GAIN x log(grid area / mesh area), at most AREA_ITERS
  rounds or until every counted section lies within AREA_TOL; the round with the least mean |log(mesh / grid)| over
  the counted sections is kept (round 0 is the input depth).
"""
import numpy as np

from meshom.geo import R_EARTH_M, arc_length, xyz_to_lonlat
from meshom.mesh import Mesh

AREA_STEP_H = 0.5            # B250: sections every this many fine cells along the axis
AREA_BANK_H = 1.0            # B250: the grid's crossing: the mesh's run widened by this many local cells on each side
AREA_OPEN_H = 3.0            # B250: a section whose grid water runs on this many local cells past a mesh bank: skipped
AREA_SHARE = 0.9             # B250: a section counts when this share of its mesh run lies in the insert's cells
AREA_TAPER_H = 2.0           # B250: the log factor tapers to 0 over this many fine cells past the outer sections
AREA_SMOOTH_H = 1.0          # B250: the log factor is smoothed along the axis by a Gaussian of this many fine cells
AREA_FACTOR = (0.25, 4.0)    # B250: the factor's range
AREA_GAIN = 0.8              # B250: a round moves a section's log factor by this x log(grid / mesh)
AREA_ITERS = 20              # B250: at most this many rounds
AREA_TOL = 0.05              # B250: the rounds stop when every counted section is within this of the grid's area


def stations(axis: np.ndarray, h_fine: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The axis (K,3) resampled every AREA_STEP_H x h_fine on the sphere: (arc length m, points, unit normals: the
    point x the chord over h_fine on each side)."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    seg = arc_length(a[:-1], a[1:]) * R_EARTH_M
    keep = np.r_[True, seg > 1e-6]
    a = a[keep]
    s = np.r_[0.0, np.cumsum(arc_length(a[:-1], a[1:]) * R_EARTH_M)]
    step = AREA_STEP_H * h_fine
    st = np.linspace(0.0, s[-1], max(int(np.ceil(s[-1] / step)), 1) + 1)

    def at(t):
        t = np.clip(t, 0.0, s[-1])
        p = np.stack([np.interp(t, s, a[:, k]) for k in range(3)], -1)
        return p / np.linalg.norm(p, axis=1, keepdims=True)

    p = at(st)
    t = at(st + h_fine) - at(st - h_fine)
    t -= np.sum(t * p, axis=1)[:, None] * p
    u = np.cross(p, t)
    u /= np.maximum(np.linalg.norm(u, axis=1), 1e-15)[:, None]
    return st, p, u


def _elem_size(sub: Mesh) -> np.ndarray:
    """(E,) the mean great-circle edge of each element (m), triangles first."""
    out = []
    for c in (sub.tris, sub.quads):
        if len(c):
            k = c.shape[1]
            L = [arc_length(sub.points[c[:, i]], sub.points[c[:, (i + 1) % k]]) for i in range(k)]
            out.append(np.mean(L, axis=0) * R_EARTH_M)
    return np.concatenate(out) if out else np.zeros(0)


def sections(sub: Mesh, ins: np.ndarray, grid, st, p, u, reach_m: float, floor_m: float):
    """Each station's section (module docstring): a list of dicts with `ok` (bool), `why` (the reason it does not
    count), and for a counted one `elem` / `w` (the elements of the mesh's run and their sample lengths, m),
    `gebco_m2`, `width_m` (the mesh's run), `gebco_width_m` (the grid's wet length in the crossing)."""
    from meshom.straits import _elem_tree, locate

    tree = _elem_tree(sub)
    T = sub.n_tris
    elem_ins = np.r_[ins[sub.tris].any(1) if T else np.zeros(0, bool), ins[sub.quads].any(1)]
    size = _elem_size(sub)
    ds = 0.5 * R_EARTH_M * np.radians(abs(grid.dlat))
    tau = np.arange(-np.ceil(reach_m / ds), np.ceil(reach_m / ds) + 1) * ds
    i0 = len(tau) // 2
    la0, la1 = float(grid.lat.min()), float(grid.lat.max())
    lo0, lo1 = float(grid.lon.min()), float(grid.lon.max())
    out = []
    for k in range(len(st)):
        q = (p[k][None, :] * np.cos(tau / R_EARTH_M)[:, None] + u[k][None, :] * np.sin(tau / R_EARTH_M)[:, None])
        e = locate(sub, q, tree)
        if e[i0] < 0:
            out.append({"ok": False, "why": "the axis point lies outside the mesh's water"})
            continue
        a = i0
        while a > 0 and e[a - 1] >= 0:
            a -= 1
        b = i0
        while b < len(e) - 1 and e[b + 1] >= 0:
            b += 1
        run = e[a:b + 1]
        rec = {"ok": False, "width_m": float((b - a + 1) * ds)}
        if a == 0 or b == len(e) - 1:
            out.append(rec | {"why": "the mesh's run reaches the samples' end"})
            continue
        if elem_ins[run].mean() < AREA_SHARE:
            out.append(rec | {"why": "the mesh's run lies in the host's cells"})
            continue
        h = float(np.median(size[run]))
        m, M = int(np.ceil(AREA_BANK_H * h / ds)), int(np.ceil(AREA_OPEN_H * h / ds))
        ga, gb = a - M, b + M
        if ga < 0 or gb > len(e) - 1:
            out.append(rec | {"why": "the grid's crossing reaches the samples' end"})
            continue
        lon, lat = xyz_to_lonlat(q[ga:gb + 1])
        ci, cj, _, _ = grid.boxes(lon, lat, np.zeros(len(lon)))
        ul = grid.unwrap(lon)
        outside = (lat < la0) | (lat > la1) | (ul < lo0) | (ul > lo1)
        z = grid.elev[ci, cj].astype(np.float64)
        wet = outside | (z < 0)
        dep = np.where(wet & ~outside, -z, 0.0)
        lo, hi = M - m, len(wet) - 1 - (M - m)               # the crossing: the mesh's run +- m samples
        lab = np.cumsum(np.r_[True, wet[1:] != wet[:-1]])    # runs of equal wetness
        cross = np.unique(lab[lo:hi + 1][wet[lo:hi + 1]])
        if np.isin(cross, [lab[0], lab[-1]]).any():
            out.append(rec | {"why": "the grid's water runs on past a mesh bank"})
            continue
        if outside[lo:hi + 1].any():
            out.append(rec | {"why": "the crossing leaves the grid's window"})
            continue
        A = float(dep[lo:hi + 1].sum() * ds)
        if A < floor_m * ds:
            out.append(rec | {"why": "the grid is dry across the mesh's run"})
            continue
        el, cnt = np.unique(run, return_counts=True)
        out.append(rec | {"ok": True, "why": "", "elem": el, "w": cnt * ds, "gebco_m2": A,
                          "gebco_width_m": float(wet[lo:hi + 1].sum() * ds), "h_m": h})
    return out


def narrows(secs: list, st: np.ndarray, p: np.ndarray) -> dict:
    """The strait's narrows among the counted sections: the run of stations round the one whose grid crossing is
    narrowest, up to the first counted section on each side wider than CHANNEL_WIDTH_K times it (the channel raise's
    rule, B213, on the sections' own wet crossings, so that islands do not cut a width short). Returns the record and,
    under `outside`, the counted stations beyond it."""
    from meshom.insert.channel import CHANNEL_WIDTH_K

    iv = [k for k, s in enumerate(secs) if s["ok"]]
    if not iv:
        return {"outside": []}
    w = {k: secs[k]["gebco_width_m"] for k in iv}
    k0 = min(iv, key=lambda k: w[k])
    lim = CHANNEL_WIDTH_K * w[k0]
    a = k0
    while a > 0 and not (secs[a - 1]["ok"] and w[a - 1] > lim):
        a -= 1
    b = k0
    while b < len(secs) - 1 and not (secs[b + 1]["ok"] and w[b + 1] > lim):
        b += 1
    lon, lat = xyz_to_lonlat(p[[k0, a, b]])
    pt = lambda i: [round(float(lon[i]), 4), round(float(lat[i]), 4)]          # noqa: E731
    return {"width_min_km": w[k0] / 1e3, "at": pt(0), "limit_km": lim / 1e3, "from": pt(1), "to": pt(2),
            "km": float(st[b] - st[a]) / 1e3, "outside": [k for k in iv if k < a or k > b]}


def area_depth(sub: Mesh, depth: np.ndarray, ins: np.ndarray, axis: np.ndarray, zbar: np.ndarray, floor_m: float,
               source, variable: str, h_fine: float, held: tuple | None = None, route: np.ndarray | None = None,
               tag: str = "", log=print) -> tuple[np.ndarray, dict]:
    """B250 (module docstring) on the nodes `ins` of `sub` (the window's part of the spliced mesh, every cell round
    them in it), their depths `depth` after the other own rules; axis (K,3): the insert's centreline from one join
    to the other; held: (node ids, cap m) of the dam (B212, B215) or None; route: (N,) bool, the nodes of the cells
    the channel line crosses (never made shallower) or None.
    Returns (depth, record)."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial import cKDTree

    from meshom.insert import channel as chan
    from meshom.io.fesom import model_levels

    d0 = np.asarray(depth, dtype=np.float64)
    st, p, u = stations(axis, h_fine)
    size = _elem_size(sub)
    T = sub.n_tris
    cells_ins = np.r_[ins[sub.tris].any(1) if T else np.zeros(0, bool), ins[sub.quads].any(1)]
    r_max = float(size[cells_ins].max())
    nodes_ins = np.unique(np.r_[sub.tris[cells_ins[:T]].ravel(), sub.quads[cells_ins[T:]].ravel()])
    dense = np.linspace(0.0, st[-1], max(int(np.ceil(st[-1] / (0.125 * h_fine))), 1) + 1)
    pd = np.stack([np.interp(dense, st, p[:, k]) for k in range(3)], -1)
    pd /= np.linalg.norm(pd, axis=1, keepdims=True)
    dist, _ = cKDTree(pd).query(sub.points[nodes_ins])
    reach = float(2.0 * np.arcsin(min(dist.max() / 2.0, 1.0)) * R_EARTH_M) + (AREA_OPEN_H + 2.0) * r_max
    lo_, la_ = xyz_to_lonlat(sub.points[nodes_ins])
    grid = chan.read_window(source, variable, lo_, la_, (AREA_OPEN_H + 2.0) * r_max)
    secs = sections(sub, ins, grid, st, p, u, reach, floor_m)
    nrec = narrows(secs, st, p)
    for k in nrec.pop("outside"):
        secs[k] = {"ok": False, "why": "outside the strait's narrows", "width_m": secs[k]["width_m"]}
    ok = np.array([s["ok"] for s in secs])
    rec = {"sections": int(len(secs)), "counted": int(ok.sum()), "step_km": AREA_STEP_H * h_fine / 1e3,
           "narrows": nrec,
           "skipped": {w: int(sum(1 for s in secs if s["why"] == w)) for w in sorted({s["why"] for s in secs} - {""})}}
    if not ok.any():
        rec["applied"] = False
        log(f"{tag} depth_area: no section counts ({rec['skipped']})", flush=True)
        return d0, rec
    iv = np.flatnonzero(ok)
    Ag = np.array([secs[k]["gebco_m2"] for k in iv])
    _, jn = cKDTree(pd).query(sub.points)
    s_node = dense[jn]
    own = ins & (d0 > floor_m)
    dam = np.zeros(sub.n_nodes, bool)
    cap = None
    if held is not None and len(held[0]):
        dam[np.asarray(held[0], np.int64)] = True
        cap = float(held[1])
    kept = dam & (d0 >= cap - 1e-6) if cap is not None else dam
    own &= ~kept
    lb = np.full(sub.n_nodes, floor_m)
    path = np.zeros(sub.n_nodes, bool) if route is None else (np.asarray(route, bool) & ins)
    lb[path] = np.maximum(d0[path], floor_m)
    sig = AREA_SMOOTH_H * h_fine / max(st[1] - st[0], 1e-9) if len(st) > 1 else 0.0
    taper = AREA_TAPER_H * h_fine
    lg = np.log(AREA_FACTOR)

    def field(g):
        G = np.interp(st, st[iv], g)
        G = np.where(st < st[iv[0]], G * np.clip(1.0 - (st[iv[0]] - st) / taper, 0.0, 1.0), G)
        G = np.where(st > st[iv[-1]], G * np.clip(1.0 - (st - st[iv[-1]]) / taper, 0.0, 1.0), G)
        return gaussian_filter1d(G, sig, mode="nearest") if sig > 0 else G

    def apply(G):
        f = np.exp(np.interp(s_node, st, G))
        d = d0.copy()
        d[own] = np.maximum(floor_m + (d0[own] - floor_m) * f[own], lb[own])
        if cap is not None:
            d[own & dam] = np.minimum(d[own & dam], cap)
        return d, f

    def areas(d):
        bot = model_levels(sub, d, zbar)[2]
        return np.array([float(secs[k]["w"] @ bot[secs[k]["elem"]]) for k in iv])

    g = np.zeros(len(iv))
    best, hist = None, []
    for it in range(AREA_ITERS + 1):
        G = field(g)
        d, f = apply(G)
        Am = areas(d)
        r = np.log(Ag / Am)
        err = np.abs(Am / Ag - 1.0)
        hist.append({"round": it, "mean": float(err.mean()), "max": float(err.max()),
                     "within": int((err <= AREA_TOL).sum())})
        if best is None or np.abs(r).mean() < best[0]:
            best = (float(np.abs(r).mean()), it, d, G, Am)
        if it == 0:
            A0 = Am
        if err.max() <= AREA_TOL:
            break
        g = np.clip(g + AREA_GAIN * r, lg[0], lg[1])
    _, it, d, G, Am = best
    e0, e1 = np.abs(A0 / Ag - 1.0), np.abs(Am / Ag - 1.0)
    fac = np.exp(G)
    ch = d - d0
    curve = []
    pos = {int(k): n for n, k in enumerate(iv)}
    for k, s in enumerate(secs):
        lon, lat = xyz_to_lonlat(p[k][None])
        c = {"s_km": round(float(st[k]) / 1e3, 3), "lonlat": [round(float(lon[0]), 4), round(float(lat[0]), 4)],
             "factor": round(float(fac[k]), 4), "ok": bool(s["ok"])}
        if "width_m" in s:
            c["width_km"] = round(s["width_m"] / 1e3, 2)
        if s["ok"]:
            n = pos[k]
            c |= {"gebco_km2": round(float(Ag[n]) / 1e6, 4), "before_km2": round(float(A0[n]) / 1e6, 4),
                  "after_km2": round(float(Am[n]) / 1e6, 4), "gebco_width_km": round(s["gebco_width_m"] / 1e3, 2)}
        else:
            c["why"] = s["why"]
        curve.append(c)
    rec |= {"applied": True, "rounds": hist, "kept_round": it, "tol": AREA_TOL,
            "before": {"mean": float(e0.mean()), "max": float(e0.max()), "within": int((e0 <= AREA_TOL).sum())},
            "after": {"mean": float(e1.mean()), "max": float(e1.max()), "within": int((e1 <= AREA_TOL).sum())},
            "factor": {"min": float(fac.min()), "max": float(fac.max()),
                       "max_step": float(np.abs(np.diff(np.log(fac))).max()) if len(fac) > 1 else 0.0},
            "nodes_changed": int((np.abs(ch) > 1e-6).sum()), "change_m": {"min": float(ch.min()),
                                                                           "max": float(ch.max())},
            "dam": {"nodes": int(dam.sum()), "kept_at_cap": int(kept.sum()), "cap_m": cap},
            "route_nodes": int(path.sum()), "curve": curve}
    log(f"{tag} depth_area: {len(iv)} of {len(secs)} sections counted (every {AREA_STEP_H * h_fine / 1e3:.2f} km); "
        f"mesh / grid area off by {e0.mean():.1%} on average (max {e0.max():.1%}) -> {e1.mean():.1%} (max "
        f"{e1.max():.1%}), within {AREA_TOL:.0%}: {int((e0 <= AREA_TOL).sum())} -> {int((e1 <= AREA_TOL).sum())}; "
        f"round {it} of {len(hist) - 1}; factor {fac.min():.2f}-{fac.max():.2f}; {rec['nodes_changed']} nodes "
        f"changed by {ch.min():.0f} to {ch.max():.0f} m; dam {int(dam.sum())} nodes, {int(kept.sum())} kept at its cap",
        flush=True)
    return d, rec
