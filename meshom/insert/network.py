"""A block network: the strait's main block plus one block per bay on its banks (decision B166).

Submapping: walking the region's boundary with the water on the left, a bay adds the corner
classes -1 (the mouth ends: the coast turns away from the water), +1, +1 (the bay's head), -1: the classes still
sum to 4, so the parameter domain is a rectilinear polygon (the strait's rectangle with a bump per bay) and no
interior irregular node is needed. It is built as blocks with shared interval counts:

- the main block's bank crosses the bay's mouth on the chord (meshom.insert.realcoast bridges it), so the block
  never wraps the bay (wrapping folded the Denmark Strait in step 1, B163);
- the main block's bank nodes nearest to the two mouth ends are moved onto them exactly; the bank nodes between
  them (n edges: the main block's row count along the mouth, fixed) are the bay block's first row (Dirichlet);
- the bay block (meshom.insert.block.build, rows by its size target) runs from the mouth to the bay's head; its walls
  (the bay's coast from each mouth end to a head corner) slide; its head is the bay's coast between the two head
  corners at equal arc steps, round(head length / mouth cell) of them (at most n): a bay that narrows loses columns
  by Y-splits, its cells never below BAY_H_FLOOR of the mouth's. Head corners: where the bay's turning reaches HEAD_TURN and 1 - HEAD_TURN
  of its total (the +1 classes);
- a mouth end node gets 2 quads of the main block and 1 of the bay block: a -1 boundary node (a cape of the
  region), every other node keeps valence 4.

The bay blocks are cleaned with the join ring (their cells are not core: the mesh-fixing pipeline may move their
nodes; a rounded bay puts its head corners on a smooth arc, ~135 deg cells before cleaning).

A bay is skipped (reported, its water left out) when its mouth holds fewer than 2 main-block rows or touches the
first or last RING_GAP rows (the join ring).
"""
from dataclasses import dataclass, field

import numpy as np

from meshom.insert import block as blockmod
from meshom.insert.geom import Polyline

HEAD_TURN = 0.25             # B166: head corners where the bay's turning reaches 1/4 and 3/4 of its total
RING_GAP = 1                 # B166: a bay's mouth this near a join row starts on the join's end node (B185)
BAY_H_FLOOR = 0.7            # B166: a bay block's cells stay >= this x the mouth's cell (a narrowing bay drops columns
                             # by Y-splits instead of thinning them; the head gets round(head length / mouth cell))
CLEAN_DEG = 120.0            # B166: block cells above this corner angle (chart) are released to the ring cleaning ...
CLEAN_RINGS = 1              # B166: ... with this many node rings around them
MATCH_H = 0.5                # B166: a bay belongs to a bank when both mouth ends lie within this many cells of it
                             # (the bank blends onto the join ends; the mouth nodes are moved onto the ends)


@dataclass
class Net:
    xy: np.ndarray               # (n,2) chart
    quads: np.ndarray            # (Q,4) local ids, CCW in the chart
    tris: np.ndarray             # (T,3)
    ids_a: np.ndarray            # local ids of join A's nodes (chain order)
    ids_b: np.ndarray
    core_q: np.ndarray           # (Q,) bool: cells the ring cleaning must not touch
    core_t: np.ndarray
    main: object                 # the main Block
    bays: list = field(default_factory=list)       # per bay: dict record
    samples: tuple = (np.zeros((0, 2)), np.zeros(0))   # (pts chart, h m chart) of the size targets (h layer)
    bank_s: np.ndarray | None = None               # the banks with the meshed bays' coast in place of their mouths
    bank_n: np.ndarray | None = None
    fine_inverted: int = 0       # fine-lattice cells with a non-positive Jacobian over all blocks (fold predictor)


def ccw(xy, cells):
    if not len(cells):
        return cells
    p = xy[cells]
    a = 0.5 * np.sum(p[:, :, 0] * np.roll(p[:, :, 1], -1, 1) - p[:, :, 1] * np.roll(p[:, :, 0], -1, 1), axis=1)
    out = cells.copy()
    out[a < 0] = out[a < 0][:, ::-1]
    return out


def fine_inverted(X) -> int:
    """Cells of a fine Winslow lattice (M+1, N+1, 2) whose corners turn the wrong way (any corner's cross product
    against the majority sign, or zero)."""
    a = X[:-1, :-1]
    b = X[1:, :-1]
    c = X[1:, 1:]
    d = X[:-1, 1:]

    def cr(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
    cs = np.stack([cr(d, a, b), cr(a, b, c), cr(b, c, d), cr(c, d, a)])
    sgn = np.sign(np.median(cs))
    return int(np.any(cs * sgn <= 0, axis=0).sum())


def _samples(blk, chart_scale):
    st = max(1, blk.fine_xy.shape[1] // 24)
    sr = max(1, blk.fine_xy.shape[0] // max(4 * blk.M, 1))
    fx = blk.fine_xy[::sr, ::st].reshape(-1, 2)
    fh = np.repeat(blk.fine_h[::sr], blk.fine_xy[::sr, ::st].shape[1])
    return fx, fh / chart_scale(fx)


def _head_corners(arc: np.ndarray) -> tuple[int, int]:
    d = np.diff(arc, axis=0)
    th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    turn = th - th[0]
    T = turn[-1]
    if abs(T) < 1e-6:
        n = len(arc) - 1
        return max(1, n // 3), min(n - 1, 2 * n // 3)
    f = turn / T
    c1 = int(np.searchsorted(np.maximum.accumulate(f), HEAD_TURN)) + 1
    c2 = int(np.searchsorted(np.maximum.accumulate(f), 1.0 - HEAD_TURN)) + 1
    n = len(arc) - 1
    c1 = min(max(c1, 1), n - 2)
    c2 = min(max(c2, c1 + 1), n - 1)
    return c1, c2


def _anchor(arc: np.ndarray, a=None, b=None) -> np.ndarray:
    """B185: a bay's coast cut at the point nearest to a join's host end node (a, at its start; b, at its end) and
    blended onto it over one tenth of the arc (smoothstep), so the bay block's mouth starts on that fixed node."""
    from meshom.insert.realcoast import blend_ends

    if a is not None:
        arc = arc[int(np.argmin(np.linalg.norm(arc - a, axis=1))):]
    if b is not None:
        arc = arc[:int(np.argmin(np.linalg.norm(arc - b, axis=1))) + 1]
    L = float(np.linalg.norm(np.diff(arc, axis=0), axis=1).sum())
    return blend_ends(arc, arc[0] if a is None else a, arc[-1] if b is None else b,
                      0.1 * L if a is not None else 1e-9, 0.1 * L if b is not None else 1e-9)


def _bank_nodes(blk, side: str) -> np.ndarray:
    """Local ids of the main block's bank nodes, row 0 .. M (S: k = 0, N: k = N_i)."""
    if side == "S":
        return np.array([blk.off[i] for i in range(blk.M + 1)])
    return np.array([blk.off[i] + blk.N[i] for i in range(blk.M + 1)])


def build(blk, bank_s: Polyline, bank_n: Polyline, bays: list, chart_scale, h_min: float = 0.0, ring_rows: int = 2,
          log=print, clean_bays: bool = True, fill_rows: int | None = None) -> Net:
    """The network of the main block `blk` (meshom.insert.block.Block, banks bank_s / bank_n: the main coast with
    the bays bridged) and its bays (meshom.insert.realcoast.Bay of the window; those whose two mouth ends lie on a
    bank are used)."""
    xy = blk.xy.copy()
    M = blk.M
    quads = [ccw(xy, blk.quads)]
    tris = [ccw(xy, blk.tris)]
    rq = np.minimum(blk.row[blk.quads].min(1), M)
    rt = blk.row[blk.tris].min(1) if len(blk.tris) else np.zeros(0, int)
    core_q = [(rq >= ring_rows) & (rq < M - ring_rows)]
    core_t = [(rt >= ring_rows) & (rt < M - ring_rows)]
    samples = [_samples(blk, chart_scale)]
    fine_bad = fine_inverted(blk.fine_xy)
    recs = []
    banks_out = {"S": bank_s.xy.copy(), "N": bank_n.xy.copy()}
    used_rows = {"S": np.zeros(M + 1, bool), "N": np.zeros(M + 1, bool)}
    nxt = len(xy)
    for bay in bays:
        best = None
        for side, pl in (("S", bank_s), ("N", bank_n)):
            if pl.L <= 0:
                continue
            da, db = pl.distance(bay.a[None])[0], pl.distance(bay.b[None])[0]
            tol = MATCH_H * bay.h_m
            if da <= tol and db <= tol:
                best = (side, pl)
        rec = bay.record()
        if best is None:
            continue                                      # not on this insert's banks
        side, pl = best
        rec["side"] = side
        if not bay.meshed:
            rec["status"] = "bridged (too small for a block of its own, B190; its water is left out)"
            recs.append(rec)
            continue
        ids = _bank_nodes(blk, side)
        sb = pl.project(xy[ids])
        sa_, sb_ = float(pl.project(bay.a[None])[0]), float(pl.project(bay.b[None])[0])
        arc = bay.arc
        if sa_ > sb_:                                     # orient the mouth along the bank (row order)
            sa_, sb_ = sb_, sa_
            arc = arc[::-1]
        ia, ib = int(np.argmin(np.abs(sb - sa_))), int(np.argmin(np.abs(sb - sb_)))
        at = []
        if ia < RING_GAP and ib - ia >= 2:                # B185: a bay at a join: its mouth starts on the join's
            ia, at = 0, at + ["A"]                        # host end node, the bay's coast blended onto it
        if ib > M - RING_GAP and ib - ia >= 2:
            ib, at = M, at + ["B"]
        if ib - ia < 2 or used_rows[side][ia:ib + 1].any():
            rec["status"] = (f"skipped (mouth rows {ia}..{ib} of {M}: fewer than 2 rows or overlapping another "
                             "bay; its water is left out)")
            recs.append(rec)
            continue
        if at:
            arc = _anchor(arc, xy[ids[0]] if "A" in at else None, xy[ids[M]] if "B" in at else None)
            rec["at_join"] = at
        used_rows[side][ia:ib + 1] = True
        xy[ids[ia]], xy[ids[ib]] = arc[0], arc[-1]
        mouth = xy[ids[ia:ib + 1]]
        n = ib - ia
        c1, c2 = _head_corners(arc)
        wall_a, head, wall_b = arc[:c1 + 1], arc[c1:c2 + 1], arc[c2:][::-1]
        hp = Polyline(head)
        h_mouth = float(np.mean(np.linalg.norm(np.diff(mouth, axis=0), axis=1)))
        n_head = int(min(n, max(2, round(hp.L / h_mouth))))      # B166: a narrowing bay loses columns (Y-splits)
        rows = None
        if bay.fill >= 0:                                 # B204: the strip behind a fill line: its rows by its depth,
            rows = fill_rows if fill_rows is not None else max(1, int(round(bay.depth_m / h_mouth)))
            while n_head < n and blockmod.birth_rows(n, n_head) > rows:      # its column changes within them
                n_head += 1
        head_xy = hp.at(np.linspace(0, hp.L, n_head + 1))
        try:
            bb = blockmod.build(mouth, head_xy, Polyline(wall_a), Polyline(wall_b), n, fixed_count=n_head == n,
                                h_min=max(h_min, BAY_H_FLOOR * h_mouth), cell_floor=h_min, scale=chart_scale,
                                rows=rows)
        except Exception as ex:                           # a degenerate bay: report and leave it out
            rec["status"] = f"failed ({type(ex).__name__}: {ex}); its water is left out"
            recs.append(rec)
            xy[ids[ia]], xy[ids[ib]] = blk.xy[ids[ia]], blk.xy[ids[ib]]
            used_rows[side][ia:ib + 1] = False
            continue
        loc = np.full(len(bb.xy), -1, np.int64)
        loc[bb.off[0]:bb.off[0] + n + 1] = ids[ia:ib + 1]
        fresh = loc < 0
        loc[fresh] = nxt + np.arange(fresh.sum())
        nxt += int(fresh.sum())
        xy = np.vstack([xy, bb.xy[fresh]])
        bq, bt = loc[bb.quads], loc[bb.tris]
        quads.append(ccw(xy, bq))
        tris.append(ccw(xy, bt))
        core_q.append(np.full(len(bq), not clean_bays))       # B166: bays are cleaned with the join ring
        core_t.append(np.full(len(bt), not clean_bays))
        samples.append(_samples(bb, chart_scale))
        fb = fine_inverted(bb.fine_xy)
        fine_bad += fb
        rec.update(status="meshed", rows=int(bb.M), cells=int(n), mouth_rows=[ia, ib], fine_inverted=fb,
                   quads=int(len(bq)))
        recs.append(rec)
        # the bank with the bay's coast in place of its mouth (for the outline and the coast measure)
        bo = Polyline(banks_out[side])
        s0, s1 = float(bo.project(arc[0][None])[0]), float(bo.project(arc[-1][None])[0])
        banks_out[side] = np.vstack([bo.xy[bo.s < s0], arc, bo.xy[bo.s > s1]])
        log(f"[insert] bay on bank {side}: mouth {bay.mouth_m / 1e3:.1f} km ({n} rows), depth {bay.depth_m / 1e3:.1f} "
            f"km, {bb.M} rows deep, {len(bq)} quads", flush=True)
    Q = np.vstack(quads)
    T = np.vstack(tris) if sum(len(t) for t in tris) else np.zeros((0, 3), np.int64)
    sp = np.vstack([s[0] for s in samples])
    sh = np.concatenate([s[1] for s in samples])
    ia, ib = blk.joins()                                  # B208: the join chains' host nodes
    net = Net(xy, Q.astype(np.int64), T.astype(np.int64), ia, ib, np.concatenate(core_q), np.concatenate(core_t), blk,
              recs, (sp, sh), banks_out["S"], banks_out["N"], fine_bad)
    return net


def release_bad(net: Net, deg: float | None = None, rings: int | None = None) -> int:
    """B166: block cells with a corner above `deg` (CLEAN_DEG; chart angles) and every cell within `rings` node rings
    of them stop being core, so the ring cleaning repairs them (a cell wrapping a tight cape of the real coast:
    Ceuta, the Westfjords). Returns the number of released cells."""
    deg = CLEAN_DEG if deg is None else deg
    rings = CLEAN_RINGS if rings is None else rings
    xy = net.xy
    Q = len(net.quads)
    if not Q:
        return 0
    p = xy[net.quads]
    a = np.roll(p, -1, 1) - p
    b = np.roll(p, 1, 1) - p
    ang = np.degrees(np.mod(np.arctan2(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0], (a * b).sum(-1)), 2 * np.pi))
    bad = ang.max(1) > deg
    if not bad.any():
        return 0
    near = np.zeros(len(xy), bool)
    near[net.quads[bad].ravel()] = True
    for _ in range(rings):
        hit_q = near[net.quads].any(1)
        near[net.quads[hit_q].ravel()] = True
        if len(net.tris):
            near[net.tris[near[net.tris].any(1)].ravel()] = True
    rel_q = near[net.quads].all(1) | bad
    rel_t = near[net.tris].all(1) if len(net.tris) else np.zeros(0, bool)
    n = int((net.core_q & rel_q).sum() + (net.core_t & rel_t).sum())
    net.core_q = net.core_q & ~rel_q
    net.core_t = net.core_t & ~rel_t
    return n


def from_block(blk, ring_rows: int, chart_scale) -> Net:
    """A single block as a network (no bays): core = rows ring_rows .. M - ring_rows."""
    M = blk.M
    rq = np.minimum(blk.row[blk.quads].min(1), M)
    rt = blk.row[blk.tris].min(1) if len(blk.tris) else np.zeros(0, int)
    ia, ib = blk.joins()                                  # B208: the join chains' host nodes
    return Net(blk.xy, blk.quads, blk.tris, ia, ib, (rq >= ring_rows) & (rq < M - ring_rows),
               (rt >= ring_rows) & (rt < M - ring_rows), blk, [], _samples(blk, chart_scale),
               fine_inverted=fine_inverted(blk.fine_xy))
