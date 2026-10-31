"""Where an insert goes: the strait's width profile, the three rules, the candidate joins and the growth of the
region (decisions B158-B160).

The rules (measured on synthetic and real straits), checked for every candidate join:

- **corner**: a join meets each bank at an angle within CORNER_DEG (70-110 deg); measured as chords over
  CORNER_CHORD_H join cells along the chain and along the coast toward the neck. The corner quads' max angle grows
  by at least the deviation (59 deg gave 135 deg quads, 90 deg gave 111 deg).
- **funnel**: with N(s) cells across a section, the row-to-row size ratio of a square-cell block is exp(alpha / N),
  alpha the banks' opening angle there (alpha = 2 atan(|dW/ds| / 2) from the smoothed width profile W(s)); every
  section between the neck and the join must have alpha <= N(s) ln FUNNEL_RATIO, N(s) = W(s) / min(h_t(s),
  h_join).
- **bend** (a known risk; recorded, not a rule): a bank's radius of curvature R (over 2 x BEND_SIGMA_H join
  cells, on the bank the block slides on) against W / (FUNNEL_RATIO^N - 1), and the cell-scale wiggle h / R; neither
  separated the good blocks from the bad ones on the step-1 cases (B158), so they are only reported (bend_rule).
- **room**: the size target h_t(s) (W(s) / N limited at slope G_RULE = ln RATIO along the axis) must have grown to
  the join's host size within one ratio step: RATIO h_t(s_join) >= h_join. For a single neck in a uniform host this
  is the distance (h_host - h_fine) / ln 1.15.

Growth (B159): candidates every STEP_H host cells along the axis, outward from the neck up to `reach` (default
REACH_H host cells), three shapes each (a straight transect and two arcs bulging away from the neck, BULGE), the
join = the host edge path closest to the shape (B160: Dijkstra, cost = length + W_DEV x distance from the shape, edges
farther than CORRIDOR_H x their length from it and every coast node but the two ends excluded). The first candidate
(smallest distance from the neck, a tier-2 chain counted TIER2_SLACK_H host cells farther) that passes all three
rules is taken. Without one the region is refused with the rule that fails and what would fix it.

Tiers (B160): tier 1 = the chain follows a host grid line (every inner chain node an interior pure-quad node of
valence 4 with the chain going straight through it); tier 2 = anything else (zigzag, triangle host), cleaned later by
the mesh-fixing pipeline.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from meshom.insert.geom import Axis, Polyline, ray_hits
from meshom.insert.host import Graph, coast_path, coast_walk

RATIO = 1.15                       # B158: the neighbour size ratio the rules aim at (room rule)
G_RULE = float(np.log(RATIO))      # B158: ln 1.15 = 0.140: the room rule's size slope along the axis
FUNNEL_RATIO = 1.2                 # B158: the funnel refuses a predicted row ratio exp(alpha / N) above this
                                   # (calibration: the best synthetic strait, alpha 84 deg, N 10, predicts
                                   # 1.158 and measured p99 1.19 with max angle 111 deg; 1.15 would refuse it)
CORNER_DEG = (70.0, 110.0)         # B158: the corner rule
CORNER_CHORD_H = 1.5               # B158: the chain's direction at its end: chord over this many join cells
BANK_CHORD_H = 3.0                 # B158: the bank's direction: chord over this many join cells along the coast
                                   # (a staircase coast has a direction only over several of its steps)
DEFAULT_CELLS = 10                 # B159: cells across the neck when the spec gives neither cells nor h_fine_m
REACH_H = 12.0                     # B159: default growth limit per side, in host cells at the neck
STEP_H = 0.5                       # B159: candidate spacing along the axis, in host cells
BEND_SIGMA_H = 1.5                 # B158: the bend rule's wide window (half length), in join cells
# The bend terms are recorded, not rules: they did not separate good from bad blocks (B158): the cell-scale wiggle
# h / R was 1.17 and 1.51 for clean cores, 1.29, 1.32, 1.48, 1.6 for bad ones; the wide term failed clean ones.
TRANSECT_H = 0.25                  # B159: transect spacing of the width profile, in host cells
NECK_INNER = (0.1, 0.9)            # B159: the neck is the narrowest section of the path's inner part
BULGE = (None, 1.0, 0.0)           # B159: candidate shapes: a transect; arcs centred k d behind the neck (radius
                                   # ~ (1 + k) d): k = 1 (radius 2d), k = 0 (centred at the neck, radius d)
W_DEV = 4.0                        # B160: chain cost = length + W_DEV x distance of the edge midpoint from the shape
CORRIDOR_H = 3.0                   # B160: edges farther than this x their length from the shape are excluded
TIER2_SLACK_H = 3.0                # B160 / B167: a join counts zig x this many host cells farther (zig = the share
                                   # of its inner nodes where it does not run straight along a host grid line)
TURN_W = 1.0                       # B167: chain cost of a turn: TURN_W x edge length x min(1, turn / 90 deg), 0 where
                                   # the chain runs straight through an interior pure-quad valence-4 node


TIER2_GRID_H = 12.0                # B193: on a structured host a tier-2 join counts this many host cells farther out
TURN_W_GRID = 4.0                  # B193: the turn cost on a structured host (ORCA): joins along long grid-line runs
STRUCTURED = 0.98                  # B193: a host window is structured when this share of its interior nodes are
                                   # pure-quad valence-4 nodes (ORCA; meshom's own meshes stay below)


@dataclass
class SideRule:
    """B200 / B201: what the block will do on one side of the neck, for the rules: `cells` = the count of the join
    row (None: any count, the block's counts follow the size target); `growth` = the size target's slope along the
    block on this side (dh/ds; block.G_LAYOUT by default). With `cells` the join must have exactly that many edges
    and the block's counts run monotonically from the neck's count to it (equal counts: the rows only widen), so the
    funnel rule takes N(s) within those two counts; `growth` scales the room rule's slope and the funnel's ratio
    (ln 1.15 and ln 1.2 at the default 0.12) by growth / 0.12: a short, fast transition where it is asked for.
    `widen` (B207): with `cells` the counts may rise above the neck's and the join's where the rows are long and come
    back to the join's count before the join; the funnel rule then takes N(s) held only at or above the smaller."""
    cells: int | None = None
    growth: float | None = None
    widen: bool = False

    @property
    def scale(self) -> float:
        from meshom.insert.block import G_LAYOUT

        return 1.0 if self.growth is None else float(self.growth) / G_LAYOUT

    @property
    def g_rule(self) -> float:
        return G_RULE * self.scale

    @property
    def ln_funnel(self) -> float:
        return float(np.log(FUNNEL_RATIO)) * self.scale


def side_target(prof, n: int, rule: "SideRule | None") -> np.ndarray:
    """The room rule's size target with n cells across the neck and the side's slope (B158, B201)."""
    if rule is None or rule.growth is None:
        if n == prof.n_cells:
            return prof.h_t
        return cone_min(prof.s, np.where(np.isfinite(prof.W), prof.W / n, np.inf), G_RULE)
    return cone_min(prof.s, np.where(np.isfinite(prof.W), prof.W / n, np.inf), rule.g_rule)


def funnel_counts(Ns: np.ndarray, n_neck: int, rule: "SideRule | None") -> np.ndarray:
    """B200: the cells across the funnel's sections as the block lays them: the square-cell count W / h, or with a
    set join count clipped to between the neck's count and the join's (equal counts: the neck's count throughout);
    with `widen` (B207) only held at or above the smaller of the two."""
    if rule is None or rule.cells is None:
        return Ns
    if rule.widen:
        return np.maximum(Ns, min(n_neck, rule.cells))
    return np.clip(Ns, min(n_neck, rule.cells), max(n_neck, rule.cells))


def structured(g) -> tuple[bool, float]:
    """B193: (structured, share): the share of the window's interior nodes that are pure-quad valence-4 nodes."""
    inner = ~g.boundary
    share = float(((g.tris_per_node == 0) & (g.elems_per_node == 4))[inner].mean()) if inner.any() else 0.0
    return share >= STRUCTURED, share


class InsertError(RuntimeError):
    """The insert cannot be built as specified; the message says which rule fails and what would fix it."""


@dataclass
class Profile:
    s: np.ndarray                  # (K,) axis arc length (m, chart)
    xy: np.ndarray                 # (K,2) axis points
    t: np.ndarray                  # (K,2) unit tangents
    n: np.ndarray                  # (K,2) unit left normals
    W: np.ndarray                  # (K,) width (inf where the transect is not across water)
    wet: np.ndarray                # (K,) bool: the axis point lies in the host's water
    alpha: np.ndarray              # (K,) opening angle (rad, nan where W is not defined on both sides)
    h_t: np.ndarray                # (K,) size target
    i0: int                        # neck index
    n_cells: int                   # cells across the neck
    h_fine: float                  # W at the neck / n_cells
    h_min: float = 0.0             # the size floor (B165)
    n_max: int = 10 ** 9           # most cells across the neck the floor allows: floor(W0 / h_min)
    floor_note: str = ""           # what the floor changed in the request (empty: nothing)
    tl: np.ndarray | None = None   # (K,) distance to the coast left / right of the axis (inf: none; B183)
    tr: np.ndarray | None = None
    h_host: float = 0.0

    @property
    def s0(self) -> float:
        return float(self.s[self.i0])


@dataclass
class Join:
    side: str                      # "A" (before the neck along the axis) or "B"
    nodes: np.ndarray              # window node ids, right bank -> left bank
    shape: str                     # "transect", "arc2d", "arcd", "hint"
    d_m: float                     # distance from the neck along the axis
    s_m: float                     # axis arc length of the join
    tier: int
    n: int                         # edges
    h_join_m: float                # mean edge length (sphere)
    corner_deg: tuple              # (right end, left end)
    room: float                    # RATIO h_t(s_join) / h_join (>= 1 passes)
    funnel: float                  # max over sections of alpha / (N(s) ln RATIO) (<= 1 passes)
    funnel_at_m: float             # distance from the neck of the worst section
    fails: list = field(default_factory=list)
    bend: float = 0.0              # max over the banks of W / ((FUNNEL_RATIO^N - 1) R) (<= 1 passes)
    bend_cells: int = 0            # cells across the neck that would pass the bend rule's wide term
    bend_wide: float = 0.0
    bend_cell: float = 0.0
    zig: float = 0.0               # B167: share of inner nodes off a host grid line (0 = tier 1)
    sec: tuple | None = None       # B184: (room index, section indices, widths, opening angles) to re-check the
                                   # room and funnel rules at another cell count
    square: tuple = ()             # B202: per end, the squared bank end's depth into land over the allowance
                                   # (<= 1: the end may be squared; empty: no allowance)
    rule: object = None            # B200 / B201: the side's SideRule (None: the default rules)

    @property
    def ok(self) -> bool:
        return not self.fails

    def record(self) -> dict:
        return {"side": self.side, "shape": self.shape, "d_km": round(self.d_m / 1e3, 3), "tier": self.tier,
                "n": self.n, "h_join_km": round(self.h_join_m / 1e3, 3),
                "corner_deg": [round(float(a), 1) for a in self.corner_deg], "room": round(self.room, 3),
                "funnel": round(self.funnel, 3), "funnel_at_km": round(self.funnel_at_m / 1e3, 2),
                "bend": round(self.bend, 3), "bend_cells": self.bend_cells,
                "bend_wide": round(self.bend_wide, 3), "bend_cell": round(self.bend_cell, 3),
                "zig": round(self.zig, 3), "fails": self.fails,
                "square": [round(float(v), 3) for v in self.square]}


def _nan_smooth(v: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    ok = np.isfinite(v)
    if sigma <= 0:
        return np.where(ok, v, np.nan)
    a = gaussian_filter1d(np.where(ok, v, 0.0), sigma, mode="nearest")
    b = gaussian_filter1d(ok.astype(float), sigma, mode="nearest")
    return np.where(ok & (b > 1e-6), a / np.maximum(b, 1e-12), np.nan)


def cone_min(s: np.ndarray, v: np.ndarray, g: float) -> np.ndarray:
    """min over s' of v(s') + g |s - s'| (inf entries ignored)."""
    ok = np.isfinite(v)
    if not ok.any():
        return np.full(len(s), np.inf)
    return np.min(v[ok][None, :] + g * np.abs(s[:, None] - s[ok][None, :]), axis=1)


def width_profile(axis: Axis, seg: np.ndarray, in_water, h_host: float, n_cells: int | None,
                  h_fine: float | None, h_min: float = 0.0) -> Profile:
    """Transects across the axis every TRANSECT_H host cells: the width W(s) (first coast crossings either side,
    the axis point in water), the neck (narrowest in NECK_INNER of the path), the size target and alpha."""
    ds = TRANSECT_H * h_host
    s = np.arange(0.0, axis.L + 1e-6, ds)
    o, t, n = axis.frame(s, max(ds, 0.5 * h_host))
    tl, _ = ray_hits(o, n, seg)
    tr, _ = ray_hits(o, -n, seg)
    wet = in_water(o)
    W = np.where(wet & np.isfinite(tl) & np.isfinite(tr), tl + tr, np.inf)
    a, b = axis.s_path
    inner = (s >= a + NECK_INNER[0] * (b - a)) & (s <= a + NECK_INNER[1] * (b - a)) & np.isfinite(W)
    if not inner.any():
        raise InsertError("the host has no water across the strait's path (the strait is closed in this mesh, or "
                          "the path misses it): nothing to insert into")
    i0 = int(np.flatnonzero(inner)[np.argmin(W[inner])])
    W0 = float(W[i0])
    what = (f"{n_cells} cells" if n_cells is not None else
            f"h_fine_m {h_fine / 1e3:.2f} km" if h_fine else f"the default {DEFAULT_CELLS} cells")
    if n_cells is None:
        n_cells = max(2, int(round(W0 / h_fine))) if h_fine else DEFAULT_CELLS
    n_max, note = 10 ** 9, ""
    if h_min > 0:                                      # B165: the tool never goes below h_min
        n_max = int(np.floor(W0 / h_min + 1e-9))
        if n_max < 2:
            raise InsertError(f"the neck ({W0 / 1e3:.2f} km) holds fewer than 2 cells of the size floor h_min_m "
                              f"{h_min / 1e3:.2f} km; fix: a smaller h_min_m (<= {W0 / 2e3:.2f} km)")
        if W0 / n_cells < h_min * (1 - 1e-9):
            note = (f"{what} across the {W0 / 1e3:.2f} km neck would need cells of {W0 / n_cells / 1e3:.2f} km, below "
                    f"the size floor h_min_m {h_min / 1e3:.2f} km: using {n_max} cells ({W0 / n_max / 1e3:.2f} km)")
            n_cells = n_max
    hf = W0 / n_cells
    h_t = cone_min(s, np.where(np.isfinite(W), W / n_cells, np.inf), G_RULE)
    Ws = _nan_smooth(np.where(np.isfinite(W), W, np.nan), h_host / ds)
    dW = np.gradient(Ws, ds)
    alpha = 2.0 * np.arctan(0.5 * np.abs(dW))
    return Profile(s, o, t, n, W, wet, alpha, h_t, i0, int(n_cells), hf, float(h_min), n_max, note,
                   np.where(wet, tl, np.inf), np.where(wet, tr, np.inf), float(h_host))


def first_crossing(pts: np.ndarray, seg: np.ndarray) -> tuple[int, np.ndarray, int]:
    """First crossing of the polyline pts (K,2) with the segments seg (S,2,2): (index of the polyline segment, the
    point, the coast segment) or (-1, None, -1)."""
    if len(pts) < 2:
        return -1, None, -1
    a, e = seg[:, 0], seg[:, 1] - seg[:, 0]
    for lo in range(0, len(pts) - 1, 128):
        p = pts[lo:min(lo + 129, len(pts))]
        o, d = p[:-1], p[1:] - p[:-1]
        den = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
        w = a[None] - o[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (w[..., 0] * e[None, :, 1] - w[..., 1] * e[None, :, 0]) / den
            u = (w[..., 0] * d[:, None, 1] - w[..., 1] * d[:, None, 0]) / den
        ok = (np.abs(den) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        if ok.any():
            tt = np.where(ok, t, np.inf)
            r = int(np.flatnonzero(ok.any(1))[0])                          # first polyline segment with a hit
            j = int(np.argmin(tt[r]))
            return lo + r, o[r] + tt[r, j] * d[r], j
    return -1, None, -1


class Chains:
    """Chain finding on the window's host edge graph."""

    def __init__(self, g: Graph, xy: np.ndarray, coast_nb: list, quads: np.ndarray):
        self.g, self.xy, self.nb, self.quads = g, xy, coast_nb, quads
        e = g.edges
        self.mid = 0.5 * (xy[e[:, 0]] + xy[e[:, 1]])
        self.len = np.linalg.norm(xy[e[:, 0]] - xy[e[:, 1]], axis=1)
        self.bnd_edge = g.boundary[e[:, 0]] | g.boundary[e[:, 1]]
        self.turn_w = TURN_W                               # B167; TURN_W_GRID on a structured host (B193)

    def path(self, a: int, b: int, shape: Polyline, exclude=None) -> np.ndarray | None:
        """The host edge path from a to b closest to `shape` (B160) and straightest (B167): Dijkstra over directed
        edges, cost = length + W_DEV x distance from the shape + TURN_W x length x min(1, turn / 90 deg), the turn
        free where the path runs straight through an interior pure-quad valence-4 node (along a host grid line)."""
        import heapq

        e = self.g.edges
        dev = shape.distance(self.mid)
        ok = dev <= CORRIDOR_H * self.len + 1e-6
        touch = self.bnd_edge.copy()
        for v in (a, b):                                   # the two ends may be coast nodes
            touch &= ~(((e[:, 0] == v) & ~self.g.boundary[e[:, 1]]) | ((e[:, 1] == v) & ~self.g.boundary[e[:, 0]]))
        ok &= ~touch
        if exclude is not None and len(exclude):           # nodes another chain already uses (a corridor, B182)
            bad = np.zeros(len(self.xy), bool)
            bad[np.asarray(list(exclude), dtype=np.int64)] = True
            bad[[a, b]] = False
            ok &= ~(bad[e[:, 0]] | bad[e[:, 1]])
        w = self.len + W_DEV * dev
        adj = {}
        for k in np.flatnonzero(ok):
            u, v = int(e[k, 0]), int(e[k, 1])
            adj.setdefault(u, []).append((v, k))
            adj.setdefault(v, []).append((u, k))
        if a not in adj or b not in adj:
            return None
        best = {(-1, a): 0.0}
        prev = {}
        heap = [(0.0, -1, a)]
        goal = None
        while heap:
            c, p, u = heapq.heappop(heap)
            if c > best.get((p, u), np.inf):
                continue
            if u == b:
                goal = (p, u)
                break
            for v, k in adj[u]:
                if v == p:
                    continue
                cc = c + w[k] + (0.0 if p < 0 else self.turn_w * self.len[k] * self.turn(p, u, v))
                if cc < best.get((u, v), np.inf):
                    best[(u, v)] = cc
                    prev[(u, v)] = (p, u)
                    heapq.heappush(heap, (cc, u, v))
        if goal is None:
            return None
        out = [goal[1]]
        st = goal
        while st in prev:
            st = prev[st]
            out.append(st[1])
        return np.array(out[::-1], dtype=np.int64)

    def straight(self, u: int, v: int, w: int) -> bool:
        """The chain u-v-w runs straight along a host grid line at v (interior pure-quad valence 4, u and w in no
        common quad of v)."""
        g = self.g
        if g.boundary[v] or g.tris_per_node[v] or g.elems_per_node[v] != 4:
            return False
        return not any(u in self.quads[q] and w in self.quads[q] for q in g.node_quads[v])

    def turn(self, u: int, v: int, w: int) -> float:
        if self.straight(u, v, w):
            return 0.0
        d1, d2 = self.xy[v] - self.xy[u], self.xy[w] - self.xy[v]
        c = float(np.dot(d1, d2) / max(np.linalg.norm(d1) * np.linalg.norm(d2), 1e-12))
        return min(1.0, float(np.degrees(np.arccos(np.clip(c, -1, 1)))) / 90.0)

    def zig(self, chain: np.ndarray) -> float:
        """Share of the chain's inner nodes where it does not run straight along a host grid line (0: tier 1)."""
        if len(chain) < 3:
            return 0.0
        k = sum(0 if self.straight(int(u), int(v), int(w)) else 1 for u, v, w in zip(chain[:-2], chain[1:-1], chain[2:]))
        return k / (len(chain) - 2)

    def neighbours(self, v: int) -> list:
        if not hasattr(self, "_adj"):
            self._adj = [[] for _ in range(len(self.xy))]
            for a, b in self.g.edges:
                self._adj[a].append(int(b))
                self._adj[b].append(int(a))
        return self._adj[v]

    def gridline(self, v0: int, max_steps: int) -> np.ndarray | None:
        """The host grid line from the coast node v0 (two quads, no triangle: its one interior edge) straight through
        interior pure-quad valence-4 nodes to the next coast node; None when it meets an irregular node first."""
        g = self.g
        if g.tris_per_node[v0] or g.elems_per_node[v0] != 2:
            return None
        inner = [w for w in self.neighbours(v0)
                 if sum(1 for q in g.node_quads[v0] if w in self.quads[q]) == 2]
        if len(inner) != 1:
            return None
        path = [v0, inner[0]]
        for _ in range(max_steps):
            u, prev = path[-1], path[-2]
            if g.boundary[u]:
                return np.array(path)
            if g.tris_per_node[u] or g.elems_per_node[u] != 4:
                return None
            nxt = [w for w in self.neighbours(u) if w != prev and
                   not any(prev in self.quads[q] and w in self.quads[q] for q in g.node_quads[u])]
            if len(nxt) != 1:
                return None
            path.append(nxt[0])
        return None

    def tier(self, chain: np.ndarray) -> int:
        g = self.g
        for u, v, w in zip(chain[:-2], chain[1:-1], chain[2:]):
            if g.boundary[v] or g.tris_per_node[v] or g.elems_per_node[v] != 4:
                return 2
            for q in g.node_quads[v]:
                if u in self._quad(q) and w in self._quad(q):
                    return 2
        return 1

    def _quad(self, q):
        return self.quads[q]


def _chain_point(xy: np.ndarray, length: float) -> np.ndarray:
    pl = Polyline(xy)
    return pl.at(min(length, pl.L))[0]


def evaluate(side: str, sigma: int, chain: np.ndarray, shape: str, s_join: float, prof: Profile, ch: Chains,
             axis: Axis, length_m: np.ndarray, bank_fn=None, bank_dir=None, bay_hit=None, leg=None, rule=None,
             square=None) -> Join:
    """A candidate join against the three rules (module docstring). bank_dir(end xy, length) -> the point of the
    bank the block will follow `length` from the join end toward the neck (the real coast, B164); None: the host
    coast. rule: the side's SideRule (B200, B201). square(end xy, unit chain direction into the join, join cell) ->
    the squared bank end's depth into land over the allowance (B202): an end whose corner fails the angle passes
    when it is <= 1."""
    xy = ch.xy[chain]
    n = len(chain) - 1
    hj = float(length_m / max(n, 1))
    fails = []
    sp = axis.project(xy)
    if np.any(sigma * (sp - prof.s0) < prof.h_fine):
        fails.append("crosses the neck")
    if n < 2:
        fails.append("fewer than 2 join edges")
    if rule is not None and rule.cells is not None and n != rule.cells:
        fails.append("count")                          # B200: the join row has the side's count, 1:1
    corner, banks, sq = [], [], []
    for end, inner in ((0, xy), (len(chain) - 1, xy[::-1])):
        v0 = int(chain[end])
        u = _chain_point(inner, CORNER_CHORD_H * hj) - xy[end]
        nb = ch.nb[v0]
        if len(nb) != 2:
            corner.append(np.nan)
            continue
        far = [coast_walk(ch.nb, v0, f, BANK_CHORD_H * hj, ch.xy) for f in nb]
        dn = [abs(float(axis.project(ch.xy[f][None])[0]) - prof.s0) for f in far]
        toward = int(np.argmin(dn))
        v = ch.xy[far[toward]] - xy[end]
        if bank_dir is not None:
            q = bank_dir(xy[end], BANK_CHORD_H * hj)
            if q is not None:
                v = q - xy[end]
        c = float(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12))
        corner.append(float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
        if square is not None:
            sq.append(float(square(xy[end], u / max(np.linalg.norm(u), 1e-12), hj)))
        stop = lambda w: abs(float(axis.project(ch.xy[w][None])[0]) - prof.s0) <= prof.h_fine   # noqa: E731
        bank = coast_path(ch.nb, v0, nb[toward], ch.xy, stop, 3.0 * (abs(s_join - prof.s0) + prof.W[prof.i0]))
        banks.append(bank)
    ends_ok = [np.isfinite(a) and CORNER_DEG[0] <= a <= CORNER_DEG[1] for a in corner]
    if sq and len(sq) == len(ends_ok):                 # B202: a squared end within the allowance passes
        ends_ok = [e or q <= 1.0 for e, q in zip(ends_ok, sq)]
    if not all(ends_ok):
        fails.append("corner")
    if bay_hit is not None and bay_hit(xy):                # B166: a join must not cut a bay of the real coast
        fails.append("bay")
    k = int(np.argmin(np.abs(prof.s - s_join)))
    W_use, alpha_use = prof.W, prof.alpha
    if leg is not None:                                # B183: an L-join: the room at its node nearest to the neck,
        near = float(np.min(sigma * (sp - prof.s0)))     # the funnel on the width the L's leg bounds
        k = int(np.argmin(np.abs(prof.s - (prof.s0 + sigma * near))))
        W_use, alpha_use = l_width(prof, *leg)
    h_t = side_target(prof, prof.n_cells, rule)
    room = float(RATIO * h_t[k] / hj)
    if room < 1.0:
        fails.append("room")
    kj = int(np.argmin(np.abs(prof.s - s_join)))
    lo, hi = sorted((prof.i0, kj))
    sec = np.arange(lo, hi + 1)
    hh = np.minimum(h_t[sec], hj)
    Ns = funnel_counts(W_use[sec] / hh, prof.n_cells, rule)
    r = alpha_use[sec] / (np.maximum(Ns, 1e-9) * (rule.ln_funnel if rule is not None else np.log(FUNNEL_RATIO)))
    r = np.where(np.isfinite(r), r, 0.0)
    j = int(np.argmax(r)) if len(r) else 0
    funnel = float(r[j]) if len(r) else 0.0
    if funnel > 1.0:
        fails.append("funnel")
    bend, need, bw, bc = bend_rule(banks, ch.xy, prof, axis, hj, bank_fn)
    return Join(side, chain, shape, abs(s_join - prof.s0), s_join, ch.tier(chain), n, hj, tuple(corner), room,
                funnel, float(abs(prof.s[sec[j]] - prof.s0)) if len(sec) else 0.0, fails, bend, need, bw, bc,
                ch.zig(chain), (k, sec, W_use[sec], alpha_use[sec]), tuple(sq), rule)


def bend_rule(banks: list, xy: np.ndarray, prof: Profile, axis: Axis, hj: float, bank_fn=None) -> tuple:
    """The bend rule (B158, a known risk) on the bank the block will slide on (bank_fn: the capped smoothing
    of B161): square cells across a bend of a channel of width W with N cells grow from the inner to the outer bank
    by (R_in + W) / R_in, so the bank's radius of curvature over 2 x BEND_SIGMA_H join cells needs R >= W /
    (FUNNEL_RATIO^N - 1), N = N(s) as in the funnel rule ("wide" term); and no block cell wraps a bank wiggle
    tighter than its own size h = W / N, the radius taken over +-1 block cell ("cell" term, recorded only).
    Returns (the wide term (the rule), cells across the neck that pass it, wide, cell)."""
    from scipy.ndimage import gaussian_filter1d

    wide = cell = 0.0
    need = 0
    for b in banks:
        if b is None or len(b) < 3:
            continue
        p = Polyline(bank_fn(xy[b]) if bank_fn is not None else xy[b])
        k_all = np.array([int(np.argmin(np.abs(prof.s - s))) for s in axis.project(p.xy)])
        W_all = prof.W[k_all]
        hh_all = np.minimum(prof.h_t[k_all], hj)
        fin = np.isfinite(W_all)
        if not fin.any():
            continue
        hloc = float(np.min(hh_all[fin]))                               # the smallest block cell along the bank
        step = max(min(0.25 * hloc, 0.25 * BEND_SIGMA_H * hj), 1.0)
        if p.L < 8 * step:
            continue
        t = np.arange(0.0, p.L, step)
        q = p.at(t)
        if bank_fn is None:
            q = gaussian_filter1d(q, 4.0, axis=0, mode="nearest")
        d = np.diff(q, axis=0)
        th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        mid = 0.5 * (q[1:] + q[:-1])
        k = np.array([int(np.argmin(np.abs(prof.s - s))) for s in axis.project(mid)])
        W = prof.W[k]
        hh = np.minimum(prof.h_t[k], hj)
        N = W / hh
        ok = np.isfinite(W) & np.isfinite(N)
        n = len(th)
        idx = np.arange(n)
        for term, half in (("wide", np.full(n, BEND_SIGMA_H * hj)), ("cell", hh)):
            w = np.maximum(1, np.round(half / step).astype(int))
            lo, hi = np.clip(idx - w, 0, n - 1), np.clip(idx + w, 0, n - 1)
            R = (hi - lo) * step / np.maximum(np.abs(th[hi] - th[lo]), 1e-9)
            good = ok & (hi - lo >= 2)
            if not good.any():
                continue
            if term == "wide":
                r = W[good] / ((FUNNEL_RATIO ** N[good] - 1.0) * R[good])
                i = int(np.argmax(r))
                if r[i] > wide:
                    wide = float(r[i])
                    n_need = np.log1p(W[good][i] / R[good][i]) / np.log(FUNNEL_RATIO)
                    need = int(np.ceil(prof.n_cells * n_need / max(N[good][i], 1e-9)))
            else:
                cell = max(cell, float(np.max(hh[good] / R[good])))
    return wide, need, wide, cell


def _shape_points(prof: Profile, axis: Axis, sigma: int, d: float, bulge, span: float, step: float):
    """Sample points of a candidate shape through the axis point at s0 + sigma d, ordered from its right end to its
    left end (tau from -span to span; tau > 0 = left of the axis)."""
    s_k = prof.s0 + sigma * d
    p, t, nrm = axis.frame(np.array([s_k]), max(step, 1.0))
    p, nrm = p[0], nrm[0]
    tau = np.arange(-span, span + 1e-9, step)
    if bulge is None:
        return s_k, p + tau[:, None] * nrm[None], tau
    c = axis.at(np.clip(prof.s0 - sigma * bulge * d, 0, axis.L))[0]
    r = float(np.linalg.norm(p - c))
    if r < step:
        return s_k, None, None
    tau = tau[np.abs(tau) <= 0.5 * np.pi * r * 1.2]
    th0 = np.arctan2(*(p - c)[::-1])
    th = th0 + sigma * tau / r
    return s_k, c + r * np.c_[np.cos(th), np.sin(th)], tau


def candidates(side: str, prof: Profile, axis: Axis, seg: np.ndarray, coast_seg_nodes: np.ndarray, ch: Chains,
               sphere_len, h_host: float, reach: float, bank_fn=None, bank_dir=None, bay_hit=None, rule=None,
               square=None) -> list[Join]:
    """Every candidate join of one side (module docstring), nearest first."""
    sigma = -1 if side == "A" else 1
    step = 0.25 * h_host
    out, seen = [], set()
    d = max(h_host, 2.0 * prof.h_fine)
    while d <= reach + 1e-6:
        s_k = prof.s0 + sigma * d
        if not (0.0 <= s_k <= axis.L):
            break
        k = int(np.argmin(np.abs(prof.s - s_k)))
        if prof.wet[k]:
            for bulge in BULGE:
                name = {None: "transect", 1.0: "arc2d", 0.0: "arcd"}[bulge]
                _, pts, tau = _shape_points(prof, axis, sigma, d, bulge, 4.0 * reach + 4 * h_host, step)
                if pts is None:
                    continue
                c = int(np.argmin(np.abs(tau)))
                iL, qL, jL = first_crossing(pts[c:], seg)
                iR, qR, jR = first_crossing(pts[c::-1], seg)
                if iL < 0 or iR < 0:
                    continue
                shape = Polyline(np.vstack([qR, pts[c - iR:c + iL + 1], qL]))
                nR = int(coast_seg_nodes[jR][np.argmin(np.linalg.norm(ch.xy[coast_seg_nodes[jR]] - qR, axis=1))])
                nL = int(coast_seg_nodes[jL][np.argmin(np.linalg.norm(ch.xy[coast_seg_nodes[jL]] - qL, axis=1))])
                if nR == nL:
                    continue
                chain = ch.path(nR, nL, shape)
                if chain is None:
                    continue
                key = tuple(chain.tolist())
                if key in seen:
                    continue
                seen.add(key)
                out.append(evaluate(side, sigma, chain, name, s_k, prof, ch, axis, sphere_len(chain), bank_fn,
                                    bank_dir, bay_hit, rule=rule, square=square))
        d += STEP_H * h_host
    # tier-1 candidates: host grid lines from the coast nodes of this side (B160)
    cn = np.unique(coast_seg_nodes)
    sp = axis.project(ch.xy[cn])
    far = sigma * (sp - prof.s0)
    for v0 in cn[(far >= max(h_host, 2.0 * prof.h_fine)) & (far <= reach)]:
        line = ch.gridline(int(v0), int(4 * reach / max(prof.h_fine, 1.0)))
        if line is None or len(line) < 3:
            continue
        lxy = ch.xy[line]
        sl = axis.project(lxy)
        k = np.clip(np.searchsorted(prof.s, sl), 0, len(prof.s) - 1)
        side_n = np.einsum("ij,ij->i", lxy - prof.xy[k], prof.n[k])
        if side_n[0] * side_n[-1] >= 0:                    # both ends on one bank
            continue
        if side_n[0] > 0:
            line = line[::-1]
        key = tuple(line.tolist())
        if key in seen:
            continue
        seen.add(key)
        s_k = float(sl[np.argmin(np.abs(side_n))])
        out.append(evaluate(side, sigma, line, "gridline", s_k, prof, ch, axis, sphere_len(line), bank_fn, bank_dir,
                            bay_hit, rule=rule, square=square))
    out.sort(key=lambda c: c.d_m)
    return out


def hinted(side: str, pts_xy: np.ndarray, prof: Profile, axis: Axis, coast_nodes: np.ndarray, ch: Chains,
           sphere_len, bank_fn=None, bank_dir=None, bay_hit=None, rule=None, square=None) -> Join:
    """The join of a hint polyline (its ends snap to the nearest coast nodes; ordered right -> left)."""
    sigma = -1 if side == "A" else 1
    shape = Polyline(pts_xy)
    a = int(coast_nodes[np.argmin(np.linalg.norm(ch.xy[coast_nodes] - pts_xy[0], axis=1))])
    b = int(coast_nodes[np.argmin(np.linalg.norm(ch.xy[coast_nodes] - pts_xy[-1], axis=1))])
    s_mid = float(axis.project(shape.at(0.5 * shape.L))[0])
    k = int(np.argmin(np.abs(prof.s - s_mid)))
    if np.dot(ch.xy[a] - prof.xy[k], prof.n[k]) > np.dot(ch.xy[b] - prof.xy[k], prof.n[k]):
        a, b = b, a
        shape = Polyline(pts_xy[::-1])
    chain = ch.path(a, b, shape)
    if chain is None:
        raise InsertError(f"side {side}: no host edge path follows the hint")
    return evaluate(side, sigma, chain, "hint", s_mid, prof, ch, axis, sphere_len(chain), bank_fn, bank_dir, bay_hit,
                    rule=rule, square=square)


def choose(cands: list[Join], h_host: float, force: bool, tier1: bool = False) -> Join | None:
    """The passing candidate nearest to the neck, a tier-2 one paying B167's slack; tier1 (a structured host, B193):
    a tier-2 one pays TIER2_GRID_H host cells instead (the host's nodes stay, so a straight grid line is worth a
    longer block)."""
    ok = [c for c in cands if c.ok]
    if ok:
        def slack(c):
            if c.tier == 1:
                return 0.0
            return TIER2_GRID_H * h_host if tier1 else max(c.zig, 0.25) * TIER2_SLACK_H * h_host
        return min(ok, key=lambda c: (c.d_m + slack(c), max(abs(a - 90.0) for a in c.corner_deg)))
    if not force or not cands:
        return None
    valid = [c for c in cands if "crosses the neck" not in c.fails and c.n >= 2] or cands
    valid = [c for c in valid if "count" not in c.fails] or valid        # B200: the side's count first

    def violation(c):
        dev = max((max(CORNER_DEG[0] - a, a - CORNER_DEG[1], 0.0) if np.isfinite(a) else 90.0)
                  for a in c.corner_deg)
        return (max(0.0, 1.0 / max(c.room, 1e-9) - 1.0) + max(0.0, c.funnel - 1.0) + dev / 20.0)
    return min(valid, key=lambda c: (round(violation(c), 2), c.d_m))


def refusal(side: str, cands: list[Join], prof: Profile, reach: float, h_host: float) -> str:
    """Why no candidate of a side passes, and what would fix it."""
    km = 1e-3
    if not cands:
        return (f"side {side}: no host edge path crosses the strait from bank to bank within {reach * km:.0f} km of "
                f"the neck (the coast leaves the window, or an island blocks every transect); fix: join hints "
                f"(mouth_{side.lower()}) or a larger reach_km")
    valid = [c for c in cands if "crosses the neck" not in c.fails and c.n >= 2]
    lines = [f"side {side}: none of {len(cands)} candidate joins within {reach * km:.0f} km passes the rules"]
    if not valid:
        return lines[0] + " (every candidate crosses the neck or has fewer than 2 edges)"
    room = [c for c in valid if "room" not in c.fails]
    corner = [c for c in valid if "corner" not in c.fails]
    funnel = [c for c in valid if "funnel" not in c.fails]
    best_room = max(valid, key=lambda c: c.room)
    if not room:
        lines.append(f"room fails everywhere: the size target grows from {prof.h_fine * km:.2f} km "
                     f"(neck {prof.W[prof.i0] * km:.1f} km / {prof.n_cells} cells) only to "
                     f"{best_room.room * best_room.h_join_m / RATIO * km:.2f} km at the best join "
                     f"({best_room.d_m * km:.0f} km out), "
                     f"host {best_room.h_join_m * km:.1f} km")
    else:
        lines.append(f"room holds from {min(c.d_m for c in room) * km:.0f} km out")
    if not corner:
        angs = [a for c in valid for a in c.corner_deg if np.isfinite(a)]
        lines.append(f"corner fails everywhere (joins meet the banks at {min(angs):.0f}-{max(angs):.0f} deg, need "
                     f"{CORNER_DEG[0]:.0f}-{CORNER_DEG[1]:.0f})" if angs else "corner: no measurable bank direction")
    else:
        ds = sorted({round(c.d_m * km) for c in corner})
        lines.append(f"corner holds at {ds[:8]}{'...' if len(ds) > 8 else ''} km")
    if not funnel:
        worst = min(valid, key=lambda c: c.funnel)
        lines.append(f"funnel fails everywhere (best {worst.funnel:.2f} x the limit at {worst.funnel_at_m * km:.0f} km)")
    def only(rule):
        return [c for c in valid if c.fails == [rule]]
    fixes = []
    c_room = only("room")
    if c_room:
        c = max(c_room, key=lambda c: c.room)
        fixes.append(f"room only fails at the join {c.d_m * km:.0f} km out: a finer host (host cells <= "
                     f"{c.room * c.h_join_m * km:.1f} km there, now {c.h_join_m * km:.1f} km)")
        n_ok = None
        for n2 in range(prof.n_cells - 1, 1, -1):
            ht = side_target(prof, n2, c.rule)
            k = int(np.argmin(np.abs(prof.s - c.s_m)))
            if RATIO * ht[k] >= c.h_join_m:
                n_ok = n2
                break
        if n_ok:
            fixes.append(f"fewer cells across the neck (cells <= {n_ok}, fine size >= "
                         f"{prof.W[prof.i0] / n_ok * km:.2f} km)")
    c_fun = only("funnel")
    if c_fun:
        c = min(c_fun, key=lambda c: c.funnel)
        sg = -1.0 if side == "A" else 1.0
        k = int(np.argmin(np.abs(prof.s - (prof.s0 + sg * c.funnel_at_m))))
        need = int(np.ceil(prof.n_cells * c.funnel))
        msg = (f"funnel only fails at the join {c.d_m * km:.0f} km out (worst section {c.funnel_at_m * km:.0f} "
               f"km from the neck, opening {np.degrees(prof.alpha[k]):.0f} deg): more cells across (cells >= {need})")
        if need > prof.n_max:                          # B165: say when the fix would cross the size floor
            msg += (f", which needs cells of {prof.W[prof.i0] / need * km:.2f} km, below the size floor h_min_m "
                    f"{prof.h_min * km:.2f} km (at most {prof.n_max} cells): only with a smaller h_min_m")
        fixes.append(msg)
    c_cnt = [c for c in valid if "count" in c.fails]
    if c_cnt:                                          # B200: the join count the side asks for
        want = c_cnt[0].rule.cells
        right = [c for c in valid if "count" not in c.fails]
        other = sorted({c.n for c in valid if c.fails == ["count"]})
        lines.append(f"{len(right)} candidate(s) have the side's {want} edges"
                     + (f" (failing {sorted({f for c in right for f in c.fails})})" if right else "")
                     + (f"; candidates passing every other rule have {other} edges: cells_{side.lower()} = one of "
                        f"them" if other else ""))
    c_bay = [c for c in valid if "bay" in c.fails]
    if len(c_bay) == len(valid):
        fixes.append("every candidate cuts a bay of the real coast: a larger reach_km (joins beyond the bay) or "
                     "bays = false")
    c_cor = only("corner")
    if c_cor:
        c = min(c_cor, key=lambda c: c.d_m)
        fixes.append(f"corner only fails at the join {c.d_m * km:.0f} km out (meets the banks at "
                     f"{c.corner_deg[0]:.0f} / {c.corner_deg[1]:.0f} deg): a join hint (mouth_{side.lower()}) there")
    if not fixes:
        fixes.append(f"no candidate fails a single rule only; a finer host, more cells across or a larger region "
                     f"(reach_km > {reach * km:.0f}) with join hints")
    if fixes:
        lines.append("fix: " + "; or ".join(fixes))
    return "; ".join(lines)


L_LEGS = (1.0, 2.0, 3.0)          # B183: an L-join's leg runs at these multiples of max(neck width / 2, 2 host cells)
                                  # from the axis


def l_width(prof: Profile, side_l: float, ell: float) -> tuple[np.ndarray, np.ndarray]:
    """B183: the width profile of a region an L-join closes: the coast on one side, the L's leg (ell from the axis)
    on side side_l (+1 left, -1 right) where the coast there is farther; (W, alpha) as in width_profile."""
    a, b = (prof.tr, prof.tl) if side_l > 0 else (prof.tl, prof.tr)       # a: the other side, b: the leg's side
    W = np.where(np.isfinite(a), a + np.minimum(b, ell), np.inf)
    ds = float(prof.s[1] - prof.s[0]) if len(prof.s) > 1 else 1.0
    Ws = _nan_smooth(np.where(np.isfinite(W), W, np.nan), max(prof.h_host, ds) / ds)
    return W, 2.0 * np.arctan(0.5 * np.abs(np.gradient(Ws, ds)))


def candidates_l(side: str, prof: Profile, axis: Axis, seg: np.ndarray, coast_seg_nodes: np.ndarray, ch: Chains,
                 sphere_len, h_host: float, reach: float, bank_fn=None, bank_dir=None, bay_hit=None, rule=None,
                 square=None) -> list[Join]:
    """B183: L-shaped candidate joins (a strait mouth opening onto a coast, a cape at the mouth): from the axis point
    d beyond the neck a transect toward one side, but only out to ell from the axis (the coast there is farther), then
    a leg back toward the neck, parallel to the axis, to the coast beyond the cape; the region then holds the water
    round the cape. Rules as for any join, the room at the chain's node nearest to the neck and the funnel on the
    width the leg bounds (l_width). Nearest first."""
    sigma = -1 if side == "A" else 1
    step = 0.25 * h_host
    out, seen = [], set()
    base = max(0.5 * float(prof.W[prof.i0]), 2.0 * h_host)
    d = max(h_host, 2.0 * prof.h_fine)
    while d <= reach + 1e-6:
        s_k = prof.s0 + sigma * d
        if not (0.0 <= s_k <= axis.L):
            break
        k = int(np.argmin(np.abs(prof.s - s_k)))
        if prof.wet[k]:
            p, t, nrm = axis.frame(np.array([s_k]), max(step, 1.0))
            p, t, nrm = p[0], t[0], nrm[0]
            toward = -sigma * t
            for side_l in (1.0, -1.0):
                far = np.arange(0.0, 4.0 * reach + 4 * h_host, step)
                q_far = p - side_l * far[:, None] * nrm[None]
                i_f, x_f, j_f = first_crossing(q_far, seg)
                if i_f < 0:
                    continue
                for m in L_LEGS:
                    ell = m * base
                    tt = np.arange(0.0, ell + 1e-9, step)
                    q_t = p + side_l * tt[:, None] * nrm[None]
                    if first_crossing(q_t, seg)[0] >= 0:
                        continue                          # the coast is nearer than the leg: a plain transect
                    X = q_t[-1]
                    uu = np.arange(step, d + float(prof.W[prof.i0]) + 2 * h_host, step)
                    q_l = X + uu[:, None] * toward[None]
                    i_l, x_l, j_l = first_crossing(np.vstack([X[None], q_l]), seg)
                    if i_l < 4:                           # no coast, or the leg shorter than a host cell (a
                        continue                          # transect to the cape's tip)
                    pts = np.vstack([x_f, q_far[:i_f + 1][::-1], q_t[1:], q_l[:i_l], x_l])
                    if side_l < 0:                        # chains run from the right bank to the left bank
                        pts = pts[::-1]
                    jr, jl_ = (j_f, j_l) if side_l > 0 else (j_l, j_f)
                    qr, ql = pts[0], pts[-1]
                    nR = int(coast_seg_nodes[jr][np.argmin(np.linalg.norm(ch.xy[coast_seg_nodes[jr]] - qr, axis=1))])
                    nL = int(coast_seg_nodes[jl_][np.argmin(np.linalg.norm(ch.xy[coast_seg_nodes[jl_]] - ql, axis=1))])
                    if nR == nL:
                        continue
                    chain = ch.path(nR, nL, Polyline(pts))
                    if chain is None:
                        continue
                    key = tuple(chain.tolist())
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(evaluate(side, sigma, chain, "L", s_k, prof, ch, axis, sphere_len(chain), bank_fn,
                                        bank_dir, bay_hit, leg=(side_l, ell), rule=rule, square=square))
        d += STEP_H * h_host
    out.sort(key=lambda c: c.d_m)
    return out


def passes_at(c: Join, prof: Profile, n: int) -> bool:
    """B184: whether the candidate passes every rule with n cells across the neck (the corner and bay rules do not
    depend on n; the room and the funnel are re-checked on the size target of n cells)."""
    if any(f not in ("room", "funnel") for f in c.fails) or c.sec is None:
        return False
    k, sec, Wsec, asec = c.sec
    rule = c.rule
    ht = side_target(prof, n, rule)
    if RATIO * ht[k] < c.h_join_m:
        return False
    hh = np.minimum(ht[sec], c.h_join_m)
    Ns = funnel_counts(Wsec / hh, n, rule)
    r = asec / (np.maximum(Ns, 1e-9) * (rule.ln_funnel if rule is not None else np.log(FUNNEL_RATIO)))
    r = np.where(np.isfinite(r), r, 0.0)
    return not (len(r) and float(r.max()) > 1.0)


def advise(cands: dict, prof: Profile, reach: float | None = None) -> str:
    """B184: the cell counts across the neck (2 .. the floor's limit or 4 x the request, fine enough for the no-zoom
    rule: RATIO x neck / n <= the host size) for which each side has a passing candidate, and the counts both sides
    accept; the advice of a refusal (a per-side fix can ask for fewer cells on one side and more on the other)."""
    n_hi = min(prof.n_max, 4 * prof.n_cells)
    W0 = float(prof.W[prof.i0])
    zoom = [n for n in range(2, n_hi + 1) if not prof.h_host or RATIO * W0 / n <= prof.h_host]   # the no-zoom rule
    ok = {sd: [n for n in zoom if any(passes_at(c, prof, n) for c in cs)] for sd, cs in cands.items()}
    both = sorted(set.intersection(*[set(v) for v in ok.values()])) if ok else []

    def rng(v):
        if not v:
            return "none"
        runs, a = [], v[0]
        for x, y in zip(v, v[1:] + [None]):
            if y != x + 1:
                runs.append(f"{a}" if a == x else f"{a}-{x}")
                a = y
        return ", ".join(runs)
    per = "; ".join(f"side {sd} passes with cells {rng(v)}" for sd, v in ok.items())
    far = ""
    if reach is not None:                              # the room rule passes farther out: a larger region
        roomy = [sd for sd, cs in cands.items() if cs and "room" in max(cs, key=lambda c: c.d_m).fails]
        if roomy:
            far = (f"; the farthest candidates of side {', '.join(roomy)} still fail the room rule: a larger "
                   f"reach_km (now {reach / 1e3:.0f})")
    if both:
        return f"cell counts across the neck that pass both sides: {rng(both)} ({per}){far}"
    return (f"no cell count across the neck from 2 to {n_hi} passes both sides ({per}){far}; or join hints, a finer "
            "host near the joins or a smaller h_min_m")
