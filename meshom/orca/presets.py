# SPDX-License-Identifier: GPL-3.0-or-later
"""The ORCA family as presets of NEMO's generator (meshom.orca.generator; decisions B78-B81). A preset's parameters come
from the code where the code has them; the integers otherwise are counted on the reference grid; continuous
parameters are fitted only where no source gives them (ORCA2's). Every preset was checked against its reference
(`scripts/grids/orca_presets_compare.py`); ORCA05's is not a NEMO system-team file but MEOM / DRAKKAR's eORCA05.L121.

Layout: "code" = the original generator's own grid (T at I = ji - 1/2, the two wrap columns, the stretch); "lattice" =
T at I = i_offset + i / q, J = J_pole - pole_row / q + j / q (the generator refined q times, the user guide's
"quadrature"; q = 1 with another column offset for ORCA2), `pole_row` the 0-based row of the pole line in NEMO's
standard grid. Rows south of a preset's own first row are added by moving its southern end (jpeqt or pole_row), which
is all the code's jpeqt does (the user guide, 4.1).

Reference grids (`ref`, converted by scripts/grids/orca_refs_npz.py; `ref_row_offset`: the reference's row of our row
0): ORCA2 is NEMO 5.0's ORCA_R2 domain_cfg (its halo column 0 and fold row 148 cut off, so `ref_col_offset` -1);
eORCA1 / eORCA025 / eORCA12 are the Met Office GO6 grids and eORCA05 MEOM's eORCA05.L121 (Grenoble), ORCA plus the
ice-shelf rows of Mathiot et al. (2017) in the south (those rows are not ORCA's: the mask there falls back to the depth
rule). eORCA36 is MEOM / IGE's eORCA36.L121 (Grenoble, NEMO 4.2 files without halos: no wrap column and no fold row,
so `ref_col_offset` -1 as ORCA2's: our column 0, near 73 E, has no reference point).
"""
from dataclasses import dataclass, field, replace

import numpy as np

from meshom.orca.generator import GenParams, OrcaGrid, Stretch, grid


@dataclass(frozen=True)
class Preset:
    name: str
    params: GenParams
    layout: str = "code"                # "code" | "lattice"
    q: int = 1
    i_offset: float = 0.5
    pole_row: int = 0                   # lattice: the pole line's row in NEMO's standard grid
    rows: int = 0                       # NEMO's standard row count (the fold row included)
    ref: str | None = None              # npz name of the reference grid (None: no reference)
    ref_row_offset: int = 0             # the reference's row of our row 0 (standard rows)
    ref_col_offset: int = 0
    ref_mask: str = "tmask"             # the reference's water flag (> 0)
    verified: bool = True
    sources: dict = field(default_factory=dict)


ORCA1_STRETCH = Stretch(ppres=1.0, ppresmin=1.0 / 3.0, jpexp=4, ppa=12.0, ppb=8.0, pphi_join=20.0)
# ORCA2: Sellar's form at ppres 2, Rmin 0.5 (counted: the equator's spacing), the rest fitted (B80)
ORCA2_STRETCH = Stretch(ppres=2.0, ppresmin=0.5, jpexp=2, ppa=8.52813796, ppb=11.1328968, pphi_join=20.0)
ORCA05_BASE = GenParams(jpi=721, jpeqt=250, jpeqn=41, jpnord=220)

PRESETS = {
    "ORCA2": Preset(
        "ORCA2", GenParams(jpi=181, jpeqt=100, jpeqn=10, jpnord=57, rlampo=260.0, rphi1=50.0, rphi2=110.0,
                           ra1=10 ** -2.13501212, rb2=0.95761178, pivot="T", stretch=ORCA2_STRETCH),
        layout="lattice", q=1, i_offset=0.0, pole_row=147, rows=149, ref="orca2", ref_row_offset=0,
        ref_col_offset=-1, ref_mask="bottom_level",
        sources={"jpi": "counted (180 columns)", "jpeqn": "counted (the last exact parallel 19.61 N)",
                 "jpnord": "counted (equator to pivot 74 rows = njc - 1 + jpnord - 1)",
                 "pivot": "user guide 5.2.4 (T-U fold) and the file (NFtype T)",
                 "rlampo, rphi1, rphi2": "counted (the pole line's ends 50 N 80 E, 70 N 100 W)",
                 "ra1, rb2": "fitted to ORCA2's north rows (B80)", "stretch": "fitted (B80)"}),
    "ORCA1": Preset(
        "ORCA1", GenParams(jpi=361, jpeqt=132, jpeqn=20, jpnord=110, stretch=ORCA1_STRETCH),
        rows=292, ref="eorca1", ref_row_offset=40,
        sources={"jpi, jpeqn, jpnord, poles": "the original generator's parameters and coefficients",
                 "jpeqt": "the original generator's commented value 132 (292 rows, NEMO's ORCA1)",
                 "stretch": "the original generator's tropical stretch (ORCA1's)", "l_trop_stretch": "on (eORCA1 is stretched)"}),
    "ORCA05": Preset(
        "ORCA05", ORCA05_BASE, rows=511, ref="eorca05", ref_row_offset=93,
        sources={"jpi": "720 columns", "jpeqn, jpnord": "counted on eORCA025 / eORCA12 (their base grid)",
                 "jpeqt": "511 rows (NEMO's ORCA05)", "stretch": "none (ORCA025 has none)",
                 "reference": "MEOM / DRAKKAR eORCA05.L121 mesh_mask (F pivot at row 602, equator T row 342)"}),
    "ORCA025": Preset(
        "ORCA025", ORCA05_BASE, layout="lattice", q=2, i_offset=0.5, pole_row=1019, rows=1021, ref="eorca025",
        ref_row_offset=186,
        sources={"q": "user guide 5.2.4: ORCA025 from ORCA05 by quadrature", "base": "ORCA05 (counted)",
                 "i_offset, pole_row": "counted on eORCA025 (T pivot)"}),
    "ORCA12": Preset(
        "ORCA12", ORCA05_BASE, layout="lattice", q=6, i_offset=5.0 / 6.0, pole_row=3057, rows=3059, ref="eorca12",
        ref_row_offset=547,
        sources={"q": "counted: eORCA12 is ORCA05's map on the 1/6 lattice (ORCA025 x 3)",
                 "i_offset, pole_row": "counted on eORCA12 (T pivot)"}),
    "ORCA36": Preset(
        "ORCA36", ORCA05_BASE, layout="lattice", q=18, i_offset=17.0 / 18.0, pole_row=9171, rows=9173, ref="eorca36",
        ref_row_offset=1670, ref_col_offset=-1,
        sources={"q": "counted: eORCA36 is ORCA05's map on the 1/18 lattice (ORCA12 x 3, ORCA025 x 9)",
                 "i_offset, pole_row": "counted on eORCA36 (T pivot; its file has no halo column, ORCA12's layout)",
                 "reference": "MEOM / IGE eORCA36.L121 coordinates, NEMO 4.2 (12960 x 10842, no halos)"}),
}


def _merc(lat):
    return np.degrees(np.arcsinh(np.tan(np.radians(lat))))


def extra_rows_for(pr: Preset, lat_south: float | None) -> int:
    """Rows to add in the south so that the first F row lies south of lat_south (0 when it already does)."""
    if lat_south is None:
        return 0
    p = pr.params
    step = p.rdx / pr.q
    if pr.layout == "code":
        y0 = p.rdx * (1.5 - p.jpeqt)                              # F row 0 at J = 1.5, Mercator (deg)
    else:
        from meshom.orca.generator import row_map, set_coefficients

        c = set_coefficients(p)
        rm = row_map(p, c)
        jf0 = (c.rjpj + rm.shift) - pr.pole_row / pr.q + 0.5 / pr.q
        y0 = p.rdx * (jf0 - p.jpeqt)
    return max(0, int(np.ceil((y0 - _merc(lat_south)) / step - 1e-9)))


def build(name: str, extra_south: int = 0, check: bool = True) -> OrcaGrid:
    """The preset's grid with `extra_south` more rows in the south (NEMO's standard rows when 0)."""
    pr = PRESETS[name]
    if pr.layout == "code":
        g = grid(replace(pr.params, jpeqt=pr.params.jpeqt + extra_south), check=check)
        g.info["i_offset"] = 0.5
    else:
        from meshom.orca.generator import row_map, set_coefficients

        c = set_coefficients(pr.params)
        rm = row_map(pr.params, c)
        joff = (c.rjpj + rm.shift) - (pr.pole_row + extra_south) / pr.q
        g = grid(pr.params, q=pr.q, i_offset=pr.i_offset, j_offset=joff, check=check)
    g.info.update(preset=name, extra_south=int(extra_south), verified=pr.verified)
    return g
