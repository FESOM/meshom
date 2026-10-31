# SPDX-License-Identifier: GPL-3.0-or-later
"""meshom.orca: NEMO's ORCA grids by NEMO's own generator (`generator`: G. Madec's tripolar code with A. Sellar's
tropical stretch, rewritten), the family as presets (`presets`: ORCA2, ORCA1, ORCA05, ORCA025, ORCA12, ORCA36), the
grid as a meshom mesh with NEMO's land-sea mask (`mesh`)."""
from meshom.orca.generator import GenParams, OrcaGrid, Stretch, grid  # noqa: F401
from meshom.orca.presets import PRESETS, Preset, build, extra_rows_for  # noqa: F401
