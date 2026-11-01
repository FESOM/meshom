"""Curvilinear inserts fitted into a finished mesh: a strait as one structured block (decisions B158-B163) with banks
on the real coast, a size floor, bays as blocks of their own and straighter joins (B164-B169); L-shaped joins round a
cape, bays at a join, a refusal advice over both sides, coastal strips and current corridors (B180-B185); bays by
size, a cleaned collar round the joins on irregular hosts and ORCA hosts kept as they are (B190-B194).

    python -m meshom.insert --run RUN_DIR --insert 'strait=Gibraltar; cells=10' [--name N] [--out DIR]
    python -m meshom.insert --run RUN_DIR --config FILE.toml        (every [[inserts]] entry of FILE)
    python -m meshom.insert --run RUN_DIR --insert 'kind="strip"; path=[[lon, lat], ...]; width_km=60; h_fine_m=2000'
    python -m meshom.insert --run RUN_DIR --insert 'kind="corridor"; path=[[lon, lat], ...]; width_km=200; h_fine_m=5000'
    python -m meshom.insert --run RUN_DIR --insert 'kind="depth"; strait="Lancaster_Barrow"; width_km=160'   (B223)

In a region of interest a logically rectangular quad grid that zooms in (a "banana" grid) replaces the host's
cells; on the two joins its nodes are the host's nodes, fixed, so the result conforms without hanging nodes. Any host
works (ORCA, meshom's quad-dominant meshes, triangles). Steps (meshom.insert.core):

1. region (meshom.insert.region): the strait's width profile along its axis, the neck, the size target; candidate
   joins grow outward from the neck until the three rules hold (corner, funnel, room), or the region is refused
   with the failing rule and what would fix it;
2. joins: ordered chains of host edges (tier 1 along host grid lines, tier 2 otherwise);
3. block (meshom.insert.block): Winslow harmonic map in a local stereographic chart, joins fixed, banks sliding along
   the real coast (meshom.insert.realcoast: the run's coast smoothed at the insert's cell size, bays bridged; step
   1's capped host coast with coast = "host"); rows along the strait at equal steps of the integral of ds / h, never
   below the size floor h_min_m; Y-splits (staggered) when the counts change; bays as blocks of their own
   (meshom.insert.network);
4. merge by shared node ids (meshom.insert.assemble); the collar cleaned by the mesh-fixing pipeline
   (meshom.meshfix.pipeline.run_stages): the host cells within `collar_rings` element rings of the joins and the
   block's edge rows, the rest of the block's core fixed (cut out of the case: its outline is artificial boundary),
   against the run's h composed with the insert's h layer; on a structured host (ORCA) the host's cells stay and
   only the block's edge rows are cleaned, the joins on straight grid lines where one passes;
5. scores by zone (core, join ring, outside; meshom.insert.measure), the splice into the whole mesh, depth by the
   run's bathymetry rules and model files when the run wrote them.

Band inserts (meshom.insert.band, meshom.insert.bandblock): a coastal strip (the real coast as the spine of a band of
fine rows, its sea side and both ends joined to the host along one host edge chain) and a current corridor (an
open-water band along an axis, closed by a loop of host edges, two half-blocks sharing the axis); rows offset from the
spine, wrapping round its ends, grown to the host size at the 15 % rule.

Later: inserts reserved inside the builder.
"""
