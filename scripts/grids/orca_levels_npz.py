"""Convert NEMO's bottom levels / model bathymetry of the ORCA presets once to .npz for meshom.orca.fesom3 (the
water and the levels of FESOM3's rules on every preset, decisions B91-B92).

    python scripts/grids/orca_levels_npz.py OUT_DIR NAME=PATH ...

NAME is the source name meshom.orca.fesom3.SOURCES uses:
- orca2=<...>/ORCA_R2_zps_domcfg.nc (NEMO 5.0 ORCA2 domain_cfg): kb = bottom_level, z = the layer bottoms from e3t_1d;
- eorca05=<...>/eORCA05_mesh_mask.nc4 (MEOM eORCA05.L121 mesh_mask): kb = mbathy, z = gdepw_1d plus the last bottom
  (2 gdept_1d - gdepw_1d of the last level);
- eorca1=<...>/bathy_eORCA1_...nc, eorca025=<...>/bathy_eORCA025_...nc, eorca12=<...>/bathy_eORCA12_...nc (Met Office
  GO6 model bathymetry from GEBCO 2021 with its hand edits, Zenodo 22010172, 15494369, 15495870): depth_m =
  Bathymetry (positive down, 0 land); the T points from the GO6 grid_mask file of the same grid, given as
  `eorca12_grid=<...>/grid_mask_eORCA12-GO6.nc` when the bathymetry file has no nav_lon / nav_lat;
- eorca36=<...>/eORCA36.L121_domain_cfg_v7_4.2.nc (MEOM / IGE eORCA36.L121, NEMO 4.2 without halos,
  ige-meom-opendap.univ-grenoble-alpes.fr thredds meomopendap/extract/MEOM/eORCA36.L121/eORCA36.L121-I): depth_m =
  bathy_metry where bottom_level > 0 and top_level = 1, else 0 (an ice-shelf cavity is land: FESOM3 has none).
Every npz: glamt, gphit (float64, the file's rows x columns), and kb (int16) + z (float64, Nz + 1, 0 first) or
depth_m (float32).
"""
import sys
from pathlib import Path

import netCDF4
import numpy as np


def _v(d, name):
    v = d.variables[name]
    v.set_auto_maskandscale(False)
    return np.asarray(v[:]).squeeze()


def main(out: Path, src: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, path in src.items():
        if name.endswith("_grid"):
            continue
        arr = {}
        with netCDF4.Dataset(path) as d:
            if name == "orca2":
                arr["kb"] = _v(d, "bottom_level").astype(np.int16)
                arr["z"] = np.r_[0.0, np.cumsum(_v(d, "e3t_1d").astype(np.float64))]
                arr["glamt"], arr["gphit"] = _v(d, "glamt").astype(float), _v(d, "gphit").astype(float)
            elif name == "eorca36":
                wet = (_v(d, "bottom_level") > 0) & (_v(d, "top_level") == 1)
                arr["depth_m"] = np.where(wet, _v(d, "bathy_metry"), 0.0).astype(np.float32)
                arr["glamt"], arr["gphit"] = _v(d, "glamt").astype(float), _v(d, "gphit").astype(float)
            elif name == "eorca05":
                arr["kb"] = _v(d, "mbathy").astype(np.int16)
                gw = _v(d, "gdepw_1d").astype(np.float64)
                gt = _v(d, "gdept_1d").astype(np.float64)
                arr["z"] = np.r_[gw, 2.0 * gt[-1] - gw[-1]]              # the last bottom: T mid-layer
                arr["glamt"], arr["gphit"] = _v(d, "glamt").astype(float), _v(d, "gphit").astype(float)
            else:
                arr["depth_m"] = _v(d, "Bathymetry").astype(np.float32)
                if "nav_lon" in d.variables:
                    arr["glamt"], arr["gphit"] = _v(d, "nav_lon").astype(float), _v(d, "nav_lat").astype(float)
        if "glamt" not in arr:
            with netCDF4.Dataset(src[f"{name}_grid"]) as g:
                arr["glamt"], arr["gphit"] = _v(g, "glamt").astype(float), _v(g, "gphit").astype(float)
        np.savez(out / f"{name}.npz", **arr)
        print(name, {k: (v.dtype.name, v.shape) for k, v in arr.items()}, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), dict(a.split("=", 1) for a in sys.argv[2:]))
