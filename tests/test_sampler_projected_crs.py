"""Regression test: the sampler must convert pixel centres from the RASTER's CRS.

Found on real data: GFM rasters sit on projected Equi7 grids (metres), and the
sampler treated their x/y as lon/lat whenever the AOI was EPSG:4326. This is a
coordinate round-trip check of the conversion maths (a real Nyando point
projected to UTM 36N and back), not a stand-in for real flood data.
"""

import numpy as np
import pyproj
import pytest
import xarray as xr
import rioxarray  # noqa: F401  (registers the .rio accessor)

from src.data.case_control_sampler import _pixel_centers_to_lonlat

REAL_LON, REAL_LAT = 34.95, -0.18   # inside the Nyando ward bounds


def _projected_mask(epsg):
    to_proj = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x0, y0 = to_proj.transform(REAL_LON, REAL_LAT)
    xs = np.array([x0 - 20.0, x0, x0 + 20.0])
    ys = np.array([y0 + 20.0, y0, y0 - 20.0])
    mask = xr.DataArray(np.zeros((3, 3), dtype=bool), coords={"y": ys, "x": xs}, dims=("y", "x"))
    mask[1, 1] = True   # the centre pixel is exactly the real point
    return mask.rio.write_crs(f"EPSG:{epsg}")


def test_projected_raster_is_converted_from_its_own_crs():
    lons, lats = _pixel_centers_to_lonlat(_projected_mask(32636), "EPSG:4326")
    assert lons[0] == pytest.approx(REAL_LON, abs=1e-6)
    assert lats[0] == pytest.approx(REAL_LAT, abs=1e-6)


def test_mask_without_crs_is_rejected_not_guessed():
    mask = xr.DataArray(np.ones((2, 2), dtype=bool),
                        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]}, dims=("y", "x"))
    with pytest.raises(ValueError):
        _pixel_centers_to_lonlat(mask, "EPSG:4326")
