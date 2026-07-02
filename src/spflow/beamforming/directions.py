"""spflow.beamforming.directions を実装するモジュール。"""

from __future__ import annotations

import numpy as np

_DEFAULT_ELEVATION_PRESET_DEG = np.sort(np.array([18.1, 10.6, 6.0, -30.0], dtype=np.float32))


def _normalize_array_side(array_side: str) -> str:
    key = array_side.strip().lower()
    if key in {"right", "right side", "starboard"}:
        return "right"
    if key in {"left", "left side", "port"}:
        return "left"
    if key == "forward":
        return "forward"
    raise ValueError("array_side must be 'right side', 'left side', or 'forward'.")


def make_directions(
    az_min_deg: float,
    az_max_deg: float,
    el_min_deg: float,
    el_max_deg: float,
    n_beam_az_real: int,
    n_beam_az_virtual: int,
    n_beam_el: int | None = None,
    array_side: str = "right side",
    el_preset_deg: np.ndarray | list[float] | tuple[float, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct 3D beam direction cosines and plotting axes.

    Returns
    -------
    dir3d:
        Direction cosines with shape ``(3, n_beam_az * n_beam_el)``.
    axis_az_deg:
        Horizontal beam axis values with shape ``(n_beam_az,)``.
    axis_el_deg:
        Elevation beam axis values with shape ``(n_beam_el,)``.
    """

    side = _normalize_array_side(array_side)
    az_min = float(az_min_deg)
    az_max = float(az_max_deg)

    single_beam_exact_direction = side != "forward" and n_beam_az_real == 1 and n_beam_az_virtual == 0 and az_min == az_max

    if side != "forward" and not single_beam_exact_direction:
        if az_min < 0.0 or az_max > 180.0:
            raise ValueError(
                "For 'right side' and 'left side', azimuth must satisfy 0 <= az_min_deg <= az_max_deg <= 180."
            )
        if az_min > az_max:
            raise ValueError(
                "For 'right side' and 'left side', azimuth must satisfy 0 <= az_min_deg <= az_max_deg <= 180."
            )
        if az_min == az_max and n_beam_az_real != 1:
            raise ValueError(
                "For 'right side' and 'left side', az_min_deg == az_max_deg is only valid when n_beam_az_real == 1."
            )

    if n_beam_az_real <= 0:
        raise ValueError("n_beam_az_real must be positive.")
    if n_beam_az_virtual < 0:
        raise ValueError("n_beam_az_virtual must be non-negative.")
    n_beam_az = n_beam_az_real + n_beam_az_virtual
    if n_beam_az <= 0:
        raise ValueError("The total number of horizontal beams must be positive.")

    el_preset = _DEFAULT_ELEVATION_PRESET_DEG if el_preset_deg is None else np.sort(np.asarray(el_preset_deg, dtype=np.float32))
    if el_preset.ndim != 1 or el_preset.size == 0:
        raise ValueError("el_preset_deg must be a non-empty 1-D array.")
    if np.any(el_preset < el_min_deg) or np.any(el_preset > el_max_deg):
        raise ValueError("el_preset_deg must lie within [el_min_deg, el_max_deg].")
    if n_beam_el is None:
        n_beam_el = int(el_preset.size)
    if n_beam_el != int(el_preset.size):
        raise ValueError("n_beam_el must match len(el_preset_deg).")

    diff_cos_az = abs(np.cos(np.deg2rad(az_max)) - np.cos(np.deg2rad(az_min)))

    if side == "forward":
        axis_az_deg = np.linspace(az_min, az_max, n_beam_az, dtype=np.float32)
        cos_az = np.cos(np.deg2rad(axis_az_deg))
        sin_az = np.sin(np.deg2rad(axis_az_deg))
    else:
        if single_beam_exact_direction:
            axis_az_deg = np.array([az_min], dtype=np.float32)
            cos_az = np.cos(np.deg2rad(axis_az_deg))
            sin_az = np.sin(np.deg2rad(axis_az_deg))
        else:
            if n_beam_az_real == 1:
                cos_az = np.array([np.cos(np.deg2rad(0.5 * (az_min + az_max)))], dtype=np.float32)
            else:
                step = diff_cos_az / (n_beam_az_real - 1)
                start = np.cos(np.deg2rad(az_min)) + step * n_beam_az_virtual / 2.0
                stop = np.cos(np.deg2rad(az_max)) - step * n_beam_az_virtual / 2.0
                cos_az = np.linspace(start, stop, n_beam_az, dtype=np.float32)
            cos_az = np.clip(cos_az, -1.0, 1.0)
            sin_az = np.sqrt(np.maximum(0.0, 1.0 - cos_az**2))
            axis_az_deg = np.rad2deg(np.arccos(cos_az))

    sin_el = np.sin(np.deg2rad(el_preset))
    cos_el = np.sqrt(np.maximum(0.0, 1.0 - sin_el**2))

    dircos_x = cos_az[:, np.newaxis] * cos_el[np.newaxis, :]
    dircos_y_base = np.real(sin_az)[:, np.newaxis] * cos_el[np.newaxis, :]
    if side in {"right", "forward"}:
        dircos_y = dircos_y_base
    else:
        dircos_y = -dircos_y_base
    dircos_z = np.ones((n_beam_az, 1), dtype=np.float32) * sin_el[np.newaxis, :]

    n_beam_all = n_beam_az * n_beam_el
    dir3d = np.zeros((3, n_beam_all), dtype=np.float32)
    dir3d[0, :] = dircos_x.reshape(-1)
    dir3d[1, :] = dircos_y.reshape(-1)
    dir3d[2, :] = dircos_z.reshape(-1)

    axis_el_deg = np.rad2deg(np.arcsin(np.clip(sin_el, -1.0, 1.0)))
    return dir3d, axis_az_deg, axis_el_deg

