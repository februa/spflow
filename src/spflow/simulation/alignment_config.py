"""整相シミュレーション条件の保持と境界検証。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spflow.simulation.numerics import SimulationPrecision

FloatArray = NDArray[np.floating[Any]]


@dataclass(frozen=True)
class AlignmentSimulationConfig:
    """整相重み設計に必要な物理条件と離散化条件を保持する。

    センサ位置はULA軸上の座標 ``[n_ch]``、方位は ``[n_beam]``、単位はそれぞれ
    mとdegである。このクラスは条件の検証だけを担い、遅延、共分散、重み、FIR、
    評価量を計算しない。
    """

    fs_hz: float
    fft_size: int
    sound_speed_m_per_s: float
    sensor_positions_m: FloatArray
    beam_azimuth_deg: FloatArray
    target_azimuth_deg: float
    target_band_hz: tuple[float, float]
    analysis_width_hz: float
    source_band_rms_power: float
    noise_power_per_bin_re_input_rms2: float
    ebae_diagonal_loading: float = 1.0
    mvdr_diagonal_loading_ratio: float = 1.0e-3
    precision: SimulationPrecision = SimulationPrecision.DOUBLE

    def __post_init__(self) -> None:
        """shape、単位上の範囲、DFT鏡映条件を構築時に検証する。

        Raises:
            ValueError: 精度、配列shape、有限性、帯域、power、loadingが不正な場合。

        Notes:
            呼出側配列の変更で条件が変化しないよう、位置と方位は読取専用copyにする。
        """
        if not isinstance(self.precision, SimulationPrecision):
            raise ValueError("precision must be a SimulationPrecision value.")
        positions = np.asarray(self.sensor_positions_m, dtype=self.precision.real_dtype)
        azimuths = np.asarray(self.beam_azimuth_deg, dtype=self.precision.real_dtype)
        if positions.ndim != 1 or positions.size == 0 or not bool(np.all(np.isfinite(positions))):
            raise ValueError("sensor_positions_m must be a finite non-empty 1-D array.")
        if azimuths.ndim != 1 or azimuths.size == 0 or not bool(np.all(np.isfinite(azimuths))):
            raise ValueError("beam_azimuth_deg must be a finite non-empty 1-D array.")
        if self.fs_hz <= 0.0 or self.sound_speed_m_per_s <= 0.0:
            raise ValueError("fs_hz and sound_speed_m_per_s must be positive.")
        if self.fft_size <= 0 or self.fft_size % 2 != 0:
            raise ValueError("fft_size must be a positive even integer.")
        band_low_hz, band_high_hz = self.target_band_hz
        if not 0.0 <= band_low_hz <= band_high_hz <= self.fs_hz / 2.0:
            raise ValueError("target_band_hz must be ordered inside [0, fs_hz / 2].")
        if self.analysis_width_hz < 0.0 or self.source_band_rms_power < 0.0:
            raise ValueError("analysis width and source power must be non-negative.")
        if self.noise_power_per_bin_re_input_rms2 <= 0.0:
            raise ValueError("noise power must be positive.")
        if self.ebae_diagonal_loading < 0.0 or self.mvdr_diagonal_loading_ratio < 0.0:
            raise ValueError("diagonal loading values must be non-negative.")
        positions = positions.copy()
        azimuths = azimuths.copy()
        positions.setflags(write=False)
        azimuths.setflags(write=False)
        object.__setattr__(self, "sensor_positions_m", positions)
        object.__setattr__(self, "beam_azimuth_deg", azimuths)
