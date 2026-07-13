"""beamforming評価支援部品のshape・軸・level規約を検証する。"""

from __future__ import annotations

import numpy as np
import pytest

from spflow.beamforming.time_delay import (
    FractionalDelayAndSumBeamformer,
    design_fractional_delay_filter_bank,
)
from spflow.beamforming_evaluation import (
    build_beam_scan_grid,
    calculate_block_rms_levels_db20,
    calculate_fractional_beam_response_matrix,
    calculate_one_sided_rms_spectrum_db20,
    calculate_real_tone_response_rms_level_db20,
    calculate_rms_level_db20,
    calculate_tone_projection_rms_level_db20,
)


def test_rms_level_metrics_preserve_explicit_reference_and_two_sideband_power() -> None:
    """RMS levelが基準振幅と実toneの正負周波数powerを正しく扱うことを確認する。"""

    # RMS=2.0をreference_rms=2.0で測ると0 dB re 2.0 RMSになる。
    signal = np.array([2.0, -2.0], dtype=np.float64)
    assert calculate_rms_level_db20(signal, reference_rms=2.0) == pytest.approx(0.0)

    symmetric_level = calculate_real_tone_response_rms_level_db20(
        np.array([1.0 + 0.0j], dtype=np.complex128),
        np.array([1.0 + 0.0j], dtype=np.complex128),
        1.0,
    )
    positive_only_level = calculate_real_tone_response_rms_level_db20(
        np.array([1.0 + 0.0j], dtype=np.complex128),
        np.array([0.0 + 0.0j], dtype=np.complex128),
        1.0,
    )

    np.testing.assert_allclose(symmetric_level, np.array([0.0]), atol=1.0e-12)
    # 一方の側帯だけに応答する場合、powerは対称応答の1/2なので-3.0103 dBになる。
    np.testing.assert_allclose(positive_only_level, np.array([-3.010299956639812]), atol=1.0e-12)


def test_beam_scan_grid_fixes_shape_and_equal_cos_direction_contract() -> None:
    """走査gridがbeam軸と方向余弦軸を固定shapeで返すことを確認する。"""

    grid = build_beam_scan_grid(
        azimuth_min_deg=20.0,
        azimuth_max_deg=160.0,
        display_elevation_deg=0.0,
        n_real_azimuth_beams=11,
    )

    assert grid.directions.shape == (11, 3)
    assert grid.azimuth_deg.shape == (11,)
    assert grid.elevation_deg.shape == (1,)
    assert grid.display_elevation_index == 0
    # directionsのaxis=1はx/y/z方向余弦なので、各beamのノルムは1になる。
    np.testing.assert_allclose(np.linalg.norm(grid.directions, axis=1), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(grid.azimuth_deg[[0, -1]], np.array([20.0, 160.0]), atol=1.0e-12)


def test_signal_level_helpers_keep_per_bin_and_block_axes_explicit() -> None:
    """整数bin toneで射影・per-bin spectrum・block RMSのlevel規約を確認する。"""

    fs_hz = 8.0
    time_axis_s = np.arange(8, dtype=np.float64) / fs_hz
    signal = np.sqrt(2.0) * np.cos(2.0 * np.pi * 1.0 * time_axis_s)

    tone_level = calculate_tone_projection_rms_level_db20(signal, 1.0, fs_hz)
    frequency_hz, spectrum_level = calculate_one_sided_rms_spectrum_db20(
        signal[np.newaxis, :],
        fs_hz,
    )
    block_level, block_time_s = calculate_block_rms_levels_db20(
        signal[np.newaxis, :],
        fs_hz,
        block_size=8,
    )

    assert tone_level == pytest.approx(0.0, abs=1.0e-12)
    assert frequency_hz[1] == pytest.approx(1.0)
    assert spectrum_level.shape == (1, 5)
    assert spectrum_level[0, 1] == pytest.approx(0.0, abs=1.0e-12)
    assert block_level.shape == (1, 1)
    assert block_level[0, 0] == pytest.approx(0.0, abs=1.0e-12)
    np.testing.assert_array_equal(block_time_s, np.array([0.0]))


def test_fractional_response_matrix_uses_observation_and_look_beam_axes() -> None:
    """一つの同位置channelでは全beam間応答が1になることを確認する。"""

    positions_m = np.zeros((1, 3), dtype=np.float64)
    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    filter_bank = design_fractional_delay_filter_bank(n_frac_filter=5, n_tap=7)
    beamformer = FractionalDelayAndSumBeamformer.from_geometry(
        array_pos_m=positions_m,
        dir_cos=directions,
        fs_hz=8000.0,
        sound_speed_m_s=1500.0,
        fractional_filter_bank=filter_bank,
    )

    response = calculate_fractional_beam_response_matrix(beamformer, frequency_hz=1000.0)

    # axis=0がobservation beam、axis=1がlook beam。
    assert response.shape == (2, 2)
    # 因果小数遅延FIRの群遅延は全beamへ共通位相として現れる。
    # 同位置1 channelでは方向差がないため、絶対位相ではなく全要素の一致と振幅1を確認する。
    np.testing.assert_allclose(response, np.full((2, 2), response[0, 0]), atol=1.0e-12)
    np.testing.assert_allclose(np.abs(response), 1.0, atol=1.0e-7)
