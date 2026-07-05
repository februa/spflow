"""小数遅延固定整相の回帰試験。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from spflow.beamforming.fractional_delay_performance import (
    FractionalDelayPerformanceConfig,
    run_fractional_delay_performance_report,
)
from spflow.beamforming.time_delay import (
    DelayTable,
    FractionalDelayAndSumBeamformer,
    IntegerDelayAndSumBeamformer,
    design_fractional_delay_filter_bank,
)


def _tone_peak_level_db20(signal: np.ndarray, frequency_hz: float, fs_hz: float) -> float:
    """単一トーンの peak レベルを dB20 で評価する。"""
    time_axis_s = np.arange(signal.shape[-1], dtype=np.float64) / float(fs_hz)
    reference = np.exp(-1j * 2.0 * np.pi * float(frequency_hz) * time_axis_s)
    coefficient = np.vdot(reference, np.asarray(signal, dtype=np.complex128)) / signal.shape[-1]
    peak_amplitude = 2.0 * np.abs(coefficient)
    return float(20.0 * np.log10(max(peak_amplitude, np.finfo(np.float64).tiny)))


def _build_reference_sparse_positions() -> np.ndarray:
    """69 ch 片舷スパースアレイ座標を返す。"""
    positive_indices = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 27, 31, 36, 42, 49, 57, 66, 76, 87, 99, 112, 126],
        dtype=np.float64,
    )
    sensor_index = np.concatenate([-positive_indices[::-1], [0.0], positive_indices])
    positions_m = np.zeros((sensor_index.size, 3), dtype=np.float64)
    positions_m[:, 0] = 0.05 * sensor_index
    return positions_m


def test_fractional_delay_and_sum_beamformer_recovers_half_sample_tone_alignment() -> None:
    """0.5 sample ずれた高域トーンについて 小数遅延固定整相の方が整数遅延より整相利得が高いことを確認する。"""
    fs_hz = 32768.0
    frequency_hz = 10000.0
    n_sample = 8192
    time_axis_s = np.arange(n_sample, dtype=np.float64) / fs_hz
    filter_bank = design_fractional_delay_filter_bank(n_frac_filter=65, n_tap=63)

    # 時間領域固定整相は早着チャネルを遅らせる因果系なので、
    # 0.5 sample 早着した ch0 に対して +0.5 sample の小数遅延を与える条件で比較する。
    delay_frac = np.array([[0.5], [0.0]], dtype=np.float64)
    delay_table_fractional = DelayTable(
        arrival_delay_sec=np.zeros((2, 1), dtype=np.float64),
        steering_delay_sample=delay_frac.copy(),
        delay_int=np.zeros((2, 1), dtype=np.int64),
        delay_frac=delay_frac,
        frac_filter_index=filter_bank.select_indices(delay_frac),
    )
    delay_table_integer = DelayTable(
        arrival_delay_sec=np.zeros((2, 1), dtype=np.float64),
        steering_delay_sample=np.zeros((2, 1), dtype=np.float64),
        delay_int=np.zeros((2, 1), dtype=np.int64),
        delay_frac=np.zeros((2, 1), dtype=np.float64),
    )

    integer_beamformer = IntegerDelayAndSumBeamformer(delay_table=delay_table_integer, average_channels=True)
    fractional_beamformer = FractionalDelayAndSumBeamformer(
        delay_table=delay_table_fractional,
        fractional_filter_bank=filter_bank,
        average_channels=True,
        fs_hz=fs_hz,
    )

    # ch0 を 0.5 sample 早着させることで、高域では整数遅延だけでは位相差が残る条件を作る。
    # 小数遅延器は ch0 にだけ +0.5 sample を与え、2 チャネルの相対位相を揃える。
    input_signal = np.stack(
        [
            np.cos(2.0 * np.pi * frequency_hz * (time_axis_s + 0.5 / fs_hz)),
            np.cos(2.0 * np.pi * frequency_hz * time_axis_s),
        ],
        axis=0,
    ).astype(np.float64)

    integer_output = integer_beamformer.process(input_signal)[0]
    fractional_output = fractional_beamformer.process(input_signal)[0]

    # FIR 立ち上がりの過渡を避けるため、先頭 256 sample を捨てた定常区間でレベルを比較する。
    integer_level_db20 = _tone_peak_level_db20(integer_output[256:], frequency_hz=frequency_hz, fs_hz=fs_hz)
    fractional_level_db20 = _tone_peak_level_db20(fractional_output[256:], frequency_hz=frequency_hz, fs_hz=fs_hz)
    assert fractional_level_db20 > integer_level_db20 + 0.8


def test_fractional_delay_performance_report_improves_sparse_array_high_frequency_margin() -> None:
    """保存済み FIR バンクを読み出した小数遅延固定整相が 10 kHz off-broadside で整数遅延より高い peak margin を持つことを確認する。"""
    output_dir = Path.cwd() / "artifacts" / "beamforming" / "fractional_delay_performance_test"
    save_path = Path.cwd() / "artifacts" / f"fractional_delay_filter_bank_perf_{os.getpid()}.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    filter_bank = design_fractional_delay_filter_bank(n_frac_filter=65, n_tap=63)
    filter_bank.save_npz(save_path)
    try:
        summary = run_fractional_delay_performance_report(
            FractionalDelayPerformanceConfig(
                output_dir=output_dir,
                array_positions_m=_build_reference_sparse_positions(),
                fractional_delay_filter_bank_path=save_path,
                fs_hz=32768.0,
                sound_speed_m_s=1500.0,
                frequency_grid_hz=(4096.0, 8192.0, 10000.0),
                evaluation_azimuths_deg=(60.0, 90.0, 120.0),
                n_beam_az_real=151,
                comparison_specs=((10000.0, 60.0),),
            )
        )
    finally:
        if save_path.exists():
            save_path.unlink()

    assert Path(str(summary["performance_summary_json_path"])).exists()
    assert Path(str(summary["performance_table_csv_path"])).exists()
    assert Path(str(summary["margin_summary_png_path"])).exists()
    for comparison_png_path in summary["comparison_png_paths"]:
        assert Path(str(comparison_png_path)).exists()

    integer_worst_margin_db = np.asarray(summary["integer_worst_margin_db"], dtype=np.float64)
    fractional_worst_margin_db = np.asarray(summary["fractional_worst_margin_db"], dtype=np.float64)
    assert np.all(fractional_worst_margin_db > integer_worst_margin_db)
    assert fractional_worst_margin_db[-1] >= 13.0
