"""時間領域固定整相に関する回帰試験。"""

# ここでは、設計書の幾何式から求めた遅延表と、整数サンプル遅延による
# 固定整相の因果実装が噛み合うことを固定条件で確認する。

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from spflow import (
    DelayTable,
    FractionalDelayFilterBank,
    IntegerDelayAndSumBeamformer,
    design_fractional_delay_filter_bank,
)


def test_delay_table_from_geometry_matches_expected_integer_delays_and_fractional_indices():
    """遅延表設計について 既知幾何で整数遅延と小数遅延 index が期待通りになることを確認する。"""
    array_pos_m = np.array(
        [
            [-0.05, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
        ]
    )
    dir_cos = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    # fs/c = 8000/400 = 20 より、x 方向 5 cm の位置差はちょうど 1 sample に対応する。
    # この条件を使うことで、整数遅延分解が丸め誤差ではなく幾何式そのものを検証できる。
    fractional_filter_bank = design_fractional_delay_filter_bank(n_frac_filter=5, n_tap=7)
    delay_table = DelayTable.from_geometry(
        array_pos_m=array_pos_m,
        dir_cos=dir_cos,
        fs_hz=8000.0,
        sound_speed_m_s=400.0,
        fractional_filter_bank=fractional_filter_bank,
    )

    np.testing.assert_allclose(
        delay_table.arrival_delay_sec,
        np.array(
            [
                [0.000125, 0.0],
                [0.0, 0.0],
                [-0.000125, 0.0],
            ]
        ),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        delay_table.steering_delay_sample,
        np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ]
        ),
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        delay_table.delay_int,
        np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
            ]
        ),
    )
    np.testing.assert_allclose(delay_table.delay_frac, 0.0, atol=1e-12)
    np.testing.assert_array_equal(delay_table.frac_filter_index, np.full((3, 2), 2, dtype=np.int64))


def test_fractional_delay_filter_bank_save_and_load_round_trips():
    """小数遅延 FIR バンクについて 保存後に読み直しても grid と係数が変わらないことを確認する。"""
    filter_bank = design_fractional_delay_filter_bank(n_frac_filter=9, n_tap=11)

    # 保存形式そのものを検証したいので、一時ディレクトリではなくワークスペース内の
    # 単一 `.npz` を使い、保存先の権限制約に依存しないようにする。
    save_path = Path.cwd() / "artifacts" / f"fractional_delay_filter_bank_{os.getpid()}.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        filter_bank.save_npz(save_path)
        loaded = FractionalDelayFilterBank.load_npz(save_path)
    finally:
        if save_path.exists():
            save_path.unlink()

    np.testing.assert_allclose(loaded.frac_grid, filter_bank.frac_grid, atol=0.0)
    np.testing.assert_allclose(loaded.frac_filters, filter_bank.frac_filters, atol=0.0)


def test_integer_delay_and_sum_beamformer_aligns_impulses_to_common_beam_peak():
    """整数遅延固定整相について 先行到達を模したインパルス群を同一時刻へ整列できることを確認する。"""
    delay_table = DelayTable(
        arrival_delay_sec=np.zeros((3, 1), dtype=np.float64),
        steering_delay_sample=np.array([[0.0], [1.0], [2.0]], dtype=np.float64),
        delay_int=np.array([[0], [1], [2]], dtype=np.int64),
        delay_frac=np.zeros((3, 1), dtype=np.float64),
    )
    beamformer = IntegerDelayAndSumBeamformer(delay_table=delay_table, average_channels=True)

    x = np.zeros((3, 8), dtype=np.float32)
    # 整相前は各チャネルの到来時刻を 2 sample ずつ前倒ししておき、
    # delay_int を適用した後に全チャネルのピークが sample 4 へ揃うようにする。
    x[0, 4] = 1.0
    x[1, 3] = 1.0
    x[2, 2] = 1.0

    beam_output, steered_channel_output = beamformer.process(x, return_steered_channels=True)

    expected_steered = np.zeros((1, 3, 8), dtype=np.float32)
    expected_steered[0, :, 4] = 1.0
    expected_beam = np.zeros((1, 8), dtype=np.float32)
    expected_beam[0, 4] = 1.0

    np.testing.assert_allclose(steered_channel_output, expected_steered, atol=1e-6)
    np.testing.assert_allclose(beam_output, expected_beam, atol=1e-6)
