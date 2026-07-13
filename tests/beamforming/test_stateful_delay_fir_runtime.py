"""正式 T2a 評価用の整数遅延・残差 FIR 段の回帰試験。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from evaluations.beamforming.stateful_delay_fir_runtime import (
    ResidualCausalFIRStage,
    StatefulIntegerDelayStage,
)


def test_integer_delay_and_residual_fir_match_one_shot_when_split_into_blocks() -> None:
    """整数 buffer と残差 FIR の直列処理が block 分割に依存しないことを確認する。"""
    rng = np.random.default_rng(20260713)
    signal = rng.standard_normal((3, 53))
    delays = np.array([0, 3, 11], dtype=np.int64)
    taps = np.array(
        [[1.0, 0.25, -0.1, 0.05], [0.5, -0.2, 0.1, 0.0], [0.75, 0.1, 0.0, -0.05]],
        dtype=np.complex128,
    )

    one_delay = StatefulIntegerDelayStage(delays)
    one_fir = ResidualCausalFIRStage(taps)
    one_delayed = one_delay.process(signal)
    one = one_fir.process(one_delayed.data, one_delayed.valid_mask)

    split_delay = StatefulIntegerDelayStage(delays)
    split_fir = ResidualCausalFIRStage(taps)
    split_data: list[NDArray[Any]] = []
    split_valid: list[NDArray[Any]] = []
    for start, stop in ((0, 5), (5, 17), (17, 18), (18, 41), (41, 53)):
        delayed = split_delay.process(signal[:, start:stop])
        filtered = split_fir.process(delayed.data, delayed.valid_mask)
        split_data.append(filtered.data)
        split_valid.append(filtered.valid_mask)

    np.testing.assert_allclose(np.concatenate(split_data, axis=1), one.data, atol=0.0)
    np.testing.assert_array_equal(np.concatenate(split_valid, axis=1), one.valid_mask)


def test_first_valid_sample_includes_integer_delay_and_fir_history() -> None:
    """初回有効境界が系列ごとに `delay + n_tap - 1` になることを確認する。"""
    n_sample = 24
    delays = np.array([0, 2, 7], dtype=np.int64)
    taps = np.ones((3, 5), dtype=np.complex128)
    signal = np.ones((3, n_sample), dtype=np.float64)

    delayed = StatefulIntegerDelayStage(delays).process(signal)
    filtered = ResidualCausalFIRStage(taps).process(delayed.data, delayed.valid_mask)

    for series_index, delay_sample_value in enumerate(delays.tolist()):
        first_valid = int(delay_sample_value) + taps.shape[1] - 1
        assert not bool(np.any(filtered.valid_mask[series_index, :first_valid]))
        assert bool(np.all(filtered.valid_mask[series_index, first_valid:]))


def test_pending_coefficients_are_latched_atomically_at_next_block_start() -> None:
    """更新係数が次block全体へ適用され、旧新係数がblock内で混在しないことを確認する。"""
    old_taps = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.complex128)
    new_taps = np.array([[0.0, 3.0], [0.0, 4.0]], dtype=np.complex128)
    stage = ResidualCausalFIRStage(old_taps, active_version=5)

    first = stage.process(np.array([[1.0, 2.0], [10.0, 20.0]], dtype=np.float64))
    np.testing.assert_allclose(first.data, np.array([[1.0, 2.0], [20.0, 40.0]]))
    assert stage.active_version == 5

    stage.request_coefficient_update(new_taps, version=6)
    # request時点では旧係数がactiveのままで、次のprocess先頭が唯一の公開切替境界となる。
    assert stage.active_version == 5
    second = stage.process(np.array([[5.0, 6.0, 7.0], [50.0, 60.0, 70.0]], dtype=np.float64))

    assert stage.active_version == 6
    # 新FIRは一sample前だけを見る。先頭出力が前block末尾を参照することで履歴保持も同時に固定する。
    np.testing.assert_allclose(second.data, np.array([[6.0, 15.0, 18.0], [80.0, 200.0, 240.0]]))
