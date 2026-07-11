"""方式3の中心サンプル表、重複snapshot抽出、方位一致積分を検証する。"""

import numpy as np

from spflow.beamforming import (
    DirectionMatchedCovarianceAccumulator,
    build_two_second_covariance_snapshot_schedule,
    calculate_maximum_spatial_correlation_table,
)


def test_schedule_assigns_one_overlapping_snapshot_to_each_beam() -> None:
    """1秒159方位に中心を1個ずつ置き、各中心から128 sampleを取得する。"""

    positions_m = np.array([[-15.0, 0.0, 0.0], [0.0, 0.0, 0.0], [15.0, 0.0, 0.0]], dtype=np.float32)
    schedule = build_two_second_covariance_snapshot_schedule(
        positions_m,
        fs_hz=32768.0,
        sound_speed_m_s=1500.0,
        snapshot_length_samples=128,
        beams_per_half=159,
    )

    assert schedule.channel_center_samples.shape == (2, 3, 159)
    assert schedule.direction_match_indices.shape == (2, 159)
    assert schedule.global_direction_azimuth_deg.shape == (317,)
    np.testing.assert_array_equal(schedule.direction_match_indices[0, [0, -1]], [0, 158])
    np.testing.assert_array_equal(schedule.direction_match_indices[1, [0, -1]], [316, 158])
    np.testing.assert_allclose(
        schedule.global_direction_azimuth_deg[schedule.direction_match_indices],
        schedule.beam_azimuth_deg,
        atol=1.0e-5,
    )
    np.testing.assert_array_equal(schedule.channel_center_samples[1], np.flip(schedule.channel_center_samples[0], axis=0))
    assert int(np.min(schedule.channel_center_samples - 64)) >= 0
    assert int(np.max(schedule.channel_center_samples + 64)) <= 32768

    # 中心間隔から非重複block数を逆算せず、159方位の中心をそのまま保持する。
    # 条件変更でsnapshot区間が重なっても抽出を拒否しない。
    center_spacing = np.diff(schedule.channel_center_samples[0], axis=1)
    assert bool(np.all(center_spacing > 0))
    assert center_spacing.shape == (3, 158)


def test_extract_snapshots_returns_n_ch_by_128_by_159_and_reuses_table() -> None:
    """1秒信号から`[n_ch,128,159]`を抽出し、中心表を再生成しない。"""

    positions_m = np.zeros((1, 3), dtype=np.float32)
    schedule = build_two_second_covariance_snapshot_schedule(
        positions_m,
        fs_hz=32768.0,
        sound_speed_m_s=1500.0,
        snapshot_length_samples=128,
        beams_per_half=159,
    )
    center_table_identity = id(schedule.channel_center_samples)
    signal = np.arange(32768, dtype=np.int32)[np.newaxis, :]

    snapshots = schedule.extract_snapshots(signal, azimuth_segment_index=0)
    assert snapshots.shape == (1, 128, 159)
    for beam_index in (0, 79, 158):
        center = int(schedule.channel_center_samples[0, 0, beam_index])
        np.testing.assert_array_equal(snapshots[0, :, beam_index], signal[0, center - 64 : center + 64])

    second_snapshots = schedule.extract_snapshots(signal + np.int32(1), azimuth_segment_index=1)
    assert id(schedule.channel_center_samples) == center_table_identity
    np.testing.assert_array_equal(second_snapshots, snapshots + np.int32(1))
    snapshot_chunk = schedule.extract_snapshot_chunk(
        signal,
        azimuth_segment_index=0,
        beam_start_index=20,
        beam_stop_index=28,
    )
    np.testing.assert_array_equal(snapshot_chunk, snapshots[:, :, 20:28])


def test_direction_matched_accumulator_updates_each_selected_direction_once() -> None:
    """各秒159方位を1 snapshotずつ更新し、非選択方位を保持する。"""

    positions_m = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    schedule = build_two_second_covariance_snapshot_schedule(
        positions_m,
        fs_hz=1024.0,
        sound_speed_m_s=1500.0,
        snapshot_length_samples=128,
        beams_per_half=5,
    )
    accumulator = DirectionMatchedCovarianceAccumulator(schedule, coef=0.25)
    sample_axis = np.arange(1024, dtype=np.float32)
    signal = np.stack(
        [np.sin(np.float32(2.0 * np.pi * (8 + channel) / 128.0) * sample_axis) for channel in range(3)],
        axis=0,
    ).astype(np.float32)

    first_update = accumulator.process_one_second(signal)
    np.testing.assert_array_equal(first_update.global_direction_indices, [0, 1, 2, 3, 4])
    covariance_after_first = accumulator.direction_covariance.copy()
    assert bool(np.any(np.abs(covariance_after_first[:5]) > 0.0))
    assert bool(np.all(covariance_after_first[5:] == 0.0))

    second_update = accumulator.process_one_second(signal)
    np.testing.assert_array_equal(second_update.global_direction_indices, [8, 7, 6, 5, 4])
    np.testing.assert_array_equal(accumulator.direction_covariance[:4], covariance_after_first[:4])
    assert bool(np.any(np.abs(accumulator.direction_covariance[5:]) > 0.0))
    # 共有90度だけは両segmentで同じ積分先へ入り、2回目のEMA更新を受ける。
    assert bool(np.any(accumulator.direction_covariance[4] != covariance_after_first[4]))


def test_maximum_spatial_correlation_excludes_diagonal_and_handles_zero_power() -> None:
    """最大相関が非対角pairだけを使い、無power binを0とすることを確認する。"""

    covariance = np.zeros((2, 3, 3, 2), dtype=np.complex64)
    covariance[:, :, :, 0] = np.eye(3, dtype=np.complex64)[np.newaxis, :, :]
    covariance[0, 0, 1, 0] = np.complex64(0.6 + 0.0j)
    covariance[0, 1, 0, 0] = np.complex64(0.6 + 0.0j)
    covariance[0, 1, 2, 0] = np.complex64(0.8 + 0.0j)
    covariance[0, 2, 1, 0] = np.complex64(0.8 + 0.0j)
    covariance[1, 0, 2, 0] = np.complex64(0.4 + 0.0j)
    covariance[1, 2, 0, 0] = np.complex64(0.4 + 0.0j)

    result = calculate_maximum_spatial_correlation_table(
        covariance,
        np.array([0.0, 90.0], dtype=np.float32),
        fs_hz=2.0,
    )

    assert result.maximum_correlation.shape == (2, 2)
    np.testing.assert_allclose(result.maximum_correlation[:, 0], [0.8, 0.4], atol=1.0e-6)
    np.testing.assert_array_equal(result.maximum_correlation[:, 1], [0.0, 0.0])
    chunked_result = calculate_maximum_spatial_correlation_table(
        covariance,
        np.array([0.0, 90.0], dtype=np.float32),
        fs_hz=2.0,
        pair_chunk_size=1,
    )
    np.testing.assert_array_equal(chunked_result.maximum_correlation, result.maximum_correlation)


def test_ten_second_integration_uses_actual_direction_update_rate() -> None:
    """通常方位0.5回/sと共有90度1回/sで10秒coefを分ける。"""

    schedule = build_two_second_covariance_snapshot_schedule(
        np.zeros((1, 3), dtype=np.float32),
        fs_hz=1024.0,
        sound_speed_m_s=1500.0,
        snapshot_length_samples=128,
        beams_per_half=5,
    )
    accumulator = DirectionMatchedCovarianceAccumulator(
        schedule,
        integration_time_seconds=10.0,
    )

    # 通常方位は2秒周期に1回なのでrate=0.5/s、共有90度は2回なのでrate=1/s。
    np.testing.assert_allclose(accumulator.direction_update_coef[[0, 8]], [2.0 / 6.0, 2.0 / 6.0])
    np.testing.assert_allclose(accumulator.direction_update_coef[4], 2.0 / 11.0)
