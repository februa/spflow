"""方式3の中心サンプル表、重複snapshot抽出、方位一致積分を検証する。"""

import numpy as np

from spflow.beamforming import (
    CovarianceSnapshotCenterSchedule,
    DirectionMatchedCovarianceAccumulator,
    build_two_second_covariance_snapshot_schedule,
    calculate_maximum_spatial_correlation_table,
)


def test_time_axis_restoration_removes_phase_from_different_snapshot_centers() -> None:
    """同じtoneを異なる中心で切り出しても、位相復元後はchannel間位相が一致する。"""

    schedule = CovarianceSnapshotCenterSchedule(
        beam_azimuth_deg=np.array([[0.0], [180.0]], dtype=np.float32),
        global_direction_azimuth_deg=np.array([0.0], dtype=np.float32),
        direction_match_indices=np.array([[0], [0]], dtype=np.int32),
        # ch1をch0より8 sample後で切り出し、FFTに既知の時間原点差を与える。
        channel_center_samples=np.array([[[256], [264]], [[256], [264]]], dtype=np.int32),
        fs_hz=1024.0,
        snapshot_length_samples=128,
    )
    sample_index = np.arange(1024, dtype=np.float32)
    tone = np.cos(np.float32(2.0 * np.pi * 64.0 / 1024.0) * sample_index).astype(np.float32)
    signal = np.stack((tone, tone), axis=0)
    snapshots = schedule.extract_snapshots(signal, azimuth_segment_index=0)
    spectrum = np.fft.rfft(snapshots, axis=1).astype(np.complex64)
    corrected = spectrum * schedule.calculate_time_axis_restoration_phase(azimuth_segment_index=0)

    tone_bin_index = 8
    # 補正前は8 sample中心差に対応するπ radの位相差があり、補正後は同一時間軸で一致する。
    assert not np.isclose(spectrum[0, tone_bin_index, 0], spectrum[1, tone_bin_index, 0])
    np.testing.assert_allclose(
        corrected[0, tone_bin_index, 0],
        corrected[1, tone_bin_index, 0],
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    accumulator = DirectionMatchedCovarianceAccumulator(schedule, coef=1.0)
    update = accumulator.process_one_second(signal)
    corrected_cross_covariance = update.active_direction_covariance[0, 0, 1, tone_bin_index]
    # Accumulatorが補正spectrumからX X^Hを作るため、交差共分散は正の実軸上へ揃う。
    assert float(np.real(corrected_cross_covariance)) > 0.0
    np.testing.assert_allclose(np.imag(corrected_cross_covariance), 0.0, atol=1.0e-3)


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
    assert schedule.global_direction_azimuth_deg.shape == (159,)
    np.testing.assert_array_equal(schedule.direction_match_indices[0, [0, 1, -1]], [0, 0, 79])
    np.testing.assert_array_equal(schedule.direction_match_indices[1, [0, 1, -1]], [158, 158, 79])
    for segment_index in (0, 1):
        unique_indices, observation_counts = np.unique(
            schedule.direction_match_indices[segment_index],
            return_counts=True,
        )
        assert unique_indices.size == 80
        assert int(np.count_nonzero(observation_counts == 2)) == 79
        assert int(np.count_nonzero(observation_counts == 1)) == 1
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


def test_direction_matched_accumulator_updates_duplicate_directions_in_observation_order() -> None:
    """片側方位を2回順次更新し、非選択側は減衰させず保持する。"""

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
    np.testing.assert_array_equal(first_update.global_direction_indices, [0, 0, 1, 1, 2])
    covariance_after_first = accumulator.direction_covariance.copy()
    assert bool(np.any(np.abs(covariance_after_first[:3]) > 0.0))
    assert bool(np.all(covariance_after_first[3:] == 0.0))

    # global 0へ入るlocal snapshot 0,1の瞬時共分散Q1,Q2を取り出し、
    # R1=(1-a)R0+aQ1、R2=(1-a)R1+aQ2の2回更新と一致することを直接確認する。
    snapshots = schedule.extract_snapshot_chunk(
        signal,
        azimuth_segment_index=0,
        beam_start_index=0,
        beam_stop_index=2,
    )
    spectrum = np.asarray(np.fft.rfft(snapshots, axis=1), dtype=np.complex64)
    spectrum *= schedule.calculate_time_axis_restoration_phase(azimuth_segment_index=0)[:, :, :2]
    instantaneous = np.asarray(
        np.einsum("ikb,jkb->bijk", spectrum, spectrum.conj(), optimize=True),
        dtype=np.complex64,
    )
    expected_after_two_updates = np.asarray(
        np.float32(0.25 * 0.75) * instantaneous[0] + np.float32(0.25) * instantaneous[1],
        dtype=np.complex64,
    )
    np.testing.assert_allclose(covariance_after_first[0], expected_after_two_updates, rtol=1.0e-5, atol=1.0e-4)

    second_update = accumulator.process_one_second(signal)
    np.testing.assert_array_equal(second_update.global_direction_indices, [4, 4, 3, 3, 2])
    np.testing.assert_array_equal(accumulator.direction_covariance[:2], covariance_after_first[:2])
    assert bool(np.any(np.abs(accumulator.direction_covariance[3:]) > 0.0))
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


def test_integration_time_uses_one_update_per_second_for_all_directions() -> None:
    """片側方位の2重観測を含め、全方位の平均更新率を1回/sとする。"""

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

    # 非境界方位は2秒に2回、90度も2秒に2回なので、全てrate=1/sである。
    np.testing.assert_allclose(accumulator.direction_update_coef, np.full(5, 2.0 / 11.0), rtol=1.0e-6)

    # rate*Tが概算有効snapshot数なので、10/40/128 sはそれぞれ10/40/128観測となる。
    for integration_time_seconds in (10.0, 40.0, 128.0):
        integration_accumulator = DirectionMatchedCovarianceAccumulator(
            schedule,
            integration_time_seconds=integration_time_seconds,
        )
        expected_coef = 2.0 / (1.0 + integration_time_seconds)
        np.testing.assert_allclose(
            integration_accumulator.direction_update_coef,
            np.full(5, expected_coef),
            rtol=1.0e-6,
        )
