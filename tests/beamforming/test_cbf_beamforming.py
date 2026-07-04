"""cbf beamforming に関する回帰試験。"""

# ここでは steering の向き、共分散推定、重み適用後の再構成が噛み合うことを
# 決定論的な入力で固定し、ビームフォーミング変更時の退行を早期に検知する。

import numpy as np

from spflow import (
    BandwiseArrayDesign,
    CBFBeamformer,
    CBFOverlapSaveBeamformer,
    apply_beamformer,
    apply_beamformer_bands,
    apply_channel_window_to_steering,
    design_cbf_overlap_save_filters,
    design_cbf_weights,
    design_cbf_weights_with_channel_window,
    make_directions,
)


def _collect_overlap_save_output(records: list[tuple[int, np.ndarray]], n_beam: int, n_band: int) -> np.ndarray:
    """`_collect_overlap_save_output` を実行する。"""
    per_band: list[list[np.ndarray]] = [[] for _ in range(n_band)]
    for band_idx, valid in records:
        per_band[band_idx].append(valid)

    pieces = []
    for band_idx in range(n_band):
        if per_band[band_idx]:
            pieces.append(np.concatenate(per_band[band_idx], axis=-1))
        else:
            pieces.append(np.zeros((n_beam, 0), dtype=np.complex64))
    return np.stack(pieces, axis=1)


def test_make_directions_right_side_returns_positive_y_direction_cosines():
    """方向ベクトル生成について `right side` 指定で正の y 方向余弦を返す を確認する。"""
    dir3d, axis_az, axis_el = make_directions(
        az_min_deg=30.0,
        az_max_deg=70.0,
        el_min_deg=-30.0,
        el_max_deg=20.0,
        n_beam_az_real=5,
        n_beam_az_virtual=2,
        array_side='right side',
    )

    assert dir3d.shape == (3, 28)
    assert axis_az.shape == (7,)
    assert axis_el.shape == (4,)
    assert np.all(dir3d[1] >= 0.0)
    np.testing.assert_allclose(np.sum(dir3d**2, axis=0), 1.0, atol=1e-6)


def test_make_directions_left_side_flips_y_sign():
    """方向ベクトル生成について `left side` 指定で y 成分の符号を反転する を確認する。"""
    right_dir3d, _, _ = make_directions(
        az_min_deg=30.0,
        az_max_deg=70.0,
        el_min_deg=-30.0,
        el_max_deg=20.0,
        n_beam_az_real=5,
        n_beam_az_virtual=2,
        array_side='right side',
    )
    left_dir3d, _, _ = make_directions(
        az_min_deg=30.0,
        az_max_deg=70.0,
        el_min_deg=-30.0,
        el_max_deg=20.0,
        n_beam_az_real=5,
        n_beam_az_virtual=2,
        array_side='left side',
    )

    np.testing.assert_allclose(left_dir3d[0], right_dir3d[0], atol=1e-6)
    np.testing.assert_allclose(left_dir3d[1], -right_dir3d[1], atol=1e-6)
    np.testing.assert_allclose(left_dir3d[2], right_dir3d[2], atol=1e-6)


def test_make_directions_side_array_requires_0_to_180_azimuth_range():
    """方向ベクトル生成について side-array モードでは 0..180 度の方位範囲だけを受け付けることを確認する。"""
    try:
        make_directions(
            az_min_deg=-90.0,
            az_max_deg=90.0,
            el_min_deg=-30.0,
            el_max_deg=20.0,
            n_beam_az_real=5,
            n_beam_az_virtual=2,
            array_side='right side',
        )
    except ValueError as exc:
        assert '0 <= az_min_deg <= az_max_deg <= 180' in str(exc)
    else:
        raise AssertionError('side-array mode must reject signed azimuth ranges.')

def test_make_directions_side_array_requires_0_to_180_azimuth_range():
    """方向ベクトル生成について side-array モードでは 0..180 度の方位範囲だけを受け付けることを確認する。"""
    try:
        make_directions(
            az_min_deg=-90.0,
            az_max_deg=90.0,
            el_min_deg=-30.0,
            el_max_deg=20.0,
            n_beam_az_real=5,
            n_beam_az_virtual=2,
            array_side='right side',
        )
    except ValueError as exc:
        assert '0 <= az_min_deg <= az_max_deg <= 180' in str(exc)
    else:
        raise AssertionError('side-array mode must reject signed azimuth ranges.')


def test_make_directions_forward_uses_uniform_azimuth_space():
    """方向ベクトル生成について `forward` 指定で一様な方位角空間を使う を確認する。"""
    dir3d, axis_az, axis_el = make_directions(
        az_min_deg=-40.0,
        az_max_deg=40.0,
        el_min_deg=-30.0,
        el_max_deg=20.0,
        n_beam_az_real=5,
        n_beam_az_virtual=3,
        array_side='forward',
    )

    np.testing.assert_allclose(axis_az, np.linspace(-40.0, 40.0, 8), atol=1e-6)
    assert np.any(dir3d[1] < 0.0)
    assert np.any(dir3d[1] > 0.0)
    np.testing.assert_allclose(axis_el, np.array([-30.0, 6.0, 10.6, 18.1]), atol=1e-6)


def test_bandwise_array_design_accepts_external_ndarrays_directly():
    """帯域別アレイ設計が外部 ndarray をそのまま受け入れることを確認する。"""
    design = BandwiseArrayDesign.from_ndarrays(
        channel_positions_m=np.array([-1.0, 0.0, 1.0]),
        shading_table=np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
    )

    np.testing.assert_allclose(design.channel_positions_m, np.array([-1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(design.active_channel_indices(0), np.array([0, 1]))
    np.testing.assert_array_equal(design.active_channel_indices(1), np.array([1, 2]))


def test_bandwise_array_design_accepts_external_3d_positions():
    """帯域別アレイ設計が外部 3D 座標を受け入れることを確認する。"""
    design = BandwiseArrayDesign.from_ndarrays(
        channel_positions_m=np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, 0.0, 0.0]]),
        shading_table=np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
    )

    assert design.is_explicit_3d
    np.testing.assert_allclose(design.positions_3d(), np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, 0.0, 0.0]]))
    np.testing.assert_allclose(design.active_aperture_m(0), np.sqrt(1.25), atol=1e-6)
    np.testing.assert_allclose(design.minimum_spacing_m(1), np.sqrt(1.25), atol=1e-6)


def test_bandwise_array_design_builds_centered_rectangular_masks():
    """帯域別アレイ設計が中央寄せの矩形マスクを構成することを確認する。"""
    design = BandwiseArrayDesign.from_uniform_linear_centered_rectangular(
        n_ch=7,
        spacing_m=0.5,
        n_band=3,
        active_counts=[3, 5, 7],
    )

    np.testing.assert_allclose(design.channel_positions_m, np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]))
    np.testing.assert_array_equal(design.active_channel_indices(0), np.array([2, 3, 4]))
    np.testing.assert_array_equal(design.active_channel_indices(1), np.array([1, 2, 3, 4, 5]))


def test_bandwise_array_design_frequency_progressive_rectangular_shrinks_with_frequency():
    """帯域別アレイ設計について 周波数上昇に従って矩形開口が縮小する を確認する。"""
    design = BandwiseArrayDesign.from_uniform_linear_frequency_progressive_rectangular(
        n_ch=9,
        spacing_m=0.1,
        fs=16000.0,
        n_band=16,
        sound_speed=343.0,
        aperture_wavelengths=1.0,
        min_active_ch=3,
    )

    counts = design.active_channel_counts_per_band()
    assert counts[0] == 9
    assert np.min(counts) >= 3
    assert counts[1] >= counts[2] >= counts[3] >= counts[4]


def test_bandwise_array_design_nested_sparse_builds_dense_center_sparse_edges():
    """帯域別アレイ設計について 中央密・端疎の配置を構成する を確認する。"""
    design = BandwiseArrayDesign.from_nested_sparse_linear_frequency_progressive(
        n_dense_ch=5,
        dense_spacing_m=0.1,
        n_outer_pairs=2,
        outer_spacing_m=0.4,
        fs=16000.0,
        n_band=16,
        sound_speed=343.0,
        aperture_wavelengths=1.0,
        min_active_ch=3,
    )

    np.testing.assert_allclose(
        design.channel_positions_m,
        np.array([-1.0, -0.6, -0.2, -0.1, 0.0, 0.1, 0.2, 0.6, 1.0]),
    )
    assert design.active_channel_count(0) == 9
    assert design.active_channel_count(4) <= design.active_channel_count(2)
    np.testing.assert_allclose(design.minimum_spacing_m(4), 0.1, atol=1e-6)
    assert design.active_aperture_m(4) <= design.active_aperture_m(2)


def test_bandwise_array_design_positions_3d_places_sensors_on_requested_axis():
    """帯域別アレイ設計について `positions_3d` が指定軸上にセンサを配置する を確認する。"""
    design = BandwiseArrayDesign.from_uniform_linear_centered_rectangular(
        n_ch=3,
        spacing_m=0.2,
        n_band=1,
        active_counts=[3],
    )

    pos = design.positions_3d(axis=1)

    np.testing.assert_allclose(pos[:, 0], 0.0)
    np.testing.assert_allclose(pos[:, 1], np.array([-0.2, 0.0, 0.2]))
    np.testing.assert_allclose(pos[:, 2], 0.0)


def test_apply_channel_window_to_steering_supports_bandwise_table():
    """steering への channel window 適用が帯域別テーブル入力をサポートすることを確認する。"""
    steering = np.ones((3, 1, 2), dtype=np.complex64)
    window = np.array(
        [
            [1.0, 0.0],
            [0.5, 1.0],
            [0.0, 1.0],
        ]
    )

    shaded = apply_channel_window_to_steering(steering, window)

    np.testing.assert_allclose(shaded[:, 0, 0], np.array([1.0, 0.5, 0.0]))
    np.testing.assert_allclose(shaded[:, 0, 1], np.array([0.0, 1.0, 1.0]))


def test_design_cbf_weights_is_distortionless_for_target():
    """CBF 重み設計が target に対して無歪みであることを確認する。"""
    steering = np.array([[1.0 + 0.0j], [1.0j]])

    weights = design_cbf_weights(steering)
    response = weights.conj().T @ steering

    np.testing.assert_allclose(response, np.ones((1, 1)), atol=1e-6)


def test_design_cbf_weights_with_channel_window_is_distortionless_on_shaded_steering():
    """channel window 付き CBF 重み設計が shading 付き steering に対して無歪みであることを確認する。"""
    steering = np.array([[1.0 + 0.0j], [1.0 + 0.0j], [1.0 + 0.0j]])
    window = np.array([0.0, 1.0, 1.0])

    shaded = apply_channel_window_to_steering(steering, window)
    weights = design_cbf_weights_with_channel_window(steering, window)

    np.testing.assert_allclose(weights[0, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(weights.conj().T @ shaded, np.ones((1, 1)), atol=1e-6)


def test_design_cbf_weights_handles_bandwise_input():
    """CBF 重み設計が帯域別入力を処理できることを確認する。"""
    steering = np.stack(
        [
            np.array([[1.0 + 0.0j], [1.0 + 0.0j]]),
            np.array([[1.0 + 0.0j], [0.0 + 1.0j]]),
        ],
        axis=-1,
    )

    weights = design_cbf_weights(steering)

    assert weights.shape == steering.shape
    for band_idx in range(weights.shape[-1]):
        response = weights[:, :, band_idx].conj().T @ steering[:, :, band_idx]
        np.testing.assert_allclose(response, np.ones((1, 1)), atol=1e-6)


def test_design_cbf_overlap_save_filters_embed_conjugation_in_filter_fft():
    """CBF overlap-save フィルタ設計が複素共役を filter FFT に埋め込むことを確認する。"""
    steering = np.array([[[1.0 + 0.0j], [1.0j]]]).transpose(0, 2, 1)
    filters = design_cbf_overlap_save_filters(steering, frame_size=8)
    weights = design_cbf_weights(steering)

    taps = np.fft.ifft(filters, axis=-1)
    np.testing.assert_allclose(taps[..., 0], weights.conj(), atol=1e-6)
    np.testing.assert_allclose(taps[..., 1:], 0.0, atol=1e-6)


def test_cbf_beamformer_matches_manual_projection():
    """CBF ビームフォーマが手計算の射影結果と一致することを確認する。"""
    steering = np.array([[1.0 + 0.0j], [1.0j]])
    X = np.array(
        [
            [1.0 + 0.0j, 2.0 + 0.0j],
            [0.0 + 1.0j, 0.0 + 2.0j],
        ]
    )

    beamformer = CBFBeamformer(steering)
    out = beamformer.process(X)
    expected = apply_beamformer(X, design_cbf_weights(steering))

    np.testing.assert_allclose(out, expected)


def test_cbf_beamformer_uses_channel_window():
    """CBF ビームフォーマが channel window を使うことを確認する。"""
    steering = np.array([[1.0 + 0.0j], [1.0 + 0.0j], [1.0 + 0.0j]])
    window = np.array([0.0, 1.0, 1.0])
    X = np.array(
        [
            [10.0 + 0.0j, 10.0 + 0.0j],
            [1.0 + 0.0j, 2.0 + 0.0j],
            [3.0 + 0.0j, 4.0 + 0.0j],
        ]
    )

    beamformer = CBFBeamformer(steering, channel_window=window)
    out = beamformer.process(X)
    expected = np.array([[(1.0 + 3.0) / 2.0, (2.0 + 4.0) / 2.0]], dtype=np.complex64)

    np.testing.assert_allclose(out, expected)


def test_apply_beamformer_bands_projects_all_bins():
    """帯域別ビームフォーマ適用が全周波数ビンへ射影を適用することを確認する。"""
    X = np.array(
        [
            [1.0 + 0.0j, 2.0 + 0.0j],
            [0.0 + 1.0j, 0.0 + 2.0j],
        ]
    )
    steering = np.stack(
        [
            np.array([[1.0 + 0.0j], [1.0 + 0.0j]]),
            np.array([[1.0 + 0.0j], [0.0 + 1.0j]]),
        ],
        axis=-1,
    )
    weights = design_cbf_weights(steering)

    out = apply_beamformer_bands(X, weights)

    expected = np.array([
        [weights[:, 0, 0].conj() @ X[:, 0], weights[:, 0, 1].conj() @ X[:, 1]],
    ])
    np.testing.assert_allclose(out, expected)


def test_cbf_overlap_save_beamformer_matches_pointwise_projection_for_length1_filters():
    """CBF overlap-save ビームフォーマについて 長さ 1 フィルタでは点ごとの射影結果と一致する を確認する。"""
    steering = np.array([[1.0 + 0.0j], [1.0j]])[:, :, np.newaxis]
    X = np.array(
        [
            np.arange(1, 17, dtype=np.float32),
            1j * np.arange(1, 17, dtype=np.float32),
        ],
        dtype=np.complex64,
    )[:, np.newaxis, :]

    beamformer = CBFOverlapSaveBeamformer(steering, frame_size=8, valid_size=4)
    records: list[tuple[int, np.ndarray]] = []
    for start in range(0, X.shape[-1], 3):
        records.extend(beamformer.process(X[:, :, start : start + 3]))
    records.extend(beamformer.flush())

    out = _collect_overlap_save_output(records, n_beam=1, n_band=1)
    expected = apply_beamformer(X[:, 0, :], design_cbf_weights(steering[:, :, 0]))

    np.testing.assert_allclose(out[0, 0, : X.shape[-1]], expected[0], atol=1e-6)

