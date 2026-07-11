"""scene_renderer波面をspflowの通常DASへ接続し、水平・俯仰方位を検証する。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scene_renderer import (
    AcousticSource,
    AmbientField,
    BandLimitedNoiseSpectrum,
    ConstantEnvelope,
    FreeField,
    Receiver,
    Scene,
    SceneRenderer,
    SourceComponent,
    StaticPose,
    tone_component_from_rms_level_db,
)
from scene_renderer.receiver import ArrayGeometry
from spflow import (
    design_cbf_weights,
    integrate_one_sided_band_rms_power,
    one_sided_rfft_bin_rms_power,
    relative_arrival_delay,
    steering_from_relative_delay,
)
from spflow.beamforming import apply_beamformer, apply_beamformer_bands


@dataclass(frozen=True)
class VolumetricArray(ArrayGeometry):
    """水平・俯仰を識別する非共面3次元アレイを表す。

    `sensor_positions_m`のshapeは[n_channel, 3]、axis=0はchannel、axis=1は
    ArrayFrameの[Bow, Starboard, Up]、単位はmである。

    音源生成、steering、DAS重みは責務に含めない。scene_rendererとspflowへ同一の幾何を渡す
    結合試験用ArrayGeometryである。
    """

    sensor_positions_m: NDArray[np.float64]

    def __post_init__(self) -> None:
        """3次元アレイ位置のshapeと有限性を検証する。"""

        positions = np.asarray(self.sensor_positions_m, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[0] < 4 or positions.shape[1] != 3:
            raise ValueError("sensor_positions_m must have shape [n_channel>=4, 3]")
        if not bool(np.all(np.isfinite(positions))):
            raise ValueError("sensor_positions_m must be finite")
        object.__setattr__(self, "sensor_positions_m", positions)

    def positions(self) -> NDArray[Any]:
        """ArrayFrame上の受波器位置を返す。

        Returns:
            受波器位置。shapeは[n_channel, 3]、単位はm。

        Raises:
            なし。生成時にshapeと有限性を検証済み。
        """

        return self.sensor_positions_m.copy()


def _direction_from_azimuth_elevation(
    azimuth_deg: NDArray[Any], elevation_deg: NDArray[Any]
) -> NDArray[np.float64]:
    """ArrayFrameの方位・俯仰から単位方向ベクトルを作る。

    Args:
        azimuth_deg: 相対方位。shapeは[n_direction]、単位はdeg。0は艦首、正は右舷。
        elevation_deg: 俯仰角。shapeは[n_direction]、単位はdeg。上向きを正とする。

    Returns:
        単位方向。shapeは[n_direction, 3]、列は[Bow, Starboard, Up]。

    Raises:
        ValueError: azimuthとelevationのshapeが一致しない場合。
    """

    azimuth = np.asarray(azimuth_deg, dtype=np.float64)
    elevation = np.asarray(elevation_deg, dtype=np.float64)
    if azimuth.shape != elevation.shape:
        raise ValueError("azimuth_deg and elevation_deg must have the same shape")
    azimuth_rad = np.deg2rad(azimuth)
    elevation_rad = np.deg2rad(elevation)
    # direction shape: [n_direction, xyz=3]。水平射影へcos(el)、Upへsin(el)を割り当てる。
    return np.column_stack(
        [
            np.cos(elevation_rad) * np.cos(azimuth_rad),
            np.cos(elevation_rad) * np.sin(azimuth_rad),
            np.sin(elevation_rad),
        ]
    ).astype(np.float64)


def _circular_azimuth_error_deg(observed_deg: float, expected_deg: float) -> float:
    """方位差を[-180, 180) degへ折り返して絶対値を返す。"""

    return abs((observed_deg - expected_deg + 180.0) % 360.0 - 180.0)


# 各tupleは独立したpytestケースであり、複数音源を同時にsceneへ入れない。
# 水平面の全四象限と背面、正負俯仰を組み合わせ、方位・俯仰の符号規約を広く確認する。
_SOURCE_DIRECTIONS_DEG = [
    (-180.0, -60.0),
    (-140.0, 30.0),
    (-100.0, 0.0),
    (-60.0, 60.0),
    (-20.0, -30.0),
    (0.0, 0.0),
    (20.0, 30.0),
    (60.0, -60.0),
    (80.0, 30.0),
    (100.0, 60.0),
    (140.0, 0.0),
    (160.0, -30.0),
]


@pytest.mark.parametrize(("source_azimuth_deg", "source_elevation_deg"), _SOURCE_DIRECTIONS_DEG)
def test_scene_renderer_single_source_is_steered_to_requested_azimuth_and_elevation(
    source_azimuth_deg: float,
    source_elevation_deg: float,
) -> None:
    """個別に生成した全方位分散音源が通常DASの所望azimuth/elevationへ集束する。"""

    sampling_frequency_hz = 8192.0
    sample_count = 4096
    tone_frequency_hz = 2048.0
    tone_bin_index = int(round(tone_frequency_hz * sample_count / sampling_frequency_hz))
    sound_speed_m_per_s = 1500.0

    # 3x3x3の非共面配置により、ULAでは識別不能な左右・前後・上下の方位縮退を避ける。
    # spacing=0.12 mはtone波長0.732 mの半波長未満であり、空間aliasを避ける。
    coordinate_m = np.array([-0.12, 0.0, 0.12], dtype=np.float64)
    sensor_positions_m = np.asarray(
        [[x, y, z] for x in coordinate_m for y in coordinate_m for z in coordinate_m],
        dtype=np.float64,
    )
    receiver = Receiver(
        trajectory=StaticPose([0.0, 0.0, 0.0]),
        array=VolumetricArray(sensor_positions_m),
    )
    source = AcousticSource.from_relative_bearing(
        bearing_deg=source_azimuth_deg,
        elevation_deg=source_elevation_deg,
        distance=1000.0,
        receiver_pose=receiver.trajectory.pose(0.0),
        components=[
            tone_component_from_rms_level_db(
                tone_frequency_hz,
                0.0,
                ConstantEnvelope(),
            )
        ],
        identifier="target",
        role="target",
    )
    # このsceneには1音源だけを入れる。各方位条件はpytest parameterごとに別々に描画する。
    rendered = np.real(
        SceneRenderer().render(
            Scene([source], [], FreeField(sound_speed_m_per_s)),
            receiver,
            np.arange(sample_count, dtype=np.float64) / sampling_frequency_hz,
        )
    )
    channel_spectrum = np.fft.rfft(rendered, axis=1)
    tone_snapshot = np.asarray(channel_spectrum[:, tone_bin_index, np.newaxis], dtype=np.complex128)

    azimuth_axis_deg = np.arange(-180.0, 180.0, 20.0, dtype=np.float64)
    elevation_axis_deg = np.arange(-60.0, 61.0, 30.0, dtype=np.float64)
    scan_azimuth_deg, scan_elevation_deg = np.meshgrid(
        azimuth_axis_deg,
        elevation_axis_deg,
        indexing="xy",
    )
    scan_directions = _direction_from_azimuth_elevation(
        scan_azimuth_deg.reshape(-1),
        scan_elevation_deg.reshape(-1),
    )
    relative_delay_s = relative_arrival_delay(
        sensor_positions_m,
        scan_directions,
        sound_speed_m_per_s=sound_speed_m_per_s,
    )
    steering = steering_from_relative_delay(
        relative_delay_s,
        np.array([tone_frequency_hz], dtype=np.float64),
    )[:, :, 0]
    weights = design_cbf_weights(steering)
    # 通常DASはw=a/(a^Ha)を用いたw^H x。snapshot axisを1として全waiting directionへ適用する。
    beam_tone_bin = apply_beamformer(tone_snapshot, weights)[:, 0]
    beam_rms = np.sqrt(2.0) * np.abs(beam_tone_bin) / float(sample_count)
    peak_index = int(np.argmax(beam_rms))
    observed_azimuth_deg = float(scan_azimuth_deg.reshape(-1)[peak_index])
    observed_elevation_deg = float(scan_elevation_deg.reshape(-1)[peak_index])

    assert _circular_azimuth_error_deg(observed_azimuth_deg, source_azimuth_deg) <= 1.0e-9
    assert abs(observed_elevation_deg - source_elevation_deg) <= 1.0e-9
    # 矩形DASは所望方向で無歪応答を持つため、SL=0 dB re input RMSをRMS=1として復元する。
    np.testing.assert_allclose(float(beam_rms[peak_index]), 1.0, atol=2.0e-6)


def _volumetric_sensor_positions_m() -> NDArray[np.float64]:
    """空間aliasを避けた3x3x3受波器位置を返す。"""

    coordinate_m = np.array([-0.12, 0.0, 0.12], dtype=np.float64)
    return np.asarray(
        [[x, y, z] for x in coordinate_m for y in coordinate_m for z in coordinate_m],
        dtype=np.float64,
    )


def _operational_scan_directions() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """等θ水平beamと俯仰preset、極方向から走査候補を作る。

    Returns:
        `(directions, azimuth_deg, elevation_deg)`。directions shapeは[n_direction,3]。
        極方向のazimuth表記は0 degだが、極の評価には方向ベクトルだけを使う。
    """

    azimuth_axis_deg = np.arange(-180.0, 180.0, 20.0, dtype=np.float64)
    elevation_presets_deg = np.array([-60.0, -30.0, 0.0, 30.0, 60.0], dtype=np.float64)
    azimuth_grid, elevation_grid = np.meshgrid(
        azimuth_axis_deg,
        elevation_presets_deg,
        indexing="xy",
    )
    regular_azimuth = azimuth_grid.reshape(-1)
    regular_elevation = elevation_grid.reshape(-1)
    # 真上・真下では水平射影が0となりazimuthは物理的に定義されないため、各極を1候補だけ追加する。
    all_azimuth = np.concatenate([regular_azimuth, np.array([0.0, 0.0])])
    all_elevation = np.concatenate([regular_elevation, np.array([90.0, -90.0])])
    return (
        _direction_from_azimuth_elevation(all_azimuth, all_elevation),
        all_azimuth,
        all_elevation,
    )


def _direction_angular_error_deg(
    observed_direction: NDArray[Any], expected_direction: NDArray[Any]
) -> float:
    """2つの単位方向ベクトル間の角距離をdegで返す。"""

    observed = np.asarray(observed_direction, dtype=np.float64)
    expected = np.asarray(expected_direction, dtype=np.float64)
    cosine = float(np.clip(np.dot(observed, expected), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cosine)))


@pytest.mark.parametrize(
    ("receiver_heading_deg", "source_azimuth_deg", "source_elevation_deg"),
    [
        (45.0, 80.0, 30.0),
        (123.0, -100.0, -30.0),
        (270.0, 160.0, 60.0),
        (30.0, 0.0, 90.0),
        (210.0, 120.0, -90.0),
    ],
)
def test_tone_das_preserves_array_frame_direction_for_heading_and_poles(
    receiver_heading_deg: float,
    source_azimuth_deg: float,
    source_elevation_deg: float,
) -> None:
    """heading回転後も相対方向へ整相し、極では方向ベクトル誤差が0になる。"""

    fs = 8192.0
    n_sample = 4096
    frequency_hz = 2048.0
    sound_speed = 1500.0
    positions = _volumetric_sensor_positions_m()
    receiver = Receiver(
        StaticPose([0.0, 0.0, 0.0], heading_deg=receiver_heading_deg),
        VolumetricArray(positions),
    )
    source = AcousticSource.from_relative_bearing(
        source_azimuth_deg,
        1000.0,
        receiver.trajectory.pose(0.0),
        [tone_component_from_rms_level_db(frequency_hz, 0.0, ConstantEnvelope())],
        elevation_deg=source_elevation_deg,
    )
    rendered = np.real(
        SceneRenderer().render(
            Scene([source], [], FreeField(sound_speed)),
            receiver,
            np.arange(n_sample, dtype=np.float64) / fs,
        )
    )
    tone_bin = int(round(frequency_hz * n_sample / fs))
    snapshot = np.fft.rfft(rendered, axis=1)[:, tone_bin, np.newaxis]
    scan_directions, _, _ = _operational_scan_directions()
    delays = relative_arrival_delay(positions, scan_directions, sound_speed_m_per_s=sound_speed)
    steering = steering_from_relative_delay(delays, np.array([frequency_hz]))[:, :, 0]
    beam = apply_beamformer(snapshot, design_cbf_weights(steering))[:, 0]
    observed_direction = scan_directions[int(np.argmax(np.abs(beam)))]
    expected_direction = _direction_from_azimuth_elevation(
        np.array([source_azimuth_deg]), np.array([source_elevation_deg])
    )[0]
    assert _direction_angular_error_deg(observed_direction, expected_direction) <= 1.0e-6


@pytest.mark.parametrize(
    ("source_azimuth_deg", "source_elevation_deg"),
    [(-140.0, 30.0), (-60.0, 60.0), (20.0, -30.0), (100.0, 60.0), (160.0, 0.0)],
)
def test_broadband_das_band_power_peaks_at_requested_direction(
    source_azimuth_deg: float,
    source_elevation_deg: float,
) -> None:
    """俯仰presetを使う広帯域DASのband積分powerが入力方向で最大になる。"""

    fs = 4096.0
    n_sample = 4096
    f_low_hz = 500.0
    f_high_hz = 1500.0
    sound_speed = 1500.0
    positions = _volumetric_sensor_positions_m()
    receiver = Receiver(StaticPose([0.0, 0.0, 0.0]), VolumetricArray(positions))
    source = AcousticSource.from_relative_bearing(
        source_azimuth_deg,
        1000.0,
        receiver.trajectory.pose(0.0),
        [
            SourceComponent(
                BandLimitedNoiseSpectrum(f_low_hz, f_high_hz),
                ConstantEnvelope(),
                amplitude=1.0,
                noise_seed=444,
                noise_filter_length=257,
            )
        ],
        elevation_deg=source_elevation_deg,
    )
    rendered = np.real(
        SceneRenderer().render(
            Scene([source], [], FreeField(sound_speed)),
            receiver,
            np.arange(n_sample, dtype=np.float64) / fs,
        )
    )
    frequencies_hz = np.fft.rfftfreq(n_sample, d=1.0 / fs)
    band_mask = (frequencies_hz >= f_low_hz) & (frequencies_hz <= f_high_hz)
    band_frequencies_hz = frequencies_hz[band_mask]
    channel_band_spectrum = np.fft.rfft(rendered, axis=1)[:, band_mask]
    scan_directions, _, _ = _operational_scan_directions()
    delays = relative_arrival_delay(positions, scan_directions, sound_speed_m_per_s=sound_speed)
    steering = steering_from_relative_delay(delays, band_frequencies_hz)
    beam_band_spectrum = apply_beamformer_bands(
        channel_band_spectrum,
        design_cbf_weights(steering),
    )
    # band内はDC/Nyquistを含まないため、one-sided RMS powerは2|Y/N|^2である。
    beam_band_power = np.sum(
        2.0 * np.abs(beam_band_spectrum / float(n_sample)) ** 2,
        axis=1,
    )
    observed_direction = scan_directions[int(np.argmax(beam_band_power))]
    expected_direction = _direction_from_azimuth_elevation(
        np.array([source_azimuth_deg]), np.array([source_elevation_deg])
    )[0]
    assert _direction_angular_error_deg(observed_direction, expected_direction) <= 1.0e-6
    np.testing.assert_allclose(float(np.sqrt(np.max(beam_band_power))), 1.0, rtol=0.03)


def _das_band_output_rms(
    signal: NDArray[Any],
    positions_m: NDArray[np.float64],
    direction: NDArray[np.float64],
    *,
    sampling_frequency_hz: float,
    sound_speed_m_per_s: float,
    f_low_hz: float,
    f_high_hz: float,
) -> float:
    """周波数依存DASを適用し、指定帯域の出力RMSを返す。"""

    signal_array = np.asarray(signal, dtype=np.float64)
    n_sample = int(signal_array.shape[1])
    frequencies_hz = np.fft.rfftfreq(n_sample, d=1.0 / sampling_frequency_hz)
    band_mask = (frequencies_hz >= f_low_hz) & (frequencies_hz <= f_high_hz)
    channel_spectrum = np.fft.rfft(signal_array, axis=1)
    delays = relative_arrival_delay(
        positions_m,
        direction,
        sound_speed_m_per_s=sound_speed_m_per_s,
    )
    steering = steering_from_relative_delay(delays, frequencies_hz[band_mask])[:, np.newaxis, :]
    beam_spectrum = apply_beamformer_bands(
        channel_spectrum[:, band_mask],
        design_cbf_weights(steering),
    )[0]
    # 全rFFT格子へ戻して既存のone-sided power部品と同じDC/Nyquist規約で積分する。
    full_beam_spectrum = np.zeros(frequencies_hz.size, dtype=np.complex128)
    full_beam_spectrum[band_mask] = beam_spectrum
    bin_power = one_sided_rfft_bin_rms_power(full_beam_spectrum, sample_count=n_sample)
    return float(
        np.sqrt(integrate_one_sided_band_rms_power(bin_power, band_mask))
    )


def test_noise_only_das_matches_white_noise_array_gain() -> None:
    """空間白色雑音のDAS出力がw^H Rwと10log10(N) array gainへ一致する。"""

    fs = 4096.0
    n_sample = 65536
    f_low_hz = 500.0
    f_high_hz = 1500.0
    sound_speed = 1500.0
    positions = _volumetric_sensor_positions_m()
    receiver = Receiver(StaticPose([0.0, 0.0, 0.0]), VolumetricArray(positions))
    field = AmbientField.from_asd_level_db(
        BandLimitedNoiseSpectrum(f_low_hz, f_high_hz),
        -40.0,
        noise_seed=500,
        noise_filter_length=513,
    )
    noise = np.real(
        SceneRenderer().render(
            Scene([], [field], FreeField(sound_speed)),
            receiver,
            np.arange(n_sample, dtype=np.float64) / fs,
        )
    )
    target_direction = _direction_from_azimuth_elevation(np.array([40.0]), np.array([30.0]))[0]
    output_rms = _das_band_output_rms(
        noise,
        positions,
        target_direction,
        sampling_frequency_hz=fs,
        sound_speed_m_per_s=sound_speed,
        f_low_hz=f_low_hz,
        f_high_hz=f_high_hz,
    )
    expected_channel_rms = 10.0 ** (-40.0 / 20.0) * np.sqrt(f_high_hz - f_low_hz)
    expected_output_rms = expected_channel_rms / np.sqrt(float(positions.shape[0]))
    np.testing.assert_allclose(output_rms, expected_output_rms, rtol=0.06)
    observed_array_gain_db = 20.0 * np.log10(expected_channel_rms / output_rms)
    np.testing.assert_allclose(observed_array_gain_db, 10.0 * np.log10(positions.shape[0]), atol=0.5)


def test_target_plus_noise_das_power_is_explained_by_separated_components() -> None:
    """target+noise出力powerが同じDASの分離成分power和で説明できる。"""

    fs = 4096.0
    n_sample = 65536
    f_low_hz = 500.0
    f_high_hz = 1500.0
    tone_frequency_hz = 1024.0
    sound_speed = 1500.0
    positions = _volumetric_sensor_positions_m()
    receiver = Receiver(StaticPose([0.0, 0.0, 0.0]), VolumetricArray(positions))
    target_azimuth_deg = 40.0
    target_elevation_deg = 30.0
    target_direction = _direction_from_azimuth_elevation(
        np.array([target_azimuth_deg]), np.array([target_elevation_deg])
    )[0]
    source = AcousticSource.from_relative_bearing(
        target_azimuth_deg,
        1000.0,
        receiver.trajectory.pose(0.0),
        [tone_component_from_rms_level_db(tone_frequency_hz, 0.0, ConstantEnvelope())],
        elevation_deg=target_elevation_deg,
        identifier="target",
        role="target",
    )
    field = AmbientField.from_asd_level_db(
        BandLimitedNoiseSpectrum(f_low_hz, f_high_hz),
        -40.0,
        noise_seed=600,
        noise_filter_length=513,
        identifier="ambient",
        role="noise",
    )
    rendered = SceneRenderer().render_components(
        Scene([source], [field], FreeField(sound_speed)),
        receiver,
        np.arange(n_sample, dtype=np.float64) / fs,
    )
    target = np.real(rendered.sum_by_role("target"))
    noise = np.real(rendered.sum_by_role("noise"))
    mixed = np.real(rendered.mixed)
    common_args = {
        "sampling_frequency_hz": fs,
        "sound_speed_m_per_s": sound_speed,
        "f_low_hz": f_low_hz,
        "f_high_hz": f_high_hz,
    }
    target_rms = _das_band_output_rms(target, positions, target_direction, **common_args)
    noise_rms = _das_band_output_rms(noise, positions, target_direction, **common_args)
    mixed_rms = _das_band_output_rms(mixed, positions, target_direction, **common_args)
    predicted_mixed_power = target_rms**2 + noise_rms**2
    np.testing.assert_allclose(mixed_rms**2, predicted_mixed_power, rtol=0.03)
    np.testing.assert_allclose(target_rms, 1.0, rtol=0.01)
    assert target_rms > noise_rms * 10.0
