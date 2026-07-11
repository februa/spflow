"""scene_renderer波面をspflowの通常DASへ接続し、水平・俯仰方位を検証する。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scene_renderer import (
    AcousticSource,
    ConstantEnvelope,
    FreeField,
    Receiver,
    Scene,
    SceneRenderer,
    StaticPose,
    tone_component_from_rms_level_db,
)
from scene_renderer.receiver import ArrayGeometry
from spflow import (
    design_cbf_weights,
    relative_arrival_delay,
    steering_from_relative_delay,
)
from spflow.beamforming import apply_beamformer


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
