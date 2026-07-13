"""再利用可能なtone scene生成部品のlevel・shape・dtypeを検証する。"""

from __future__ import annotations

import numpy as np

from spflow.simulation import (
    SimulationPrecision,
    ToneSceneSource,
    direction_from_azimuth_elevation,
    synthesize_tone_scene,
)


def test_tone_scene_preserves_rms_level_shape_and_selected_precision() -> None:
    """整数周期toneのRMSとmulti-channel shapeが指定精度で保たれることを確認する。"""

    positions_m = np.zeros((2, 3), dtype=np.float64)
    scene = synthesize_tone_scene(
        array_positions_m=positions_m,
        sources=(ToneSceneSource(azimuth_deg=90.0, frequency_hz=100.0, level_db20=0.0),),
        fs_hz=1000.0,
        duration_s=0.1,
        sound_speed_m_s=1500.0,
        # noiseを数値上無視できるlevelにし、tone RMS規約だけを検証する。
        noise_level_db20=-300.0,
        random_seed=1234,
        precision=SimulationPrecision.DOUBLE,
    )

    assert scene.signal.shape == (2, 100)
    assert scene.time_axis_s.shape == (100,)
    assert scene.signal.dtype == np.float64
    assert scene.time_axis_s.dtype == np.float64
    np.testing.assert_allclose(np.sqrt(np.mean(scene.signal**2, axis=1)), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(scene.signal[0], scene.signal[1], atol=1.0e-12)


def test_direction_from_azimuth_elevation_uses_xyz_axis_order() -> None:
    """方位90deg・俯仰0degが+y方向を表すことを確認する。"""

    direction = direction_from_azimuth_elevation(90.0, 0.0)

    assert direction.shape == (3,)
    np.testing.assert_allclose(direction, np.array([0.0, 1.0, 0.0]), atol=1.0e-12)
