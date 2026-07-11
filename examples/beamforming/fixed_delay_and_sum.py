"""外部 renderer を使わず、固定整相の BL と校正用特徴量を作る最小例。"""

from __future__ import annotations

import numpy as np

from spflow import (
    build_beam_level_display_arrays,
    calculate_bl_shape_features,
    relative_arrival_delay,
    steering_from_relative_delay,
)
from spflow.beamforming import build_source_sector_mask_from_azimuths, design_cbf_weights


def main() -> None:
    """単一 source に対する waiting-beam BL と未校正の形状特徴量を表示する。

    入力と出力:
        8 channel 線状アレイと単一周波数 source を解析式で構成し、
        `[n_beam]` の BL と `BlShapeFeatures` を標準出力へ表示する。

    単位と境界条件:
        位置は m、周波数は Hz、方位は deg、BL は `dB re input RMS`。
        この例は特徴量を観測するだけで、方式比較、sweep、採否判定を行わない。
    """
    sound_speed_m_per_s = 1500.0
    source_frequency_hz = 1500.0
    source_azimuth_deg = 65.0
    channel_count = 8
    sensor_spacing_m = 0.25
    waiting_beam_azimuth_deg = np.linspace(0.0, 180.0, 181, dtype=np.float64)

    # sensor_positions_m shape: [n_channel, 3]。axis=0 は channel、axis=1 は x/y/z [m]。
    sensor_positions_m = np.zeros((channel_count, 3), dtype=np.float64)
    sensor_positions_m[:, 0] = np.arange(channel_count, dtype=np.float64) * sensor_spacing_m
    waiting_azimuth_rad = np.deg2rad(waiting_beam_azimuth_deg)
    # waiting_directions shape: [n_beam, 3]。水平面 source 方向を単位ベクトルで表す。
    waiting_directions = np.stack(
        [
            np.cos(waiting_azimuth_rad),
            np.sin(waiting_azimuth_rad),
            np.zeros_like(waiting_azimuth_rad),
        ],
        axis=1,
    )
    source_azimuth_rad = np.deg2rad(source_azimuth_deg)
    source_direction = np.array(
        [np.cos(source_azimuth_rad), np.sin(source_azimuth_rad), 0.0],
        dtype=np.float64,
    )

    waiting_delay_s = relative_arrival_delay(
        sensor_positions_m,
        waiting_directions,
        sound_speed_m_per_s=sound_speed_m_per_s,
    )
    source_delay_s = relative_arrival_delay(
        sensor_positions_m,
        source_direction,
        sound_speed_m_per_s=sound_speed_m_per_s,
    )
    frequency_hz = np.array([source_frequency_hz], dtype=np.float64)
    # steering shape: [n_channel, n_beam, n_frequency]。
    waiting_steering = steering_from_relative_delay(waiting_delay_s, frequency_hz)
    source_steering = steering_from_relative_delay(source_delay_s, frequency_hz)
    waiting_weights = design_cbf_weights(waiting_steering)

    # 各 waiting beam の w^H a_source を channel 軸で内積する。
    # response shape: [n_beam, n_frequency] -> spectrum shape: [n_beam, n_frequency, n_frame=1]。
    response = np.einsum(
        "cbf,cf->bf",
        np.conjugate(waiting_weights),
        source_steering,
    )
    beam_spectrum = response[:, :, np.newaxis]
    display_arrays = build_beam_level_display_arrays(
        beam_spectrum,
        target_frequency_index=0,
        source_frequency_indices=np.array([0], dtype=np.int64),
        reference_rms=1.0,
        level_reference_label="dB re input RMS",
        floor_db=-120.0,
    )
    source_mask = build_source_sector_mask_from_azimuths(
        waiting_beam_azimuth_deg,
        np.array([source_azimuth_deg], dtype=np.float64),
        guard_deg=10.0,
    )
    features = calculate_bl_shape_features(
        waiting_beam_azimuth_deg,
        display_arrays.target_frequency_bl_level_db,
        source_mask.source_mask,
        source_beam_indices=source_mask.source_beam_indices,
        level_reference_label=display_arrays.level_reference_label,
    )

    print(f"peak_azimuth_deg={features.peak_azimuth_deg:.3f}")
    print(f"peak_width_3db_deg={features.peak_width_3db_deg:.3f}")
    print(f"source_to_guard_peak_margin_db={features.source_to_guard_peak_margin_db:.3f}")
    print("これらは校正前の観測特徴量であり、採否判定値ではありません。")


if __name__ == "__main__":
    main()
