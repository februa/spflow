"""検証済み平面波toneから単一source BLと現行形状特徴量を出力する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spflow import (
    relative_arrival_delay,
    rms_amplitude_to_level_db,
    steering_from_relative_delay,
    synthesize_plane_wave_tone,
)
from spflow.beamforming import design_cbf_coefficients
from spflow.beamforming_evaluation import (
    BlComponentEvaluation,
    build_source_sector_mask_from_azimuths,
    evaluate_target_only_bl,
)
from spflow.beamforming_evaluation.diagnostic_plotting import plot_bl_response


def parse_args() -> argparse.Namespace:
    """BL評価条件をcommand lineから取得する。

    Returns:
        `frequency_hz`、`sampling_frequency_hz`、`output_dir`を持つ引数。

    境界条件:
        toneはFFT binへ一致させる必要があり、`main()`で不一致を検出する。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency-hz", type=float, default=1500.0)
    parser.add_argument("--sampling-frequency-hz", type=float, default=12000.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/beamforming/bl_metric_baseline"),
    )
    return parser.parse_args()


def main() -> None:
    """単一sourceのwaiting-beam BL、NPZ、現行特徴量JSONを生成する。

    入力と出力:
        `delay_and_sum.py`と同じ8 channel平面波toneを入力とし、181本のwaiting beamへ
        周波数領域delay-and-sumを適用する。出力はBL PNG、描画元NPZ、特徴量JSONである。

    shapeと単位:
        channel signalは`[n_channel,n_sample]`、BLは`[n_beam]`。
        位置はm、周波数はHz、方位はdeg、levelは`dB re input RMS`。

    境界条件:
        toneを整数FFT binへ置き、leakageをBL形状へ混入させない。
        現行特徴量は校正前の観測値であり、採否判定には使用しない。
    """
    args = parse_args()
    sound_speed_m_per_s = 1500.0
    sampling_frequency_hz = float(args.sampling_frequency_hz)
    sample_count = 4096
    requested_frequency_hz = float(args.frequency_hz)
    tone_bin_index = int(round(requested_frequency_hz * sample_count / sampling_frequency_hz))
    tone_frequency_hz = tone_bin_index * sampling_frequency_hz / sample_count
    if not np.isclose(tone_frequency_hz, requested_frequency_hz, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            "frequency_hz must coincide with an FFT bin for the configured sample count."
        )
    source_level_db_re_input_rms = 0.0
    source_azimuth_deg = 65.0
    channel_count = 8
    sensor_spacing_m = 0.25
    source_guard_deg = 10.0
    display_floor_db = -80.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # sensor_positions_m shape: [n_channel,3]。axis=0はchannel、axis=1はx/y/z [m]。
    sensor_positions_m = np.zeros((channel_count, 3), dtype=np.float64)
    sensor_positions_m[:, 0] = np.arange(channel_count, dtype=np.float64) * sensor_spacing_m
    source_azimuth_rad = np.deg2rad(source_azimuth_deg)
    source_direction = np.array(
        [np.cos(source_azimuth_rad), np.sin(source_azimuth_rad), 0.0],
        dtype=np.float64,
    )
    generated = synthesize_plane_wave_tone(
        sensor_positions_m,
        source_direction,
        sound_speed_m_per_s=sound_speed_m_per_s,
        sampling_frequency_hz=sampling_frequency_hz,
        sample_count=sample_count,
        frequency_hz=tone_frequency_hz,
        level_db_re_rms=source_level_db_re_input_rms,
    )
    spectrum = np.fft.rfft(generated.signal, axis=1)

    waiting_beam_azimuth_deg = np.linspace(0.0, 180.0, 181, dtype=np.float64)
    waiting_azimuth_rad = np.deg2rad(waiting_beam_azimuth_deg)
    # waiting_directions shape: [n_beam,3]。各rowはreceiverからsourceへ向く単位方向。
    waiting_directions = np.stack(
        [
            np.cos(waiting_azimuth_rad),
            np.sin(waiting_azimuth_rad),
            np.zeros_like(waiting_azimuth_rad),
        ],
        axis=1,
    )
    # tau shape: [n_channel,n_beam]。各waiting方向の相対到達遅延[s]を解析式で求める。
    tau_s = relative_arrival_delay(
        sensor_positions_m,
        waiting_directions,
        sound_speed_m_per_s=sound_speed_m_per_s,
    )
    steering = steering_from_relative_delay(
        tau_s,
        np.array([tone_frequency_hz], dtype=np.float64),
    )[:, :, 0]
    coefficients = design_cbf_coefficients(steering)
    # tone_bin_output shape: [n_beam]。h^T Xをchannel軸で内積する。
    tone_bin_output = np.einsum(
        "cb,c->b",
        coefficients,
        spectrum[:, tone_bin_index],
    )
    # 実toneの内部正周波数binは負周波数共役対を持つため、2|Y/N|^2がRMS powerとなる。
    beam_rms = np.sqrt(2.0 * np.abs(tone_bin_output / sample_count) ** 2)
    bl_level_db = rms_amplitude_to_level_db(
        beam_rms,
        reference_rms=1.0,
        floor_db=display_floor_db,
    )
    source_mask = build_source_sector_mask_from_azimuths(
        waiting_beam_azimuth_deg,
        np.array([source_azimuth_deg], dtype=np.float64),
        guard_deg=source_guard_deg,
    )
    target_only_metrics = evaluate_target_only_bl(
        waiting_beam_azimuth_deg,
        bl_level_db,
        source_azimuth_deg=source_azimuth_deg,
        source_level_db=source_level_db_re_input_rms,
        level_reference_label="dB re input RMS",
    )
    component_evaluation = BlComponentEvaluation(target_only=target_only_metrics)
    wavelength_m = sound_speed_m_per_s / tone_frequency_hz
    spacing_to_wavelength_ratio = sensor_spacing_m / wavelength_m
    source_direction_cosine = float(np.cos(source_azimuth_rad))
    theoretical_alias_azimuths_deg: list[float] = []
    # ULAの同一空間位相条件はu_alias=u_source+m*lambda/dである。
    # m!=0のうち可視方向余弦[-1,1]に入る方位だけを理論grating-lobe候補として保存する。
    for alias_order in range(-channel_count, channel_count + 1):
        if alias_order == 0:
            continue
        alias_direction_cosine = source_direction_cosine + alias_order / spacing_to_wavelength_ratio
        if -1.0 <= alias_direction_cosine <= 1.0:
            theoretical_alias_azimuths_deg.append(
                float(np.rad2deg(np.arccos(alias_direction_cosine)))
            )
    metrics = component_evaluation.as_dict()
    metrics.update(
        {
            "source_azimuth_deg": source_azimuth_deg,
            "source_frequency_hz": tone_frequency_hz,
            "sampling_frequency_hz": sampling_frequency_hz,
            "source_level_db_re_input_rms": source_level_db_re_input_rms,
            "wavelength_m": wavelength_m,
            "sensor_spacing_m": sensor_spacing_m,
            "spacing_to_wavelength_ratio": spacing_to_wavelength_ratio,
            "aperture_length_m": (channel_count - 1) * sensor_spacing_m,
            "theoretical_alias_azimuths_deg": theoretical_alias_azimuths_deg,
            "source_guard_deg": source_guard_deg,
            "display_floor_db": display_floor_db,
            "evaluation_pattern": "fixed_beam_single_source",
            "metric_status": "uncalibrated_observation",
            "component_status": {
                "target_only": "evaluated",
                "noise_only": "not_generated",
                "mixed": "not_generated",
            },
        }
    )
    metrics_path = output_dir / "bl_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez(
        output_dir / "bl_plot_data.npz",
        waiting_beam_azimuth_deg=waiting_beam_azimuth_deg,
        bl_level_db_re_input_rms=bl_level_db,
        source_mask=source_mask.source_mask,
        source_azimuth_deg=np.array(source_azimuth_deg),
        source_frequency_hz=np.array(tone_frequency_hz),
    )
    plot_bl_response(
        waiting_beam_azimuth_deg,
        bl_level_db,
        target_azimuth_deg=source_azimuth_deg,
        peak_azimuth_deg=target_only_metrics.peak_azimuth_deg,
        title="Delay-and-sum beam response for one plane-wave tone",
        caption=(
            f"Source={source_azimuth_deg:.1f} deg, {tone_frequency_hz:.1f} Hz, "
            f"SL={source_level_db_re_input_rms:.1f} dB re input RMS; "
            f"guard=±{source_guard_deg:.1f} deg; display floor={display_floor_db:.1f} dB."
        ),
        output_path=output_dir / "bl.png",
        response_label="Delay-and-sum BL",
        level_unit_label="dB re input RMS",
        source_guard_deg=source_guard_deg,
        level_limits_db=(display_floor_db, 3.0),
        diagnostic_peak_points=[
            (peak.azimuth_deg, "Grating-lobe candidate")
            for peak in target_only_metrics.grating_lobe_candidates
        ],
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
