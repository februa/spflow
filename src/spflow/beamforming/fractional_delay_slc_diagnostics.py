"""小数遅延固定整相の後段へ周波数選択 SLC を適用した BL/FRAZ/BTR 診断を行うモジュール。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .diagnostic_plotting import (
    build_beam_diagnostic_plot_usage_notes,
    plot_bl_comparison,
    plot_bl_response,
    plot_btr_heatmap,
    plot_fraz_heatmap,
    require_matplotlib,
    write_beam_diagnostic_plot_usage_notes,
)
from .slc import SlcConfig
from .time_delay import FractionalDelayAndSumBeamformer
from .time_delay_diagnostics import (
    TimeDelayDiagnosticConfig,
    TimeDelayDiagnosticSource,
    _build_array_positions,
    _build_beam_grid,
    _compress_time_rms_levels,
    _format_source_caption_fragment,
    _generate_target_scene,
    _resolve_source_specs,
    _rfft_levels_db20,
    _sanitize_label_for_filename,
    _source_label,
    _tone_level_db20_rms,
    _validate_config,
)
from .time_delay_slc_diagnostics import (
    _apply_frequency_selective_scan_slc,
    _build_source_comparisons,
    _build_tone_snapshots,
    _evaluate_stage_source_metrics,
    _resolve_target_source_indices,
    _synthesize_tone_from_snapshots,
)


def _build_interference_source_comparisons(
    fixed_source_metrics: list[dict[str, float | str]],
    slc_source_metrics: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    """interferer 方位での before/after レベル差を評価用辞書へまとめる。

    Args:
        fixed_source_metrics: SLC 前の interferer 指標。各要素は `_evaluate_stage_source_metrics()`
            が返す辞書であり、`level_at_nearest_source_grid_db20` などを含む。
        slc_source_metrics: SLC 後の interferer 指標。shape と順序は `fixed_source_metrics`
            と一致している必要がある。

    Returns:
        interferer 方位近傍の抑圧量一覧。各要素は同一 interferer source に対する
        nearest beam level reduction、local peak reduction、nonlocal peak reduction を dB 単位で持つ。
    """
    comparisons: list[dict[str, float | str]] = []
    for fixed_metric, slc_metric in zip(fixed_source_metrics, slc_source_metrics, strict=True):
        # interferer 評価では「維持」ではなく「低下」が目的である。
        # target 評価用の mainlobe_preserved 判定とは分け、方位近傍レベルと局所 peak の低下量を直接記録する。
        nearest_level_reduction_db = float(fixed_metric["level_at_nearest_source_grid_db20"]) - float(
            slc_metric["level_at_nearest_source_grid_db20"]
        )
        local_peak_reduction_db = float(fixed_metric["peak_level_db20"]) - float(slc_metric["peak_level_db20"])
        max_nonlocal_reduction_db = float(fixed_metric["max_nonlocal_level_db20"]) - float(
            slc_metric["max_nonlocal_level_db20"]
        )
        comparisons.append(
            {
                "label": str(fixed_metric["label"]),
                "source_azimuth_deg": float(fixed_metric["source_azimuth_deg"]),
                "source_frequency_hz": float(fixed_metric["source_frequency_hz"]),
                "nearest_level_reduction_db": nearest_level_reduction_db,
                "local_peak_reduction_db": local_peak_reduction_db,
                "max_nonlocal_reduction_db": max_nonlocal_reduction_db,
                "fixed_nearest_level_db20": float(fixed_metric["level_at_nearest_source_grid_db20"]),
                "slc_nearest_level_db20": float(slc_metric["level_at_nearest_source_grid_db20"]),
                "fixed_peak_azimuth_deg": float(fixed_metric["peak_azimuth_deg"]),
                "slc_peak_azimuth_deg": float(slc_metric["peak_azimuth_deg"]),
            }
        )
    return comparisons


def _build_fractional_beam_response_matrix(
    beamformer: FractionalDelayAndSumBeamformer,
    frequency_hz: float,
) -> np.ndarray:
    """小数遅延固定整相の理論ビーム応答行列を返す。

    Args:
        beamformer: 小数遅延固定整相器。`steering_response()` と `delay_table` を持つ。
        frequency_hz: 評価周波数。単位は Hz。

    Returns:
        observation beam と look beam の複素応答行列。shape は `[n_beam, n_beam]`。
    """
    arrival_delay_sec = np.asarray(beamformer.delay_table.arrival_delay_sec, dtype=np.float64)

    # steering_response[ch, beam_obs] は、整数遅延位相と保存済み小数遅延 FIR の
    # 複素周波数応答を含む observation beam 側の整相器応答である。
    # これと look 方向ごとの到来位相 arrival_phase[ch, beam_look] をセンサ平均することで、
    # 小数遅延込みの beam-to-beam 理論応答行列を得る。
    steering_response = np.asarray(beamformer.steering_response(float(frequency_hz)), dtype=np.complex128)
    arrival_phase = np.exp(-1j * 2.0 * np.pi * float(frequency_hz) * arrival_delay_sec)
    return (steering_response.T @ arrival_phase) / float(arrival_delay_sec.shape[0])


def _evaluate_fractional_source_metrics_and_save_bl(
    beam_output: np.ndarray,
    axis_az_deg: np.ndarray,
    btr_relative_levels_db: np.ndarray,
    source_specs: tuple[TimeDelayDiagnosticSource, ...],
    config: TimeDelayDiagnosticConfig,
    output_dir: Path,
) -> list[dict[str, float | str]]:
    """小数遅延固定整相の source ごとの BL 指標を計算し、BL 図を保存する。"""
    source_metrics: list[dict[str, float | str]] = []

    for source_index, source_spec in enumerate(source_specs):
        beam_levels_db20 = np.array(
            [
                _tone_level_db20_rms(
                    beam_output[beam_index],
                    frequency_hz=float(source_spec.frequency_hz),
                    fs_hz=float(config.fs_hz),
                )
                for beam_index in range(beam_output.shape[0])
            ],
            dtype=np.float64,
        )
        peak_beam_index = int(np.argmax(beam_levels_db20))
        nearest_beam_index = int(np.argmin(np.abs(axis_az_deg - float(source_spec.azimuth_deg))))
        mirror_beam_index = int(np.argmin(np.abs(axis_az_deg - (180.0 - float(source_spec.azimuth_deg)))))
        label = _source_label(source_spec, source_index)

        if len(source_specs) == 1:
            bl_output_path = output_dir / "bl.png"
        else:
            bl_output_path = output_dir / f"bl_{source_index:02d}_{_sanitize_label_for_filename(label)}.png"

        plot_bl_response(
            axis_az_deg=axis_az_deg,
            beam_levels_db20=beam_levels_db20,
            target_azimuth_deg=float(source_spec.azimuth_deg),
            peak_azimuth_deg=float(axis_az_deg[peak_beam_index]),
            title=f"BL: Fractional-delay beam response ({label}, {float(source_spec.frequency_hz):.1f} Hz)",
            caption=(
                f"source={float(source_spec.frequency_hz):.1f} Hz, target azimuth={float(source_spec.azimuth_deg):.2f} deg, "
                f"peak azimuth={float(axis_az_deg[peak_beam_index]):.2f} deg"
            ),
            output_path=bl_output_path,
            response_label=f"Fractional delay beam response ({label})",
        )

        source_metrics.append(
            {
                "label": label,
                "source_azimuth_deg": float(source_spec.azimuth_deg),
                "source_elevation_deg": float(source_spec.elevation_deg),
                "source_frequency_hz": float(source_spec.frequency_hz),
                "source_level_db20": float(source_spec.level_db20),
                "nearest_beam_azimuth_deg": float(axis_az_deg[nearest_beam_index]),
                "bl_peak_azimuth_deg": float(axis_az_deg[peak_beam_index]),
                "bl_peak_level_db20": float(beam_levels_db20[peak_beam_index]),
                "bl_level_at_nearest_source_grid_db20": float(beam_levels_db20[nearest_beam_index]),
                "mirror_azimuth_deg": float(axis_az_deg[mirror_beam_index]),
                "mirror_level_db20": float(beam_levels_db20[mirror_beam_index]),
                "btr_mean_relative_level_db": float(np.mean(btr_relative_levels_db[:, nearest_beam_index])),
                "btr_min_relative_level_db": float(np.min(btr_relative_levels_db[:, nearest_beam_index])),
                "bl_png_path": str(bl_output_path.resolve()),
            }
        )

    return source_metrics


def _run_fractional_delay_diagnostics(
    config: TimeDelayDiagnosticConfig,
    fractional_delay_filter_bank_path: str | Path,
) -> tuple[dict[str, object], FractionalDelayAndSumBeamformer, np.ndarray, np.ndarray, tuple[TimeDelayDiagnosticSource, ...], np.ndarray]:
    """小数遅延固定整相の BL/FRAZ/BTR を保存し、SLC 前段の評価結果を返す。"""
    require_matplotlib()
    _validate_config(config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 表示方法の解釈を固定整相 before/after で揃えるため、
    # 小数遅延版でも整数遅延版と同じ注意事項ファイルを出力しておく。
    plot_usage_notes = build_beam_diagnostic_plot_usage_notes()
    write_beam_diagnostic_plot_usage_notes(output_dir / "plot_usage_notes.md", plot_usage_notes)

    source_specs = _resolve_source_specs(config)
    array_positions_m, array_geometry_name, array_is_sparse = _build_array_positions(config)
    beam_grid = _build_beam_grid(config)
    multichannel_signal, _ = _generate_target_scene(array_positions_m, source_specs, config)

    beamformer = FractionalDelayAndSumBeamformer.from_geometry_and_filter_bank_path(
        array_pos_m=array_positions_m,
        dir_cos=np.asarray(beam_grid["directions"], dtype=np.float64),
        fs_hz=float(config.fs_hz),
        sound_speed_m_s=float(config.sound_speed_m_s),
        fractional_filter_bank_path=fractional_delay_filter_bank_path,
    )
    beam_output = beamformer.process(multichannel_signal)

    axis_az_deg = np.asarray(beam_grid["axis_az_deg"], dtype=np.float64)
    freqs_hz, fraz_levels_db20 = _rfft_levels_db20(beam_output, fs_hz=float(config.fs_hz))
    fraz_global_peak_beam_index, fraz_global_peak_frequency_index = np.unravel_index(
        np.argmax(fraz_levels_db20),
        fraz_levels_db20.shape,
    )
    fraz_global_peak_azimuth_deg = float(axis_az_deg[int(fraz_global_peak_beam_index)])
    fraz_global_peak_frequency_hz = float(freqs_hz[int(fraz_global_peak_frequency_index)])
    fraz_global_peak_level_db20 = float(
        fraz_levels_db20[int(fraz_global_peak_beam_index), int(fraz_global_peak_frequency_index)]
    )

    btr_levels_db20, btr_times_s = _compress_time_rms_levels(
        beam_output,
        fs_hz=float(config.fs_hz),
        block_size=int(config.btr_block_size),
    )
    btr_relative_levels_db = btr_levels_db20 - np.max(btr_levels_db20, axis=1, keepdims=True)
    btr_peak_beam_indices = np.argmax(btr_levels_db20, axis=1)
    btr_peak_azimuths_deg = axis_az_deg[btr_peak_beam_indices]

    source_metrics = _evaluate_fractional_source_metrics_and_save_bl(
        beam_output=beam_output,
        axis_az_deg=axis_az_deg,
        btr_relative_levels_db=btr_relative_levels_db,
        source_specs=source_specs,
        config=config,
        output_dir=output_dir,
    )

    source_caption_fragment = _format_source_caption_fragment(source_specs)
    fraz_target_points = [
        (
            float(source_spec.azimuth_deg),
            float(source_spec.frequency_hz),
            f"Target {source_index + 1}",
        )
        for source_index, source_spec in enumerate(source_specs)
    ]
    fraz_peak_points = [
        (
            float(source_metrics[source_index]["bl_peak_azimuth_deg"]),
            float(source_spec.frequency_hz),
            f"Peak {source_index + 1}",
        )
        for source_index, source_spec in enumerate(source_specs)
    ]
    plot_fraz_heatmap(
        axis_az_deg=axis_az_deg,
        freqs_hz=freqs_hz,
        fraz_levels_db20=fraz_levels_db20,
        target_points=fraz_target_points,
        peak_points=fraz_peak_points,
        title="FRAZ: Fractional-delay frequency-azimuth response",
        caption=f"sources={source_caption_fragment}",
        output_path=output_dir / "fraz.png",
    )

    if len(source_specs) == 1:
        btr_caption = (
            f"各時刻で最大ビームを 0 dB に正規化, "
            f"mean peak azimuth={float(np.mean(btr_peak_azimuths_deg)):.2f} deg"
        )
        btr_track = btr_peak_azimuths_deg
    else:
        btr_caption = (
            "複数同時音源のため peak track は代表値にならない。"
            f" target azimuths={', '.join([f'{float(source_spec.azimuth_deg):.1f}' for source_spec in source_specs])} deg"
        )
        btr_track = None

    plot_btr_heatmap(
        axis_az_deg=axis_az_deg,
        times_s=btr_times_s,
        btr_relative_levels_db=btr_relative_levels_db,
        btr_peak_azimuths_deg=btr_track,
        target_azimuths_deg=np.array([float(source_spec.azimuth_deg) for source_spec in source_specs], dtype=np.float64),
        title="BTR: Fractional-delay beam-time record",
        caption=btr_caption,
        output_path=output_dir / "btr.png",
    )

    sensor_positions_x = np.sort(array_positions_m[:, 0])
    sensor_spacings_m = np.diff(sensor_positions_x)
    summary: dict[str, object] = {
        "fs_hz": float(config.fs_hz),
        "duration_s": float(config.duration_s),
        "sound_speed_m_s": float(config.sound_speed_m_s),
        "noise_level_db20": float(config.noise_level_db20),
        "random_seed": int(config.random_seed),
        "n_source": int(len(source_specs)),
        "source_metrics": source_metrics,
        "array_geometry_name": array_geometry_name,
        "array_is_sparse": bool(array_is_sparse),
        "array_n_ch": int(array_positions_m.shape[0]),
        "array_sensor_spacing_unit_m": float(config.array_sensor_spacing_m),
        "array_aperture_m": float(sensor_positions_x[-1] - sensor_positions_x[0]),
        "array_min_sensor_spacing_m": float(np.min(sensor_spacings_m)) if sensor_spacings_m.size > 0 else 0.0,
        "array_max_sensor_spacing_m": float(np.max(sensor_spacings_m)) if sensor_spacings_m.size > 0 else 0.0,
        "n_ch": int(array_positions_m.shape[0]),
        "n_beam": int(beam_output.shape[0]),
        "fractional_delay_filter_bank_path": str(Path(fractional_delay_filter_bank_path).resolve()),
        "fraz_global_peak_azimuth_deg": fraz_global_peak_azimuth_deg,
        "fraz_global_peak_frequency_hz": fraz_global_peak_frequency_hz,
        "fraz_global_peak_level_db20": fraz_global_peak_level_db20,
        "btr_global_peak_azimuth_mean_deg": float(np.mean(btr_peak_azimuths_deg)),
        "btr_global_peak_azimuth_std_deg": float(np.std(btr_peak_azimuths_deg)),
        "fraz_png_path": str((output_dir / "fraz.png").resolve()),
        "btr_png_path": str((output_dir / "btr.png").resolve()),
        "plot_usage_notes_path": str((output_dir / "plot_usage_notes.md").resolve()),
    }

    if len(source_metrics) == 1:
        source_metric = source_metrics[0]
        summary.update(
            {
                "source_frequency_hz": float(source_metric["source_frequency_hz"]),
                "source_level_db20": float(source_metric["source_level_db20"]),
                "source_azimuth_deg": float(source_metric["source_azimuth_deg"]),
                "source_elevation_deg": float(source_metric["source_elevation_deg"]),
                "nearest_beam_azimuth_deg": float(source_metric["nearest_beam_azimuth_deg"]),
                "bl_peak_azimuth_deg": float(source_metric["bl_peak_azimuth_deg"]),
                "bl_peak_level_db20": float(source_metric["bl_peak_level_db20"]),
                "fraz_peak_azimuth_deg": float(source_metric["bl_peak_azimuth_deg"]),
                "fraz_peak_frequency_hz": float(source_metric["source_frequency_hz"]),
                "fraz_peak_level_db20": float(source_metric["bl_peak_level_db20"]),
                "fraz_level_at_nearest_source_grid_db20": float(source_metric["bl_level_at_nearest_source_grid_db20"]),
                "btr_mean_peak_azimuth_deg": float(np.mean(btr_peak_azimuths_deg)),
                "btr_peak_azimuth_std_deg": float(np.std(btr_peak_azimuths_deg)),
                "mirror_azimuth_deg": float(source_metric["mirror_azimuth_deg"]),
                "mirror_level_db20": float(source_metric["mirror_level_db20"]),
                "bl_png_path": str(source_metric["bl_png_path"]),
            }
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary, beamformer, beam_output, axis_az_deg, source_specs, array_positions_m


def run_fractional_delay_slc_diagnostics(
    config: TimeDelayDiagnosticConfig,
    slc_config: SlcConfig,
    fractional_delay_filter_bank_path: str | Path,
    *,
    target_source_indices: tuple[int, ...] | None = None,
    slc_analysis_block_size: int = 64,
    max_reference_beams: int = 48,
) -> dict[str, object]:
    """小数遅延固定整相後段へ周波数選択 SLC を適用した BL/FRAZ/BTR 診断を実行する。"""
    require_matplotlib()
    if slc_analysis_block_size <= 0:
        raise ValueError("slc_analysis_block_size must be positive.")
    if max_reference_beams <= 0:
        raise ValueError("max_reference_beams must be positive.")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_summary, beamformer, fixed_beam_output, axis_az_deg, source_specs, _ = _run_fractional_delay_diagnostics(
        config=config,
        fractional_delay_filter_bank_path=fractional_delay_filter_bank_path,
    )

    protected_target_indices = _resolve_target_source_indices(source_specs, target_source_indices)
    protected_target_index_set = {int(source_index) for source_index in protected_target_indices}
    target_source_specs = tuple(source_specs[int(source_index)] for source_index in protected_target_indices)
    interference_source_specs = tuple(
        source_spec for source_index, source_spec in enumerate(source_specs) if int(source_index) not in protected_target_index_set
    )

    slc_beam_output = np.array(fixed_beam_output, copy=True)
    per_frequency_design_summaries: list[dict[str, object]] = []
    unique_target_frequencies_hz = np.unique(
        np.array([float(source_spec.frequency_hz) for source_spec in target_source_specs], dtype=np.float64)
    )

    for target_frequency_hz in unique_target_frequencies_hz:
        before_snapshots, trimmed_length = _build_tone_snapshots(
            real_signals=slc_beam_output,
            frequency_hz=float(target_frequency_hz),
            fs_hz=float(config.fs_hz),
            block_size=int(slc_analysis_block_size),
        )
        response_matrix = _build_fractional_beam_response_matrix(
            beamformer=beamformer,
            frequency_hz=float(target_frequency_hz),
        )
        after_snapshots, tone_design_summary = _apply_frequency_selective_scan_slc(
            tone_snapshots=before_snapshots,
            response_matrix=response_matrix,
            slc_config=slc_config,
            max_reference_beams=int(max_reference_beams),
        )
        before_tone = _synthesize_tone_from_snapshots(
            tone_snapshots=before_snapshots,
            frequency_hz=float(target_frequency_hz),
            fs_hz=float(config.fs_hz),
            block_size=int(slc_analysis_block_size),
            trimmed_length=int(trimmed_length),
        )
        after_tone = _synthesize_tone_from_snapshots(
            tone_snapshots=after_snapshots,
            frequency_hz=float(target_frequency_hz),
            fs_hz=float(config.fs_hz),
            block_size=int(slc_analysis_block_size),
            trimmed_length=int(trimmed_length),
        )

        # 固定整相出力の対象周波数トーンだけを SLC 後へ差し替えることで、
        # 小数遅延で揃えた mainlobe を保ったまま、その周波数成分に限った sidelobe キャンセル量を評価する。
        slc_beam_output[:, :trimmed_length] += after_tone - before_tone
        tone_design_summary["frequency_hz"] = float(target_frequency_hz)
        tone_design_summary["trimmed_length"] = int(trimmed_length)
        per_frequency_design_summaries.append(tone_design_summary)

    fixed_btr_levels_db20, _ = _compress_time_rms_levels(
        fixed_beam_output,
        fs_hz=float(config.fs_hz),
        block_size=int(config.btr_block_size),
    )
    fixed_btr_relative_levels_db = fixed_btr_levels_db20 - np.max(fixed_btr_levels_db20, axis=1, keepdims=True)

    slc_freqs_hz, slc_fraz_levels_db20 = _rfft_levels_db20(slc_beam_output, fs_hz=float(config.fs_hz))
    slc_btr_levels_db20, slc_btr_times_s = _compress_time_rms_levels(
        slc_beam_output,
        fs_hz=float(config.fs_hz),
        block_size=int(config.btr_block_size),
    )
    slc_btr_relative_levels_db = slc_btr_levels_db20 - np.max(slc_btr_levels_db20, axis=1, keepdims=True)
    slc_btr_peak_beam_indices = np.argmax(slc_btr_levels_db20, axis=1)
    slc_btr_peak_azimuths_deg = axis_az_deg[slc_btr_peak_beam_indices]

    evaluation_half_width_beam = max(1, int(slc_config.guard) + 1)
    fixed_source_metrics, fixed_bl_curves = _evaluate_stage_source_metrics(
        beam_output=fixed_beam_output,
        axis_az_deg=axis_az_deg,
        btr_relative_levels_db=fixed_btr_relative_levels_db,
        source_specs=target_source_specs,
        fs_hz=float(config.fs_hz),
        output_dir=output_dir,
        stage_prefix="fixed_eval",
        bl_title_prefix="Fractional fixed BL evaluation",
        local_half_width_beam=evaluation_half_width_beam,
        save_bl=False,
    )
    slc_source_metrics, slc_bl_curves = _evaluate_stage_source_metrics(
        beam_output=slc_beam_output,
        axis_az_deg=axis_az_deg,
        btr_relative_levels_db=slc_btr_relative_levels_db,
        source_specs=target_source_specs,
        fs_hz=float(config.fs_hz),
        output_dir=output_dir,
        stage_prefix="slc",
        bl_title_prefix="Fractional fixed + SLC BL evaluation",
        local_half_width_beam=evaluation_half_width_beam,
        save_bl=True,
    )

    if interference_source_specs:
        # SLC の目的は protected target の維持だけではなく、guard 外 interferer の低下である。
        # source 周波数が target と同一の場合もあるため、周波数分離ではなく interferer 方位近傍の BL 指標として評価する。
        fixed_interference_source_metrics, _ = _evaluate_stage_source_metrics(
            beam_output=fixed_beam_output,
            axis_az_deg=axis_az_deg,
            btr_relative_levels_db=fixed_btr_relative_levels_db,
            source_specs=interference_source_specs,
            fs_hz=float(config.fs_hz),
            output_dir=output_dir,
            stage_prefix="fixed_interference_eval",
            bl_title_prefix="Fractional fixed interference BL evaluation",
            local_half_width_beam=evaluation_half_width_beam,
            save_bl=False,
        )
        slc_interference_source_metrics, _ = _evaluate_stage_source_metrics(
            beam_output=slc_beam_output,
            axis_az_deg=axis_az_deg,
            btr_relative_levels_db=slc_btr_relative_levels_db,
            source_specs=interference_source_specs,
            fs_hz=float(config.fs_hz),
            output_dir=output_dir,
            stage_prefix="slc_interference",
            bl_title_prefix="Fractional fixed + SLC interference BL evaluation",
            local_half_width_beam=evaluation_half_width_beam,
            save_bl=False,
        )
    else:
        fixed_interference_source_metrics = []
        slc_interference_source_metrics = []

    for source_index, source_spec in enumerate(target_source_specs):
        label = _source_label(source_spec, source_index)
        if len(target_source_specs) == 1:
            compare_output_path = output_dir / "slc_bl_compare.png"
        else:
            compare_output_path = output_dir / f"slc_bl_compare_{source_index:02d}_{_sanitize_label_for_filename(label)}.png"

        plot_bl_comparison(
            axis_az_deg=axis_az_deg,
            before_levels_db20=fixed_bl_curves[source_index],
            after_levels_db20=slc_bl_curves[source_index],
            target_azimuth_deg=float(source_spec.azimuth_deg),
            before_peak_azimuth_deg=float(fixed_source_metrics[source_index]["peak_azimuth_deg"]),
            after_peak_azimuth_deg=float(slc_source_metrics[source_index]["peak_azimuth_deg"]),
            title=f"BL comparison before/after SLC ({label}, {float(source_spec.frequency_hz):.1f} Hz)",
            caption=(
                f"mainlobe delta={float(slc_source_metrics[source_index]['level_at_nearest_source_grid_db20']) - float(fixed_source_metrics[source_index]['level_at_nearest_source_grid_db20']):.2f} dB, "
                f"sidelobe reduction={float(fixed_source_metrics[source_index]['max_nonlocal_level_db20']) - float(slc_source_metrics[source_index]['max_nonlocal_level_db20']):.2f} dB"
            ),
            output_path=compare_output_path,
        )
        slc_source_metrics[source_index]["bl_compare_png_path"] = str(compare_output_path.resolve())

    all_source_caption_fragment = _format_source_caption_fragment(source_specs)
    target_caption_fragment = _format_source_caption_fragment(target_source_specs)
    interference_caption_fragment = _format_source_caption_fragment(interference_source_specs) if interference_source_specs else "none"
    slc_target_points = [
        (
            float(source_spec.azimuth_deg),
            float(source_spec.frequency_hz),
            f"Target {source_index + 1}",
        )
        for source_index, source_spec in enumerate(target_source_specs)
    ]
    slc_peak_points = [
        (
            float(slc_source_metrics[source_index]["peak_azimuth_deg"]),
            float(source_spec.frequency_hz),
            f"SLC peak {source_index + 1}",
        )
        for source_index, source_spec in enumerate(target_source_specs)
    ]
    plot_fraz_heatmap(
        axis_az_deg=axis_az_deg,
        freqs_hz=slc_freqs_hz,
        fraz_levels_db20=slc_fraz_levels_db20,
        target_points=slc_target_points,
        peak_points=slc_peak_points,
        title="FRAZ after SLC (fractional fixed)",
        caption=(
            f"protected targets={target_caption_fragment}, interferers={interference_caption_fragment}, "
            f"all sources={all_source_caption_fragment}"
        ),
        output_path=output_dir / "slc_fraz.png",
    )

    if len(target_source_specs) == 1:
        slc_btr_caption = (
            f"SLC後 BTR: protected target={target_caption_fragment}, "
            f"mean peak azimuth={float(np.mean(slc_btr_peak_azimuths_deg)):.2f} deg"
        )
        slc_btr_track = slc_btr_peak_azimuths_deg
    else:
        slc_btr_caption = (
            f"SLC後 BTR: protected targets={target_caption_fragment}, interferers={interference_caption_fragment}"
        )
        slc_btr_track = None

    plot_btr_heatmap(
        axis_az_deg=axis_az_deg,
        times_s=slc_btr_times_s,
        btr_relative_levels_db=slc_btr_relative_levels_db,
        btr_peak_azimuths_deg=slc_btr_track,
        target_azimuths_deg=np.array([float(source_spec.azimuth_deg) for source_spec in target_source_specs], dtype=np.float64),
        title="BTR after SLC (fractional fixed)",
        caption=slc_btr_caption,
        output_path=output_dir / "slc_btr.png",
    )

    source_comparisons = _build_source_comparisons(fixed_source_metrics, slc_source_metrics)
    interference_source_comparisons = _build_interference_source_comparisons(
        fixed_interference_source_metrics,
        slc_interference_source_metrics,
    )
    aggregate_design_summary: dict[str, object] = {
        "analysis_block_size": int(slc_analysis_block_size),
        "max_reference_beams": int(max_reference_beams),
        "protected_target_indices": [int(source_index) for source_index in protected_target_indices],
        "protected_target_labels": [str(_source_label(source_specs[int(source_index)], int(source_index))) for source_index in protected_target_indices],
        "frequencies_hz": [float(frequency_hz) for frequency_hz in unique_target_frequencies_hz],
        "normal_beam_count": int(np.sum([int(summary["normal_beam_count"]) for summary in per_frequency_design_summaries])),
        "limited_beam_count": int(np.sum([int(summary["limited_beam_count"]) for summary in per_frequency_design_summaries])),
        "disabled_beam_count": int(np.sum([int(summary["disabled_beam_count"]) for summary in per_frequency_design_summaries])),
        "mean_selected_reference_beams": float(np.mean([float(summary["mean_selected_reference_beams"]) for summary in per_frequency_design_summaries])) if per_frequency_design_summaries else 0.0,
        "per_frequency": per_frequency_design_summaries,
    }

    combined_summary: dict[str, object] = {
        "fixed_summary_path": str((output_dir / "summary.json").resolve()),
        "fixed_summary": fixed_summary,
        "fractional_delay_filter_bank_path": str(Path(fractional_delay_filter_bank_path).resolve()),
        "target_source_indices": [int(source_index) for source_index in protected_target_indices],
        "target_source_labels": [str(_source_label(source_specs[int(source_index)], int(source_index))) for source_index in protected_target_indices],
        "interference_source_labels": [
            str(_source_label(source_spec, source_index)) for source_index, source_spec in enumerate(interference_source_specs)
        ],
        "slc_config": {
            "guard": int(slc_config.guard),
            "loading": float(slc_config.loading),
            "memory_time_sec": float(slc_config.memory_time_sec),
            "heading_scale_deg": float(slc_config.heading_scale_deg),
            "min_ref": int(slc_config.min_ref),
            "sample_per_dof": float(slc_config.sample_per_dof),
            "tap_len": int(slc_config.tap_len),
            "eta_normal": float(slc_config.eta_normal),
            "eta_limited": float(slc_config.eta_limited),
            "enable_heading_forgetting": bool(slc_config.enable_heading_forgetting),
        },
        "slc_design_summary": aggregate_design_summary,
        "slc_source_metrics": slc_source_metrics,
        "fixed_interference_source_metrics": fixed_interference_source_metrics,
        "slc_interference_source_metrics": slc_interference_source_metrics,
        "slc_fraz_png_path": str((output_dir / "slc_fraz.png").resolve()),
        "slc_btr_png_path": str((output_dir / "slc_btr.png").resolve()),
        "source_comparisons": source_comparisons,
        "interference_source_comparisons": interference_source_comparisons,
        "all_mainlobes_preserved": bool(all(bool(comparison["mainlobe_preserved"]) for comparison in source_comparisons)),
        "mean_mainlobe_level_delta_db": float(np.mean([float(comparison["mainlobe_level_delta_db"]) for comparison in source_comparisons])),
        "mean_mainlobe_margin_improvement_db": float(np.mean([float(comparison["mainlobe_margin_improvement_db"]) for comparison in source_comparisons])),
        "mean_local_margin_improvement_db": float(np.mean([float(comparison["local_margin_improvement_db"]) for comparison in source_comparisons])),
        "mean_sidelobe_reduction_db": float(np.mean([float(comparison["sidelobe_reduction_db"]) for comparison in source_comparisons])),
        "mean_nominal_error_improvement_db": float(np.mean([float(comparison["nominal_error_improvement_db"]) for comparison in source_comparisons])),
        "mean_mirror_reduction_db": float(np.mean([float(comparison["mirror_reduction_db"]) for comparison in source_comparisons])),
        "slc_btr_global_peak_azimuth_mean_deg": float(np.mean(slc_btr_peak_azimuths_deg)),
        "slc_btr_global_peak_azimuth_std_deg": float(np.std(slc_btr_peak_azimuths_deg)),
    }
    (output_dir / "slc_summary.json").write_text(json.dumps(combined_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return combined_summary


__all__ = [
    "run_fractional_delay_slc_diagnostics",
]
