"""時間領域固定整相の BL/FRAZ/BTR 診断を行うモジュール。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .._validation import (
    require,
    require_non_negative_float,
    require_positive_float,
    require_positive_int,
)
from ..beamforming_evaluation.diagnostic_plotting import (
    build_beam_diagnostic_plot_usage_notes,
    plot_bl_response,
    plot_btr_heatmap,
    plot_fraz_heatmap,
    require_matplotlib,
    write_beam_diagnostic_plot_usage_notes,
)
from ..beamforming_evaluation.scan_grid import build_beam_scan_grid
from ..beamforming_evaluation.signal_levels import (
    calculate_block_rms_levels_db20,
    calculate_one_sided_rms_spectrum_db20,
    calculate_tone_projection_rms_level_db20,
)
from ..simulation.numerics import SimulationPrecision
from ..simulation.tone_scene import ToneSceneSource, synthesize_tone_scene
from .time_delay import IntegerDelayAndSumBeamformer


class TimeDelayDiagnosticSource(ToneSceneSource):
    """整数遅延固定整相の検証に使う単一トーン音源条件を表す。

    このクラスは、到来方位、周波数、レベル、初期位相を持つ 1 本の平面波音源を表す。
    複数本の音源を `TimeDelayDiagnosticConfig.source_specs` へ並べることで、
    複数方位・複数周波数の固定整相評価シーンを構成する。

    入力は方位角、俯仰角、周波数、dB20 レベル、位相、必要に応じて
    包絡変調条件、任意ラベルであり、
    出力はシーン生成および source ごとの BL/FRAZ/BTR 指標計算に使う内部条件である。

    音源間の適応抑圧、SLC 係数更新、時変包絡制御は責務に含めない。
    信号処理上は、固定整相前段を検証するための理想平面波 source 記述子に位置づく。

    入力・出力・単位・境界条件は`ToneSceneSource`と同じである。
    この互換名は既存の診断設定から通常のPython importを保つためだけに残し、
    scene生成やbeamforming処理は責務に含めない。
    """


@dataclass(frozen=True)
class TimeDelayDiagnosticConfig:
    """整数遅延固定整相の BL/FRAZ/BTR 評価条件を保持する。

    このクラスは、単一音源または複数音源の時間領域固定整相シーン、
    ならびに uniform / sparse / 明示座標指定アレイの評価条件を保持する。

    入力はサンプリング周波数、音速、音源条件、走査ビーム数、出力先ディレクトリ、
    および必要に応じてアレイ座標であり、出力は `run_integer_delay_diagnostics()` が
    保存する BL, FRAZ, BTR の画像と summary JSON である。

    小数遅延 FIR の適用、SLC の適応更新、複数 target の保護領域設計は責務に含めない。
    信号処理上は、時間領域固定 Delay-and-Sum ビームフォーマ単体の診断条件を定義する。
    """

    output_dir: Path
    fs_hz: float = 32768.0
    duration_s: float = 1.0
    sound_speed_m_s: float = 1500.0
    source_frequency_hz: float = 1536.0
    source_level_db20: float = 0.0
    source_azimuth_deg: float = 20.0
    source_elevation_deg: float = 0.0
    source_phase_deg: float = 0.0
    source_specs: tuple[TimeDelayDiagnosticSource, ...] | None = None
    noise_level_db20: float = -40.0
    random_seed: int = 1234
    array_n_ch: int = 160
    array_sensor_spacing_m: float = 0.05
    sparse_stride_pattern: tuple[int, ...] | None = None
    array_positions_m: np.ndarray | None = None
    az_min_deg: float = 0.0
    az_max_deg: float = 180.0
    n_beam_az_real: int = 241
    n_beam_az_virtual: int = 0
    display_elevation_deg: float = 0.0
    btr_block_size: int = 1024


def _validate_config(config: TimeDelayDiagnosticConfig) -> None:
    """診断条件の範囲と単位系を検証する。"""
    require_positive_float("fs_hz", config.fs_hz)
    require_positive_float("duration_s", config.duration_s)
    require_positive_float("sound_speed_m_s", config.sound_speed_m_s)
    require_positive_float("source_frequency_hz", config.source_frequency_hz)
    require_non_negative_float("noise_level_db20", -float(config.noise_level_db20) + float(config.noise_level_db20))
    require_positive_int("array_n_ch", config.array_n_ch)
    require_positive_float("array_sensor_spacing_m", config.array_sensor_spacing_m)
    require_positive_int("n_beam_az_real", config.n_beam_az_real)
    require(config.n_beam_az_virtual >= 0, "n_beam_az_virtual must be non-negative.")
    require_positive_int("btr_block_size", config.btr_block_size)

    if config.sparse_stride_pattern is not None:
        stride_pattern = np.asarray(config.sparse_stride_pattern, dtype=np.int64)
        require(stride_pattern.ndim == 1 and stride_pattern.size > 0, "sparse_stride_pattern must be a non-empty 1-D sequence.")
        require(bool(np.all(stride_pattern > 0)), "sparse_stride_pattern must contain only positive integers.")
        require(
            int(config.array_n_ch) % 2 == 1,
            "array_n_ch must be odd when sparse_stride_pattern is used so that the sparse array keeps a center sensor.",
        )

    if config.array_positions_m is not None:
        positions = np.asarray(config.array_positions_m, dtype=np.float64)
        require(positions.ndim == 2 and positions.shape[1] == 3, "array_positions_m must have shape (n_ch, 3).")
        require(positions.shape[0] > 0, "array_positions_m must not be empty.")
        require(bool(np.all(np.isfinite(positions))), "array_positions_m must contain only finite values.")


def _normalize_explicit_array_positions(array_positions_m: np.ndarray) -> np.ndarray:
    """明示指定されたアレイ座標を shape `[n_ch, 3]` の `float64` 配列へ正規化する。"""
    positions = np.asarray(array_positions_m, dtype=np.float64)
    require(positions.ndim == 2 and positions.shape[1] == 3, "array_positions_m must have shape (n_ch, 3).")
    require(positions.shape[0] > 0, "array_positions_m must not be empty.")
    require(bool(np.all(np.isfinite(positions))), "array_positions_m must contain only finite values.")
    return positions


def build_sparse_single_side_array_positions(
    n_ch: int,
    sensor_spacing_unit_m: float,
    sparse_stride_pattern: tuple[int, ...],
) -> np.ndarray:
    """対称 1 列片舷スパースアレイ座標を構成する。

    Args:
        n_ch: チャネル数。shape 概念上はセンサ本数であり、奇数を要求する。
        sensor_spacing_unit_m: 最小格子間隔。単位は m。
        sparse_stride_pattern: 中心から外側へ進む整数格子ステップ列。
            例えば `(1, 2, 1, 3)` の場合、x 軸正側の格子位置は
            `1, 3, 4, 7, 8, 10, ...` のように周期反復で生成する。

    Returns:
        センサ位置。shape は `[n_ch, 3]`、単位は m。
        axis=0 がセンサ番号、axis=1 が `x, y, z` 座標である。

    Raises:
        ValueError: `n_ch` が奇数でない、または `sparse_stride_pattern` が不正な場合。
    """
    require_positive_int("n_ch", n_ch)
    require_positive_float("sensor_spacing_unit_m", sensor_spacing_unit_m)
    require(n_ch % 2 == 1, "n_ch must be odd for symmetric sparse side-array construction.")

    stride_pattern = np.asarray(sparse_stride_pattern, dtype=np.int64)
    require(stride_pattern.ndim == 1 and stride_pattern.size > 0, "sparse_stride_pattern must be a non-empty 1-D sequence.")
    require(bool(np.all(stride_pattern > 0)), "sparse_stride_pattern must contain only positive integers.")

    n_positive_sensor = n_ch // 2
    positive_indices: list[int] = []
    current_index = 0

    # sparse_stride_pattern を周期反復して中心から外側の格子位置を作る。
    # 非一様な整数格子間隔にすることで、等間隔 ULA よりグレーティングローブの周期性を崩す。
    while len(positive_indices) < n_positive_sensor:
        current_index += int(stride_pattern[len(positive_indices) % stride_pattern.size])
        positive_indices.append(current_index)

    mirrored_indices = [-grid_index for grid_index in reversed(positive_indices)]
    sensor_indices = np.array(mirrored_indices + [0] + positive_indices, dtype=np.float64)

    positions = np.zeros((n_ch, 3), dtype=np.float64)
    positions[:, 0] = sensor_indices * float(sensor_spacing_unit_m)
    return positions


def _build_uniform_single_side_array_positions(config: TimeDelayDiagnosticConfig) -> np.ndarray:
    """整数遅延整相に合わせた 1 列片舷 ULA 座標を返す。"""
    centered_positions_x = (
        np.arange(int(config.array_n_ch), dtype=np.float64) - 0.5 * (int(config.array_n_ch) - 1)
    ) * float(config.array_sensor_spacing_m)
    positions = np.zeros((int(config.array_n_ch), 3), dtype=np.float64)
    positions[:, 0] = centered_positions_x
    return positions


def _build_array_positions(config: TimeDelayDiagnosticConfig) -> tuple[np.ndarray, str, bool]:
    """診断条件から評価対象アレイ座標を決定する。"""
    if config.array_positions_m is not None:
        positions = _normalize_explicit_array_positions(config.array_positions_m)
        sensor_positions_x = positions[:, 0]
        spacing_steps = np.diff(np.sort(sensor_positions_x))
        is_sparse = bool(spacing_steps.size > 1 and not np.allclose(spacing_steps, spacing_steps[0], atol=1e-12))
        return positions, "explicit_array_positions", is_sparse

    if config.sparse_stride_pattern is not None:
        return (
            build_sparse_single_side_array_positions(
                n_ch=int(config.array_n_ch),
                sensor_spacing_unit_m=float(config.array_sensor_spacing_m),
                sparse_stride_pattern=tuple(int(step) for step in config.sparse_stride_pattern),
            ),
            "generated_sparse_single_side",
            True,
        )

    return _build_uniform_single_side_array_positions(config), "uniform_single_side", False


def _resolve_source_specs(config: TimeDelayDiagnosticConfig) -> tuple[TimeDelayDiagnosticSource, ...]:
    """legacy 単一 source 設定と複数 source 設定を統一形式へ揃える。"""
    if config.source_specs is not None:
        require(len(config.source_specs) > 0, "source_specs must not be empty when provided.")
        return tuple(config.source_specs)

    return (
        TimeDelayDiagnosticSource(
            azimuth_deg=float(config.source_azimuth_deg),
            elevation_deg=float(config.source_elevation_deg),
            frequency_hz=float(config.source_frequency_hz),
            level_db20=float(config.source_level_db20),
            phase_deg=float(config.source_phase_deg),
            label="source_00",
        ),
    )


def _source_label(source_spec: TimeDelayDiagnosticSource, source_index: int) -> str:
    """source ごとの表示・保存に使う安定ラベルを返す。"""
    if source_spec.label is not None:
        return str(source_spec.label)
    return f"source_{source_index:02d}"


def _sanitize_label_for_filename(label: str) -> str:
    """ファイル名へ使いやすい ASCII 断片へ整形する。"""
    sanitized = [character if character.isalnum() or character in ("-", "_") else "_" for character in label]
    stem = "".join(sanitized).strip("_")
    return stem or "source"


def _format_source_caption_fragment(source_specs: tuple[TimeDelayDiagnosticSource, ...]) -> str:
    """複数 source を短いキャプション文字列へ整形する。"""
    return ", ".join(
        [
            f"{_source_label(source_spec, source_index)}=({float(source_spec.azimuth_deg):.1f} deg, {float(source_spec.frequency_hz):.1f} Hz)"
            for source_index, source_spec in enumerate(source_specs)
        ]
    )


def _evaluate_source_metrics_and_save_bl(
    beam_output: np.ndarray,
    axis_az_deg: np.ndarray,
    btr_relative_levels_db: np.ndarray,
    source_specs: tuple[TimeDelayDiagnosticSource, ...],
    config: TimeDelayDiagnosticConfig,
    output_dir: Path,
) -> list[dict[str, float | str]]:
    """source ごとの BL 指標を計算し、周波数別 BL 図を保存する。"""
    source_metrics: list[dict[str, float | str]] = []

    for source_index, source_spec in enumerate(source_specs):
        beam_levels_db20 = np.array(
            [
                calculate_tone_projection_rms_level_db20(
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
            title=f"BL: Integer-delay beam response ({label}, {float(source_spec.frequency_hz):.1f} Hz)",
            caption=(
                f"source={float(source_spec.frequency_hz):.1f} Hz, target azimuth={float(source_spec.azimuth_deg):.2f} deg, "
                f"peak azimuth={float(axis_az_deg[peak_beam_index]):.2f} deg"
            ),
            output_path=bl_output_path,
            response_label=f"Integer delay beam response ({label})",
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


def run_integer_delay_diagnostics(config: TimeDelayDiagnosticConfig) -> dict[str, Any]:
    """整数遅延固定整相の BL/FRAZ/BTR を画像保存し、ピーク位置を要約する。

    Args:
        config: 診断条件。画像保存先、音源条件、走査ビーム数、必要なら sparse 配置を含む。

    Returns:
        summary 辞書。単一音源時は従来互換の top-level キーを含み、
        複数音源時は `source_metrics` に source ごとの BL/BTR 指標を格納する。

    Raises:
        RuntimeError: matplotlib が使えず画像を保存できない場合。
        ValueError: 入力条件が不正な場合。
    """
    require_matplotlib()
    _validate_config(config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # BL/FRAZ/BTR は表示解釈を誤ると診断結果の読み違いに直結するため、
    # 各 run で注意事項を図と同じディレクトリへ保存し、再利用時の契約を明示する。
    plot_usage_notes = build_beam_diagnostic_plot_usage_notes()
    write_beam_diagnostic_plot_usage_notes(output_dir / "plot_usage_notes.md", plot_usage_notes)

    source_specs = _resolve_source_specs(config)
    array_positions_m, array_geometry_name, array_is_sparse = _build_array_positions(config)
    beam_grid = build_beam_scan_grid(
        azimuth_min_deg=float(config.az_min_deg),
        azimuth_max_deg=float(config.az_max_deg),
        display_elevation_deg=float(config.display_elevation_deg),
        n_real_azimuth_beams=int(config.n_beam_az_real),
        n_virtual_azimuth_beams=int(config.n_beam_az_virtual),
    )
    scene = synthesize_tone_scene(
        array_positions_m=array_positions_m,
        sources=source_specs,
        fs_hz=float(config.fs_hz),
        duration_s=float(config.duration_s),
        sound_speed_m_s=float(config.sound_speed_m_s),
        noise_level_db20=float(config.noise_level_db20),
        random_seed=int(config.random_seed),
        precision=SimulationPrecision.SINGLE,
    )
    multichannel_signal = scene.signal

    beamformer = IntegerDelayAndSumBeamformer.from_geometry(
        array_pos_m=array_positions_m,
        dir_cos=beam_grid.directions,
        fs_hz=float(config.fs_hz),
        sound_speed_m_s=float(config.sound_speed_m_s),
    )
    beam_output = beamformer.process(multichannel_signal)
    # process はデバッグ用 steered channel を返す構成も型契約に含むため、診断で使う主出力を明示的に確定する。
    if isinstance(beam_output, tuple):
        beam_output = beam_output[0]

    axis_az_deg = beam_grid.azimuth_deg
    freqs_hz, fraz_levels_db20 = calculate_one_sided_rms_spectrum_db20(
        beam_output,
        fs_hz=float(config.fs_hz),
    )

    fraz_global_peak_beam_index, fraz_global_peak_frequency_index = np.unravel_index(
        np.argmax(fraz_levels_db20),
        fraz_levels_db20.shape,
    )
    fraz_global_peak_azimuth_deg = float(axis_az_deg[int(fraz_global_peak_beam_index)])
    fraz_global_peak_frequency_hz = float(freqs_hz[int(fraz_global_peak_frequency_index)])
    fraz_global_peak_level_db20 = float(
        fraz_levels_db20[int(fraz_global_peak_beam_index), int(fraz_global_peak_frequency_index)]
    )

    btr_levels_db20, btr_times_s = calculate_block_rms_levels_db20(
        beam_output,
        fs_hz=float(config.fs_hz),
        block_size=int(config.btr_block_size),
    )

    # BTR は各時刻で最大ビームを 0 dB に正規化した相対表示にし、
    # 複数音源時でも各 target 方位の ridge を比較しやすくする。
    btr_relative_levels_db = btr_levels_db20 - np.max(btr_levels_db20, axis=1, keepdims=True)
    btr_peak_beam_indices = np.argmax(btr_levels_db20, axis=1)
    btr_peak_azimuths_deg = axis_az_deg[btr_peak_beam_indices]

    source_metrics = _evaluate_source_metrics_and_save_bl(
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
        title="FRAZ: Frequency-azimuth response",
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
        title="BTR: Beam-time record",
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
        "fraz_global_peak_azimuth_deg": fraz_global_peak_azimuth_deg,
        "fraz_global_peak_frequency_hz": fraz_global_peak_frequency_hz,
        "fraz_global_peak_level_db20": fraz_global_peak_level_db20,
        "btr_global_peak_azimuth_mean_deg": float(np.mean(btr_peak_azimuths_deg)),
        "btr_global_peak_azimuth_std_deg": float(np.std(btr_peak_azimuths_deg)),
        "fraz_png_path": str((output_dir / "fraz.png").resolve()),
        "btr_png_path": str((output_dir / "btr.png").resolve()),
        "plot_usage_notes_path": str((output_dir / "plot_usage_notes.md").resolve()),
    }

    # 従来の単一音源診断 API との互換を維持するため、source が 1 本のときは
    # 既存テストが参照している top-level キーへ代表値を展開する。
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
    return summary
