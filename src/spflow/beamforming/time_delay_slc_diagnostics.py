"""時間領域固定整相の後段へ周波数選択 SLC を適用した BL/FRAZ/BTR 診断を行うモジュール。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .diagnostic_plotting import plot_bl_comparison, plot_btr_heatmap, plot_fraz_heatmap, require_matplotlib
from .slc import BeamGuardSelector, BlockLeastSquaresSlcSolver, SlcConfig, SlcReferenceCapacityChecker
from .time_delay import IntegerDelayAndSumBeamformer
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
    run_integer_delay_diagnostics,
)


def _resolve_target_source_indices(
    source_specs: tuple[TimeDelayDiagnosticSource, ...],
    target_source_indices: tuple[int, ...] | None,
) -> np.ndarray:
    """SLC で保護する target source index を一意化して返す。"""
    if target_source_indices is None:
        return np.array([0], dtype=np.int64)

    resolved_indices = np.unique(np.asarray(target_source_indices, dtype=np.int64))
    if resolved_indices.ndim != 1 or resolved_indices.size == 0:
        raise ValueError("target_source_indices must be a non-empty 1-D sequence.")
    if np.any((resolved_indices < 0) | (resolved_indices >= len(source_specs))):
        raise ValueError("target_source_indices contain out-of-range entries.")
    return resolved_indices.astype(np.int64)


def _build_tone_snapshots(
    real_signals: np.ndarray,
    frequency_hz: float,
    fs_hz: float,
    block_size: int,
) -> tuple[np.ndarray, int]:
    """単一周波数の block 複素係数列を返す。"""
    beam_signals = np.asarray(real_signals, dtype=np.float64)
    if beam_signals.ndim != 2:
        raise ValueError("real_signals must have shape (n_beam, n_sample).")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    trimmed_length = (beam_signals.shape[1] // int(block_size)) * int(block_size)
    if trimmed_length <= 0:
        raise ValueError("real_signals must contain at least one full SLC analysis block.")

    n_snapshot = trimmed_length // int(block_size)

    # reshaped shape: [n_beam, n_snapshot, block_size]
    # axis=1 を SLC の snapshot 軸、axis=2 を 1 snapshot 内の時間サンプル軸として扱う。
    reshaped = beam_signals[:, :trimmed_length].reshape(beam_signals.shape[0], n_snapshot, int(block_size))
    block_time_s = np.arange(int(block_size), dtype=np.float64) / float(fs_hz)
    reference = np.exp(-1j * 2.0 * np.pi * float(frequency_hz) * block_time_s)[None, None, :]

    # 各 block で e^{-j2πft} と複素内積を取ることで、当該トーンの複素振幅 snapshot を取り出す。
    # 同一周波数 source でも包絡変調が異なれば snapshot 間で振幅が変化し、SLC の block 共分散に反映される。
    tone_snapshots = np.mean(reshaped * reference, axis=2)
    return tone_snapshots.astype(np.complex128), int(trimmed_length)


def _synthesize_tone_from_snapshots(
    tone_snapshots: np.ndarray,
    frequency_hz: float,
    fs_hz: float,
    block_size: int,
    trimmed_length: int,
) -> np.ndarray:
    """block 複素係数列から実数トーン波形を再合成する。"""
    snapshots = np.asarray(tone_snapshots, dtype=np.complex128)
    if snapshots.ndim != 2:
        raise ValueError("tone_snapshots must have shape (n_beam, n_snapshot).")
    if trimmed_length != snapshots.shape[1] * int(block_size):
        raise ValueError("trimmed_length must equal n_snapshot * block_size.")

    n_beam, n_snapshot = snapshots.shape
    synthesized = np.zeros((n_beam, trimmed_length), dtype=np.float64)

    for snapshot_index in range(n_snapshot):
        sample_start = snapshot_index * int(block_size)
        sample_stop = sample_start + int(block_size)
        time_axis_s = np.arange(sample_start, sample_stop, dtype=np.float64) / float(fs_hz)

        # x[n] = 2 Re{a_k exp(j2πft)}。
        # snapshot 係数 a_k は複素片側振幅に相当するため、実トーンへ戻すときは 2 倍して実部を取る。
        synthesized[:, sample_start:sample_stop] = 2.0 * np.real(
            snapshots[:, snapshot_index][:, None] * np.exp(1j * 2.0 * np.pi * float(frequency_hz) * time_axis_s)[None, :]
        )
    return synthesized


def _build_beam_response_matrix(
    beamformer: IntegerDelayAndSumBeamformer,
    frequency_hz: float,
    fs_hz: float,
) -> np.ndarray:
    """固定整相の理論ビーム応答行列を返す。"""
    arrival_delay_sec = np.asarray(beamformer.delay_table.arrival_delay_sec, dtype=np.float64)
    steering_delay_sec = np.asarray(beamformer.delay_table.delay_int, dtype=np.float64) / float(fs_hz)

    # steering_phase[ch, beam_obs] と arrival_phase[ch, beam_look] の積をセンサ平均することで、
    # look 方向ごとの observation beam 複素応答を一括生成する。
    steering_phase = np.exp(-1j * 2.0 * np.pi * float(frequency_hz) * steering_delay_sec)
    arrival_phase = np.exp(-1j * 2.0 * np.pi * float(frequency_hz) * arrival_delay_sec)
    return (steering_phase.T @ arrival_phase) / float(arrival_delay_sec.shape[0])


def _select_reference_beams(
    raw_reference_beams: np.ndarray,
    n_snapshot: int,
    slc_config: SlcConfig,
    max_reference_beams: int,
) -> tuple[np.ndarray, bool]:
    """snapshot 数と計算量制約から参照ビームを間引く。"""
    raw_indices = np.asarray(raw_reference_beams, dtype=np.int64)
    if raw_indices.ndim != 1:
        raise ValueError("raw_reference_beams must be a 1-D array.")
    if raw_indices.size == 0:
        return raw_indices, False
    if max_reference_beams <= 0:
        raise ValueError("max_reference_beams must be positive.")

    # guard を狭める代わりに参照を等方位間隔で間引く。
    # これにより target mainlobe 保護幅は維持したまま、snapshot 数に対して過大な自由度を抑える。
    max_from_snapshot = int(np.floor(float(n_snapshot) / (float(slc_config.sample_per_dof) * float(slc_config.tap_len))))
    allowed_reference_beams = min(int(raw_indices.size), int(max_reference_beams), int(max_from_snapshot))
    if allowed_reference_beams < int(slc_config.min_ref):
        return np.empty(0, dtype=np.int64), True
    if allowed_reference_beams >= raw_indices.size:
        return raw_indices, False

    selected_positions = np.linspace(0, raw_indices.size - 1, allowed_reference_beams, dtype=np.int64)
    return raw_indices[selected_positions], True


def _apply_frequency_selective_scan_slc(
    tone_snapshots: np.ndarray,
    response_matrix: np.ndarray,
    slc_config: SlcConfig,
    max_reference_beams: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """scan 全ビームへ周波数選択 SLC を掛けた複素 snapshot 列を返す。"""
    snapshots = np.asarray(tone_snapshots, dtype=np.complex128)
    if snapshots.ndim != 2:
        raise ValueError("tone_snapshots must have shape (n_beam, n_snapshot).")
    if response_matrix.shape != (snapshots.shape[0], snapshots.shape[0]):
        raise ValueError("response_matrix must have shape (n_beam, n_beam).")

    n_beam, n_snapshot = snapshots.shape
    guard_selector = BeamGuardSelector(n_beam=n_beam, guard=int(slc_config.guard))
    capacity_checker = SlcReferenceCapacityChecker(
        min_ref=int(slc_config.min_ref),
        sample_per_dof=float(slc_config.sample_per_dof),
        tap_len=int(slc_config.tap_len),
    )
    solver = BlockLeastSquaresSlcSolver(loading=float(slc_config.loading))

    slc_snapshots = np.array(snapshots, copy=True)
    selected_reference_counts: list[int] = []
    raw_reference_counts: list[int] = []
    mode_counts = {
        "NORMAL": 0,
        "LIMITED_REFERENCE": 0,
        "DISABLED": 0,
    }

    for beam_index in range(n_beam):
        raw_reference_beams = guard_selector.make_reference_beams(np.array([beam_index], dtype=np.int64))
        selected_reference_beams, is_limited = _select_reference_beams(
            raw_reference_beams=raw_reference_beams,
            n_snapshot=int(n_snapshot),
            slc_config=slc_config,
            max_reference_beams=int(max_reference_beams),
        )
        raw_reference_counts.append(int(raw_reference_beams.size))
        selected_reference_counts.append(int(selected_reference_beams.size))

        capacity = capacity_checker.check(n_ref=int(selected_reference_beams.size), block_size=int(n_snapshot))
        if not capacity.is_feasible:
            mode_counts["DISABLED"] += 1
            continue

        desired_response = np.asarray(response_matrix[selected_reference_beams, beam_index], dtype=np.complex128)
        desired_response_norm_sq = float(np.real(np.vdot(desired_response, desired_response)))
        if desired_response_norm_sq <= 1.0e-12:
            # look 方向の理論応答がほぼゼロなら blocking が定義できない。
            # この条件で無理に更新すると任意の位相基準で自己消去し得るため、安全側で固定整相を維持する。
            mode_counts["DISABLED"] += 1
            continue

        # P = I - aa^H / (a^H a) により、look 方向の理論応答 a を参照空間から射影除去する。
        # これが beam-domain の簡易 blocking matrix に相当し、desired 成分の self-nulling を抑える。
        projector = np.eye(selected_reference_beams.size, dtype=np.complex128) - (
            np.outer(desired_response, np.conj(desired_response)) / desired_response_norm_sq
        )
        reference_snapshots = projector @ snapshots[selected_reference_beams, :]
        target_snapshots = snapshots[beam_index : beam_index + 1, :]

        covariance_matrix = (reference_snapshots @ reference_snapshots.conj().T) / float(n_snapshot)
        cross_correlation = ((reference_snapshots @ target_snapshots[0].conj()) / float(n_snapshot))[None, :]
        weights = solver.solve(R=covariance_matrix, r=cross_correlation)[0]
        eta = float(slc_config.eta_limited if is_limited else slc_config.eta_normal)
        slc_snapshots[beam_index, :] = target_snapshots[0] - eta * (np.conj(weights) @ reference_snapshots)
        mode_counts["LIMITED_REFERENCE" if is_limited else "NORMAL"] += 1

    design_summary: dict[str, object] = {
        "guard": int(slc_config.guard),
        "loading": float(slc_config.loading),
        "n_beam": int(n_beam),
        "n_snapshot": int(n_snapshot),
        "max_reference_beams": int(max_reference_beams),
        "min_ref": int(slc_config.min_ref),
        "sample_per_dof": float(slc_config.sample_per_dof),
        "tap_len": int(slc_config.tap_len),
        "normal_beam_count": int(mode_counts["NORMAL"]),
        "limited_beam_count": int(mode_counts["LIMITED_REFERENCE"]),
        "disabled_beam_count": int(mode_counts["DISABLED"]),
        "mean_selected_reference_beams": float(np.mean(selected_reference_counts)) if selected_reference_counts else 0.0,
        "min_selected_reference_beams": int(np.min(selected_reference_counts)) if selected_reference_counts else 0,
        "max_selected_reference_beams": int(np.max(selected_reference_counts)) if selected_reference_counts else 0,
        "mean_raw_reference_beams": float(np.mean(raw_reference_counts)) if raw_reference_counts else 0.0,
    }
    return slc_snapshots, design_summary


def _evaluate_stage_source_metrics(
    beam_output: np.ndarray,
    axis_az_deg: np.ndarray,
    btr_relative_levels_db: np.ndarray,
    source_specs: tuple[TimeDelayDiagnosticSource, ...],
    fs_hz: float,
    output_dir: Path,
    *,
    stage_prefix: str,
    bl_title_prefix: str,
    local_half_width_beam: int,
    save_bl: bool,
) -> tuple[list[dict[str, float | str]], list[np.ndarray]]:
    """指定 stage の source ごとの BL 指標と BL 曲線を返す。"""
    stage_metrics: list[dict[str, float | str]] = []
    beam_level_curves: list[np.ndarray] = []

    for source_index, source_spec in enumerate(source_specs):
        beam_levels_db20 = np.array(
            [
                _tone_level_db20_rms(
                    beam_output[beam_index],
                    frequency_hz=float(source_spec.frequency_hz),
                    fs_hz=float(fs_hz),
                )
                for beam_index in range(beam_output.shape[0])
            ],
            dtype=np.float64,
        )
        beam_level_curves.append(beam_levels_db20)

        nearest_beam_index = int(np.argmin(np.abs(axis_az_deg - float(source_spec.azimuth_deg))))
        local_start = max(0, nearest_beam_index - int(local_half_width_beam))
        local_stop = min(axis_az_deg.size, nearest_beam_index + int(local_half_width_beam) + 1)
        local_peak_offset = int(np.argmax(beam_levels_db20[local_start:local_stop]))
        local_peak_beam_index = int(local_start + local_peak_offset)
        mirror_beam_index = int(np.argmin(np.abs(axis_az_deg - (180.0 - float(source_spec.azimuth_deg)))))
        nonlocal_mask = np.ones(axis_az_deg.size, dtype=bool)

        # target 方位の近傍だけを mainlobe と見なし、その外側最大値を sidelobe 指標として分離する。
        nonlocal_mask[local_start:local_stop] = False
        max_nonlocal_level_db20 = float(np.max(beam_levels_db20[nonlocal_mask])) if np.any(nonlocal_mask) else float(beam_levels_db20[local_peak_beam_index])
        label = _source_label(source_spec, source_index)

        if len(source_specs) == 1:
            bl_output_path = output_dir / f"{stage_prefix}_bl.png"
        else:
            bl_output_path = output_dir / f"{stage_prefix}_bl_{source_index:02d}_{_sanitize_label_for_filename(label)}.png"

        if save_bl:
            from .diagnostic_plotting import plot_bl_response

            plot_bl_response(
                axis_az_deg=axis_az_deg,
                beam_levels_db20=beam_levels_db20,
                target_azimuth_deg=float(source_spec.azimuth_deg),
                peak_azimuth_deg=float(axis_az_deg[local_peak_beam_index]),
                title=f"{bl_title_prefix} ({label}, {float(source_spec.frequency_hz):.1f} Hz)",
                caption=(
                    f"source={float(source_spec.frequency_hz):.1f} Hz, target azimuth={float(source_spec.azimuth_deg):.2f} deg, "
                    f"local peak azimuth={float(axis_az_deg[local_peak_beam_index]):.2f} deg"
                ),
                output_path=bl_output_path,
                response_label=f"{bl_title_prefix} ({label})",
            )

        stage_metrics.append(
            {
                "label": label,
                "source_azimuth_deg": float(source_spec.azimuth_deg),
                "source_elevation_deg": float(source_spec.elevation_deg),
                "source_frequency_hz": float(source_spec.frequency_hz),
                "source_level_db20": float(source_spec.level_db20),
                "nearest_beam_azimuth_deg": float(axis_az_deg[nearest_beam_index]),
                "peak_azimuth_deg": float(axis_az_deg[local_peak_beam_index]),
                "peak_level_db20": float(beam_levels_db20[local_peak_beam_index]),
                "level_at_nearest_source_grid_db20": float(beam_levels_db20[nearest_beam_index]),
                "max_nonlocal_level_db20": max_nonlocal_level_db20,
                "local_to_nonlocal_margin_db": float(beam_levels_db20[nearest_beam_index] - max_nonlocal_level_db20),
                "mirror_azimuth_deg": float(axis_az_deg[mirror_beam_index]),
                "mirror_level_db20": float(beam_levels_db20[mirror_beam_index]),
                "btr_mean_relative_level_db": float(np.mean(btr_relative_levels_db[:, nearest_beam_index])),
                "btr_min_relative_level_db": float(np.min(btr_relative_levels_db[:, nearest_beam_index])),
                "nominal_level_error_db": float(beam_levels_db20[nearest_beam_index] - float(source_spec.level_db20)),
                "bl_png_path": str(bl_output_path.resolve()),
            }
        )

    return stage_metrics, beam_level_curves


def _build_source_comparisons(
    fixed_source_metrics: list[dict[str, float | str]],
    slc_source_metrics: list[dict[str, float | str]],
) -> list[dict[str, float | str | bool]]:
    """固定整相前後の target source 指標差分を評価用辞書へまとめる。"""
    comparisons: list[dict[str, float | str | bool]] = []
    for fixed_metric, slc_metric in zip(fixed_source_metrics, slc_source_metrics, strict=True):
        mainlobe_level_delta_db = float(slc_metric["level_at_nearest_source_grid_db20"]) - float(fixed_metric["level_at_nearest_source_grid_db20"])
        peak_shift_deg = float(slc_metric["peak_azimuth_deg"]) - float(fixed_metric["peak_azimuth_deg"])
        local_margin_improvement_db = float(slc_metric["local_to_nonlocal_margin_db"]) - float(fixed_metric["local_to_nonlocal_margin_db"])
        sidelobe_reduction_db = float(fixed_metric["max_nonlocal_level_db20"]) - float(slc_metric["max_nonlocal_level_db20"])
        mirror_reduction_db = float(fixed_metric["mirror_level_db20"]) - float(slc_metric["mirror_level_db20"])
        nominal_error_improvement_db = abs(float(fixed_metric["nominal_level_error_db"])) - abs(float(slc_metric["nominal_level_error_db"]))
        mainlobe_preserved = abs(mainlobe_level_delta_db) <= 1.5 and abs(peak_shift_deg) <= 4.0

        comparisons.append(
            {
                "label": str(fixed_metric["label"]),
                "source_azimuth_deg": float(fixed_metric["source_azimuth_deg"]),
                "source_frequency_hz": float(fixed_metric["source_frequency_hz"]),
                "mainlobe_level_delta_db": mainlobe_level_delta_db,
                "peak_azimuth_shift_deg": peak_shift_deg,
                "mainlobe_margin_improvement_db": local_margin_improvement_db,
                "local_margin_improvement_db": local_margin_improvement_db,
                "sidelobe_reduction_db": sidelobe_reduction_db,
                "mirror_reduction_db": mirror_reduction_db,
                "nominal_error_improvement_db": nominal_error_improvement_db,
                "mainlobe_preserved": bool(mainlobe_preserved),
            }
        )
    return comparisons


def run_integer_delay_slc_diagnostics(
    config: TimeDelayDiagnosticConfig,
    slc_config: SlcConfig,
    *,
    target_source_indices: tuple[int, ...] | None = None,
    slc_analysis_block_size: int = 64,
    max_reference_beams: int = 48,
) -> dict[str, object]:
    """固定整相後段へ周波数選択 SLC を適用した BL/FRAZ/BTR 診断を実行する。"""
    require_matplotlib()
    if slc_analysis_block_size <= 0:
        raise ValueError("slc_analysis_block_size must be positive.")
    if max_reference_beams <= 0:
        raise ValueError("max_reference_beams must be positive.")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_summary = run_integer_delay_diagnostics(config)

    source_specs = _resolve_source_specs(config)
    protected_target_indices = _resolve_target_source_indices(source_specs, target_source_indices)
    protected_target_index_set = {int(source_index) for source_index in protected_target_indices}
    target_source_specs = tuple(source_specs[int(source_index)] for source_index in protected_target_indices)
    interference_source_specs = tuple(
        source_spec for source_index, source_spec in enumerate(source_specs) if int(source_index) not in protected_target_index_set
    )

    array_positions_m, _, _ = _build_array_positions(config)
    beam_grid = _build_beam_grid(config)
    multichannel_signal, _ = _generate_target_scene(array_positions_m, source_specs, config)

    beamformer = IntegerDelayAndSumBeamformer.from_geometry(
        array_pos_m=array_positions_m,
        dir_cos=np.asarray(beam_grid["directions"], dtype=np.float64),
        fs_hz=float(config.fs_hz),
        sound_speed_m_s=float(config.sound_speed_m_s),
    )
    fixed_beam_output = beamformer.process(multichannel_signal)
    axis_az_deg = np.asarray(beam_grid["axis_az_deg"], dtype=np.float64)
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
        response_matrix = _build_beam_response_matrix(
            beamformer=beamformer,
            frequency_hz=float(target_frequency_hz),
            fs_hz=float(config.fs_hz),
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

        # 固定整相波形から対象周波数トーンだけを差し替えることで、
        # SLC が作用した成分と未処理の残差雑音・他周波数成分を分離して保持する。
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
        bl_title_prefix="Fixed BL evaluation",
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
        bl_title_prefix="SLC BL evaluation",
        local_half_width_beam=evaluation_half_width_beam,
        save_bl=True,
    )

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
        title="FRAZ after SLC",
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
        title="BTR after SLC",
        caption=slc_btr_caption,
        output_path=output_dir / "slc_btr.png",
    )

    source_comparisons = _build_source_comparisons(fixed_source_metrics, slc_source_metrics)
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
        "slc_fraz_png_path": str((output_dir / "slc_fraz.png").resolve()),
        "slc_btr_png_path": str((output_dir / "slc_btr.png").resolve()),
        "source_comparisons": source_comparisons,
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
    "run_integer_delay_slc_diagnostics",
]

