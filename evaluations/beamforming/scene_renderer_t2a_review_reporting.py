"""T2a streaming評価の完成結果をreview packへ直列化する。

本モジュールは、完成済みの評価結果からCSV、NPZ、JSON、Markdown、PNGを生成する。
scene生成、重み設計、FIR化、逐次信号処理、評価指標の計算は責務に含めない。
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from evaluations.beamforming.scene_renderer_t2a_waveform_reporting import (
    T2aWaveformScenario,
    WaveformIntegrityResult,
    write_input_waveform_diagnostics,
    write_output_waveform_diagnostics,
    write_target_waveform_integrity,
)
from spflow.beamforming_evaluation.diagnostic_plotting import centers_to_edges

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


class T2aReviewScenario(T2aWaveformScenario, Protocol):
    """review pack生成に必要なT2a scenario属性を定義する。

    入力はsampling、source、FFT、block条件であり、reporting関数が軸、表示範囲、
    metadataを構築するために参照する。scenario検証や信号生成は責務に含めない。
    """

    @property
    def duration_s(self) -> float:
        """評価対象信号長を秒で返す。"""
        ...

    @property
    def target_azimuth_deg(self) -> float:
        """target真値方位をdegで返す。"""
        ...

    @property
    def interferer_azimuth_deg(self) -> float:
        """interferer真値方位をdegで返す。"""
        ...

    @property
    def interferer_frequency_hz(self) -> float:
        """interferer真値周波数をHzで返す。"""
        ...

    @property
    def analysis_hop_size(self) -> int:
        """FRAZ解析hopをsample数で返す。"""
        ...


@dataclass(frozen=True)
class ScenarioSummaryRow:
    """一方式のT2a評価指標を固定列で保持する。

    入力は完成済みのBL、雑音、FIR、波形完全性、block境界指標であり、出力は
    `scenario_summary.csv`と`worst_cases.csv`の一行である。指標計算や採否判定は
    責務に含めず、本型の値はすべて観測値として扱う。
    """

    scenario: str
    method: str
    evaluation_pattern: str
    target_frequency_hz: float
    target_azimuth_deg: float
    target_peak_azimuth_deg: float
    target_peak_error_deg: float
    target_level_db_re_input_rms: float
    sidelobe_peak_db_re_mainlobe_peak: float
    output_snr_db: float
    interferer_level_at_target_beam_db_re_input_rms: float
    minimum_fir_energy_containment: float
    target_waveform_rms_delta_db: float
    target_waveform_correlation_after_phase_alignment: float
    target_waveform_residual_rms_db_re_input_rms: float
    target_phase_delay_samples_modulo_period: float
    streaming_one_block_max_abs_error: float
    streaming_boundary_max_abs_error: float
    streaming_valid_mask_matches_one_block: bool
    ebae_signal_count_at_target: int
    ebae_music_peak_azimuth_deg_at_target: float
    ebae_fallback_at_target: bool
    runtime_factor: float
    finite: bool


@dataclass(frozen=True)
class T2aReviewData:
    """T2a評価計算からreportingへ渡す完成結果を保持する。

    `frequency_hz`は`[n_frequency]`、`beam_azimuth_deg`は`[n_beam]`、各FRAZは
    `[n_beam,n_frequency]`で、level基準は`dB re input RMS`である。streamed波形と
    valid maskは`[n_beam,n_sample]`、source-frequency BLは`[n_beam]`である。

    本型は完成結果の受け渡しだけを担い、信号処理、指標計算、ファイル保存を担わない。
    """

    frequency_hz: FloatArray
    beam_azimuth_deg: FloatArray
    fraz_by_component: dict[str, dict[str, FloatArray]]
    valid_sample_counts: dict[str, int]
    streamed_waveforms: dict[str, dict[str, tuple[ComplexArray, BoolArray]]]
    one_block_mixed: dict[str, tuple[ComplexArray, BoolArray]]
    waveform_integrity_by_method: dict[str, WaveformIntegrityResult]
    streaming_overall_error_by_method: dict[str, float]
    streaming_boundary_error_by_method: dict[str, float]
    streaming_valid_match_by_method: dict[str, bool]
    diagnostic_zoom_by_method: dict[str, tuple[int, int, int]]
    source_frequency_bl_by_method: dict[str, FloatArray]
    summary_rows: tuple[ScenarioSummaryRow, ...]
    runtime_s: float
    runtime_factor: float
    target_frequency_index: int
    interferer_frequency_index: int
    target_beam_index: int
    reference_channel_index: int


@dataclass(frozen=True)
class T2aReviewContext:
    """T2a review packの再現条件と設計診断量を保持する。

    入力はscenario、外部係数path、方式、アレイ規模、完成重みの診断配列、rendered mixed
    信号である。出力先や評価結果そのものは保持せず、信号処理も責務に含めない。
    """

    scenario: T2aReviewScenario
    scenario_metadata: dict[str, Any]
    selected_method_ids: tuple[str, ...]
    review_title: str
    positions_path: Path
    shading_path: Path
    shading_frequency_step_hz: float
    n_channel: int
    predicted_aliases_deg: dict[str, tuple[float, ...]]
    rendered_mixed: FloatArray
    active_channel_count: FloatArray
    causal_delays_samples: IntArray
    ebae_signal_count: IntArray
    ebae_music_peak_azimuth_deg: FloatArray
    ebae_fallback_mask: BoolArray
    covariance_snapshot_count_by_beam: IntArray


def _write_input_spectrum(
    output_path: Path,
    mixed: FloatArray,
    scenario: T2aReviewScenario,
) -> None:
    """整相前mixed信号の片側per-bin RMS spectrumをPNGへ保存する。"""
    spectrum = np.fft.rfft(mixed, axis=1)
    power = np.abs(spectrum / float(mixed.shape[1])) ** 2
    # 実信号の非DC/Nyquist binは正負周波数powerを合算し、片側RMS powerへ変換する。
    power[:, 1:-1] *= 2.0
    level = 10.0 * np.log10(np.maximum(np.mean(power, axis=0), np.finfo(float).tiny))
    frequency = np.fft.rfftfreq(mixed.shape[1], d=1.0 / scenario.fs_hz)
    figure, axis = plt.subplots(figsize=(10.0, 4.0))
    axis.plot(frequency, level)
    axis.set(
        title="Pre-beamforming rendered target + interferer + noise",
        xlabel="Frequency [Hz]",
        ylabel="Per-bin RMS Level [dB re input RMS]",
        xlim=(0.0, scenario.fs_hz / 2.0),
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _write_bl_fraz_fl(
    output_path: Path,
    data: T2aReviewData,
    scenario: T2aReviewScenario,
    predicted_target_aliases_deg: tuple[float, ...],
) -> None:
    """同一軸・同一level基準でmixed信号のBL、FL、FRAZを保存する。"""
    fraz_by_method = data.fraz_by_component["mixed"]
    if len(fraz_by_method) == 1:
        # 単独方式は2×2へ詰め、比較方式が存在しない空panelを成果物へ残さない。
        figure, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
        bl_axis = axes[0, 0]
        fl_axis = axes[0, 1]
        source_bl_axis = axes[1, 0]
        fraz_axes = (axes[1, 1],)
    else:
        figure, axes = plt.subplots(2, 3, figsize=(18.0, 9.0))
        bl_axis = axes[0, 0]
        fl_axis = axes[0, 1]
        source_bl_axis = axes[0, 2]
        fraz_axes = tuple(axes[1])
    for method_id, fraz in fraz_by_method.items():
        bl_axis.plot(
            data.beam_azimuth_deg,
            fraz[:, data.target_frequency_index],
            label=method_id,
        )
        fl_axis.plot(data.frequency_hz, fraz[data.target_beam_index], label=method_id)
        source_bl_axis.plot(
            data.beam_azimuth_deg,
            data.source_frequency_bl_by_method[method_id],
            label=method_id,
        )
    bl_axis.axvline(scenario.target_azimuth_deg, color="black", linestyle="--")
    for alias_deg in predicted_target_aliases_deg:
        # 理論aliasは計算BLを見てから付ける説明ではなく、宣言幾何からの事前予測として描く。
        bl_axis.axvline(alias_deg, color="tab:orange", linestyle=":", alpha=0.8)
    bl_axis.set(
        title=f"BL at {data.frequency_hz[data.target_frequency_index]:g} Hz",
        xlabel="Waiting-beam azimuth [deg]",
        ylabel="RMS Level [dB re input RMS]",
    )
    fl_axis.set(
        title=f"FL at {data.beam_azimuth_deg[data.target_beam_index]:g} deg",
        xlabel="Frequency [Hz]",
        ylabel="RMS Level [dB re input RMS]",
    )
    source_bl_axis.set(
        title="Source-frequency BL",
        xlabel="Waiting-beam azimuth [deg]",
        ylabel="RMS Level [dB re input RMS]",
    )
    azimuth_edges = centers_to_edges(data.beam_azimuth_deg)
    frequency_edges = centers_to_edges(data.frequency_hz)
    finite = np.concatenate([values[np.isfinite(values)] for values in fraz_by_method.values()])
    upper = float(np.max(finite))
    lower = upper - 80.0
    for axis, (method_id, fraz) in zip(fraz_axes, fraz_by_method.items(), strict=False):
        image = axis.pcolormesh(
            azimuth_edges,
            frequency_edges,
            fraz.T,
            shading="auto",
            vmin=lower,
            vmax=upper,
        )
        axis.set(
            title=f"FRAZ: {method_id}",
            xlabel="Waiting-beam azimuth [deg]",
            ylabel="Frequency [Hz]",
        )
        figure.colorbar(image, ax=axis, label="RMS Level [dB re input RMS]")
    for axis in fraz_axes[len(fraz_by_method) :]:
        # 選択方式より多い空panelを表示せず、存在しない方式の結果と誤認させない。
        axis.set_visible(False)
    for axis in (bl_axis, fl_axis, source_bl_axis):
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _write_source_frequency_bl(
    output_path: Path,
    data: T2aReviewData,
    scenario: T2aReviewScenario,
) -> None:
    """全source真値周波数を統合したBL overlayを保存する。"""
    figure, axis = plt.subplots(figsize=(10.0, 4.5))
    for method_id, levels in data.source_frequency_bl_by_method.items():
        axis.plot(data.beam_azimuth_deg, levels, label=method_id)
    axis.axvline(scenario.target_azimuth_deg, color="black", linestyle="--", label="target")
    axis.axvline(
        scenario.interferer_azimuth_deg,
        color="gray",
        linestyle=":",
        label="interferer",
    )
    axis.set(
        title="Source-frequency BL overlay",
        xlabel="Waiting-beam azimuth [deg]",
        ylabel="RMS Level [dB re input RMS]",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _build_plot_arrays(
    context: T2aReviewContext,
    data: T2aReviewData,
) -> dict[str, Any]:
    """PNG再描画と数値監査に必要な描画直前配列を構築する。"""
    # NumPyのsavez型stubはkeyword引数を固定できないため、serializer境界だけAnyを許容する。
    # 実際に格納する値は以下で明示したndarrayに限定する。
    arrays: dict[str, Any] = {
        "azimuth_deg": data.beam_azimuth_deg,
        "frequency_hz": data.frequency_hz,
        "active_channel_count": context.active_channel_count,
        "causal_integer_delays_samples": context.causal_delays_samples,
        "t2a_ebae_signal_count": context.ebae_signal_count,
        "t2a_ebae_music_peak_azimuth_deg": context.ebae_music_peak_azimuth_deg,
        "t2a_ebae_fallback_mask": context.ebae_fallback_mask,
        "covariance_snapshot_count_by_beam": context.covariance_snapshot_count_by_beam,
        "diagnostic_time_s": (
            np.arange(context.rendered_mixed.shape[1], dtype=np.float64) / context.scenario.fs_hz
        ),
        "diagnostic_reference_channel_index": np.asarray(
            data.reference_channel_index, dtype=np.int64
        ),
        "diagnostic_input_mixed_reference_channel": context.rendered_mixed[
            data.reference_channel_index
        ],
    }
    for component_id, method_levels in data.fraz_by_component.items():
        for method_id, levels in method_levels.items():
            arrays[f"{component_id}_{method_id}_fraz_db_re_input_rms"] = levels
    for method_id in data.fraz_by_component["mixed"]:
        mixed_output, mixed_valid = data.streamed_waveforms["mixed"][method_id]
        one_block_output, _ = data.one_block_mixed[method_id]
        integrity = data.waveform_integrity_by_method[method_id]
        arrays[f"{method_id}_source_frequency_bl_db_re_input_rms"] = (
            data.source_frequency_bl_by_method[method_id]
        )
        arrays[f"{method_id}_target_beam_mixed_output_real"] = np.real(
            mixed_output[data.target_beam_index]
        )
        arrays[f"{method_id}_target_beam_mixed_valid_mask"] = mixed_valid[data.target_beam_index]
        arrays[f"{method_id}_target_beam_mixed_one_block_real"] = np.real(
            one_block_output[data.target_beam_index]
        )
        arrays[f"{method_id}_target_integrity_input"] = integrity.reference_signal
        arrays[f"{method_id}_target_integrity_phase_aligned_output"] = (
            integrity.phase_aligned_output
        )
    return arrays


def _write_summary_csv(output_dir: Path, rows: tuple[ScenarioSummaryRow, ...]) -> None:
    """完成指標を主表とレビュー優先順表へ保存する。"""
    fieldnames = [field.name for field in fields(ScenarioSummaryRow)]
    csv_rows = [asdict(row) for row in rows]
    with (output_dir / "scenario_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    # worst_casesは採否の自動判定ではなく、peak誤差とFIR包含率が悪い方式を先に見る索引である。
    worst_rows = sorted(
        rows,
        key=lambda row: (-row.target_peak_error_deg, row.minimum_fir_energy_containment),
    )
    with (output_dir / "worst_cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in worst_rows)


def _write_waveform_figures(
    output_dir: Path,
    context: T2aReviewContext,
    data: T2aReviewData,
) -> None:
    """完成波形から入力、出力、target完全性の診断PNGを保存する。"""
    first_method_id = next(iter(data.fraz_by_component["mixed"]))
    _, input_zoom_start, input_zoom_stop = data.diagnostic_zoom_by_method[first_method_id]
    write_input_waveform_diagnostics(
        output_dir / "input_waveform_diagnostics.png",
        context.rendered_mixed,
        data.reference_channel_index,
        input_zoom_start,
        input_zoom_stop,
        context.scenario,
    )
    for method_id in data.fraz_by_component["mixed"]:
        mixed_output, mixed_valid = data.streamed_waveforms["mixed"][method_id]
        one_block_output, _ = data.one_block_mixed[method_id]
        _, zoom_start, zoom_stop = data.diagnostic_zoom_by_method[method_id]
        write_output_waveform_diagnostics(
            output_dir / f"output_waveform_diagnostics_{method_id}.png",
            method_id,
            mixed_output[data.target_beam_index],
            one_block_output[data.target_beam_index],
            mixed_valid[data.target_beam_index],
            zoom_start,
            zoom_stop,
            context.scenario,
        )
        write_target_waveform_integrity(
            output_dir / f"target_waveform_integrity_{method_id}.png",
            method_id,
            data.waveform_integrity_by_method[method_id],
            context.scenario,
        )


def _write_metadata(
    output_dir: Path,
    context: T2aReviewContext,
    data: T2aReviewData,
) -> None:
    """scenario条件、level基準、診断指標をJSONへ保存する。"""
    metadata = {
        "scenario": context.scenario_metadata,
        "positions_path": str(context.positions_path),
        "shading_path": str(context.shading_path),
        "shading_frequency_step_hz": context.shading_frequency_step_hz,
        "n_channel": context.n_channel,
        "active_channel_count_by_frequency": context.active_channel_count.tolist(),
        "t2a_ebae_fallback_count": int(np.count_nonzero(context.ebae_fallback_mask)),
        "covariance_snapshot_count_by_beam": (context.covariance_snapshot_count_by_beam.tolist()),
        "valid_sample_counts": data.valid_sample_counts,
        "runtime_s": data.runtime_s,
        "runtime_factor": data.runtime_factor,
        "level_reference": "BL/FRAZ/FL: dB re input RMS",
        "evaluation_patterns": ["sparse_array_design", "fixed_beam_multi_source"],
        "selected_method_ids": list(context.selected_method_ids),
        "waveform_diagnostics": {
            "input_reference_channel_index": data.reference_channel_index,
            "output_beam_azimuth_deg": float(data.beam_azimuth_deg[data.target_beam_index]),
            "spectrum_reference": "per-bin RMS level, dB re input RMS",
            "phase_delay_definition": "sample delay modulo one target-tone period",
            "method_metrics": {
                method_id: {
                    "target_waveform_rms_delta_db": (
                        data.waveform_integrity_by_method[method_id].rms_delta_db
                    ),
                    "target_waveform_correlation_after_phase_alignment": (
                        data.waveform_integrity_by_method[
                            method_id
                        ].correlation_after_phase_alignment
                    ),
                    "target_waveform_residual_rms_db_re_input_rms": (
                        data.waveform_integrity_by_method[method_id].residual_rms_db_re_input_rms
                    ),
                    "target_phase_delay_samples_modulo_period": (
                        data.waveform_integrity_by_method[
                            method_id
                        ].phase_delay_samples_modulo_period
                    ),
                    "streaming_one_block_max_abs_error": (
                        data.streaming_overall_error_by_method[method_id]
                    ),
                    "streaming_boundary_max_abs_error": (
                        data.streaming_boundary_error_by_method[method_id]
                    ),
                    "streaming_valid_mask_matches_one_block": (
                        data.streaming_valid_match_by_method[method_id]
                    ),
                }
                for method_id in data.fraz_by_component["mixed"]
            },
        },
        "predicted_uniform_subset_grating_azimuths_deg": context.predicted_aliases_deg,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_review_index(output_dir: Path, context: T2aReviewContext) -> None:
    """成果物の意味、用途、非対応条件を日本語Markdownへ保存する。"""
    if context.selected_method_ids == ("t2a_ebae",):
        method_description = (
            "MATLAB係数の周波数別active channelとshadingを適用し、候補方位別T共分散から"
            "T2a-EBAE残差重みだけを設計、block逐次処理、評価、表示した。EBAE内部の成立条件を"
            "満たさない場合に使う固定整相fallbackは安全契約として残すが、`fixed_baseline`と"
            "`t2a_mvdr`の独立branchは生成しない。比較baselineを含まないため、本pack単独で"
            "方式間の採否は判断しない。"
        )
    else:
        method_description = (
            "MATLAB係数の周波数別active channelとshadingを適用し、選択方式 "
            f"{', '.join(context.selected_method_ids)} を同じblock反復、完成区間、"
            "表示軸で評価した。"
        )
    (output_dir / "review_index.md").write_text(
        f"# {context.review_title}\n\n"
        f"{method_description}\n\n"
        "- `rendered_input_spectrum.png`: 整相前target+interferer+noiseのper-bin RMS。\n"
        "- `input_waveform_diagnostics.png`: 基準channel入力の全体・block境界拡大波形とspectrum。\n"
        "- `output_waveform_diagnostics_<method>.png`: target待受beam出力、境界拡大、"
        "一括block差、spectrum。\n"
        "- `target_waveform_integrity_<method>.png`: target-only入力と位相整列後出力の"
        "波形、残差、spectrum。\n"
        "- `bl_fraz_fl.png`: 整相後mixed信号のBL、FL、FRAZ。\n"
        "- `source_frequency_bl_overlay.png`: 全source真値周波数の最大BL。\n"
        "- `scenario_summary.csv`: peak、sidelobe、SNR、FIR、波形完全性、境界、runtime観測値。\n"
        "- `worst_cases.csv`: レビュー優先順で並べた同じ観測値。自動採否には使わない。\n"
        "- `plot_arrays.npz`: PNGの描画直前配列。図の再描画、shape・軸・dB基準の"
        "数値監査に使い、方式の主たる数値判定は`scenario_summary.csv`を使う。\n\n"
        "波形完全性はtarget-only入力の原点最近傍channelとtarget待受beam出力を比較する。"
        "単一toneの位相遅延は1周期ごとに同値なため、絶対伝搬遅延ではなく1周期を法とする。"
        "分割streamingと同じ係数を一括blockへ適用した差によりblock境界由来の不連続を確認する。\n\n"
        "本scenarioは自由音場、水平固定音源、channel独立帯域雑音である。海面・海底反射、"
        "音速プロファイル、係数更新過渡は扱わず、それらの成立性を本結果から判断しない。\n",
        encoding="utf-8",
    )


def write_t2a_review_pack(
    output_dir: Path,
    context: T2aReviewContext,
    data: T2aReviewData,
) -> None:
    """完成済みT2a評価結果をreview packへ保存する。

    Args:
        output_dir: CSV、NPZ、JSON、Markdown、PNGの保存先。
        context: scenario、外部係数、設計診断量。配列のshapeは各field docstringに従う。
        data: 完成評価結果。FRAZは`[n_beam,n_frequency]`、波形は
            `[n_beam,n_sample]`、level基準は`dB re input RMS`。

    Returns:
        なし。全方式の完成結果だけを指定directoryへ保存する。

    Raises:
        KeyError: contextとdataの方式またはcomponent対応が欠落している場合。
        ValueError: 描画対象に有限なFRAZ値が存在しない場合。

    境界条件:
        NPZはPNGの描画直前配列を保存し、再描画と数値監査に用いる。CSVを主たる数値表、
        PNGを表示確認、review indexを意味と制約の記録として使い分ける。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(output_dir, data.summary_rows)
    _write_input_spectrum(
        output_dir / "rendered_input_spectrum.png",
        context.rendered_mixed,
        context.scenario,
    )
    _write_waveform_figures(output_dir, context, data)
    _write_bl_fraz_fl(
        output_dir / "bl_fraz_fl.png",
        data,
        context.scenario,
        context.predicted_aliases_deg["target"],
    )
    _write_source_frequency_bl(
        output_dir / "source_frequency_bl_overlay.png",
        data,
        context.scenario,
    )
    # 全方式のFRAZ、波形、境界参照、source-frequency BL完成後に一つのNPZを公開する。
    np.savez_compressed(output_dir / "plot_arrays.npz", **_build_plot_arrays(context, data))
    _write_metadata(output_dir, context, data)
    _write_review_index(output_dir, context)
