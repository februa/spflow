"""ChatGPT レビュー用の軽量 ABF-like review_pack を生成する。

このスクリプトは、8月評価向け shortlist と同じ effective 出力だけを使い、
scenario ごとの比較図、描画前配列、summary CSV、review index をまとめて保存する。

出力先は `artifacts/beamforming/lightweight_abf_like_comparison/review_pack/` である。
"""

from __future__ import annotations

import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from evaluations.beamforming.evaluate_lightweight_abf_like_august_shortlist import (
    LEVEL_UNIT_LABEL,
    TONE_FREQUENCIES_HZ,
    CandidateEvaluation,
    ScenarioDefinition,
    _btr_relative_levels,
    _evaluate_scenario,
    _load_assets,
    _negative_scenarios,
    _representative_scenarios,
    _rms_levels_db20,
    _tone_projection_levels_db20,
)
from spflow.beamforming import SourceSectorMask
from spflow.beamforming import diagnostic_plotting as plotting
from spflow.beamforming.diagnostic_plotting import centers_to_edges, require_matplotlib

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]

REVIEW_PACK_DIR = Path("artifacts/beamforming/lightweight_abf_like_comparison/review_pack")
FIGURE_DIR = REVIEW_PACK_DIR / "figures"
DATA_DIR = REVIEW_PACK_DIR / "data"
REVIEW_INDEX_PATH = REVIEW_PACK_DIR / "review_index.md"
SCENARIO_SUMMARY_PATH = REVIEW_PACK_DIR / "scenario_summary.csv"
WORST_CASES_PATH = REVIEW_PACK_DIR / "worst_cases.csv"
BTR_COLOR_RANGE_DB = (-12.0, 0.0)
METHOD_ORDER = ("fixed_baseline", "A2_safe", "A2_aggressive")
METHOD_LABELS = {"fixed_baseline": "fixed", "A2_safe": "A2_safe", "A2_aggressive": "A2_aggressive"}
METHOD_COLORS = {"fixed_baseline": "black", "A2_safe": "tab:blue", "A2_aggressive": "tab:orange"}
SUMMARY_COLUMNS = (
    "scenario",
    "method",
    "mask_type",
    "candidate",
    "status",
    "source_peak_delta_db",
    "source_azimuth_error_deg",
    "non_source_global_peak_delta_db",
    "non_source_p95_level_delta_db",
    "non_source_p99_level_delta_db",
    "non_source_integrated_level_delta_db",
    "source_to_non_source_margin_delta_db",
    "false_peak_count_delta",
    "max_local_worsening_db_gated",
    "fallback_required",
    "fallback_reason",
    "runtime_factor",
)


@dataclass(frozen=True)
class ScenarioReviewData:
    """review_pack で扱う 1 scenario の評価結果と描画前配列を保持する。

    このクラスは、scenario 条件、source mask、fixed / A2_safe / A2_aggressive の
    effective 出力から作った BL / FRAZ / BTR 配列をまとめる。

    入力は beam-domain 評価結果であり、出力は review 図と npz 保存に使う数値配列である。

    SLC 係数推定や source mask 検出そのものは責務に含めない。
    信号処理上は、source-preserving scan として source visibility と non-source 抑圧を
    同じ軸でレビューするための中間データである。
    """

    scenario: ScenarioDefinition
    source_mask: SourceSectorMask
    rows: list[dict[str, object]]
    evaluations: dict[str, CandidateEvaluation]
    azimuth_deg: FloatArray
    frequency_hz: FloatArray
    time_sec: FloatArray
    bl_levels_db: dict[str, FloatArray]
    fraz_levels_db: dict[str, FloatArray]
    btr_levels_db: dict[str, FloatArray]


def _plt() -> Any:
    """matplotlib.pyplot module を返す。"""
    require_matplotlib()
    pyplot = plotting.plt
    if pyplot is None:
        raise RuntimeError("matplotlib is required to build review pack figures.")
    return pyplot


def _scenario_definitions() -> tuple[ScenarioDefinition, ...]:
    """review_pack に含める scenario を返す。"""
    scenarios: list[ScenarioDefinition] = []
    seen: set[str] = set()
    for scenario in (*_representative_scenarios(), *_negative_scenarios()):
        if scenario.scenario_id in seen:
            continue
        scenarios.append(scenario)
        seen.add(scenario.scenario_id)
    return tuple(scenarios)


def _safe_float(value: object, default: float = 0.0) -> float:
    """CSV / row の値を Python float へ変換する。"""
    if value is None or value == "":
        return float(default)
    if isinstance(value, bool):
        raise TypeError("bool value cannot be converted to metric float.")
    if isinstance(value, int | float | np.integer | np.floating):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"unsupported metric value type: {type(value)!r}")


def _safe_bool(value: object) -> bool:
    """CSV / row の値を Python bool へ変換する。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _sanitize_csv_value(value: object) -> object:
    """CSV に保存しやすい scalar へ変換する。"""
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, list | tuple):
        return "|".join(str(item) for item in value)
    return value


def _write_csv(rows: list[dict[str, object]], path: Path, columns: tuple[str, ...] | None) -> None:
    """CSV を保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        columns if columns is not None else tuple(sorted({key for row in rows for key in row}))
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _sanitize_csv_value(row.get(key, "")) for key in fieldnames})


def _effective_row(rows: list[dict[str, object]], method_id: str) -> dict[str, object]:
    """指定 method の effective row を返す。"""
    for row in rows:
        if str(row.get("method_id", "")) != method_id:
            continue
        if str(row.get("output_stage", "")) == "effective":
            return row
    raise KeyError(f"effective row not found: {method_id}")


def _evaluate_review_scenario(scenario: ScenarioDefinition, assets: Any) -> ScenarioReviewData:
    """1 scenario を評価し、review 用配列を作る。"""
    rows, evaluations, source_mask = _evaluate_scenario(
        scenario=scenario, assets=assets, include_raw=True
    )
    fs_hz = float(assets.array_definition.fs_hz)
    bl_levels: dict[str, FloatArray] = {}
    fraz_levels: dict[str, FloatArray] = {}
    btr_levels: dict[str, FloatArray] = {}
    time_sec = np.empty(0, dtype=np.float64)
    for method_id in METHOD_ORDER:
        output = np.asarray(evaluations[method_id].effective_output, dtype=np.complex128)
        bl_levels[method_id] = _rms_levels_db20(output)
        fraz_levels[method_id] = _tone_projection_levels_db20(
            beam_output=output,
            fs_hz=fs_hz,
            frequencies_hz=TONE_FREQUENCIES_HZ,
        )
        btr_time_sec, btr_level_db, _ = _btr_relative_levels(
            beam_output=output,
            fs_hz=fs_hz,
            block_size=128,
        )
        if time_sec.size == 0:
            time_sec = btr_time_sec
        btr_levels[method_id] = btr_level_db
    return ScenarioReviewData(
        scenario=scenario,
        source_mask=source_mask,
        rows=rows,
        evaluations=evaluations,
        azimuth_deg=np.asarray(assets.axis_azimuth_deg, dtype=np.float64),
        frequency_hz=np.asarray(TONE_FREQUENCIES_HZ, dtype=np.float64),
        time_sec=time_sec,
        bl_levels_db=bl_levels,
        fraz_levels_db=fraz_levels,
        btr_levels_db=btr_levels,
    )


def _mask_runs(mask: BoolArray) -> list[tuple[int, int]]:
    """bool mask の連続 run を返す。"""
    normalized = np.asarray(mask, dtype=np.bool_)
    if normalized.ndim != 1:
        raise ValueError("mask must have shape (n_beam,).")
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, enabled in enumerate(normalized.tolist()):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, int(normalized.size)))
    return runs


def _add_mask_spans(axis: Any, azimuth_deg: FloatArray, source_mask: BoolArray) -> None:
    """plot axis へ source mask / non-source sector を描画する。

    Args:
        axis: matplotlib axis。
        azimuth_deg: 方位軸。shape は `[n_beam]`、単位は deg。
        source_mask: source sector mask。shape は `[n_beam]`。
    """
    azimuth_edges = centers_to_edges(np.asarray(azimuth_deg, dtype=np.float64))
    source = np.asarray(source_mask, dtype=np.bool_)
    non_source_runs = _mask_runs(np.logical_not(source))
    source_runs = _mask_runs(source)
    for run_index, (start, stop) in enumerate(non_source_runs):
        axis.axvspan(
            float(azimuth_edges[start]),
            float(azimuth_edges[stop]),
            color="0.92",
            alpha=0.45,
            linewidth=0.0,
            label="non-source sector" if run_index == 0 else None,
        )
    for run_index, (start, stop) in enumerate(source_runs):
        axis.axvspan(
            float(azimuth_edges[start]),
            float(azimuth_edges[stop]),
            color="tab:green",
            alpha=0.16,
            linewidth=0.0,
            label="source mask" if run_index == 0 else None,
        )


def _source_caption(scenario: ScenarioDefinition) -> str:
    """scenario の source 条件を短く表す。"""
    return "; ".join(
        (
            f"{source.label}: az={float(source.azimuth_deg):.1f} deg, "
            f"f={float(source.frequency_hz):.0f} Hz, level={float(source.level_db):.1f} dB"
        )
        for source in scenario.source_specs
    )


def _mask_caption(data: ScenarioReviewData) -> str:
    """source mask / non-source beam 数の説明を返す。"""
    source_count = int(np.count_nonzero(data.source_mask.source_mask))
    non_source_count = int(np.count_nonzero(data.source_mask.non_source_mask))
    return f"source_mask_beams={source_count}, non_source_beams={non_source_count}"


def _save_figure(fig: Any, path: Path) -> None:
    """figure を PNG 保存して閉じる。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    _plt().close(fig)


def _plot_bl_overlay(data: ScenarioReviewData, output_path: Path) -> None:
    """fixed / A2_safe / A2_aggressive の BL overlay を保存する。"""
    plt = _plt()
    fig, axis = plt.subplots(figsize=(10.5, 5.0))
    all_levels = np.concatenate([data.bl_levels_db[method] for method in METHOD_ORDER])
    y_min = float(np.min(all_levels) - 1.0)
    y_max = float(np.max(all_levels) + 1.0)
    _add_mask_spans(axis, data.azimuth_deg, data.source_mask.source_mask)
    for method_id in METHOD_ORDER:
        axis.plot(
            data.azimuth_deg,
            data.bl_levels_db[method_id],
            linewidth=1.6,
            color=METHOD_COLORS[method_id],
            label=METHOD_LABELS[method_id],
        )
    axis.set_ylim(y_min, y_max)
    axis.set_xlabel("Azimuth [deg]")
    axis.set_ylabel(f"RMS Level [{LEVEL_UNIT_LABEL}]")
    axis.set_title(f"{data.scenario.scenario_id}: BL overlay")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.text(
        0.5,
        0.01,
        f"{_source_caption(data.scenario)}\n{_mask_caption(data)}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.03, 0.08, 1.0, 0.96))
    _save_figure(fig, output_path)


def _plot_bl_delta(data: ScenarioReviewData, output_path: Path) -> None:
    """A2 - fixed の BL delta を保存する。"""
    plt = _plt()
    fig, axis = plt.subplots(figsize=(10.5, 4.8))
    fixed = data.bl_levels_db["fixed_baseline"]
    deltas = {
        "A2_safe": data.bl_levels_db["A2_safe"] - fixed,
        "A2_aggressive": data.bl_levels_db["A2_aggressive"] - fixed,
    }
    max_abs = max(float(np.max(np.abs(delta))) for delta in deltas.values())
    y_limit = max(1.0, max_abs + 0.5)
    _add_mask_spans(axis, data.azimuth_deg, data.source_mask.source_mask)
    for method_id, delta in deltas.items():
        axis.plot(
            data.azimuth_deg,
            delta,
            linewidth=1.5,
            color=METHOD_COLORS[method_id],
            label=f"{METHOD_LABELS[method_id]} - fixed",
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylim(-y_limit, y_limit)
    axis.set_xlabel("Azimuth [deg]")
    axis.set_ylabel("BL Delta [dB re fixed BL level]")
    axis.set_title(f"{data.scenario.scenario_id}: BL delta")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.text(0.5, 0.01, _mask_caption(data), ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0.03, 0.07, 1.0, 0.96))
    _save_figure(fig, output_path)


def _plot_fraz_delta(data: ScenarioReviewData, output_path: Path) -> None:
    """A2 - fixed の FRAZ delta を保存する。"""
    plt = _plt()
    fixed = data.fraz_levels_db["fixed_baseline"]
    safe_delta = data.fraz_levels_db["A2_safe"] - fixed
    aggressive_delta = data.fraz_levels_db["A2_aggressive"] - fixed
    max_abs = max(float(np.max(np.abs(safe_delta))), float(np.max(np.abs(aggressive_delta))), 1.0)
    az_edges = centers_to_edges(data.azimuth_deg)
    freq_edges = centers_to_edges(data.frequency_hz)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    image = None
    for axis, method_id, delta in (
        (axes[0], "A2_safe", safe_delta),
        (axes[1], "A2_aggressive", aggressive_delta),
    ):
        image = axis.pcolormesh(
            az_edges,
            freq_edges,
            delta.T,
            shading="flat",
            cmap="coolwarm",
            vmin=-max_abs,
            vmax=max_abs,
        )
        _add_mask_spans(axis, data.azimuth_deg, data.source_mask.source_mask)
        axis.set_title(f"{METHOD_LABELS[method_id]} - fixed")
        axis.set_xlabel("Azimuth [deg]")
    axes[0].set_ylabel("Frequency [Hz]")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="FRAZ Delta [dB re fixed FRAZ level]")
    fig.suptitle(f"{data.scenario.scenario_id}: FRAZ delta")
    fig.text(0.5, 0.01, _mask_caption(data), ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.16, top=0.86, wspace=0.22)
    _save_figure(fig, output_path)


def _plot_btr_panel(data: ScenarioReviewData, output_path: Path) -> None:
    """fixed / A2_safe / A2_aggressive の BTR panel を保存する。

    BTR は各 method の frame max 基準であり、抑圧量の定量比較ではなく source track の
    連続性確認用である。color scale は 3 panel で揃える。
    """
    plt = _plt()
    az_edges = centers_to_edges(data.azimuth_deg)
    time_edges = centers_to_edges(data.time_sec)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8), sharey=True)
    image = None
    for axis, method_id in zip(axes, METHOD_ORDER, strict=True):
        image = axis.pcolormesh(
            az_edges,
            time_edges,
            data.btr_levels_db[method_id],
            shading="flat",
            cmap="viridis",
            vmin=BTR_COLOR_RANGE_DB[0],
            vmax=BTR_COLOR_RANGE_DB[1],
        )
        _add_mask_spans(axis, data.azimuth_deg, data.source_mask.source_mask)
        axis.set_title(METHOD_LABELS[method_id])
        axis.set_xlabel("Azimuth [deg]")
    axes[0].set_ylabel("Time [s]")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Relative Level [dB re frame max]")
    fig.suptitle(f"{data.scenario.scenario_id}: BTR source-track continuity")
    fig.text(
        0.5,
        0.01,
        "BTR is normalized per frame; use for source track continuity, not suppression amount.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.16, top=0.86, wspace=0.22)
    _save_figure(fig, output_path)


def _save_npz(data: ScenarioReviewData, output_path: Path) -> None:
    """BL / FRAZ / BTR の描画前配列を npz 保存する。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        azimuth_deg=data.azimuth_deg,
        frequency_hz=data.frequency_hz,
        time_sec=data.time_sec,
        fixed_level_db=data.bl_levels_db["fixed_baseline"],
        a2_safe_level_db=data.bl_levels_db["A2_safe"],
        a2_aggressive_level_db=data.bl_levels_db["A2_aggressive"],
        fixed_fraz_level_db=data.fraz_levels_db["fixed_baseline"],
        a2_safe_fraz_level_db=data.fraz_levels_db["A2_safe"],
        a2_aggressive_fraz_level_db=data.fraz_levels_db["A2_aggressive"],
        fixed_btr_level_db=data.btr_levels_db["fixed_baseline"],
        a2_safe_btr_level_db=data.btr_levels_db["A2_safe"],
        a2_aggressive_btr_level_db=data.btr_levels_db["A2_aggressive"],
        source_mask=np.asarray(data.source_mask.source_mask, dtype=np.bool_),
        non_source_mask=np.asarray(data.source_mask.non_source_mask, dtype=np.bool_),
    )


def _build_scenario_summary_rows(data_items: list[ScenarioReviewData]) -> list[dict[str, object]]:
    """scenario_summary.csv の row を作る。"""
    rows: list[dict[str, object]] = []
    for data in data_items:
        for method_id in METHOD_ORDER:
            row = _effective_row(data.rows, method_id)
            rows.append(
                {
                    "scenario": data.scenario.scenario_id,
                    "method": method_id,
                    "mask_type": data.scenario.mask_type,
                    "candidate": data.evaluations[method_id].candidate.candidate_id,
                    "status": str(row.get("status", "")),
                    "source_peak_delta_db": _safe_float(row.get("max_abs_source_peak_delta_db")),
                    "source_azimuth_error_deg": _safe_float(
                        row.get("max_source_azimuth_error_deg")
                    ),
                    "non_source_global_peak_delta_db": _safe_float(
                        row.get("non_source_global_peak_delta_db")
                    ),
                    "non_source_p95_level_delta_db": _safe_float(
                        row.get("non_source_p95_level_delta_db")
                    ),
                    "non_source_p99_level_delta_db": _safe_float(
                        row.get("non_source_p99_level_delta_db")
                    ),
                    "non_source_integrated_level_delta_db": _safe_float(
                        row.get("non_source_integrated_level_delta_db")
                    ),
                    "source_to_non_source_margin_delta_db": _safe_float(
                        row.get("source_to_non_source_margin_delta_db")
                    ),
                    "false_peak_count_delta": int(
                        round(_safe_float(row.get("false_peak_count_delta")))
                    ),
                    "max_local_worsening_db_gated": _safe_float(
                        row.get("max_local_worsening_db_gated")
                    ),
                    "fallback_required": _safe_bool(row.get("fallback_required")),
                    "fallback_reason": str(row.get("fallback_reason", "")),
                    "runtime_factor": _safe_float(row.get("realtime_factor")),
                    "source_count": int(round(_safe_float(row.get("source_count")))),
                    "source_count_in_mask": int(
                        round(_safe_float(row.get("source_count_in_mask")))
                    ),
                    "scenario_group": str(row.get("scenario_group", "")),
                    "mask_outside_source_count": int(
                        round(_safe_float(row.get("mask_outside_source_count")))
                    ),
                    "mask_outside_source_labels": str(row.get("mask_outside_source_labels", "")),
                    "max_mask_outside_source_suppression_db": _safe_float(
                        row.get("max_mask_outside_source_suppression_db")
                    ),
                }
            )
    return rows


def _append_metric_worst_rows(
    worst_rows: list[dict[str, object]], summary_rows: list[dict[str, object]]
) -> None:
    """各 metric の worst top 10 を worst_rows へ追加する。"""
    metric_specs = (
        ("source_peak_delta_db", True),
        ("source_azimuth_error_deg", True),
        ("non_source_global_peak_delta_db", True),
        ("non_source_p95_level_delta_db", True),
        ("non_source_p99_level_delta_db", True),
        ("non_source_integrated_level_delta_db", True),
        ("source_to_non_source_margin_delta_db", False),
        ("false_peak_count_delta", True),
        ("max_local_worsening_db_gated", True),
        ("runtime_factor", True),
        ("max_mask_outside_source_suppression_db", False),
    )
    a2_rows = [row for row in summary_rows if str(row.get("method", "")) != "fixed_baseline"]
    for metric, descending in metric_specs:
        ranked = sorted(a2_rows, key=lambda row: _safe_float(row.get(metric)), reverse=descending)
        for rank, row in enumerate(ranked[:10], start=1):
            worst_rows.append(
                {
                    "category": "metric_worst_top10",
                    "metric": metric,
                    "rank": rank,
                    "scenario": row.get("scenario", ""),
                    "method": row.get("method", ""),
                    "value": row.get(metric, ""),
                    "status": row.get("status", ""),
                    "details": "descending" if descending else "ascending",
                }
            )


def _append_detected_mismatch_rows(
    worst_rows: list[dict[str, object]], summary_rows: list[dict[str, object]]
) -> None:
    """detected mask の source count mismatch を worst_rows へ追加する。"""
    for row in summary_rows:
        if row.get("mask_type") != "detected":
            continue
        if row.get("source_count") == row.get("source_count_in_mask"):
            continue
        worst_rows.append(
            {
                "category": "detected_mask_source_count_mismatch",
                "metric": "source_count_in_mask",
                "rank": "",
                "scenario": row.get("scenario", ""),
                "method": row.get("method", ""),
                "value": row.get("source_count_in_mask", ""),
                "status": row.get("status", ""),
                "details": f"source_count={row.get('source_count', '')}",
            }
        )


def _append_fallback_rows(
    worst_rows: list[dict[str, object]], summary_rows: list[dict[str, object]]
) -> None:
    """fallback_required row を worst_rows へ追加する。"""
    for row in summary_rows:
        if not _safe_bool(row.get("fallback_required")):
            continue
        worst_rows.append(
            {
                "category": "fallback_rows",
                "metric": "fallback_required",
                "rank": "",
                "scenario": row.get("scenario", ""),
                "method": row.get("method", ""),
                "value": True,
                "status": row.get("status", ""),
                "details": row.get("fallback_reason", ""),
            }
        )


def _append_negative_rows(
    worst_rows: list[dict[str, object]], summary_rows: list[dict[str, object]]
) -> None:
    """negative scenario の A2 row を worst_rows へ追加する。"""
    for row in summary_rows:
        if not str(row.get("scenario_group", "")).startswith("negative"):
            continue
        if str(row.get("method", "")) == "fixed_baseline":
            continue
        worst_rows.append(
            {
                "category": "negative_case_rows",
                "metric": "scenario_group",
                "rank": "",
                "scenario": row.get("scenario", ""),
                "method": row.get("method", ""),
                "value": row.get("scenario_group", ""),
                "status": row.get("status", ""),
                "details": (
                    f"outside_source_count={row.get('mask_outside_source_count', '')}; "
                    f"outside_suppression={row.get('max_mask_outside_source_suppression_db', '')}"
                ),
            }
        )


def _append_a2_difference_rows(
    worst_rows: list[dict[str, object]], summary_rows: list[dict[str, object]]
) -> None:
    """A2_safe / A2_aggressive の差が大きい row を追加する。"""
    metrics = (
        "non_source_p95_level_delta_db",
        "non_source_global_peak_delta_db",
        "source_peak_delta_db",
        "max_local_worsening_db_gated",
        "max_mask_outside_source_suppression_db",
    )
    by_scenario: dict[str, dict[str, dict[str, object]]] = {}
    for row in summary_rows:
        by_scenario.setdefault(str(row.get("scenario", "")), {})[str(row.get("method", ""))] = row

    diff_rows: list[dict[str, object]] = []
    for scenario, method_rows in by_scenario.items():
        safe = method_rows.get("A2_safe")
        aggressive = method_rows.get("A2_aggressive")
        if safe is None or aggressive is None:
            continue
        for metric in metrics:
            safe_value = _safe_float(safe.get(metric))
            aggressive_value = _safe_float(aggressive.get(metric))
            diff_rows.append(
                {
                    "category": "a2_safe_aggressive_large_difference",
                    "metric": metric,
                    "rank": "",
                    "scenario": scenario,
                    "method": "A2_safe_vs_A2_aggressive",
                    "value": abs(aggressive_value - safe_value),
                    "status": (
                        f"safe={safe.get('status', '')}; "
                        f"aggressive={aggressive.get('status', '')}"
                    ),
                    "details": f"safe={safe_value:.6f}; aggressive={aggressive_value:.6f}",
                }
            )
    ranked = sorted(diff_rows, key=lambda row: _safe_float(row.get("value")), reverse=True)
    for rank, row in enumerate(ranked[:10], start=1):
        row["rank"] = rank
        worst_rows.append(row)


def _build_worst_case_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """worst_cases.csv の row を作る。"""
    worst_rows: list[dict[str, object]] = []
    _append_metric_worst_rows(worst_rows, summary_rows)
    _append_detected_mismatch_rows(worst_rows, summary_rows)
    _append_fallback_rows(worst_rows, summary_rows)
    _append_negative_rows(worst_rows, summary_rows)
    _append_a2_difference_rows(worst_rows, summary_rows)
    return worst_rows


def _relative(path: Path) -> str:
    """review_pack からの相対 path を返す。"""
    return path.relative_to(REVIEW_PACK_DIR).as_posix()


def _scenario_summary_lines(data: ScenarioReviewData) -> list[str]:
    """review_index.md の scenario 節を作る。"""
    lines = [
        f"### {data.scenario.scenario_id}",
        "",
        f"- 目的: {data.scenario.selection_basis}",
        f"- scenario group: `{data.scenario.scenario_group}`",
        f"- mask: `{data.scenario.mask_type}`",
        f"- source: {_source_caption(data.scenario)}",
        f"- mask beams: {_mask_caption(data)}",
        "",
        "| method | candidate | status | p95 delta | source peak delta | fallback |",
        "|---|---|---|---:|---:|---|",
    ]
    for method_id in METHOD_ORDER:
        row = _effective_row(data.rows, method_id)
        lines.append(
            (
                "| `{method}` | `{candidate}` | `{status}` | {p95:.3f} | "
                "{source_delta:.3f} | {fallback} |"
            ).format(
                method=method_id,
                candidate=data.evaluations[method_id].candidate.candidate_id,
                status=str(row.get("status", "")),
                p95=_safe_float(row.get("non_source_p95_level_delta_db")),
                source_delta=_safe_float(row.get("max_abs_source_peak_delta_db")),
                fallback=str(row.get("fallback_required", "")),
            )
        )
    scenario_dir = FIGURE_DIR / data.scenario.scenario_id
    npz_path = DATA_DIR / f"{data.scenario.scenario_id}.npz"
    lines.extend(
        [
            "",
            "参照:",
            f"- BL overlay: `{_relative(scenario_dir / 'bl_overlay.png')}`",
            f"- BL delta: `{_relative(scenario_dir / 'bl_delta.png')}`",
            f"- FRAZ delta: `{_relative(scenario_dir / 'fraz_delta.png')}`",
            f"- BTR panel: `{_relative(scenario_dir / 'btr_panel.png')}`",
            f"- plot arrays: `{_relative(npz_path)}`",
            f"- scenario CSV: `{_relative(SCENARIO_SUMMARY_PATH)}`",
            f"- worst cases CSV: `{_relative(WORST_CASES_PATH)}`",
            "",
        ]
    )
    return lines


def _write_review_index(data_items: list[ScenarioReviewData]) -> None:
    """review_index.md を保存する。"""
    lines: list[str] = [
        "# Lightweight ABF-like Review Pack",
        "",
        "この review_pack は ChatGPT レビュー用に、fixed_baseline / A2_safe / "
        "A2_aggressive の effective 出力だけを横並びにしたものです。",
        "",
        "## 読み方",
        "",
        "- 採否は raw ではなく effective のみで判断する。",
        "- fixed_baseline は常に fallback として残す。",
        "- BL / FRAZ delta は fixed_baseline に対する dB 差分で読む。",
        "- BTR は dB re frame max であり、抑圧量の定量比較ではなく source track の連続性確認用。",
        "- すべての図で source mask と non-source sector を背景色で表示する。",
        "",
        "## ファイル",
        "",
        f"- scenario summary: `{_relative(SCENARIO_SUMMARY_PATH)}`",
        f"- worst cases: `{_relative(WORST_CASES_PATH)}`",
        "- figures: `figures/<scenario>/`",
        "- plot arrays: `data/<scenario>.npz`",
        "",
        "## Scenarios",
        "",
    ]
    for data in data_items:
        lines.extend(_scenario_summary_lines(data))
    REVIEW_INDEX_PATH.write_text("\n".join(lines), encoding="utf-8")


def _build_review_pack() -> None:
    """review_pack を生成する。"""
    if REVIEW_PACK_DIR.exists():
        shutil.rmtree(REVIEW_PACK_DIR)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    require_matplotlib()
    assets = _load_assets()
    data_items: list[ScenarioReviewData] = []
    for scenario in _scenario_definitions():
        data = _evaluate_review_scenario(scenario, assets)
        data_items.append(data)
        scenario_dir = FIGURE_DIR / scenario.scenario_id
        _plot_bl_overlay(data, scenario_dir / "bl_overlay.png")
        _plot_bl_delta(data, scenario_dir / "bl_delta.png")
        _plot_fraz_delta(data, scenario_dir / "fraz_delta.png")
        _plot_btr_panel(data, scenario_dir / "btr_panel.png")
        _save_npz(data, DATA_DIR / f"{scenario.scenario_id}.npz")

    scenario_rows = _build_scenario_summary_rows(data_items)
    worst_rows = _build_worst_case_rows(scenario_rows)
    _write_csv(scenario_rows, SCENARIO_SUMMARY_PATH, SUMMARY_COLUMNS)
    _write_csv(worst_rows, WORST_CASES_PATH, None)
    _write_review_index(data_items)


def main() -> None:
    """CLI entrypoint。"""
    _build_review_pack()
    print(f"saved review index to {REVIEW_INDEX_PATH}")
    print(f"saved scenario summary to {SCENARIO_SUMMARY_PATH}")
    print(f"saved worst cases to {WORST_CASES_PATH}")
    print(f"saved figures and npz under {REVIEW_PACK_DIR}")


if __name__ == "__main__":
    main()

