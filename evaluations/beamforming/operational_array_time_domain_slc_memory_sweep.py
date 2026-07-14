"""時間領域SLCの共分散積分時間を評価する。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from evaluations.beamforming.scenarios.operational_time_domain_slc_diagnostics import (
    OperationalTimeDomainSlcDiagnosticConfig,
    run_operational_time_domain_slc_leakage_diagnostics,
)
from spflow.sidelobe_cancellation import SlcConfig


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    """JSON 境界の object 値を mapping として検証して返す。

    Args:
        value: summary から取り出した値。
        name: エラー表示用の項目名。

    Returns:
        文字列 key を持つ mapping。

    Raises:
        TypeError: 値が mapping でない、または key が文字列でない場合。

    境界条件:
        診断 summary は `dict[str, object]` で返るため、Pylance / Pyright 上も
        入れ子構造をここで確定してから数値 metric を読む。
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    for key in value.keys():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings.")
    return value


def _require_number(value: object, name: str) -> float:
    """summary 内の scalar 数値を Python float として取り出す。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    return float(value)


def _require_optional_number(value: object, name: str) -> float | None:
    """summary 内の None 許容 scalar 数値を Python float として取り出す。"""
    if value is None:
        return None
    return _require_number(value, name)


def _relative_std(stats: Mapping[str, object], name: str) -> float | None:
    """block 間統計から相対標準偏差を計算する。

    Args:
        stats: `block_condition_number_stats` などの統計辞書。
        name: エラー表示用の項目名。

    Returns:
        `std / abs(mean)`。mean が 0 または統計量が存在しない場合は `None`。

    境界条件:
        条件数や重みノルムの絶対値は条件によって桁が変わるため、
        積分時間の安定性比較では標準偏差を平均値で割った無次元量も併記する。
    """
    mean_value = _require_optional_number(stats.get("mean"), f"{name}.mean")
    std_value = _require_optional_number(stats.get("std"), f"{name}.std")
    if mean_value is None or std_value is None or abs(mean_value) <= 0.0:
        return None
    return float(std_value / abs(mean_value))


def _case_metrics(summary: dict[str, object], *, has_effective_interferer: bool) -> dict[str, object]:
    """SLC summary から積分時間評価用 metric を抽出する。

    Args:
        summary: `run_operational_time_domain_slc_leakage_diagnostics()` の戻り値。
        has_effective_interferer: interferer を評価対象に含める場合は `True`。

    Returns:
        target 保護、interferer 低減、共分散健全性、runtime の metric 辞書。

    境界条件:
        `dB` 差分はすべて `dB re before level` として読む。
        絶対 RMS レベルは元 summary の `level_reference = dB re input RMS` に従う。
    """
    levels = _require_mapping(summary["levels"], "levels")
    slc_process = _require_mapping(summary["slc_process"], "slc_process")
    covariance_memory = _require_mapping(slc_process["covariance_memory"], "slc_process.covariance_memory")
    condition_stats = _require_mapping(
        slc_process["block_condition_number_stats"],
        "slc_process.block_condition_number_stats",
    )
    weight_stats = _require_mapping(slc_process["block_weight_norm_stats"], "slc_process.block_weight_norm_stats")

    raw_target_delta_db = _require_number(levels["raw_target_power_delta_db"], "raw_target_power_delta_db")
    raw_interferer_reduction_db = _require_number(
        levels["raw_interferer_reduction_db"],
        "raw_interferer_reduction_db",
    )
    effective_block_count = _require_optional_number(
        covariance_memory.get("asymptotic_effective_independent_block_count"),
        "asymptotic_effective_independent_block_count",
    )
    enabled_duration_sec = _require_number(covariance_memory["enabled_duration_sec"], "enabled_duration_sec")
    memory_time_sec = _require_number(covariance_memory["memory_time_sec"], "memory_time_sec")
    condition_relative_std = _relative_std(condition_stats, "condition_stats")
    weight_relative_std = _relative_std(weight_stats, "weight_stats")

    # target 低下 1.5 dB は、方式比較時に desired 自己消去を疑う実用上の警戒線である。
    # interferer 低減 3 dB は、SLC を入れる価値がある最低限の改善幅として扱う。
    target_preserved = bool(raw_target_delta_db >= -1.5)
    interferer_reduced = bool(raw_interferer_reduction_db >= 3.0) if has_effective_interferer else True

    # 3 tau 以上観測できると指数応答は約 95% まで進む。
    # これ未満では、最初の block 初期化の影響が残る可能性があるため、積分時間評価で分けて読む。
    observed_at_least_three_tau = bool(enabled_duration_sec >= 3.0 * memory_time_sec)
    has_three_effective_blocks = bool(effective_block_count is not None and effective_block_count >= 3.0)

    return {
        "raw_target_power_delta_db_re_before": raw_target_delta_db,
        "raw_interferer_reduction_db_re_before": raw_interferer_reduction_db,
        "target_preserved": target_preserved,
        "interferer_reduced": interferer_reduced,
        "raw_slc_usable": bool(target_preserved and interferer_reduced),
        "memory_time_sec": memory_time_sec,
        "alpha": _require_optional_number(covariance_memory.get("alpha"), "alpha"),
        "e_folding_block_count": _require_optional_number(
            covariance_memory.get("e_folding_block_count"),
            "e_folding_block_count",
        ),
        "asymptotic_effective_independent_block_count": effective_block_count,
        "enabled_duration_sec": enabled_duration_sec,
        "observed_at_least_three_tau": observed_at_least_three_tau,
        "has_three_effective_blocks": has_three_effective_blocks,
        "condition_number_mean": _require_optional_number(condition_stats.get("mean"), "condition_stats.mean"),
        "condition_number_std": _require_optional_number(condition_stats.get("std"), "condition_stats.std"),
        "condition_number_relative_std": condition_relative_std,
        "weight_norm_mean": _require_optional_number(weight_stats.get("mean"), "weight_stats.mean"),
        "weight_norm_std": _require_optional_number(weight_stats.get("std"), "weight_stats.std"),
        "weight_norm_relative_std": weight_relative_std,
        "slc_realtime_factor": _require_number(slc_process["realtime_factor"], "realtime_factor"),
        "summary_path": str(summary["summary_path"]) if isinstance(summary.get("summary_path"), str) else "",
    }


def main() -> None:
    """1 秒から 5 秒の memory_time_sec を振り、共分散積分時間が SLC 指標へ与える影響を評価する。"""
    output_root = Path("artifacts/beamforming/operational_time_domain_slc_memory_sweep")
    array_path = Path("artifacts/beamforming/operational_sparse_array/operational_sparse_array_fs32768.json")
    filter_bank_path = Path("artifacts/beamforming/fractional_delay_filter_bank_65x63.npz")

    memory_times_sec = (1.0, 2.0, 3.0, 5.0)
    cases = (
        {
            "case_name": "target_only_high_snr",
            "interferer_frequency_hz": 10000.0,
            "interferer_level_db20": -300.0,
            "has_effective_interferer": False,
        },
        {
            "case_name": "different_frequency_interferer_8192Hz_m6dB",
            "interferer_frequency_hz": 8192.0,
            "interferer_level_db20": -6.0,
            "has_effective_interferer": True,
        },
    )

    case_summaries: list[dict[str, object]] = []
    for memory_time_sec in memory_times_sec:
        for case in cases:
            case_name = f"{str(case['case_name'])}_memory{float(memory_time_sec):.4f}s"

            # 積分時間だけの影響を見るため、tap_len は既定候補の L=1 に固定する。
            # safety gate は無効化し、raw SLC の target 保護と interferer 低減を直接評価する。
            # 5 秒 memory は 5 秒評価では 3 tau まで観測できないため、startup 条件として明示して読む。
            summary = run_operational_time_domain_slc_leakage_diagnostics(
                config=OperationalTimeDomainSlcDiagnosticConfig(
                    output_dir=output_root / case_name,
                    operational_array_definition_path=array_path,
                    fractional_delay_filter_bank_path=filter_bank_path,
                    processing_frequency_hz=10000.0,
                    interferer_frequency_hz=float(case["interferer_frequency_hz"]),
                    target_azimuth_deg=90.0,
                    interferer_azimuth_deg=60.0,
                    target_level_db20=0.0,
                    interferer_level_db20=float(case["interferer_level_db20"]),
                    duration_s=5.0,
                    n_beam_az_real=151,
                    slc_analysis_block_size=8192,
                    noise_level_db20=-60.0,
                ),
                slc_config=SlcConfig(
                    guard=10,
                    loading=3.0e-2,
                    memory_time_sec=float(memory_time_sec),
                    heading_scale_deg=5.0,
                    min_ref=8,
                    sample_per_dof=5.0,
                    tap_len=1,
                    eta_normal=1.0,
                    eta_limited=1.0,
                    enable_heading_forgetting=False,
                    enable_output_safety_gate=False,
                ),
            )
            summary["summary_path"] = str((output_root / case_name / "time_domain_slc_leakage_summary.json").resolve())
            case_summaries.append(
                {
                    "case_name": case_name,
                    "source_case_name": str(case["case_name"]),
                    "has_effective_interferer": bool(case["has_effective_interferer"]),
                    "interferer_frequency_hz": float(case["interferer_frequency_hz"]),
                    "interferer_level_db20": float(case["interferer_level_db20"]),
                    "metrics": _case_metrics(summary, has_effective_interferer=bool(case["has_effective_interferer"])),
                }
            )

    output_root.mkdir(parents=True, exist_ok=True)
    sweep_summary = {
        "evaluation_patterns": ["slc_target_only", "slc_different_frequency_interference", "slc_runtime"],
        "level_reference": "absolute RMS levels are dB re input RMS; before/after deltas are dB re before level",
        "tap_len": 1,
        "slc_analysis_block_size": 8192,
        "duration_s": 5.0,
        "noise_level_db20": -60.0,
        "memory_times_sec": [float(value) for value in memory_times_sec],
        "case_summaries": case_summaries,
    }
    (output_root / "memory_sweep_summary.json").write_text(
        json.dumps(sweep_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(sweep_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
