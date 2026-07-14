"""運用スパースアレイで時間領域SLCが使える条件をsafety gate無効で調べる。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from spflow.beamforming import (
    OperationalTimeDomainSlcDiagnosticConfig,
    SlcConfig,
    run_operational_time_domain_slc_leakage_diagnostics,
)


def _require_float_levels(summary: dict[str, object]) -> dict[str, float]:
    """診断 summary から dB metric 辞書を型検証して取り出す。

    Args:
        summary: `run_operational_time_domain_slc_leakage_diagnostics()` の戻り値。

    Returns:
        `levels` の各 metric を `dict[str, float]` として返す。

    Raises:
        TypeError: `levels` が mapping でない、key が文字列でない、または値が数値でない場合。

    境界条件:
        summary は JSON 出力と同じ構造を `dict[str, object]` で返すため、
        Pyright 上も実行時上もここで metric 辞書の型を確定してから評価へ渡す。
    """
    levels_value = summary.get("levels")
    if not isinstance(levels_value, Mapping):
        raise TypeError("summary['levels'] must be a mapping.")

    levels: dict[str, float] = {}
    for key, value in levels_value.items():
        if not isinstance(key, str):
            raise TypeError("summary['levels'] keys must be strings.")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"summary['levels'][{key!r}] must be numeric.")
        levels[key] = float(value)
    return levels


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    """summary の入れ子 mapping を実行時検証して返す。

    Args:
        value: summary から取り出した値。
        name: エラー表示用の項目名。

    Returns:
        文字列 key を持つ mapping。

    Raises:
        TypeError: 値が mapping でない、または key が文字列でない場合。

    境界条件:
        SLC 評価では `levels` 以外に `slc_process` と `capacity` も参照する。
        JSON 境界の値を `object` のまま使わず、ここで shape の代わりに構造を確定する。
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    for key in value.keys():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings.")
    return value


def _require_number(value: object, name: str) -> float:
    """summary の数値 metric を Python float として取り出す。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    return float(value)


def _classify_case(levels: dict[str, float], *, has_effective_interferer: bool) -> dict[str, float | bool | str]:
    """raw SLC の target 維持と interferer 抑圧から利用可否を判定する。

    Args:
        levels: `run_operational_time_domain_slc_leakage_diagnostics()` の `levels`。
            dB20 レベルと before/after 差分を含む。
        has_effective_interferer: interferer を実質的に入れている場合は `True`。

    Returns:
        判定結果。target 自己消去、干渉悪化、利用可否を含む。
    """
    raw_target_delta_db = float(levels["raw_target_power_delta_db"])
    raw_interferer_reduction_db = float(levels["raw_interferer_reduction_db"])

    # target が 1.5 dB 以上落ちる場合は、SLC が desired 成分を学習している可能性が高い。
    # interferer がない条件では、この target 維持だけを利用可否の主判定にする。
    target_preserved = bool(raw_target_delta_db >= -1.5)
    interferer_reduced = bool(raw_interferer_reduction_db >= 3.0) if has_effective_interferer else True
    raw_slc_usable = bool(target_preserved and interferer_reduced)

    if not target_preserved:
        reason = "target_self_nulling"
    elif has_effective_interferer and not interferer_reduced:
        reason = "insufficient_or_negative_interference_reduction"
    else:
        reason = "usable_in_this_case"

    return {
        "target_preserved": target_preserved,
        "interferer_reduced": interferer_reduced,
        "raw_slc_usable": raw_slc_usable,
        "reason": reason,
        "raw_target_power_delta_db": raw_target_delta_db,
        "raw_interferer_reduction_db": raw_interferer_reduction_db,
    }


def main() -> None:
    """target-only / 同一周波数 / 異周波数 interferer 条件を比較する。"""
    output_root = Path("artifacts/beamforming/operational_time_domain_slc_condition_sweep")
    array_path = Path("artifacts/beamforming/operational_sparse_array/operational_sparse_array_fs32768.json")
    filter_bank_path = Path("artifacts/beamforming/fractional_delay_filter_bank_65x63.npz")

    cases = (
        {
            "case_name": "target_only_high_snr",
            "interferer_frequency_hz": 10000.0,
            "interferer_level_db20": -300.0,
            "has_effective_interferer": False,
        },
        {
            "case_name": "same_frequency_interferer_m6dB",
            "interferer_frequency_hz": 10000.0,
            "interferer_level_db20": -6.0,
            "has_effective_interferer": True,
        },
        {
            "case_name": "different_frequency_interferer_8192Hz_m6dB",
            "interferer_frequency_hz": 8192.0,
            "interferer_level_db20": -6.0,
            "has_effective_interferer": True,
        },
        {
            "case_name": "different_frequency_interferer_6144Hz_0dB",
            "interferer_frequency_hz": 6144.0,
            "interferer_level_db20": 0.0,
            "has_effective_interferer": True,
        },
    )

    tap_lengths = (1, 3, 5, 8)

    case_summaries: list[dict[str, object]] = []
    for tap_len in tap_lengths:
        for case in cases:
            case_name = f"{str(case['case_name'])}_L{int(tap_len)}"

            # ここでは「悪化時に固定整相へ戻る」安全機構を無効化し、raw SLC の素性を見る。
            # L>1 は短い FIR 型キャンセラなので、時間領域で吸収できる相対遅延が増えるかを確認する。
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
                    duration_s=1.0,
                    n_beam_az_real=151,
                ),
                slc_config=SlcConfig(
                    guard=10,
                    loading=3.0e-2,
                    memory_time_sec=3.0,
                    heading_scale_deg=5.0,
                    min_ref=8,
                    sample_per_dof=5.0,
                    tap_len=int(tap_len),
                    eta_normal=1.0,
                    eta_limited=1.0,
                    enable_heading_forgetting=False,
                    enable_output_safety_gate=False,
                ),
            )

            levels = _require_float_levels(summary)
            slc_process = _require_mapping(summary["slc_process"], "slc_process")
            capacity = _require_mapping(slc_process["capacity"], "slc_process.capacity")
            classification = _classify_case(
                levels=levels,
                has_effective_interferer=bool(case["has_effective_interferer"]),
            )
            case_summary: dict[str, object] = {
                "case_name": case_name,
                "tap_len": int(tap_len),
                "target_frequency_hz": 10000.0,
                "interferer_frequency_hz": float(case["interferer_frequency_hz"]),
                "interferer_level_db20": float(case["interferer_level_db20"]),
                "has_effective_interferer": bool(case["has_effective_interferer"]),
                "classification": classification,
                "condition_number": _require_number(slc_process["condition_number"], "condition_number"),
                "weight_norm": _require_number(slc_process["weight_norm"], "weight_norm"),
                "reference_beam_count": int(_require_number(slc_process["reference_beam_count"], "reference_beam_count")),
                "capacity_is_feasible": bool(capacity["is_feasible"]),
                "slc_elapsed_sec": _require_number(slc_process["elapsed_sec"], "elapsed_sec"),
                "slc_realtime_factor": _require_number(slc_process["realtime_factor"], "realtime_factor"),
                "summary_path": str((output_root / case_name / "time_domain_slc_leakage_summary.json").resolve()),
                "target_leakage_levels_png_path": str((output_root / case_name / "target_leakage_levels.png").resolve()),
            }
            case_summaries.append(case_summary)

    output_root.mkdir(parents=True, exist_ok=True)
    sweep_summary = {
        "safety_gate": "disabled",
        "target_frequency_hz": 10000.0,
        "target_azimuth_deg": 90.0,
        "interferer_azimuth_deg": 60.0,
        "tap_lengths": [int(value) for value in tap_lengths],
        "case_summaries": case_summaries,
    }
    (output_root / "condition_sweep_summary.json").write_text(
        json.dumps(sweep_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(sweep_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
