"""運用スパースアレイで時間領域 SLC の tap 長と eta 依存性を調べる。"""

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
        eta / tap 長 sweep では summary の metric をそのまま比較表に使う。
        JSON 境界の `object` を未確認で添字アクセスすると、Pylance が型を確定できないためここで検証する。
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
        eta sweep では target / interferer の level だけでなく、係数健全性と runtime も比較する。
        `dict[str, object]` の境界で構造を確定し、Pylance が object 添字アクセスを警告しない形にする。
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


def main() -> None:
    """safety gate を無効化し、tap 長と eta ごとの target 維持と干渉低減を保存する。"""
    output_root = Path("artifacts/beamforming/operational_time_domain_slc_eta_probe")
    array_path = Path("artifacts/beamforming/operational_sparse_array/operational_sparse_array_fs32768.json")
    filter_bank_path = Path("artifacts/beamforming/fractional_delay_filter_bank_65x63.npz")

    cases = (
        ("target_only", 10000.0, -300.0),
        ("same_freq_m6", 10000.0, -6.0),
        ("diff8192_m6", 8192.0, -6.0),
        ("diff6144_0", 6144.0, 0.0),
    )
    eta_values = (0.25, 0.5, 0.75, 1.0)
    tap_lengths = (1, 3, 5, 8)

    case_summaries: list[dict[str, object]] = []
    for tap_len in tap_lengths:
        for eta in eta_values:
            for case_name, interferer_frequency_hz, interferer_level_db20 in cases:
                output_dir = output_root / f"{case_name}_L{int(tap_len)}_eta{int(round(eta * 100.0))}"

                # eta は SLC 推定成分をどれだけ引くかを決める。
                # tap_len は u[n], u[n-1], ... を何 sample 分使うかを決める FIR 型自由度である。
                # safety gate を無効化し、raw SLC の target 低下と interferer 低減の素性を見る。
                summary = run_operational_time_domain_slc_leakage_diagnostics(
                    config=OperationalTimeDomainSlcDiagnosticConfig(
                        output_dir=output_dir,
                        operational_array_definition_path=array_path,
                        fractional_delay_filter_bank_path=filter_bank_path,
                        processing_frequency_hz=10000.0,
                        interferer_frequency_hz=float(interferer_frequency_hz),
                        target_azimuth_deg=90.0,
                        interferer_azimuth_deg=60.0,
                        target_level_db20=0.0,
                        interferer_level_db20=float(interferer_level_db20),
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
                        eta_normal=float(eta),
                        eta_limited=float(eta),
                        enable_heading_forgetting=False,
                        enable_output_safety_gate=False,
                    ),
                )
                levels = _require_float_levels(summary)
                slc_process = _require_mapping(summary["slc_process"], "slc_process")
                capacity = _require_mapping(slc_process["capacity"], "slc_process.capacity")
                case_summaries.append(
                    {
                        "case_name": f"{case_name}_L{int(tap_len)}_eta{int(round(eta * 100.0))}",
                        "tap_len": int(tap_len),
                        "eta": float(eta),
                        "target_frequency_hz": 10000.0,
                        "interferer_frequency_hz": float(interferer_frequency_hz),
                        "interferer_level_db20": float(interferer_level_db20),
                        "raw_target_power_delta_db": float(levels["raw_target_power_delta_db"]),
                        "raw_interferer_reduction_db": float(levels["raw_interferer_reduction_db"]),
                        "condition_number": _require_number(slc_process["condition_number"], "condition_number"),
                        "weight_norm": _require_number(slc_process["weight_norm"], "weight_norm"),
                        "capacity_is_feasible": bool(capacity["is_feasible"]),
                        "slc_elapsed_sec": _require_number(slc_process["elapsed_sec"], "elapsed_sec"),
                        "slc_realtime_factor": _require_number(slc_process["realtime_factor"], "realtime_factor"),
                        "summary_path": str((output_dir / "time_domain_slc_leakage_summary.json").resolve()),
                        "target_leakage_levels_png_path": str((output_dir / "target_leakage_levels.png").resolve()),
                    }
                )

    output_root.mkdir(parents=True, exist_ok=True)
    sweep_summary = {
        "safety_gate": "disabled",
        "uses_desired_response_blocking": True,
        "eta_values": [float(value) for value in eta_values],
        "tap_lengths": [int(value) for value in tap_lengths],
        "case_summaries": case_summaries,
    }
    (output_root / "eta_probe_summary.json").write_text(
        json.dumps(sweep_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(sweep_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
