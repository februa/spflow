"""運用スパースアレイ定義を使い、小数遅延固定整相の BL/FRAZ/BTR を保存する。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spflow.beamforming import TimeDelayDiagnosticConfig, TimeDelayDiagnosticSource, load_operational_sparse_array
from spflow.beamforming.fractional_delay_slc_diagnostics import _run_fractional_delay_diagnostics


def main() -> None:
    """運用アレイ JSON の active channel subset で小数遅延時間領域診断を実行する。"""
    array_definition_path = Path("artifacts/beamforming/operational_sparse_array/operational_sparse_array_fs32768.json")
    fractional_delay_filter_bank_path = Path("artifacts/beamforming/fractional_delay_filter_bank_65x63.npz")
    output_root = Path("artifacts/beamforming/operational_fractional_delay_diagnostics")

    array_definition = load_operational_sparse_array(array_definition_path)

    diagnostic_cases = (
        (256.0, 90.0, "low_broadside"),
        (10000.0, 60.0, "high_off_broadside"),
        (10000.0, 90.0, "high_broadside"),
    )
    case_summaries: list[dict[str, object]] = []

    for frequency_hz, azimuth_deg, label in diagnostic_cases:
        active_indices = array_definition.active_channel_indices_for_frequency(float(frequency_hz))

        # positions_m は物理 125 ch 全体の shape `[n_ch, 3]` である。
        # ここでは周波数ごとの active_indices で ch 軸を抽出し、
        # 高域で設計外の外側疎配置を混ぜない narrowband 診断条件にする。
        active_positions_m = np.asarray(array_definition.positions_m[active_indices], dtype=np.float64)
        output_dir = output_root / f"{int(frequency_hz):05d}Hz_{int(azimuth_deg):03d}deg_{label}"

        config = TimeDelayDiagnosticConfig(
            output_dir=output_dir,
            fs_hz=float(array_definition.fs_hz),
            duration_s=1.0,
            sound_speed_m_s=float(array_definition.sound_speed_m_s),
            source_specs=(
                TimeDelayDiagnosticSource(
                    azimuth_deg=float(azimuth_deg),
                    frequency_hz=float(frequency_hz),
                    level_db20=0.0,
                    label=label,
                ),
            ),
            noise_level_db20=-80.0,
            random_seed=1234,
            array_n_ch=int(active_positions_m.shape[0]),
            array_positions_m=active_positions_m,
            n_beam_az_real=303,
            n_beam_az_virtual=0,
            btr_block_size=1024,
        )

        summary, _, _, _, _, _ = _run_fractional_delay_diagnostics(
            config=config,
            fractional_delay_filter_bank_path=fractional_delay_filter_bank_path,
        )

        # 既存診断 summary は時間領域整相の結果を保存する。
        # 運用アレイとして再現可能にするため、使用した active channel 情報を別 summary に残す。
        case_summary: dict[str, object] = {
            "label": label,
            "frequency_hz": float(frequency_hz),
            "azimuth_deg": float(azimuth_deg),
            "active_channel_count": int(active_indices.size),
            "active_channel_indices": [int(index) for index in active_indices.tolist()],
            "active_aperture_m": float(np.max(active_positions_m[:, 0]) - np.min(active_positions_m[:, 0])),
            "summary_path": str((output_dir / "summary.json").resolve()),
            "bl_png_path": str((output_dir / "bl.png").resolve()),
            "fraz_png_path": str((output_dir / "fraz.png").resolve()),
            "btr_png_path": str((output_dir / "btr.png").resolve()),
            "bl_peak_azimuth_deg": float(summary["bl_peak_azimuth_deg"]),
            "fraz_global_peak_azimuth_deg": float(summary["fraz_global_peak_azimuth_deg"]),
            "fraz_global_peak_frequency_hz": float(summary["fraz_global_peak_frequency_hz"]),
            "btr_global_peak_azimuth_mean_deg": float(summary["btr_global_peak_azimuth_mean_deg"]),
            "btr_global_peak_azimuth_std_deg": float(summary["btr_global_peak_azimuth_std_deg"]),
        }
        (output_dir / "operational_case_summary.json").write_text(
            json.dumps(case_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        case_summaries.append(case_summary)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "operational_fractional_delay_diagnostics_summary.json").write_text(
        json.dumps(
            {
                "array_definition_path": str(array_definition_path.resolve()),
                "fractional_delay_filter_bank_path": str(fractional_delay_filter_bank_path.resolve()),
                "cases": case_summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
