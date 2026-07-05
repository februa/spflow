"""運用スパースアレイで小数遅延固定整相 + SLC の BL/FRAZ/BTR を保存する。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spflow.beamforming import (
    SlcConfig,
    TimeDelayDiagnosticConfig,
    TimeDelayDiagnosticSource,
    load_operational_shading,
    load_operational_sparse_array,
)
from spflow.beamforming.fractional_delay_slc_diagnostics import run_fractional_delay_slc_diagnostics


def main() -> None:
    """運用アレイ JSON と保存済み小数遅延 FIR バンクを使い、高域 SLC 診断を実行する。

    入力:
        - 運用スパースアレイ JSON。物理 CH 数と周波数別 active CH をこのファイルから読む。
        - 151 本ビーム固定のシェーディング JSON。現状 beta=0 のため矩形窓と等価である。
        - 保存済み小数遅延 FIR バンク。

    出力:
        `artifacts/beamforming/operational_fractional_delay_slc_diagnostics/10000Hz_151beam`
        以下へ、固定整相 before と SLC after の BL/FRAZ/BTR、BL 比較図、summary JSON を保存する。

    境界条件:
        シェーディング係数が active CH 内で非一様な場合、このスクリプトは停止する。
        現在の `run_fractional_delay_slc_diagnostics()` は channel shading を整相器へ渡さないため、
        非ゼロ beta の評価では weighted delay-and-sum 実装を追加してから使う必要がある。
    """
    array_definition_path = Path("artifacts/beamforming/operational_sparse_array/operational_sparse_array_fs32768.json")
    shading_definition_path = Path("artifacts/beamforming/operational_shading_fixed_beam/operational_kaiser_bessel_shading_151beam.json")
    fractional_delay_filter_bank_path = Path("artifacts/beamforming/fractional_delay_filter_bank_65x63.npz")
    output_dir = Path("artifacts/beamforming/operational_fractional_delay_slc_diagnostics/10000Hz_151beam")

    if not fractional_delay_filter_bank_path.exists():
        raise FileNotFoundError(
            "小数遅延 FIR バンクが見つかりません。"
            " examples/beamforming/design_fractional_delay_filter_bank.py で 65x63 条件のバンクを作成してください。"
        )

    array_definition = load_operational_sparse_array(array_definition_path)
    shading_definition = load_operational_shading(shading_definition_path)

    target_frequency_hz = 10000.0
    active_indices = array_definition.active_channel_indices_for_frequency(target_frequency_hz)
    shading_active_indices = shading_definition.active_channel_indices_for_frequency(target_frequency_hz)
    if not np.array_equal(active_indices, shading_active_indices):
        raise ValueError("アレイ定義とシェーディング定義の active channel index が一致しません。")

    full_shading_weights = shading_definition.coefficients_for_frequency(target_frequency_hz)
    active_shading_weights = np.asarray(full_shading_weights[active_indices], dtype=np.float64)
    if not bool(np.allclose(active_shading_weights, active_shading_weights[0], rtol=0.0, atol=1.0e-12)):
        raise ValueError(
            "この診断は beta=0 の固定 151 本ビーム用シェーディングだけを対象にします。"
            " 非一様シェーディングを使う場合は、channel 加重付き小数遅延整相へ拡張してください。"
        )

    # positions_m は物理 125 CH 全体の shape `[n_ch, 3]` である。
    # 高域では外側の疎配置を active から外す設計なので、10000 Hz 行の active index で ch 軸を抽出する。
    active_positions_m = np.asarray(array_definition.positions_m[active_indices], dtype=np.float64)
    active_aperture_m = float(np.max(active_positions_m[:, 0]) - np.min(active_positions_m[:, 0]))

    # 同一周波数・異方位の interferer を入れると、FRAZ では周波数で分離できない。
    # ここでは target を 90 deg、interferer を 60 deg に置き、SLC が target mainlobe を守りつつ
    # guard 外応答を下げられるかを BL 比較と BTR で確認する。
    summary = run_fractional_delay_slc_diagnostics(
        config=TimeDelayDiagnosticConfig(
            output_dir=output_dir,
            fs_hz=float(array_definition.fs_hz),
            duration_s=1.0,
            sound_speed_m_s=float(array_definition.sound_speed_m_s),
            source_specs=(
                TimeDelayDiagnosticSource(
                    azimuth_deg=90.0,
                    frequency_hz=target_frequency_hz,
                    level_db20=0.0,
                    amplitude_modulation_hz=0.7,
                    amplitude_modulation_depth=0.6,
                    label="target_10000Hz_090deg",
                ),
                TimeDelayDiagnosticSource(
                    azimuth_deg=60.0,
                    frequency_hz=target_frequency_hz,
                    level_db20=-6.0,
                    amplitude_modulation_hz=1.3,
                    amplitude_modulation_depth=0.8,
                    amplitude_modulation_phase_deg=70.0,
                    label="interferer_10000Hz_060deg",
                ),
            ),
            noise_level_db20=-80.0,
            random_seed=1234,
            array_n_ch=int(active_positions_m.shape[0]),
            array_positions_m=active_positions_m,
            n_beam_az_real=151,
            n_beam_az_virtual=0,
            btr_block_size=1024,
        ),
        slc_config=SlcConfig(
            guard=10,
            loading=3.0e-2,
            memory_time_sec=1.0,
            heading_scale_deg=5.0,
            min_ref=8,
            sample_per_dof=5.0,
            tap_len=1,
            eta_normal=0.25,
            eta_limited=0.15,
            enable_heading_forgetting=False,
        ),
        fractional_delay_filter_bank_path=fractional_delay_filter_bank_path,
        target_source_indices=(0,),
        slc_analysis_block_size=64,
        max_reference_beams=48,
    )

    case_summary: dict[str, object] = {
        "array_definition_path": str(array_definition_path.resolve()),
        "shading_definition_path": str(shading_definition_path.resolve()),
        "fractional_delay_filter_bank_path": str(fractional_delay_filter_bank_path.resolve()),
        "frequency_hz": float(target_frequency_hz),
        "active_channel_count": int(active_indices.size),
        "active_aperture_m": active_aperture_m,
        "n_beam_az_real": 151,
        "slc_guard_beam": 10,
        "max_reference_beams": 48,
        "summary_path": str((output_dir / "slc_summary.json").resolve()),
        "fixed_bl_png_path": str((output_dir / "bl_00_target_10000Hz_090deg.png").resolve()),
        "fixed_fraz_png_path": str((output_dir / "fraz.png").resolve()),
        "fixed_btr_png_path": str((output_dir / "btr.png").resolve()),
        "slc_bl_png_path": str((output_dir / "slc_bl.png").resolve()),
        "slc_bl_compare_png_path": str((output_dir / "slc_bl_compare.png").resolve()),
        "slc_fraz_png_path": str((output_dir / "slc_fraz.png").resolve()),
        "slc_btr_png_path": str((output_dir / "slc_btr.png").resolve()),
        "all_mainlobes_preserved": bool(summary["all_mainlobes_preserved"]),
        "mean_mainlobe_level_delta_db": float(summary["mean_mainlobe_level_delta_db"]),
        "mean_sidelobe_reduction_db": float(summary["mean_sidelobe_reduction_db"]),
        "mean_mainlobe_margin_improvement_db": float(summary["mean_mainlobe_margin_improvement_db"]),
        "slc_design_summary": summary["slc_design_summary"],
        "source_comparisons": summary["source_comparisons"],
        "interference_source_comparisons": summary["interference_source_comparisons"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "operational_slc_case_summary.json").write_text(
        json.dumps(case_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(case_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
