"""運用スパースアレイの小数遅延固定整相とSLCについてBL、FRAZ、BTRを保存する。"""

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


def _require_summary_float(summary: dict[str, object], key: str) -> float:
    """診断 summary から実数値を型検証して取り出す。

    Args:
        summary: `run_fractional_delay_slc_diagnostics()` が返す summary。
        key: 取り出す実数指標名。

    Returns:
        Python の `float` に確定した指標値。

    Raises:
        TypeError: 指標値が実数でない場合。
    """
    value = summary[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{key} must be numeric.")
    return float(value)


def _require_summary_bool(summary: dict[str, object], key: str) -> bool:
    """診断 summary から bool 値を型検証して取り出す。

    Args:
        summary: `run_fractional_delay_slc_diagnostics()` が返す summary。
        key: 取り出す bool 指標名。

    Returns:
        Python の `bool` に確定した指標値。

    Raises:
        TypeError: 指標値が bool でない場合。
    """
    value = summary[key]
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be bool.")
    return bool(value)

def main() -> None:
    """運用アレイ JSON と保存済み小数遅延 FIR バンクを使い、高域 SLC 診断を実行する。

    入力:
        - 運用スパースアレイ JSON。`positions_m` の shape は `[n_physical_ch, 3]`、単位は m。
        - 周波数別 channel shading JSON。係数 shape は `[n_physical_ch]`。
        - 保存済み小数遅延 FIR バンク。

    出力:
        `artifacts/beamforming/operational_fractional_delay_slc_diagnostics/10000Hz_151beam`
        以下へ、固定整相 before と SLC after の BL/FRAZ/BTR、BL 比較図、summary JSON を保存する。

    境界条件:
        10000 Hz の active channel index が、アレイ定義と shading 定義で一致することを要求する。
        channel shading は固定整相出力と SLC 応答行列の両方へ同じ `sum(w_ch y_ch) / sum(w_ch)` 正規化で渡す。
        これにより、非一様 shading 条件でも SLC の desired response と評価対象の beam output を一致させる。
    """
    array_definition_path = Path("artifacts/beamforming/operational_sparse_array/operational_sparse_array_fs32768.json")
    shading_definition_path = Path("artifacts/beamforming/operational_shading/operational_kaiser_bessel_shading_fs32768.json")
    fractional_delay_filter_bank_path = Path("artifacts/beamforming/fractional_delay_filter_bank_65x63.npz")
    output_dir = Path("artifacts/beamforming/operational_fractional_delay_slc_diagnostics/10000Hz_151beam")

    if not fractional_delay_filter_bank_path.exists():
        raise FileNotFoundError(
            "小数遅延 FIR バンクが見つかりません。"
            " tools/design_fractional_delay_filter_bank.py で 65x63 条件のバンクを作成してください。"
        )

    array_definition = load_operational_sparse_array(array_definition_path)
    shading_definition = load_operational_shading(shading_definition_path)

    target_frequency_hz = 10000.0
    active_indices = array_definition.active_channel_indices_for_frequency(target_frequency_hz)
    shading_active_indices = shading_definition.active_channel_indices_for_frequency(target_frequency_hz)
    if not bool(np.array_equal(active_indices, shading_active_indices)):
        raise ValueError("アレイ定義とシェーディング定義の active channel index が一致しません。")

    full_shading_weights = shading_definition.coefficients_for_frequency(target_frequency_hz)
    active_shading_weights = np.asarray(full_shading_weights[active_indices], dtype=np.float64)
    active_weight_sum = float(np.sum(active_shading_weights))
    active_weight_power_sum = float(np.sum(active_shading_weights**2))
    if active_weight_sum <= 0.0 or active_weight_power_sum <= 0.0:
        raise ValueError("active channel の shading 係数和が正ではありません。")

    # active_positions_m shape: [n_active_ch, 3]、axis=0 は active channel、axis=1 は x/y/z 座標[m]である。
    # 10000 Hz では高域 grating lobe を避けるため、物理 305 CH 全体ではなく周波数別 active subset を使う。
    active_positions_m = np.asarray(array_definition.positions_m[active_indices], dtype=np.float64)
    active_aperture_m = float(np.max(active_positions_m[:, 0]) - np.min(active_positions_m[:, 0]))

    # 有効 channel 数は shading 後の白色雑音利得に対応する Kish の有効標本数 `(sum w)^2 / sum(w^2)` として記録する。
    # Beamforming Evaluation では channel 数だけでなく、shading による利得低下も SNR / BL 評価の前提条件として扱う。
    effective_channel_count = float((active_weight_sum * active_weight_sum) / active_weight_power_sum)

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
        channel_weights=active_shading_weights,
    )

    case_summary: dict[str, object] = {
        "array_definition_path": str(array_definition_path.resolve()),
        "shading_definition_path": str(shading_definition_path.resolve()),
        "fractional_delay_filter_bank_path": str(fractional_delay_filter_bank_path.resolve()),
        "frequency_hz": float(target_frequency_hz),
        "active_channel_count": int(active_indices.size),
        "active_aperture_m": active_aperture_m,
        "active_weight_sum": active_weight_sum,
        "active_weight_min": float(np.min(active_shading_weights)),
        "active_weight_max": float(np.max(active_shading_weights)),
        "effective_channel_count": effective_channel_count,
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
        "all_mainlobes_preserved": _require_summary_bool(summary, "all_mainlobes_preserved"),
        "mean_mainlobe_level_delta_db": _require_summary_float(summary, "mean_mainlobe_level_delta_db"),
        "mean_sidelobe_reduction_db": _require_summary_float(summary, "mean_sidelobe_reduction_db"),
        "mean_mainlobe_margin_improvement_db": _require_summary_float(summary, "mean_mainlobe_margin_improvement_db"),
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
