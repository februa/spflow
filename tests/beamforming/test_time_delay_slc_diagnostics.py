"""時間領域固定整相後段の SLC 診断に関する回帰試験。"""

from __future__ import annotations

import json
from pathlib import Path

from spflow.beamforming.slc import SlcConfig
from spflow.beamforming.time_delay import design_fractional_delay_filter_bank
from spflow.beamforming.time_delay_diagnostics import TimeDelayDiagnosticConfig, TimeDelayDiagnosticSource
from spflow.beamforming.time_delay_slc_diagnostics import run_integer_delay_slc_diagnostics
from spflow.beamforming.fractional_delay_slc_diagnostics import run_fractional_delay_slc_diagnostics


def test_integer_delay_slc_diagnostics_save_before_after_figures_and_preserve_mainlobe() -> None:
    """SLC 診断について固定整相 before/after の BL/FRAZ/BTR を保存し、target mainlobe 維持と sidelobe 抑圧を要約できることを確認する。"""
    output_dir = Path.cwd() / "artifacts" / "beamforming" / "time_delay_slc_diagnostics_test"
    summary = run_integer_delay_slc_diagnostics(
        config=TimeDelayDiagnosticConfig(
            output_dir=output_dir,
            fs_hz=32768.0,
            duration_s=1.0,
            sound_speed_m_s=1500.0,
            noise_level_db20=-45.0,
            random_seed=1234,
            array_n_ch=31,
            array_sensor_spacing_m=0.05,
            sparse_stride_pattern=(1, 2, 1, 3, 1, 4),
            n_beam_az_real=81,
            btr_block_size=512,
            source_specs=(
                TimeDelayDiagnosticSource(
                    azimuth_deg=20.0,
                    frequency_hz=1536.0,
                    level_db20=0.0,
                    amplitude_modulation_hz=0.7,
                    amplitude_modulation_depth=0.9,
                    label="target",
                ),
                TimeDelayDiagnosticSource(
                    azimuth_deg=60.0,
                    frequency_hz=1536.0,
                    level_db20=-3.0,
                    amplitude_modulation_hz=1.1,
                    amplitude_modulation_depth=0.9,
                    amplitude_modulation_phase_deg=70.0,
                    label="interferer",
                ),
            ),
        ),
        slc_config=SlcConfig(
            guard=4,
            loading=3.0e-2,
            memory_time_sec=2.0,
            heading_scale_deg=5.0,
            min_ref=8,
            sample_per_dof=5.0,
            tap_len=1,
            eta_normal=0.2,
            eta_limited=0.1,
            enable_heading_forgetting=False,
        ),
        target_source_indices=(0,),
        slc_analysis_block_size=64,
        max_reference_beams=48,
    )

    fixed_summary_path = Path(str(summary["fixed_summary_path"]))
    slc_summary_path = output_dir / "slc_summary.json"
    slc_fraz_path = output_dir / "slc_fraz.png"
    slc_btr_path = output_dir / "slc_btr.png"

    assert fixed_summary_path.exists()
    assert slc_summary_path.exists()
    assert slc_fraz_path.exists()
    assert slc_btr_path.exists()

    saved_summary = json.loads(slc_summary_path.read_text(encoding="utf-8"))
    assert saved_summary["slc_fraz_png_path"] == summary["slc_fraz_png_path"]
    assert saved_summary["slc_btr_png_path"] == summary["slc_btr_png_path"]
    assert bool(summary["all_mainlobes_preserved"])
    assert float(summary["mean_sidelobe_reduction_db"]) > 0.5
    assert int(summary["slc_design_summary"]["normal_beam_count"]) + int(summary["slc_design_summary"]["limited_beam_count"]) > 0

    source_comparisons = summary["source_comparisons"]
    assert isinstance(source_comparisons, list)
    assert len(source_comparisons) == 1
    source_comparison = source_comparisons[0]
    assert bool(source_comparison["mainlobe_preserved"])
    # mainlobe margin improvement は target レベル低下と guard 外 peak 低下の差で決まるため、
    # 同一周波数・複数音源条件では sidelobe_reduction_db より小さくなる。
    # ここでは固定しきい値を強くしすぎず、mainlobe を維持したまま改善側に倒れることを確認する。
    assert float(source_comparison["mainlobe_margin_improvement_db"]) > 0.1

    for source_metric in summary["slc_source_metrics"]:
        assert Path(str(source_metric["bl_png_path"])).exists()
        assert Path(str(source_metric["bl_compare_png_path"])).exists()



def test_fractional_delay_slc_diagnostics_save_before_after_figures_and_preserve_mainlobe() -> None:
    """小数遅延固定整相を前段にした SLC 診断について before/after の BL/FRAZ/BTR を保存し、mainlobe 維持と sidelobe 改善を要約できることを確認する。"""
    output_dir = Path.cwd() / "artifacts" / "beamforming" / "fractional_delay_slc_diagnostics_test"
    filter_bank_path = Path.cwd() / "artifacts" / "beamforming" / "fractional_delay_filter_bank_slc_test.npz"
    filter_bank_path.parent.mkdir(parents=True, exist_ok=True)

    filter_bank = design_fractional_delay_filter_bank(n_frac_filter=65, n_tap=63)
    filter_bank.save_npz(filter_bank_path)
    try:
        summary = run_fractional_delay_slc_diagnostics(
            config=TimeDelayDiagnosticConfig(
                output_dir=output_dir,
                fs_hz=32768.0,
                duration_s=1.0,
                sound_speed_m_s=1500.0,
                noise_level_db20=-45.0,
                random_seed=1234,
                array_n_ch=31,
                array_sensor_spacing_m=0.05,
                sparse_stride_pattern=(1, 2, 1, 3, 1, 4),
                n_beam_az_real=81,
                btr_block_size=512,
                source_specs=(
                    TimeDelayDiagnosticSource(
                        azimuth_deg=20.0,
                        frequency_hz=6144.0,
                        level_db20=0.0,
                        amplitude_modulation_hz=0.7,
                        amplitude_modulation_depth=0.9,
                        label="target",
                    ),
                    TimeDelayDiagnosticSource(
                        azimuth_deg=60.0,
                        frequency_hz=6144.0,
                        level_db20=-3.0,
                        amplitude_modulation_hz=1.1,
                        amplitude_modulation_depth=0.9,
                        amplitude_modulation_phase_deg=70.0,
                        label="interferer",
                    ),
                ),
            ),
            slc_config=SlcConfig(
                guard=4,
                loading=3.0e-2,
                memory_time_sec=2.0,
                heading_scale_deg=5.0,
                min_ref=8,
                sample_per_dof=5.0,
                tap_len=1,
                eta_normal=0.2,
                eta_limited=0.1,
                enable_heading_forgetting=False,
            ),
            fractional_delay_filter_bank_path=filter_bank_path,
            target_source_indices=(0,),
            slc_analysis_block_size=64,
            max_reference_beams=48,
        )
    finally:
        if filter_bank_path.exists():
            filter_bank_path.unlink()

    fixed_summary_path = Path(str(summary["fixed_summary_path"]))
    slc_summary_path = output_dir / "slc_summary.json"
    slc_fraz_path = output_dir / "slc_fraz.png"
    slc_btr_path = output_dir / "slc_btr.png"

    assert fixed_summary_path.exists()
    assert slc_summary_path.exists()
    assert slc_fraz_path.exists()
    assert slc_btr_path.exists()

    saved_summary = json.loads(slc_summary_path.read_text(encoding="utf-8"))
    assert saved_summary["slc_fraz_png_path"] == summary["slc_fraz_png_path"]
    assert saved_summary["slc_btr_png_path"] == summary["slc_btr_png_path"]
    assert bool(summary["all_mainlobes_preserved"])
    assert float(summary["mean_sidelobe_reduction_db"]) > 0.2
    assert int(summary["slc_design_summary"]["normal_beam_count"]) + int(summary["slc_design_summary"]["limited_beam_count"]) > 0

    source_comparisons = summary["source_comparisons"]
    assert isinstance(source_comparisons, list)
    assert len(source_comparisons) == 1
    source_comparison = source_comparisons[0]
    assert bool(source_comparison["mainlobe_preserved"])
    assert float(source_comparison["sidelobe_reduction_db"]) > 0.2

    # 複数音源条件では、target 主ローブ維持とは別に interferer 方位の before/after 指標が必要になる。
    # ここでは方式検討中のため抑圧量の符号は固定せず、summary 契約として保存されることを確認する。
    interference_source_comparisons = summary["interference_source_comparisons"]
    assert isinstance(interference_source_comparisons, list)
    assert len(interference_source_comparisons) == 1
    assert "nearest_level_reduction_db" in interference_source_comparisons[0]

    for source_metric in summary["slc_source_metrics"]:
        assert Path(str(source_metric["bl_png_path"])).exists()
        assert Path(str(source_metric["bl_compare_png_path"])).exists()
