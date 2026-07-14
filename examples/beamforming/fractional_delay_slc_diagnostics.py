"""小数遅延固定整相の後段へ SLC を適用した BL/FRAZ/BTR 診断を保存するサンプル。"""

# 保存済み小数遅延 FIR バンクを読み出した固定整相を before として残しつつ、
# 同一周波数 interferer に対する SLC の抑圧量と mainlobe 維持を after 図で比較する。

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spflow.beamforming.fractional_delay_slc_diagnostics import (  # noqa: E402
    run_fractional_delay_slc_diagnostics,
)
from spflow.beamforming.time_delay_diagnostics import (  # noqa: E402
    TimeDelayDiagnosticConfig,
    TimeDelayDiagnosticSource,
)
from spflow.sidelobe_cancellation import SlcConfig  # noqa: E402


def main() -> None:
    """小数遅延固定整相後段へ SLC を適用した診断を実行する。"""
    filter_bank_path = ROOT / "artifacts" / "beamforming" / "fractional_delay_filter_bank_65x63.npz"
    if not filter_bank_path.exists():
        raise FileNotFoundError(
            "fractional delay filter bank not found. Run examples/beamforming/design_fractional_delay_filter_bank.py "
            "with --n-frac-filter 65 --n-tap 63 first."
        )

    output_dir = ROOT / "artifacts" / "beamforming" / "fractional_delay_slc_diagnostics"
    summary = run_fractional_delay_slc_diagnostics(
        config=TimeDelayDiagnosticConfig(
            output_dir=output_dir,
            fs_hz=32768.0,
            duration_s=2.0,
            sound_speed_m_s=1500.0,
            noise_level_db20=-45.0,
            random_seed=1234,
            array_n_ch=61,
            array_sensor_spacing_m=0.05,
            sparse_stride_pattern=(1, 2, 1, 3, 1, 4, 2, 5),
            n_beam_az_real=121,
            btr_block_size=1024,
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
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
