"""固定整相の後段へ SLC を適用した BL/FRAZ/BTR 診断を保存するサンプル。"""

# 固定整相だけの診断結果を残したまま、保護したい target の mainlobe を維持しつつ、
# 同一周波数 interferer の sidelobe をどこまで抑えられるかを before/after で比較する。

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spflow.beamforming.slc import SlcConfig
from spflow.beamforming.time_delay_diagnostics import TimeDelayDiagnosticConfig, TimeDelayDiagnosticSource
from spflow.beamforming.time_delay_slc_diagnostics import run_integer_delay_slc_diagnostics


def main() -> None:
    """固定整相後段へ SLC を適用した診断を実行する。"""
    output_dir = ROOT / "artifacts" / "beamforming" / "time_delay_slc_diagnostics"
    summary = run_integer_delay_slc_diagnostics(
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
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
