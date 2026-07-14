"""運用スパースアレイで時間領域SLCとMVDR、LCMV、GSCのBL応答を比較する。"""

from __future__ import annotations

import json
from pathlib import Path

from spflow.beamforming import (
    OperationalTimeDomainAdaptiveComparisonConfig,
    SlcConfig,
    run_operational_time_domain_adaptive_comparison,
)


def main() -> None:
    """SLC baseline と時間領域適応方式の before/after BL 改善量を出力する。"""
    summary = run_operational_time_domain_adaptive_comparison(
        config=OperationalTimeDomainAdaptiveComparisonConfig(
            output_dir=Path("artifacts/beamforming/operational_time_domain_adaptive_comparison/10000Hz_151beam_memory3s_8192Hz_interferer"),
            operational_array_definition_path=Path(
                "artifacts/beamforming/operational_sparse_array/operational_sparse_array_fs32768.json"
            ),
            fractional_delay_filter_bank_path=Path("artifacts/beamforming/fractional_delay_filter_bank_65x63.npz"),
            processing_frequency_hz=10000.0,
            target_azimuth_deg=90.0,
            interferer_azimuth_deg=60.0,
            interferer_frequency_hz=8192.0,
            target_level_db20=0.0,
            interferer_level_db20=-6.0,
            duration_s=5.0,
            n_beam_az_real=151,
            tap_len=3,
            diagonal_loading=3.0e-2,
        ),
        slc_config=SlcConfig(
            guard=10,
            loading=3.0e-2,
            memory_time_sec=3.0,
            heading_scale_deg=5.0,
            min_ref=8,
            sample_per_dof=5.0,
            tap_len=1,
            eta_normal=1.0,
            eta_limited=1.0,
            enable_heading_forgetting=False,
        ),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
