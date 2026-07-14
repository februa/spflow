"""運用スパースアレイでL=1時間領域SLC漏れ込み診断を実行する。"""

from __future__ import annotations

import json
from pathlib import Path

from evaluations.beamforming.scenarios.operational_time_domain_slc_diagnostics import (
    OperationalTimeDomainSlcDiagnosticConfig,
    run_operational_time_domain_slc_leakage_diagnostics,
)
from spflow.sidelobe_cancellation import SlcConfig


def main() -> None:
    """小数遅延固定整相後の beam output から時間領域共分散を作る SLC を評価する。

    このスクリプトは、narrowband 診断用 SLC ではなく、L=1 時間領域方式である
    `beam_output [n_beam, n_sample]` から `R_uu` と `r_ud` を 1 組だけ作る方式を使う。

    評価では mixed / target-only / interferer-only を同じ固定整相に通し、
    mixed で学習した SLC 係数を各成分へ適用する。これにより、target beam 上の
    interferer leakage が本当に下がったかを直接確認する。
    """
    summary = run_operational_time_domain_slc_leakage_diagnostics(
        config=OperationalTimeDomainSlcDiagnosticConfig(
            output_dir=Path("artifacts/beamforming/operational_time_domain_slc_diagnostics/10000Hz_151beam_memory3s_8192Hz_interferer"),
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
