"""運用アレイで同一方位・複数周波数の時間領域固定整相後分離を評価する。"""

from __future__ import annotations

import json
from pathlib import Path

from spflow.beamforming import (
    OperationalSameAzimuthFrequencySeparationConfig,
    run_operational_same_azimuth_frequency_separation_diagnostics,
)


def main() -> None:
    """10000 Hz 設計の運用 active subset で、同一方位の 3 周波数を成分別に評価する。"""
    output_dir = Path("artifacts/beamforming/operational_same_azimuth_frequency_separation")

    # 代表処理周波数 10000 Hz の active subset と fractional delay FIR を使い、
    # リアルタイム経路は時間領域固定整相のまま、評価側の単一 tone 射影で周波数成分を分ける。
    summary = run_operational_same_azimuth_frequency_separation_diagnostics(
        OperationalSameAzimuthFrequencySeparationConfig(
            output_dir=output_dir,
            operational_array_definition_path=Path(
                "artifacts/beamforming/operational_sparse_array/operational_sparse_array_fs32768.json"
            ),
            fractional_delay_filter_bank_path=Path("artifacts/beamforming/fractional_delay_filter_bank_65x63.npz"),
            processing_frequency_hz=10000.0,
            source_azimuth_deg=90.0,
            source_frequencies_hz=(6144.0, 8192.0, 10000.0),
            source_levels_db20=(0.0, 0.0, 0.0),
            duration_s=1.0,
            noise_level_db20=-120.0,
            n_beam_az_real=151,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
