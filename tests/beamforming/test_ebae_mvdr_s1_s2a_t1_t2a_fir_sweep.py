"""EBAE/MVDR S1・S2a・T1・T2a FIR長sweepの回帰試験。"""

from __future__ import annotations

import numpy as np

from evaluations.beamforming.ebae_mvdr_s1_s2a_t1_t2a_fir_sweep import (
    ALGORITHM_IDS,
    FIR_TAP_COUNTS,
    calculate_fir_sweep,
)


def test_s1_s2a_and_t1_t2a_are_unitary_coordinate_equivalents() -> None:
    """EBAE/MVDR双方で正しいS1=S2a、T1=T2aが成立することを確認する。"""
    _, _, arrays = calculate_fir_sweep()

    for algorithm in ALGORITHM_IDS:
        assert float(arrays[f"{algorithm}_s1_s2a_relative_error"][0]) < 1.0e-10
        assert float(arrays[f"{algorithm}_t1_t2a_relative_error"][0]) < 1.0e-10


def test_full_length_fir_reconstructs_all_reference_weights() -> None:
    """512 tapでは全方式のfull DFT重みを数値床近傍で再構成することを確認する。"""
    _, rows, _ = calculate_fir_sweep()

    full_rows = [row for row in rows if int(row["tap_count"]) == max(FIR_TAP_COUNTS)]
    assert len(full_rows) == 8
    for row in full_rows:
        assert float(row["relative_weight_error"]) < 1.0e-12
        assert abs(float(row["target_level_delta_db_re_reference"])) < 1.0e-10
        # full DFT完成重みではT1を含む全方式が正しいsource方位を保持する。
        # 短tap T1のpeak移動が共分散設計ではなくFIR打切りに起因することを固定する。
        assert float(row["target_peak_error_deg"]) == 0.0


def test_integer_delay_residual_coordinates_improve_short_fir_reconstruction() -> None:
    """短いFIRでS2a/T2aが対応するS1/T1より再構成誤差を下げることを確認する。"""
    _, rows, _ = calculate_fir_sweep()
    lookup = {
        (str(row["algorithm"]), str(row["method"]), int(row["tap_count"])): float(
            row["relative_weight_error"]
        )
        for row in rows
    }
    for algorithm in ALGORITHM_IDS:
        assert lookup[(algorithm, "S2a", 32)] < lookup[(algorithm, "S1", 32)]
        assert lookup[(algorithm, "T2a", 32)] < lookup[(algorithm, "T1", 32)]
        assert np.isfinite(lookup[(algorithm, "S2a", 32)])
