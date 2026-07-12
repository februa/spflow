"""単一信号・全整相方式直交試験の契約を確認する。"""

from evaluations.beamforming.alignment_single_source_validation_matrix import (
    AZIMUTHS_DEG,
    BAND_CASES,
    TAP_COUNTS,
    _method_rows,
)


def test_single_source_matrix_has_all_methods_and_finite_snr() -> None:
    """代表条件で全方式・tapのSNRとlevelが有限になることを確認する。"""
    rows = _method_rows(BAND_CASES[0], AZIMUTHS_DEG[1], 0.0, 0.0)

    assert {str(row["method"]) for row in rows} == {"S1", "S2a", "T1", "T2a"}
    assert {int(row["tap_count"]) for row in rows} == set(TAP_COUNTS)
    assert {str(row["algorithm"]) for row in rows} == {"ebae", "mvdr"}
    assert all(bool(row["finite"]) for row in rows)
    assert {str(row["equivalent_difference_branch_method"]) for row in rows} >= {
        "S2b",
        "T2b",
    }
