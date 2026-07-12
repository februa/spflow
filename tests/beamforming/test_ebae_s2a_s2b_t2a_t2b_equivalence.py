"""S2a/S2bおよびT2a/T2bのFIR実現同値性を検証する。"""

from evaluations.beamforming.ebae_s2a_s2b_t2a_t2b_equivalence import (
    BRANCH_METHOD_IDS,
    DIRECT_METHOD_IDS,
    TAP_COUNTS,
    calculate_equivalence_rows,
)


def test_direct_and_difference_branch_realizations_are_equivalent() -> None:
    """同一線形FIR射影ではa方式とb方式の重み・応答・波形が一致する。"""
    rows = calculate_equivalence_rows()

    assert len(rows) == 2 * len(DIRECT_METHOD_IDS) * len(TAP_COUNTS)
    assert {str(row["difference_branch_method"]) for row in rows} == set(
        BRANCH_METHOD_IDS.values()
    )
    assert all(bool(row["equivalent"]) for row in rows)
    assert max(float(row["maximum_complex_weight_error"]) for row in rows) < 1.0e-12
    assert max(float(row["maximum_w_h_a_error"]) for row in rows) < 1.0e-12
    assert max(float(row["maximum_bl_power_error"]) for row in rows) < 1.0e-12
    assert max(float(row["maximum_waveform_error"]) for row in rows) < 1.0e-12
