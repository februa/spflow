"""粗い共分散の整数遅延・方位別切り出しMVDR比較を検証する。"""

import numpy as np

from evaluations.beamforming.coarse_covariance_integer_delay_mvdr import (
    evaluate_methods,
    method_covariances,
)


def test_t0_t1_covariance_eigenvalues_are_unitarily_equivalent() -> None:
    """T0/T1は座標変換だけなので共分散固有値が一致する。"""

    covariance = method_covariances()
    # unitary変換は物理的なcoherenceやrankを変えないことを固有値で確認する。
    np.testing.assert_allclose(
        np.linalg.eigvalsh(covariance["T0"]),
        np.linalg.eigvalsh(covariance["T1"]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_integer_delay_and_time_cut_avoid_s0_coherence_failure() -> None:
    """S1/T0はS0よりtarget整相と干渉漏れを改善する。"""

    rows, _ = evaluate_methods()
    by_method = {str(row["method"]): row for row in rows}

    assert float(by_method["S1"]["weight_norm"]) < float(by_method["S0"]["weight_norm"])
    assert float(by_method["T0"]["weight_norm"]) < float(by_method["S0"]["weight_norm"])
    assert float(by_method["T0"]["t0_t1_target_complex_error"]) < 1.0e-10
    assert abs(float(by_method["T0"]["target_level_db_re_input_rms"])) < 1.0e-10
