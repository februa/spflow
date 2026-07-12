"""粗い共分散の整数遅延・方位別切り出しMVDR比較を検証する。"""

import numpy as np

from evaluations.beamforming.coarse_covariance_integer_delay_mvdr import (
    DIRECT_METHOD_ID,
    INTEGER_DELAY_METHOD_ID,
    evaluate_methods,
    method_covariances,
)


def test_direct_and_integer_delay_covariance_eigenvalues_are_equivalent() -> None:
    """直接適用と整数遅延前段用の共分散固有値が一致する。"""

    covariance = method_covariances()
    # unitary変換は物理的なcoherenceやrankを変えないことを固有値で確認する。
    np.testing.assert_allclose(
        np.linalg.eigvalsh(covariance[DIRECT_METHOD_ID]),
        np.linalg.eigvalsh(covariance[INTEGER_DELAY_METHOD_ID]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_direct_and_integer_delay_methods_preserve_the_same_target() -> None:
    """方位別共分散を使う2方式が同じtarget応答を保つ。"""

    rows, _ = evaluate_methods()
    by_method = {str(row["method"]): row for row in rows}

    direct = by_method[DIRECT_METHOD_ID]
    integer_delay = by_method[INTEGER_DELAY_METHOD_ID]
    assert float(direct["direct_integer_delay_target_complex_error"]) < 1.0e-10
    assert abs(float(direct["target_level_db_re_input_rms"])) < 1.0e-10
    assert abs(float(integer_delay["target_level_db_re_input_rms"])) < 1.0e-10
    np.testing.assert_allclose(
        float(direct["interferer_leakage_db_re_target"]),
        float(integer_delay["interferer_leakage_db_re_target"]),
        rtol=0.0,
        atol=1.0e-9,
    )
