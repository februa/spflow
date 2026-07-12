"""方位別MVDR 3方式review配列を検証する。"""

import numpy as np

from evaluations.beamforming.direction_cut_mvdr_spatial_spectral_review import (
    AZIMUTH_DEG,
    FREQUENCY_HZ,
    METHOD_IDS,
    calculate_review_arrays,
)


def test_review_arrays_have_fixed_beam_and_frequency_axes() -> None:
    """BL、FRAZ、スペクトルが明示したbeam・周波数axisを持つ。"""

    arrays = calculate_review_arrays()
    for method_id in METHOD_IDS:
        assert arrays[f"{method_id}_mixed_bl_level_db"].shape == AZIMUTH_DEG.shape
        assert arrays[f"{method_id}_fraz_level_db"].shape == (
            AZIMUTH_DEG.size,
            FREQUENCY_HZ.size,
        )
        assert arrays[f"{method_id}_output_spectrum_db"].shape == FREQUENCY_HZ.shape
        assert bool(np.all(np.isfinite(arrays[f"{method_id}_fraz_level_db"])))


def test_t1_t2_fraz_are_equivalent_in_static_model() -> None:
    """位相基準を揃えたT1/T2の静的FRAZが数値誤差内で一致する。"""

    arrays = calculate_review_arrays()
    np.testing.assert_allclose(
        arrays[f"{METHOD_IDS[3]}_fraz_level_db"],
        arrays[f"{METHOD_IDS[4]}_fraz_level_db"],
        rtol=0.0,
        atol=1.0e-8,
    )


def test_s1_s2_fraz_are_equivalent_in_static_model() -> None:
    """位相基準を揃えたS1/S2の静的FRAZが数値誤差内で一致する。"""

    arrays = calculate_review_arrays()
    np.testing.assert_allclose(
        arrays[f"{METHOD_IDS[1]}_fraz_level_db"],
        arrays[f"{METHOD_IDS[2]}_fraz_level_db"],
        rtol=0.0,
        atol=1.0e-8,
    )
