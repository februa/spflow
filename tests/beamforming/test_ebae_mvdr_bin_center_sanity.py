"""EBAE/MVDR bin中心・beam直上sanity評価の回帰試験。"""

from __future__ import annotations

import numpy as np

from evaluations.beamforming.ebae_mvdr_bin_center_sanity import (
    TARGET_AZIMUTH_DEG,
    calculate_ebae_mvdr_bin_center_sanity,
)


def test_ebae_and_mvdr_preserve_bin_center_beam_center_source() -> None:
    """理想単一信号で両方式が同じtarget方位と0 dB応答を保つことを確認する。"""
    result = calculate_ebae_mvdr_bin_center_sanity()

    assert result.signal_count == 1
    assert result.associated_azimuth_deg == TARGET_AZIMUTH_DEG
    for row in result.summary_rows:
        assert float(row["target_peak_error_deg"]) == 0.0
        assert abs(float(row["target_level_db_re_input_rms"])) < 1.0e-10
        assert float(row["distortionless_error"]) < 1.0e-10


def test_ebae_dl1_is_close_to_but_more_robust_than_mvdr() -> None:
    """DL=1のEBAEがMVDRに近い形状を保ちつつ非target除外量を弱めることを確認する。"""
    result = calculate_ebae_mvdr_bin_center_sanity()
    ebae_row = result.summary_rows[0]
    mvdr_row = result.summary_rows[1]

    # DL=1は除外量を弱めてCBFへ寄せるため、MVDRよりguard外levelが高いことを期待する。
    assert float(ebae_row["guard_outside_peak_db_re_input_rms"]) > float(
        mvdr_row["guard_outside_peak_db_re_input_rms"]
    )
    # このsanity条件ではBL全体の差が約6 dBであり、方式実装の符号誤りを疑う大差ではない。
    assert float(ebae_row["target_bl_max_abs_delta_db_re_mvdr"]) < 7.0
    assert np.isfinite(float(ebae_row["target_bl_rms_delta_db_re_mvdr"]))
