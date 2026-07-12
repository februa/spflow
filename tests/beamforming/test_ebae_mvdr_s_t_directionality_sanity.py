"""EBAE/MVDR S/T方位推定sanityの回帰試験。"""

from __future__ import annotations

from evaluations.beamforming.ebae_mvdr_s_t_directionality_sanity import (
    calculate_directionality_sanity,
)


def _lookup_rows():
    result = calculate_directionality_sanity()
    return {
        (str(row["signal_type"]), str(row["algorithm"]), str(row["method"])): row
        for row in result.rows
    }


def test_tone_is_localized_by_s_and_t_for_ebae_and_mvdr() -> None:
    """bin中心toneではS/Tの両方が正方位を推定することを確認する。"""
    rows = _lookup_rows()
    for algorithm in ("ebae_music", "mvdr_capon"):
        for method in ("S", "T"):
            assert float(rows[("bin_center_tone", algorithm, method)]["peak_error_deg"]) == 0.0


def test_flat_bin_broadband_breaks_s_but_t_preserves_direction() -> None:
    """粗い1-bin広帯域ではSが破綻し、Tが正方位を維持することを確認する。"""
    rows = _lookup_rows()
    # EBAEはSのMUSIC peak自体がsourceから外れ、Tでは0 degへ戻る。
    assert float(rows[("flat_one_bin_broadband", "ebae_music", "S")]["peak_error_deg"]) > 2.0
    assert float(rows[("flat_one_bin_broadband", "ebae_music", "T")]["peak_error_deg"]) == 0.0
    # MVDR/CaponはSで主ローブが著しく広がり、Tでは狭い方位peakを回復する。
    s_width = float(rows[("flat_one_bin_broadband", "mvdr_capon", "S")]["three_db_width_deg"])
    t_width = float(rows[("flat_one_bin_broadband", "mvdr_capon", "T")]["three_db_width_deg"])
    assert s_width > 2.0 * t_width
    assert float(rows[("flat_one_bin_broadband", "mvdr_capon", "T")]["peak_error_deg"]) == 0.0


def test_t_restores_single_signal_model_for_ebae() -> None:
    """広帯域source候補でTがEBAE信号数を1へ戻すことを確認する。"""
    rows = _lookup_rows()
    s_count = int(
        rows[("flat_one_bin_broadband", "ebae_music", "S")]["source_count_at_source_candidate"]
    )
    t_count = int(
        rows[("flat_one_bin_broadband", "ebae_music", "T")]["source_count_at_source_candidate"]
    )
    assert s_count > 1
    assert t_count == 1
