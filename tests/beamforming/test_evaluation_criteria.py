"""beamforming 評価基準カタログのテスト。"""

from pathlib import Path

import pytest

from spflow.beamforming import (
    get_evaluation_criteria_for_pattern,
    list_beamforming_evaluation_criteria,
    list_beamforming_evaluation_patterns,
    write_beamforming_evaluation_criteria_markdown,
)


def test_evaluation_criteria_ids_are_unique() -> None:
    """評価基準 ID が一意で、重要な故障モードを含むことを確認する。

    評価基準は診断値そのものではなく、検討時に見るべき軸のカタログである。
    ここでは重複 ID により別評価が上書きされないことと、SLC 自己消去・時間波形・
    sidelobe margin という今回不足していた観点が含まれることを確認する。
    """
    criteria = list_beamforming_evaluation_criteria()
    criterion_ids = [criterion.criterion_id for criterion in criteria]

    assert len(criterion_ids) == len(set(criterion_ids))
    assert "target_leakage_components" in criterion_ids
    assert "waveform_integrity" in criterion_ids
    assert "sidelobe_peak_margin" in criterion_ids
    assert "input_output_level_consistency" in criterion_ids
    assert "frequency_component_separation" in criterion_ids
    assert "adaptive_constraint_response" in criterion_ids
    assert "source_visibility_preservation" in criterion_ids


    for criterion in criteria:
        assert criterion.unit_reference


def test_sidelobe_margin_requires_first_sidelobe_reduction() -> None:
    """SLC 前後比較では第一副極改善量を必須 metric として扱う。

    干渉方位 marker の一点だけが落ちても、第一副極が下がらない場合は
    BL 図上の sidelobe 改善とは言えないため、評価基準カタログで明示する。
    """
    criteria = {
        criterion.criterion_id: criterion
        for criterion in list_beamforming_evaluation_criteria()
    }
    criterion = criteria["sidelobe_peak_margin"]

    assert "first_sidelobe_reduction_db" in criterion.required_metrics
    assert "第一副極" in criterion.purpose
    assert "第一副極" in criterion.pass_guideline


def test_evaluation_pattern_ids_are_unique() -> None:
    """検討パターン ID が一意で、固定整相と SLC を別用途として扱うことを確認する。

    全パターンに全評価を強制すると診断が過剰になるため、検討目的ごとの pattern ID を
    安定した公開契約として持つ。
    """
    patterns = list_beamforming_evaluation_patterns()
    pattern_ids = [pattern.pattern_id for pattern in patterns]

    assert len(pattern_ids) == len(set(pattern_ids))
    assert "fixed_beam_single_source" in pattern_ids
    assert "slc_target_only" in pattern_ids
    assert "slc_different_frequency_interference" in pattern_ids
    assert "slc_same_azimuth_multi_frequency" in pattern_ids
    assert "time_domain_adaptive_mvdr_lcmv_gsc" in pattern_ids
    assert "slc_scan_multi_source_display" in pattern_ids


def test_slc_scan_multi_source_display_does_not_require_interferer_cancellation() -> None:
    """全方位 scan 用途では interferer を別信号として残す評価 pattern を使う。

    SLC を scan 表示へ掛ける場合、別方位 source のピークは観測対象であり、
    target beam leakage canceller のように interferer 自体の低減を必須にしない。
    """
    required, recommended = get_evaluation_criteria_for_pattern("slc_scan_multi_source_display")
    required_ids = {criterion.criterion_id for criterion in required}
    recommended_ids = {criterion.criterion_id for criterion in recommended}
    criteria = {
        criterion.criterion_id: criterion
        for criterion in list_beamforming_evaluation_criteria()
    }

    assert {
        "source_visibility_preservation",
        "mainlobe_preservation",
        "fraz_btr_consistency",
        "waveform_integrity",
    } <= required_ids
    assert "target_leakage_components" not in required_ids
    assert "sidelobe_peak_margin" in recommended_ids
    assert "interferer 自体を消す必要はない" in criteria["source_visibility_preservation"].pass_guideline

def test_slc_target_only_requires_self_cancellation_checks() -> None:
    """target-only SLC では自己消去と時間波形健全性を必須評価にする。

    目標信号が 1 つで SNR が悪くない条件では SLC が使えるべきなので、
    mainlobe 維持、target 成分別漏れ込み、処理後時間波形を必須にしている。
    """
    required, recommended = get_evaluation_criteria_for_pattern("slc_target_only")
    required_ids = {criterion.criterion_id for criterion in required}
    recommended_ids = {criterion.criterion_id for criterion in recommended}

    assert {
        "mainlobe_preservation",
        "target_leakage_components",
        "waveform_integrity",
        "input_output_level_consistency",
    } <= required_ids
    assert "slc_covariance_health" in recommended_ids


def test_fixed_single_source_requires_peak_sidelobe_and_display_checks() -> None:
    """単一音源の固定整相では peak、sidelobe、FRAZ/BTR 整合を必須評価にする。

    固定整相の基本性能は BL だけでは判断できないため、方位・周波数 peak と
    FRAZ/BTR 表示軸の整合も同時に確認する。
    """
    required, recommended = get_evaluation_criteria_for_pattern("fixed_beam_single_source")
    required_ids = {criterion.criterion_id for criterion in required}
    recommended_ids = {criterion.criterion_id for criterion in recommended}

    assert {
        "beam_peak_position",
        "sidelobe_peak_margin",
        "fraz_btr_consistency",
        "input_output_level_consistency",
    } <= required_ids
    assert "grating_lobe_and_ambiguity" in recommended_ids


def test_input_output_level_criterion_records_spatial_and_analysis_snr_gain() -> None:
    """入力対出力レベル評価が空間利得と分析利得を分けて記録することを確認する。

    無相関雑音では整相により有効チャネル数に応じた SN 改善が期待できる。
    一方で BL/FRAZ のような分析幅を持つ表示では時間・周波数方向の平均による改善も混ざるため、
    空間合成利得と分析幅由来の利得を別 metric として保持する。
    """
    criteria = {
        criterion.criterion_id: criterion
        for criterion in list_beamforming_evaluation_criteria()
    }
    criterion = criteria["input_output_level_consistency"]

    assert "expected_spatial_snr_gain_db" in criterion.required_metrics
    assert "expected_analysis_snr_gain_db" in criterion.required_metrics
    assert "observed_snr_gain_db" in criterion.required_metrics
    assert "20log10(sqrt(N_eff))" in criterion.pass_guideline
    assert "10log10(N_eff)" in criterion.pass_guideline


def test_slc_same_azimuth_multi_frequency_requires_frequency_separation_checks() -> None:
    """同一方位・複数周波数 SLC では空間分離ではなく周波数成分分離を必須評価にする。

    同じ方位にある成分はアレイ応答が同じになるため、同一周波数の空間分離を要求しない。
    その代わり、STFT bin または帯域別処理で目的周波数成分を維持し、別周波数成分の漏れ込みを
    下げられるかを方式評価の中心に置く。
    """
    required, recommended = get_evaluation_criteria_for_pattern("slc_same_azimuth_multi_frequency")
    required_ids = {criterion.criterion_id for criterion in required}
    recommended_ids = {criterion.criterion_id for criterion in recommended}

    assert {
        "frequency_component_separation",
        "mainlobe_preservation",
        "fraz_btr_consistency",
        "waveform_integrity",
    } <= required_ids
    assert "slc_covariance_health" in recommended_ids
    assert "input_output_level_consistency" in recommended_ids


def test_time_domain_adaptive_pattern_requires_constraint_and_runtime_checks() -> None:
    """時間領域 MVDR / LCMV / GSC では制約応答と実時間性を必須評価にする。

    SLC の BL 改善量が不足した後段検討では、target 保護と null が設計通りか、
    さらに channel×tap 自由度の共分散が安定しているかを同時に確認する必要がある。
    """
    required, recommended = get_evaluation_criteria_for_pattern("time_domain_adaptive_mvdr_lcmv_gsc")
    required_ids = {criterion.criterion_id for criterion in required}
    recommended_ids = {criterion.criterion_id for criterion in recommended}

    assert {
        "adaptive_constraint_response",
        "target_leakage_components",
        "mainlobe_preservation",
        "slc_covariance_health",
        "runtime_budget",
    } <= required_ids
    assert "waveform_integrity" in recommended_ids
    assert "fraz_btr_consistency" in recommended_ids

def test_unknown_pattern_is_rejected() -> None:
    """未知 pattern ID は評価不要として扱わず、明示的に失敗させる。

    未知 ID を空の評価セットとして通すと、評価漏れを成功と誤認するため危険である。
    """
    with pytest.raises(ValueError):
        get_evaluation_criteria_for_pattern("unknown_pattern")


def test_evaluation_criteria_markdown_can_be_written() -> None:
    """評価基準カタログを Markdown として再生成できることを確認する。

    設計書は人がレビューする契約でもあるため、コード上のカタログから doc/SLC 配下と
    同じ形式の Markdown を生成できる必要がある。
    """
    output_path = Path("artifacts/beamforming/evaluation_criteria_test/criteria.md")

    write_beamforming_evaluation_criteria_markdown(output_path)
    text = output_path.read_text(encoding="utf-8")

    assert "# Beamforming 評価基準カタログ" in text
    assert "target_leakage_components" in text
    assert "input_output_level_consistency" in text
    assert "frequency_component_separation" in text
    assert "slc_same_azimuth_multi_frequency" in text
    assert "単位・基準" in text
    assert "dB re" in text
    assert "20log10(sqrt(N_eff))" in text
    assert "10log10(N_eff)" in text
    assert "slc_target_only" in text
