"""beamforming 評価基準のカタログと検討パターン別の選定規則を提供するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .._validation import require


@dataclass(frozen=True)
class BeamformingEvaluationCriterion:
    """1 つの beamforming 評価基準を表す。

    このクラスは、BL / FRAZ / BTR / 時間波形 / SLC 係数などの評価観点を、
    後続の診断コードや設計書から参照しやすい粒度で保持する。

    入力は評価基準 ID、分類、目的、使う metric、推奨図、判定の目安、失敗時の解釈、
    dB 指標の基準量または物理単位の説明である。
    出力は検討パターンから参照される不変の評価基準定義である。

    実際の BL 計算、FRAZ 描画、SLC 係数推定は責務に含めない。
    信号処理上は、評価不足や誤った単一指標判断を避けるための評価設計カタログに位置づく。
    """

    criterion_id: str
    category: str
    title: str
    purpose: str
    required_metrics: tuple[str, ...]
    recommended_figures: tuple[str, ...]
    pass_guideline: str
    failure_interpretation: str
    unit_reference: str


@dataclass(frozen=True)
class BeamformingEvaluationPattern:
    """検討パターンごとに使う評価基準の組を表す。

    このクラスは、固定整相、アレイ設計、シェーディング、SLC 条件スイープなどの
    検討パターンに対して、必須評価基準と推奨評価基準を紐付ける。

    入力は pattern ID、説明、必須 criterion ID、推奨 criterion ID である。
    出力は診断スクリプトや設計レビューが参照する評価セットである。

    評価値の計算や pass/fail 判定の実行は責務に含めない。
    信号処理上は、方式検討ごとに見るべき軸を明示する評価計画に位置づく。
    """

    pattern_id: str
    title: str
    description: str
    required_criterion_ids: tuple[str, ...]
    recommended_criterion_ids: tuple[str, ...]


def _build_criteria() -> tuple[BeamformingEvaluationCriterion, ...]:
    """評価基準カタログを作る。

    Returns:
        評価基準定義の tuple。各 ID は一意である。

    Notes:
        ここでは数値閾値を固定しすぎない。アレイ設計、固定整相、SLC では
        合格条件が異なるため、各 criterion には「何を見るべきか」と「失敗時の解釈」を持たせる。
    """
    return (
        BeamformingEvaluationCriterion(
            criterion_id="beam_peak_position",
            category="BL/FRAZ/BTR",
            title="ピーク方位・ピーク周波数の正しさ",
            purpose="到来方位と周波数に対して、最大応答が正しい位置に出ているかを確認する。",
            required_metrics=("bl_peak_azimuth_deg", "fraz_global_peak_azimuth_deg", "fraz_global_peak_frequency_hz"),
            recommended_figures=("bl.png", "fraz.png", "btr.png"),
            pass_guideline="単一音源では peak 方位が最近傍待受方位に一致し、FRAZ peak 周波数が入力周波数に一致すること。",
            failure_interpretation="方位軸、等 cos 軸、遅延符号、周波数ビン対応、またはアレイ側面定義が誤っている可能性が高い。",
            unit_reference="方位は deg、周波数は Hz。レベルを併記する場合は dB re input RMS、dB re 1 uPa RMS など基準量を明記する。",
        ),
        BeamformingEvaluationCriterion(
            criterion_id="mainlobe_preservation",
            category="BL",
            title="メインローブ維持",
            purpose="SLC やシェーディング後に target mainlobe の位置とレベルが壊れていないかを確認する。",
            required_metrics=("mainlobe_level_delta_db", "peak_azimuth_shift_deg", "target_power_delta_db"),
            recommended_figures=("slc_bl_compare.png", "target_leakage_levels.png"),
            pass_guideline="方式比較では mainlobe level delta と target power delta を別々に見る。SLC 評価では target-only 条件も必ず確認する。",
            failure_interpretation="desired 成分を reference へ混ぜている、blocking が不足している、eta が大きすぎる、または guard が狭すぎる。",
            unit_reference="mainlobe_level_delta_db と target_power_delta_db は処理前または固定整相出力に対する比率 dB。絶対レベルは dB re 1 uPa RMS などで別記する。",
        ),
        BeamformingEvaluationCriterion(
            criterion_id="sidelobe_peak_margin",
            category="BL",
            title="サイドローブ peak margin",
            purpose="mainlobe peak と guard 外 sidelobe peak の差、および第一副極レベルの改善量が十分かを確認する。",
            required_metrics=("local_to_nonlocal_margin_db", "max_nonlocal_level_db20", "first_sidelobe_reduction_db", "worst_peak_margin_db"),
            recommended_figures=("bl.png", "slc_bl_compare.png", "margin_summary.png"),
            pass_guideline="固定整相・アレイ設計では 13 dB 以上を基準にする。SLC / 適応方式の before/after 比較では第一副極と guard 外 peak が実際に下がること。複数音源では既知 source 方位の mainlobe を除外した指標も併用する。",
            failure_interpretation="開口長、受波器間隔、active channel 選定、シェーディング、または評価 mask が不適切。",
            unit_reference="margin は mainlobe peak に対する相対 dB。max_nonlocal_level_db20 など絶対レベルを出す場合は dB re input RMS または dB re 1 uPa RMS を明記する。",
        ),
        BeamformingEvaluationCriterion(
            criterion_id="grating_lobe_and_ambiguity",
            category="BL",
            title="グレーティングローブ・鏡像曖昧性",
            purpose="疎配置や高周波で設計外の高い別 peak が出ていないかを確認する。",
            required_metrics=("mirror_level_db20", "outside_peak_level_db20", "active_max_gap_alias_limit_hz"),
            recommended_figures=("bl.png", "fraz.png"),
            pass_guideline="高域では active subset の最大間隔が波長に対して妥当で、mirror / grating lobe が mainlobe より十分低いこと。",
            failure_interpretation="高域で外側疎配置を使いすぎている、片舷アレイの方位定義が誤っている、または等 cos 表示軸が崩れている。",
            unit_reference="mirror/outside peak level は RMS 振幅レベル。シミュレーションでは dB re input RMS、実データでは dB re 1 uPa RMS などの基準量を付ける。",
        ),
        BeamformingEvaluationCriterion(
            criterion_id="three_db_overlap",
            category="Shading",
            title="隣接待受ビームの -3 dB 主ローブ overlap",
            purpose="後段のビーム補間が成立するよう、隣接待受方位の -3 dB 範囲が交差するかを確認する。",
            required_metrics=("minimum_three_db_overlap_margin_deg", "minimum_three_db_width_deg", "maximum_peak_error_deg"),
            recommended_figures=("operational_kaiser_bessel_shading_summary.png",),
            pass_guideline="全評価周波数で minimum overlap margin が 0 deg 以上であること。",
            failure_interpretation="待受ビーム数が少なすぎる、シェーディングで主ローブ幅が不足している、または評価軸が粗すぎる。",
            unit_reference="-3 dB は peak RMS 振幅に対する半 power 点の相対比。方位幅は deg。",
        ),
        BeamformingEvaluationCriterion(
            criterion_id="fraz_btr_consistency",
            category="FRAZ/BTR",
            title="FRAZ / BTR の表示整合性",
            purpose="BL で見たピークが FRAZ と BTR でも同じ方位・周波数・時間に出るかを確認する。",
            required_metrics=("fraz_global_peak_azimuth_deg", "fraz_global_peak_frequency_hz", "btr_global_peak_azimuth_mean_deg"),
            recommended_figures=("fraz.png", "btr.png"),
            pass_guideline="単一音源では FRAZ peak と BTR peak track が target 方位近傍にあること。複数音源では global peak だけで判断しない。",
            failure_interpretation="表示正規化、BTR 方位軸、FRAZ 周波数軸、または複数音源時の代表 peak の解釈が誤っている。",
            unit_reference="FRAZ は RMS レベルなら dB re input RMS または dB re 1 uPa RMS、スペクトル密度なら dB re uPa/sqrt(Hz) または dB re uPa/Hz@ch を明記する。BTR 相対表示は dB re frame max。",
        ),
        BeamformingEvaluationCriterion(
            criterion_id="source_visibility_preservation",
            category="SLC/Scan",
            title="全方位 scan での音源可視性維持",
            purpose="全方位 BL/FRAZ/BTR 表示で、target と interferer を別信号のピークとして残せているかを確認する。",
            required_metrics=(
                "known_source_peak_azimuths_deg",
                "known_source_level_delta_db",
                "false_peak_increase_db",
                "known_source_mainlobe_exclusion_mask",
            ),
            recommended_figures=("slc_bl_compare.png", "fraz.png", "btr.png"),
            pass_guideline="source-preserving scan では interferer 自体を消す必要はない。既知 source の mainlobe を維持し、別方位 false peak や guard 外 envelope を悪化させないことを確認する。",
            failure_interpretation="scan 出力を局所キャンセラとして解釈している、既知 source 方位を sidelobe と誤計上している、または SLC により別信号の可視性が落ちている可能性がある。",
            unit_reference="方位は deg。known_source_level_delta_db と false_peak_increase_db は処理前後の相対 dB。絶対レベルは dB re input RMS または dB re 1 uPa RMS を明記する。",
        ),
        BeamformingEvaluationCriterion(
            criterion_id="frequency_component_separation",
            category="Frequency/SLC",
            title="同一方位の周波数成分分離",
            purpose="同一方位に複数周波数成分が重畳する条件で、周波数ごとの target / interferer 成分を分けて評価する。",
            required_metrics=(
                "target_frequency_power_delta_db",
                "off_frequency_reduction_db",
                "frequency_bin_leakage_db",
                "analysis_bandwidth_hz",
            ),
            recommended_figures=("fraz.png", "target_leakage_levels.png"),
            pass_guideline="同一方位では空間的な null ではなく、STFT bin または帯域ごとの処理で目的周波数の target 成分を維持し、別周波数成分の漏れ込みを下げること。",
            failure_interpretation="時間領域 L=1 で周波数を混ぜている、分析帯域幅が広すぎる、窓漏れが大きい、または周波数ごとの guard / eta / loading を持っていない。",
            unit_reference="周波数は Hz、analysis_bandwidth_hz は Hz。レベル差と leakage は処理前または目的周波数成分に対する相対 dB。絶対レベルは dB re input RMS などで別記する。",
        ),
        BeamformingEvaluationCriterion(
            criterion_id="target_leakage_components",
            category="SLC",
            title="target beam 成分別漏れ込み",
            purpose="mixed / target-only / interferer-only を分け、SLC が target beam 上のどの成分を削っているかを確認する。",
            required_metrics=("raw_target_power_delta_db", "raw_interferer_reduction_db", "effective_interferer_reduction_db"),
            recommended_figures=("target_leakage_levels.png",),
            pass_guideline="local_leakage_canceller として採用する場合は、target-only で target 低下が小さく、interferer-only で protected target beam への漏れ込みが下がること。source-preserving scan では interferer 自体の低減を要求しない。raw と safety fallback 後を分けること。",
            failure_interpretation="target 自己消去、eta 過大、desired blocking 不足、または同一周波数・高相関条件で SLC が不適。",
            unit_reference="成分別 level は RMS 振幅レベル。差分や reduction は処理前後の相対 dB、絶対値は dB re input RMS または dB re 1 uPa RMS。",
        ),
        BeamformingEvaluationCriterion(
            criterion_id="adaptive_constraint_response",
            category="Adaptive BF",
            title="時間領域適応重みの制約応答",
            purpose="MVDR / LCMV / GSC の target 保護応答と null 制約が設計通り満たされているかを確認する。",
            required_metrics=(
                "target_constraint_response_error_db",
                "null_constraint_response_db20",
                "constraint_matrix_rank",
                "degree_of_freedom",
            ),
            recommended_figures=("protected_target_response_bl_overlay.png", "constraint_response_summary.png"),
            pass_guideline="target 制約は 0 dB re desired response 近傍、明示 null は十分低いこと。GSC は同じ制約の LCMV 解と応答が一致することを確認する。",
            failure_interpretation="制約ベクトルの位相符号、正負周波数制約、tap の並び、blocking matrix、または対角 loading の扱いが誤っている可能性が高い。",
            unit_reference="制約応答誤差は desired response に対する相対 dB。null 応答は振幅比 dB20。rank と自由度は count。",
        ),
        BeamformingEvaluationCriterion(
            criterion_id="slc_covariance_health",
            category="SLC",
            title="SLC 共分散・係数の健全性",
            purpose="SLC が数値的に安定し、参照自由度と snapshot 数が足りているかを確認する。",
            required_metrics=("reference_beam_count", "capacity", "weight_norm", "condition_number"),
            recommended_figures=(),
            pass_guideline="capacity が feasible で、weight norm や condition number が異常に大きくないこと。参照を間引いた場合は LIMITED として記録する。",
            failure_interpretation="参照不足、snapshot 不足、reference beam 間の高相関、loading 不足、または過大な自由度設定。",
            unit_reference="reference_beam_count と capacity は count、condition_number は無次元比、weight_norm は重みベクトル norm。dB 表記は使わない。",
        ),
        BeamformingEvaluationCriterion(
            criterion_id="waveform_integrity",
            category="Time waveform",
            title="処理後時間波形の健全性",
            purpose="SLC や固定整相後に時間波形の RMS、ピーク、NaN/inf、不要な発振がないかを確認する。",
            required_metrics=("output_rms_db20", "output_peak_db20", "nan_inf_count", "power_delta_db"),
            recommended_figures=("target_leakage_levels.png", "btr.png"),
            pass_guideline="処理前後の power delta が設計意図に沿い、NaN/inf がなく、target-only で不自然な大低下がないこと。",
            failure_interpretation="適応係数の発散、複素出力の扱いミス、fallback 不足、または時間波形再合成ミス。",
            unit_reference="output_rms/peak は RMS または peak 振幅レベル。実音圧なら dB re 1 uPa、シミュレーションなら dB re input RMS など基準量を付ける。power_delta_db は処理前後の相対 dB。",
        ),
        BeamformingEvaluationCriterion(
            criterion_id="input_output_level_consistency",
            category="Level/SNR",
            title="入力レベルに対する出力レベル・SN 改善の妥当性",
            purpose="入力信号・入力雑音レベルに対して、整相後の信号レベル、雑音レベル、SN 改善量が理論的に妥当かを確認する。",
            required_metrics=(
                "input_signal_rms_db20",
                "input_noise_rms_db20",
                "output_signal_rms_db20",
                "output_noise_rms_db20",
                "observed_snr_gain_db",
                "expected_spatial_snr_gain_db",
                "expected_analysis_snr_gain_db",
                "snr_gain_error_db",
            ),
            recommended_figures=("bl.png", "fraz.png", "btr.png", "target_leakage_levels.png"),
            pass_guideline="無相関雑音では空間合成により 20log10(sqrt(N_eff)) = 10log10(N_eff) dB の SN 改善を期待する。BL/FRAZ の dB20 RMS 表示では signal_db20 - noise_db20 の差で同じ値を確認する。",
            failure_interpretation="重み正規化、チャネル窓の N_eff 算出、入力レベル設定、相関雑音、分析幅、または振幅比と power 比の dB 表記対応が誤っている可能性がある。",
            unit_reference="信号・雑音の RMS レベルは dB re input RMS、dB re 1 uPa RMS、または dB re uPa/sqrt(Hz)。SN 改善量は signal_db20 - noise_db20 の相対 dB。",
        ),
        BeamformingEvaluationCriterion(
            criterion_id="array_file_consistency",
            category="Array",
            title="アレイ定義ファイルと active channel の整合",
            purpose="CH 数、active index、開口長、受波器間隔、周波数ごとの active subset が想定通りか確認する。",
            required_metrics=("physical_n_ch", "active_channel_count", "active_aperture_m", "active_min_spacing_m", "active_max_spacing_m"),
            recommended_figures=("sector_margin_summary.png", "operational_fractional_margin_summary.png"),
            pass_guideline="処理側は CH 数を手入力せず、ファイルから読み出すこと。高域では外側疎配置を active にしないこと。",
            failure_interpretation="ファイル入力方式が崩れている、周波数別 active subset が不一致、または高域 grating lobe の原因になる。",
            unit_reference="CH 数は count、位置・開口長・受波器間隔は m、周波数は Hz。dB 表記は使わない。",
        ),
        BeamformingEvaluationCriterion(
            criterion_id="runtime_budget",
            category="Runtime",
            title="CPU 実時間性",
            purpose="固定整相 + SLC が 1 秒入力を 1 秒以内に処理できるかを確認する。",
            required_metrics=("elapsed_sec", "input_duration_sec", "realtime_factor", "n_ref", "n_beam", "n_sample"),
            recommended_figures=(),
            pass_guideline="代表条件で realtime_factor <= 1 を満たすこと。SLC の n_ref と solve 回数を必ず記録する。",
            failure_interpretation="参照ビーム数過大、全ビームごとの solve、STFT bin 別共分散過多、または Python 実装の最適化不足。",
            unit_reference="時間は s、realtime_factor は無次元比、n_ref/n_beam/n_sample は count。dB 表記は使わない。",
        ),
    )


def _build_patterns() -> tuple[BeamformingEvaluationPattern, ...]:
    """検討パターンごとの評価基準セットを作る。

    Returns:
        検討パターン定義の tuple。各 pattern ID は一意である。
    """
    return (
        BeamformingEvaluationPattern(
            pattern_id="fixed_beam_single_source",
            title="固定整相 単一音源",
            description="整数遅延または小数遅延固定整相の基本性能を確認する。",
            required_criterion_ids=(
                "beam_peak_position",
                "sidelobe_peak_margin",
                "fraz_btr_consistency",
                "input_output_level_consistency",
            ),
            recommended_criterion_ids=("grating_lobe_and_ambiguity", "array_file_consistency", "waveform_integrity"),
        ),
        BeamformingEvaluationPattern(
            pattern_id="fixed_beam_multi_source",
            title="固定整相 複数音源",
            description="複数方位・複数周波数・同一方位異周波の表示と分離を確認する。",
            required_criterion_ids=("beam_peak_position", "fraz_btr_consistency", "grating_lobe_and_ambiguity"),
            recommended_criterion_ids=("sidelobe_peak_margin", "waveform_integrity", "input_output_level_consistency"),
        ),
        BeamformingEvaluationPattern(
            pattern_id="sparse_array_design",
            title="スパースアレイ設計",
            description="周波数ごとの active channel、開口長、grating lobe 余裕を確認する。",
            required_criterion_ids=("array_file_consistency", "sidelobe_peak_margin", "grating_lobe_and_ambiguity"),
            recommended_criterion_ids=("beam_peak_position", "input_output_level_consistency", "runtime_budget"),
        ),
        BeamformingEvaluationPattern(
            pattern_id="shading_design",
            title="シェーディング設計",
            description="Kaiser-Bessel 窓と待受ビーム数の妥当性を確認する。",
            required_criterion_ids=("three_db_overlap", "sidelobe_peak_margin", "array_file_consistency"),
            recommended_criterion_ids=("grating_lobe_and_ambiguity", "input_output_level_consistency"),
        ),
        BeamformingEvaluationPattern(
            pattern_id="slc_scan_multi_source_display",
            title="SLC 全方位 scan 複数音源表示",
            description="target と interferer を別方位・別周波数の観測対象として残す scan 表示用途を確認する。",
            required_criterion_ids=(
                "source_visibility_preservation",
                "mainlobe_preservation",
                "fraz_btr_consistency",
                "waveform_integrity",
            ),
            recommended_criterion_ids=("sidelobe_peak_margin", "array_file_consistency", "input_output_level_consistency"),
        ),
        BeamformingEvaluationPattern(
            pattern_id="slc_target_only",
            title="SLC target-only 高 SNR",
            description="SLC が目標信号だけの条件で自己消去しないことを確認する。",
            required_criterion_ids=(
                "mainlobe_preservation",
                "target_leakage_components",
                "waveform_integrity",
                "input_output_level_consistency",
            ),
            recommended_criterion_ids=("slc_covariance_health",),
        ),
        BeamformingEvaluationPattern(
            pattern_id="slc_same_frequency_interference",
            title="SLC 同一周波数干渉",
            description="local leakage canceller として、target と interferer が同一周波数で高相関になる厳しい条件を確認する。source-preserving scan では別信号として残す判断も許容する。",
            required_criterion_ids=("target_leakage_components", "mainlobe_preservation", "slc_covariance_health"),
            recommended_criterion_ids=("waveform_integrity", "fraz_btr_consistency", "input_output_level_consistency"),
        ),
        BeamformingEvaluationPattern(
            pattern_id="slc_different_frequency_interference",
            title="SLC 異周波数干渉",
            description="local leakage canceller として、時間領域 SLC L=1 が効きやすい異周波数 interferer 条件を確認する。別信号として出力する scan 用途とは分ける。",
            required_criterion_ids=("target_leakage_components", "mainlobe_preservation", "waveform_integrity"),
            recommended_criterion_ids=("slc_covariance_health", "fraz_btr_consistency", "input_output_level_consistency"),
        ),
        BeamformingEvaluationPattern(
            pattern_id="slc_same_azimuth_multi_frequency",
            title="SLC 同一方位・複数周波数",
            description="同一方位に複数周波数成分が重なる条件で、周波数軸上の分離と target 成分維持を確認する。",
            required_criterion_ids=(
                "frequency_component_separation",
                "mainlobe_preservation",
                "fraz_btr_consistency",
                "waveform_integrity",
            ),
            recommended_criterion_ids=("slc_covariance_health", "input_output_level_consistency"),
        ),
        BeamformingEvaluationPattern(
            pattern_id="time_domain_adaptive_mvdr_lcmv_gsc",
            title="時間領域 MVDR / LCMV / GSC",
            description="channel×tap 制約で target 保護、干渉 null、共分散健全性、実時間性を比較する。",
            required_criterion_ids=(
                "adaptive_constraint_response",
                "target_leakage_components",
                "mainlobe_preservation",
                "slc_covariance_health",
                "runtime_budget",
            ),
            recommended_criterion_ids=("waveform_integrity", "fraz_btr_consistency", "input_output_level_consistency"),
        ),
        BeamformingEvaluationPattern(
            pattern_id="slc_runtime",
            title="SLC 実時間性",
            description="固定整相 + SLC の CPU 処理量を確認する。",
            required_criterion_ids=("runtime_budget", "slc_covariance_health", "array_file_consistency"),
            recommended_criterion_ids=("waveform_integrity",),
        ),
    )


def list_beamforming_evaluation_criteria() -> tuple[BeamformingEvaluationCriterion, ...]:
    """評価基準カタログを返す。

    Returns:
        評価基準定義。shape 的には `[n_criterion]` に相当する 1 次元 tuple であり、
        各要素は `criterion_id` で識別する。

    境界条件:
        評価基準は静的な設計カタログであるため、入力引数は持たない。
        診断スクリプト側で評価値を計算し、この関数は評価すべき軸だけを返す。
    """
    # 評価値計算と評価項目選定を分離することで、BL/FRAZ/BTR/SLC 診断が増えても、
    # 「どの検討で何を必ず見るか」という設計判断を一箇所でレビューできるようにする。
    return _build_criteria()


def list_beamforming_evaluation_patterns() -> tuple[BeamformingEvaluationPattern, ...]:
    """検討パターン一覧を返す。

    Returns:
        検討パターン定義。shape 的には `[n_pattern]` に相当する 1 次元 tuple であり、
        各要素は `pattern_id` で識別する。

    境界条件:
        検討パターンは固定整相、疎アレイ、シェーディング、SLC などの用途分類であり、
        評価データの有無には依存しない。
    """
    # 全パターンで全評価を必須にすると検討が重くなりすぎるため、
    # 方式ごとの故障モードに合わせた評価セットをここで明示する。
    return _build_patterns()


def get_evaluation_criteria_for_pattern(
    pattern_id: str,
) -> tuple[tuple[BeamformingEvaluationCriterion, ...], tuple[BeamformingEvaluationCriterion, ...]]:
    """検討パターンに対応する必須・推奨評価基準を返す。

    Args:
        pattern_id: 検討パターン ID。単位はなく、`list_beamforming_evaluation_patterns`
            が返す `pattern_id` のいずれかである。

    Returns:
        `(required, recommended)`。
        `required` は必須評価基準、`recommended` は状況に応じて見る推奨評価基準である。
        どちらも shape 的には `[n_selected_criterion]` に相当する 1 次元 tuple である。

    Raises:
        ValueError: `pattern_id` が未定義、または pattern が未知の criterion ID を参照する場合。
    """
    # ID 参照を辞書化してから解決する。未知 ID を黙って無視すると、
    # 評価漏れを「評価不要」と誤認するため、require で設計不整合を即時に検出する。
    criteria_by_id = {criterion.criterion_id: criterion for criterion in _build_criteria()}
    patterns_by_id = {pattern.pattern_id: pattern for pattern in _build_patterns()}
    require(pattern_id in patterns_by_id, f"unknown evaluation pattern: {pattern_id}")
    pattern = patterns_by_id[pattern_id]

    required: list[BeamformingEvaluationCriterion] = []
    recommended: list[BeamformingEvaluationCriterion] = []
    for criterion_id in pattern.required_criterion_ids:
        require(criterion_id in criteria_by_id, f"unknown required criterion: {criterion_id}")
        required.append(criteria_by_id[criterion_id])
    for criterion_id in pattern.recommended_criterion_ids:
        require(criterion_id in criteria_by_id, f"unknown recommended criterion: {criterion_id}")
        recommended.append(criteria_by_id[criterion_id])
    return tuple(required), tuple(recommended)


def write_beamforming_evaluation_criteria_markdown(output_path: Path) -> None:
    """評価基準カタログと pattern 対応表を Markdown として保存する。

    Args:
        output_path: 保存先 Markdown パス。単位はファイルシステム上の path である。

    Returns:
        なし。評価基準を設計書として確認できるよう、`output_path` へ UTF-8 Markdown を保存する。

    Raises:
        OSError: 保存先ディレクトリ作成またはファイル書き込みに失敗した場合。

    境界条件:
        親ディレクトリがない場合は作成する。設計書を再生成しやすくするため、
        既存ファイルは同じ内容で上書きする。
    """
    criteria = _build_criteria()
    patterns = _build_patterns()
    criteria_by_id = {criterion.criterion_id: criterion for criterion in criteria}

    lines: list[str] = [
        "# Beamforming 評価基準カタログ",
        "",
        "## 1. 評価基準一覧",
        "",
    ]
    for criterion in criteria:
        lines.extend(
            [
                f"### {criterion.criterion_id}: {criterion.title}",
                "",
                f"- 分類: {criterion.category}",
                f"- 目的: {criterion.purpose}",
                f"- metric: {', '.join(criterion.required_metrics)}",
                f"- 推奨図: {', '.join(criterion.recommended_figures) if criterion.recommended_figures else 'なし'}",
                f"- 判定目安: {criterion.pass_guideline}",
                f"- 失敗時の解釈: {criterion.failure_interpretation}",
                f"- 単位・基準: {criterion.unit_reference}",
                "",
            ]
        )

    lines.extend(["## 2. 検討パターン別の評価基準", ""])
    for pattern in patterns:
        required_titles = [criteria_by_id[criterion_id].title for criterion_id in pattern.required_criterion_ids]
        recommended_titles = [criteria_by_id[criterion_id].title for criterion_id in pattern.recommended_criterion_ids]
        lines.extend(
            [
                f"### {pattern.pattern_id}: {pattern.title}",
                "",
                f"- 説明: {pattern.description}",
                f"- 必須: {', '.join(required_titles)}",
                f"- 推奨: {', '.join(recommended_titles) if recommended_titles else 'なし'}",
                "",
            ]
        )

    # dB は比率表現であり単位ではないため、設計書にも基準量の明記を要求する。
    lines.extend(
        [
            "## 3. dB 表記と単位・基準量の方針",
            "",
            "dB は比率表現であり、単独では物理単位を表さない。",
            "絶対音圧レベル、スペクトル密度、チャネル入力基準、正規化表示を区別するため、図・JSON・設計書では `dB re ...` の基準量を明記する。",
            "例として、音圧 RMS は `dB re 1 uPa RMS`、振幅スペクトル密度は `dB re uPa/sqrt(Hz)`、power spectral density は `dB re uPa^2/Hz` または設計で定義した `dB re uPa/Hz@ch`、シミュレーション正規化値は `dB re input RMS` と書く。",
            "BTR のような相対正規化図では `dB re frame max` のように正規化基準を示す。",
            "",
            "## 4. 入力レベルに対する出力レベル・SN 改善の判断式",
            "",
            "### 3.1 信号レベル",
            "",
            "固定整相重みが target 方向で無歪み正規化されている場合、target 信号の出力 RMS は入力 target RMS と同程度になる。",
            "CBF の `w^H a = 1` や時間領域 delay-and-sum の `sum(weights)=1` は、この確認の前提である。",
            "SLC 後は mainlobe 維持評価と合わせて、target-only 条件で `target_power_delta_db` が過大に低下していないことを確認する。",
            "",
            "### 3.2 無相関雑音の空間 SN 改善",
            "",
            "各 channel の雑音が同一分散かつ無相関で、target 方向の信号だけが整相される場合、出力雑音分散は `sigma_out^2 = sigma_in^2 sum(|w_ch|^2)` となる。",
            "したがって、無歪み正規化された重みに対する空間 SN 改善量は power 比では `-10log10(sum(|w_ch|^2)) dB` で評価できる。",
            "BL / FRAZ / BTR の表示は RMS 振幅の dB20 なので、同じ改善量を `20log10(1 / sqrt(sum(|w_ch|^2))) dB` と書いても数値は一致する。",
            "矩形窓の delay-and-sum では `w_ch = 1/N` なので、期待改善量は `20log10(sqrt(N)) = 10log10(N) dB` である。",
            "シェーディングを使う場合は単純な active channel 数ではなく、`N_eff = (sum(g_ch))^2 / sum(g_ch^2)` を使い、期待改善量を `20log10(sqrt(N_eff)) = 10log10(N_eff) dB` とする。",
            "",
            "### 3.3 分析幅による SN 改善",
            "",
            "BL / FRAZ のように時間方向または周波数方向へ平均した power を表示する場合、独立平均数 `M` に応じて雑音床の推定ばらつきは下がる。",
            "この効果は空間合成利得とは別に `expected_analysis_snr_gain_db` として記録する。",
            "時間領域の瞬時波形出力では、明示的な平均、帯域制限、STFT 積分を入れていない限り、この分析幅由来の改善を期待しない。",
            "",
            "### 3.4 判定時の注意",
            "",
            "信号レベルと雑音レベルを RMS 振幅として表示する場合は dB20 で扱い、`signal_db20 - noise_db20` が SN である。",
            "同じ SN 改善量を power 比から導くと dB10、RMS 振幅比から導くと dB20 になるが、`10log10(N_eff) = 20log10(sqrt(N_eff))` なので数値は一致する。",
            "入力雑音が channel 間で相関している場合、または干渉音が coherent source の場合、`20log10(sqrt(N_eff))` の改善は成り立たない。",
            "高域で active subset やシェーディングが変わる場合は、周波数ごとに `N_eff` と分析幅を記録して比較する。",
            "",
        ]
    )

    path = Path(output_path)
    # 設計書は生成物だが、レビュー対象として doc/SLC 配下に保存する。
    # 親ディレクトリがない場合でもスクリプト単体で再生成できるようにする。
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "BeamformingEvaluationCriterion",
    "BeamformingEvaluationPattern",
    "get_evaluation_criteria_for_pattern",
    "list_beamforming_evaluation_criteria",
    "list_beamforming_evaluation_patterns",
    "write_beamforming_evaluation_criteria_markdown",
]
