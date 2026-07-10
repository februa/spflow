# Beamforming 評価基準カタログ

## 1. 評価基準一覧

### beam_peak_position: ピーク方位・ピーク周波数の正しさ

- 分類: BL/FRAZ/BTR
- 目的: 到来方位と周波数に対して、最大応答が正しい位置に出ているかを確認する。
- metric: bl_peak_azimuth_deg, fraz_global_peak_azimuth_deg, fraz_global_peak_frequency_hz
- 推奨図: bl.png, fraz.png, btr.png
- 判定目安: 単一音源では peak 方位が最近傍待受方位に一致し、FRAZ peak 周波数が入力周波数に一致すること。
- 失敗時の解釈: 方位軸、等 cos 軸、遅延符号、周波数ビン対応、またはアレイ側面定義が誤っている可能性が高い。
- 単位・基準: 方位は deg、周波数は Hz。レベルを併記する場合は dB re input RMS、dB re 1 uPa RMS など基準量を明記する。

### mainlobe_preservation: メインローブ維持

- 分類: BL
- 目的: SLC やシェーディング後に target mainlobe の位置とレベルが壊れていないかを確認する。
- metric: mainlobe_level_delta_db, peak_azimuth_shift_deg, target_power_delta_db
- 推奨図: slc_bl_compare.png, target_leakage_levels.png
- 判定目安: 方式比較では mainlobe level delta と target power delta を別々に見る。SLC 評価では target-only 条件も必ず確認する。
- 失敗時の解釈: desired 成分を reference へ混ぜている、blocking が不足している、eta が大きすぎる、または guard が狭すぎる。
- 単位・基準: mainlobe_level_delta_db と target_power_delta_db は処理前または固定整相出力に対する比率 dB。絶対レベルは dB re 1 uPa RMS などで別記する。

### sidelobe_peak_margin: サイドローブ peak margin

- 分類: BL
- 目的: mainlobe peak と guard 外 sidelobe peak の差、および第一副極レベルの改善量が十分かを確認する。
- metric: local_to_nonlocal_margin_db, max_nonlocal_level_db20, first_sidelobe_reduction_db, worst_peak_margin_db
- 推奨図: bl.png, slc_bl_compare.png, margin_summary.png
- 判定目安: 固定整相・アレイ設計では 13 dB 以上を基準にする。SLC / 適応方式の before/after 比較では第一副極と guard 外 peak が実際に下がること。複数音源では既知 source 方位の mainlobe を除外した指標も併用する。
- 失敗時の解釈: 開口長、受波器間隔、active channel 選定、シェーディング、または評価 mask が不適切。
- 単位・基準: margin は mainlobe peak に対する相対 dB。max_nonlocal_level_db20 など絶対レベルを出す場合は dB re input RMS または dB re 1 uPa RMS を明記する。

### grating_lobe_and_ambiguity: グレーティングローブ・鏡像曖昧性

- 分類: BL
- 目的: 疎配置や高周波で設計外の高い別 peak が出ていないかを確認する。
- metric: mirror_level_db20, outside_peak_level_db20, active_max_gap_alias_limit_hz
- 推奨図: bl.png, fraz.png
- 判定目安: 高域では active subset の最大間隔が波長に対して妥当で、mirror / grating lobe が mainlobe より十分低いこと。
- 失敗時の解釈: 高域で外側疎配置を使いすぎている、片舷アレイの方位定義が誤っている、または等 cos 表示軸が崩れている。
- 単位・基準: mirror/outside peak level は RMS 振幅レベル。シミュレーションでは dB re input RMS、実データでは dB re 1 uPa RMS などの基準量を付ける。

### three_db_overlap: 隣接待受ビームの -3 dB 主ローブ overlap

- 分類: Shading
- 目的: 後段のビーム補間が成立するよう、隣接待受方位の -3 dB 範囲が交差するかを確認する。
- metric: minimum_three_db_overlap_margin_deg, minimum_three_db_width_deg, maximum_peak_error_deg
- 推奨図: operational_kaiser_bessel_shading_summary.png
- 判定目安: 全評価周波数で minimum overlap margin が 0 deg 以上であること。
- 失敗時の解釈: 待受ビーム数が少なすぎる、シェーディングで主ローブ幅が不足している、または評価軸が粗すぎる。
- 単位・基準: -3 dB は peak RMS 振幅に対する半 power 点の相対比。方位幅は deg。

### fraz_btr_consistency: FRAZ / BTR の表示整合性

- 分類: FRAZ/BTR
- 目的: BL で見たピークが FRAZ と BTR でも同じ方位・周波数・時間に出るかを確認する。
- metric: fraz_global_peak_azimuth_deg, fraz_global_peak_frequency_hz, btr_global_peak_azimuth_mean_deg
- 推奨図: fraz.png, btr.png
- 判定目安: 単一音源では FRAZ peak と BTR peak track が target 方位近傍にあること。複数音源では global peak だけで判断しない。
- 失敗時の解釈: 表示正規化、BTR 方位軸、FRAZ 周波数軸、または複数音源時の代表 peak の解釈が誤っている。
- 単位・基準: FRAZ は RMS レベルなら dB re input RMS または dB re 1 uPa RMS、スペクトル密度なら dB re uPa/sqrt(Hz) または dB re uPa/Hz@ch を明記する。BTR 相対表示は dB re frame max。

### abf_like_non_source_suppression: ABF-like non-source sector 抑圧

- 分類: Adaptive BF
- 目的: 既知 source 方位を source mask として保持し、その guard 外 non-source sector の応答包絡線が下がるかを確認する。
- metric: source_peak_delta_db, source_azimuth_error_deg, non_source_global_peak_delta_db, non_source_p95_level_delta_db, non_source_integrated_level_delta_db, source_to_non_source_margin_delta_db, false_peak_count, angular_robustness_min_reduction_db, weight_norm_or_wng, condition_number, realtime_factor
- 推奨図: abf_like_non_source_envelope_overlay.png, source_mask_response_summary.png
- 判定目安: 採否は exact marker null ではなく、source mask 外の global peak、p95、integrated level、source-to-non-source margin、false peak count で判定する。known source peak は維持する。
- 失敗時の解釈: 点 null や target beam leakage だけを最適化し、non-source sector へ sidelobe を押し出している可能性が高い。source mask、guard 幅、train/test 方位ずれ、または sector 制約が不足している。
- 単位・基準: source / non-source level は RMS dB20。絶対値は dB re input RMS、処理前後差は dB re before level として基準を明記する。WNG/weight_norm、condition_number、realtime_factor は無次元比。

### source_visibility_preservation: 全方位 scan での音源可視性維持

- 分類: SLC/Scan
- 目的: 全方位 BL/FRAZ/BTR 表示で、target と interferer を別信号のピークとして残せているかを確認する。
- metric: known_source_peak_azimuths_deg, known_source_level_delta_db, false_peak_increase_db, known_source_mainlobe_exclusion_mask
- 推奨図: slc_bl_compare.png, fraz.png, btr.png
- 判定目安: source-preserving scan では interferer 自体を消す必要はない。既知 source の mainlobe を維持し、別方位 false peak や guard 外 envelope を悪化させないことを確認する。
- 失敗時の解釈: scan 出力を局所キャンセラとして解釈している、既知 source 方位を sidelobe と誤計上している、または SLC により別信号の可視性が落ちている可能性がある。
- 単位・基準: 方位は deg。known_source_level_delta_db と false_peak_increase_db は処理前後の相対 dB。絶対レベルは dB re input RMS または dB re 1 uPa RMS を明記する。

### frequency_component_separation: 同一方位の周波数成分分離

- 分類: Frequency/SLC
- 目的: 同一方位に複数周波数成分が重畳する条件で、周波数ごとの target / interferer 成分を分けて評価する。
- metric: target_frequency_power_delta_db, off_frequency_reduction_db, frequency_bin_leakage_db, analysis_bandwidth_hz
- 推奨図: fraz.png, target_leakage_levels.png
- 判定目安: 同一方位では空間的な null ではなく、STFT bin または帯域ごとの処理で目的周波数の target 成分を維持し、別周波数成分の漏れ込みを下げること。
- 失敗時の解釈: 時間領域 L=1 で周波数を混ぜている、分析帯域幅が広すぎる、窓漏れが大きい、または周波数ごとの guard / eta / loading を持っていない。
- 単位・基準: 周波数は Hz、analysis_bandwidth_hz は Hz。レベル差と leakage は処理前または目的周波数成分に対する相対 dB。絶対レベルは dB re input RMS などで別記する。

### target_leakage_components: target beam 成分別漏れ込み

- 分類: SLC
- 目的: mixed / target-only / interferer-only を分け、SLC が target beam 上のどの成分を削っているかを確認する。
- metric: raw_target_power_delta_db, raw_interferer_reduction_db, effective_interferer_reduction_db
- 推奨図: target_leakage_levels.png
- 判定目安: local_leakage_canceller として採用する場合は、target-only で target 低下が小さく、interferer-only で protected target beam への漏れ込みが下がること。source-preserving scan では interferer 自体の低減を要求しない。raw と safety fallback 後を分けること。
- 失敗時の解釈: target 自己消去、eta 過大、desired blocking 不足、または同一周波数・高相関条件で SLC が不適。
- 単位・基準: 成分別 level は RMS 振幅レベル。差分や reduction は処理前後の相対 dB、絶対値は dB re input RMS または dB re 1 uPa RMS。

### adaptive_constraint_response: 時間領域適応重みの制約応答

- 分類: Adaptive BF
- 目的: MVDR / LCMV / GSC の target 保護応答と null 制約が設計通り満たされているかを確認する。
- metric: target_constraint_response_error_db, null_constraint_response_db20, constraint_matrix_rank, degree_of_freedom
- 推奨図: protected_target_response_bl_overlay.png, constraint_response_summary.png
- 判定目安: target 制約は 0 dB re desired response 近傍、明示 null は十分低いこと。GSC は同じ制約の LCMV 解と応答が一致することを確認する。
- 失敗時の解釈: 制約ベクトルの位相符号、正負周波数制約、tap の並び、blocking matrix、または対角 loading の扱いが誤っている可能性が高い。
- 単位・基準: 制約応答誤差は desired response に対する相対 dB。null 応答は振幅比 dB20。rank と自由度は count。

### slc_covariance_health: SLC 共分散・係数の健全性

- 分類: SLC
- 目的: SLC が数値的に安定し、参照自由度と snapshot 数が足りているかを確認する。
- metric: reference_beam_count, capacity, weight_norm, condition_number
- 推奨図: なし
- 判定目安: capacity が feasible で、weight norm や condition number が異常に大きくないこと。参照を間引いた場合は LIMITED として記録する。
- 失敗時の解釈: 参照不足、snapshot 不足、reference beam 間の高相関、loading 不足、または過大な自由度設定。
- 単位・基準: reference_beam_count と capacity は count、condition_number は無次元比、weight_norm は重みベクトル norm。dB 表記は使わない。

### waveform_integrity: 処理後時間波形の健全性

- 分類: Time waveform
- 目的: SLC や固定整相後に時間波形の RMS、ピーク、NaN/inf、不要な発振がないかを確認する。
- metric: output_rms_db20, output_peak_db20, nan_inf_count, power_delta_db
- 推奨図: target_leakage_levels.png, btr.png
- 判定目安: 処理前後の power delta が設計意図に沿い、NaN/inf がなく、target-only で不自然な大低下がないこと。
- 失敗時の解釈: 適応係数の発散、複素出力の扱いミス、fallback 不足、または時間波形再合成ミス。
- 単位・基準: output_rms/peak は RMS または peak 振幅レベル。実音圧なら dB re 1 uPa、シミュレーションなら dB re input RMS など基準量を付ける。power_delta_db は処理前後の相対 dB。

### input_output_level_consistency: 入力レベルに対する出力レベル・SN 改善の妥当性

- 分類: Level/SNR
- 目的: 入力信号・入力雑音レベルに対して、整相後の信号レベル、雑音レベル、SN 改善量が理論的に妥当かを確認する。
- metric: input_signal_rms_db20, input_noise_rms_db20, output_signal_rms_db20, output_noise_rms_db20, observed_snr_gain_db, expected_spatial_snr_gain_db, expected_analysis_snr_gain_db, snr_gain_error_db
- 推奨図: bl.png, fraz.png, btr.png, target_leakage_levels.png
- 判定目安: 無相関雑音では空間合成により 20log10(sqrt(N_eff)) = 10log10(N_eff) dB の SN 改善を期待する。BL/FRAZ の dB20 RMS 表示では signal_db20 - noise_db20 の差で同じ値を確認する。
- 失敗時の解釈: 重み正規化、チャネル窓の N_eff 算出、入力レベル設定、相関雑音、分析幅、または振幅比と power 比の dB 表記対応が誤っている可能性がある。
- 単位・基準: 信号・雑音の RMS レベルは dB re input RMS、dB re 1 uPa RMS、または dB re uPa/sqrt(Hz)。SN 改善量は signal_db20 - noise_db20 の相対 dB。

### array_file_consistency: アレイ定義ファイルと active channel の整合

- 分類: Array
- 目的: CH 数、active index、開口長、受波器間隔、周波数ごとの active subset が想定通りか確認する。
- metric: physical_n_ch, active_channel_count, active_aperture_m, active_min_spacing_m, active_max_spacing_m
- 推奨図: sector_margin_summary.png, operational_fractional_margin_summary.png
- 判定目安: 処理側は CH 数を手入力せず、ファイルから読み出すこと。高域では外側疎配置を active にしないこと。
- 失敗時の解釈: ファイル入力方式が崩れている、周波数別 active subset が不一致、または高域 grating lobe の原因になる。
- 単位・基準: CH 数は count、位置・開口長・受波器間隔は m、周波数は Hz。dB 表記は使わない。

### runtime_budget: CPU 実時間性

- 分類: Runtime
- 目的: 固定整相 + SLC が 1 秒入力を 1 秒以内に処理できるかを確認する。
- metric: elapsed_sec, input_duration_sec, realtime_factor, n_ref, n_beam, n_sample
- 推奨図: なし
- 判定目安: 代表条件で realtime_factor <= 1 を満たすこと。SLC の n_ref と solve 回数を必ず記録する。
- 失敗時の解釈: 参照ビーム数過大、全ビームごとの solve、STFT bin 別共分散過多、または Python 実装の最適化不足。
- 単位・基準: 時間は s、realtime_factor は無次元比、n_ref/n_beam/n_sample は count。dB 表記は使わない。

## 2. 検討パターン別の評価基準

### fixed_beam_single_source: 固定整相 単一音源

- 説明: 整数遅延または小数遅延固定整相の基本性能を確認する。
- 必須: ピーク方位・ピーク周波数の正しさ, サイドローブ peak margin, FRAZ / BTR の表示整合性, 入力レベルに対する出力レベル・SN 改善の妥当性
- 推奨: グレーティングローブ・鏡像曖昧性, アレイ定義ファイルと active channel の整合, 処理後時間波形の健全性

### fixed_beam_multi_source: 固定整相 複数音源

- 説明: 複数方位・複数周波数・同一方位異周波の表示と分離を確認する。
- 必須: ピーク方位・ピーク周波数の正しさ, FRAZ / BTR の表示整合性, グレーティングローブ・鏡像曖昧性
- 推奨: サイドローブ peak margin, 処理後時間波形の健全性, 入力レベルに対する出力レベル・SN 改善の妥当性

### sparse_array_design: スパースアレイ設計

- 説明: 周波数ごとの active channel、開口長、grating lobe 余裕を確認する。
- 必須: アレイ定義ファイルと active channel の整合, サイドローブ peak margin, グレーティングローブ・鏡像曖昧性
- 推奨: ピーク方位・ピーク周波数の正しさ, 入力レベルに対する出力レベル・SN 改善の妥当性, CPU 実時間性

### shading_design: シェーディング設計

- 説明: Kaiser-Bessel 窓と待受ビーム数の妥当性を確認する。
- 必須: 隣接待受ビームの -3 dB 主ローブ overlap, サイドローブ peak margin, アレイ定義ファイルと active channel の整合
- 推奨: グレーティングローブ・鏡像曖昧性, 入力レベルに対する出力レベル・SN 改善の妥当性

### slc_scan_multi_source_display: SLC 全方位 scan 複数音源表示

- 説明: target と interferer を別方位・別周波数の観測対象として残す scan 表示用途を確認する。
- 必須: 全方位 scan での音源可視性維持, メインローブ維持, FRAZ / BTR の表示整合性, 処理後時間波形の健全性
- 推奨: サイドローブ peak margin, アレイ定義ファイルと active channel の整合, 入力レベルに対する出力レベル・SN 改善の妥当性

### slc_target_only: SLC target-only 高 SNR

- 説明: SLC が目標信号だけの条件で自己消去しないことを確認する。
- 必須: メインローブ維持, target beam 成分別漏れ込み, 処理後時間波形の健全性, 入力レベルに対する出力レベル・SN 改善の妥当性
- 推奨: SLC 共分散・係数の健全性

### slc_same_frequency_interference: SLC 同一周波数干渉

- 説明: local leakage canceller として、target と interferer が同一周波数で高相関になる厳しい条件を確認する。source-preserving scan では別信号として残す判断も許容する。
- 必須: target beam 成分別漏れ込み, メインローブ維持, SLC 共分散・係数の健全性
- 推奨: 処理後時間波形の健全性, FRAZ / BTR の表示整合性, 入力レベルに対する出力レベル・SN 改善の妥当性

### slc_different_frequency_interference: SLC 異周波数干渉

- 説明: local leakage canceller として、時間領域 SLC L=1 が効きやすい異周波数 interferer 条件を確認する。別信号として出力する scan 用途とは分ける。
- 必須: target beam 成分別漏れ込み, メインローブ維持, 処理後時間波形の健全性
- 推奨: SLC 共分散・係数の健全性, FRAZ / BTR の表示整合性, 入力レベルに対する出力レベル・SN 改善の妥当性

### slc_same_azimuth_multi_frequency: SLC 同一方位・複数周波数

- 説明: 同一方位に複数周波数成分が重なる条件で、周波数軸上の分離と target 成分維持を確認する。
- 必須: 同一方位の周波数成分分離, メインローブ維持, FRAZ / BTR の表示整合性, 処理後時間波形の健全性
- 推奨: SLC 共分散・係数の健全性, 入力レベルに対する出力レベル・SN 改善の妥当性

### ABF_like_non_source_suppression: ABF-like non-source sector 抑圧

- 説明: 既知 source 方位を source mask とし、guard 外 non-source sector の包絡線抑圧で固定整相、SLC、LCMV/GSC、STFT/Capon を比較する。
- 必須: ABF-like non-source sector 抑圧, メインローブ維持, SLC 共分散・係数の健全性, CPU 実時間性
- 推奨: FRAZ / BTR の表示整合性, 入力レベルに対する出力レベル・SN 改善の妥当性, 処理後時間波形の健全性

### time_domain_adaptive_mvdr_lcmv_gsc: 時間領域 MVDR / LCMV / GSC

- 説明: channel×tap 制約で target 保護、干渉 null、共分散健全性、実時間性を比較する。
- 必須: 時間領域適応重みの制約応答, target beam 成分別漏れ込み, メインローブ維持, SLC 共分散・係数の健全性, CPU 実時間性
- 推奨: 処理後時間波形の健全性, FRAZ / BTR の表示整合性, 入力レベルに対する出力レベル・SN 改善の妥当性

### slc_runtime: SLC 実時間性

- 説明: 固定整相 + SLC の CPU 処理量を確認する。
- 必須: CPU 実時間性, SLC 共分散・係数の健全性, アレイ定義ファイルと active channel の整合
- 推奨: 処理後時間波形の健全性

## 3. dB 表記と単位・基準量の方針

dB は比率表現であり、単独では物理単位を表さない。
絶対音圧レベル、スペクトル密度、チャネル入力基準、正規化表示を区別するため、図・JSON・設計書では `dB re ...` の基準量を明記する。
例として、音圧 RMS は `dB re 1 uPa RMS`、振幅スペクトル密度は `dB re uPa/sqrt(Hz)`、power spectral density は `dB re uPa^2/Hz` または設計で定義した `dB re uPa/Hz@ch`、シミュレーション正規化値は `dB re input RMS` と書く。
BTR のような相対正規化図では `dB re frame max` のように正規化基準を示す。

## 4. 入力レベルに対する出力レベル・SN 改善の判断式

### 3.1 信号レベル

固定整相重みが target 方向で無歪み正規化されている場合、target 信号の出力 RMS は入力 target RMS と同程度になる。
CBF の `w^H a = 1` や時間領域 delay-and-sum の `sum(weights)=1` は、この確認の前提である。
SLC 後は mainlobe 維持評価と合わせて、target-only 条件で `target_power_delta_db` が過大に低下していないことを確認する。

### 3.2 無相関雑音の空間 SN 改善

各 channel の雑音が同一分散かつ無相関で、target 方向の信号だけが整相される場合、出力雑音分散は `sigma_out^2 = sigma_in^2 sum(|w_ch|^2)` となる。
したがって、無歪み正規化された重みに対する空間 SN 改善量は power 比では `-10log10(sum(|w_ch|^2)) dB` で評価できる。
BL / FRAZ / BTR の表示は RMS 振幅の dB20 なので、同じ改善量を `20log10(1 / sqrt(sum(|w_ch|^2))) dB` と書いても数値は一致する。
矩形窓の delay-and-sum では `w_ch = 1/N` なので、期待改善量は `20log10(sqrt(N)) = 10log10(N) dB` である。
シェーディングを使う場合は単純な active channel 数ではなく、`N_eff = (sum(g_ch))^2 / sum(g_ch^2)` を使い、期待改善量を `20log10(sqrt(N_eff)) = 10log10(N_eff) dB` とする。

### 3.3 分析幅による SN 改善

BL / FRAZ のように時間方向または周波数方向へ平均した power を表示する場合、独立平均数 `M` に応じて雑音床の推定ばらつきは下がる。
この効果は空間合成利得とは別に `expected_analysis_snr_gain_db` として記録する。
時間領域の瞬時波形出力では、明示的な平均、帯域制限、STFT 積分を入れていない限り、この分析幅由来の改善を期待しない。

### 3.4 判定時の注意

信号レベルと雑音レベルを RMS 振幅として表示する場合は dB20 で扱い、`signal_db20 - noise_db20` が SN である。
同じ SN 改善量を power 比から導くと dB10、RMS 振幅比から導くと dB20 になるが、`10log10(N_eff) = 20log10(sqrt(N_eff))` なので数値は一致する。
入力雑音が channel 間で相関している場合、または干渉音が coherent source の場合、`20log10(sqrt(N_eff))` の改善は成り立たない。
高域で active subset やシェーディングが変わる場合は、周波数ごとに `N_eff` と分析幅を記録して比較する。
