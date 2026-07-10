# 運用スパースアレイへの SLC 導入検討

## 1. 目的

本書は、305 ch / 300.6 m の運用スパースアレイ、保存済み小数遅延 FIR バンク、151 本待受ビームを前提に、固定整相後段へ SLC を入れる条件を整理する。

SLC の評価では、target mainlobe が維持されることだけでは採用判定にしない。また、interferer を常に消すことも要求しない。Beamforming Evaluation の観点に従い、`source-preserving scan`、`local_leakage_canceller`、`BL_sidelobe_reducer` の role を分け、target-only 保護、音源可視性、target beam leakage、BL/FRAZ/BTR の整合、runtime、condition number、weight norm を確認する。

---

## 2. 評価入力

評価で使う入力ファイルは次である。

```text
array:
  artifacts/beamforming/operational_sparse_array/operational_sparse_array_fs32768.json

channel shading:
  artifacts/beamforming/operational_shading/operational_kaiser_bessel_shading_fs32768.json

fractional delay FIR bank:
  artifacts/beamforming/fractional_delay_filter_bank_65x63.npz
```

10000 Hz の active 条件は次である。

```text
physical channel count = 305
physical aperture      = 300.6 m
active channel count   = 61
active aperture        = 6.0 m
shading weight min/max = 0.2048847564 / 1.0
effective channel count = 53.236
```

channel shading は、固定整相出力と SLC の理論 response matrix の両方に同じ正規化で掛ける。

```text
y_beam = sum(w_ch * y_ch) / sum(w_ch)
```

片側だけに shading を掛けると、SLC が保護すべき desired response と実際の beam output が一致せず、target 自己消去または誤った干渉推定を起こす。

---

## 3. 評価方式

### 3.1 narrowband scan SLC

`examples/beamforming/operational_array_fractional_delay_slc_diagnostics.py` で評価する。小数遅延固定整相後の全 beam output を周波数選択 snapshot に変換し、各 scan beam を順に保護 target として SLC を適用する。

この方式は BL/FRAZ/BTR の before / after を作る診断に向いている。一方、全 beam scan では interferer 方位の beam もその beam 自身にとっては desired mainlobe であるため、干渉源ピークそのものを消す評価には使わない。採否 pattern は `slc_scan_multi_source_display` とし、既知 source が別ピークとして残ること、false peak や guard 外 envelope が悪化しないことを確認する。

### 3.2 target-centric 時間領域 SLC

`examples/beamforming/operational_array_time_domain_slc_diagnostics.py` で評価する。固定整相後の target beam を保護出力 `d[n]`、guard 外 beam を reference `u[n]` とし、時間領域の共分散を block ごとに指数忘却積分する。

評価では mixed / target-only / interferer-only を同じ固定整相に通し、mixed で得た SLC 係数を各成分へ適用する。これにより、target beam 上の target 保護量と interferer leakage 低減量を分けて測る。

---

## 4. 2026-07-05 narrowband scan SLC 評価

評価条件は次である。

```text
processing frequency = 10000 Hz
target               = 90 deg, 10000 Hz, 0 dB re input RMS
interferer           = 60 deg, 10000 Hz, -6 dB re input RMS
beam count           = 151
guard                = 10 beam
max_reference_beams  = 48
snapshot block       = 64 sample
loading              = 3.0e-2
eta_normal/limited   = 0.25 / 0.15
```

生成物:

```text
artifacts/beamforming/operational_fractional_delay_slc_diagnostics/10000Hz_151beam/operational_slc_case_summary.json
artifacts/beamforming/operational_fractional_delay_slc_diagnostics/10000Hz_151beam/slc_summary.json
artifacts/beamforming/operational_fractional_delay_slc_diagnostics/10000Hz_151beam/slc_bl_compare.png
artifacts/beamforming/operational_fractional_delay_slc_diagnostics/10000Hz_151beam/slc_fraz.png
artifacts/beamforming/operational_fractional_delay_slc_diagnostics/10000Hz_151beam/slc_btr.png
```

結果:

```text
all_mainlobes_preserved             = true
mean_mainlobe_level_delta_db        = +0.0013 dB re before level
mean_sidelobe_reduction_db          = -0.0007 dB re before level
mean_mainlobe_margin_improvement_db = +0.0006 dB re before level
interferer nearest reduction        = -0.0007 dB re before level
normal_beam_count                   = 0
limited_beam_count                  = 151
```

判定:

- target mainlobe は維持された。
- 同一周波数 interferer は別方位 source として残る。source-preserving scan ではこれは許容される。
- `interferer nearest reduction` は scan 用途の採否指標ではなく、局所 leakage canceller として解釈した場合の参考値である。
- 全 beam が `LIMITED_REFERENCE` であり、`max_reference_beams=48` の制限下での評価である。
- 実装は設計通り、現行 shading を固定整相出力と response matrix の両方へ適用している。矩形重みの取りこぼしではない。

---

## 5. 2026-07-05 target-centric 時間領域 SLC 評価

評価条件は次である。

```text
processing frequency = 10000 Hz
target               = 90 deg, 10000 Hz, 0 dB re input RMS
interferer           = 60 deg, 8192 Hz, -6 dB re input RMS
duration             = 5.0 s
beam count           = 151
guard                = 10 beam
reference_beam_count = 130
block size           = 8192 sample
memory_time_sec      = 3.0 s
loading              = 3.0e-2
tap_len              = 1
eta_normal/limited   = 1.0 / 1.0
```

生成物:

```text
artifacts/beamforming/operational_time_domain_slc_diagnostics/10000Hz_151beam_memory3s_8192Hz_interferer/time_domain_slc_leakage_summary.json
artifacts/beamforming/operational_time_domain_slc_diagnostics/10000Hz_151beam_memory3s_8192Hz_interferer/protected_target_response_bl_overlay.png
artifacts/beamforming/operational_time_domain_slc_diagnostics/10000Hz_151beam_memory3s_8192Hz_interferer/protected_target_interferer_response_bl_overlay.png
artifacts/beamforming/operational_time_domain_slc_diagnostics/10000Hz_151beam_memory3s_8192Hz_interferer/slc_component_spectrum_overlay.png
```

成分別評価。これは block ごとの streaming SLC 係数を、実際の target-only / interferer-only 時系列へ適用した RMS である。

```text
target_power_delta_db                    = -0.0027 dB re before level
streaming_interferer_reduction_db        = +23.0457 dB re before level
mixed_power_delta_db                     = -0.0041 dB re before level
condition_number                         = 2181.72
weight_norm                              = 1.7136
realtime_factor                          = 0.259
safety_fallback_required                 = false
```

BL 評価:

```text
slc_bl_improvement_pass = false
failure_reason          = guard_outside_peak_or_first_sidelobe_not_reduced_or_local_worsening_detected

target frequency:
  target level delta at target = 0.000 dB
  guard outside peak delta     = +2.025 dB
  first sidelobe peak delta    = +12.111 dB

interferer frequency:
  target-beam leakage reduction at target = +0.519 dB
  interferer marker reduction             = -10.149 dB、すなわち 10.149 dB 悪化
  guard outside peak delta                = +3.146 dB
  first sidelobe peak delta               = +8.470 dB
```

BL 図は、最後に有効だった SLC 係数を固定して source 方位を sweep した応答である。一方、`streaming_interferer_reduction_db` は block ごとの SLC 係数を実際の interferer-only 時系列に適用した RMS である。したがって、この 2 つは同じ量ではない。

判定:

- target beam 上の streaming interferer 成分 RMS は 23.0 dB 低下している。
- target-only 低下は -0.0027 dB であり、成分別時系列では target 保護は成立している。
- ただし representative BL 図では、60.44 deg marker が 10.149 dB 悪化し、guard 外 peak と first sidelobe も悪化している。
- この図単体からは、interferer 方位の local leakage 低減や第一副極低減は確認できない。
- 現設定は `local_leakage_canceller` としては保留、`BL_sidelobe_reducer` としては不採用とする。

---

## 6. 採否判断

現時点の判断は次である。

```text
role = source_preserving_scan:
  narrowband scan SLC は、target と同一周波数 interferer を別方位 source として残せている。
  interferer 自体の低減は要求しないため、この用途では不採用とはしない。

role = local_leakage_canceller:
  time-domain L=1 target-centric SLC は、streaming 成分別 RMS では 8192 Hz interferer 成分を低減している。
  ただし final fixed-weight BL の interferer marker は悪化しており、図示する local leakage 低減方式としては保留にする。

role = BL_sidelobe_reducer:
  time-domain L=1 target-centric SLC は、guard 外 peak と first sidelobe が悪化するため不採用。

fixed fractional beamformer with operational shading:
  10000 Hz 代表点では BL margin を満たしており、SLC を掛けない基準出力として採用。
```

SLC の採否は role で決める。別信号として観測したい interferer は消さない。target beam に混入した成分を下げたい場合だけ `local_leakage_canceller` として leakage 低減を要求する。BL 全体の sidelobe envelope を下げたい場合は、`BL_sidelobe_reducer` として guard 外 peak、first sidelobe、最大局所悪化量を必須にする。

---

## 7. 未検討・次に確認する項目

1. `source_preserving_scan` では既知 source の mainlobe を除外した false peak / envelope 指標を追加し、interferer を誤って sidelobe と数えない。
2. `max_reference_beams=48` の narrowband scan SLC では全 beam が `LIMITED_REFERENCE` になるため、snapshot 数、block 長、reference 数の組み合わせを再評価する。
3. target absent / training 区間を使い、desired target を含まない共分散推定で同一周波数条件を再評価する。
4. `local_leakage_canceller` の safety gate は target-only 保護、interferer-only leakage 低減、mixed 出力の過大変化で判定する。BL 悪化は `BL_sidelobe_reducer` role の不合格理由として分けて記録する。
5. streaming 成分別 RMS と final fixed-weight BL marker の符号が一致しない原因を、block-wise BL、exact 60 deg 応答、最終係数と時間平均係数の差で切り分ける。
6. `tap_len=3` の時間タップ付き SLC を、同じ 10000 Hz target / 8192 Hz interferer 条件で再評価する。
7. 低域 `f < 200 Hz` の全 CH 使用条件では、別途 BL/FRAZ/BTR を生成し、長開口時の SLC reference 選定を確認する。

## 8. 2026-07-05 ABF-like non-source suppression への再定義

目的を `local_leakage_canceller` から `ABF_like_non_source_suppression` へ切り替える。
既知 source 方位は target / interferer とも source mask とし、source mask guard 外を non-source sector と定義する。
SLC の exact marker reduction は補助指標に落とし、採否は non-source sector の包絡線抑圧で判定する。

必須指標は次である。

```text
source_peak_delta_db
source_azimuth_error_deg
non_source_global_peak_delta_db
non_source_p95_level_delta_db
non_source_integrated_level_delta_db
source_to_non_source_margin_delta_db
false_peak_count
angular_robustness_min_reduction_db over source/interferer +/-0.5 deg and +/-1.0 deg
WNG または weight_norm
condition_number
realtime_factor
```

比較対象は次とする。

```text
fixed_fractional_delay
beam_domain_slc_l1
time_domain_lcmv
time_domain_gsc
stft_capon
```

train/test 分離は次で固定する。

```text
train source azimuth:
  target     = 90.0 deg
  interferer = nominal interferer azimuth

test source azimuth:
  target     = 90.0 deg
  interferer = nominal interferer azimuth + offgrid_deg

source mask:
  test source 方位を中心に guard を取る。
  known source peak は sidelobe / false peak として数えない。
```

sweep 条件は次である。

```text
target_frequency_hz     = 6144 / 8192 / 10000
interferer_frequency_hz = 6144 / 8192 / 10000
interferer_azimuth_deg  = 30 / 45 / 60 / 75 / 105 / 120 / 150
offgrid_deg             = 0.0 / 0.25 / 0.5
```

判定規則は次とする。

```text
ABF_like_non_source_suppression pass:
  source_peak_delta_db が許容内で、source_azimuth_error_deg が小さい。
  non_source_global_peak_delta_db、non_source_p95_level_delta_db、
  non_source_integrated_level_delta_db がすべて負、または許容範囲内。
  source_to_non_source_margin_delta_db が非負。
  false_peak_count が増えない。
  angular_robustness_min_reduction_db が一点 null だけの改善になっていない。
  condition_number / weight_norm / realtime_factor が運用可能範囲。

fail:
  known source を落とす、non-source sector global / p95 / integrated が悪化する、
  false peak を増やす、または off-grid で抑圧が崩れる。

hold:
  一部条件だけ改善するが、train/test off-grid、周波数組み合わせ、runtime、
  condition_number のいずれかで追加確認が必要。
```

この再定義により、従来の `raw_interferer_reduction_db` や exact 60 deg marker reduction は採否主指標ではなくなる。
代表条件で exact 60.0 deg が大きく下がっても、60.44 deg grid や non-source sector 包絡線が悪化する方式は pass にしない。

