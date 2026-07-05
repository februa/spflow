# 運用スパースアレイへの SLC 導入検討

## 1. 目的

本書は、305 ch / 300.6 m の運用スパースアレイ、保存済み小数遅延 FIR バンク、151 本待受ビームを前提に、固定整相後段へ SLC を入れる条件を整理する。

SLC の評価では、target mainlobe が維持されることだけでは採用判定にしない。Beamforming Evaluation の観点に従い、target-only 保護、同一周波数干渉、異周波数干渉、BL/FRAZ/BTR の整合、runtime、condition number、weight norm を分けて確認する。

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

この方式は BL/FRAZ/BTR の before / after を作る診断に向いている。一方、全 beam scan では interferer 方位の beam もその beam 自身にとっては desired mainlobe であるため、干渉源ピークそのものを消す評価には使わない。

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
- 同一周波数 interferer に対して、実質的な低減は出ていない。
- 全 beam が `LIMITED_REFERENCE` であり、`max_reference_beams=48` の制限下での評価である。
- 実装は設計通り、現行 shading を固定整相出力と response matrix の両方へ適用している。したがって、この結果は矩形重みの取りこぼしではなく、同一周波数・高相関条件での方式限界として扱う。

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

成分別評価:

```text
target_power_delta_db        = -0.0027 dB re before level
interferer_reduction_db      = +23.0457 dB re before level
mixed_power_delta_db         = -0.0041 dB re before level
condition_number             = 2181.72
weight_norm                  = 1.7136
realtime_factor              = 0.259
safety_fallback_required     = false
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

判定:

- target beam 上の interferer 成分だけを見ると 23.0 dB 低下している。
- しかし SLC 後の protected-target BL は guard 外 peak と first sidelobe が悪化している。
- このため、現設定を「BL を改善する SLC」として採用しない。ユーザ提示時の representative BL は、上記 overlay を使って、成分別低減と BL 悪化を分けて説明する。
- `recommended_output` は現段階では固定整相とする。

---

## 6. 採否判断

現時点の判断は次である。

```text
narrowband scan SLC, same-frequency interferer:
  target mainlobe は維持するが、干渉低減は出ないため不採用。

time-domain L=1 target-centric SLC, different-frequency interferer:
  成分別 leakage は下がるが、BL が悪化するため現設定は不採用。

fixed fractional beamformer with operational shading:
  10000 Hz 代表点では BL margin を満たしており、現段階の出力として採用。
```

SLC は効果が全くないとは言わない。異周波数 interferer の target beam leakage には低減が出ている。一方、BL 形状の悪化を許容してよい方式ではないため、運用出力へ採用するには追加制約が必要である。

---

## 7. 未検討・次に確認する項目

1. `eta` と `loading` を BL 悪化量で制約し、成分別 leakage 低減と guard 外 peak 悪化の両方を満たす領域を探す。
2. `max_reference_beams=48` の narrowband scan SLC では全 beam が `LIMITED_REFERENCE` になるため、snapshot 数、block 長、reference 数の組み合わせを再評価する。
3. target absent / training 区間を使い、desired target を含まない共分散推定で同一周波数条件を再評価する。
4. target-centric BL の悪化を safety gate に入れ、target beam 出力成分だけでなく空間応答の悪化でも固定整相へ戻す。
5. `tap_len=3` の時間タップ付き SLC を、同じ 10000 Hz target / 8192 Hz interferer 条件で再評価する。
6. 低域 `f < 200 Hz` の全 CH 使用条件では、別途 BL/FRAZ/BTR を生成し、長開口時の SLC reference 選定を確認する。