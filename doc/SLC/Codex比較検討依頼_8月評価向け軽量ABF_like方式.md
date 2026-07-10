# Codex比較検討依頼: 8月評価向け軽量 ABF-like 非信号方位抑圧方式

## 0. この文書の目的

8月までに評価へ入れることを優先し、リアルタイム経路へ重い ABF、STFT、filterbank、新規 channel-domain GSC を入れない前提で、次の 2 系統だけを比較・検討する。

1. **固定整相後に SLC を導入する方式**
2. **固定整相の小数遅延 FIR / channel 重みを、信号入力または評価結果に基づいて適応的に作成・選択する方式**

最終目的は、単なる干渉一点 null ではなく、**ABF を最初から適用した場合に近い「信号が存在しない方位をなるべく低くする」こと**である。したがって、評価 role は次を主役にする。

```text
primary_role = ABF_like_non_source_suppression
```

従来の `local_leakage_canceller` や `BL_sidelobe_reducer` の指標は参考に残すが、採否主指標は `non_source sector` の抑圧と source 保護に置く。

---

## 1. 背景と制約

### 1.1 既存前段

既存前段は次で固定する。

```text
array:
  artifacts/beamforming/operational_sparse_array/operational_sparse_array_fs32768.json

shading:
  artifacts/beamforming/operational_shading/operational_kaiser_bessel_shading_fs32768.json

fractional delay FIR bank:
  artifacts/beamforming/fractional_delay_filter_bank_65x63.npz

beam count:
  151

fs:
  32768 Hz
```

運用アレイは 305 ch / 300.6 m で、200 Hz 以上は周波数ごとの active channel と Kaiser-Bessel shading を使う。処理側は CH 数、active channel、shading をコード直書きせず、JSON から読み込む。

### 1.2 現状の固定整相 baseline

現行の小数遅延固定整相 + 周波数別 active aperture + shading は、代表評価点で required peak margin 13 dB を満たしている。したがって、**固定整相単体を安全な fallback baseline** とする。

```text
fallback_output = fixed_fractional_delay_beamformer
```

SLC または入力適応小数遅延 FIR が採用基準を満たさない場合は、必ず固定整相へ戻す。

### 1.3 実装制約

8月評価に間に合わせるため、次は今回スコープ外とする。

```text
NG:
  - 固定整相前の channel-domain ABF を本番経路へ入れる
  - STFT / filterbank を本番リアルタイム経路へ入れる
  - 全周波数 bin ごとの covariance / weight solve をリアルタイムで回す
  - MUSIC / EVD / LCMV sector 探索を毎フレーム実行する
  - 既存 BL / FRAZ / BTR 評価系を壊す大改修
```

許容する処理は次とする。

```text
OK:
  - 固定整相後 beam_output[beam, sample] への軽量 SLC
  - L=1、条件付きで L=3 までの時間領域 SLC
  - 低頻度または評価前の小数遅延 FIR / shading / active set の選択・更新
  - リアルタイム側では保存済み係数の適用だけにする
  - safety gate による固定整相 fallback
```

---

## 2. 目的の再定義

### 2.1 従来 role と今回 role の違い

これまでの SLC 評価では、次の role を分けていた。

```text
source_preserving_scan:
  既知 source を別方位 source として残す。

local_leakage_canceller:
  target beam に混入する特定方位・特定周波数の漏れを下げる。

BL_sidelobe_reducer:
  固定整相後 BL の guard 外 peak / 第一副極 / 最大悪化量を下げる。
```

今回の主目的はこれらより実運用表示寄りで、次とする。

```text
ABF_like_non_source_suppression:
  既知 source 方位の peak は維持する。
  source mask 外、すなわち信号が存在しないと判断した方位の出力を低くする。
  false peak を増やさない。
  non-source sector の global peak / percentile / integrated level を悪化させない。
```

### 2.2 重要な評価上の注意

`exact marker reduction` は採否主指標にしない。

例: 60.0 deg exact では深い null が出ても、60.44 deg grid、第一副極、guard 外 global peak が悪化するなら、ABF-like non-source suppression としては不合格である。

```text
marker 一点 null:       参考指標
source mask 外 envelope: 採否主指標
```

---

## 3. 共通用語と評価 mask

### 3.1 source mask

評価時には、target と interferer をどちらも source として扱う。source を消すのではなく、source 以外を落とす。

```text
source_list:
  - target source
  - interferer source
  - 評価上既知の追加 source
```

source mask は、各 source 方位を中心に guard 幅を取る。

```text
source_mask(theta):
  true if theta is inside any source guard sector

non_source_sector:
  all azimuth grid points outside source_mask
```

### 3.2 oracle mask と detected mask

2 種類の mask で評価する。

```text
oracle_source_mask:
  simulation の正解 source 方位から作る。
  方式の上限性能を見る。

detected_source_mask:
  固定整相 before の BL / FRAZ / BTR から source peak を検出して作る。
  実運用に近い性能を見る。
```

採用には、oracle mask だけでなく detected mask でも大きく破綻しないことを要求する。

### 3.3 non-source 指標

non-source sector では次を測る。

```text
non_source_global_peak_db
non_source_p95_level_db
non_source_p99_level_db
non_source_integrated_level_db
source_to_non_source_margin_db
false_peak_count
max_local_worsening_db_gated
```

`max_local_worsening_db_gated` は、before が極端な谷だった点の差分で過大評価しないよう、次の gate を入れる。

```text
max_local_worsening_db_gated:
  count only points where
    before_level_db > source_peak_db - 60 dB
    or after_level_db > source_peak_db - 40 dB
```

---

## 4. 候補方式 A: 固定整相後 SLC 系

### A0. 固定整相 baseline

比較基準であり、常に出力可能な fallback とする。

```text
method_id = fixed_fractional_delay_baseline
```

採否:

```text
- required_peak_margin_db >= 13 dB を満たすことを確認する。
- SLC / adaptive FIR が fail の場合はこの出力へ戻す。
```

### A1. 既存 target-centric 時間領域 SLC

既存方式。

```text
method_id = target_centric_time_domain_slc

d[n]:
  protected target beam output

u[n]:
  guard 外 reference beam outputs

y[n] = d[n] - eta * w^H u[n]
```

この方式は、target beam に混入する特定 interferer 成分を下げる用途には有効な条件がある。ただし、過去評価では final fixed-weight BL 上で、interferer marker、第一副極、guard 外 peak が悪化する条件があった。

今回の扱い:

```text
primary candidate にはしない。
local_leakage_canceller として参考評価に残す。
ABF_like_non_source_suppression の採否には使わない。
```

Codex 作業:

```text
- 既存結果を再利用し、ABF_like_non_source_suppression の summary へ含める。
- exact marker reduction は補助指標に落とす。
- non_source_global_peak_delta_db、p95、ISL、false_peak_count を追加する。
```

### A2. source-mask non-source leakage subtractor SLC

今回の SLC 系の本命候補。

#### 考え方

従来の target-centric SLC は「target beam を守って、target beam に入る interferer を消す」方式だった。今回の目的は「source は残し、source ではない方位を落とす」ことである。

そのため、source mask 内の beam を reference とし、non-source beam に漏れた source-correlated 成分を引く。

```text
source beam outputs:
  x_S[n] = beam_output[source_reference_beams, n]

non-source beam b:
  d_b[n] = beam_output[b, n]

estimate leakage:
  c_b[n] = h_b^H x_S[n]

output:
  y_b[n] = d_b[n] - eta * c_b[n]
```

source mask 内の beam は原則として変更しない。

```text
if b in source_mask:
  y_b[n] = beam_output[b, n]
else:
  y_b[n] = beam_output[b, n] - eta * h_b^H x_S[n]
```

これは「source-preserving de-leakage SLC」として扱う。

#### 期待効果

```text
- 実 source の sidelobe leakage が non-source sector へ出ている場合、それを source reference から推定して下げられる可能性がある。
- source beam 自体を処理しないため、target / interferer の main peak を壊しにくい。
- source 数が少ない場合、reference 自由度が小さく、通常の guard 外全 beam SLC より軽い。
```

#### 限界

```text
- 実在する未検出 source も non-source と見なすと、その成分を削る可能性がある。
- source-correlated leakage 以外の無相関雑音は下がらない。
- deterministic な固定整相の理論 BL response 自体を改善する方式ではなく、実データ中の source leakage を下げる方式である。
```

#### 実装方針

```text
input:
  beam_output[beam, sample]
  source_mask[beam]
  source_reference_beams
  non_source_beams

for each block:
  Xs = beam_output[source_reference_beams, block]
  for each non-source beam b:
    db = beam_output[b, block]
    Rss = Xs Xs^H / K
    rsd = Xs db^* / K
    h_b = (Rss + loading * mean(diag(Rss)) I)^-1 rsd
    y_b = db - eta * h_b^H Xs

source beams:
  copy through
```

計算量を抑えるため、source_reference_beams は source ごとに中心 beam + ±少数 beam に限定する。

```text
source_reference_beams:
  each source center beam
  + optional adjacent beams within source_ref_guard

recommended initial:
  source_ref_guard = 1 or 2 beams
  eta = 0.25 / 0.5 / 0.75 sweep
  loading = 1e-2 / 3e-2 / 1e-1 relative loading sweep
  tap_len = 1 first
  tap_len = 3 only if L=1 pass but insufficient
```

#### 採否

この方式は `ABF_like_non_source_suppression` として採否する。

```text
pass candidate if:
  source peaks are preserved
  non-source global / p95 / integrated levels improve
  false peaks do not increase
  runtime factor <= 1
  no excessive local worsening
```

A1 のような exact interferer marker 低下は主指標にしない。

### A3. source-preserving scan SLC

既存の全 beam scan SLC は、各 beam をそれぞれ desired として保護するため、interferer beam 自体も source として残る。これは source-preserving display としては自然だが、non-source suppression には弱い。

今回の扱い:

```text
- A2 の比較対象として残す。
- 採否は source-preserving scan としてのみ行う。
- ABF_like_non_source_suppression の本命にはしない。
```

---

## 5. 候補方式 B: 入力適応 小数遅延 FIR / channel 重み系

### B0. 現行 fixed fractional-delay + shading baseline

現行 baseline は採用済みの安全出力である。

```text
method_id = fixed_fractional_delay_current_shading
```

この baseline は、SLC を使わずに固定整相側だけでどこまで sidelobe envelope を下げられているかを見る基準である。

### B1. 入力・評価結果に基づく係数セット選択

新しい STFT / filterbank を入れず、既存の小数遅延 FIR と channel shading の候補を複数用意し、評価区間の BL/FRAZ/BTR 指標で最もよい係数セットを選ぶ。

```text
method_id = input_informed_fd_shading_selection
```

候補は次に限定する。

```text
- 既存 fractional delay FIR bank
- 既存または軽微に追加する Kaiser beta 候補
- 周波数別 active aperture 候補
- sum(weights) 正規化は維持
```

リアルタイム経路では選択済み係数を掛けるだけにする。

```text
learning/evaluation side:
  choose coefficient profile

real-time side:
  apply selected fixed coefficients
```

採否上は、SLC よりも固定整相 baseline 改善として扱う。

### B2. residual delay / phase calibration 型

信号入力から各 channel の小さな遅延誤差または位相誤差を推定し、既存 fractional delay FIR の選択位置または小さな fractional offset を補正する。

```text
method_id = input_adaptive_residual_delay_calibration
```

目的は、アレイ座標、音速、FIR 近似、実機応答のずれを補正し、source peak の coherence と sidelobe の乱れを改善すること。

#### 制約

この方式は ABF ではなく、固定整相の校正である。したがって制約を強く置く。

```text
- per-channel delay correction は小さい範囲に制限する
- delay correction は block 間で急変させない
- source peak 方位を動かしすぎない
- target-only / source-only で source power を落とさない
- coefficient symmetry / FIR health を確認する
```

推奨初期制約:

```text
abs(delta_delay_samples) <= 0.25 sample
median_filter_or_smooth_over_channels = true
update_period_sec >= 1.0
freeze_when_source_mask_unstable = true
```

#### 限界

```text
- サイドローブがアレイ設計そのものから来ている場合、校正だけでは下がらない。
- 入力信号が複数 source / 低 SNR の場合、誤った delay correction を学習するリスクがある。
- source 方位を既知または安定推定できない区間では更新しない。
```

### B3. constrained adaptive fractional-delay / shading synthesis

入力データから source mask と non-source sector を作り、source を維持しながら non-source response を下げるように小数遅延 FIR / channel 重みを再設計する。

```text
method_id = constrained_adaptive_fd_shading_synthesis
```

ただし、これは自由に複素 channel 重みを作ると ABF そのものになるため、今回のスコープでは強く制限する。

```text
allowed:
  - real-valued channel shading
  - fractional delay FIR prototype の小さな perturbation
  - linear-phase / group-delay constraint
  - source direction distortionless constraint
  - coefficient smoothness constraint

not allowed:
  - arbitrary complex channel weight per source / per frequency
  - full covariance MVDR equivalent solve in real-time
  - source 方位ごとに全 channel 複素重みを毎回更新する方式
```

B3 は開発リスクが高いため、8月評価では **hold candidate** とする。B1/B2 で不十分かつ実装時間が残る場合だけ、offline 設計として試す。

---

## 6. 共通採否基準

### 6.1 primary role: ABF_like_non_source_suppression

採用には、代表条件だけでなく off-grid 条件を含む評価で次を満たすこと。

#### source 保護

```text
source_peak_delta_db:
  pass: abs(delta) <= 0.5 dB
  hold: abs(delta) <= 1.0 dB
  fail: abs(delta) > 1.0 dB

source_azimuth_error_deg:
  pass: <= max(0.5 deg, 0.5 * local_beam_spacing_deg)
  fail: > 1 beam spacing or peak swaps to non-source sector

target_only_power_delta_db:
  pass: >= -0.5 dB
  hold: >= -1.0 dB
  fail: < -1.0 dB
```

#### non-source 抑圧

```text
non_source_global_peak_delta_db:
  pass: <= -1.0 dB
  hold: -1.0 dB < delta <= +0.5 dB and p95/ISL improve
  fail: > +0.5 dB

non_source_p95_level_delta_db:
  pass: <= -1.0 dB
  hold: <= 0.0 dB
  fail: > 0.0 dB

non_source_integrated_level_delta_db:
  pass: <= -1.0 dB
  hold: <= 0.0 dB
  fail: > 0.0 dB

source_to_non_source_margin_delta_db:
  pass: >= +0.5 dB
  hold: >= 0.0 dB
  fail: < 0.0 dB

false_peak_count_delta:
  pass: <= 0
  fail: > 0

max_local_worsening_db_gated:
  pass: <= +3.0 dB
  hold: <= +6.0 dB and after level remains below source_peak - 25 dB
  fail: > +6.0 dB or creates visible false peak
```

#### robust / off-grid

```text
angular_robustness:
  pass if offgrid 0.0 / 0.25 / 0.5 deg で同じ判定
  hold if 0.5 deg だけ hold
  fail if 0.25 deg で fail
```

一点 null は pass としない。

```text
fail if:
  exact source/interferer marker improves
  but surrounding grid or non-source envelope worsens
```

#### runtime / safety

```text
realtime_factor:
  pass: <= 0.7
  hold: <= 1.0
  fail: > 1.0

nan_inf_count:
  pass: 0
  fail: > 0

condition_number:
  pass: <= 1e6
  hold: <= 1e8 with stable output
  fail: > 1e8 or unstable block-to-block variation

weight_norm:
  pass: not larger than 2x baseline median or configured limit
  fail: unstable, spikes, or causes output power anomaly
```

### 6.2 SLC 系の追加基準

```text
raw_candidate と effective_output を分けて記録する。
```

```text
raw_candidate:
  SLC が実際に何をしたかを見る。

effective_output:
  safety gate / fallback 後に運用へ出る出力。
```

SLC raw が fail した場合でも、effective output が fixed fallback なら運用安全上は可。ただし、方式採用としては改善なし扱いにする。

### 6.3 小数遅延 FIR / shading 系の追加基準

```text
passband_ripple_db:
  pass: <= 0.2 dB in evaluation band
  hold: <= 0.5 dB
  fail: > 0.5 dB

group_delay_error_samples:
  pass: <= 0.05 sample RMS
  hold: <= 0.10 sample RMS
  fail: > 0.10 sample RMS

snr_loss_vs_current_baseline_db:
  pass: <= 0.5 dB
  hold: <= 1.0 dB
  fail: > 1.0 dB

coefficient_update_stability:
  fail if coefficients jump between adjacent updates without source mask change
```

---

## 7. 評価条件

### 7.1 tiered evaluation

8月までの作業量を抑えるため、評価を tier に分ける。

#### Tier 0: smoke / representative

```text
target:
  azimuth = 90 deg
  frequency = 10000 Hz
  level = 0 dB re input RMS

interferer:
  azimuth = 60 deg
  frequency = 8192 Hz
  level = -6 dB re input RMS

offgrid_deg:
  0.0 / 0.5

duration:
  5.0 s
```

#### Tier 1: high-priority matrix

```text
target_frequency_hz:
  6144 / 8192 / 10000

interferer_frequency_hz:
  6144 / 8192 / 10000

interferer_azimuth_deg:
  45 / 60 / 75 / 105 / 120 / 150

offgrid_deg:
  0.0 / 0.25 / 0.5
```

同一周波数条件は SLC で消すことを主目的にしないが、source-preserving と non-source mask の安全確認として必ず含める。

#### Tier 2: robustness / weak source

```text
source level combinations:
  target 0 dB, interferer -6 / -12 / -20 dB

weak additional source:
  non-source sector に追加し、detected mask で検出できる場合は保存されるか確認する。

noise:
  -60 dB re input RMS and practical noise cases if available
```

### 7.2 評価単位

各 method で以下を出す。

```text
BL:
  before fixed baseline
  after candidate raw
  after candidate effective

FRAZ/BTR:
  before / after effective

component RMS:
  target-only
  interferer-only
  mixed

runtime:
  elapsed_sec
  input_duration_sec
  realtime_factor
```

---

## 8. summary JSON schema

Codex は各評価 run で次の summary を出す。

```json
{
  "method_id": "source_mask_non_source_slc",
  "primary_role": "ABF_like_non_source_suppression",
  "status": "pass|hold|fail",
  "failure_reasons": [],
  "input_config": {
    "fs_hz": 32768,
    "beam_count": 151,
    "target_frequency_hz": 10000,
    "target_azimuth_deg": 90,
    "interferer_frequency_hz": 8192,
    "interferer_azimuth_deg": 60,
    "offgrid_deg": 0.5
  },
  "source_mask": {
    "mask_type": "oracle|detected",
    "source_count": 2,
    "guard_deg_or_beams": null
  },
  "source_preservation": {
    "max_abs_source_peak_delta_db": null,
    "max_source_azimuth_error_deg": null,
    "target_only_power_delta_db": null
  },
  "non_source_suppression": {
    "non_source_global_peak_delta_db": null,
    "non_source_p95_level_delta_db": null,
    "non_source_p99_level_delta_db": null,
    "non_source_integrated_level_delta_db": null,
    "source_to_non_source_margin_delta_db": null,
    "false_peak_count_before": null,
    "false_peak_count_after": null,
    "false_peak_count_delta": null,
    "max_local_worsening_db_gated": null,
    "max_local_worsening_azimuth_deg": null
  },
  "slc_health": {
    "tap_len": null,
    "eta": null,
    "loading": null,
    "n_ref": null,
    "condition_number": null,
    "weight_norm": null,
    "safety_fallback_required": null
  },
  "fd_filter_health": {
    "passband_ripple_db": null,
    "group_delay_error_samples_rms": null,
    "snr_loss_vs_current_baseline_db": null,
    "max_abs_delta_delay_samples": null
  },
  "runtime": {
    "elapsed_sec": null,
    "input_duration_sec": null,
    "realtime_factor": null
  },
  "artifacts": {
    "bl_before_after_png": null,
    "fraz_before_after_png": null,
    "btr_before_after_png": null,
    "summary_csv": null
  }
}
```

---

## 9. Codex 実装タスク

### Task 1: 評価 mask と ABF-like metrics の追加

追加対象:

```text
src/spflow/beamforming/evaluation_criteria.py
既存 diagnostics summary generator
```

実装内容:

```text
- source_mask / non_source_sector を作る関数
- source peak preservation 指標
- non_source_global_peak / p95 / p99 / ISL
- source_to_non_source_margin
- false_peak_count
- gated max local worsening
- oracle mask / detected mask の両対応
```

完了条件:

```text
- fixed baseline だけで summary が出る
- 既知 source mainlobe を sidelobe / false peak と誤計上しない
- dB re input RMS / dB re before level の基準を JSON に明記する
```

### Task 2: A2 source-mask non-source leakage subtractor SLC

実装候補ファイル:

```text
src/spflow/beamforming/source_mask_slc.py
examples/beamforming/operational_array_source_mask_slc_diagnostics.py
tests/beamforming/test_source_mask_slc.py
```

初期実装:

```text
- tap_len = 1
- source beam は copy-through
- non-source beam のみ source reference から leakage subtract
- relative diagonal loading
- eta sweep
- runtime summary
```

必須テスト:

```text
- source beams are unchanged when copy-through is enabled
- no NaN / inf
- eta=0 で fixed baseline と完全一致
- source reference が空なら DISABLED / fixed fallback
- shape and dB reference are recorded
```

### Task 3: B1/B2 adaptive FD / shading candidate evaluation

実装候補:

```text
examples/beamforming/evaluate_adaptive_fractional_delay_candidates.py
src/spflow/beamforming/adaptive_fractional_delay_selection.py
```

初期対象:

```text
B1:
  beta 候補と既存 FD bank の選択評価

B2:
  residual delay correction の offline/simulated prototype
```

必須制約:

```text
- sum(weights) normalization を維持
- filter / group delay health を summary へ出す
- source_peak_delta と non_source metrics を同じ JSON で出す
- update 後の coefficients をいきなり運用採用せず candidate として保存する
```

### Task 4: 比較表の生成

全方式の summary を集約し、次の CSV / Markdown を出す。

```text
artifacts/beamforming/lightweight_abf_like_comparison/comparison_summary.csv
artifacts/beamforming/lightweight_abf_like_comparison/comparison_report.md
```

比較列:

```text
method_id
status
max_abs_source_peak_delta_db
non_source_global_peak_delta_db
non_source_p95_level_delta_db
non_source_integrated_level_delta_db
source_to_non_source_margin_delta_db
false_peak_count_delta
max_local_worsening_db_gated
realtime_factor
condition_number
snr_loss_vs_current_baseline_db
failure_reasons
```

---

## 10. 最終採用フロー

### 10.1 判定順

```text
1. fixed baseline を必ず生成する。
2. A2 source-mask SLC を Tier 0 で評価する。
3. A2 が pass/hold なら Tier 1 へ広げる。
4. A2 が fail なら A1 target-centric SLC の局所効果は参考に落とし、採用候補から外す。
5. B1/B2 を固定整相 baseline 改善として評価する。
6. A2 と B1/B2 を比較し、8月評価へ入れる方式を 1 つ、または fallback 付きで 2 つ選ぶ。
```

### 10.2 採用優先順位

```text
Priority 1:
  B1/B2 が固定整相 baseline を安全に改善する場合
  -> 出力波形への副作用が小さいため最優先。

Priority 2:
  A2 が source 保護と non-source 抑圧を両立する場合
  -> fixed baseline + source-mask SLC を fallback 付きで採用。

Priority 3:
  A1 target-centric SLC only
  -> local leakage 用途に限定。ABF-like 目的では採用しない。

Fail-safe:
  どの candidate も fail の場合は fixed_fractional_delay_baseline を採用し、評価資料には「軽量後段処理では非信号方位抑圧の改善なし」と明記する。
```

### 10.3 8月評価へ入れる条件

8月評価へ入れる方式は、最低限 Tier 0 と Tier 1 の主要条件で次を満たすこと。

```text
required:
  - source_peak_delta_db pass または hold
  - non_source_global_peak_delta_db が fail でない
  - non_source_p95_level_delta_db が fail でない
  - false_peak_count_delta <= 0
  - realtime_factor <= 1.0
  - NaN / inf なし
  - fixed baseline fallback が実装済み
```

正式採用には pass が必要。8月評価へ「比較候補」として入れるだけなら hold を許容する。ただし hold の理由を comparison_report.md に明記する。

---

## 11. Codex への最初の具体依頼文

以下をそのまま Codex に渡す。

```text
目的を ABF_like_non_source_suppression に固定し、固定整相後 SLC と入力適応小数遅延 FIR / shading の 2 系統だけを比較してください。

まず、source_mask / non_source_sector 評価を実装し、fixed baseline に対して以下の指標を出してください。

- source_peak_delta_db
- source_azimuth_error_deg
- non_source_global_peak_delta_db
- non_source_p95_level_delta_db
- non_source_p99_level_delta_db
- non_source_integrated_level_delta_db
- source_to_non_source_margin_delta_db
- false_peak_count_delta
- max_local_worsening_db_gated
- runtime_factor

次に、source-mask non-source leakage subtractor SLC を実装してください。
source mask 内の beam は copy-through とし、non-source beam のみ source beam reference から source-correlated leakage を差し引く構成にしてください。
eta と loading を sweep し、採否は exact marker reduction ではなく non_source sector 指標で判定してください。

最後に、固定整相の小数遅延 FIR / shading について、入力または評価結果に基づく候補選択・残差遅延補正として実装可能な最小案を評価してください。
任意複素 channel weight を毎回解く方式にはしないでください。これは ABF と同等になり、今回の処理量制約から外れます。

すべての方式は fixed_fractional_delay_baseline への fallback を持たせ、raw candidate と effective output を分けて summary に記録してください。
```

---

## 12. 期待される結論パターン

### パターン 1: B1/B2 が改善、A2 は hold/fail

固定整相係数側の改善を採用し、SLC は導入しない。最も安全。

### パターン 2: A2 が改善、B1/B2 は改善小

fixed baseline + source-mask SLC を fallback 付きで採用候補にする。source mask 誤りと弱 source 抑圧のリスクを明記する。

### パターン 3: どちらも改善なし

固定整相 baseline のまま評価へ入る。資料には、処理量制約下では ABF-like 非信号方位抑圧は不足し、次段は STFT / 学習側 ABF が必要と記録する。

### パターン 4: A1 target-centric SLC だけ局所改善

local leakage 用途に限定し、ABF-like 非信号方位抑圧方式としては採用しない。

---

## 13. 注意事項

1. **source を消さない。** 今回の目的は non-source を落とすことであり、target / interferer として観測したい source peak は残す。
2. **一点 null で合格にしない。** exact marker が深く落ちても、grid / off-grid / 第一副極 / p95 / ISL が悪化すれば fail。
3. **SLC の BL source-response と実データ non-source 抑圧を混同しない。** A2 は実 source leakage を下げる方式であり、理論 BL envelope 改善とは別指標で見る。
4. **小数遅延 FIR の適応を ABF 化しない。** 任意複素 channel 重みを信号ごとに解く方式は今回スコープ外。
5. **fallback を必須にする。** fail/unstable/unknown condition では固定整相を出す。
