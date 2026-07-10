# 軽量 ABF-like 非信号方位抑圧方式 検討設計書

## 1. 目的とスコープ

本設計書は、8月評価へ向けた軽量 ABF-like 方式の検討結果を、実装・評価・採否判断が追跡できる形で蓄積するための単一文書である。

主目的は、固定整相後または固定整相係数側の軽量処理によって、既知 source 方位を維持しながら non-source sector の出力包絡線を下げることである。評価 role は `ABF_like_non_source_suppression` とする。

今回のスコープに含める方式は次の 2 系統に限定する。

- A 系統: 固定整相後 `beam_output[beam, sample]` に対する source-mask SLC
- B 系統: 小数遅延 FIR / shading / active aperture の候補選択または残差遅延補正

今回のリアルタイム経路には、STFT、filterbank、channel-domain GSC、全周波数 bin ごとの covariance solve、任意複素 channel 重みの逐次更新は入れない。どの候補方式も、固定整相出力を安全な fallback baseline として保持する。

## 2. Beamforming Evaluation に基づく評価原則

Beamforming Evaluation の基準に従い、BL / FRAZ / BTR の peak 位置、source 可視性、mainlobe 保護、SLC covariance health、waveform integrity、runtime budget を分けて確認する。レベル値は単独の dB として扱わず、絶対値では `dB re input RMS` または校正済みなら `dB re 1 uPa RMS`、before/after 差分では level difference であることを summary に残す。

`ABF_like_non_source_suppression` では、一点 marker の null 深さを採否主指標にしない。source mask 外の global peak、p95、p99、integrated level、source-to-non-source margin、false peak count、gated local worsening を採否中心に置く。

採否時に必ず分ける出力は次の通りである。

- `before`: 固定整相 baseline
- `raw`: 候補方式が計算した出力
- `effective`: safety gate / fallback 適用後に運用へ出せる出力

SLC raw が改善しても source peak を壊す、non-source false peak を増やす、NaN / inf を含む、または covariance が悪条件になる場合は、effective output を固定整相へ戻す。

## 3. 共通 source mask と ABF-like metrics

### 3.1 実装対象

`src/spflow/beamforming/abf_like_metrics.py` に、source mask と non-source sector 評価を追加した。

主な公開 API は次の通りである。

- `SourceSectorMask`
- `build_source_sector_mask`
- `build_source_sector_mask_from_azimuths`
- `detect_source_beam_indices_from_level_peaks`
- `calculate_abf_like_non_source_metrics`
- `judge_abf_like_non_source_metrics`

### 3.2 mask 設計

`SourceSectorMask.source_mask[beam]` は既知または検出 source の mainlobe 保護領域を表す。`non_source_mask[beam] = ~source_mask[beam]` とし、source 自体を sidelobe や false peak として数えない。

source が beam 端にある場合、guard 領域は `[0, n_beam)` へ clip する。存在しない beam を評価対象へ入れないためである。角度指定 mask では、`guard_deg=0` かつ source 方位が grid 点と一致しない場合でも、最近傍 beam を必ず source として保護する。

### 3.3 評価指標

`calculate_abf_like_non_source_metrics` は、同じ source mask を before / after に適用し、次を返す。

- `max_abs_source_peak_delta_db`
- `max_source_azimuth_error_deg`
- `non_source_global_peak_delta_db`
- `non_source_p95_level_delta_db`
- `non_source_p99_level_delta_db`
- `non_source_integrated_level_delta_db`
- `source_to_non_source_margin_delta_db`
- `false_peak_count_delta`
- `max_local_worsening_db_gated`

integrated level は、RMS dB20 レベルを power 比へ戻して積分し、dB10 へ戻す。before/after で同じ beam 集合を使うため、総和か平均かの選択は差分値には影響しない。

gated local worsening は、before が source peak から 60 dB より低く、after も source peak から 40 dB より低い点を除外する。深い谷だけが変化した条件を、表示上の false peak と同じ重みで扱わないためである。

## 4. A2 source-mask non-source leakage subtractor SLC

### 4.1 方式

`src/spflow/beamforming/source_mask_slc.py` に、固定整相後 beam-domain の source-mask SLC を追加した。

入力は `beam_output[beam, sample]` であり、axis=0 が scan beam、axis=1 が時間 sample である。source reference は source mask 内 beam に限定し、non-source beam を reference として誤用した場合は `ValueError` にする。

non-source beam `b` に対して、source reference 行列を `X_S`、対象 beam を `d_b` とすると、次を解く。

```text
R_ss = X_S X_S^H / K
r_sd = X_S d_b^* / K
h_b = (R_ss + lambda I)^-1 r_sd
c_b[n] = h_b^H X_S[n]
y_b[n] = d_b[n] - eta c_b[n]
```

`lambda` は平均対角 power に対する相対 diagonal loading である。reference が無音に近い場合でも loaded covariance が作れるよう、平均対角 power が非正または非 finite なら 1.0 を基準にする。

source mask 内 beam は通常設定で copy-through する。

```text
if b in source_mask:
  y_b[n] = beam_output[b, n]
else:
  y_b[n] = beam_output[b, n] - eta * h_b^H X_S[n]
```

### 4.2 安全診断

結果型 `SourceMaskSlcResult` は raw/effective/cancel/weights/reference beams/source mask/health を保持する。`SourceMaskSlcHealth` には、mode、eta、loading、tap_len、reference 数、non-source 数、condition number、weight norm、NaN/inf 数、fallback 要否、理由を残す。

次の条件では SLC を無効化し、effective output を固定整相へ戻す。

- 入力に NaN / inf がある
- non-source sector が空
- source reference が空、または snapshot 数に対して reference 自由度が足りない
- linear solve が失敗する
- raw output に NaN / inf が出る
- loaded covariance condition number が制限値を超える
- weight norm が設定上限を超える

condition number の既定制限は、依頼資料の fail 条件に合わせて `1e8` とした。悪条件のまま採用すると、block 間の重み急変や過大キャンセルが起きるためである。

### 4.3 A2 の設計上の限界

A2 は実データ中の source-correlated leakage を下げる方式であり、固定整相の理論 BL response 自体を再設計する方式ではない。未検出 source を non-source と見なすと、その source を削る可能性がある。無相関雑音や配列幾何由来の deterministic sidelobe は、source reference から説明できない限り下がらない。

そのため、A2 が fail した場合は、まず次を確認する。

- source mask が source mainlobe を十分に含んでいるか
- reference beam が source mask 内だけから選ばれているか
- `eta`、relative loading、tap_len、sample_per_dof が設計どおりか
- raw output と effective output を取り違えて評価していないか
- dB reference と source/non-source mask が before/after で一致しているか

## 5. B1/B2 固定整相係数側の候補

### 5.1 B1 係数セット選択

B1 は、既存 fractional delay FIR bank、Kaiser beta 候補、周波数別 active aperture 候補を評価側で比較し、保存済み係数をリアルタイム経路で適用する方式である。これは ABF ではなく、固定整相 baseline の係数選択として扱う。

採否には、ABF-like non-source metrics に加え、passband ripple、group delay error、SNR loss、active channel / shading の一貫性を記録する。`sum(weights)` 正規化は維持する。

### 5.2 B2 残差遅延補正

B2 は、アレイ座標、音速、FIR 近似、実機応答の小さなずれを補正する方式である。per-channel delay correction は小さい範囲に制限し、block 間で急変させない。source mask が不安定な区間では更新しない。

この方式も任意複素 channel weight を毎回解くものではない。source peak coherence と sidelobe の乱れを改善できるかを、固定整相係数側の calibration として評価する。

## 6. 実装と検証結果

### 6.1 追加・更新ファイル

- `src/spflow/beamforming/abf_like_metrics.py`
- `src/spflow/beamforming/source_mask_slc.py`
- `src/spflow/beamforming/__init__.py`
- `tests/beamforming/test_abf_like_metrics.py`
- `tests/beamforming/test_source_mask_slc.py`

`__init__.py` では、A2 と ABF-like metrics の公開 API を `spflow.beamforming` から import できるようにした。

### 6.2 テストで確認した条件

ABF-like metrics では、source guard が non-source sector から除外されること、non-source 包絡線改善を pass 判定できること、深い谷だけの変化を visible false peak として数えないこと、detected mask 用 peak 検出が同じ mainlobe を重複検出しないことを確認した。

source-mask SLC では、source beam が copy-through で維持されること、non-source beam の source-correlated leakage が下がること、`eta=0` で固定整相 baseline と一致すること、source reference が空なら fixed fallback になること、non-source beam を reference に入れられないことを確認した。

### 6.3 検証コマンド

```text
.venv\Scripts\python.exe -m pytest tests\beamforming\test_evaluation_criteria.py tests\beamforming\test_slc.py tests\beamforming\test_abf_like_metrics.py tests\beamforming\test_source_mask_slc.py
結果: 26 passed

.venv\Scripts\python.exe -m ruff check src\spflow\beamforming\abf_like_metrics.py src\spflow\beamforming\source_mask_slc.py src\spflow\beamforming\__init__.py tests\beamforming\test_abf_like_metrics.py tests\beamforming\test_source_mask_slc.py
結果: All checks passed

.venv\Scripts\python.exe -m pyright -p artifacts\beamforming\lightweight_abf_like_comparison\pyright_check\pyrightconfig.json --verbose
結果: Found 4 source files, 0 errors
```

Pyright の検査対象は、本件で追加した 2 つの実装ファイルと 2 つのテストファイルである。

## 7. 次に実施する評価

次段では、固定整相 baseline、A2 raw、A2 effective、B1/B2 candidate を同じ `ABF_like_non_source_suppression` summary へ並べる。出力先は次を標準とする。

- `artifacts/beamforming/lightweight_abf_like_comparison/comparison_summary.csv`
- `artifacts/beamforming/lightweight_abf_like_comparison/comparison_report.md`

Tier 0 では、target 90 deg / 10000 Hz、interferer 60 deg / 8192 Hz、offgrid 0.0 / 0.5 deg を確認する。Tier 1 では、6144 / 8192 / 10000 Hz、interferer 方位 45 / 60 / 75 / 105 / 120 / 150 deg、offgrid 0.0 / 0.25 / 0.5 deg へ広げる。

各 run では、oracle mask と detected mask を分け、source peak delta、source azimuth error、non-source global/p95/p99/integrated level、source-to-non-source margin、false peak count、gated local worsening、runtime、SLC health、FD filter health を同じ schema で保存する。

方式が期待どおりに改善しない場合は、採否判断へ進む前に、実装が本設計どおりかを確認する。特に、source mask、dB reference、raw/effective の取り違え、fallback 条件、loaded covariance、reference beam 選択を先に点検する。


## 8. Tier 0 / Tier 1 横並び評価結果

### 8.1 実行内容

`examples/beamforming/evaluate_lightweight_abf_like_comparison.py` を追加し、固定整相 baseline、A2 source-mask SLC raw/effective、B1 FD/shading candidate を同じ `ABF_like_non_source_suppression` 指標で比較した。

出力は次へ保存した。

- `artifacts/beamforming/lightweight_abf_like_comparison/comparison_summary.csv`
- `artifacts/beamforming/lightweight_abf_like_comparison/comparison_report.md`

評価は beam-domain 周波数応答から mixed 波形を合成して行った。これは、A2 が実際に入力として受け取る `beam_output[beam, sample]` に対する source-correlated leakage 抑圧を確認するためである。FRAZ/BTR 図はこの run では生成していないため、次段で代表 pass/hold/fail 条件に絞って図示する。

### 8.2 評価条件

Tier 0 は target 90 deg / 10000 Hz、interferer 60 deg / 8192 Hz、offgrid 0.0 / 0.25 / 0.5 deg とした。Tier 1 は target frequency 6144 / 8192 / 10000 Hz、interferer frequency 6144 / 8192 / 10000 Hz、interferer azimuth 45 / 60 / 75 / 105 / 120 / 150 deg、offgrid 0.0 / 0.25 / 0.5 deg とした。

mask は oracle / detected の両方を出した。source guard は狭め 1.0 deg、既定 2.0 deg、広め 3.0 deg の half width とし、CSV には `source_mask_width_deg` として左右合計幅を保存した。

A2 は eta 0.25 / 0.5 / 0.75 / 1.0、loading 1e-3 / 1e-2 / 3e-2 / 1e-1、same-block と one-block-delay の両方を評価した。判定は raw ではなく effective output で行い、raw 行は `diagnostic_raw` として保存した。

B1 は current operational shading、beta 0.0 / 2.0 / 4.0 / 6.0 を比較し、既存 operational active aperture 上で first sidelobe、p95、p99、ISL、N_eff、SNR loss を保存した。

### 8.3 結果要約

CSV は 69301 行で、依頼された必須列を含む。最終 report の方式別集計は次である。

| 方式 | 判定 | 集計 |
|---|---:|---|
| fixed_baseline | baseline | baseline=990 |
| A2_source_mask_slc | hold | fail=6, hold=265, pass=31409 |
| B1_fd_shading_selection | hold | fail=3697, hold=1253 |
| B2_residual_delay_correction | not_evaluated | not_evaluated=1 |

A2 は effective 判定で pass が多数ある一方、mask / offgrid / loading / eta 条件により hold/fail が残るため、方式全体は hold とした。same-block だけでなく one-block-delay でも pass が出ている点は有望だが、代表条件の FRAZ/BTR 図と未検出 source 条件を追加確認する必要がある。

B1 は beta 候補により non-source p95 が改善する条件があるが、source 保護や false peak / local worsening を含めた ABF-like 判定では fail が多い。固定整相係数側の改善候補としては、単純な beta sweep だけで採用するには不足である。

B2 は補正係数 candidate を今回生成していないため未評価とした。B2 を比較表へ入れるには、残差遅延推定条件、補正上限、更新 freeze 条件、FIR health の summary を先に定義する必要がある。

### 8.4 注意点と次の確認

今回の横並び評価は、SLC source-correlated leakage 抑圧を beam-domain で高速に比較するための評価である。運用資料へ載せる代表図としては、A2 の pass / hold / fail 条件から数例を選び、BL / FRAZ / BTR を追加生成する必要がある。

A2 が上手くいかない条件は、まず実装ではなく評価 mask と source reference の整合を確認する。特に detected mask が source を 1 本しか検出しない条件、source guard が広すぎて non-source sector が狭くなる条件、eta=1.0 で source-correlated 成分を過大に引く条件を分けて読む。

## 9. 8月評価向け候補絞り込み

### 9.1 絞り込み対象

`comparison_summary.csv` の effective 判定を受け、8月評価へ残す候補を次に限定した。

| 区分 | 候補 | 条件 |
|---|---|---|
| baseline | fixed_baseline | current operational shading |
| A2_safe | A2 source-mask SLC | eta=0.5、loading=0.03、source_guard=wide、train_mode=one_block_delay |
| A2_aggressive | A2 source-mask SLC | eta=1.0、loading=0.1、source_guard=wide、train_mode=one_block_delay |

既存 full sweep では、A2_safe と A2_aggressive はどちらも effective 判定で 330 条件すべて pass であった。方式判定は raw ではなく effective で行い、raw は診断用に分けて保存した。

### 9.2 生成した artifact

絞り込み評価は `examples/beamforming/evaluate_lightweight_abf_like_august_shortlist.py` で実行した。実行時は repo root から module として呼び出す。

```text
.venv\Scripts\python.exe -m examples.beamforming.evaluate_lightweight_abf_like_august_shortlist
```

出力は次へ保存した。

- `artifacts/beamforming/lightweight_abf_like_comparison/august_shortlist/shortlist_summary.csv`
- `artifacts/beamforming/lightweight_abf_like_comparison/august_shortlist/negative_summary.csv`
- `artifacts/beamforming/lightweight_abf_like_comparison/august_shortlist/safety_gate_summary.csv`
- `artifacts/beamforming/lightweight_abf_like_comparison/august_shortlist/runtime_summary.csv`
- `artifacts/beamforming/lightweight_abf_like_comparison/august_shortlist/representative_figure_manifest.csv`
- `artifacts/beamforming/lightweight_abf_like_comparison/august_shortlist/august_candidate_report.md`
- `artifacts/beamforming/lightweight_abf_like_comparison/august_shortlist/figures/`

代表図は 4 scenario × 3 method × BL/FRAZ/BTR の 36 PNG を保存した。BL は mixed RMS、FRAZ は評価 tone 周波数への複素投影、BTR は短時間 RMS の frame 最大基準相対 dB とした。

### 9.3 代表図 scenario

代表図は次の 4 条件で生成した。

| scenario | mask | 選定理由 |
|---|---|---|
| representative_pass_tier0_oracle | oracle | Tier0 の標準 pass 条件 |
| representative_hold_tier1_detected | detected | full sweep で弱い A2 条件が p95 未達となった hold 代表条件 |
| representative_detected_tier0 | detected | detected mask の代表条件 |
| representative_offgrid_0p5_tier0_oracle | oracle | offgrid 0.5 deg の代表条件 |

A2_safe / A2_aggressive は、上記 4 条件の effective 判定でいずれも pass であった。fixed_baseline は基準行として baseline とした。

### 9.4 負例評価

負例評価は次の 5 条件で行った。

| scenario | 条件 | 目的 |
|---|---|---|
| negative_unknown_source_outside_mask | 135 deg / 6144 Hz / -3 dB source を source mask 外に追加 | 未知 source を non-source と扱う危険の確認 |
| negative_weak_source_outside_mask | 135 deg / 6144 Hz / -18 dB source を source mask 外に追加 | 弱 source の検出漏れ時の挙動確認 |
| negative_three_or_more_known_sources | 3 本 source を oracle mask 内に含める | source 数増加時の source 保護確認 |
| negative_detected_mask_single_source | detected mask が target 1 本だけを検出 | detected 漏れ条件の確認 |
| negative_source_azimuth_offgrid_0p5_mask_nominal | source は +0.5 deg、mask は nominal 方位 | source 方位ずれと wide guard の確認 |

A2_safe / A2_aggressive は 5 条件の effective 判定でいずれも pass であった。ただし、unknown source / weak source を source mask 外に置く条件で pass になることは、運用上その source を non-source 抑圧対象として扱うことを意味する。したがって、この結果は未知 source 保護の合格ではなく、source 検出と mask 管理を運用条件として必須にする根拠として扱う。

### 9.5 Safety gate 確認

raw 悪化を人工的に作るため、代表 pass 条件の non-source sector だけを +6 dB 増幅した。この raw は `status=fail` となり、metric safety gate により effective は fixed_baseline と完全一致する後半 block へ fallback した。

`safety_gate_summary.csv` では、effective 行に次を保存した。

- `fallback_required=True`
- `fallback_reason=metric_safety_gate_gated_local_worsening|metric_safety_gate_ungated_local_worsening|metric_safety_gate_false_peak_increase|metric_safety_gate_raw_metric_fail`
- `effective_equals_fixed_baseline=True`
- `max_abs_diff_vs_fixed_baseline=0.0`

この確認により、raw が改善して見える条件でも、effective safety gate 後に改善しない場合は採用判定へ進めない運用を明示した。

### 9.6 実時間性

runtime は Python example 全体ではなく、A2 kernel である `_run_a2_source_mask_slc` の処理時間だけを測った。対象は代表 4 scenario と A2_safe / A2_aggressive の 2 候補であり、各条件 3 回 warmup、20 回 repeat とした。

| 候補 | 最大 p95 runtime_factor | 備考 |
|---|---:|---|
| A2_safe | 0.199 | A2 kernel only |
| A2_aggressive | 0.202 | A2 kernel only |

評価窓は one-block-delay の後半 1024 sample であり、runtime_factor は kernel elapsed / 評価窓時間である。この範囲では両候補とも代表条件で実時間予算内に収まった。

### 9.7 最終整理

| method | original sweep | representative | negative | worst p95 delta | max gated worsening | fallback rows | runtime p95 |
|---|---|---|---|---:|---:|---:|---:|
| fixed_baseline | baseline=990 | baseline=4 | baseline=5 | 0.000 | 0.000 | 0 | |
| A2_safe | pass=330 | pass=4 | pass=5 | -3.911 | -0.005 | 0 | 0.199 |
| A2_aggressive | pass=330 | pass=4 | pass=5 | -5.453 | -0.006 | 0 | 0.202 |

8月評価の主候補は A2_safe とする。理由は、one-block-delay、wide guard、effective pass を満たし、eta が 0.5 で過大キャンセルの余裕を残すためである。A2_aggressive は比較候補として残すが、eta=1.0 のため unknown source や detected 漏れを source-correlated leakage として強く引く危険を個別に確認する。

不採用条件は、effective で false peak が増えること、source peak delta が 1 dB を超えること、source mask 外 source を運用上保護したいのに A2 が抑圧すること、detected mask 1 本条件で局所悪化が safety gate なしに残ること、A2 kernel p95 runtime_factor が実時間予算を超えること、とする。

### 9.8 検証コマンド

```text
.venv\Scripts\python.exe -m ruff check src\spflow\beamforming\abf_like_metrics.py src\spflow\beamforming\source_mask_slc.py examples\beamforming\evaluate_lightweight_abf_like_comparison.py examples\beamforming\evaluate_lightweight_abf_like_august_shortlist.py tests\beamforming\test_abf_like_metrics.py tests\beamforming\test_source_mask_slc.py
結果: All checks passed

.venv\Scripts\pyright.exe -p artifacts\beamforming\lightweight_abf_like_comparison\pyright_check\pyrightconfig.json
結果: 0 errors, 0 warnings, 0 informations

.venv\Scripts\python.exe -m pytest tests\beamforming\test_abf_like_metrics.py tests\beamforming\test_source_mask_slc.py
結果: 8 passed
```

### 9.9 Source mask 外 source 消失リスクの追加確認

運用上の最大リスクである「source mask 外の source を A2 が source leakage とみなして消す」条件を、次の 4 scenario で追加確認した。

- `negative_source_azimuth_offgrid_0p5_mask_nominal`
- `negative_detected_mask_single_source`
- `negative_unknown_source_outside_mask`
- `negative_weak_source_outside_mask`

`negative_summary.csv` へ true source 方位最近傍 beam の tone projection delta を追加した。追加列は次である。

- `source_visibility_labels`
- `source_visibility_nearest_beam_indices`
- `source_visibility_nearest_azimuths_deg`
- `source_visibility_in_mask_flags`
- `source_visibility_level_delta_db`
- `mask_outside_source_count`
- `mask_outside_source_labels`
- `mask_outside_source_level_delta_db`
- `max_mask_outside_source_suppression_db`
- `max_mask_outside_source_abs_delta_db`

結果は次であった。

| scenario | method | mask 外 source | true source tone delta |
|---|---|---|---:|
| negative_unknown_source_outside_mask | A2_safe | unknown_outside_mask | -3.895 dB |
| negative_unknown_source_outside_mask | A2_aggressive | unknown_outside_mask | -5.074 dB |
| negative_weak_source_outside_mask | A2_safe | weak_outside_mask | -0.335 dB |
| negative_weak_source_outside_mask | A2_aggressive | weak_outside_mask | -0.206 dB |
| negative_detected_mask_single_source | A2_safe | weak_interferer_detected_miss | +0.004 dB |
| negative_detected_mask_single_source | A2_aggressive | weak_interferer_detected_miss | +0.008 dB |
| negative_source_azimuth_offgrid_0p5_mask_nominal | A2_safe | なし | 0.000 dB |
| negative_source_azimuth_offgrid_0p5_mask_nominal | A2_aggressive | なし | 0.000 dB |

この結果から、未知 source が source mask 外にあり、かつ source reference と相関を持つ条件では、A2 は non-source 指標を改善しながら未知 source の可視性を数 dB 下げ得ることが確認できた。特に `negative_unknown_source_outside_mask` では、A2_safe で約 3.9 dB、A2_aggressive で約 5.1 dB の低下が出ているため、8月評価では「source mask 外 source が保護対象になる運用」では A2 を採用しない。

一方、`negative_detected_mask_single_source` は detected mask が target 1 本だけを検出する条件であるが、弱 interferer の tone delta は +0.01 dB 未満であり、この合成条件では消失は起きていない。ただし、source mask 外である事実は変わらないため、検出漏れ source のレベル、周波数、相関条件を変えた追加評価なしに安全とはみなさない。

`negative_source_azimuth_offgrid_0p5_mask_nominal` は nominal mask 条件であるが、wide guard では true source が mask 内に入るため、mask 外 source count は 0 であった。この条件は「0.5 deg offgrid 自体では wide guard が source を保護できる」確認として扱う。

### 9.10 図注記の改善

資料用に BL/FRAZ/BTR 図の注記を更新した。offgrid 図では、target peak と true target 方位のわずかなずれを誤読しないよう、caption に次を出す。

```text
Nominal target azimuth = 90.0 deg
True target azimuth = 90.5 deg
Nominal source 2 = 60.0 deg
True source 2 = 60.5 deg
mask = oracle, centered at true source azimuths
```

nominal mask を使う負例では、最後の行を `mask = oracle, centered at nominal source azimuths` とした。detected mask では `mask = detected, centered at fixed-baseline RMS peaks` とした。

FRAZ の周波数ごとの peak marker は実在 source ではなく、その周波数 slice の最大方位である。そのため、凡例を `Peak 6144 Hz` から `Diagnostic slice max 6144 Hz` へ変更した。

また、指定 4 負例条件についても fixed_baseline / A2_safe / A2_aggressive の BL / FRAZ / BTR 図を保存した。図は合計 72 PNG である。

### 9.11 ChatGPT レビュー用 review_pack

PNG を個別に貼り付けずにレビューできるよう、`review_pack` を生成した。生成スクリプトは次である。

```text
.venv\Scripts\python.exe -m examples.beamforming.build_lightweight_abf_like_review_pack
```

出力先は次である。

- `artifacts/beamforming/lightweight_abf_like_comparison/review_pack/review_index.md`
- `artifacts/beamforming/lightweight_abf_like_comparison/review_pack/scenario_summary.csv`
- `artifacts/beamforming/lightweight_abf_like_comparison/review_pack/worst_cases.csv`
- `artifacts/beamforming/lightweight_abf_like_comparison/review_pack/figures/`
- `artifacts/beamforming/lightweight_abf_like_comparison/review_pack/data/`

`review_index.md` には、各 scenario の目的、source 方位 / 周波数、mask 種別、fixed / A2_safe / A2_aggressive の effective 判定要約、参照すべき図と CSV / npz の相対 path を記載した。

`scenario_summary.csv` は採否に使う effective row だけを対象とし、source peak delta、source azimuth error、non-source global / p95 / p99 / integrated delta、source-to-non-source margin、false peak count、gated local worsening、fallback、runtime factor を保存した。

`worst_cases.csv` には、各 metric の worst top 10、detected mask source count mismatch、fallback rows、negative case rows、A2_safe / A2_aggressive の差が大きい rows を保存した。

図は scenario ごとに次の 4 枚を保存した。

- `bl_overlay.png`: fixed / A2_safe / A2_aggressive を同一 y 軸で重ねる。
- `bl_delta.png`: A2_safe - fixed、A2_aggressive - fixed。
- `fraz_delta.png`: A2_safe - fixed、A2_aggressive - fixed。
- `btr_panel.png`: fixed / A2_safe / A2_aggressive を同一 color scale で横並び。

全図で source mask と non-source sector を背景色として表示した。BTR は `dB re frame max` であり、抑圧量の定量比較ではなく source track の連続性確認用として明記した。

`data/<scenario>.npz` には、描画前配列として `azimuth_deg`、`frequency_hz`、`time_sec`、`fixed_level_db`、`a2_safe_level_db`、`a2_aggressive_level_db`、FRAZ / BTR の各 level 配列、`source_mask`、`non_source_mask` を保存した。
