# 時間領域 MVDR / LCMV / GSC 方式検討

## 1. 目的

現行の時間領域 SLC は、target beam を壊さず interferer marker を局所的に下げる条件はあるが、BL 全体の guard 外 sidelobe peak を下げる方式としては不十分だった。したがって、次段の方式として channel×tap 空間で MVDR / LCMV / GSC を検討する。

本検討では、STFT へ進む前に時間領域 FIR 重みで実現できる範囲を確認する。重み更新は学習側、重み適用はリアルタイム側に分ける。リアルタイム側へ EVD、MUSIC、全ビーム全周波数の逐次 solve を入れない。

## 2. 共通表現

入力 channel 信号を次で表す。

```text
x[ch, n]
```

FIR tap 数を `L` とし、channel×tap snapshot を次で作る。

```text
x_tap[n] = [x[:, n], x[:, n-1], ..., x[:, n-L+1]]
shape: [n_ch * L]
```

実装では `build_time_tapped_snapshot_matrix()` が次の行列を返す。

```text
X_tap[dof, snapshot]
dof = n_ch * L
```

共分散は次で推定する。

```text
R = X_tap X_tap^H / K
shape: [n_dof, n_dof]
```

この `R` へ平均対角 power に対する比で diagonal loading を加える。絶対値 loading にしないのは、入力レベルが変わったときに正則化の相対強度が変わることを避けるためである。

## 3. tone 制約

複素 tone を次で表す。

```text
x_ch[n] = a_ch exp(j 2π f n / fs)
```

lag 付き snapshot では、過去 sample により次の位相が掛かる。

```text
c[lag, ch] = a_ch exp(-j 2π f lag / fs)
```

したがって、制約ベクトル `c` は shape `[n_ch * L]` であり、歪みなし条件は次になる。

```text
w^H c = 1
```

実信号 tone は正負周波数の共役対を持つため、target 保護では必要に応じて `[c, conj(c)]` を同時に制約する。これは現行 SLC の desired response blocking で正負周波数制約が必要だった理由と同じである。

## 4. MVDR

MVDR は LCMV の 1 制約である。

```text
minimize    w^H R w
subject to  c_t^H w = 1
```

解は次である。

```text
w = R_loaded^-1 c_t / (c_t^H R_loaded^-1 c_t)
```

MVDR は干渉方位を明示的に null 指定しない。干渉が共分散内で支配的で、target 制約と独立していれば出力 power 最小化により null が形成される。一方、target と干渉が高相関、または snapshot が不足する場合は、loading と制約不足により狙った BL 改善にならない可能性がある。

## 5. LCMV

LCMV は複数制約を同時に扱う。

```text
minimize    w^H R w
subject to  C^H w = f
```

例えば、target 保護と interferer null は次で表す。

```text
C = [c_target, c_interferer]
f = [1, 0]^T
```

解は次である。

```text
w = R_loaded^-1 C (C^H R_loaded^-1 C)^-1 f
```

LCMV は干渉方位・周波数が既知または推定済みである場合に使う。MUSIC やピーク追跡で null 候補を学習側で得て、リアルタイム側は固定済み FIR 重みを適用する構成にする。

## 6. GSC

GSC は LCMV を主経路と blocked reference に分解した実装形である。

```text
w = w_q - B g
C^H w_q = f
C^H B = 0
```

`w_q` は制約を満たす quiescent weight、`B` は制約空間を消す blocking matrix、`g` は blocked 空間で出力 power を下げる適応重みである。

今回追加した `design_time_domain_gsc_weights()` では、同じ `R_loaded` と同じ制約 `C, f` を使う場合に LCMV 解と一致することをテストで確認している。したがって、GSC は別の性能を持つ方式ではなく、制約保持と適応更新を分離しやすい実装方式として扱う。

## 7. リアルタイム経路と学習経路

リアルタイム経路は次だけを行う。

```text
1. channel×tap snapshot を作る
2. 保存済み FIR 重み w を適用する
3. RMS、NaN/inf、target power delta などの軽い監視を行う
```

学習経路は低周期で次を行う。

```text
1. 忘却積分で R を更新する
2. 必要なら MUSIC / peak tracking で null 候補を更新する
3. MVDR / LCMV / GSC 重みを解く
4. 制約応答、条件数、weight norm、runtime を確認してから publish する
```

この分離により、STFT 方式へ進んだ場合でも、リアルタイム側の処理量を固定 FIR 適用に近づけられる。

## 8. 評価基準

Beamforming Evaluation の pattern は `time_domain_adaptive_mvdr_lcmv_gsc` を使う。必須評価は次とする。

```text
adaptive_constraint_response
target_leakage_components
mainlobe_preservation
slc_covariance_health
runtime_budget
```

特に、BL 図だけでは採否を決めない。次を同時に確認する。

```text
target_constraint_response_error_db
null_constraint_response_db20
constraint_matrix_rank
degree_of_freedom
loaded_condition_number
raw_target_power_delta_db
raw_interferer_reduction_db
guard_outside_peak_delta_db
max_guard_outside_worsening_db
realtime_factor
```

`target が悪化しない` は採用条件ではなく前提条件である。採用には、target 保護、干渉低減、guard 外 peak 改善、最大悪化量抑制、実時間性を同時に満たす必要がある。

## 9. 現時点の実装範囲

`src/spflow/beamforming/time_domain_adaptive.py` に、次の共通部品を追加した。

```text
build_time_tapped_snapshot_matrix
estimate_time_domain_covariance
build_time_domain_tone_constraint_vector
build_real_tone_constraint_matrix
design_time_domain_mvdr_weights
design_time_domain_lcmv_weights
design_time_domain_gsc_weights
apply_time_domain_fir_beamformer
evaluate_constraint_response
diagnose_time_domain_adaptive_weights
```

テストでは次を確認している。

```text
- MVDR が target 制約 w^H c = 1 を満たす
- LCMV が target 保護と interferer null を同時に満たす
- GSC が同じ制約の LCMV 解と一致する
- FIR 適用時に full tap 以降で制約 tone が歪まず出る
- covariance health 診断が自由度、制約数、条件数を返す
```

次の評価では、運用スパースアレイの active channel と 3 秒忘却共分散を使い、現行 SLC で悪化した 8192 Hz interferer 条件に対して MVDR / LCMV / GSC の BL 重ね合わせと guard 外 peak 改善量を比較する。
## 10. 2026-07-05 代表条件での SLC / MVDR / LCMV / GSC 比較

代表条件は `target=10000 Hz, 90 deg`、`interferer=8192 Hz, 60 deg`、`memory_time_sec=3.0 s`、`active_channel_count=19`、`tap_len=3` である。比較では、必ず固定整相 target beam 応答を before とし、各方式の target beam 応答を after として重ねた。

注意点として、151 本 beam の source 方位 grid では interferer 真方位 `60.0 deg` が grid 点に一致せず、最近傍は `60.44 deg` になる。LCMV / GSC の明示 null は真方位 `60.0 deg` に置かれるため、grid 最近傍の BL 値だけを見ると null 深さを過小評価する。そのため、summary には grid 最近傍指標に加えて `interferer_frequency_exact_reduction_at_interferer_db` を追加した。

```text
method                                  grid_marker_reduction  exact_marker_reduction  guard_outside_peak_delta  max_guard_outside_worsening
SLC baseline                             23.915 dB              n/a                     +0.440 dB                 +25.848 dB
Time-domain MVDR real constraint         26.698 dB              45.271 dB               +0.900 dB                 +26.595 dB
Time-domain LCMV target+interferer null  27.736 dB              304.258 dB              +0.903 dB                 +26.643 dB
Time-domain GSC equivalent LCMV          27.736 dB              316.901 dB              +0.903 dB                 +26.643 dB
```

この結果から、LCMV / GSC の null 制約そのものは実装通り効いている。`constraint_response.max_null_constraint_response_db20` も -300 dB 程度であり、制約式と tap 並びは整合している。一方で、BL 全体の guard 外 peak は全方式で悪化している。したがって、現条件の時間領域 MVDR / LCMV / GSC は、真の干渉方位に鋭い null を作る方式としては成立するが、固定整相後 BL の guard 外 sidelobe peak を広く下げる方式としては採用できない。

方式判断としては、次を満たすまで合格にしない。

```text
- target_frequency_delta_at_target_db が許容範囲内
- interferer_frequency_exact_reduction_at_interferer_db が正
- interferer_frequency_sidelobe_metrics.guard_outside_peak_delta_db が負
- interferer_frequency_sidelobe_metrics.max_guard_outside_worsening_db が許容範囲内
```

今回満たせていないのは後半 2 条件である。次の検討では、単一 null の深さではなく BL 全体の peak 改善を目的に、複数 null 制約、方位幅を持つ null 制約、または channel shading / active aperture と LCMV の組み合わせを評価する。

## 11. 第一副極改善量の追加

BL before/after の評価では、guard 外最大 peak だけでなく第一副極のレベル改善量も必須にする。第一副極は、target mainlobe guard の外側に最初に現れる左右の局所 peak のうち、レベルが高い側として定義する。

summary には次を追加する。

```text
before_first_sidelobe_peak_db20
after_first_sidelobe_peak_db20
first_sidelobe_peak_delta_db
first_sidelobe_reduction_db
```

`first_sidelobe_reduction_db <= 0` の場合、干渉 marker が落ちていても方式は不合格とする。SLC / MVDR / LCMV / GSC の採否条件は次に更新する。

```text
- target_frequency_delta_at_target_db が許容範囲内
- interferer_frequency_exact_reduction_at_interferer_db が正
- interferer_frequency_sidelobe_metrics.first_sidelobe_reduction_db が正
- interferer_frequency_sidelobe_metrics.guard_outside_peak_delta_db が負
- interferer_frequency_sidelobe_metrics.max_guard_outside_worsening_db が許容範囲内
```

この更新により、BL 曲線として第一副極が落ちていない方式を、marker 一点の低下だけで改善と誤認しない。

## 12. 実装不足の洗い出しと方式見直し

SLC 前後 BL 図で改善が見えない原因が実装ミスだけでないかを確認した。Beamforming Evaluation の基準に従い、干渉 marker だけでなく guard 外 peak、最大局所悪化、第一副極改善量を見る。

確認した実装不足は次の 2 点である。

- 実信号 tone の BL レベルを `H(+f)` だけで評価していた。複素 SLC / MVDR / LCMV / GSC 重みでは `H(-f)` が `conj(H(+f))` にならないため、`sqrt((|H(+f)|^2 + |H(-f)|^2) / 2)` で RMS レベルを評価するよう修正した。
- SLC の対角 loading が絶対 power 値だった。入力レベルや blocking 後 reference power によって正則化の相対強度が変わるため、`loading * mean(diag(R_uu)) I` に修正した。

修正後の代表条件では、SLC baseline は `marker reduction = 15.99 dB` である一方、`guard_outside_peak_delta = +2.72 dB`、`max_guard_outside_worsening = +33.85 dB`、`first_sidelobe_reduction = -2.56 dB` であり、BL 改善としては不合格である。

時間領域 MVDR / LCMV / GSC も、制約点の null や target 保護は満たすが、BL 全体の第一副極は改善しない。代表条件では LCMV / GSC の `interferer_reduction` は約 `51.59 dB`、null 制約応答は十分小さいが、`guard_outside_peak_delta = +0.90 dB`、`max_guard_outside_worsening = +26.64 dB`、`first_sidelobe_reduction = -0.38 dB` である。

したがって、現行の「一点 null または target beam 後段キャンセル」方式は、固定整相 BL のサイドローブ改善方式としては目的関数が不足している。次の方式では、干渉方位の一点 null だけでなく、第一副極または guard 外 peak を目的関数または制約に含める必要がある。

次に検討する方式は、時間領域で続ける場合は sector-constrained LCMV / GSC とする。制約は target 正負周波数を歪みなし、interferer 方位を null、さらに interferer 周波数の第一副極候補または guard 外 peak 方位を抑圧制約として追加する。自由度が不足する場合は、STFT 方式へ切り替え、学習部で sector 制約重みを設計し、リアルタイム側は固定済みの周波数別重み適用に限定する。

### 12.1 sector-constrained LCMV / GSC の追加確認

方式を時間領域のまま改める候補として、干渉方位だけでなく固定整相 BL 上の第一副極方位と guard 外 peak 方位も null 制約に加えた sector-constrained LCMV / GSC を追加した。代表条件の制約方位は `60.0 deg`, `73.7398 deg`, `81.5662 deg` である。

結果は不合格である。sector-constrained LCMV / GSC は `interferer_reduction` を約 `50.09 dB` 得る一方、`guard_outside_peak_delta = +5.11 dB`、`max_guard_outside_worsening = +40.84 dB`、`first_sidelobe_reduction = -2.32 dB` となった。

この結果から、時間領域の少数点 null 制約では、抑えた方位の代わりに別方位へ sidelobe を押し出す。第一副極改善量を合格条件に入れる限り、SLC / MVDR / LCMV / GSC / sector-constrained LCMV は現条件の BL 改善方式として採用しない。

次の方式は STFT の周波数別重み設計だけへ固定しない。BL 全体の sidelobe envelope を下げる要求では、固定整相側の shading / active aperture 再設計と、dense sector envelope 制約付きの offline 重み設計を優先して比較する。STFT / サブバンド方式へ進む場合も、リアルタイム側は学習済みの周波数別重みを適用する処理に限定し、共分散推定と制約探索は学習側に寄せる。

## 13. レビュー反映後の結論

本検討の棄却対象は、channel×tap MVDR / LCMV / GSC を固定整相後の BL 改善器として使い、少数の interferer 方位、第一副極方位、guard 外 peak 方位へ null 制約を置く方式である。これは、時間領域で実装される重み設計全体を棄却する結論ではない。

現評価で分かったことは次の通りである。

```text
1. target 制約と interferer null 制約は数式通り満たせる。
2. 制約点の null は深くできる。
3. しかし、BL 全体の guard 外 peak と第一副極は別方位へ押し出される。
4. したがって、少数点 null 制約は BL sidelobe envelope 低減の目的関数として不足している。
```

今後、時間領域側で継続して見る価値があるのは、以下のように BL envelope を直接制約または目的関数へ入れる方式である。

```text
dense sidelobe sector:
  theta in guard 外の高密度方位 grid

目的または制約:
  minimize max |w^H a(theta, f)|
  または |w^H a(theta, f)| <= bound

同時に必要な制約:
  target response = 1
  passband ripple <= epsilon
  white noise gain >= lower_bound
  ||w|| <= upper_bound
```

この方向は、名称としては LCMV より minimax / inequality constrained beam pattern synthesis に近い。リアルタイム実装は時間領域 FIR 重みの適用だけにできるが、設計側は複数周波数・複数方位 grid を使うため、実質的には周波数別の offline 設計問題として扱う。

## 14. 未検討の時間領域系候補

現時点で未検討として残す項目は次である。

| 候補 | 位置づけ | 評価で必須にする指標 |
|---|---|---|
| 固定整相側の shading / active aperture | 後段キャンセラではなく、固定整相 BL パターン自体を作り直す方式。第一副極低減の本命候補。 | first_sidelobe_reduction_db, guard_outside_peak_delta_db, 3dB beamwidth, N_eff, expected_snr_gain_db |
| dense sector minimax / inequality LCMV | 少数点 null ではなく guard 外 sector 全体の envelope を抑える方式。 | max sidelobe, percentile sidelobe, integrated sidelobe level, WNG, target ripple |
| broadband FIR beamformer | channel×tap 重みを複数周波数・複数方位で offline 設計し、適用は時間領域 FIR とする方式。 | passband ripple, stopband sector bound, tap_len, runtime_factor, mismatch robustness |
| train / test 分離つき adaptive weight | 学習方位・周波数にだけ過適合した null を検出する評価。 | train/test reduction gap, off-grid worsening, unseen interferer reduction |

一方、SLC / MVDR / LCMV / GSC を局所 leakage canceller として使う評価は残す。採否は `role = local_leakage_canceller` として、target-only 保護、interferer-only 低減、mixed 条件、fallback、runtime を見る。`role = BL_sidelobe_reducer` では、第一副極、guard 外 peak、最大局所悪化量を満たさない限り不採用とする。


