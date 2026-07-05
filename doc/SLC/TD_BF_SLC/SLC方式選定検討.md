# SLC方式選定検討

## 1. 目的

本書は、`TD_BF_SLC` 系で採用する SLC 方式を選定するための検討記録である。

ここでは、現在の固定整相前段、評価済みアレイ条件、BL / FRAZ / BTR による評価方法、ユーザ要求を前提として、候補方式を比較し、現時点で採るべき方式と今後の段階的な進め方を定める。

---

## 2. 前提条件

### 2.1 固定整相前段

現時点の前提は次である。

- 固定整相は時間領域 Delay-and-Sum とする。
- 小数遅延は保存済み FIR バンクを読み出して適用する。
- 固定整相の BL / FRAZ / BTR 表示方法は既に部品化済みであり、SLC 前後比較でも同じ表示系を使う。

### 2.2 現時点で確認済みの固定整相性能

`artifacts/beamforming/fractional_delay_performance/performance_summary.json` によれば、小数遅延固定整相は `512 Hz` から `10000 Hz` の評価範囲で、最悪 peak margin `13 dB` 以上を満たしている。

代表値:

```text
6144 Hz : 14.003 dB
8192 Hz : 13.734 dB
10000 Hz: 13.705 dB
```

したがって、SLC の主目的は、固定整相が作った mainlobe を大きく壊さずに、特に高域での sidelobe / mirror をさらに抑えることである。

### 2.3 現時点で確認済みの SLC 挙動

`artifacts/beamforming/fractional_delay_slc_diagnostics/slc_summary.json` では、`6144 Hz` 単一周波数条件で次が確認できている。

```text
mainlobe_preserved    = True
mainlobe_level_delta  = -0.709 dB
sidelobe_reduction    = +0.588 dB
mirror_reduction      = +0.836 dB
```

つまり、現行の簡易 SLC は mainlobe 保持の傾向はあるが、mainlobe をやや削りつつ sidelobe を下げる状態にある。

---

## 3. 方式選定で満たすべき要求

今回の SLC 方式選定では、少なくとも次を満たす必要がある。

### 3.1 機能要求

- 固定整相前後の BL / FRAZ / BTR を比較できること。
- 複数方位、複数周波数で動作できること。
- 同一方位に異なる周波数の信号が重なっても扱えること。
- スパース片舷アレイで成立すること。
- 高域での sidelobe / mirror 抑圧を重点的に確認できること。

### 3.2 安全側要求

- mainlobe peak 位置を大きく動かさないこと。
- mainlobe peak level を過度に落とさないこと。
- guard 幅を安易に削って参照を確保しないこと。
- 参照不足時は弱化または無効化できること。

### 3.3 実装要求

- 既存の固定整相前段を大きく壊さないこと。
- 保存済み小数遅延 FIR バンク方式と矛盾しないこと。
- 将来の実時間化を見据えて、計算量と状態量が管理できること。

---

## 4. 候補方式

## 4.1 候補A: 時間領域ビーム出力に直接かける broadband beam-domain SLC

### 概要

固定整相後の `beam_output[beam, sample]` に対して、そのまま block least-squares 型 SLC を掛ける方式である。

数式上は、

```text
y[n] = d[n] - w^H u[n]
```

であり、`u[n]` は guard 外ビーム群、`d[n]` は target beam である。

### 利点

- 実装が最も単純である。
- 固定整相後の時系列をそのまま扱える。
- 周波数分解を導入しないため、パイプラインが短い。

### 欠点

- 広帯域成分を 1 組の係数で扱うため、周波数ごとの最適キャンセルがしにくい。
- 高域と低域で mainlobe 幅が異なる問題を guard 1 つで吸収しにくい。
- 同一方位・異周波の分離設計が難しい。
- 固定整相後の target 成分が参照に広帯域で混入すると self-nulling を起こしやすい。

### 判定

L=1 時間領域方式として採用する。ただし、target-only、同一周波数干渉、異周波数干渉、実時間性の各評価基準を満たす条件に限って運用出力へ採用する。

理由は、固定整相が時間領域で完結しており、`beam_output[beam, sample]` からそのまま統計サンプルを多く取れるためである。1 秒以内処理を目標にする場合、STFT / 周波数ビンごとの共分散を最初から持つより、共分散を 1 つにした時間領域 SLC の方が状態量、計算量、安定性を管理しやすい。

---

## 4.2 候補B: 単一周波数ごとに複素 snapshot を作る narrowband beam-domain SLC

### 概要

固定整相後のビーム時系列から、評価したい周波数だけを複素係数化し、その周波数でだけ SLC を掛ける方式である。

現在の `time_delay_slc_diagnostics.py` と `fractional_delay_slc_diagnostics.py` はこの考え方に近い。

### 利点

- 高域だけ、あるいは特定周波数だけを選んで評価できる。
- BL / FRAZ / BTR で周波数選択的な before / after 比較がしやすい。
- 小数遅延固定整相の理論応答行列 `steering_response()` をそのまま利用できる。
- 単一トーン条件では、blocking 的な射影を入れやすい。

### 欠点

- 単一または少数の離散周波数に寄った評価方式であり、広帯域運用の本番実装にはそのまま使いにくい。
- 実音場が複数帯域にまたがる場合、各周波数成分を個別に追う必要がある。
- 現状は診断用途の色が強く、連続処理器としては未完成である。

### 判定

方式選定用の評価基盤としては有効だが、運用方式の最終形ではない。

---

## 4.3 候補C: STFT / サブバンド化した beam-domain frequency-selective SLC

### 概要

固定整相後のビーム出力を STFT またはサブバンド分解し、各周波数ビンごとに独立した SLC を掛ける方式である。

この候補は、より具体的には beam-domain GSC として設計する。固定整相後のビーム出力から共分散を作り、guard 外 reference beam に含まれる target 応答を blocking projector で除去してから適応キャンセル係数を求める。

処理イメージ:

```text
fixed beam output [n_beam, n_sample]
  ↓
STFT / subband analysis
  ↓
X[beam, freq, frame]
  ↓
for each freq:
  guard / reference selection
  target response blocking
  covariance update
  SLC weight solve
  Y[target, freq, frame]
  ↓
必要に応じて可視化または再合成
```

### 利点

- guard 幅を周波数依存で変えられる。
- `eta`、`loading`、`min_ref` も周波数依存で設計できる。
- 同一方位・異周波の重畳に自然に対応できる。
- 複数方位・複数周波数を同時に扱いやすい。
- 高域だけ重点的に強いキャンセル、低域は保守的運用、という設計が可能である。
- 現在の narrowband 診断ロジックを、そのまま各周波数ビンへ拡張する方向に近い。
- fixed beam output から共分散を作るため、チャネル領域 GSC より現行評価系との接続が明確である。

### 欠点

- 実装量が増える。
- 周波数ビンごとの共分散行列を持つため、状態量が増える。
- 再合成する場合は STFT / overlap-add 側の整合が必要になる。

### 判定

性能面では最も有利な拡張候補である。

ただし、時間領域 L=1 が評価基準を満たさない条件を確認してから本番経路へ入れる。周波数ごとの共分散は、広帯域干渉やビームパターンの周波数依存を表現しやすい一方、共分散行列と重みを周波数ビン数分だけ持つ必要があり、各ビンの統計サンプル数も STFT フレーム数に制限される。リアルタイム CPU 処理と初期安定性を優先し、まずは時間領域 SLC で効果と限界を測る。

---

## 4.4 候補D: チャネル領域 GSC / blocking matrix 型 SLC

### 概要

固定整相後のビームを使う代わりに、チャネル信号から blocking matrix を作り、主経路と参照経路を分ける古典的 GSC に近い方式である。

### 利点

- 理論的には最も正統である。
- blocking matrix が適切に作れれば、desired 保護を厳密にしやすい。

### 欠点

- 現在の固定整相後評価パイプラインから大きく外れる。
- 小数遅延固定整相後の BL / FRAZ / BTR 比較という現在の評価軸と直結しにくい。
- スパース片舷アレイでの blocking 設計が重くなる。
- 実装変更量が大きい。

### 判定

将来の比較対象としては持つ価値があるが、現段階の主方式にはしない。

---

## 5. 候補比較表

| 候補 | 実装難度 | 多周波対応 | 同方位異周波 | mainlobe保護設計 | 現行資産との整合 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| A: broadband beam-domain | 低 | 低 | 低 | 中 | 高 | 初期採用 |
| B: narrowband beam-domain | 中 | 中 | 低〜中 | 中 | 高 | 診断基盤 |
| C: STFT/subband beam-domain | 中〜高 | 高 | 高 | 高 | 高 | 拡張本命 |
| D: channel-domain GSC | 高 | 高 | 高 | 高 | 低 | 将来比較 |

---

## 6. 現時点の選定結論

### 6.1 採用方針

現時点では、**候補A: 時間領域ビーム出力に直接かける broadband beam-domain SLC L=1** を、評価基準を満たす条件に限定した正式方式として採用する。

これは、周波数ごとの違いを 1 つの平均的なキャンセラで吸収する設計である。広帯域に対して完全ではないが、軽く、安定し、固定整相後の `beam_output[beam, sample]` をそのまま使える。

一方で、性能上の最終候補としては **候補C: STFT / サブバンド化した beam-domain frequency-selective SLC** を維持する。周波数ごと、または帯域ごとに共分散を持つ方式は、ビームパターン、サイドローブ、FIR 残留誤差、干渉スペクトルの周波数依存を扱いやすいためである。

### 6.2 選定理由

理由は次の通りである。

1. 固定整相前段は既に小数遅延込みで 0 Hz から 10000 Hz の性能土台ができている。
2. 第1段階の正式実装では 1 秒以内処理を意識し、CPU で扱いやすい状態量に抑える必要がある。
3. 時間領域 SLC では統計サンプル数を `n_sample` として使えるため、STFT ビン別の `n_frame` 統計より初期安定性を確保しやすい。
4. 既存 `BeamDomainSLC` が `R_uu[n_ref, n_ref]` と `r_ud[n_target, n_ref]` を 1 組持つ設計であり、固定整相後段に自然に接続できる。
5. 時間領域 SLC で不足する条件が見えた時点で、時間タップ付き、帯域別、周波数ビン別へ段階拡張できる。

---

## 7. 実装はどう段階化するか

## 7.1 第1段階: 時間領域 SLC L=1 を評価基準付き正式方式として使う

まずは固定整相後の `beam_output[beam, sample]` に対して、時間領域 SLC を適用する。

この段階の状態量:

```text
R_uu: [n_ref, n_ref]
r_ud: [n_target, n_ref]
w:    [n_target, n_ref]
```

この段階で振る項目:

- `guard_effective`
- `eta_normal`
- `eta_limited`
- `loading`
- `max_reference_beams`
- `block_size`
- `memory_time_sec`

目的:

- 固定整相の mainlobe を大きく壊さずに SLC が効く条件を確認する。
- CPU で 1 秒以内処理できる計算量、参照本数、ブロック長を見積もる。
- 広帯域条件でキャンセル不足が出る帯域を特定する。

時間領域 SLC は共分散を 1 つだけ持つため、実行時の guard も周波数ごとには切り替えない。固定整相 BL から設計した `guard(f)` 表は、L=1 時間領域方式では評価帯域または運用帯域から単一の `guard_effective` を決めるために使う。低域 target を保護する必要がある場合は広い guard、高域重点の評価では高域側 guard を候補にする。

## 7.2 第2段階: 時間タップ付き SLC を検討する

時間領域 L=1 で不足する場合、まず時間タップ付き SLC を検討する。

```text
u_L[n] = [u[n], u[n-1], ..., u[n-L+1]]
```

この方式は FFT / STFT を導入せず、参照信号の短い FIR 形キャンセラとして周波数依存を部分的に吸収する。

候補タップ長は `L=3〜8` を優先する。タップ長を増やすと自由度が `n_ref * L` へ増えるため、`sample_per_dof` による有効条件を必ず維持する。

## 7.3 第3段階: 帯域別 beam-domain SLC を検討する

時間タップ付きでも、低域と高域で効き方が大きく違う場合は、帯域別 SLC を検討する。

この段階の状態量:

```text
R_uu: [n_band, n_ref, n_ref]
r_ud: [n_band, n_target, n_ref]
w:    [n_band, n_target, n_ref]
```

帯域例:

```text
band 0: 100-300 Hz
band 1: 300-800 Hz
band 2: 800-1500 Hz
band 3: 1500-3000 Hz
band 4: 3000-10000 Hz
```

全 FFT ビン別より状態量が少なく、周波数依存のビームパターンもある程度吸収できる。

## 7.4 第4段階: STFT / 周波数ビン別 beam-domain GSC を実装する

最後に、固定整相後の `beam_output` を STFT / サブバンドへ変換し、周波数ビン単位で SLC を掛ける方式を実装する。

この段階の処理単位:

```text
X[beam, freq, frame]
```

少なくとも次が必要である。

- 周波数ビンごとの guard 設定
- 周波数ビンごとの reference 選択
- 周波数ビンごとの target response blocking
- 周波数ビンごとの共分散更新
- 周波数ビンごとの `eta` / `loading`

## 7.5 第5段階: channel-domain GSC は比較用に留める

beam-domain 方式で mainlobe 保持と sidelobe reduction が両立できない場合に限り、channel-domain GSC を比較対象として検討する。

現時点では主線にしない。

---

## 8. 設計上の重要論点

## 8.1 mainlobe 保護指標を 1 つにしない

現状の評価結果では、

- `mainlobe_preserved = True`
- `sidelobe_reduction_db > 0`
- しかし `mainlobe_margin_improvement_db < 0`

が起きている。

したがって、SLC 方式選定では次を独立に見る。

- mainlobe peak shift
- mainlobe level delta
- sidelobe reduction
- mirror reduction
- local margin improvement

`margin` だけで方式優劣を決めてはいけない。

## 8.2 guard は周波数依存設計表から決める

固定整相だけで求めた `guard(f)` は、SLC の guard 設計入力として使う。

時間領域 SLC L=1 では共分散が 1 つであり、実行時に周波数ごとの guard を切り替えられない。そのため、L=1 時間領域方式では `guard(f)` 表から運用帯域に対する単一の `guard_effective` を決める。低域 target を守る必要がある場合は広い guard を採用し、高域重点の評価では高域側 guard を候補にする。

帯域別 SLC または周波数ビン別 SLC へ進んだ段階では、`guard(f)` を帯域または周波数ビンごとに直接適用する。

理由は次の通りである。

- 低域では mainlobe が広い。
- 高域では mainlobe が狭い。
- guard 一定では、低域で自己消去、高域で参照不足のどちらかに寄りやすい。

## 8.3 参照本数不足は guard を削って解かない

現実装の方針は正しい。

- guard は維持する。
- 参照は等角度間引きで減らす。
- それでも不足なら `LIMITED_REFERENCE` または `DISABLED` にする。

この方針は L=1 時間領域方式と周波数依存方式のどちらでも維持する。

## 8.4 高域は強め、低域は保守的にする

現在の課題は高域で顕著であるため、周波数依存制御は次の方向が妥当である。

- 低域: `eta` 小さめ、guard 広め、保守運用
- 高域: `eta` やや強め、guard は固定整相設計値準拠

---

## 9. 直近の設計判断

現時点での具体判断を次に固定する。

### 9.1 今すぐ本線とする方式

- **時間領域 beam-domain SLC**
- 共分散は周波数ごとに持たず、固定整相後 `beam_output[beam, sample]` から 1 組だけ推定する。
- 初期タップ長は `L=1` とする。

### 9.2 今すぐ使う評価方式

- 現在実装済みの narrowband 診断方式を使う
- 小数遅延固定整相を前段に固定する
- 高域 `6144 / 8192 / 10000 Hz` を重点評価する
- narrowband / frequency-selective 診断は、L=1 時間領域方式の限界確認と周波数依存方式の比較基準として使う

### 9.3 今は採らない方式

- 最初から周波数ビン別の共分散を本番経路へ入れない
- 最初から STFT 再合成を本番経路へ入れない
- channel-domain GSC を主線にはしない

---

## 10. 次の作業項目

次は、以下の順で進めるのが妥当である。

1. 時間領域 SLC L=1 で、複数方位・複数周波数・同一方位異周波の before / after を評価する。
2. 1 秒入力を 1 秒以内に処理できるか、固定整相 + SLC の CPU 実測を追加する。
3. 高域で `mainlobe_level_delta` と `sidelobe_reduction` の両立点を探す。
4. 不足する場合は `L=3〜8` の時間タップ付き SLC を評価する。
5. それでも不足する場合に、帯域別 SLC、周波数ビン別 SLC の順で拡張する。

---

## 11. 結論

現時点の方式選定としては、次で進める。

```text
前段:
  小数遅延固定整相

SLC方式:
  評価基準を満たす条件では時間領域 beam-domain SLC L=1

評価段階:
  時間領域 before/after を主評価とし、narrowband beam-domain GSC 診断を比較基準に使う

拡張順:
  時間タップ付き SLC
  → 帯域別 SLC
  → 周波数ビン別 beam-domain GSC
```

この選定は、現行資産を活かしつつ、リアルタイム CPU 処理、複数方位、複数周波数、同一方位異周波、高域重点、スパース片舷アレイという要件を段階的に満たすためのものである。


---

## 12. 2026-07-05 レビュー結果と設計更新

### 12.1 Beamforming Evaluation に基づく評価パターン

今回の SLC 方式検討では、以下の評価パターンを必須の見方として扱う。

```text
slc_target_only:
  mainlobe preservation, target leakage components, waveform integrity, input/output level consistency

slc_same_frequency_interference:
  target leakage components, mainlobe preservation, slc covariance health

slc_different_frequency_interference:
  target leakage components, mainlobe preservation, waveform integrity

slc_runtime:
  runtime budget, slc covariance health, array file consistency
```

BL / FRAZ / BTR だけで SLC の採否を判断しない。target-only、interferer-only、mixed を分け、raw SLC 候補と safety fallback 後の運用出力を別々に記録する。

### 12.2 洗い出した問題点

1. `slc_covariance_health` の必須 metric である condition number が時間領域 SLC 診断 summary に出ていなかった。これは評価漏れであるため、`R_uu + loading I` の condition number を `condition_number` として記録する。
2. `BeamDomainSLC` は `tap_len=1` では `y[n] = d[n] - w^H u[n]`、`tap_len > 1` では `u[n], u[n-1], ...` を `[n_ref * L, n_sample - L + 1]` へ展開する時間タップ付き SLC として扱う。先頭 `L-1` sample は履歴不足で適応 cancellation を定義できないため、固定整相出力を通す。
3. 10000 Hz 同一周波数 deterministic tone 条件では、target と interferer が高相関になり、L=1 時間領域 SLC は target beam 上の干渉漏れ込みを減らせない。eta 調整だけで方式成立と見なしてはいけない。
4. target-only では desired response blocking により自己消去は改善するが、eta=1.0 は target 低下が大きい。target 保護条件を満たす eta を評価で決める必要がある。
5. 異周波数 interferer では干渉低減が出る条件があるが、target 低下との trade-off を `mainlobe_preservation` と `target_leakage_components` の両方で判定する必要がある。
6. 診断図と summary の dB 表記は、単なる `dB` や `dB20` ではなく、シミュレーション基準として `dB re input RMS` を併記する。

### 12.3 設計判断

L=1 時間領域 SLC は、次を満たす条件でのみ採用する。

```text
raw_target_power_delta_db が許容範囲内
raw_interferer_reduction_db が正で、要求下限を満たす
condition_number が異常に大きくない
weight_norm が過大でない
NaN / inf がない
1 秒入力を 1 秒以内に処理できる
```

同一周波数・高相関条件では、L=1 時間領域 SLC を eta だけで押し通さない。次のいずれかへ進める。

```text
1. target absent / training 区間で desired を含まない共分散を推定する
2. target response blocking を multi-constraint 化する
3. 帯域別 SLC で周波数依存 guard / eta / loading を持つ
4. STFT-bin beam-domain GSC で周波数ビンごとの共分散を持つ
```

### 12.4 次に進める設計

次の作業は、方式そのものを正しく分けて評価できるようにすることである。

1. 時間領域 SLC 診断 summary に condition number、weight norm、reference count、capacity を必ず出す。
2. `tap_len > 1` は時間タップ付き SLC として扱い、reference 行列を `[n_ref * L, n_sample - L + 1]` へ明示的に展開する。
3. 同一周波数干渉では、target absent 学習または multi-constraint blocking を優先して評価する。
4. 異周波数干渉では eta sweep を `slc_different_frequency_interference` の必須基準で評価し、target 低下と干渉低減の採用範囲を決める。
5. L=1 が採用条件を満たさない帯域・条件を確認した場合、帯域別 SLC の詳細設計へ進む。

---

## 13. 同一方位・複数周波数を主対象にする設計整理

### 13.1 方針

同一周波数の複数 source を SLC だけで分離することは主目標にしない。同一周波数かつ高相関の deterministic tone では、target と interferer の信号部分空間が分離しにくく、L=1 時間領域 SLC でも narrowband GSC でも自己消去や干渉増加が起きやすい。

一方、同一方位に複数周波数が重なる条件は、空間分離ではなく周波数分離として扱う。同じ方位にある成分は steering が同じなので、reference beam から空間的に片方だけを消すのではなく、STFT bin または帯域別出力で周波数成分を分ける。

### 13.2 処理単位

同一方位・複数周波数では、固定整相後の target beam 出力を次のように周波数軸へ分解する。

```text
beam_output[beam, sample]
  -> STFT / subband analysis
X[beam, freq, frame]
  -> target beam または target 周辺 beam を選択
Y[target, freq, frame]
```

評価対象は、方位 peak を分けることではなく、同じ target 方位上で周波数成分が分離して見えることである。FRAZ では同じ方位に複数の周波数 ridge が立ち、BTR では同じ方位 ridge に各周波数成分が重畳して見えることを正常とする。

### 13.3 SLC との関係

同一方位・異周波数成分は、target beam の main path に入る成分であり、guard 外 reference beam から推定して差し引く sidelobe 干渉とは性質が違う。そのため、SLC は次の用途に限定する。

```text
1. 別方位から target beam へ漏れる sidelobe 干渉を下げる
2. 周波数別に分けた後、各 band / bin で必要な場合だけ beam-domain GSC を掛ける
3. 同一方位の別周波数成分は、SLC ではなく周波数選択で分ける
```

### 13.4 評価パターン

この条件は `slc_same_azimuth_multi_frequency` として扱う。必須評価は次である。

```text
frequency_component_separation:
  target_frequency_power_delta_db
  off_frequency_reduction_db
  frequency_bin_leakage_db
  analysis_bandwidth_hz

mainlobe_preservation:
  同一方位の target beam レベルを維持する

fraz_btr_consistency:
  FRAZ 上で周波数 ridge が分かれ、BTR 方位 ridge が破綻しない

waveform_integrity:
  STFT / subband 後に NaN / inf や不自然な power 変化がない
```

### 13.5 次の設計判断

次に実装する場合は、L=1 時間領域 SLC の eta 調整ではなく、固定整相後 beam output の STFT / subband 分解を先に設計する。まずは再合成を要求せず、`X[beam, freq, frame]` 上で同一方位・複数周波数の成分別 level と leakage を評価する。

---

## 14. 時間領域優先の段階評価と MVDR / MUSIC / STFT への切替条件

### 14.1 基本方針

固定整相は可能な限り時間領域で進める。STFT は最初から本番経路へ入れず、時間領域方式で評価基準を満たせない条件が明確になった場合の切替先とする。

段階は次の順に固定する。

```text
1. 時間領域 SLC L=1
2. 時間タップ付き SLC L=3 / 5 / 8
3. 時間領域での constrained beamformer / MVDR 系
4. MUSIC は source 数・方位・周波数候補の診断または学習補助に使う
5. それでも性能が出ない条件だけ STFT / subband 方式へ切り替える
```

### 14.2 時間タップ付き SLC の評価

時間タップ付き SLC は、

```text
u_L[n] = [u[n], u[n-1], ..., u[n-L+1]]
```

を reference とする短い FIR 型キャンセラである。実装では reference 行列を次の shape へ展開する。

```text
U:   [n_ref, n_sample]
U_L: [n_ref * L, n_sample - L + 1]
D_L: [n_target, n_sample - L + 1]
W:   [n_target, n_ref * L]
```

先頭 `L-1` サンプルは full tap が揃わないため、固定整相出力を通す。評価では `dof = n_ref * L`、`block_size = n_sample - L + 1`、`block_size / dof`、condition number、weight norm を必ず記録する。

### 14.3 MVDR / MUSIC の位置づけ

時間タップ付き SLC でも target 維持と interferer 低減が両立しない場合、次に検討するのは MVDR / LCMV 系である。target 方位を distortionless 制約に置き、推定した interferer 方位または guard 外方向に null / 抑圧制約を置く。

MUSIC は、リアルタイム出力に直接入れる処理ではなく、まず学習・診断側で使う。用途は次である。

```text
- source 数の推定
- interferer 方位候補の推定
- 同一周波数・高相関条件で分離不能に近い状態の検出
- MVDR / LCMV の制約方向候補の生成
```

### 14.4 リアルタイム処理と学習処理の分離

STFT / subband 方式へ進む場合でも、リアルタイム経路へ重い EVD や全 bin の重み更新を毎フレーム入れない。

```text
学習側:
  STFT / subband covariance
  MUSIC / EVD
  MVDR / LCMV 重み更新
  guard / eta / loading / null 方位の更新

リアルタイム側:
  固定整相
  既に決まった時間領域 FIR SLC または固定重みの適用
  必要最小限の監視指標計算
```

重み更新周期は出力サンプル周期とは分離する。例えば 1 秒ごとの学習更新で重みを決め、リアルタイム側は次の 1 秒ブロックへ固定係数を適用する。これにより、STFT を使う場合でも処理量を学習側へ寄せ、リアルタイムの遅延と CPU 負荷を抑える。

### 14.5 STFT へ切り替える条件

以下を満たせない場合に STFT / subband 方式へ進む。

```text
- target-only で target_power_delta_db が許容範囲に入らない
- 異周波数 interferer で off-frequency reduction が不足する
- 時間タップ付き SLC の condition number / weight norm が悪化する
- 同一方位・複数周波数で frequency_bin_leakage_db が要求を満たさない
- MVDR / LCMV の時間領域更新で runtime_budget を超える
```

---

## 15. 時間領域方針の評価結果

### 15.1 実信号 blocking の扱い

時間領域 SLC の desired response blocking は、実信号 tone を扱う場合 `A(f)` だけでは不足する。実信号は正負周波数の共役成分を持つため、reference 空間から target を落とす制約は `[A(f), conj(A(f))]` とする。これにより target-only 条件の約 3 dB 自己消去は解消した。

### 15.2 採用判断

`operational_time_domain_slc_condition_sweep` の結果から、現条件では既定を `tap_len=1` とする。

- target-only: `raw_target_power_delta_db = -0.000005 dB re before level`
- 異周波数 interferer 8192 Hz: `raw_interferer_reduction_db = 33.009 dB re before level`
- 異周波数 interferer 6144 Hz: `raw_interferer_reduction_db = 37.345 dB re before level`
- 同一周波数 interferer: `raw_interferer_reduction_db = -26.768 dB re before level`

L=3 / L=5 は条件付き候補に残すが、L=5 は realtime factor が 0.838 と余裕が小さい。L=8 は realtime factor が 1.831 のため、リアルタイム経路では採用しない。

### 15.3 同一方位・複数周波数

`operational_same_azimuth_frequency_separation` の結果では、同一方位 6144 / 8192 / 10000 Hz の固定整相後 target beam に対して、`analysis_bandwidth_hz = 1.0`、`max_abs_target_frequency_power_delta_db = 0.0068`、`worst_frequency_bin_leakage_db = -75.982` だった。

この条件では、時間領域固定整相後の周波数成分分析で目的を満たす。MVDR / MUSIC / STFT は、非定常条件、周波数近接条件、または `frequency_bin_leakage_db` 悪化を確認してから、学習側の処理として検討する。

### 15.4 共分散の忘却積分評価への更新

共分散推定は 1 秒一括ではなく、`slc_analysis_block_size` ごとの block-wise 更新とし、`alpha = exp(-block_time_sec / memory_time_sec)` で忘却積分する。成分別評価も各 block の係数を同じ block へ適用し、逐次運用に近い条件で判定する。

`slc_analysis_block_size = 8192`、`memory_time_sec = 1.0` の再評価では、L=1 が target 保護と異周波数 interferer 低減を両立し、`realtime_factor` も 0.2 未満だった。L=3 は `realtime_factor = 0.828` で条件付き候補、L=5 / L=8 は 1 を超えるためリアルタイム経路では採用しない。

### 15.5 共分散積分時間の採用範囲

`memory_time_sec` は短すぎると共分散の有効平均 block 数が不足する。`slc_analysis_block_size = 8192` の評価では、`memory_time_sec = 0.25 s` 以下は有効独立 block 数が 3 未満であり、異周波数 interferer 条件の条件数相対標準偏差も 0.57 以上だった。

採用範囲は `0.5 s` から `1.0 s` とし、既定は `1.0 s` とする。`0.5 s` は追従性を優先する条件の候補、`1.0 s` は条件数のばらつき低下と干渉低減を両立する標準条件である。`0.25 s` 以下は、干渉低減量が出ていても共分散値が安定していないため採用しない。`2.0 s` は定常条件に限定する。

### 15.6 1 秒から 5 秒オーダーでの積分時間見直し

共分散積分時間は 1 秒から 5 秒のオーダーを主評価範囲とする。`memory_time_sec = 1.0, 2.0, 3.0, 5.0` の再評価では、target-only 条件の自己消去は全条件で無視でき、異周波数 interferer 8192 Hz は全条件で 30 dB re before level 以上低減した。

既定は `memory_time_sec = 1.0 s` とする。理由は、5 秒評価で 3 tau 以上を観測でき、異周波数 interferer 低減が 34.225 dB re before level と最大だったためである。共分散値の安定性を優先する場合は `2.0 s` から `3.0 s` を候補にする。`5.0 s` は条件数ばらつきが最も小さいが、追従が遅くなるため定常条件に限定する。

### 15.7 積分時間 3 秒の採用

共分散積分時間は `memory_time_sec = 3.0 s` を運用設定とする。1 秒は追従性と干渉低減量を優先する候補、5 秒は定常条件限定の候補として残す。3 秒は、異周波数 interferer 8192 Hz で `31.542 dB re before level` の低減を維持しつつ、条件数の block 間相対標準偏差を 1 秒条件より抑えられるため、現時点の標準条件にする。

### 15.8 SLC 後 BL 評価定義の訂正

SLC 後 BL は、全 beam を個別に target として処理した曲線ではなく、保護 target beam を固定した source 方位応答として評価する。この定義で、固定整相後の target beam 応答と SLC 後の target beam 応答を重ねる。

3 秒設定では、target 方位の target 周波数応答は維持され、interferer 方位の 8192 Hz 応答は `23.913 dB re before level` 低下した。以後、SLC の BL 図はこの定義を採用する。

### 15.9 BL サイドローブ改善量の採否基準

SLC の採否は、target が悪化しないことだけでは判断しない。target 保護は前提条件であり、そのうえで固定整相後 BL と SLC 後 BL の差から改善量を測る。

採否で必ず見る指標は次の通り。

```text
target_frequency_delta_at_target_db
interferer_frequency_reduction_at_interferer_db
target_frequency_sidelobe_metrics.guard_outside_peak_delta_db
target_frequency_sidelobe_metrics.max_guard_outside_worsening_db
interferer_frequency_sidelobe_metrics.guard_outside_peak_delta_db
interferer_frequency_sidelobe_metrics.reduction_at_marker_db
interferer_frequency_sidelobe_metrics.max_guard_outside_worsening_db
```

`interferer_frequency_sidelobe_metrics.reduction_at_marker_db` が正でも、`guard_outside_peak_delta_db` が改善しない場合は、方式の効果を「BL 全体の sidelobe 低減」とは呼ばない。この場合は、target beam に混入する特定方位・特定周波数の leakage を落とす局所キャンセラとして評価する。BL 全体の sidelobe peak を下げる要求がある場合は、固定整相の shading / active aperture 設計、MVDR / LCMV、または STFT / subband 学習側の方式へ進む。

### 15.10 時間領域 MVDR / LCMV / GSC への移行

現行 SLC は局所的な leakage canceller としては効果があるが、BL 全体の guard 外 sidelobe peak を下げる方式としては不十分である。そのため、次段の方式は channel×tap 空間の時間領域 MVDR / LCMV / GSC とする。

MVDR は 1 制約 LCMV として扱い、LCMV は target 保護と interferer null を同じ制約行列 `C` で扱う。GSC は LCMV の `w = w_q - B g` 分解として扱い、同じ制約と共分散では LCMV と同じ解になることを実装テストで確認する。

評価 pattern は `time_domain_adaptive_mvdr_lcmv_gsc` を使う。target が悪化しないことは前提条件であり、採否は `adaptive_constraint_response`、`target_leakage_components`、`mainlobe_preservation`、`slc_covariance_health`、`runtime_budget` を同時に満たすかで決める。

詳細は `時間領域MVDR_LCMV_GSC方式検討.md` に分離した。

### 15.11 第一副極改善量の追加

BL before/after 比較では、guard 外最大 peak と最大悪化量に加え、第一副極レベルの改善量を必須評価にする。`first_sidelobe_reduction_db <= 0` の場合、target が維持され、干渉 marker が落ちていても方式は改善なしとして扱う。

### 15.12 実装不足確認後の方式判断

SLC の BL 改善不足について、実装ミスの可能性を優先して確認した。修正した点は、実信号 tone の BL 評価で正負周波数応答を RMS 合成すること、および SLC の対角 loading を `loading * mean(diag(R_uu)) I` として入力 power に対する相対値にすることである。

修正後も、SLC は干渉 marker を落とす一方で guard 外 peak と第一副極を悪化させる。代表条件では `first_sidelobe_reduction_db < 0` であり、方式として採用しない。

この結果から、SLC の後段キャンセラ方式ではなく、第一副極または guard 外 peak を明示的に抑える sector-constrained LCMV / GSC へ検討を移す。時間領域で自由度が足りない場合は、STFT 方式に切り替える。ただし STFT 方式でも、リアルタイム側は学習済み重みの適用に寄せ、共分散推定や sector 制約探索を常時処理へ入れない。

### 15.13 sector-constrained LCMV / GSC の棄却

時間領域のまま方式を改める候補として、干渉方位、第一副極方位、guard 外 peak 方位を null 制約に入れる sector-constrained LCMV / GSC を評価した。制約点では null が形成されるが、BL 全体では sidelobe が別方位へ押し出され、第一副極改善量は負のままである。

代表条件では `first_sidelobe_reduction_db = -2.32 dB`、`guard_outside_peak_delta_db = +5.11 dB`、`max_guard_outside_worsening_db = +40.84 dB` である。よって、この方式も採用しない。

この棄却は、固定整相後段に置く SLC / MVDR / LCMV / GSC と、少数点 null 制約で BL 全体の sidelobe envelope を下げようとする方式に限定する。時間領域で実装される方式全体を棄却する意味ではない。特に、固定整相前段の shading / active aperture 再設計、channel×tap の broadband FIR beamformer、dense sector envelope 制約付きの minimax / inequality LCMV は、まだ同じ評価条件では潰し切れていない。

以降の主検討は、STFT の周波数別重み設計だけに固定しない。BL 全体の sidelobe 低減を要求する場合は、まず固定整相側の shading / active aperture 設計を評価し、必要に応じて周波数別または dense sector envelope 制約付きの重み設計へ進む。リアルタイム処理量を抑えるため、重み更新や sector 探索は学習側へ寄せ、リアルタイム側は固定整相、保存済み FIR、学習済み重みの適用を中心に設計する。

### 15.14 結論の限定表現

現時点の結論は次のように限定して扱う。

```text
現行の固定整相後段 SLC、および時間領域 channel×tap MVDR / LCMV / GSC /
少数点 sector-constrained LCMV は、target 保護や特定干渉方位の null 形成には
有効な条件がある。

しかし、固定整相後 BL 全体の guard 外 sidelobe peak、第一副極、
最大局所悪化量を同時に改善する方式としては、現評価条件では成立していない。

したがって、BL 全体の sidelobe 低減を要求する場合は、後段キャンセラではなく、
固定整相側の shading / active aperture 設計、または周波数別・sector envelope
制約付きの重み設計へ進む。
```

この整理により、SLC は「BL 全体の sidelobe reducer」としては不採用だが、「特定方位・特定周波数の target beam leakage を下げる局所キャンセラ」としては用途を残す。方式選定では `role = BL_sidelobe_reducer` と `role = local_leakage_canceller` を分け、前者では第一副極、guard 外 peak、最大悪化量を必須にし、後者では target 保護、marker 低減、悪化時 fallback を必須にする。

### 15.15 未検討事項

現資料で未検討または追加評価が必要な項目は次である。

| 項目 | 未検討内容 | 次に見る指標 |
|---|---|---|
| 固定整相側の shading / active aperture 再設計 | Kaiser / Taylor / Dolph-Chebyshev / DPSS 的 shading、周波数別 active subset、mainlobe 幅と SNR loss の trade-off をまだ網羅していない。 | first_sidelobe_reduction_db, guard_outside_peak_delta_db, 3dB beamwidth, N_eff, expected_snr_gain_db |
| dense sector envelope 制約 | 少数点 null は評価済みだが、guard 外 sector 全体に `max |w^H a(theta, f)| <= bound` を課す minimax / inequality 設計は未評価である。 | max_sidelobe_db, 95/99 percentile sidelobe, integrated sidelobe level, WNG |
| broadband FIR beamformer | channel×tap 重みを複数周波数・複数方位の制約で offline 設計し、リアルタイムでは FIR 適用だけにする方式は未評価である。 | passband ripple, stopband sector bound, tap_len, runtime_factor, target mismatch robustness |
| train / test 分離 | 学習に使った干渉方位・周波数でだけ null が深くなる過適合をまだ十分に分離できていない。 | train/test marker reduction gap, off-grid peak worsening, unseen interferer reduction |
| dense angle grid / off-grid 補間 | 151 本 beam grid 上の評価だけでは null 深さや peak 位置を見誤る可能性がある。 | dense_grid_guard_peak, exact_source_response, local_peak_interpolation |
| target mismatch robustness | 方位誤差、音速誤差、FIR 残留誤差、target 近接条件で self-nulling しないか未評価である。 | target_power_delta_db under mismatch, constraint_response_error_db, fallback_rate |
| WNG / DI | sidelobe を下げても白色雑音増幅や directivity 低下が大きい重みを棄却する基準が不足している。 | white_noise_gain_db, directivity_index_delta_db, output_noise_rms_db20 |
| multi-target blocking | 複数 target または保持方位がある条件で、単一 target blocking が別 target を消さないか未評価である。 | multi_target_power_delta_db, reference_target_leakage_db, constraint_matrix_rank |
| beammap deconvolution | CLEAN / DAMAS / NNLS 系は BL/FRAZ 表示改善として未検討である。ただし波形出力の物理的 sidelobe 低減とは分ける。 | display_sidelobe_reduction_db, source_position_error_deg, waveform_output_applicability |

次の優先順位は、固定整相側の shading / active aperture sweep、SLC の局所 leakage canceller としての再分類、dense sector / minimax LCMV の offline 評価、broadband FIR beamformer の順とする。STFT / subband 方式へ進む場合も、bin 別 SLC を単純に並べるだけではなく、周波数別 guard、WNG、sector envelope 制約を同時に設計する。

