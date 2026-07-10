# 固定遅延+差分補正 MVDR 方式実装検討設計書

## 1. 本書の目的

本書は、`fixed_delay_diff_mvdr_basic_design.md` と
`fixed_delay_diff_mvdr_detailed_design.md` を確認したうえで、固定遅延+差分補正
MVDR を spflow 上で扱うための方式、実装単位、評価条件、残課題を 1 箇所へ追記していく
検討設計書である。

本書では「後で正しくする実装」として扱わない。検討対象は、次の式を
満たす正式な方式である。

```text
q[k] = w0[k] - w_mvdr[k]
y[k] = w0[k]^H X[k] - q[k]^H X[k]
     = w_mvdr[k]^H X[k]
```

ただし、`q[k]` を有限長 FIR 化した後は近似誤差が生じ得るため、
`w0[k] - q_fir[k]`、`q_fir[k]^H a[k]`、`w_final[k]^H a[k]` を評価対象に含める。

## 2. 確認した設計書

確認対象は以下である。

- `doc/fixed_delay_diff_mvdr/fixed_delay_diff_mvdr_basic_design.md`
- `doc/fixed_delay_diff_mvdr/fixed_delay_diff_mvdr_detailed_design.md`

設計の中核は、固定整相枝を主経路として残し、MVDR 重みとの差分だけを補正 FIR にする
構成である。これは明示的な blocking matrix を持たない GSC と解釈できる。

注意点として、既存詳細設計書には段階導入を示す表現が残っている。本検討書では、その表現を
採用条件として扱わず、実装済み部品と評価済み条件だけを方式検討の根拠にする。

## 3. 実装方針

今回追加した実装単位は、既存の固定遅延整相器や既存 MVDR API を置き換えない。
固定遅延+差分補正 MVDR 固有の shape、共役規約、fallback、診断量を閉じ込めるため、
専用モジュールを追加した。

追加ファイル:

```text
src/spflow/beamforming/fixed_delay_diff_mvdr.py
tests/beamforming/test_fixed_delay_diff_mvdr.py
```

公開 API には `src/spflow/beamforming/__init__.py` から接続した。

## 4. 実装済み部品

### 4.1 短 FFT 共分散更新

`ShortFFTCovarianceAccumulator` は、入力 `x` shape `[n_ch, n_sample]` を
`block_size` ごとに区切り、FFT axis を時間 sample 軸として `X[k]` を作る。

共分散は次式で更新する。

```text
R_b[k] = alpha R_{b-1}[k] + (1 - alpha) X_b[k] X_b[k]^H
alpha = exp(-block_duration_sec / covariance_time_constant_sec)
```

`block_size` 未満の端数はゼロ詰めせず内部バッファに保持する。端数をゼロ詰めして統計へ
入れると、短時間だけ入力 power を過小評価し、MVDR 重みの不安定化要因になるためである。

### 4.2 対角ローディング付き MVDR

`LoadedMVDRWeightDesigner` は、各 bin で次を解く。

```text
R_load[k] u[k] = a[k]
w_mvdr[k] = u[k] / (a[k]^H u[k])
```

対角ローディングは次式で入れる。

```text
R_load[k] = R[k] + epsilon[k] I
epsilon[k] = diagonal_loading_ratio * trace(R[k]).real / n_ch
```

`solve` 失敗、分母 floor 未満、非有限重みの場合は fallback する。前回安定重みがあれば前回値、
なければ固定整相重みを使う。異常な MVDR 更新で target を削るより、固定整相へ退避する方が
安全側である。

### 4.3 固定整相重み

`design_distortionless_fixed_weights` は、ステアリング `a[k]` から
次の固定整相重みを作る。

```text
w0[k] = a[k] / (a[k]^H a[k])
```

これにより `w0[k]^H a[k] = 1` を満たす。入力 shape は `[n_bin, n_ch]` である。

### 4.4 差分補正 FIR 設計

`DifferenceCorrectionFIRDesigner` は、数式上の差分重みを次のように作る。

```text
q_weight_freq[k] = w0[k] - w_mvdr[k]
```

spflow の適用側は `w^H x` の規約を使うため、信号へ掛ける周波数応答は
`conj(q_weight_freq)` とし、IFFT で `q_apply_taps[ch, tap]` を得る。

FIR 化後は再 FFT し、次を診断する。

- `w0[k]^H a[k]`
- `w_mvdr[k]^H a[k]`
- `w_final[k]^H a[k]`
- `q_fir[k]^H a[k]`
- `q_weight_freq[k] - q_fir[k]`

### 4.5 差分補正 FIR 適用

`DifferenceCorrectionFIR` は、次式の因果 FIR を chunk 分割に依存しない形で実行する。

```text
z[n] = sum_ch sum_l h[ch,l] x[ch,n-l]
```

初回 chunk の過去 sample はゼロとし、以後は `fir_taps - 1` sample を履歴として保持する。

## 5. Beamforming Evaluation に基づく評価計画

Beamforming Evaluation スキルの基準に従い、本方式は単一の BL/FRAZ/BTR 図だけでは
採否判断しない。

### 5.1 適用する評価 pattern

固定整相主経路の確認:

- `fixed_beam_single_source`
- 必須: peak position、sidelobe margin、FRAZ/BTR consistency、input/output level consistency

target 保護の確認:

- `slc_target_only`
- 必須: mainlobe preservation、target leakage components、waveform integrity、
  input/output level consistency

同一周波数干渉の確認:

- `slc_same_frequency_interference`
- local leakage canceller として扱う
- 必須: target leakage components、mainlobe preservation、covariance health

実時間性の確認:

- `slc_runtime`
- 必須: runtime budget、covariance health、array consistency

本方式は source-preserving scan としてではなく、保護 target beam の local leakage canceller
としてまず評価する。したがって、全方位 scan 上の interferer peak を必ず消すことを
採否条件にはしない。

### 5.2 dB 表記

絶対値に見えるレベルは `dB re ...` の基準量を付ける。

- シミュレーション入力基準: `dB re input RMS`
- BTR frame 正規化: `dB re frame max`
- mainlobe / sidelobe margin: `dB re mainlobe peak`
- before/after 差分: `dB re before level`

## 6. 今回実施した検証

追加テストで確認した内容は以下である。

- 短 FFT 共分散が block 境界でのみ更新され、Hermitian を保つこと
- 白色共分散で MVDR が固定整相重みと一致すること
- 特異共分散で固定整相重みへ fallback すること
- `q = w0 - w_mvdr`、`w0 - q = w_mvdr`、`q^H a = 0` が成立すること
- `fir_taps == n_bin` 条件で IFFT/FFT round trip により差分重みが再構成されること
- 差分補正 FIR が chunk 分割に依存しないこと
- impulse 入力で FIR tap 順が設計どおりであること

実行結果:

```text
.venv\Scripts\python.exe -m pytest -q tests\beamforming\test_fixed_delay_diff_mvdr.py
5 passed

.venv\Scripts\python.exe -m ruff check src\spflow\beamforming\fixed_delay_diff_mvdr.py tests\beamforming\test_fixed_delay_diff_mvdr.py src\spflow\beamforming\__init__.py
All checks passed

.venv\Scripts\pyright.exe -p artifacts\beamforming\fixed_delay_diff_mvdr\pyright_check\pyrightconfig.json
0 errors, 0 warnings, 0 informations
```

## 7. Pyright / Pylance 自己レビュー

今回追加した Python コードでは、以下を確認した。

- NumPy scalar を `require` の `bool` 引数へ渡す箇所は `bool(...)` で明示変換した。
- `Optional` な前回重み `_previous_weights` は `is not None` で分岐した。
- フラグで戻り値型を変えず、結果は dataclass で固定した。
- 入出力の NumPy 配列には `NDArray` 型注釈を付けた。
- `typing.cast` で型を握りつぶしていない。
- shape validation を各公開入口に入れた。

## 8. 未実施の評価と次章で追記する内容

今回の検証は、数式対応と時間領域 FIR 適用の単体確認である。方式採用判断にはまだ不足がある。

次に追記すべき評価は以下である。

1. target-only 波形で `z[n]` が十分小さく、`y[n]` が `y0[n]` を保つこと
2. interferer-only で保護 target beam への漏れ込みが固定整相より下がること
3. mixed 条件で target power delta と interferer reduction を分けて記録すること
4. 係数更新周期 1 秒で BTR 上の縦縞や waveform jump が出ないこと
5. `fir_taps != n_bin` または実固定遅延応答由来 `w0` の場合の `q` 近似誤差を測ること
6. runtime budget と loaded covariance condition number を記録すること

上記が終わるまで、BL/FRAZ/BTR の見た目だけで方式採用とは判断しない。


## 9. 小数遅延 FIR 事前計算方式

今回の方式検証では、小数遅延 FIR を実行時に設計しない。小数遅延範囲
`-0.5 sample` から `0.5 sample` を 51 パターンに等分し、各パターンについて
128 tap の窓付き sinc FIR を事前計算する。

実装上の標準条件は以下である。

```text
小数遅延範囲: -0.5 sample 〜 0.5 sample
小数遅延パターン数: 51
小数遅延 grid 間隔: 0.02 sample
FIR tap 数: 128
FIR バンク shape: [51, 128]
```

この条件は `design_standard_fractional_delay_filter_bank()` で固定した。
内部的には既存の `FractionalDelayFilterBank` を使い、`frac_grid` shape `[51]` と
`frac_filters` shape `[51, 128]` を保持する。

各チャネル・各整相方位では、`DelayTable.delay_frac[ch, beam]` に最も近い
`frac_grid` の index を `frac_filter_index[ch, beam]` として選ぶ。時間領域固定整相では、
次の順に処理する。

```text
1. チャネル ch、整相方位 beam ごとの整数遅延 delay_int[ch, beam] を適用する。
2. frac_filter_index[ch, beam] で 51 本の FIR バンクから 1 本を選ぶ。
3. 選択した 128 tap FIR を、そのチャネル・その整相方位の信号へ畳み込む。
4. チャネル方向に合成して固定整相出力 y0[beam, n] を作る。
```

この方式では、同じチャネルでも整相方位が変われば `delay_frac[ch, beam]` が変わるため、
選択される FIR も変わり得る。したがって、FIR 適用単位は channel 単位ではなく
`channel × beam` 単位である。

## 10. 固定整相重み w0 の実応答化

差分補正 MVDR の差分重みは `q[k] = w0[k] - w_mvdr[k]` で定義する。
この `w0[k]` が理想 steering だけから作られると、時間領域主経路で実際に使う
整数遅延・小数遅延 FIR・チャネル平均の応答とずれる。

そのため、今回の実装では `design_fixed_delay_fractional_weights_from_delay_table()` を追加し、
`DelayTable` と事前計算済み FIR バンクから、固定整相主経路の実周波数応答に対応する
`w0` を生成できるようにした。

返す shape は以下である。

```text
w0 shape: [n_bin, n_beam, n_ch]
axis=0: 周波数 bin
axis=1: 整相方位 beam
axis=2: センサ channel
```

時間領域主経路が入力 `X[ch,k]` に掛ける応答を `G[ch, beam, k]` とすると、
spflow の重み適用規約は `y = w^H X` であるため、返す重みは次の関係を満たす。

```text
conj(w0[k, beam, ch]) = G[ch, beam, k]
```

ここで `G` には、整数遅延の位相回転、選択済み小数遅延 FIR の周波数応答、
およびチャネル平均時の `1/n_ch` を含める。

## 11. 追記検証結果

今回追加した検証は、Beamforming Evaluation のうち固定整相単一音源の前提確認、
target 保護評価の前段となる `w0` 整合性確認、waveform integrity の chunk 境界確認に対応する。

追加で確認した内容は以下である。

- 標準小数遅延 FIR バンクが 51 パターン、128 tap、grid 間隔 0.02 sample であること。
- `-0.5 sample`、`0.0 sample`、`0.5 sample` がそれぞれ index 0、25、50 に対応すること。
- `delay_frac[ch, beam]` に従い、channel×beam ごとに FIR index が選ばれること。
- 固定整相重み `w0[k, beam, ch]` が、選択済み FIR の実周波数応答と整数遅延位相を含むこと。
- `FractionalDelayAndSumBeamformer` が、channel×beam ごとに選択済み 128 tap FIR を適用すること。
- impulse 入力で、整数遅延位置から選択済み FIR tap がそのまま channel 別整相出力に現れること。

実行結果は以下である。

```text
.venv\Scripts\python.exe -m pytest -q tests\beamforming\test_fixed_delay_diff_mvdr.py tests\beamforming\test_evaluation_criteria.py
20 passed

.venv\Scripts\python.exe -m ruff check src\spflow\beamforming\fixed_delay_diff_mvdr.py tests\beamforming\test_fixed_delay_diff_mvdr.py src\spflow\beamforming\__init__.py
All checks passed

.venv\Scripts\pyright.exe -p artifacts\beamforming\fixed_delay_diff_mvdr\pyright_check\pyrightconfig.json
0 errors, 0 warnings, 0 informations
```

## 12. 次に検証する評価項目

小数遅延 FIR の選択と `w0` 実応答化は確認できた。次は、Beamforming Evaluation の基準に従い、
BL/FRAZ/BTR だけで採否判断せず、少なくとも以下を分けて評価する。

1. `fixed_beam_single_source`: 事前計算小数遅延 FIR を使った固定整相の peak 方位、
   sidelobe margin、FRAZ/BTR consistency、input/output level consistency。
2. `slc_target_only`: `q^H a` と target-only 波形で、差分補正枝が target を自己消去しないこと。
3. `slc_same_frequency_interference`: interferer-only と mixed を分け、protected target beam への
   漏れ込み低減、mainlobe preservation、covariance health を確認すること。
4. `slc_runtime`: 51×128 の小数遅延 FIR と 128 tap 差分補正 FIR を含めた runtime budget を測ること。

## 13. review_pack 生成と結果分析

### 13.1 review_pack の出力構成

方式検証結果は以下に保存した。

```text
artifacts/beamforming/fixed_delay_diff_mvdr/review_pack/
```

必須ファイルと役割は以下である。

```text
review_index.md        scenario の目的、source 条件、mask 種別、判定要約、図と npz への相対 path
scenario_summary.csv   source / non-source 分離 metric と fallback / runtime の一覧
worst_cases.csv        worst top10、negative/watch、mask count、MVDR と FIR 化後の差分が大きい行
figures/<scenario>/    BL overlay、BL delta、FRAZ delta、BTR panel
data/<scenario>.npz    BL / FRAZ / BTR 描画前配列
```

mask は `source_guard_3deg_non_source_complement` とした。source 方位の ±3 deg を
source mask とし、その補集合を non-source sector とする。BL / FRAZ は `dB re input RMS`、
BTR は `dB re frame max` である。BTR は frame max 正規化なので、抑圧量の採否には使わず、
source track の連続性確認だけに使う。

### 13.2 review_pack 生成時に検出した実装不整合

最初の評価では、差分補正枝 `q` の target blocking 応答が大きく、target-only でも
`diff_mvdr_fir128` の target mainlobe が低下していた。方式が上手くいかない場合の確認として、
固定主経路の target 応答を直接調べたところ、事前計算小数遅延 FIR の群遅延により
`w0^H a` は振幅ほぼ 1 だが複素位相を持つことが分かった。

差分枝が target を通さない条件は次である。

```text
q = w0 - w_mvdr
q^H a = w0^H a - w_mvdr^H a = 0
```

したがって MVDR の distortionless 目標は常に `1+0j` ではなく、固定主経路の複素応答
`g = w0^H a` に合わせる必要がある。`w^H a = g` を満たす重みは、MVDR 解に `conj(g)` を
掛ける形になるため、`LoadedMVDRWeightDesigner` をこの制約へ修正した。

この修正を固定するため、白色共分散で `w_mvdr^H a = w0^H a` と
`q^H a = 0` を確認する単体テストを追加した。

### 13.3 評価 scenario と最終判定

評価 scenario は以下の 3 条件である。

1. `target_only_20deg_1536hz`: target-only で差分補正枝が target を自己消去しないこと。
2. `same_frequency_interferer_60deg`: 同一周波数 interferer の target beam への漏れ込み低減。
3. `different_frequency_interferer_75deg`: 異周波 interferer 条件で target 維持と副作用確認。

最終結果の要点は以下である。

```text
scenario                            method             status
 target_only_20deg_1536hz            diff_mvdr_fir128   pass
 same_frequency_interferer_60deg     diff_mvdr_fir128   watch_low_leakage_reduction
 different_frequency_interferer_75deg diff_mvdr_fir128  watch_non_source_worsening
```

`diff_mvdr_fir128` は target-only では target mainlobe delta が -0.322 dB に収まり、
source 自己消去は抑えられた。一方、同一周波数 interferer では leakage reduction が
-0.355 dB で、固定整相より漏れ込みがわずかに悪化した。異周波 interferer では
leakage reduction が -0.187 dB であり、さらに `max_local_worsening_db_gated` が
12.566 dB と大きいため、non-source 局所悪化として watch 判定にした。

`mvdr_oracle` は interferer leakage reduction が 28 dB から 35 dB 程度と大きいが、
target mainlobe delta が -1 dB を超えて低下するため fail とした。これは target 方位が
scan grid の中心からずれている条件で、MVDR が強い null / suppression を作ることによる
mainlobe preservation 不足として扱う。

### 13.4 現時点の方式判断

51×128 小数遅延 FIR の事前計算、channel×beam ごとの FIR 選択、固定主経路 `w0` の実応答化、
および `w_mvdr^H a = w0^H a` の複素制約は実装・単体検証済みである。

ただし、128 tap の差分補正 FIR 化後は、MVDR oracle が持つ interferer 抑圧が十分に再現されていない。
特に interferer 条件で leakage reduction が負になるため、この構成のまま採用判断には進めない。
次に確認すべき点は以下である。

1. 評価周波数 17 bin から 128 tap FIR へ変換する際の周波数グリッド不足と reconstruction error。
2. 差分補正 FIR の tap 数増加、または設計周波数 bin の増加による leakage reduction 改善。
3. target 方位が beam grid 中心から外れた場合の mainlobe preservation と guard 幅の感度。
4. fixed_baseline を fallback として残したまま、watch 条件で差分補正を無効化する運用条件。

### 13.5 実行した検証

```text
.venv\Scripts\python.exe -m pytest -q tests\beamforming\test_fixed_delay_diff_mvdr.py tests\beamforming\test_evaluation_criteria.py
21 passed

.venv\Scripts\python.exe -m ruff check src\spflow\beamforming\fixed_delay_diff_mvdr.py tests\beamforming\test_fixed_delay_diff_mvdr.py examples\beamforming\build_fixed_delay_diff_mvdr_review_pack.py src\spflow\beamforming\__init__.py
All checks passed

.venv\Scripts\pyright.exe -p artifacts\beamforming\fixed_delay_diff_mvdr\review_pack_pyright_check\pyrightconfig.json
0 errors, 0 warnings, 0 informations

.venv\Scripts\python.exe examples\beamforming\build_fixed_delay_diff_mvdr_review_pack.py
saved review index / scenario summary / worst cases
```

## 14. 高周波条件での追加評価

### 14.1 追加した評価条件

「もう少し高周波」の確認として、target 4.096 kHz、interferer 5.120 kHz を含む
3 scenario を review_pack に追加した。既存の 1.536 kHz / 2.304 kHz 条件と同じ package で
比較するため、評価周波数軸は以下へ拡張した。

```text
frequency_hz: 768 Hz 〜 6144 Hz、128 Hz 間隔
n_freq: 43
```

追加 scenario は以下である。

```text
target_only_20deg_4096hz
same_frequency_interferer_60deg_4096hz
different_frequency_interferer_75deg_4096_5120hz
```

出力 package は同じ `artifacts/beamforming/fixed_delay_diff_mvdr/review_pack/` に更新した。
図は 6 scenario × 4 種で 24 PNG、描画前 npz は 6 個である。

### 14.2 高周波条件の結果

高周波側の `diff_mvdr_fir128` は以下の結果である。

```text
scenario                                        status             target_mainlobe_delta_db  leakage_reduction_db
 target_only_20deg_4096hz                       fail_target_loss                 -2.915                  0.000
 same_frequency_interferer_60deg_4096hz         fail_target_loss                 -3.126                -18.776
 different_frequency_interferer_75deg_4096_5120hz fail_target_loss               -2.967                -16.537
```

target-only でも target mainlobe が約 2.9 dB 低下しているため、target 保護条件を満たさない。
interferer 条件では leakage reduction が負であり、fixed_baseline より protected target beam への
漏れ込みが悪化している。異周波条件では `max_local_worsening_db_gated` も 16.360 dB と大きい。

MVDR oracle は高周波条件で interferer leakage reduction が 24.8 dB から 38.4 dB 程度出るが、
target mainlobe delta が -13 dB から -19 dB 程度まで低下するため、これも採用可能な参照出力ではない。
今回の oracle は「抑圧できる上限」を見るための比較であり、mainlobe preservation を満たす方式ではない。

### 14.3 分析

高周波条件では、128 tap 差分補正 FIR の近似不足がより強く出た。`q_reconstruction_rms_error` は
高周波 scenario で約 1.1 から 1.35 であり、差分重み `q = w0 - w_mvdr` を時間領域 FIR として
十分に再構成できていない。

また、周波数軸を 6.144 kHz まで広げたことで、低周波 scenario も以前の 17 bin 評価とは異なる
広帯域 FIR 設計になった。この条件では低周波側も `watch_non_source_worsening` になり、
広い設計帯域を 128 tap に押し込むほど target 保護と non-source 副作用が悪化する傾向が見えた。

### 14.4 現時点の判断

4.096 kHz / 5.120 kHz の追加評価では、`diff_mvdr_fir128` は性能が出ていない。
特に高周波では target-only から fail しており、interferer 抑圧以前に target 保護が成立していない。

次の検討では、以下のどちらかを分けて評価する必要がある。

1. 周波数帯域を狭く区切り、その帯域内だけで差分 FIR を設計する。
2. 128 tap より長い差分 FIR、または周波数領域処理のまま適用する方式を比較する。

fixed_baseline は引き続き fallback として残す。

## 15. 実装の数式化と理想式との比較

### 15.1 固定主経路 fixed の式

入力の周波数成分を `X_ch(f)`、channel `ch`、beam `b` の整数遅延を `d_int[ch,b]`、
選択済み小数遅延 FIR を `h_frac[ch,b,l]` とする。固定主経路が入力へ掛ける実応答は次である。

```text
G_ch,b(f) = (1/Nch) exp(-j 2π f d_int[ch,b] / fs)
            Σ_l h_frac[ch,b,l] exp(-j 2π f l / fs)
```

spflow の重み適用規約は `y = w^H X` であるため、固定主経路の重みは次である。

```text
w0_ch,b(f) = conj(G_ch,b(f))
y_fixed,b(f) = Σ_ch conj(w0_ch,b(f)) X_ch(f)
             = Σ_ch G_ch,b(f) X_ch(f)
```

この式は `design_fixed_delay_fractional_weights_from_delay_table()` の実装と一致している。

### 15.2 MVDR oracle の理想式と実装式

保護 steering を `a_b(f)`、loaded covariance を `R(f)` とする。固定主経路の保護方向応答は
一般に `1+0j` ではなく、小数遅延 FIR の群遅延を含む複素値になる。

```text
g_b(f) = w0_b(f)^H a_b(f)
```

差分枝 `q = w0 - w_mvdr` が保護方向を通さない条件は次である。

```text
q_b(f)^H a_b(f) = w0_b(f)^H a_b(f) - w_mvdr,b(f)^H a_b(f) = 0
```

したがって MVDR は `w^H a = g` を制約にする必要がある。標準 MVDR 解
`u = R^-1 a`、`den = a^H R^-1 a` を使うと、重みは次である。

```text
w_mvdr(f) = conj(g_b(f)) R(f)^-1 a_b(f) / (a_b(f)^H R(f)^-1 a_b(f))
```

実装は `desired_response = _weight_response(w0, a)` とし、`conj(desired_response)` を
MVDR 解に掛けているため、この式と一致している。

### 15.3 差分補正 FIR の理想式

差分重みを次で定義する。

```text
q_ch,b(f_k) = w0_ch,b(f_k) - w_mvdr,ch,b(f_k)
```

時間領域補正枝は `z = q^H X` を作る必要があるため、実際に FIR が表すべき周波数応答は
`conj(q)` である。補正 FIR 係数を `h_q[ch,l]` とすると、理想式は次である。

```text
Σ_l h_q[ch,l] exp(-j 2π f_k l / fs) ≈ conj(q_ch,b(f_k))
```

行列で書くと、`V[k,l] = exp(-j 2π f_k l / fs)` に対して次を解く。

```text
V h_q[:,ch] ≈ conj(q[:,ch])
```

### 15.4 旧実装との差分

旧実装では、`conj(q)` を
p.fft.ifft(..., n=128)` へ直接渡していた。この式は暗黙に
`conj(q[m])` が DFT bin `m`、つまり `f_m = m fs / 128` の応答であることを仮定する。
しかし review_pack の評価周波数は `768 Hz, 896 Hz, ...` のような任意 Hz 配列であり、
DFT bin `0, 1, ...` ではない。

そのため旧実装は次の誤った対応になっていた。

```text
実際の評価周波数: f_k = 768 + 128 k [Hz]
旧実装が仮定した周波数: f_k = k fs / 128 [Hz]
```

これは理想式と一致しない。実装誤りと判断し、`DifferenceCorrectionFIRDesigner` を
任意周波数 least-squares 設計へ修正した。

### 15.5 修正後の確認結果

修正後は、`q_reconstruction_rms_error` が全 scenario で約 `1.1e-6` まで低下した。
`q_blocking_max_db` と `target_response_error_db` も約 `-96 dB` から `-99 dB` であり、
設計周波数上では差分 FIR が MVDR oracle をほぼ再現している。

```text
q_reconstruction_rms_error: 約 1.1e-6
q_blocking_max_db:          約 -96 dB 〜 -99 dB
target_response_error_db:   約 -96 dB 〜 -99 dB
```

したがって、修正後に残る `fail_target_loss` は差分 FIR 近似の失敗ではなく、
MVDR oracle 自体の問題である。

### 15.6 残る問題の数式上の原因

現在の scan 評価では、各 beam `b` の MVDR は beam center の steering `a_b` を保護する。
一方、source truth は `20 deg` であり、nearest beam は完全には一致しない。

```text
保護する方向: a_b(f)       beam center
評価する source: a_src(f)   20 deg
```

MVDR が保証するのは次である。

```text
w_mvdr,b(f)^H a_b(f) = w0_b(f)^H a_b(f)
```

しかし評価で見ている target は次である。

```text
w_mvdr,b(f)^H a_src(f)
```

`a_b(f) != a_src(f)` の場合、特に高周波では steering の角度差が大きな位相差になり、
MVDR が target truth を削る。このため、差分 FIR が oracle を正しく再現しても、
`target_mainlobe_delta_db` は大きく負になる。

次の検証では、以下を分ける必要がある。

1. protected steering を source truth に合わせた target 保護評価。
2. scan 表示として beam center ごとに処理する評価。
3. target が beam center から外れた場合に MVDR を無効化する guard / fallback 条件。

この切り分けを行わないと、差分 FIR の実装誤差と MVDR oracle の steering mismatch を混同する。

## 16. MVDR 保護式の適用と 512 tap 差分 FIR の再評価

### 16.1 修正内容

前章の切り分けに従い、protected target beam では beam center steering ではなく
source truth steering を MVDR 制約に使うよう review_pack 生成を修正した。
これにより、評価対象 target に対して次の式を直接保証する。

```text
w_mvdr^H a_src = w0^H a_src
q^H a_src = 0
```

また、差分 FIR は 512 tap に増やした。実装名は `diff_mvdr_fir512` とした。
差分 FIR は任意周波数 least-squares 設計のままなので、設計周波数上では
`V h ≈ conj(q)` を解く。

### 16.2 再評価結果

512 tap 条件の `diff_mvdr_fir512` は以下である。

```text
scenario                                          status   target_mainlobe_delta_db  leakage_reduction_db
 target_only_20deg_1536hz                         pass                    0.000                  0.000
 same_frequency_interferer_60deg                  fail_target_loss        -1.367                 63.353
 different_frequency_interferer_75deg             pass                    0.000                 64.109
 target_only_20deg_4096hz                         pass                    0.000                  0.000
 same_frequency_interferer_60deg_4096hz           pass                    0.143                 66.636
 different_frequency_interferer_75deg_4096_5120hz pass                    0.000                 64.854
```

高周波 4.096 kHz / 5.120 kHz 条件はすべて pass した。
`q_reconstruction_rms_error` は約 `2e-15`、`q_blocking_max_db` は約 `-265 dB` から
`-269 dB` であり、設計周波数上では差分 FIR が MVDR oracle を数値精度まで再現している。

低周波の同一周波 interferer 条件だけは `target_mainlobe_delta_db = -1.367 dB` のため、
現在の -1 dB 閾値では fail とした。ただし leakage reduction は 63.353 dB であり、
抑圧自体は大きく改善している。

### 16.3 tap 数に関する判断

今回の 512 tap 条件では、設計周波数上の再構成誤差は数値精度まで下がった。
したがって、現在見ている BL/FRAZ の設計周波数上では、FIR tap 不足が主要因ではない。

ただし、時間波形としての帯域内連続性、設計周波数間の応答、係数更新時の waveform integrity は
まだ別途評価が必要である。512 tap は runtime_factor が約 1.6 から 2.1 になっており、
実時間運用では runtime budget と遅延も評価対象になる。

## 17. near-frequency interferer 条件への置換

### 17.1 置換理由

完全に同一周波数の target と interferer を同一 phase 付きの狭帯域成分として扱うと、
2 source が時間的に分離しない coherent な合成波になる。この条件では、局所漏れ込みキャンセルの
評価というより、合成された単一波面に近い場を評価してしまう可能性がある。

そのため、同一周波数 scenario は削除し、interferer 周波数を target から 0.1 Hz ずらした
near-frequency 条件へ置き換えた。周波数差なので単位は Hz とした。

```text
near_frequency_interferer_60deg_1536p1hz: target 1536.0 Hz, interferer 1536.1 Hz
near_frequency_interferer_60deg_4096p1hz: target 4096.0 Hz, interferer 4096.1 Hz
```

0.1 Hz offset を評価周波数軸で丸めないため、`1536.1 Hz` と `4096.1 Hz` を
`frequency_hz` に明示的に追加した。

### 17.2 再評価結果

near-frequency 条件を含む `diff_mvdr_fir512` の結果は以下である。

```text
scenario                                          status  target_mainlobe_delta_db  leakage_reduction_db
target_only_20deg_1536hz                          pass                   0.000                  0.000
near_frequency_interferer_60deg_1536p1hz          pass                   0.000                 63.354
different_frequency_interferer_75deg              pass                   0.000                 64.109
target_only_20deg_4096hz                          pass                   0.000                  0.000
near_frequency_interferer_60deg_4096p1hz          pass                   0.000                 66.638
different_frequency_interferer_75deg_4096_5120hz  pass                   0.000                 64.854
```

完全同一周波数で残っていた低周波条件の `target_mainlobe_delta = -1.367 dB` は、
near-frequency 条件では解消した。したがって、前回の fail は方式の FIR tap 不足ではなく、
coherent な同一周波数 2 source を静的に合成して評価したことによる条件依存の影響と判断する。

### 17.3 注意点

0.1 Hz offset は 1 秒程度の短い BTR では位相差の進みが小さい。より実信号に近い確認では、
解析時間長、STFT 周波数分解能、source 位相の時間変化を明示し、完全 coherent 条件と
near-coherent 条件を別 scenario として管理する必要がある。

## 18. frequency offset sweep による source 分解条件

### 18.1 問題の整理

BL overlay は target 周波数の 1 断面を表示するため、interferer 周波数を target から
少しでもずらすと、target 周波数断面には interferer peak が表示されにくい。
両方の source peak を確認するには、target は target 周波数で、interferer は interferer 周波数で
別々に source visibility を評価する必要がある。

今回の sweep では、`diff_mvdr_fir512` について次を評価した。

```text
target_frequency_hz:     1536.0 Hz, 4096.0 Hz
interferer_azimuth_deg:  60.0 deg
frequency_offset_hz:     0.0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128
visibility threshold:    fixed source peak から -1 dB 以内
```

成果物は以下である。

```text
artifacts/beamforming/fixed_delay_diff_mvdr/frequency_offset_sweep/frequency_offset_sweep.csv
artifacts/beamforming/fixed_delay_diff_mvdr/frequency_offset_sweep/frequency_offset_sweep.md
artifacts/beamforming/fixed_delay_diff_mvdr/frequency_offset_sweep/frequency_offset_sweep.png
```

### 18.2 結果

解析周波数軸へ offset 周波数を明示的に追加した場合、0.05 Hz や 0.1 Hz でも
数式上は別周波数として扱える。この条件では target peak と interferer peak はどちらも
fixed から -1 dB 以内で保持された。

ただし、1 秒観測の STFT 周波数 bin 幅は約 1 Hz である。したがって、実際に
1 秒程度の観測で周波数分解できる条件としては、offset 1 Hz 以上を分解可能とした。

```text
target 1536.0 Hz: 1 秒観測基準では offset 1.00 Hz 以上で分解可能
target 4096.0 Hz: 1 秒観測基準では offset 1.00 Hz 以上で分解可能
```

0.1 Hz offset は、長時間観測または明示的な狭帯域解析では別周波数として扱えるが、
1 秒 BTR / 1 秒 STFT 相当では分解可能とはみなさない。

### 18.3 BL overlay の扱い

添付図のような target 周波数の BL overlay だけを見ると、interferer が落ちているように見える。
しかし、この図は target 周波数断面だけなので、interferer を 1536.1 Hz に置いた時点で
interferer 周波数の peak を表示する図ではなくなる。

両 source の visibility を見る図として、source-frequency BL overlay を追加生成した。
これは target / interferer の各 source 真値周波数断面だけを FRAZ から取り出し、
方位ごとに最大値統合した BL である。target 周波数 1 断面の BL overlay と異なり、
near-frequency interferer が自分の周波数でピークとして残るかを確認できる。

```text
artifacts/beamforming/fixed_delay_diff_mvdr/review_pack/figures/near_frequency_interferer_60deg_1536p1hz/source_frequency_bl_overlay.png
artifacts/beamforming/fixed_delay_diff_mvdr/review_pack/figures/near_frequency_interferer_60deg_4096p1hz/source_frequency_bl_overlay.png
```

数値確認では、source-frequency BL 上の 20 deg 近傍 target peak と 60 deg 近傍 interferer peak は、
`fixed_baseline`、`mvdr_oracle`、`diff_mvdr_fir512` のいずれでもほぼ 0 dB re input RMS に残った。

BTR は `dB re frame max` なので、抑圧量ではなく track continuity の確認に限定する。

## 19. 単一周波数 + チャネル無相関雑音 sweep 評価

### 19.1 目的

干渉 source を持たない `fixed_beam_single_source` 条件で、
fixed delay + difference MVDR FIR512 の基礎的な beam peak、SNR gain、sidelobe margin を確認した。
この評価では、source は単一 tone、雑音はチャネル無相関白色雑音とする。

```text
source frequency: 768, 1024, 1536, 2048, 3072, 4096, 5120, 6144 Hz
source azimuth:   10, 20, 30, 45, 60, 75, 90, 105, 120, 135, 150, 170 deg
noise power:      1.0e-2 per channel
input SNR:        20.00 dB
array:            32 ch ULA, spacing 0.05 m
```

MVDR の共分散は source を含む `R = sigma_s^2 a_s a_s^H + sigma_n^2 I` とした。
この sweep は target 自己抑圧を防ぐ方式そのものの検証であるため、source を統計から除外しない。
ただし MVDR 制約には source truth steering `a_s` を使わず、各 scan beam の待ち受け方位
`theta_beam` に対応する steering `a(theta_beam, f)` を使う。
`mvdr_oracle` は真値方向 oracle ではなく、同じ待ち受け方位制約を周波数領域で直接解いた参照値である。

### 19.2 成果物

```text
artifacts/beamforming/fixed_delay_diff_mvdr/single_tone_noise_sweep/single_tone_noise_sweep_report.md
artifacts/beamforming/fixed_delay_diff_mvdr/single_tone_noise_sweep/single_tone_noise_sweep.csv
artifacts/beamforming/fixed_delay_diff_mvdr/single_tone_noise_sweep/worst_cases.csv
artifacts/beamforming/fixed_delay_diff_mvdr/single_tone_noise_sweep/data/single_tone_noise_sweep_arrays.npz
artifacts/beamforming/fixed_delay_diff_mvdr/single_tone_noise_sweep/figures/snr_gain_heatmap.png
artifacts/beamforming/fixed_delay_diff_mvdr/single_tone_noise_sweep/figures/azimuth_error_heatmap.png
artifacts/beamforming/fixed_delay_diff_mvdr/single_tone_noise_sweep/figures/sidelobe_margin_heatmap.png
artifacts/beamforming/fixed_delay_diff_mvdr/single_tone_noise_sweep/figures/sidelobe_margin_delta_heatmap.png
artifacts/beamforming/fixed_delay_diff_mvdr/single_tone_noise_sweep/figures/q_reconstruction_error_heatmap.png
artifacts/beamforming/fixed_delay_diff_mvdr/single_tone_noise_sweep/figures/representative_bl_overlay.png
```

sidelobe 評価では、固定の ±3 deg guard だけでは低周波の広い主ローブを non-source と誤判定する。
そのため、source mask は fixed BL の peak を含む -3 dB 主ローブ連続領域と source 方位 ±3 deg guard の和集合とした。

### 19.3 結果

```text
fixed_baseline:       fallback_baseline 96 rows
mvdr_oracle:          pass 24 rows, watch_snr_loss 24 rows, fail_source_loss 48 rows
diff_mvdr_fir512:     pass 24 rows, watch_snr_loss 24 rows, fail_source_loss 48 rows
```

主要 metric は以下である。

```text
fixed min SNR gain:                  15.014 dB
diff MVDR FIR512 min SNR gain:      -28.408 dB
diff MVDR FIR512 max az error:        1.039 deg
diff MVDR FIR512 min sidelobe margin: 39.529 dB
diff MVDR FIR512 max q RMS error:     3.575e-16
max source-to-waiting az error:       1.039 deg
```

差分 FIR512 は、待ち受け方位制約の周波数領域 MVDR 参照値と一致し、q 再構成誤差も十分小さい。
そのため今回の source loss は FIR 化誤差ではなく、待ち受け方位 `theta_beam` と source 真値方位の
steering mismatch に起因する。source が待ち受け方位に完全一致する 24 rows は全 pass で、
最小 source peak delta は -0.000 dB re fixed であった。一方、off-grid の 72 rows は全て watch/fail となり、
最悪で source peak delta は -23.402 dB re fixed、SNR gain は -28.408 dB まで落ちた。

この結果から、真値方向 oracle 制約の評価は方式の弱点を隠していたと判断する。
実運用を想定する場合、単一の待ち受け方位制約では高周波・off-grid source の自己抑圧を避けられない。

### 19.4 注意点

この sweep は単一 source + 無相関雑音の基礎確認であり、同一周波数干渉、近接周波数干渉、
source-preserving scan の複数 source visibility は評価対象外である。
BTR も生成していないため、時間方向 track continuity は別評価で確認する。




## 20. FIR128 と FIR512 の比較

### 20.1 評価条件

19 章と同じ `fixed_beam_single_source` 条件で、差分補正 FIR の tap 数だけを 512 から 128 に変更した。
MVDR 共分散は source を含む `R = sigma_s^2 a_s a_s^H + sigma_n^2 I` とし、
MVDR 制約には source truth steering ではなく各 scan beam の待ち受け方位 steering
`a(theta_beam, f)` を使う。

成果物は以下である。

```text
artifacts/beamforming/fixed_delay_diff_mvdr/single_tone_noise_sweep_fir128/single_tone_noise_sweep_fir128_report.md
artifacts/beamforming/fixed_delay_diff_mvdr/single_tone_noise_sweep_fir128/single_tone_noise_sweep_fir128.csv
artifacts/beamforming/fixed_delay_diff_mvdr/single_tone_noise_sweep_fir128/worst_cases_fir128.csv
```

### 20.2 結果

```text
diff_mvdr_fir128: pass 24 rows, watch_snr_loss 24 rows, fail_source_loss 48 rows
```

主要 metric は以下である。

```text
FIR128 min source peak delta:       -23.402 dB re fixed
FIR128 min SNR gain:                -28.408 dB
FIR128 max az error:                  1.039 deg
FIR128 max q RMS error:               4.374e-16
FIR512 max q RMS error:               3.575e-16
```

FIR128 は FIR512 と同じく、待ち受け方位制約の周波数領域 MVDR 参照値をほぼ数値精度で再現した。
FIR tap 数を 512 から 128 に戻しても、source loss の status、最悪 source peak delta、最悪 SNR gain は変わらない。
したがって、19 章で見えた off-grid source の自己抑圧は FIR tap 数不足ではなく、
source 真値方位と待ち受け方位の steering mismatch に起因する。

### 20.3 判断

この条件では FIR128 でも FIR512 でも方式上の結論は同じである。
差分 FIR の再構成誤差は十分小さいため、tap 数を増やしても今回の自己抑圧問題は解決しない。
次に検討すべき対象は、単一待ち受け方位制約ではなく、source mask 内の複数方位制約、
LCMV、または mainlobe 幅を持たせたロバスト制約である。

## 21. 差分 FIR tap 数と時間領域畳み込み処理量の tradeoff

### 21.1 目的

差分補正 FIR の tap 数を増やすと、周波数領域 MVDR 参照重みの再現性は上がる一方、
時間領域の直接畳み込み処理量は `n_ch * fir_taps` complex MAC/sample/beam に比例して増える。
ここでは tap 数候補ごとに、FIR 化誤差と `DifferenceCorrectionFIR.process` の実測時間を比較した。

評価パターンは `fixed_beam_single_source` と `slc_runtime` の混合である。
周波数軸は 768 Hz から 6144 Hz まで 128 Hz 間隔、beam は 121、channel は 32 とした。
MVDR 共分散は source を含む `R = sigma_s^2 a_s a_s^H + sigma_n^2 I`、
MVDR 制約は各 scan beam の待ち受け方位 steering `a(theta_beam, f)` とした。

成果物は以下である。

```text
artifacts/beamforming/fixed_delay_diff_mvdr/tap_runtime_tradeoff/tap_runtime_tradeoff_report.md
artifacts/beamforming/fixed_delay_diff_mvdr/tap_runtime_tradeoff/tap_runtime_tradeoff.csv
artifacts/beamforming/fixed_delay_diff_mvdr/tap_runtime_tradeoff/tap_runtime_tradeoff.png
```

### 21.2 結果

```text
taps  runtime factor re 128  max q RMS error  max target response error
 16        0.294              2.978e-01        2.082e+00
 24        0.339              2.694e-01        2.623e+00
 32        0.433              2.192e-01        3.904e+00
 48        0.523              6.568e-02        1.539e+00
 64        0.624              6.721e-04        7.329e-03
 96        0.806              1.806e-05        6.225e-04
128        1.000              8.415e-08        1.619e-06
192        1.420              6.443e-16        1.965e-14
256        1.819              8.921e-16        3.246e-14
384        2.707              9.603e-16        4.253e-14
512        3.547              1.007e-15        2.329e-14
```

時間領域直接畳み込みの理論処理量は `32 * fir_taps` complex MAC/sample/beam である。
121 beam 全 scan に対して全 beam 分の補正 FIR を適用する場合は、この値に 121 を掛ける。

### 21.3 判断

- `64 taps` 以下は target response error が `7.329e-3` 以上であり、FIR 化誤差が大きい。
- `96 taps` は runtime が 128 taps の約 0.806 倍だが、target response error は `6.225e-4` 残る。
- `128 taps` は q RMS error が `8.415e-8`、target response error が `1.619e-6` で、処理量とのバランスがよい。
- `192 taps` は q RMS error と target response error がほぼ数値精度まで下がるが、runtime は 128 taps の約 1.420 倍である。
- `256 taps` 以上は 192 taps に対して有意な精度改善がなく、処理量だけが増える。

したがって、通常候補は `128 taps`、厳密に `1e-6` 未満の target response error を要求する候補は `192 taps` とする。
`512 taps` は今回の評価条件では過剰であり、off-grid source 自己抑圧の改善にも寄与しない。

## 22. 実アレイ外部データ評価スクリプト

### 22.1 目的

別 PC 上で実アレイ位置、実 channel shading、小数遅延 FIR フィルタを読み込み、
fixed-delay diff-MVDR の tap 数評価と scene_renderer 入力評価を実行できるようにした。
公開 API は ndarray を受け取る形にし、raw file 読み込みは CLI 側の便利用処理として分離した。

作成した script は以下である。

```text
examples/beamforming/external_fixed_delay_diff_mvdr_inputs.py
examples/beamforming/evaluate_external_fixed_delay_diff_mvdr_tap_tradeoff.py
examples/beamforming/evaluate_external_scene_renderer_fixed_delay_diff_mvdr.py
```

### 22.2 外部データの ndarray 変換

`external_fixed_delay_diff_mvdr_inputs.py` は MATLAB 例と同じ raw 配列を Python ndarray へ変換する。

```text
COE_POS          -> array_positions_m: shape [n_ch, 3], unit m
COE_CBFSHADING   -> shading_by_channel_bin: shape [n_ch, n_bin], complex128
frac_grid        -> shape [n_frac_filter], unit sample
frac_filters     -> shape [n_frac_filter, n_tap]
```

`COE_POS` は MATLAB の `reshape(pos, 3, [])` と同じ column-major 解釈で読み、
評価 API では `[n_ch, 3]` に転置して使う。
`COE_CBFSHADING` は MATLAB の `reshape(shading, nCh, [])` と同じ column-major 解釈で読み、
前半列を real、後半列を imag として `[n_ch, n_bin]` の complex shading に変換する。
shading 周波数間隔は `dF_shad = 0.5 Hz` を既定値にしている。
`COE_DLYFILT_128` は MATLAB の `reshape(delayfilter, 128, [])` と同じ column-major 解釈で読み、
各 column を 1 本の小数遅延 FIR とみなして `[n_frac_filter, 128]` に転置する。
raw file には小数遅延 grid が含まれないため、既定では `-0.5 sample` から `0.5 sample` までの
等間隔 grid として扱う。tap 数が 128 以外の場合は `--fractional-delay-taps` で明示する。

### 22.3 tap 数 tradeoff 評価

公開 API は以下である。

```python
evaluate_external_fir_tap_tradeoff(
    array_positions_m=positions,
    shading_by_channel_bin=shading,
    shading_frequency_step_hz=0.5,
    fractional_delay_filter_bank=filter_bank,
    tap_counts=(16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512),
)
```

CLI 例は以下である。

```powershell
.venv\Scripts\python.exe examples\beamforming\evaluate_external_fixed_delay_diff_mvdr_tap_tradeoff.py `
  --coe-pos C:\path\to\COE_POS `
  --coe-cbfshading C:\path\to\COE_CBFSHADING `
  --shading-df-hz 0.5 `
  --fractional-delay-raw C:\path\to\COE_DLYFILT_128 `
  --fractional-delay-taps 128 `
  --output-dir artifacts\beamforming\fixed_delay_diff_mvdr\external_tap_tradeoff
```

出力は以下である。

```text
external_tap_tradeoff.csv
external_tap_tradeoff_report.md
```

### 22.4 scene_renderer 入力評価

公開 API は以下である。SL/NL の dB 指定は script 側で線形振幅へ変換し、API には変換後の値を渡す。

```python
sources = (
    ExternalSceneSource(
        label="S1",
        azimuth_deg=40.0,
        frequency_hz=1024.0,
        peak_amplitude=db20_rms_to_tone_peak_amplitude(0.0),
    ),
)
rows, arrays = evaluate_external_scene_renderer_inputs(
    array_positions_m=positions,
    shading_by_channel_bin=shading,
    shading_frequency_step_hz=0.5,
    fractional_delay_filter_bank=filter_bank,
    sources=sources,
    noise_rms_amplitude=db20_to_rms_amplitude(-40.0),
)
```

CLI 例は以下である。

```powershell
.venv\Scripts\python.exe examples\beamforming\evaluate_external_scene_renderer_fixed_delay_diff_mvdr.py `
  --coe-pos C:\path\to\COE_POS `
  --coe-cbfshading C:\path\to\COE_CBFSHADING `
  --shading-df-hz 0.5 `
  --fractional-delay-raw C:\path\to\COE_DLYFILT_128 `
  --fractional-delay-taps 128 `
  --source-azimuths-deg 40,80 `
  --source-frequencies-hz 1024,1536 `
  --source-levels-db20 0,-6 `
  --noise-level-db20 -40 `
  --fir-taps 128 `
  --output-dir artifacts\beamforming\fixed_delay_diff_mvdr\external_scene_renderer
```

出力は以下である。

```text
external_scene_summary.csv
external_scene_arrays.npz
external_scene_report.md
```

### 22.5 smoke test

実データがない環境でも、synthetic の `COE_POS`、`COE_CBFSHADING`、`COE_DLYFILT_128` を作成して、
両 CLI が最後まで実行できることを確認した。これは raw 読み込み、ndarray 正規化、評価 API 接続の確認であり、
実アレイ性能の判断には使わない。

```text
artifacts/beamforming/fixed_delay_diff_mvdr/external_smoke_tap_tradeoff_raw/external_tap_tradeoff_report.md
artifacts/beamforming/fixed_delay_diff_mvdr/external_smoke_scene_raw/external_scene_report.md
```
