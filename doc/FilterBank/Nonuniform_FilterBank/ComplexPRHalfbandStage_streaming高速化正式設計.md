# ComplexPRHalfbandStage streaming高速化正式設計

## 1. 目的

本書は、`ComplexPRHalfbandStage` を用いた nonuniform tree の
streaming 実装について、

- どこが現在の処理時間ボトルネックか
- 正式版としてどの実装方式を採用するか
- その方式が理論上どこまで最適か

を整理するための文書である。

ここでの対象は

- stage-level streaming analyzer
- stage-level streaming synthesizer

であり、leaf beamforming の `long FFT / short FFT` ではない。

---

## 2. 現状の問題

現在の `ComplexFIRHalfbandStageStreamingAnalyzer / Synthesizer` は、
exact-by-construction の検証器としては正しいが、
処理時間の観点では正式実装に使えない。

現行方式は、各 update ごとに

1. これまでの全入力 prefix を保持する
2. `analysis()` または `synthesis()` を全 prefix に対して再実行する
3. 既出力部分を切り捨てて新規分だけ返す

構造である。

したがって、chunk 長を `C`、最終長を `N` とすると、
1 stage あたりの総計算量は概ね

```text
O((C + 2C + 3C + ... + N) * L)
= O(N^2 * L / C)
```

となる。

ここで `L` は FIR tap 長である。

この構造は

- exactness の確認
- offline / streaming 一致の確認
- metadata 規約の確認

には有効だが、
正式高速版の骨格としては不適切である。

---

## 3. 正式版に求める条件

正式な高速実装は、少なくとも以下を満たす必要がある。

1. 現行 `ComplexFIRHalfbandStage.analysis / synthesis` と同じ数値結果を返すこと
2. `analysis_phase`, `synthesis_phase`, `delay_compensation` を正しく扱えること
3. `FormalComplexPRHalfbandStage` の lower-edge packet 規約を壊さないこと
4. causal / streaming 実装であること
5. memory が入力長 `N` に比例して増えないこと
6. chunk 与え方によらず offline と一致すること

---

## 4. 候補方式

## 4.1 候補A: 現行 prefix 再実行方式

特徴:

- 実装が最も単純
- exact-by-construction
- しかし `O(N^2)`

判断:

- 検証器としては残す
- 正式高速版としては不採用

## 4.2 候補B: full-tap 直接 FIR + parity 判定

analysis では、入力リングバッファへ 1 sample ずつ積み、
`(sample_index - analysis_phase) mod 2 == 0` のときだけ

- low filter 全 tap 内積
- high filter 全 tap 内積

を実行する。

synthesis では、child-rate の low/high ring buffer を持ち、
新しい low/high pair が来るたびに

- even 側 parent sample
- odd 側 parent sample

を直接計算する。

特徴:

- exact
- memory `O(L)`
- 総計算量 `Theta(NL)`

判断:

- 理論次数は良い
- ただし decimation / interpolation の構造を明示しないため、
  定数因子と実装見通しは polyphase より不利

## 4.3 候補C: polyphase 状態機械

analysis / synthesis を decimate-by-2 / interpolate-by-2 の
polyphase FIR として実装する。

analysis では、
位相整列後の even / odd 系列を child-rate で管理し、
1 child step ごとに

- low polyphase branch 和
- high polyphase branch 和

を計算する。

synthesis では、
low/high child 系列から

- parent 偶数位相 sample
- parent 奇数位相 sample

を polyphase branch の和として直接生成する。

特徴:

- exact
- memory `O(L)`
- 総計算量 `Theta(NL)`
- upsample 後のゼロや不要位相を一切計算しない
- analysis / synthesis 双方で decimation/interpolation 構造が明示される

判断:

- 正式採用候補

## 4.4 候補D: stage ごとの FFT overlap-save

1 stage の FIR を block FFT で処理する方式。

特徴:

- tap 長が極端に長ければ有利な場合がある
- ただし stage は leaf beamforming と異なり
  - 帯域ごとに sample rate が違う
  - packet が非同期に流れる
  - flush 規約が厳密
  - 係数長は現状それほど長くない

判断:

- 正式 v1 では不採用
- まず direct FIR streaming を完成させる

---

## 5. 採用方式

正式版 v1 では、

- `ComplexPRHalfbandStage` の streaming analyzer / synthesizer

を

- polyphase 状態機械

として実装する。

理由は以下である。

1. exactness を保ったまま `Theta(NL)` へ落とせる。
2. decimation / interpolation で不要なゼロ演算を避けられる。
3. memory を `O(L)` に抑えられる。
4. `analysis_phase`, `synthesis_phase`, `delay_compensation` を自然に組み込める。
5. nonuniform tree の packet 契約と整合する。

---

## 6. 理論上の最適性

ここでいう「理論上最適」は、以下の前提つきで定義する。

前提:

1. 対象は固定係数の dense FIR である。
2. streaming / causal に online 処理する。
3. exact な time-domain 出力を返す。
4. decimation ratio は 2、interpolation ratio も 2 である。
5. 係数の特殊な sparse 構造や lifting 分解は仮定しない。

このとき、1 child-rate step の analysis 出力 pair は
一般に `L` tap 全体へ依存する 2 本の線形形式である。
同様に、1 child-rate step の synthesis では
2 本の parent sample が `L` tap 全体へ依存する。

したがって、任意 dense FIR に対する exact online 実装は、
1 step あたり少なくとも

```text
Omega(L)
```

の係数寄与を処理する必要がある。

一方、polyphase 状態機械は
1 step あたり

```text
Theta(L)
```

で実現できる。

よって、固定 dense FIR / exact online streaming という前提では、
polyphase 状態機械は

```text
Theta(L) per child step
Theta(NL) total
```

を達成し、
漸近次数として最適である。

さらに stage tap 長 `L` を設計定数とみなすと、
全体は

```text
Theta(N)
```

となる。

---

## 7. formal 実装仕様

## 7.1 analysis

現行定義は

```text
y_low[k]  = sum_m h_low[m]  * x[2k + p_a - m]
y_high[k] = sum_m h_high[m] * x[2k + p_a - m]
```

である。

ここで `p_a = analysis_phase`。

これを even / odd へ分けると

```text
y_b[k]
  = sum_r h_b[2r]   * x_even[k-r]
  + sum_r h_b[2r+1] * x_odd[k-r]
```

となる。

正式実装では

- `x_even`
- `x_odd`

の child-rate ring buffer を持ち、
1 child step ごとに low/high の 2 出力を計算する。

## 7.2 synthesis

現行定義は

```text
x_hat = conv(upsample(low), g_low) + conv(upsample(high), g_high)
```

である。

これを polyphase 化すると、
1 child step ごとに

- parent 位相0 sample
- parent 位相1 sample

を直接計算できる。

正式実装では

- low child ring buffer
- high child ring buffer

を持ち、
新しい low/high pair ごとに parent 2 sample を生成する。

`delay_compensation` は

- 出力 cursor の初期位置
- flush 時の残長

として管理する。

## 7.3 flush 規約

exactness を保つため、flush は

- analysis: `full_analysis_length - stable_analysis_length`
- synthesis: `full_synthesis_length - stable_synthesis_length`

に相当する tail を追加で emit しなければならない。

正式実装では、内部状態へ零入力を進めるのではなく、

- 現在の state
- 既出力数
- offline と同じ長さ式

から必要 tail 長を計算し、
その分だけ stateful FIR を進めて出す。

---

## 8. tree 全体の処理量式

1 internal stage `v` の sample rate を `f_v [sample/s]`、
analysis / synthesis tap 長を `L_a(v)`, `L_s(v)` とすると、
polyphase 状態機械の常時計算量は概ね

```text
Cost_stage(v) ~= f_v * (L_a(v) + L_s(v))
```

である。

nonuniform tree 全体では

```text
Cost_tree ~= sum_v f_v * (L_a(v) + L_s(v))
```

となる。

今回の 8-leaf 非対称木では internal node rate は

```text
32768, 16384, 8192, 4096, 2048, 1024, 512
```

なので、全 stage が同一 tap 長 `L_a = L_s = L` なら

```text
Cost_tree ~= 65024 * 2L
```

である。

例:

- `L = 8` なら約 `1.04M complex MAC/s`
- `L = 64` なら約 `8.32M complex MAC/s`

したがって tree 本体は、正式 tap 長が過大でない限り
leaf beamforming core と同程度か、それ以下に抑えられる可能性が高い。

---

## 9. 実装候補の絞り込み

正式版 v1 の実装候補は以下に絞る。

1. `PolyphaseComplexFIRHalfbandStageStreamingAnalyzer`
2. `PolyphaseComplexFIRHalfbandStageStreamingSynthesizer`
3. 既存 prefix 再実行版は `Oracle*` として残す

`FormalComplexPRHalfbandStage` と `FormalNonuniformTreeStreaming*` は
外部契約を変えず、内部の stage-level engine のみ差し替える。

これにより

- packet 契約
- metadata 契約
- 既存試験

を保ったまま高速化できる。

---

## 10. 今後の実装順

処理時間を最優先にする場合、次の順で進める。

1. stage-level polyphase analyzer を実装
2. stage-level polyphase synthesizer を実装
3. 既存 oracle と bit-exact ではなく `allclose` 一致を回帰で固定
4. formal tree streaming へ差し替え
5. Daubechies beamforming streaming へ差し替え
6. その後に処理時間を再測定し、tree 本体コストを正式見積りへ反映

---

## 11. 現時点での結論

処理時間低減の観点で、
正式に採用すべき方式は

- stage-level polyphase 状態機械

である。

この方式は、固定 dense FIR / exact online streaming という前提では
漸近次数 `Theta(NL)` を達成し、理論上最適である。

したがって、次の実装段階では

- root 合成の追加整理
ではなく
- `ComplexPRHalfbandStage` 自体の streaming engine 差し替え

を最優先に進めるのが正しい。

---

## 12. 実装・確認結果（2026-07-02）

実装結果:

1. `ComplexFIRHalfbandStageStreamingAnalyzer` を stateful analyzer へ差し替えた
2. `ComplexFIRHalfbandStageStreamingSynthesizer` を sparse overlap-add 型の stateful synthesizer へ差し替えた
3. 旧 prefix 再実行版は
   - `OracleComplexFIRHalfbandStageStreamingAnalyzer`
   - `OracleComplexFIRHalfbandStageStreamingSynthesizer`
   として保存した
4. `FormalNonuniformTreeStreamingAnalyzer / Synthesizer` は public 契約を変えずに新 engine を利用する

確認結果:

- `python -m pytest -q tests/filterbank/test_complex_halfband_stage.py`
  - `5 passed in 0.17s`
- `python -m pytest -q tests/nonuniform/test_formal_nonuniform_streaming.py tests/nonuniform/test_daubechies_nonuniform_streaming.py`
  - `6 passed in 14.33s`
- `python -m pytest -q`
  - `153 passed in 21.13s`

### 12.1 stage 単体の速度確認

`daubechies_qmf_order4_taps8`, `n_sample = 32768` での最短実測値:

| chunk_size | analyzer new [s] | analyzer oracle [s] | synthesizer new [s] | synthesizer oracle [s] |
|---|---:|---:|---:|---:|
| `32` | `0.076` | `1.070` | `0.103` | `1.368` |
| `64` | `0.074` | `0.533` | `0.102` | `0.677` |
| `128` | `0.072` | `0.272` | `0.104` | `0.339` |
| `256` | `0.072` | `0.134` | `0.102` | `0.155` |
| `512` | `0.073` | `0.070` | `0.108` | `0.085` |

解釈:

- streaming らしい小〜中 chunk では stateful 版が明確に有利
- chunk が大きい条件では Python の `np.convolve` を使う oracle 版が競る
- ただし oracle 版は chunk 数が増えると `O(N^2)` に寄るため、正式版の骨格には使えない

### 12.2 formal tree の速度確認

`FormalNonuniformTreeFilterBank.default_for_fs(32768.0)`, `n_sample = 32768` の最短実測値:

| chunk_size | formal streaming [s] | oracle root rebuild [s] | speedup |
|---|---:|---:|---:|
| `32` | `1.829` | `6.043` | `3.30x` |
| `64` | `1.363` | `2.713` | `1.99x` |
| `128` | `0.986` | `1.356` | `1.37x` |
| `256` | `0.750` | `0.719` | `0.96x` |
| `512` | `0.569` | `0.436` | `0.77x` |

解釈:

- chunk が細かい streaming 条件では、新しい stage-level engine の効果が tree 全体まで届く
- chunk が大きい条件では Python 実装の定数因子が支配し、oracle 版がまだ有利な点が残る
- それでも formal v1 の設計判断としては、stateful stage へ進んだことで構造上の `O(N^2)` ボトルネックは除去できた

### 12.3 現時点の判断

現時点では、

- 正式版 streaming 構造は実装済み
- offline / oracle 一致は回帰で固定済み
- 小〜中 chunk の streaming 条件では実測でも改善を確認済み

と判断してよい。

今後の高速化は、

- Python 実装の定数因子削減
- 係数長拡大時の C++ 実装
- leaf beamforming 側の `used_channels` / covariance 最適化

へ進めるのが妥当である。
