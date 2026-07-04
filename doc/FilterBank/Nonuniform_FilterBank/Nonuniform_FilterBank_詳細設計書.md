# Nonuniform FilterBank 詳細設計書

## 1. 目的

補助図面として `doc/Nonuniform_FilterBank_正式処理詳細図.md` を併用する。

本書は、不均一複素フィルタバンク系について、
均一帯域 PRDFT 設計書と同様に

- 何を本命構造として採用するか
- どこまで段階検証が進んだか
- どこが `Pending` か
- 最終的にどの処理量比較を目標にするか

を一本で辿れる文書として整理するための詳細設計書である。

既存の均一帯域 PRDFT 系は残し、

- 基準実装
- 比較対象
- 切り分け用参照系

として今後も使う。

---

## 2. 位置付け

本系は、均一帯域 PRDFT 系の拡張ではない。

- 既存 PRDFT 系は維持する
- 非均一系は別系列として新設する
- 内部表現は最初から複素とする

という立場を採る。

また、現時点では

- `block exact PR` の基準木
- streaming 基本骨格

はすでに成立しており、今後も切り分け用基準として残す。

一方で、まだ正式版として未完了なのは

- 実用 stopband を持つ正式 `ComplexPRHalfbandStage`
- `ComplexPRHalfbandStage` の処理時間要件を満たす stateful polyphase streaming 実装
- `causal analytic front-end` の正式評価条件固定
- leaf packet の lower-edge 周波数規約と遅延 metadata の実用運用整理

である。

---

## 3. 設計目標

### 3.1 帯域分割

`fs = 32768 Hz` を基準とし、正側帯域を以下の 8 leaf に分ける。

| leaf band | target resolution |
|---|---:|
| `0 - 128 Hz` | `1 Hz` |
| `128 - 256 Hz` | `1 Hz` |
| `256 - 512 Hz` | `2 Hz` |
| `512 - 1024 Hz` | `2 Hz` |
| `1024 - 2048 Hz` | `4 Hz` |
| `2048 - 4096 Hz` | `8 Hz` |
| `4096 - 8192 Hz` | `16 Hz` |
| `8192 - 16384 Hz` | `32 Hz` |

### 3.2 目標構造

本命構造は以下とする。

1. 実入力
2. `causal analytic front-end`
3. `ComplexPRHalfbandStage` を再帰接続した非対称木
4. leaf ごとの `BandPacket`
5. leaf ごとの bandwise beamforming
6. 合成木
7. 実出力

### 3.3 採用しない方式

以下は本命構造として採用しない。

- 均一 DFT bank の高域ビングルーピングだけを最終形とすること
- leaf を `(n_ch, n_band, n_sample)` の長方形配列へ固定すること
- 実 wavelet だけで複素位相規約を曖昧にすること
- beamforming 後に単純な複素共役だけで負側を後付けすること

---

## 4. 正式構造で固定した内容

以下は、今後の詳細化で大きく崩さない前提として固定済みとみなす。

### 4.1 木構造

```text
0-16384
├── 0-8192
│   ├── 0-4096
│   │   ├── 0-2048
│   │   │   ├── 0-1024
│   │   │   │   ├── 0-512
│   │   │   │   │   ├── 0-256
│   │   │   │   │   │   ├── 0-128
│   │   │   │   │   │   └── 128-256
│   │   │   │   │   └── 256-512
│   │   │   │   └── 512-1024
│   │   │   └── 1024-2048
│   │   └── 2048-4096
│   └── 4096-8192
└── 8192-16384
```

### 4.2 leaf interface

leaf の正式出力は長方形配列ではなく、概念的に以下の packet とする。

```text
BandPacket
    band_id
    f_low_hz
    f_high_hz
    center_frequency_hz
    nominal_sample_rate_hz
    target_resolution_hz
    time_origin
    delay_samples_at_root_rate
    complex_samples
```

### 4.3 streaming 契約

正式構造では

- leaf ごとの非同期 packet emission
- packet 単位の時刻管理
- offline と streaming の一致

を前提にする。

### 4.4 実用アレイ条件

主検証条件は

- スパース直線アレイ
- 中央が密
- 端が疎

の `dense center / sparse edge` 形状とする。

---

## 5. 段階検証の流れ

均一帯域 PRDFT 系と同様に、以下の段階で積み上げる。

1. stage 単体 PR
2. 2-level tree PR
3. 非対称 full tree PR
4. streaming / offline 一致
5. `causal analytic front-end` を含む実入力から実出力までの PR
6. leaf band ごとの beamforming 接続
7. スパース直線アレイ条件での実用評価
8. 処理量比較の確定

---

## 6. 現在の進捗

### 6.1 構造骨格

以下は成立済みである。

- 非対称 8 leaf 木の定義
- `BandPacket` ベースの leaf 出力
- `block exact PR` 基準木
- tree 本体の streaming 骨格

対応実装:

- `src/spflow/filterbank/nonuniform_tree.py`
- `src/spflow/filterbank/nonuniform_streaming.py`

対応試験:

- `tests/nonuniform/test_nonuniform_filterbank.py`
- `tests/nonuniform/test_nonuniform_streaming.py`

直近確認結果:

```text
135 passed in 4.57s
```

この時点で、

- 木構造
- packet 化
- block streaming の骨格

が原因で今後の設計が全面的にやり直しになる可能性は低いと判断してよい。

### 6.2 stage 候補

正式 `ComplexPRHalfbandStage` については、
以下の 3 層で位置付けが固まっている。

1. `A-0`: 2 点 DFT 骨格確認用基準 stage
2. `試作品1`: explicit FIR 形式で PR / streaming 契約を確認
3. `A-1`: 既知 orthonormal QMF family で stopband 改善傾向を確認

対応実装:

- `src/spflow/filterbank/halfband_stage.py`
- `src/spflow/filterbank/complex_halfband_stage.py`
- `src/spflow/filterbank/halfband_stage_candidates.py`

対応試験:

- `tests/filterbank/test_halfband_stage.py`
- `tests/filterbank/test_complex_halfband_stage.py`
- `tests/filterbank/test_halfband_stage_candidates.py`

現時点の判断:

- PR / streaming 契約は成立可能
- paraunitary FIR family を延ばす方向は妥当
- streaming 高速化の正式方式は stateful polyphase へ固定してよい
- ただし `daubechies_qmf_order4_taps8` でも stopband は約 `19.55 dB` であり、
  正式要求 `80 dB` には未達

したがって、正式 stage の構造方向と streaming 高速化方向は固まっているが、
正式係数と高速版本実装はまだ未完了である。

### 6.3 causal analytic front-end

正式仕様は

- `doc/CausalAnalyticFrontend_正式仕様.md`

で固定した。

現時点の判断:

- front-end を独立部品として置く方針は妥当
- FIR Hilbert transformer 系を第一候補とする
- 最小実装と streaming / offline 一致確認は完了している
- formal tree 接続と real end-to-end 最小検証も完了している
- 残りは正式評価条件と delay 合成規約の固定である

### 6.4 beamforming 接続

beamforming との接続方針は固まっている。

- leaf ごとに `used_channels` を持つ
- 低域は広開口
- 高域は中央高密度サブアレイ
- overlap-save 条件も leaf ごとに持つ
- さらに各 leaf 内部は `long FFT + short FFT` の 2 経路を持つ

ここでの役割分担は以下とする。

- `long FFT path`: overlap-save により実際の beamformed 出力を作る
- `short FFT path`: `Rxx` 推定と重み更新のための短時間統計量を作る

また、均一帯域版で整理した通り、
`reuse_filter_fft` を long FFT の単純流用として標準方式にはしない。
正式方針では、leaf ごとに明示的な `short FFT` を持たせる。

詳細は `doc/Nonuniform_FilterBank_leaf処理構造.md` に整理した。

ただし、正式な `used_channels`、
leaf ごとの `frame size / valid size`、
`short_fft_size / short_fft_hop_size` はまだ暫定値である。

また、Daubechies 系 stage を用いた beamforming 先行試作を
`doc/Nonuniform_FilterBank_Daubechies_beamforming試作結果.md` に整理した。
現時点では、broadside CBF 条件で
`nonuniform analysis -> leaf beamforming -> synthesis`
を通しても安定に再構成できることを確認済みである。

さらに、streaming 統合と chunk-size sweep の結果を
`doc/Nonuniform_FilterBank_Daubechies_streaming_sweep結果.md` に整理した。
正式 metadata 付き streaming は formal tree / front-end 系で完了しており、
Daubechies beamforming 試作については、旧 prototype streaming での continuity 確認に加えて、
現在は formal metadata 付き増分 root 合成へ差し替え済みであり、
offline / streaming 一致も回帰試験で確認済みである。

また、formal tree 接続結果を
`doc/Nonuniform_FilterBank_formal_tree接続結果.md` に整理した。
現時点では、`FormalComplexPRHalfbandStage` を full tree へ接続し、
real input -> `causal analytic front-end` -> formal tree -> real output の
最小 end-to-end に加え、formal metadata 付き streaming が
offline と一致することまで確認済みである。

---

## 7. 構造リスクの判定

現時点では、リスクを以下のように切り分ける。

### 7.1 構造的にほぼ固定してよい部分

- 非対称木そのもの
- packet ベース leaf interface
- leaf ごとの独立 sample rate
- streaming と offline を一致させる設計思想
- `causal analytic front-end` を tree の外側に置く構造

### 7.2 まだ技術リスクが残る部分

- `>= 80 dB` stopband を満たす正式 stage 係数
- `causal analytic front-end` の定量評価条件固定
- multiband interferer / real-input streaming を含む実用 beamforming 評価

したがって今後の主な不確実性は

- 木構造の未熟さ

ではなく、

- フィルタ係数設計
- 位相規約の実装
- front-end 実装

にあると整理できる。

---

## 8. 実用評価条件

### 8.1 初期代表アレイ

初期の主評価条件として、
以下の 32 ch スパース直線アレイを代表条件に置く。

- dense center: `16 ch @ 0.01 m`
- sparse edge: 左右 `8 pair @ 0.04 m`

代表位置は以下である。

```text
[-0.395, -0.355, -0.315, -0.275, -0.235, -0.195, -0.155, -0.115,
 -0.075, -0.065, -0.055, -0.045, -0.035, -0.025, -0.015, -0.005,
  0.005,  0.015,  0.025,  0.035,  0.045,  0.055,  0.065,  0.075,
  0.115,  0.155,  0.195,  0.235,  0.275,  0.315,  0.355,  0.395] m
```

これは最終配列固定値ではなく、

- dense center / sparse edge 条件
- high band で中心部だけを使う条件

を揃えた初期代表条件である。

### 8.2 初期代表 `used_channels`

leaf ごとの暫定使用チャネル数は以下を置く。

| leaf band | used channels |
|---|---:|
| `0 - 128 Hz` | `32` |
| `128 - 256 Hz` | `32` |
| `256 - 512 Hz` | `24` |
| `512 - 1024 Hz` | `20` |
| `1024 - 2048 Hz` | `16` |
| `2048 - 4096 Hz` | `12` |
| `4096 - 8192 Hz` | `8` |
| `8192 - 16384 Hz` | `4` |

具体的には、低域から高域へ向かって
中心対称な contiguous subset を縮める代表運用を採る。

---

## 9. 処理量比較

### 9.1 現時点の比較位置付け

処理量比較は、以後以下の 3 つに分ける。

- `A. 実時間パス`
- `B. 重み更新パス`
- `C. 固定コスト`

ここで

- `A. 実時間パス` は `FFT -> beamforming 内積 -> IFFT`
- `B. 重み更新パス` は covariance 更新と MVDR weight 計算
- `C. 固定コスト` は `causal analytic front-end` と nonuniform tree

である。

方式そのもののリアルタイム性比較は `A` を主指標とし、
`B` は `1 update あたり`、`C` は別加算とする。

### 9.2 実時間パスの暫定比較

analytic 複素入力 `32768 samples` を 1 block 処理するとき、
現時点の概算では

- full-band 一括方式: 約 `16.74448M / block`
- 非均一 leaf-wise 実時間パス: 約 `6.024964M / block`

である。

したがって、現行 nonuniform leaf 方式の実時間パスは
full-band 一括方式の約 `36.0%` であり、約 `2.78x` 軽い。

また、beamforming 内積だけを見ると

- full-band 一括方式: `0.52432M / block`
- 非均一 leaf-wise: `0.290052M / block`

であり、約 `55.3%` である。

この差は

- leaf ごとの `used_channels` 削減
- 小 FFT 化
- one-sided 化

の合成効果である。

### 9.3 重み更新パスの比較方針

重み更新パスは実時間パスと更新レートが異なるため、
方式比較では `1 update あたり` の構造比較として扱う。

現行 leaf 条件では

- full-band 一括側: `16385 bins`, `32 ch`
- nonuniform 側: `1672 one-sided bins`, `m_l = 32, 32, 24, 20, 16, 12, 8, 4`

となるため、`Rxx` 更新や MVDR solve の規模は
full-band 一括方式よりかなり小さくなる見込みである。

ただし、`1 秒あたり` の重み更新コストは

- covariance 積分時間
- update scheduler
- 間引き率
- 次数変換方式

で変わるため、運用条件固定後に別途算入する。

### 9.4 まだ含んでいないもの

以下はまだ最終総処理量へ正式算入していない。

- `causal analytic front-end`
- 非均一木 analysis / synthesis の FIR コスト
- 重み更新パスの正式 update rate
- 次数変換コスト

したがって、現時点の数値は
全体処理量ではなく、主に `A. 実時間パス` の暫定比較値である。

---

## 10. Pending

### P-STAGE-1

正式版 `ComplexPRHalfbandStage` として採用できる
`>= 80 dB` 級 stopband の paraunitary FIR stage 係数設計法が未確定。

### P-STAGE-2

formal packet 規約に必要な

- upper child の lower-edge 基準周波数 shift
- `time_origin`
- `delay_samples_at_root_rate`

の最小実装に加え、leaf beamforming 出力での practical rule 固定まで完了した。

leaf beamforming では、

- output packet の `time_origin_at_root_rate` は valid region 先頭の root-rate 時刻
- 同一 leaf 内では `packet_length * 2**tree_depth` ずつ進む
- `delay_samples_at_root_rate` は v1 では保持する

規約で運用する。

残る delay 合成の全体規約は `P-FRONT-1` 側で管理する。

### P-FRONT-1

`causal analytic front-end` の最小試作、streaming / offline 一致確認、
formal metadata 付き tree との streaming 接続確認、
および real input から real output までの最小 end-to-end 検証は実施済み。
残っているのは、

- negative-frequency suppression の正式評価指標固定
- formal tree との delay 合成規約固定

である。

### P-BF-1

leaf ごとの正式 `used_channels` が未固定。

### P-BF-2

leaf ごとの正式 `frame size / valid size / integration time / update period`
が未固定。

### P-BF-3

leaf ごとの正式 `short_fft_size / short_fft_hop_size` が未固定。

### P-BF-4

Daubechies 系先行試作では、broadside / identical-channel 条件に加えて、
representative leaf の interferer 条件まで `mvdr` を確認済みである。

具体的には

- covariance 更新
- short FFT 起点の重み更新
- representative leaf での target distortionless / interferer 応答低減
- formal offline / streaming 一致

まで確認済みである。

残っているのは、

- 周波数依存 steering
- multiband interferer を含む MVDR 実用評価

である。

### P-BF-5

Daubechies 系 `cbf` / `mvdr` については、leaf beamforming を
formal metadata 付き tree と formal streaming 実装へ統合済みである。
さらに、single-band interferer 条件では formal full-tree streaming が
offline と一致し、後半区間で CBF より target 誤差を下げるところまで確認済みである。

残っているのは、

- real-input streaming を含む正式評価
- multiband interferer 条件を含む実用評価

である。

### P-STREAM-1

`ComplexPRHalfbandStage` の stage-level streaming を
stateful polyphase 実装へ差し替える本実装が未完了。

### P-COST-1

front-end と tree 本体を含めた最終処理量比較が未完了。

---

## 11. 今後の進め方

今後は以下の順で進める。

1. `ComplexPRHalfbandStage` の stateful polyphase streaming 実装へ進む
2. tree 本体を含む処理時間を再測定する
3. その後に周波数依存 steering を入れた multiband interferer 条件での MVDR 実用評価へ進む
4. real-input streaming を含む正式評価へ進む
5. その後に `A-2`: 高選択度 paraunitary FIR stage の係数設計へ戻る
6. 最後に front-end と tree 本体を含めた最終処理量比較を詰める

優先順位の詳細は `doc/Nonuniform_FilterBank_正式化優先順位.md` に整理した。

---

## 12. 現時点での結論

非均一複素フィルタバンクは、
構造骨格の段階ではすでに成立している。

したがって今後の主問題は

- 木構造そのもの

ではなく、

- 実用 stopband を持つ正式 stage
- causal analytic front-end
- delay / phase metadata を含む formal packet 実装

である。

よって、今後は

- 構造を再考する段階

ではなく、

- 正式部品を 1 つずつ埋めて実用系へ仕上げる段階

に入ったと判断する。



