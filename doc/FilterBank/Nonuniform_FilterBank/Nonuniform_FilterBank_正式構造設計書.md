# Nonuniform FilterBank 正式構造設計書

## 1. 目的

本書は、今後実装する不均一フィルタバンクの

- 実用的
- 正式
- 本命

な構造を先に固定するための設計書である。

既存の

- `Nonuniform_FilterBank_設計方針.md`
- `Nonuniform_FilterBank_検証条件.md`
- `Nonuniform_FilterBank_streaming検証結果.md`
- `Nonuniform_FilterBank_正式処理詳細図.md`

は引き続き有効である。

ただし本書では、それらの方針文書より一段具体化し、

- 何を本命構造として採用するか
- 何を基準実装として残すか
- どこまでを正式仕様にするか

を明確に定める。

---

## 2. 設計上の立場

本系は、既存の均一帯域 PRDFT 系からの移行ではない。

- 既存 PRDFT 系は比較基準として残す
- 不均一系は別系列として新設する
- 不均一系の内部表現は最初から複素とする

という立場を採る。

また、今回すでに確認できた

- block exact PR の基準木
- streaming 骨格

は、今後の実装で破棄しない。

ただしそれらは

- 骨格確認用
- PR 切り分け用
- streaming 切り分け用

の基準系とし、
最終的な正式構造そのものとは区別する。

---

## 3. 正式構造として採用する方式

正式構造として採用するのは、以下である。

### 3.1 全体像

- 実入力
- causal analytic front-end
- complex PR halfband stage を用いた非対称木
- leaf band ごとの複素サブバンド時系列
- leaf band ごとの beamforming
- 合成木
- 実出力

### 3.2 構造上の要点

- 木構造は 2-channel 分岐を基本とする
- 各 stage は complex PR halfband splitter とする
- 高選択度の FIR または IIR ではなく、まず FIR を正式候補とする
- leaf ごとに sample rate が異なることを正式仕様として受け入れる
- leaf ごとに beamforming 条件が異なることを正式仕様とする

### 3.3 採用しない方式

以下は正式構造の第一候補にはしない。

- 一様 `n_band` 配列へ無理に押し込む方式
- 均一 DFT bank の単なる高域ビングルーピングを最終形とする方式
- 実 wavelet のみで複素位相管理を曖昧にする方式
- beamforming 後に単純共役だけで負側を後付けする方式

---

## 4. 正式な帯域木

正式な leaf band は以下とする。

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

木構造は以下とする。

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

これは正式仕様とする。

---

## 5. formal front-end

## 5.1 採用方針

正式構造では、

- 実入力をそのまま木へ入れない
- causal analytic front-end を先頭に置く

方式を採用する。

理由は以下である。

- 正側帯域だけで複素包絡として整理しやすい
- beamforming と接続する位相規約を固定しやすい
- 実信号の Hermitian 対称を leaf 側の複素表現へ直接背負わせずに済む

## 5.2 front-end の要件

analytic front-end は少なくとも以下を満たす必要がある。

- streaming 可能
- causal
- 明示的な群遅延を持つ
- 正側帯域優勢の複素表現を与える
- 合成系を含めた全体で実出力へ戻せる

## 5.3 現時点の扱い

現時点では offline FFT-based helper があるが、
これは正式構造ではない。

正式構造では、

- FIR Hilbert transformer 系
- あるいは等価な causal analytic filter

を別部品として実装する。

---

## 6. formal stage

## 6.1 stage の役割

各 stage は、入力複素信号を

- low branch
- high branch

へ分け、`2:1` の decimation を行う。

## 6.2 正式候補

正式候補は以下である。

- complex analysis lowpass FIR
- complex analysis highpass FIR
- complex synthesis lowpass FIR
- complex synthesis highpass FIR
- paraunitary または biorthogonal PR 条件を満たす設計

## 6.3 基準系との関係

2 点 DFT による exact PR block stage は、
あくまで

- 骨格確認用
- streaming 切り分け用

である。

正式構造では、

- 選択度
- stopband attenuation
- 遅延整合
- passband flatness

を満たす複素 halfband stage を採用する。

## 6.4 stage 設計で重視する項目

1. PR が成り立つこと
2. low/high 分離が十分あること
3. 遅延が stage ごとに明示できること
4. streaming 実装が容易であること
5. 固定小数点や C++ 実装へ移しやすいこと

---

## 7. formal leaf interface

正式構造では、leaf band を長方形配列で持たない。

leaf 出力の正式インタフェースは、概念的に以下とする。

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

要するに、

- 周波数情報
- 時間原点
- root 系で見た遅延
- サンプル列

を 1 つの packet として持つ。

これを正式仕様とする。

---

## 8. formal synthesis

合成は、解析木の逆操作として定義する。

各 stage で

- upsample by 2
- synthesis filtering
- sibling branch の和

を行い、root まで戻す。

正式構造では特に以下を厳密に扱う。

- leaf 間の相対遅延
- stage ごとの群遅延
- analytic front-end を含む全体遅延
- 最終実出力時の虚部残差

したがって、各 stage と各 leaf packet は

- delay metadata

を持つ前提とする。

---

## 9. streaming 正式仕様

## 9.1 stage state

各 stage は少なくとも以下の state を持つ。

- analysis delay line
- synthesis delay line
- decimation phase
- upsampling phase
- output time cursor

## 9.2 packet emission

streaming では、すべての leaf が同じタイミングで
同じ長さの packet を出す必要はない。

正式仕様では、

- leaf ごとに非同期な packet emission
- packet 単位の時刻管理

を認める。

## 9.3 offline との整合

正式構造では、最低限

- same input
- same filter
- same delay convention

に対して、

- streaming result == offline result

が成り立たなければならない。

---

## 10. beamforming 接続の正式仕様

本系は最初から beamforming 前段であることを前提にする。

そのため各 leaf には以下をぶら下げられる必要がある。

- `used_channels`
- `channel_positions`
- `steering`
- `covariance_update_rate`
- `integration_time`
- `overlap_save_frame_size`
- `overlap_save_valid_size`
- `short_fft_size`
- `short_fft_hop_size`
- `weight_update_period`

さらに、各 leaf の内部処理は

- `output path`: overlap-save による実出力生成
- `statistics path`: `Rxx` 推定と重み更新

の 2 役割で整理する。

正式実装では旧 split-path 構造は廃止し、
`leaf_independent_one_sided` 1 系統だけを残す。
ここでの one-side は

- statistics 側では正側ビンだけを保持する
- output 側ではその one-side 重みから full complex `filter FFT` を構成する

という意味である。

したがって output path 自体は

1. shared `N`-point frame FFT
2. one-side 重みから設計した causal FIR (`M <= N - H + 1`)
3. その zero-padding 後の `N`-point `filter FFT`
4. overlap-save valid 抽出

で正式化する。

詳細は `doc/Nonuniform_FilterBank_leaf処理構造.md` に整理する。

これにより、

- `0 - 128 Hz` では広開口・長積分
- `8192 - 16384 Hz` では中央高密度サブアレイ・短積分

のような運用を正式仕様として自然に受け入れる。

---

## 11. アレイ条件

主検証条件および実用設計条件として、
使用アレイは以下とする。

- スパース直線アレイ
- 中央が密
- 端が疎

つまり、

- dense center
- sparse edge

を持つ直線形状である。

正式構造の評価は、このアレイ条件を主条件として行う。

均一 ULA は補助比較には使ってよいが、
正式評価の主条件にはしない。

---

## 12. 実装モジュール構成

正式構造のための推奨モジュール構成は以下とする。

```text
spflow/filterbank/nonuniform/
    band_spec.py
    analytic_frontend.py
    stage.py
    tree_definition.py
    analysis_tree.py
    synthesis_tree.py
    packet.py
    delay.py
    checker.py
```

推奨クラスは以下である。

- `NonuniformBandSpec`
- `BandPacket`
- `CausalAnalyticFrontend`
- `ComplexPRHalfbandStage`
- `NonuniformTreeDefinition`
- `NonuniformAnalysisTree`
- `NonuniformSynthesisTree`
- `NonuniformDelayTracker`
- `NonuniformPRChecker`

---

## 13. 実装順序

正式構造を実装する順序は以下とする。

1. `NonuniformTreeDefinition`
2. `BandPacket`
3. `ComplexPRHalfbandStage`
4. `NonuniformAnalysisTree`
5. `NonuniformSynthesisTree`
6. `NonuniformPRChecker`
7. `CausalAnalyticFrontend`
8. leaf band ごとの beamforming 接続

重要なのは、

- まず正式構造そのものを作る
- その後に front-end と beamforming を載せる

という順序を守ることだ。

---

## 14. 検証順序

正式構造の検証順序は以下とする。

1. stage 単体 PR
2. 2-level tree PR
3. 非対称 full tree PR
4. streaming / offline 一致
5. causal analytic front-end を含む PR
6. 実入力から実出力までの虚部残差確認
7. leaf band ごとの beamforming 接続確認
8. スパース直線アレイ条件での実用評価

---

## 15. 現時点での結論

今後の本命構造は、

- causal analytic front-end
- complex PR halfband stage
- 非対称不均一木
- packet ベースの leaf interface
- leaf ごとの独立 beamforming

から成る `practical nonuniform complex PR tree filter bank` とする。

block exact PR の基準木や現在の streaming 骨格は、
この正式構造を実装するための切り分け基準として残す。

以後は、

- まず正式 stage と正式 tree を設計どおりに実装し
- その後に front-end
- 最後に beamforming

の順で進める。



