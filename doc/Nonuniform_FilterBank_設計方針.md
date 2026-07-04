# Nonuniform FilterBank 設計方針

## 1. 目的

本書は、既存の均一帯域 PRDFT フィルタバンク系とは別系統として、

- 複素表現
- 完全再構成
- streaming 対応
- ウェーブレット的な木構造
- 不均一帯域分割

を満たす新しいフィルタバンクを設計するための方針書である。

既存の PRDFT / Polyphase DFT / prototype-bank 系は維持する。
それらは今後も

- 基準実装
- 比較対象
- 切り分け用参照系

として残す。

本系は「既存系の拡張」ではなく、
別系統の新規設計として進める。

---

## 2. 設計対象

想定する基本条件は以下とする。

- 入力は実信号を主対象とする
- 内部表現は複素サブバンド信号とする
- 出力は最終的に実信号へ戻せることを目標とする
- sampling rate はまず `fs = 32768 Hz` を基準条件とする
- 正側帯域の設計目標は `0 Hz` から `16384 Hz` までとする

不均一帯域分割は以下を採用する。

| 帯域 | 目標分解能 |
|---|---:|
| `0 - 128 Hz` | `1 Hz` |
| `128 - 256 Hz` | `1 Hz` |
| `256 - 512 Hz` | `2 Hz` |
| `512 - 1024 Hz` | `2 Hz` |
| `1024 - 2048 Hz` | `4 Hz` |
| `2048 - 4096 Hz` | `8 Hz` |
| `4096 - 8192 Hz` | `16 Hz` |
| `8192 - 16384 Hz` | `32 Hz` |

---

## 3. この帯域分割を採用する理由

この分割は、実用上の beamforming 要求と整合している。

- 低域では広い開口を使いたいため、細かい分解能が必要
- 高域では狭い開口を使うため、過剰な分解能は不要
- 高域では使用チャネル数も減るため、解析側も粗くてよい

したがって、

- 全帯域を一様に細かく見る

のではなく、

- 低域は細かく
- 高域は粗く
- その代わり帯域ごとのサブバンド時間レートも変える

方が自然である。

また、与えられた帯域境界はほぼ dyadic な分割になっており、
木構造フィルタバンクと相性がよい。

---

## 4. 基本方針

本系は、ウェーブレット的な

- 2 分岐
- decimation by 2
- 逐次的な帯域分割

を基本単位にする。

ただし、通常の実数 wavelet のままでは

- 複素表現
- 位相の明示的な追跡
- bandwise beamforming との接続

が扱いにくい。

そのため本系では、

- 複素 PR halfband splitter を基本部品とする
- それを木構造に接続して不均一分割を作る
- 各 leaf band は複素サブバンド時系列として扱う

方針とする。

要するに、

- 実装構造は wavelet tree 的
- 扱う信号は complex subband
- 再構成は解析木と対になる合成木で行う

という設計である。

---

## 5. 推奨する木構造

最上位から順に、低域側だけを深く掘る非対称木を採用する。

概念図は以下である。

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

この構造により、要求された 8 帯域を素直に表現できる。

---

## 6. 各 leaf band の設計目安

`fs = 32768 Hz` を前提に、各帯域の leaf rate と分解能の目安を以下とする。

| 帯域 | 帯域幅 | 木の深さ | leaf rate の目安 | 目標分解能 | 局所解析長の目安 |
|---|---:|---:|---:|---:|---:|
| `0 - 128 Hz` | `128 Hz` | 7 | `256 Hz` | `1 Hz` | `256` |
| `128 - 256 Hz` | `128 Hz` | 7 | `256 Hz` | `1 Hz` | `256` |
| `256 - 512 Hz` | `256 Hz` | 6 | `512 Hz` | `2 Hz` | `256` |
| `512 - 1024 Hz` | `512 Hz` | 5 | `1024 Hz` | `2 Hz` | `512` |
| `1024 - 2048 Hz` | `1024 Hz` | 4 | `2048 Hz` | `4 Hz` | `512` |
| `2048 - 4096 Hz` | `2048 Hz` | 3 | `4096 Hz` | `8 Hz` | `512` |
| `4096 - 8192 Hz` | `4096 Hz` | 2 | `8192 Hz` | `16 Hz` | `512` |
| `8192 - 16384 Hz` | `8192 Hz` | 1 | `16384 Hz` | `32 Hz` | `512` |

ここでの `leaf rate` は、
各段で `decimation by 2` を行った場合の nominal rate である。

`局所解析長` は、各 leaf band 上で

- 共分散推定
- overlap-save beamforming
- weight 更新

を考えるときの局所 FFT 長または局所観測長の目安である。

この値は固定仕様ではなく、初期設計の出発点とする。

---

## 7. 複素表現の方針

本系では、各 leaf 出力を複素サブバンド信号として扱う。

目的は以下である。

- 帯域ごとの位相を明示的に保持する
- steering / beamforming と直接接続しやすくする
- 実数直交変換だけでは扱いにくい中心周波数回りの複素包絡として整理する

このとき重要なのは、

- 単に実 wavelet を並べるだけでは不十分
- 解析側と合成側で複素位相規約を固定する必要がある

という点である。

したがって、各 2 分岐ステージは

- 振幅特性
- 群遅延
- 低域枝 / 高域枝の複素位相
- downsample 後の時間原点

を明示的に持つ必要がある。

---

## 8. 完全再構成の考え方

完全再構成の難所は、

- 不均一木
- 複素枝
- streaming
- 位相整合

が同時に入る点である。

そのため、本系では最初から

- paraunitary または biorthogonal な 2-channel PR stage
- stage 単位で PR を検証
- 木全体でも PR を検証

という順序で積み上げる。

再構成は

1. 各 leaf を対応する合成枝へ入力
2. 各 stage で upsample + synthesis filtering
3. 上位ノードで和成分を合流
4. root で全帯域時間信号を復元

という木構造の逆操作で行う。

重要なのは、leaf band 単体での位相だけでなく、

- 兄弟枝どうしの相対遅延
- 異なる深さの leaf 間の整合

まで含めて合わせることである。

---

## 9. streaming 実装方針

本系は offline 変換ではなく、逐次入力で動作できる必要がある。

そのため、各 stage は stateful な streaming processor として実装する。

各 stage が持つべき状態は以下である。

- 解析 FIR の遅延線
- 合成 FIR の遅延線
- downsample / upsample の位相位置
- 出力時刻の基準

分析木全体では、

- 入力チャンクを root stage に入れる
- 各 stage が必要なタイミングで子ノードへ出力する
- leaf 側では band ごとの rate で非同期に出力が現れる

という形を許容する。

したがって、leaf 出力のインタフェースは一様配列ではなく、
概念的には以下が必要である。

```text
BandPacket:
    band_id
    band_range_hz
    sample_rate
    complex_samples
    time_origin
```

つまり、uniform bank のような

- `(n_ch, n_band, n_sample)` の長方形配列

を前提にしない。

---

## 10. 実入力・実出力との関係

入力は実信号を主対象とするが、内部は複素表現とする。

このとき方式候補は 2 つある。

### 10.1 方式A: 実入力をそのまま PR tree に入れる

利点:

- front-end が単純

課題:

- 高域枝 / 低域枝の複素表現規約が難しい
- 実 wavelet 的な対称だけでは beamforming 接続が弱い

### 10.2 方式B: front-end で analytic 化してから木へ入れる

利点:

- 正側帯域の複素包絡として整理しやすい
- 位相規約を管理しやすい
- beamforming の複素表現と自然に接続できる

課題:

- analytic front-end 自体の遅延と近似誤差を管理する必要がある

現時点では、位相整理のしやすさを優先し、
方式Bを第一候補とする。

ただし、最終的には

- front-end を含めた全体 PR
- 実出力復元時の虚部残差

で判断する。

---

## 11. beamforming との接続方針

本系は filterbank 単体で終わらず、
bandwise beamforming の前段として使うことを前提にする。

そのため各 leaf は、少なくとも以下を持つ。

- `band_id`
- `f_low`, `f_high`
- `center_frequency`
- `sample_rate`
- `target_resolution`
- `used_channels`

さらに beamforming 側では、leaf ごとに

- 使用チャネル集合
- steering
- 共分散更新周期
- overlap-save frame size
- weight 更新周期

を個別に設定できる構造にする。

これにより、

- 低域 leaf は大開口・長積分
- 高域 leaf は小開口・短積分

という実用設計へ自然に接続できる。

---

## 12. 初期 API の方向性

クラス名は既存 PRDFT 系と分離し、例えば以下を候補とする。

```text
nonuniform/
    band_spec.py
    tree.py
    stage.py
    analysis.py
    synthesis.py
    checker.py
```

想定クラス:

- `NonuniformBandSpec`
- `ComplexPRHalfbandStage`
- `NonuniformAnalysisTree`
- `NonuniformSynthesisTree`
- `NonuniformPRChecker`
- `BandPacket`

`NonuniformBandSpec` は少なくとも以下を持つ。

- `band_id`
- `f_low_hz`
- `f_high_hz`
- `target_resolution_hz`
- `tree_depth`
- `nominal_sample_rate_hz`

---

## 13. 段階的な検証手順

既存 PRDFT 系と同じく、段階確認で積み上げる。

### 13.1 Stage 単体の PR 確認

まず 1 個の 2-channel 複素 PR stage だけを作る。

確認項目:

- `synthesis(analysis(x)) = x`
- complex tone に対する振幅誤差
- complex tone に対する位相誤差
- streaming / offline 一致

### 13.2 2 段木の PR 確認

次に 2 level の木にする。

確認項目:

- 全葉を通した再構成誤差
- 各 leaf 単独励振での出力遅延
- sibling 間の位相整合

### 13.3 非対称木の PR 確認

低域側だけ深く掘る非対称木へ進む。

確認項目:

- 深さの異なる leaf 間での再構成整合
- 合成後の遅延整合
- block 境界での不連続の有無

### 13.4 実信号復元の確認

実余弦波、複数 tone、雑音で確認する。

確認項目:

- 実出力の虚部 RMS
- 周波数 sweep での再構成誤差
- 境界近傍周波数での段差

### 13.5 beamforming 接続の確認

最後に各 leaf band で bandwise beamforming を接続する。

確認項目:

- leaf ごとの direct 計算との一致
- 全体合成後の出力整合
- uniform 基準系との比較

---

## 14. 初期実装の優先順位

初期実装は以下の順で進める。

1. `BandSpec` と帯域木定義
2. 単一の complex PR halfband stage
3. 2 level analysis/synthesis tree
4. 非対称木での full PR
5. streaming 実装
6. 実入力 / 実出力 wrapper
7. leaf band ごとの beamforming 接続

この順序であれば、

- まず PR があるか
- 次に木構造で崩れないか
- 最後に実用 beamforming へ接続できるか

を分離して確認できる。

---

## 15. 現時点での結論

新しい不均一フィルタバンクは、

- 既存 PRDFT 系から移行するのではなく
- 別系統として新設する

のが妥当である。

その構造は、

- 複素
- 完全再構成
- streaming 対応
- ウェーブレット的な非対称木

を満たす `complex PR tree filter bank` とする。

帯域分割は以下を設計目標とする。

- `0 - 128 Hz`: `1 Hz`
- `128 - 256 Hz`: `1 Hz`
- `256 - 512 Hz`: `2 Hz`
- `512 - 1024 Hz`: `2 Hz`
- `1024 - 2048 Hz`: `4 Hz`
- `2048 - 4096 Hz`: `8 Hz`
- `4096 - 8192 Hz`: `16 Hz`
- `8192 - 16384 Hz`: `32 Hz`

以後は、この設計方針に沿って

- stage 単体
- 小木
- 非対称 full tree
- 実信号復元
- beamforming 接続

の順で段階検証を行う。
