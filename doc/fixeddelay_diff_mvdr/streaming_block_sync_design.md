# 小数遅延FIR主経路・diff-MVDR FIR補正枝 ストリーミング同期設計書

## 1. 目的

本設計書は、固定遅延＋小数遅延FIR主経路と、diff-MVDR FIR補正枝を同一のストリーミング block 境界で処理し、複数アレイを同時並列に処理しても時間サンプルの同期ずれが累積しないようにするための設計を定義する。

特に以下を明文化する。

- 主経路と補正枝の入力 block 境界を一致させる。
- FIR 処理を因果 FIR として定義し、`same` / `valid` / `full` の曖昧な切り出しを禁止する。
- 各 block で入力 sample 数と出力 sample 数を一致させる。
- 小数遅延FIR主経路と diff-MVDR FIR 補正枝の出力 sample index を一致させてから加算する。
- 係数更新は block 境界でのみ行い、同一 block 内で係数が変化しないようにする。
- 複数アレイ間で `global_sample_index` を共有し、下流合成時に同期を検証する。

## 2. 前提

### 2.1 処理対象

本設計の対象は以下の2系統である。

1. **小数遅延FIR主経路**
   - 各チャンネルに対して、整数遅延および小数遅延 FIR を適用する。
   - 固定整相または目標方位への固定遅延を担う。

2. **diff-MVDR FIR補正枝**
   - 主経路に対して差分的な補正を与える FIR 経路である。
   - 主経路の出力と同一 sample index 上で加算される補正信号を出力する。

最終出力は以下で定義する。

```text
y_out[n] = y_main[n] + y_diff_mvdr[n]
```

ここで、`y_main[n]` と `y_diff_mvdr[n]` は必ず同じ `n` に対応する出力でなければならない。

### 2.2 FIR tap 数

現状の基本設定は以下とする。

```text
M = 128 tap
H = M - 1 = 127 sample
```

各 FIR 処理器は、最低限 `127 sample` の過去入力履歴を保持する。

ただし、整数遅延を FIR の外側で delay line として持つ場合は、必要履歴長は以下になる。

```text
H_required = integer_delay_max + M - 1
```

主経路と補正枝で FIR 長が異なる場合は、各経路が必要な履歴長を個別に持ってよい。ただし、出力 block の sample index は必ず一致させる。

## 3. 基本方針

### 3.1 overlap-add は使わない

本設計では、FIR 128 tap 程度を想定し、FFT overlap-add ではなく **入力履歴付き direct FIR** を用いる。

したがって、block ごとの畳み込み tail を次 block に加算する tail 加算バッファは持たない。

```text
採用方式:
  direct FIR + input history

不採用:
  FFT overlap-add
  block convolution tail add
```

### 3.2 各 block は L sample 入力、L sample 出力とする

各ストリーミング block の入力を以下とする。

```text
block_index = k
block_length = L
block_start_sample = n0 = k * L
input_block = x[n0 : n0 + L - 1]
```

各経路は、必ず同じ `n0` と `L` を持つ出力 block を返す。

```text
output_block = y[n0 : n0 + L - 1]
```

初回 block で過去履歴が不足する場合でも、原則としてゼロ初期化した履歴を用いて `L sample` 出力する。初期過渡を無効扱いにしたい場合は、sample を削除するのではなく `valid_mask` または `warmup_samples` メタデータで表現する。

### 3.3 sample を捨てて同期を取らない

以下は禁止する。

```text
禁止:
  初回 block の先頭 H sample を片経路だけ捨てる
  np.convolve(..., mode="same") による暗黙の中心合わせ
  np.convolve(..., mode="valid") の切り出し位置を経路ごとに変える
  FIR 長の違いを sample の削除で吸収する
  係数更新 block で片経路だけ出力長を変える
```

同期は sample の削除ではなく、以下で管理する。

```text
管理対象:
  global_sample_index
  block_index
  block_start_sample
  block_length
  latency_tag
  coeff_version
  valid_mask
```

## 4. 因果 FIR の定義

### 4.1 単一系列の FIR 定義

各経路の FIR は因果 FIR として定義する。

```text
y[n] = Σ_{p=0}^{M-1} h[p] x[n - p]
```

この定義では、`y[n]` は入力 sample `x[n]` に対応する出力 sample である。

FIR は過去入力を参照するため、計算には履歴が必要であるが、出力 sample index は入力 sample index からずれない。

### 4.2 block FIR の定義

block 開始 sample を `n0`、block 長を `L` とする。

過去履歴を以下とする。

```text
history = x[n0 - H : n0 - 1]
```

現在 block を以下とする。

```text
input_block = x[n0 : n0 + L - 1]
```

FIR 計算用の拡張入力を以下で構成する。

```text
extended = concat(history, input_block)
```

このとき、出力は必ず `input_block` に対応する `L sample` のみとする。

```text
for i = 0 ... L-1:
    n = n0 + i
    y[n] = Σ_{p=0}^{M-1} h[p] x[n - p]
```

処理後、次 block 用の履歴を更新する。

```text
next_history = last H samples of extended
```

### 4.3 実装上の注意

`np.convolve` を使う場合、`same` は使用しない。使用するなら切り出し位置を明示する。

推奨は、FIR 処理器を共通部品化し、経路ごとに切り出し規約が変わらないようにすることである。

```python
class CausalBlockFIR:
    def __init__(self, tap_length: int):
        self.tap_length = tap_length
        self.history_length = tap_length - 1
        self.history = None

    def process(self, x_block, h):
        # x_block: shape = [n_ch, L]
        # h:       shape = [n_ch, M] or [M]
        # return:  shape = [n_ch, L]
        #
        # y[:, i] corresponds to x_block[:, i]
        ...
```

## 5. ストリーミング block の共通データ構造

### 5.1 StreamingBlock

主経路と補正枝には、同じ `StreamingBlock` を入力する。

```python
@dataclass(frozen=True)
class StreamingBlock:
    array_id: str
    block_index: int
    start_sample: int
    length: int
    fs: float
    data: np.ndarray          # shape = [n_ch, length]
    valid_mask: np.ndarray    # shape = [length]
```

`StreamingBlock` は immutable とし、主経路・補正枝の処理中に内容を変更してはならない。

### 5.2 ProcessedBlock

各経路の出力は以下の構造に統一する。

```python
@dataclass(frozen=True)
class ProcessedBlock:
    array_id: str
    path_id: str
    block_index: int
    start_sample: int
    length: int
    fs: float
    latency_tag: str
    coeff_version: int
    data: np.ndarray
    valid_mask: np.ndarray
```

主経路と補正枝を加算する前に、以下を検証する。

```text
必須一致条件:
  array_id
  block_index
  start_sample
  length
  fs
  latency_tag
  coeff_version または compatible_coeff_version
```

## 6. 主経路と補正枝の同期設計

### 6.1 経路構成

本設計では、主経路と補正枝を同一 block から分岐させる。

```text
                 ┌───────────────────────────┐
input stream ───▶│ StreamingBlock Builder     │
                 │ start_sample = n0          │
                 │ length = L                 │
                 └─────────────┬─────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐        ┌──────────────────────┐
        │ Main FIR Path   │        │ diff-MVDR FIR Path    │
        │ fractional FIR  │        │ correction FIR        │
        └────────┬────────┘        └───────────┬──────────┘
                 │                             │
                 ▼                             ▼
        y_main[n0:n0+L-1]          y_diff[n0:n0+L-1]
                 │                             │
                 └─────────────┬───────────────┘
                               ▼
                    y_out = y_main + y_diff
```

### 6.2 主経路

主経路は以下を行う。

1. 整数遅延を適用する。
2. 小数遅延 FIR を適用する。
3. チャンネル和または beam 出力を生成する。

主経路の出力は以下で定義する。

```text
y_main[n] = MainPathFIR(x, n)
```

このとき、`y_main[n]` は `start_sample <= n < start_sample + L` の範囲で `L sample` 出力される。

### 6.3 diff-MVDR FIR補正枝

補正枝は、主経路と同じ `StreamingBlock` と同じ `start_sample` を用いる。

補正枝の出力は以下で定義する。

```text
y_diff_mvdr[n] = DiffMVDRFIR(x, n)
```

補正枝も `start_sample <= n < start_sample + L` の範囲で `L sample` 出力する。

### 6.4 合成条件

最終加算は以下の条件を満たす場合のみ許可する。

```text
y_out[n] = y_main[n] + y_diff_mvdr[n]

where:
  y_main.start_sample == y_diff_mvdr.start_sample
  y_main.length       == y_diff_mvdr.length
  y_main.latency_tag  == y_diff_mvdr.latency_tag
```

条件を満たさない場合は例外を発行し、暗黙の切り詰め・ゼロ詰め・sample shift は行わない。

## 7. 遅延・群遅延の扱い

### 7.1 FIR 128 tap の群遅延

線形位相 FIR の tap 数を `M = 128` とすると、群遅延は以下になる。

```text
G = (M - 1) / 2 = 63.5 sample
```

これは累積するずれではなく、固定の処理遅延である。

ただし、`63.5 sample` は整数 sample ではないため、設計書上は以下のどちらかを選ぶ。

### 7.2 推奨案A: 128 tap を維持し、0.5 sample を latency_tag に含める

128 tap を維持する場合、主経路・補正枝の両方で共通の遅延基準を持つ。

```text
latency_tag:
  common_algorithmic_delay = D_base + 63.5 sample
```

この場合、出力配列の index は整数 sample のまま扱う。0.5 sample の固定遅延は、係数設計および位相基準で管理する。

主経路と補正枝は、同じ `latency_tag` を持つ限り sample 同期しているとみなす。

### 7.3 推奨案B: 129 tap に変更し、群遅延を整数化する

整数 sample の群遅延を重視する場合は、FIR tap 数を奇数にする。

```text
M = 129
G = (M - 1) / 2 = 64 sample
```

この場合、履歴長は以下になる。

```text
H = 128 sample
```

block 同期設計は 128 tap の場合と同じだが、遅延管理が単純になる。

### 7.4 主経路と補正枝で FIR 長が異なる場合

主経路と補正枝で FIR 長が異なる場合は、短い側に明示的な純遅延を追加するか、係数設計時に同じ `common_algorithmic_delay` へ揃える。

禁止する対応は以下である。

```text
禁止:
  FIR 長の差を出力 sample の削除で合わせる
  補正枝だけ 1 block 遅らせて加算する
  主経路だけ valid 出力にする
```

許可する対応は以下である。

```text
許可:
  係数設計で同一 latency_tag に揃える
  短い FIR に純遅延を追加する
  delay compensation block を明示的に挿入する
```

## 8. 係数更新タイミング

### 8.1 block 境界更新

FIR 係数は block 境界でのみ更新する。

```text
許可:
  block k の開始時点で係数 version v を latch
  block k の全 sample に version v を適用

禁止:
  block k の途中で係数を変更する
  主経路と補正枝で係数 version の切り替え sample が異なる
```

### 8.2 共分散更新と係数適用の分離

diff-MVDR の係数が共分散行列から更新される場合、現在 block の観測値で更新した係数を同じ block に即時適用しない。

推奨規約は以下である。

```text
block k:
  入力 x_k を取得
  係数 version v_k を適用して y_k を出力
  x_k を用いて共分散 R を更新
  必要なら係数 version v_{k+1} を生成

block k+1:
  係数 version v_{k+1} を latch して適用
```

この設計により、係数生成と係数適用の因果関係が明確になり、主経路と補正枝の更新境界も一致する。

### 8.3 係数更新の滑らか化

係数の急変による段差が問題になる場合は、block 境界で新旧係数の crossfade を行ってもよい。

ただし、crossfade も主経路と補正枝で同じ sample index に対して定義する。

```text
h_apply[n] = (1 - a[n]) h_old + a[n] h_new
```

ここで `a[n]` は block 内 sample index に対する滑らかな係数である。主経路と補正枝で別々の sample offset を使ってはならない。

## 9. 複数アレイ並列処理の同期設計

### 9.1 global_sample_index

複数アレイを同時並列で処理する場合、各アレイは同じ `global_sample_index` を共有する。

```text
array A, block k:
  start_sample = k * L

array B, block k:
  start_sample = k * L
```

アレイごとにスレッドやプロセスが異なっても、`start_sample` と `length` は一致させる。

### 9.2 バリア同期

下流処理は、全アレイの同じ `block_index` が揃ってから実行する。

```text
ArrayBlockBarrier:
  wait until all array outputs for block k are ready
  verify start_sample and length
  pass synchronized block set downstream
```

### 9.3 欠損 block の扱い

あるアレイの block が欠損した場合、他アレイの sample を詰めて同期を取ってはならない。

許可する対応は以下である。

```text
対応案:
  欠損アレイを zero block + invalid_mask として流す
  欠損アレイを hold block + invalid_mask として流す
  block k 全体を downstream invalid とする
  例外を発行して処理停止する
```

いずれの場合も、`start_sample` と `length` は維持する。

## 10. 実装クラス案

### 10.1 クラス構成

```text
StreamingBlockBuilder
  入力ストリームを block 化し、start_sample / block_index を付与する。

HistoryStore
  array_id / path_id / channel ごとの FIR 履歴を保持する。

CausalBlockFIR
  因果 FIR を block 入力・block 出力で実行する共通部品。

FractionalDelayMainPath
  小数遅延 FIR 主経路を実行する。

DiffMVDRCorrectionPath
  diff-MVDR FIR 補正枝を実行する。

CoefficientLatch
  block 境界で使用する係数 version を固定する。

AlignedPathCombiner
  主経路と補正枝の start_sample / length / latency_tag を検証して加算する。

ArrayBlockBarrier
  複数アレイの同一 block_index を揃える。
```

### 10.2 擬似コード

```python
def process_array_block(array_id: str, x_block: np.ndarray, block_index: int):
    n0 = block_index * block_length

    frame = StreamingBlock(
        array_id=array_id,
        block_index=block_index,
        start_sample=n0,
        length=block_length,
        fs=fs,
        data=x_block,
        valid_mask=np.ones(block_length, dtype=bool),
    )

    coeff_set = coeff_latch.get_for_block(array_id, block_index)

    y_main = main_path.process(
        frame=frame,
        coeff=coeff_set.main,
    )

    y_diff = diff_mvdr_path.process(
        frame=frame,
        coeff=coeff_set.diff_mvdr,
    )

    y_out = aligned_combiner.add(y_main, y_diff)

    covariance_updater.update(frame)
    coeff_latch.prepare_next_if_needed(array_id, block_index)

    return y_out
```

### 10.3 合成器の検証

```python
class AlignedPathCombiner:
    def add(self, main: ProcessedBlock, diff: ProcessedBlock) -> ProcessedBlock:
        require(main.array_id == diff.array_id)
        require(main.block_index == diff.block_index)
        require(main.start_sample == diff.start_sample)
        require(main.length == diff.length)
        require(main.fs == diff.fs)
        require(main.latency_tag == diff.latency_tag)

        data = main.data + diff.data
        valid_mask = main.valid_mask & diff.valid_mask

        return ProcessedBlock(
            array_id=main.array_id,
            path_id="main_plus_diff_mvdr",
            block_index=main.block_index,
            start_sample=main.start_sample,
            length=main.length,
            fs=main.fs,
            latency_tag=main.latency_tag,
            coeff_version=max(main.coeff_version, diff.coeff_version),
            data=data,
            valid_mask=valid_mask,
        )
```

## 11. メタデータ規約

### 11.1 必須メタデータ

各 block には以下のメタデータを必ず付与する。

```text
array_id
path_id
block_index
start_sample
length
fs
latency_tag
coeff_version
valid_mask
```

### 11.2 latency_tag

`latency_tag` は、出力 sample がどの遅延基準で定義されているかを表す。

例を以下に示す。

```text
latency_tag = "fixed_delay_plus_frac_fir:G=63.5:Dbase=..."
```

または構造化して保持する。

```python
@dataclass(frozen=True)
class LatencyTag:
    base_delay_samples: float
    fir_group_delay_samples: float
    fractional_reference_samples: float
    convention: str
```

主経路と補正枝は同一の `LatencyTag` を持つ必要がある。

## 12. 試験項目

### 12.1 streaming / offline 一致試験

同一入力に対して、以下を比較する。

1. 全 sample を一括で direct FIR 処理した結果
2. block 分割して履歴付き direct FIR 処理した結果

評価条件:

```text
max_abs_error < tolerance
```

この試験は主経路・補正枝の両方で実施する。

### 12.2 block 境界インパルス試験

block 境界直前・境界上・境界直後に impulse を置く。

```text
n = L - 1
n = L
n = L + 1
```

期待結果:

```text
streaming 処理結果が offline 処理結果と一致する
境界で sample の欠落・重複がない
```

### 12.3 主経路・補正枝同期試験

補正枝に既知の FIR を設定し、主経路と補正枝の出力 sample index が一致することを確認する。

例:

```text
main path: identity FIR
correction path: identity FIR * small_gain
expected: y_out[n] = (1 + small_gain) x[n]
```

block 境界で位相または振幅の段差が発生しないことを確認する。

### 12.4 複数アレイ同期試験

複数アレイに同一 impulse または既知信号を入力し、各アレイの `start_sample` と出力 peak index が一致することを確認する。

```text
array A output peak index == array B output peak index
array A start_sample == array B start_sample
array A block_index == array B block_index
```

### 12.5 係数更新境界試験

係数 version を block 境界で切り替え、以下を確認する。

```text
block k 内では coeff_version が一定
block k と block k+1 の境界でのみ version が変わる
主経路と補正枝の version 切り替え start_sample が一致する
```

### 12.6 FIR tap 数差分試験

主経路と補正枝で FIR tap 数が異なる構成を試験する。

期待結果:

```text
latency_tag が一致する場合のみ加算可能
latency_tag が不一致の場合は例外発行
暗黙の sample shift は行われない
```

## 13. 実装上の禁止事項

以下は同期ずれの原因になるため禁止する。

```text
1. np.convolve(..., mode="same") を本番処理に直接使う。
2. 経路ごとに valid 出力の切り出し位置を変える。
3. 初回 block の出力 sample を片経路だけ削除する。
4. FIR 履歴更新を処理途中に行い、同一 block 内で入力履歴が変化する。
5. 係数を block 途中で更新する。
6. 主経路と補正枝で異なる block_length を使う。
7. アレイごとに block_index の原点を変える。
8. 欠損 block を詰めて後続 sample を前倒しする。
9. FIR 長の差を暗黙の sample trim で吸収する。
10. half-sample delay を無視して latency_tag を一致扱いにする。
```

## 14. 推奨実装方針

現状の FIR 128 tap 設計では、以下を推奨する。

```text
FIR方式:
  direct FIR + input history

履歴長:
  M - 1 = 127 sample
  ただし整数遅延を外部 delay line で持つ場合は integer_delay_max を加算

block入出力:
  L sample input -> L sample output

主経路と補正枝:
  同じ StreamingBlock から分岐
  同じ start_sample / length / latency_tag で出力

係数更新:
  block境界で latch
  共分散更新結果は原則として次 block から適用

複数アレイ:
  global_sample_index と block_index を共有
  ArrayBlockBarrier で同一 block を揃えてから下流へ渡す
```

## 15. 結論

FIR 128 tap 程度であれば、FFT overlap-add ではなく、入力履歴を持つ direct FIR が適している。

この場合、同期ずれを防ぐために重要なのは、FIR 処理方式そのものではなく、以下の設計規約を守ることである。

```text
1. 主経路と補正枝を同一 StreamingBlock から分岐する。
2. 各経路は L sample 入力に対して L sample 出力する。
3. 出力 sample index を start_sample / length で明示する。
4. FIR の切り出し規約を CausalBlockFIR に集約する。
5. 群遅延・小数遅延は latency_tag で管理する。
6. 係数更新は block 境界で同期する。
7. 複数アレイは global_sample_index と ArrayBlockBarrier で同期する。
```

以上により、小数遅延FIR主経路と diff-MVDR FIR 補正枝を同じストリーミング block 境界で扱い、逐次処理による累積的な sample ずれを防止できる。
