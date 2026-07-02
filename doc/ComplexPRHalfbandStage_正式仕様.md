# ComplexPRHalfbandStage 正式仕様

## 1. 目的

本書は、不均一フィルタバンク正式構造における
`ComplexPRHalfbandStage` の仕様を固定するための文書である。

ここで固定したいのは、

- 何を入出力とするか
- どの周波数規約で child band を表現するか
- 完全再構成をどう定義するか
- streaming で何を state として持つか
- どの性能を満たせば正式版とみなすか

である。

本書は係数そのものをまだ定めない。
定めるのは

- stage の役割
- 数学的契約
- 実装契約
- 検証条件

である。

---

## 2. stage の位置付け

`ComplexPRHalfbandStage` は、
不均一複素フィルタバンク木の 1 段を構成する基本部品である。

各 stage は、1 本の親 packet を

- lower child
- upper child

へ分割し、両 child を `decimation by 2` した複素 packet として出力する。

正式構造では、木全体はこの stage を再帰接続して構成する。

---

## 3. 採用する signal convention

## 3.1 基本規約

正式版では、各 `BandPacket` の複素サンプル列は

- その band の lower edge を基準周波数とする
- analytic な複素サブバンド信号

として解釈する。

すなわち、band が

- `[f_low_hz, f_high_hz]`

であるとき、packet 内の複素信号は

- 物理周波数 `f_low_hz` を `0 Hz` に周波数シフトした表現

である。

この規約を正式採用とする。

## 3.2 この規約を採用する理由

lower-edge 基準にすると、

- 親 band `[f_low, f_high]`
- 中央 `f_mid = (f_low + f_high) / 2`

に対して、

- lower child はそのまま `[0, B/2]`
- upper child は `-f_mid` だけ追加シフトすれば `[0, B/2]`

の形で統一できる。

これにより、各 child packet を

- 同じ analytic baseband 規約

で再帰的に扱える。

## 3.3 非採用規約

以下は正式規約として採用しない。

- child packet を中心周波数基準の両側帯 baseband として持つ方式
- upper child を alias したまま raw bandpass representation として持つ方式
- branch ごとに異なる曖昧な位相基準を使う方式

---

## 4. packet 契約

親 packet は少なくとも以下を持つ。

```text
band_id
f_low_hz
f_high_hz
sample_rate_hz
time_origin_at_root_rate
delay_samples_at_root_rate
complex_samples
```

ここで

- `sample_rate_hz` はその packet の複素サンプル列のレート
- `f_high_hz - f_low_hz = sample_rate_hz / 2`

を正式仕様とする。

すなわち、各 packet は

- complex one-sided representation
- 2 倍余裕を持つ一様時間サンプリング

である。

---

## 5. child packet の定義

親 band を

- `[f_low, f_high]`
- `B = f_high - f_low`
- `f_mid = f_low + B/2`

とする。

stage 出力は以下とする。

### 5.1 lower child

- 周波数帯域: `[f_low, f_mid]`
- lower-edge 基準周波数: `f_low`
- sample rate: `sample_rate / 2`

### 5.2 upper child

- 周波数帯域: `[f_mid, f_high]`
- lower-edge 基準周波数: `f_mid`
- sample rate: `sample_rate / 2`

重要なのは、upper child も

- 自分自身の lower edge を `0 Hz` に落とした analytic packet

として出すことだ。

したがって stage は内部的に、

- upper-half を抽出する
- child lower-edge へ周波数シフトする
- decimate by 2 する

処理を含む。

---

## 6. analysis の正式仕様

## 6.1 抽象仕様

analysis は、親 packet `x[n]` を

- lower child `y0[m]`
- upper child `y1[m]`

へ変換する。

この変換は、

- complex FIR analysis filtering
- branch-wise frequency translation
- decimation by 2

から成る。

## 6.2 lower child の処理

lower child は概念的に

```text
parent packet
    -> lowpass analysis filter
    -> decimation by 2
    -> lower-edge referenced analytic child
```

で定義する。

## 6.3 upper child の処理

upper child は概念的に

```text
parent packet
    -> upper-half extraction
    -> child lower-edge への周波数シフト
    -> decimation by 2
    -> lower-edge referenced analytic child
```

で定義する。

## 6.4 実装上の許容

この処理は実装上、

- 明示的な複素乗算
- 変調込み FIR
- polyphase 化された branch 演算

のどれで実現してもよい。

ただし、外から見た契約は同じでなければならない。

---

## 7. synthesis の正式仕様

synthesis は、2 つの child packet を受け取り、
親 packet を復元する。

概念的には

- upsample by 2
- child 規約から親規約への逆周波数シフト
- synthesis filtering
- 2 branch の和

で定義する。

upper child 側は、

- analysis 時に child lower-edge へ落としている

ため、synthesis では逆向きに

- parent 上側半分の位置へ周波数を戻す

必要がある。

---

## 8. 完全再構成の正式定義

`ComplexPRHalfbandStage` は、浮動小数点参照実装では
以下を満たさなければならない。

```text
synthesis(analysis(x)) = c * x[n - d]
```

ここで

- `c` は既知の定数スケール
- `d` は既知の整数遅延

である。

正式版では、原則として

- `c = 1`

を採用する。

すなわち stage 単体で

- unit-gain PR

を満たすことを正式要求とする。

また、`d` は

- stage metadata として明示できる整数遅延

でなければならない。

---

## 9. 採用する PR 形式

正式版 stage は、第一候補として

- 複素 FIR
- critically sampled
- paraunitary

方式を採用する。

理由は以下である。

- PR 条件を明示しやすい
- エネルギー挙動が分かりやすい
- streaming 実装が比較的単純
- 数値安定性の見通しがよい

biorthogonal 方式は将来的な拡張候補として残すが、
初期の正式版仕様としては第一候補にしない。

したがって、

- 正式版 `ComplexPRHalfbandStage` v1

は

- complex FIR paraunitary halfband stage

として設計する。

---

## 10. FIR 仕様

正式版 stage の FIR 仕様は以下とする。

- analysis low / high は同一 tap length
- synthesis low / high も同一 tap length
- 4 branch の係数は複素係数を許容する
- tap length は偶数長・奇数長を禁止しない
- ただし最終的な stage 遅延 `d` は整数で明示できること

この段階では係数長を固定しない。

ただし正式設計の要求として、

- stopband attenuation
- passband ripple
- transition width
- group delay

のトレードオフを後段で最適化可能な形で持つことを求める。

---

## 11. 位相規約

正式版では、位相規約を以下のように固定する。

1. packet は必ず lower-edge 基準の analytic 信号
2. upper child は child lower-edge 基準へ落としてから出力
3. synthesis はその逆規約で親へ戻す
4. stage は `delay_samples_at_root_rate` を更新する
5. child sample index 0 の物理時刻を metadata で追跡する

要するに、

- サンプル列そのもの
- 周波数参照
- 時刻参照

を同時に固定する。

これを曖昧にしないことが正式版で最も重要である。

---

## 12. delay / time_origin 契約

各 stage は child packet に対して以下を出力する。

- `time_origin_at_root_rate`
- `delay_samples_at_root_rate`

ここで、

- `time_origin_at_root_rate` は child packet の sample index 0 が指す root 系の時刻
- `delay_samples_at_root_rate` はその packet が保持する既知の累積遅延

である。

正式版では、

- delay は浮動小数点ではなく root-rate 整数サンプルで表現する

方針を採用する。

理由は、

- 実装が簡潔
- streaming 検証しやすい
- beamforming 後の整合確認で扱いやすい

ためである。

---

## 13. streaming 契約

各 stage は stateful processor として実装する。

最低限持つ state は以下である。

- analysis delay line
- synthesis delay line
- branch modulation phase
- decimation phase
- upsampling phase
- emitted child packet の time cursor

正式版では、

- chunk 境界がどこに来ても
- offline と同一結果を返す

ことを要求する。

すなわち、

```text
offline_analysis == streaming_analysis
offline_synthesis == streaming_synthesis
```

が成立しなければならない。

---

## 14. 性能要求

正式版 stage に求める最低要求を以下とする。

### 14.1 PR 要求

- float32 参照実装で `max_abs_error <= 1e-10`
- `rms_error <= 1e-12`

### 14.2 周波数特性要求

初期目標として以下を置く。

- passband ripple: `<= 0.1 dB`
- stopband attenuation: `>= 80 dB`

この値は、leaf beamforming まで進めた段階で再調整してよいが、
正式版 stage の初期設計目標として採用する。

### 14.3 streaming 要求

- 任意 chunk 分割で offline と一致
- chunk 分割依存の境界 jump を作らない

---

## 15. 受け入れ試験

正式版として受け入れるには、少なくとも以下を通す。

1. stage 単体の PR 試験
2. stage 単体の streaming / offline 一致
3. complex tone sweep
4. 実余弦波から analytic front-end を通した後の stage 応答確認
5. 2-level tree での PR
6. full nonuniform tree での PR

---

## 16. 非採用事項

正式版 stage では以下を採用しない。

- raw alias された upper branch をそのまま child とする
- child ごとに異なる ad-hoc な位相補正を後付けする
- delay を暗黙の実装依存にする
- offline だけ合って streaming でずれる構造
- packet が lower-edge 基準か center 基準か曖昧な構造

---

## 17. 現時点での結論

`ComplexPRHalfbandStage` 正式版の仕様は、以下で固定する。

- complex FIR paraunitary halfband stage
- critically sampled
- child packet は lower-edge 基準の analytic complex packet
- upper child は child lower-edge へシフトしてから出力
- synthesis は逆シフトを含めて unit-gain PR
- delay / time_origin を root-rate 整数で明示
- streaming と offline は必ず一致

以後の設計・実装は、この仕様を満たすかどうかで判断する。
