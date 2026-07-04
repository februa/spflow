# ComplexPRHalfbandStage sinc目標最適化方針

## 1. 目的

本書は、
`ComplexPRHalfbandStage` の係数設計を

- Daubechies family そのものを目標にする

のではなく、

- sinc 系 lowpass を目標応答として使う

方針へ進めるための設計メモである。

---

## 2. 基本方針

採用する方針は以下である。

1. `windowed-sinc` で lowpass の目標応答を作る
2. その目標応答への近さを、正式な設計評価軸として持つ
3. 実際の候補は
   - paraunitary
   - PR
   - streaming 一致
   を維持したまま、その目標へ近づける

つまり、

- sinc をそのまま係数として使う

のではなく、

- sinc を target response として使う constrained design

へ移る。

---

## 3. 今回追加したもの

今回追加した実装は以下である。

- `src/spflow/filterbank/design/sinc_target.py`
- `examples/filterbank/complex_halfband_stage_sinc_target_design_eval.py`
- `tests/filterbank/test_sinc_target_design.py`

### 3.1 `sinc_target.py`

役割:

- halfband 用の windowed-sinc target lowpass を定義する
- complemented power target を作る
- paraunitary 候補が target にどれだけ近いかを数値化する

評価指標:

- `fullband_rms_error`
- `passband_rms_error`
- `stopband_rms_error`
- `transition_rms_error`

### 3.2 `complex_halfband_stage_sinc_target_design_eval.py`

役割:

- 既存の paraunitary 候補群を
  sinc target に対する近さで順位付けする
- 現段階の baseline artifact を保存する

---

## 4. この段階でまだやっていないこと

今回の段階では、

- sinc target を formal に定義した
- その target に対する距離を測れるようにした

までである。

まだ実施していないのは、

- 自由パラメータを本当に最適化して
  sinc target へ寄せる constrained optimizer

である。

したがって現時点の位置付けは、

- 設計方針の転換は完了
- 最適化器の本体は次段階

である。

---

## 5. 初期の読み取り

既存 Daubechies QMF 候補を
sinc target に対して比べると、

- order が高い候補ほど target への RMS 誤差は小さくなる

傾向が確認できる。

これは、

- sinc target 自体は合理的な設計目標として使える

ことを示している。

少なくとも、

- target 指標がノイズ的で順位が不安定

という問題は見えていない。

---

## 6. 今後の最適化方針

今後は、以下の順で進めるのが妥当である。

1. sinc target との誤差指標を正式採用する
2. paraunitary 制約を壊さないパラメータ化を 1 つ定める
3. そのパラメータ空間で
   - target 誤差
   - stopband
   - PR 誤差
   を同時に最適化する

候補となる方式は以下である。

1. paraunitary lattice / rotation chain を直接最適化する
2. halfband power response を sinc target に寄せ、
   spectral factorization で QMF を作る
3. biorthogonal を許して analysis / synthesis を別最適化する

現段階では、

- v1 正式版は paraunitary を優先

なので、
まずは

- paraunitary parameterization + sinc target objective

で進めるのが本線である。

---

## 7. 現時点での判断

現時点では、

- `sinc 直接採用` は非採用
- `sinc を目標応答として使う constrained design` は正式に採用

と整理してよい。

したがって以後の係数設計は、

- 「どの family が最も PR を満たすか」

だけではなく、

- 「どの候補が sinc target に近いか」

も同時に評価して進める。
