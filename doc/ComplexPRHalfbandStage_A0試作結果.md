# ComplexPRHalfbandStage A-0 試作結果

## 1. 目的

本書は、`ComplexPRHalfbandStage` 正式版候補Aの最初の試作品として作成した
A-0 の確認結果を記録する。

A-0 は

- paraunitary
- FIR
- exact PR

を満たす最小構造である。

ただし、A-0 は最終候補ではなく、

- PR 確認
- power complementarity 確認
- 周波数特性評価の基準

として使う試作品である。

---

## 2. 実装内容

追加した実装:

```text
src/spflow/filterbank/halfband_stage.py
```

追加したクラス:

```text
ParaunitaryHalfbandStagePrototype
```

構造:

- even / odd sample pair を 2 点 DFT する
- analysis では `1 / sqrt(2)` で正規化する
- synthesis では逆変換で元の sample pair へ戻す

これは、2-channel paraunitary stage としては成立するが、
filter length が短いため選択度は低い。

---

## 3. 確認した項目

追加したテスト:

```text
tests/filterbank/test_halfband_stage.py
```

確認内容:

- 複素信号に対する stage 単体 PR
- power complementarity
- stopband attenuation が正式要求 `80 dB` に届かないこと

---

## 4. テスト結果

関連テスト:

```text
tests/filterbank/test_halfband_stage.py
tests/nonuniform/test_nonuniform_filterbank.py
tests/nonuniform/test_nonuniform_streaming.py
```

結果:

```text
12 passed in 0.15s
```

全体テスト:

```text
108 passed in 3.90s
```

---

## 5. 周波数特性メトリクス

`ParaunitaryHalfbandStagePrototype.response_metrics()` の結果は以下である。

```text
low_passband_ripple_db:       0.6876930815810858
high_passband_ripple_db:      0.6876930815810858
low_stopband_attenuation_db:  8.343206788338348
high_stopband_attenuation_db: 8.34320678833835
power_complementarity_error:  1.7763568394002505e-15
```

解釈:

- power complementarity は機械精度レベルで成立している
- stage 単体 PR も成立している
- しかし stopband attenuation は約 `8.34 dB` であり、正式要求の `80 dB` には届かない

---

## 6. 判断

A-0 は以下の点で有用である。

- paraunitary stage の最小基準として使える
- PR / streaming 骨格の切り分けに使える
- response metric の評価系として使える

一方で、以下の理由により正式版 stage としては採用しない。

- passband ripple が大きい
- stopband attenuation が不足している
- transition band が広すぎる

したがって、A-0 の位置付けは

- baseline paraunitary prototype

であり、

- practical formal stage

ではない。

---

## 7. 次の方針

次に進めるべき内容は、

- A-1: longer FIR paraunitary halfband stage

の試作である。

A-1 では、

- paraunitary / PR を維持する
- tap length を伸ばす
- stopband attenuation を改善する
- streaming / offline 一致を維持する

ことを確認する。

A-0 の結果から、候補Aの方向性そのものは継続するが、
正式採用には高選択度の paraunitary FIR stage が必要である。
