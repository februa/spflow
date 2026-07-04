# ComplexPRHalfbandStage sinc目標 constrained最適化結果

## 1. 目的

本書は、
`sinc を目標応答として使う constrained design`
の最初の試作品として実装した

- halfband power response 上の制約付き最適化
- spectral factorization による QMF 化

の結果を記録する。

---

## 2. 今回の実装

追加した主な実装:

- `src/spflow/filterbank/design/sinc_constrained_optimizer.py`
- `examples/filterbank/complex_halfband_stage_sinc_constrained_optimize.py`
- `tests/filterbank/test_sinc_constrained_optimizer.py`

今回の optimizer は、

1. odd-lag halfband power coefficient を変数に取る
2. `P(w) = 1 + 2 Σ a_m cos((2m+1)w)` 形で halfband complement を保つ
3. `P(w) >= floor` を満たす範囲で座標降下最適化する
4. 最後に spectral factorization で FIR lowpass を得る

方式である。

したがってこれは、

- sinc target に近づける
- PR / paraunitary 条件をできるだけ壊さない

ための最初の constrained optimizer と位置付けられる。

---

## 3. 実行条件

実行コマンド:

```text
python examples/filterbank/complex_halfband_stage_sinc_constrained_optimize.py ^
  --tap-list 16 24 32 48 ^
  --artifact-name sinc_constrained_optimizer_results
```

条件:

- target window: `blackman`
- cutoff: `0.25`
- optimizer FFT size: `8192`
- optimizer passes: `60`

artifact:

- `artifacts/complex_halfband_stage_design/sinc_constrained_optimizer_results.json`

---

## 4. 結果

主要結果は以下である。

| candidate | taps | weighted power rms | sinc fullband rms | stopband attenuation [dB] | PR max abs | PR rms | power complementarity error |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sinc_target_constrained_blackman_taps24` | `24` | `1.61e-03` | `1.05e-01` | `27.34` | `4.72e-03` | `1.57e-03` | `2.22e-03` |
| `sinc_target_constrained_blackman_taps16` | `16` | `1.63e-03` | `1.28e-01` | `27.00` | `5.38e-03` | `1.79e-03` | `2.53e-03` |
| `sinc_target_constrained_blackman_taps32` | `32` | `1.76e-03` | `9.18e-02` | `26.51` | `4.44e-03` | `1.48e-03` | `2.09e-03` |
| `sinc_target_constrained_blackman_taps48` | `48` | `2.00e-03` | `7.73e-02` | `25.83` | `4.23e-03` | `1.41e-03` | `2.00e-03` |

---

## 5. 読み取り

この試作で確認できたことは以下である。

1. naive sinc/QMF 直結よりは大幅に良い
2. stage 単体 PR は `1e-3` 台まで改善できる
3. power complementarity も `1e-3` 台まで改善できる
4. sinc target への一致度も、既存 Daubechies candidate より良い場合がある

一方で、まだ大きく足りない点も明確である。

1. stopband attenuation は `25 - 27 dB` 程度に留まる
2. 正式要求の `>= 80 dB` には全く届いていない
3. strict PR 要求 `1e-10 / 1e-12` にも未達

つまり、

- constrained optimizer の骨格としては成立した
- しかし実用 stage の最終設計器にはまだ遠い

という結果である。

---

## 6. 位置付け

今回の optimizer は、

- sinc target を formal objective に入れる
- paraunitary を意識した制約付き設計を実装する

という意味で前進である。

ただしこれは

- final optimizer

ではなく、

- A-2 初期試作

とみなすべきである。

---

## 7. 今後の改善方向

次に改善すべき点は以下である。

1. 座標降下だけでなく、より強い探索法を入れる
2. odd-lag power coefficient のみでなく、より表現力の高い paraunitary parameterization を使う
3. stopband 領域の重み付けを強めた多目的最適化にする
4. spectral factorization の数値精度を上げる

特に本命は、

- paraunitary parameterization を直接持つ optimizer

へ進むことである。

---

## 8. 現時点での判断

現時点では、

- sinc target constrained optimizer を導入して進める方針は妥当

と判断してよい。

ただし今回の試作品だけでは、

- practical stopband
- strict formal PR

の両方に届かない。

したがって今後の扱いは、

- 方針としては採用
- 実装としては試作1完了
- 実用化は Pending

と整理するのが妥当である。
