# ComplexPRHalfbandStage sinc系予備試験

## 1. 目的

本書は、
`ComplexPRHalfbandStage` の係数候補として

- sinc 関数ベースの lowpass
- windowed-sinc lowpass から作る素朴な QMF

が使えるかを予備的に確認した結果を記録する。

狙いは、

- Daubechies family を直接使わず
- より直感的な sinc 系設計へ寄せられるか

の切り分けである。

---

## 2. 試した内容

今回試したのは、以下の最も素朴な方式である。

1. `cutoff = fs/4` の halfband lowpass を windowed-sinc で作る
2. QMF 規約で highpass を導出する
3. synthesis 側は時間反転で構成する
4. `ComplexFIRHalfbandStage` に流し込み、
   - stopband attenuation
   - passband ripple
   - power complementarity
   - stage 単体 PR
   を評価する

この方式は

- sinc らしい周波数特性

を得やすい一方で、

- paraunitary / orthonormal 条件

を自動では満たさないことが懸念点である。

---

## 3. 再現用スクリプト

再現用として以下を追加した。

- `examples/filterbank/complex_halfband_stage_sinc_qmf_eval.py`

結果 artifact:

- `artifacts/complex_halfband_stage_design/sinc_qmf_experiment.json`

---

## 4. 実行条件

評価 sweep 条件:

- window: `hann`, `hamming`, `blackman`
- taps: `16`, `24`, `32`, `48`, `64`, `96`
- reference complex signal length: `4096`

---

## 5. 主な結果

上位の代表結果は以下である。

| candidate | stopband attenuation [dB] | ripple [dB] | power complementarity error | PR max abs | PR rms |
|---|---:|---:|---:|---:|---:|
| `sinc_blackman_taps96` | `99.878` | `1.57e-04` | `1.0000036` | `2.80e-01` | `1.22e-01` |
| `sinc_hann_taps96` | `90.671` | `4.53e-04` | `1.0000100` | `2.55e-01` | `1.11e-01` |
| `sinc_blackman_taps64` | `89.771` | `4.82e-04` | `1.0000119` | `3.59e-01` | `1.48e-01` |
| `sinc_blackman_taps48` | `82.995` | `1.02e-03` | `1.0000275` | `4.41e-01` | `1.71e-01` |
| `sinc_hann_taps64` | `80.114` | `1.45e-03` | `1.0000337` | `3.17e-01` | `1.35e-01` |

---

## 6. 読み取り

この結果から分かることは明快である。

1. sinc 系でも stopband attenuation 自体はかなり伸ばせる
2. passband ripple も十分小さくできる
3. しかし power complementarity error が約 `1.0` のままで、
   paraunitary 条件が全く満たせていない
4. その結果、stage 単体 PR は大きく崩れる

つまり、

- 周波数特性だけを見ると良い
- しかし `ComplexPRHalfbandStage` に必要な PR 構造には全く乗っていない

という結果である。

---

## 7. 結論

素朴な

- windowed-sinc lowpass
- QMF highpass

だけでは、
正式版 `ComplexPRHalfbandStage` の係数候補にはならない。

理由は、

- sinc 系の stopband 設計

と

- paraunitary / PR 条件

が別問題だからである。

したがって、
もし sinc 系へ寄せるなら、
少なくとも以下のどれかが必要になる。

1. sinc 形状を目標応答として使い、その上で paraunitary 制約付き最適化を行う
2. halfband power response を作ってから spectral factorization で orthonormal QMF に落とす
3. biorthogonal 設計として analysis / synthesis を別設計にする

---

## 8. 現時点での判断

現時点では、

- sinc をそのまま係数 family に採用するのは不適

と判断してよい。

ただし、

- sinc を目標応答や初期値として使う

方向は残る。

したがって今後の扱いとしては、

- `sinc 直接採用`: 非採用
- `sinc を目標応答として使う constrained design`: 検討価値あり

と整理するのが妥当である。
