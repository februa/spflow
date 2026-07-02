# ComplexPRHalfbandStage A-1 試作結果

## 1. 目的

本書は、A-0 に続く次段階の試作として、

- より長い FIR
- paraunitary family
- stage 単体 PR と streaming 契約の維持

を確認した A-1 の結果を記録する。

ここでの A-1 は、

- 実用 stopband へ届くかどうかの見通し
- 候補A family を継続する価値があるか

を確認するための試作である。

---

## 2. 試作内容

今回追加した実装:

- `src/spflow/filterbank/halfband_stage_candidates.py`
- `tests/filterbank/test_halfband_stage_candidates.py`

評価対象:

- `haar_qmf_taps2`
- `daubechies_qmf_order2_taps4`
- `daubechies_qmf_order3_taps6`
- `daubechies_qmf_order4_taps8`

これらは、既知の orthonormal QMF lowpass 係数を用いた

- FIR
- paraunitary family
- critically sampled 2-channel stage

の候補群である。

各候補は、QMF 規約から

- analysis high
- synthesis low
- synthesis high

を導出し、
位相と遅延補償は数値的に整合する組を固定した。

---

## 3. 確認した項目

確認内容は以下である。

1. 各候補が stage 単体で PR を満たすか
2. tap length を伸ばすと stopband attenuation が改善するか
3. 最も長い `daubechies_qmf_order4_taps8` でも正式要求 `80 dB` に届くか

関連テスト:

- `tests/filterbank/test_halfband_stage.py`
- `tests/filterbank/test_halfband_stage_candidates.py`
- `tests/filterbank/test_complex_halfband_stage.py`

---

## 4. テスト結果

関連テスト結果:

```text
9 passed in 0.23s
```

全体テスト結果:

```text
114 passed in 3.95s
```

したがって、A-1 候補群については

- stage 単体の PR
- 候補比較用メトリクス評価

が問題なく動作している。

---

## 5. 周波数特性結果

代表結果は以下である。

| candidate | low stopband attenuation [dB] | high stopband attenuation [dB] | low ripple [dB] | high ripple [dB] |
|---|---:|---:|---:|---:|
| `haar_qmf_taps2` | `8.34` | `8.34` | `0.688` | `0.688` |
| `daubechies_qmf_order2_taps4` | `12.36` | `12.36` | `0.260` | `0.260` |
| `daubechies_qmf_order3_taps6` | `16.04` | `16.04` | `0.110` | `0.110` |
| `daubechies_qmf_order4_taps8` | `19.55` | `19.55` | `0.0485` | `0.0485` |

また、power complementarity は
各候補で機械精度レベルまたはそれに近い精度で成立した。

---

## 6. 判断

今回の A-1 で確認できたことは以下である。

1. paraunitary FIR family のまま tap length を伸ばすと、
   stopband attenuation は単調に改善する
2. passband ripple も改善方向にある
3. stage 単体の PR は維持できる

一方で、最良だった `daubechies_qmf_order4_taps8` でも

- stopband attenuation は約 `19.55 dB`

であり、
正式要求である

- `80 dB`

には大きく届かない。

したがって、A-1 の位置付けは

- A-0 より前進した
- しかしまだ practical formal stage ではない

である。

---

## 7. Pending

今回の結果から、以下を Pending とする。

### P-A1-1

候補A family のまま、

- `>= 80 dB` 級 stopband attenuation
- stage 単体 PR
- streaming / offline 一致

を同時に満たす

- longer FIR paraunitary halfband stage

の係数設計法を別途検討する必要がある。

### P-A1-2

lower-edge 基準 packet 規約を満たす

- upper child の周波数 shift
- delay / time_origin metadata

は、今回の A-1 ではまだ正式実装していない。

---

## 8. 次の方針

A-1 の結果により、

- 候補A family 自体は継続してよい
- ただし係数設計は別段階として本格化が必要

と判断する。

したがって次は、

1. A-2: より高選択度の paraunitary FIR stage 設計法を詰める
2. 並行して、`causal analytic front-end` の正式仕様を設計する

の 2 本立てで進めるのが妥当である。
