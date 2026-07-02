# ComplexPRHalfbandStage 係数設計CLI設計と初期結果

## 1. 目的

本書は、`ComplexPRHalfbandStage` の係数を
均一帯域 PRDFT の prototype 設計 CLI と同様の流れで
設計・評価・保存する仕組みを正式に定め、
初回 sweep の結果を記録するための文書である。

---

## 2. 結論

`ComplexPRHalfbandStage` の係数は、
均一版 `examples/filterbank/prototype_design_eval.py` と同様に

- 候補生成
- 評価
- スコアリング
- artifact 保存

を行う専用 CLI で作成する方針でよい。

ただし最適化問題そのものは PRDFT prototype と異なるため、
CLI は別実装とする。

今回の初版実装では、

- `examples/filterbank/complex_halfband_stage_design_eval.py`

を追加し、
第一候補 family として

- Daubechies orthonormal QMF

の order sweep を実装した。

---

## 3. 実装内容

追加した主な実装は以下である。

- `src/spflow/filterbank/design/complex_halfband_stage.py`
- `examples/filterbank/complex_halfband_stage_design_eval.py`
- `tests/filterbank/test_complex_halfband_stage_design.py`

### 3.1 `design/complex_halfband_stage.py`

役割:

- order から Daubechies lowpass 係数を生成する
- QMF 規約から highpass を導出する
- `ComplexFIRHalfbandStage` に必要な
  - `analysis_phase`
  - `synthesis_phase`
  - `delay_compensation`
  を自動決定する
- `OrthonormalQMFStageCandidate` を返す

### 3.2 `complex_halfband_stage_design_eval.py`

役割:

- order sweep
- PR 評価
- streaming/offline 一致確認
- 周波数特性評価
- 候補ランキング
- 最良候補の artifact 保存

保存先:

- `artifacts/complex_halfband_stage_design/<artifact-name>/stage_filters.npz`
- `artifacts/complex_halfband_stage_design/<artifact-name>/stage_filters.json`

---

## 4. CLI の設計方針

今回の CLI は、均一版 prototype CLI と同様に

1. 候補を作る
2. 共通評価軸で比較する
3. 最良候補を artifact として保存する

流れを採用する。

ただし `ComplexPRHalfbandStage` では、
prototype pair 最適化の代わりに
stage 単体の係数列と stage metadata を決める。

評価項目は以下である。

1. stage 単体 PR 誤差
2. streaming analysis と offline analysis の一致
3. streaming synthesis と offline synthesis の一致
4. low/high の passband ripple
5. low/high の stopband attenuation
6. power complementarity

---

## 5. 候補生成の初版

初版では、まず数値的に安定な候補 family を揃えるため、

- Daubechies orthonormal QMF

を採用した。

理由:

1. 外部依存なしで `numpy.roots` により order から係数生成できる
2. 2-channel orthonormal FIR として PR 条件が明快
3. tap 長を伸ばしたときの stopband 向上傾向を素直に確認できる
4. uniform prototype CLI と同様に sweep と artifact 化がしやすい

したがって、初版 CLI は

- 汎用 optimizer

ではなく、

- 実用的な候補 family を系統的に sweep する設計器

として作成した。

---

## 6. 初回 sweep 条件

実行コマンド:

```text
python examples/filterbank/complex_halfband_stage_design_eval.py ^
  --order-list 20 22 24 26 28 30 ^
  --artifact-name db_sweep_formal_candidate
```

評価条件:

- reference complex signal length: `4096`
- streaming chunk size: `17`, `64`, `255`
- 周波数特性 FFT size: `65536`

選定方針:

1. まず `stopband >= 80 dB` かつ `ripple <= 0.1 dB` を満たす候補を優先
2. その中で PR 誤差が小さいものを優先
3. その上で係数長が短いものを優先

---

## 7. 初回 sweep 結果

主要結果は以下である。

| candidate | len | stopband attenuation [dB] | ripple [dB] | PR max abs | PR rms | exact_pr | freq target |
|---|---:|---:|---:|---:|---:|---:|---:|
| `daubechies_qmf_order20_taps40` | `40` | `70.827` | `3.59e-07` 相当 | `1.58e-12` | `5.72e-13` | `True` | `False` |
| `daubechies_qmf_order22_taps44` | `44` | `77.044` | `8.58e-08` 相当 | `6.64e-12` | `2.40e-12` | `False` | `False` |
| `daubechies_qmf_order24_taps48` | `48` | `83.245` | `2.05e-08` 相当 | `7.64e-11` | `2.78e-11` | `False` | `True` |
| `daubechies_qmf_order26_taps52` | `52` | `89.432` | `5.05e-09` 相当 | `6.05e-11` | `2.20e-11` | `False` | `True` |
| `daubechies_qmf_order28_taps56` | `56` | `95.606` | `3.65e-09` 相当 | `1.43e-09` | `5.13e-10` | `False` | `True` |
| `daubechies_qmf_order30_taps60` | `60` | `101.771` | `2.65e-08` 相当 | `1.68e-08` | `5.93e-09` | `False` | `True` |

ここで

- `exact_pr`
  は正式仕様の
  - `max_abs_error <= 1e-10`
  - `rms_error <= 1e-12`
  を同時に満たすか
- `freq target`
  は
  - `stopband >= 80 dB`
  - `ripple <= 0.1 dB`
  - streaming 一致
  を満たすか

を意味する。

---

## 8. 現時点の第一候補

初回 sweep の第一候補は

- `daubechies_qmf_order24_taps48`

とする。

採用理由:

1. `>= 80 dB` を初めて満たす最短候補である
2. streaming analysis / synthesis は機械精度レベルで offline と一致した
3. `daubechies_qmf_order26_taps52` 以降は stopband は改善するが、tap 長と遅延も増える
4. 現段階では、まず practical frequency selectivity に到達した最短候補を保持する方がよい

今回保存された artifact:

- `artifacts/complex_halfband_stage_design/db_sweep_formal_candidate/stage_filters.npz`
- `artifacts/complex_halfband_stage_design/db_sweep_formal_candidate/stage_filters.json`

---

## 9. 重要な注意

`daubechies_qmf_order24_taps48` は、

- 周波数特性目標
- streaming 一致

は満たしたが、
正式仕様の strict PR 条件

- `rms_error <= 1e-12`

はまだ満たしていない。

したがって、現時点の整理は以下である。

- 構造としては有望
- 周波数分離性能の候補としては採用価値がある
- ただし正式版係数として固定する前に
  PR 数値精度の改善が必要

---

## 10. Pending

### P-CLI-1

高次 Daubechies 係数生成は `numpy.roots` に依存しているため、
order を上げるほど PR 数値誤差が悪化する傾向がある。

したがって今後は、

- root 計算後の係数 refinement
- 係数正規化の改善
- paraunitary 制約を直接使う別 optimizer

のいずれかを検討する必要がある。

### P-CLI-2

今回の CLI は初版として

- Daubechies QMF family sweep

に限定している。

今後は必要に応じて、

- より高選択度の paraunitary family
- biorthogonal family
- 直接最適化型の candidate generator

を追加候補とする。

---

## 11. 現時点での判断

現時点では、
`ComplexPRHalfbandStage` の係数は
均一版 prototype CLI と同様の流れで
専用 CLI により作成してよい。

また、その初版として

- Daubechies QMF order sweep

を正式に採用してよい。

ただし、これで最終係数設計が完了したわけではなく、
現時点では

- `daubechies_qmf_order24_taps48` を実用周波数特性の第一候補
- strict formal PR は Pending

という位置付けで扱うのが妥当である。
