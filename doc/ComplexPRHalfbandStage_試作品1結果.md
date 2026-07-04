# ComplexPRHalfbandStage 試作品1結果

## 1. 目的

本書は、`ComplexPRHalfbandStage` 正式版 v1 の第一候補である

- `complex FIR paraunitary halfband stage`

について、最小試作品を作って確認した結果を記録する。

ここでの目的は、

- 最終係数を決めること

ではなく、

- 候補A family が stage 単体として成立するか
- PR と streaming 契約を満たせるか
- どこまでが確認済みで、何が未達か

を明確にすることである。

---

## 2. 試作品1の内容

今回の試作品1は、

- `src/spflow/filterbank/complex_halfband_stage.py`

に追加した。

構成は以下である。

- 2-channel complex FIR stage
- explicit analysis / synthesis filters
- critically sampled
- streaming analyzer / synthesizer を同時に持つ

係数としては、最小の paraunitary 例である

- Haar / QMF 型

を採用した。

具体的には、

- analysis low: `[1, 1] / sqrt(2)`
- analysis high: `[-1, 1] / sqrt(2)`
- synthesis low: `[1, 1] / sqrt(2)`
- synthesis high: `[1, -1] / sqrt(2)`

である。

これは

- 実用的な最終係数

ではなく、

- paraunitary FIR family の最小試作品

として使った。

---

## 3. 実施した確認

追加したテストは

- `tests/filterbank/test_complex_halfband_stage.py`

である。

確認項目は以下とした。

1. stage 単体で複素信号を完全再構成できるか
2. streaming analysis が offline analysis と一致するか
3. streaming synthesis が offline synthesis と一致するか

---

## 4. 結果

試作品1のテスト結果:

```text
3 passed in 0.14s
```

全体テスト結果:

```text
111 passed in 3.99s
```

したがって、試作品1については少なくとも

- stage 単体の PR
- stage 単体の streaming / offline 一致

が成立した。

---

## 5. 周波数特性の確認

試作品1の low branch フィルタについて、
簡易な周波数応答確認も行った。

代表値は以下である。

- `passband_peak = 1.4142`
- `stopband_peak = 0.5412`
- `stopband_attenuation_db = 8.34 dB`

ここでの `stopband_attenuation_db` は、
簡易な周波数帯分割に基づく概算値である。

重要なのは絶対値そのものよりも、

- 80 dB 級の stopband を要求する正式版仕様には全く足りない

という点である。

したがって、この試作品1は

- 構造確認用としては合格
- 実用フィルタとしては不合格

と整理する。

---

## 6. この結果から分かったこと

今回の試作品1で分かったことは以下である。

### 6.1 確認できたこと

1. `complex FIR paraunitary` family で stage 単体 PR は実装可能
2. analysis / synthesis の explicit filter 形式でも streaming 一致を取れる
3. stage 単体の実装契約は破綻していない

### 6.2 まだ未達なこと

1. 実用的な stopband attenuation
2. 実用的な passband ripple
3. lower-edge 基準 packet への正式な周波数 shift 規約
4. delay / time_origin metadata の正式実装
5. analytic front-end を含む接続確認

---

## 7. 解釈

この試作品1は、

- 候補A family 自体が成立するか

を確認する目的には十分成功している。

一方で、

- 実用的な halfband filter

としては全く十分ではない。

したがって、今回の結果は

- 候補Aを正式版第一候補から外す理由にはならない
- ただし係数設計は別途本格化が必要

と解釈するのが妥当である。

---

## 8. 現時点での結論

試作品1の結果から、以下を結論とする。

1. `complex FIR paraunitary halfband stage` は構造候補として妥当である
2. stage 単体 PR と streaming 契約は満たせる
3. Haar/QMF 型はあくまで family 妥当性確認用の最小例である
4. 正式版へ進むには、次に
   - より長い FIR
   - 実用 stopband
   - 正式な packet 周波数規約
   を満たす stage 係数設計へ進む必要がある

したがって、次段階では

- 候補A family のまま
- 実用係数を持つ stage 単体試作 2

へ進むのが妥当である。
