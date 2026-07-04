# Nonuniform FilterBank streaming 検証結果

## 1. 目的

本書は、不均一フィルタバンク初期実装について

- subband 内 beamforming を入れない
- 複素 PR tree filter bank 単体で
- streaming 動作が offline 動作と一致する

ことを確認した結果を記録する。

本検証は、`Nonuniform_FilterBank_設計方針.md` の
段階検証のうち、まず

- 完全再構成
- streaming / offline 一致

を確認する段階に相当する。

位置付け:

- 本書は beamforming 接続前の初期基準結果を残す履歴文書である
- 本書で「未確認」としていた項目の多くは、その後
  - `doc/Nonuniform_FilterBank_formal_tree接続結果.md`
  - `doc/Nonuniform_FilterBank_Daubechies_beamforming試作結果.md`
  - `doc/Nonuniform_FilterBank_Daubechies_streaming_sweep結果.md`
  で確認済みである
- したがって、本書の未確認項目は当時の段階差分として読むのが正しい

---

## 2. 今回検証した実装

今回の対象は、初期基準実装として追加した

- `src/spflow/filterbank/nonuniform_tree.py`
- `src/spflow/filterbank/nonuniform_streaming.py`

である。

構造は以下とした。

- 2 分岐の複素 PR stage
- stage 自体は 2 点 DFT による block exact PR
- それを非対称木へ接続した nonuniform tree
- root block size は最深 leaf 深さに合わせて `128 sample`

この実装は、最終的な高選択度フィルタではない。
まずは

- 木構造として PR が成立すること
- block streaming で破綻しないこと

を確認するための基準系である。

---

## 3. streaming 実装の範囲

今回 streaming で確認したのは、

- analytic 複素入力を nonuniform tree に逐次投入する経路
- leaf band packet を block 単位で合成木へ戻す経路

である。

実入力については、当時の時点では

- offline の FFT-based analytic helper

を使っているため、

- real input をそのまま causal に analytic 化する front-end

はまだ今回の検証範囲に含めていない。

したがって、当時の時点で確認できたことは厳密には

- complex PR tree 本体の streaming / offline 一致

である。

---

## 4. 実施したテスト

以下のテストを追加して確認した。

### 4.1 既存 PR 確認

`tests/nonuniform/test_nonuniform_filterbank.py`

- 2-channel 複素 PR stage 単体の再構成
- 指定 8 帯域の band plan と tree depth の一致
- offline 解析合成での実信号再構成
- offline 解析合成での多チャネル実信号再構成
- leaf packet 長が tree depth に応じて落ちること
- analytic 複素信号の再構成

### 4.2 streaming 確認

`tests/nonuniform/test_nonuniform_streaming.py`

- irregular chunk 分割した analytic 多チャネル入力で、streaming analysis 結果が offline analysis と一致
- irregular chunk 分割した analytic 多チャネル入力で、streaming synthesis 結果が offline synthesis と一致
- real 信号を offline analytic 化した後、その complex tree 部分では streaming 合成後に元の実信号へ戻る

---

## 5. 結果

新規テストの結果:

```text
9 passed in 0.26s
```

全体テストの結果:

```text
105 passed in 4.98s
```

したがって、少なくとも今回の初期基準実装については

- nonuniform tree 単体で完全再構成できる
- analytic 複素入力に対して streaming と offline が一致する
- streaming 化によって既存系を壊していない

と判断してよい。

---

## 6. 解釈

今回確認できた最も重要な点は、

- 不均一木そのものは block streaming で成立する

ことである。

これは今後、

- stage を高選択度フィルタへ置き換える
- causal analytic front-end を導入する
- leaf band ごとに beamforming を接続する

といった拡張へ進む前提として十分重要である。

一方で、当時の時点では以下が未確認だった。

- real input を causal に analytic 化した場合の streaming 一致
  - これは後に `doc/Nonuniform_FilterBank_formal_tree接続結果.md` で確認済み
- block exact PR stage を高選択度複素 halfband stage に置き換えた場合の PR 維持
  - これは後に `FormalComplexPRHalfbandStage` を用いた formal tree 接続で確認済み
- 不均一 leaf band ごとの時間原点・遅延整合
  - これは後に formal metadata 付き tree / leaf 実装で確認済み
- leaf ごとの beamforming 接続後の全体再構成
  - これは後に Daubechies beamforming / streaming 文書群で確認済み

したがって、本結果は

- 最終方式が完成したこと

を意味するものではなく、

- 不均一複素木の streaming 基本骨格が成立した

ことを示す段階結果である。

---

## 7. 現時点での結論

subband 内 beamforming をまだ入れない段階において、
不均一複素 PR tree filter bank は

- offline で完全再構成できる
- streaming 実装でも offline と一致する

ことを確認した。

よって、当時の次段階としては

1. causal analytic front-end の扱いを整理する
2. 2 点 DFT 基準 stage を、より実用的な複素 halfband stage へ置き換える
3. その上で leaf band ごとの beamforming 接続へ進む

という順序で進めるのが妥当である。
