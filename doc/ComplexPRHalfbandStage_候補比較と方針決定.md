# ComplexPRHalfbandStage 候補比較と方針決定

## 1. 目的

本書は、`ComplexPRHalfbandStage` の正式版実装候補を比較し、

- 採用候補
- 保留候補
- 非採用候補

を決定するとともに、
現時点の試作品確認結果を記録するための文書である。

本書の位置付けは、

- `ComplexPRHalfbandStage_正式仕様.md` で固定した仕様に対して
- 実際にどの方式で実装を進めるか

を決める意思決定記録である。

---

## 2. 今回の結論

実装の進め方は以下で決定する。

1. まず机上で候補を絞る
2. 次に、最有力候補だけを最小試作する
3. その試作で
   - PR
   - streaming / offline 一致
   - 周波数特性
   - delay / time_origin の扱いやすさ
   を確認する
4. 良ければその方式を正式版 `ComplexPRHalfbandStage` として採用する

つまり、

- いきなり複数実装を大きく作らない
- 机上で絞ってから試作で決める

という進め方を正式に採用する。

---

## 3. 比較した候補

## 3.1 候補A: complex FIR paraunitary halfband stage

内容:

- complex FIR
- critically sampled
- paraunitary
- unit-gain PR
- lower-edge 基準 packet 規約

利点:

- PR 条件が明快
- エネルギー挙動を把握しやすい
- streaming 実装と相性がよい
- delay を明示しやすい
- C++ 実装へ移しやすい
- 木全体へ再帰接続しやすい

欠点:

- stopband 性能を上げると tap 長が増えやすい
- complex 係数設計の自由度が必要
- 係数設計器の作成がやや重い

評価:

- 正式版 v1 の第一候補として採用

## 3.2 候補B: complex FIR biorthogonal halfband stage

内容:

- complex FIR
- critically sampled
- biorthogonal PR

利点:

- paraunitary より設計自由度が大きい可能性がある
- 周波数特性と遅延のトレードオフで有利な場合がある

欠点:

- エネルギー規約が分かりにくくなる
- 数値安定性や正規化規約が複雑になりやすい
- 初期版としては切り分けが難しい

評価:

- 予備候補として保持
- 正式版 v1 では第一候補にしない

## 3.3 候補C: block exact PR stage

内容:

- 2 点 DFT ベース
- exact PR
- block streaming のみで骨格確認

利点:

- PR を最も簡単に確認できる
- 木構造や packet 契約の切り分けに向く
- streaming 骨格の試験台として優秀

欠点:

- 実用的な帯域分離性能を持たない
- halfband filter の本命候補ではない

評価:

- 基準実装として維持
- 正式版 stage そのものとしては採用しない

## 3.4 候補D: IIR / all-pass / 楕円系の複素分岐 stage

内容:

- IIR または all-pass を含む分岐構造

利点:

- tap 数を減らせる可能性がある
- 鋭い遷移帯域を得やすい可能性がある

欠点:

- delay / phase / streaming の扱いが難しい
- 実装の検証コストが高い
- 初期段階での切り分けに不向き

評価:

- 正式版 v1 では採用しない

---

## 4. 採用理由

正式版 v1 で候補A

- `complex FIR paraunitary halfband stage`

を第一候補にする理由は以下である。

1. `ComplexPRHalfbandStage_正式仕様.md` で固定した
   - unit-gain PR
   - lower-edge 基準 packet
   - streaming / offline 一致
   - delay / time_origin 明示
   に最も素直に乗る

2. 不均一木全体へ再帰的に接続しやすい

3. 実用化時に
   - 数値安定性
   - 可搬性
   - C++ 実装容易性
   の点で見通しがよい

4. 今の段階では
   - 最高性能よりも
   - 構造の堅さと検証容易性
   を優先すべき

したがって、

- v1 は paraunitary FIR
- biorthogonal は保留
- block exact PR stage は基準系

という整理が妥当である。

---

## 5. 試作品の扱い

正式版候補をいきなり複数作るのではなく、
まずは以下の 2 層で試作を扱う。

### 5.1 試作品0: 骨格確認用基準実装

内容:

- `src/spflow/filterbank/nonuniform_tree.py`
- `src/spflow/filterbank/nonuniform_streaming.py`

目的:

- 木構造そのものが PR で成立するか
- packet 契約が機能するか
- streaming 骨格が壊れないか

これは正式版候補ではないが、
本命方式へ進む前の切り分け基準として有効である。

### 5.2 試作品1: 正式候補Aの最小試作

今後作るべき試作品は、

- complex FIR paraunitary halfband stage 単体

とする。

まずこの stage 単体だけを作り、

- PR
- 周波数特性
- streaming 一致
- delay / time_origin 管理

を確認する。

---

## 6. 試作品0の確認結果

既存試作品0については、以下を確認済みである。

対象:

- `tests/nonuniform/test_nonuniform_filterbank.py`
- `tests/nonuniform/test_nonuniform_streaming.py`

再確認結果:

```text
9 passed in 0.16s
```

この確認で成立していることは以下である。

- block exact PR stage を使った nonuniform tree が完全再構成できる
- analytic 複素入力に対して streaming analysis が offline analysis と一致する
- analytic 複素入力に対して streaming synthesis が offline synthesis と一致する
- real 信号を offline analytic 化した後の complex tree 部分では streaming 再構成が成立する

これにより、

- 不均一木の骨格
- packet 化
- block streaming

は、少なくとも基準実装レベルでは破綻していないことを確認できた。

---

## 7. 試作品0から分かったこと

試作品0で分かった重要点は以下である。

1. 木構造そのものは成立する
2. 非同期 leaf band を packet で持つ方針は成立する
3. streaming 骨格は成立する
4. 構造の大枠は今後の本命設計へ引き継げる

逆に、まだ未確認なのは以下である。

1. 正式候補Aの complex FIR paraunitary stage が、実用的な stopband 性能で成立するか
2. lower-edge 基準 packet 規約で、upper child の shift を入れても実装が素直か
3. delay / time_origin 規約が本命 stage でも扱いやすいか
4. causal analytic front-end を含めたときにも streaming 一致が保てるか

したがって、試作品0は

- 正式版完成

を意味しないが、

- 正式候補Aへ進んでよいだけの構造的基盤はある

と判断してよい。

---

## 8. 今後の試作順序

今後は以下の順で進める。

1. 候補Aの stage 単体試作
2. stage 単体の PR / streaming / tone sweep
3. 2-level tree へ拡張
4. full nonuniform tree へ拡張
5. causal analytic front-end を接続
6. leaf band ごとの beamforming 接続

つまり、

- まず stage
- 次に小木
- 次に full tree

の順で進める。

---

## 9. 現時点での決定

現時点で以下を正式決定とする。

1. `ComplexPRHalfbandStage` 正式版 v1 の第一候補は
   `complex FIR paraunitary halfband stage`
   とする。

2. `complex FIR biorthogonal stage` は保留候補とする。

3. `block exact PR stage` は正式版候補ではなく、
   骨格確認用の基準実装として残す。

4. 実装の進め方は
   - 机上で候補を絞る
   - 最有力候補だけを最小試作する
   - 試作結果で正式採用を確定する
   方式とする。

5. 既存試作品0の結果から、
   不均一木の骨格と streaming 骨格は成立済みとみなしてよい。

したがって、次に進めるべき内容は

- 候補Aの stage 単体試作

である。
