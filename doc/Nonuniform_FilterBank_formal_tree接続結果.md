# Nonuniform FilterBank formal tree接続結果

## 1. 目的

本書は、不均一複素フィルタバンク系について、

- `FormalComplexPRHalfbandStage`
- `FormalBandPacket`
- `CausalAnalyticFrontend`

を使った formal tree 接続の到達点を記録する文書である。

今回の目的は、

- high-stopband 係数の最適化

ではなく、

- formal packet / metadata 契約を full tree へ接続すること
- real input から real output までの最小 end-to-end を成立させること

にある。

---

## 2. 今回追加した実装

対応実装:

- `src/spflow/filterbank/formal_nonuniform_tree.py`
- `src/spflow/filterbank/formal_nonuniform_streaming.py`

追加した内容は以下である。

1. `FormalComplexPRHalfbandStage` を不均一 full tree へ再帰接続
2. leaf 出力を `FormalBandPacket` として保持
3. 各内部ノードの target length を保持し、合成時にその長さへ戻す
4. `CausalAnalyticFrontend` を前段に置いた `analyze_real()` を追加
5. `synthesize()` で
   - analytic 出力
   - real 出力
   の両方を返せるようにした
6. exact-by-construction の formal metadata 付き streaming analyzer を追加した
7. oracle 再構成器と増分 root 合成器を併設した formal metadata 付き streaming synthesizer を追加した

ここでの formal tree は、
現在の contiguous-band packet 規約を保ったまま、
各 packet の複素サンプル列については

- upper child を lower-edge 基準へ周波数シフトして保持する
- `time_origin_at_root_rate` を root-rate 整数で更新する

実装まで含めて接続した。

さらにその後、leaf beamforming 出力での practical rule も固定した。

- output packet の `time_origin_at_root_rate` は valid region 先頭の root-rate 時刻
- 同一 leaf 内では `packet_length * 2**tree_depth` ずつ進む
- `delay_samples_at_root_rate` は v1 では保持する

したがって、formal tree 単体としては `time_origin_at_root_rate` の運用整理まで完了した。

---

## 3. 実装上の要点

## 3.1 target length の保持

Daubechies 系 FIR stage を full tree へ再帰接続すると、
各 stage の full convolution により
単純な leaf packet 合成だけでは親ノード長が揃わない。

そのため今回の formal tree では、
解析時に

- 各ノード band id
- そのノード packet の sample length

を保持し、合成時には

- `synthesize_packets(..., length=node_sample_lengths[node_id])`

を明示する構造にした。

これにより、

- full tree 全体で sibling 長が崩れない
- leaf metadata を保持したまま root へ戻せる

ことを確認した。

## 3.2 front-end との接続

`analyze_real()` では、

1. `CausalAnalyticFrontend.analyze(..., pad_tail=True)` で causal analytic 化
2. front-end の delay metadata を root packet に引き継ぐ
3. formal tree で解析
4. 合成後、`recover_real(..., length=original_length)` で元実信号へ戻す

方式を採用した。

したがって今回の end-to-end 検証は、

- front-end の遅延
- tree の PR
- real output の整合

を同時に確認する最小試験になっている。

## 3.3 formal metadata 付き streaming

`formal_nonuniform_streaming.py` では、

- `ComplexFIRHalfbandStageStreamingAnalyzer` を各内部ノードへ再帰配置する
- leaf では `FormalBandPacket` chunk を emit する
- high child では chunk ごとの `time_origin_at_root_rate` を使って lower-edge shift を掛ける
- 合成側は oracle として leaf chunk 全履歴を再構成する参照経路を残す
- 正式経路は、各内部ノードに streaming synthesizer を持つ増分 root 合成木へ置き換える
- sibling の共通 prefix だけを消費し、stable prefix を親へ順次押し上げる
- high child の lower-edge 基準解除は各内部ノードで packet metadata に従って戻す

構造にした。

これにより、現在の formal streaming は

- offline formal tree と完全一致すること
- metadata continuity を崩さないこと
- root を毎回 prefix 再構成しないこと
- stage 内部も stateful streaming engine で処理すること

を同時に満たす。

現在の `ComplexFIRHalfbandStageStreamingAnalyzer / Synthesizer` は、
public 契約を保ったまま stateful 実装へ差し替えてある。
一方、旧 prefix 再実行版は

- `OracleComplexFIRHalfbandStageStreamingAnalyzer`
- `OracleComplexFIRHalfbandStageStreamingSynthesizer`

として回帰試験用に残している。
formal tree 側でも oracle 再構成器は参照系として残している。

---

## 4. 追加試験

対応試験:

- `tests/nonuniform/test_formal_nonuniform_tree.py`
- `tests/nonuniform/test_formal_nonuniform_streaming.py`

確認項目:

1. formal tree の default band plan が設計どおりか
2. analytic 入力で full tree PR が成立するか
3. leaf packet metadata が伝播するか
4. real input -> front-end -> formal tree -> real output が成立するか
5. formal metadata 付き streaming analysis が offline と一致するか
6. 増分 root 合成の formal metadata 付き streaming synthesis が offline と一致するか
7. 増分 root 合成が oracle 再構成器と一致するか
8. front-end streamer を含めても real 系 streaming が offline と一致し、oracle とも一致するか

再確認結果:

```text
python -m pytest -q tests/nonuniform/test_formal_nonuniform_tree.py tests/nonuniform/test_formal_nonuniform_streaming.py tests/nonuniform/test_formal_complex_pr_stage.py tests/filterbank/test_causal_analytic_frontend.py
12 passed in 0.38s
```

全体回帰:

```text
python -m pytest -q
153 passed in 21.13s
```

---

## 5. 結果

## 5.1 analytic full tree PR

analytic 複素入力に対して、
formal tree の合成出力は元入力へ
機械精度レベルで一致した。

代表結果:

- `atol = 1e-10` で一致

## 5.2 metadata 伝播

leaf packet では少なくとも以下を確認した。

- band id は設計どおり
- sample rate は band spec と一致
- 同一 depth の sibling では delay が一致
- 同一 depth の sibling では `time_origin_at_root_rate` も一致する
- depth の深い低域 leaf は高域 leaf より大きい delay と time origin を持つ
- root stage の upper child は lower-edge 基準へシフトされている

これは、

- formal packet に delay / time origin metadata を載せた tree
- upper child を lower-edge 基準へ落とす規約

として最低限の整合が取れていることを意味する。

## 5.3 real end-to-end

`num_taps = 31` の front-end を使った最小 end-to-end では、

- analytic 再構成誤差: 約 `1.27e-12`
- real 再構成誤差: 約 `1.19e-12`

であり、

- real input
- causal analytic front-end
- formal nonuniform tree
- real output

の最小系が成立した。

## 5.4 formal streaming 一致

追加した streaming 試験では、

- analytic 入力の analysis 結果が offline formal tree と一致
- analytic 入力の synthesis 出力が offline formal tree と一致
- 増分 root 合成が oracle 再構成器と一致
- front-end streamer を前段に置いた real 系でも analytic 出力 / real 出力の両方が offline と一致し、oracle とも一致

を確認した。

したがって、formal metadata 付き tree は
real 系を含めて streaming 検証まで到達したとみなしてよい。

## 5.5 stage-level streaming 高速化反映

2026-07-02 時点で、stage-level streaming engine は stateful 実装へ差し替え済みである。

確認済み項目:

- `tests/filterbank/test_complex_halfband_stage.py` で Haar と `daubechies_qmf_order4_taps8` の両方について streaming/oracle 一致
- `tests/nonuniform/test_formal_nonuniform_streaming.py` で formal tree streaming が offline / oracle と一致
- 全体回帰 `python -m pytest -q` で `153 passed in 21.13s`

したがって、formal tree は

- root 構造
- stage 内部 engine
- oracle 回帰経路

の 3 点が揃った状態になった。

---

## 6. 今回の意味

今回確認できたことは、

- formal packet / metadata 契約
- formal stage
- front-end

を最適化なしでも先に接続できる、という点である。

したがって現時点では、

- 係数最適化を止めているから formal 実装が進まない

のではなく、

- formal 構造の受け皿はすでに前へ進められる

と判断してよい。

---

## 7. 残る課題

今回の接続結果に関して、現在も残っている課題は以下である。

1. high-stopband stage 係数の正式最適化
2. multiband interferer 条件での MVDR 実用評価
3. real-input beamforming streaming の正式評価整理

補足:

- single-band interferer を含む representative な MVDR streaming 一致は、その後 `doc/Nonuniform_FilterBank_Daubechies_streaming_sweep結果.md` で確認済み
- `leaf_independent_one_sided` を含む root-rate 再合成後の peak angle / boundary continuity も同文書で確認済み
- したがって、本書の接続成果そのものについては
  - formal tree 接続完了
  - end-to-end 最小接続完了
  - formal metadata 付き streaming 完了
  - 増分 root 合成への差し替え完了
  - `time_origin_at_root_rate` の leaf practical rule 固定
  - formal metadata 付き tree への CBF beamforming 統合完了
- まで到達済みと見てよい
