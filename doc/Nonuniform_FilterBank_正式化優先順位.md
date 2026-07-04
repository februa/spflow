# Nonuniform FilterBank 正式化優先順位

## 1. 目的

本書は、不均一複素フィルタバンク系について、

- 係数最適化
- stopband 最適化
- 処理量最適化

に依存せず、
先に正式版へ進められる部品を優先順で固定するための文書である。

ここでいう「正式化」とは、

- 入出力契約が固定できる
- delay / time origin / packet 規約が固定できる
- offline / streaming 一致を検証で閉じられる
- 後で係数や高速化を差し替えても構造が崩れない

状態を指す。

---

## 2. 基本方針

当面は、次の 2 種類を明確に分ける。

### 2.1 先に正式化するもの

- packet 契約
- metadata 契約
- tree 構造
- front-end 構造
- streaming 契約
- 検証器

これらは、最適化の良し悪しとは独立に前へ進められる。

### 2.2 後で最適化するもの

- high-stopband stage 係数
- leaf ごとの FFT 長の最適化
- MVDR 更新周期の最適化
- 全体処理量の最適化
- C++ 実装前提の高速化

これらは、正式構造と契約が固まった後に詰める。

---

## 3. 優先順位

## 3.1 最優先

### A-1. formal packet / metadata 契約

対象:

- `FormalBandPacket`
- `time_origin_at_root_rate`
- `delay_samples_at_root_rate`
- lower-edge 周波数規約

理由:

- これは構造の根幹であり、最適化では解決しない
- 後段の beamforming / streaming / front-end すべてが依存する
- ここが曖昧なままだと、以後の結果が正式版として残らない

完了条件:

- offline tree で metadata 伝播規約が固定される
- sibling 合成時の delay / length 規約が固定される
- 設計書と実装が一致する

### A-2. formal tree 接続

対象:

- `FormalComplexPRHalfbandStage` を full nonuniform tree へ接続すること
- current Daubechies candidate を暫定係数として使ってもよい

理由:

- 係数の最適性がなくても、formal tree の構造は検証できる
- これができると、以後の front-end / beamforming / metadata 検証の受け皿ができる

完了条件:

- full tree の offline PR が成立する
- full tree の streaming 契約が明文化される

### A-3. causal analytic front-end の正式化

対象:

- `CausalAnalyticFrontend`
- `CausalAnalyticFrontendStreamer`

理由:

- すでに最小実装と streaming / offline 一致確認がある
- 残っているのは最適化ではなく、正式評価条件と tree 接続である

完了条件:

- negative-frequency suppression の正式評価指標を固定
- tree 接続時の delay 合成規約を固定
- real input -> analytic -> tree -> real output の検証を追加

### A-4. formal checker 群

対象:

- front-end 単体 checker
- formal tree checker
- metadata checker
- streaming continuity checker

理由:

- 正式版で重要なのは「何を満たしたら完了か」を閉じること
- checker は最適化に依存せず先に作れる

完了条件:

- 設計書に書いた要求が自動試験に対応づく

---

## 3.2 次優先

### B-1. beamforming の formal metadata 統合

対象:

- leaf packet の時刻情報を保持したまま beamforming を通すこと
- formal metadata 付き増分 streaming synthesizer へ統合すること

理由:

- broadside CBF で構造安定性は確認済み
- 残りは「正式版として残せる metadata 付き構造」にすること

### B-2. leaf 運用条件の固定

対象:

- `used_channels`
- `frame size / valid size`
- `short_fft_size / short_fft_hop_size`

理由:

- これは最適化そのものではなく、正式版 v1 の運用条件固定である
- ただし最優先群よりは後でよい

---

## 3.3 後回し

### C-1. stage 係数最適化

対象:

- `>= 80 dB` stopband を満たす paraunitary FIR 係数設計
- sinc target / constrained optimizer

理由:

- 重要だが、構造と契約が固まってからでも遅くない
- beamforming を通した結果で要求が動く可能性がある

### C-2. MVDR 正式安定性評価

対象:

- covariance 更新
- short FFT 起点の重み更新
- 周波数依存 steering

理由:

- CBF で構造安定性の先行確認はできる
- MVDR は構造よりも運用条件と安定化の問題が大きい

### C-3. 処理量最適化

対象:

- stage-level polyphase streaming の本実装（2026-07-02 完了）
- その後の増分 streaming 実装の処理量最適化
- C++ 前提の最適化
- 上三角共分散やチャネル制限による実装最適化

理由:

- 処理時間要件が強いため、stage-level の `O(N)` 化は後回しにせず先行してよい
- ただし packet / metadata 契約を変える最適化は避ける

---

## 4. 今すぐ進める順序

当面の実装順は以下で固定する。

1. `CausalAnalyticFrontend` の進捗を正式文書へ反映する
2. formal packet / metadata 契約の不足分を洗い出す
3. `FormalComplexPRHalfbandStage` を full tree 規約へ寄せる
4. real input -> front-end -> formal tree -> real output の最小 end-to-end 検証を作る
5. stage-level polyphase streaming の正式設計を固定する
6. stage-level polyphase streaming を実装する（2026-07-02 完了）
7. その後に stage 係数最適化へ戻る

---

## 5. 現時点の判断

現状では、

- 構造が未熟だから最適化前に止まっている

のではなく、

- 最適化しなくても正式化できる部品の整理順がまだ甘い

という状態に近い。

したがって当面は、

- 最適化課題を先頭に置かず
- 契約と構造を先に正式化し
- 最適化はその後に戻る

方針を採るのが妥当である。
