# CausalAnalyticFrontend 正式仕様

## 1. 目的

本書は、不均一複素フィルタバンク正式構造の先頭に置く
`CausalAnalyticFrontend` の仕様を固定する。

`ComplexPRHalfbandStage` が不均一木の基本部品であるのに対し、
本部品は

- 実入力
- 複素不均一木入力

の間をつなぐ正式 front-end である。

---

## 2. 位置付け

正式構造では、

- 実入力をそのまま木へ入れない
- causal analytic front-end を通してから木へ入れる

方針を採用している。

したがって本部品は、

- 実信号を複素 analytic 表現へ変換する
- その delay と time origin を明示する
- streaming で offline と一致する

ことが要求される。

---

## 3. 入出力契約

入力:

- 実数時系列 `x[n]`
- 多チャネルなら shape `(n_ch, n_sample)`

出力:

- 複素時系列 `z[n]`
- shape は入力と同じ time length を持つ
- 正側帯域優勢の analytic representation

metadata:

- `delay_samples_at_root_rate`
- `time_origin_at_root_rate`

---

## 4. 正式仕様

正式版 `CausalAnalyticFrontend` は以下を満たす。

1. causal
2. streaming 可能
3. 明示的な整数遅延を持つ
4. 負周波数成分を十分抑圧する
5. その出力を `ComplexPRHalfbandStage` へそのまま渡せる

---

## 5. 採用候補

正式版 v1 の第一候補は以下とする。

- FIR Hilbert transformer 系 analytic front-end

すなわち、

- 実信号の直交成分を FIR Hilbert 変換で生成し
- `z[n] = x[n] + j * x_hat[n]`

で複素表現を作る方式である。

理由:

- causal 化しやすい
- delay が明示的
- streaming 実装しやすい
- C++ 実装へ移しやすい

---

## 6. 非採用事項

正式版 v1 では、以下を front-end として採用しない。

- offline FFT ベース helper を正式版とすること
- 非 causal な Hilbert 変換
- 入力長全体を見ないと出力できない方式

ただし、FFT helper は

- 比較基準
- 検証基準

として残してよい。

---

## 7. 性能要求

正式版 front-end の初期要求は以下とする。

- 負周波数抑圧: 実用上十分小さいこと
- passband 振幅誤差: 実用上十分小さいこと
- group delay: 明示的整数遅延で管理できること
- streaming / offline 一致

ここで厳密な dB 値は、
stage と木を含む全体最適化の段階で再調整してよい。

ただし少なくとも、

- 後段の不均一木より front-end の誤差が支配的にならない

ことを要求する。

---

## 8. 現在の進捗

最小実装と試験はすでに追加済みである。

対応実装:

- `src/spflow/filterbank/causal_analytic_frontend.py`

対応試験:

- `tests/filterbank/test_causal_analytic_frontend.py`

確認済み内容:

1. odd-length Hilbert FIR の最小設計
2. streaming / offline 一致
3. `recover_real()` による実信号復元
4. 正周波数 tone に対する負周波数抑圧の最小確認

したがって本部品は、

- 未着手

ではなく、

- 最小実装済みで formal tree 接続済み
- formal metadata 付き streaming 接続済み

の状態である。

対応結果は `doc/Nonuniform_FilterBank_formal_tree接続結果.md` に整理した。

---

## 9. Pending

本部品には、現時点で以下の Pending がある。

### P-CAF-1

Hilbert FIR の tap 長と stopband の初期目標値を固定していない。

### P-CAF-2

front-end 単体の negative-frequency suppression を
どう評価するかの正式メトリクスが未固定である。

### P-CAF-3

front-end と `ComplexPRHalfbandStage` の delay 合成規約を
実装仕様まで落とし込めていない。

---

## 10. 次の方針

`ComplexPRHalfbandStage` の high-stopband 係数最適化とは独立に、
front-end は先に正式化を進めてよい。

したがって次は、

1. suppression / delay の正式評価指標を固定する
2. formal tree との delay 合成規約を固定する
3. beamforming を含む正式 streaming へ接続する

を進めるのが妥当である。
