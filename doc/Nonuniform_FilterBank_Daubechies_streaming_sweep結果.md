# Nonuniform FilterBank Daubechies streaming sweep結果

## 1. 目的

本書は、Daubechies 系の不均一 beamforming 試作について、

- streaming 統合を入れたか
- 小さい `chunk_size` でどの程度処理時間が増えるか
- matched beam peak level が維持されるか
- 合成後時間波形の `jump_abs` が悪化しないか

を記録する文書である。

対象実装:

- `src/spflow/filterbank/daubechies_nonuniform_streaming.py`
- `src/spflow/filterbank/daubechies_nonuniform_beamformer.py`
- `examples/nonuniform/nonuniform_daubechies_streaming_sweep.py`

---

## 2. 今回の streaming 統合

今回追加した `DaubechiesNonuniformBeamformerStreaming` は、
以下の構造を持つ。

1. stage-level streaming analyzer を不均一木へ再帰接続する
2. leaf ごとに persistent な `NonuniformLeafProcessor` を持つ
3. leaf beamforming 出力を蓄積する
4. root 出力は、formal metadata 付き増分 synthesis tree で逐次合成する

注意:

- 解析木と leaf beamforming は stateful / streaming である
- 現在の正式実装では、root 合成は各内部ノードの streaming synthesizer を使う増分方式へ差し替え済みである
- 本書のベンチマーク値は、旧 prefix 再構成 prototype を測った履歴として残す

---

## 3. 処理時間見積り

小さい `chunk_size` を使う前に、
1 周波数・`n_sample = 16384` と `8192` の代表条件で実測した。

### 3.1 実測ベンチマーク1

条件:

- `freq = 1536 Hz`
- `n_sample = 16384`
- dense center / sparse edge 32 ch

結果:

| chunk_size | process calls | elapsed [s] |
|---|---:|---:|
| `8` | `2048` | `40.94` |
| `16` | `1024` | `21.18` |
| `32` | `512` | `10.93` |
| `64` | `256` | `5.40` |
| `128` | `128` | `2.88` |
| `257` | `64` | `1.53` |

### 3.2 実測ベンチマーク2

条件:

- `freq = 1536 Hz`
- `n_sample = 8192`

結果:

| chunk_size | process calls | elapsed [s] |
|---|---:|---:|
| `16` | `512` | `5.66` |
| `32` | `256` | `2.94` |
| `64` | `128` | `1.56` |
| `128` | `64` | `0.78` |

### 3.3 見積り判断

このベンチマーク時点の試作 streaming は、root 合成を毎回 prefix 再構成していたため、
概ね `O(N^2 / chunk_size)` に近い増え方を示した。

そのため当時の sweep は以下の 2 段に分けた。

1. `n_sample = 8192`, `chunk_size = 16, 32, 64`
   : 多周波数 sweep で peak level と誤差量を見る
2. `n_sample = 65536`, `chunk_size = 128`
   : root 出力が複数 chunk に分かれて現れる条件で boundary `jump_abs` を見る

この分け方により、

- `chunk_size` を十分小さくする
- 総時間を実用的範囲に抑える
- boundary continuity も確認する

を両立した。

---

## 4. 短尺 streaming sweep

条件:

- `n_sample = 8192`
- `chunk_size = 16, 32, 64`
- `freq = 64, 192, 384, 768, 1536, 3072, 6144, 12288 Hz`
- 入力: broadside analytic tone を全チャネルへ同相で入力
- 指標:
  - `peak_response_db`
  - `max_abs_error`
  - `rms_error`
  - `max_jump_abs_error`

結果:

| freq [Hz] | chunk_size | peak_response_db | max_abs_error | rms_error | max_jump_abs_error |
|---|---:|---:|---:|---:|---:|
| `64` | `16` | `0.000000` | `1.25e-15` | `1.73e-16` | `6.75e-16` |
| `64` | `32` | `0.000000` | `1.25e-15` | `1.70e-16` | `6.75e-16` |
| `64` | `64` | `0.000000` | `1.23e-15` | `1.77e-16` | `6.75e-16` |
| `192` | `16` | `0.000000` | `8.88e-16` | `1.50e-16` | `5.66e-16` |
| `192` | `32` | `0.000000` | `8.88e-16` | `1.50e-16` | `5.66e-16` |
| `192` | `64` | `0.000000` | `1.22e-15` | `2.13e-16` | `5.80e-16` |
| `384` | `16` | `0.000000` | `1.01e-15` | `2.26e-16` | `6.67e-16` |
| `384` | `32` | `0.000000` | `1.00e-15` | `2.05e-16` | `6.67e-16` |
| `384` | `64` | `0.000000` | `9.09e-16` | `2.09e-16` | `6.66e-16` |
| `768` | `16` | `0.000000` | `8.88e-16` | `1.92e-16` | `6.75e-16` |
| `768` | `32` | `0.000000` | `9.49e-16` | `1.52e-16` | `5.61e-16` |
| `768` | `64` | `0.000000` | `5.66e-16` | `7.20e-17` | `5.55e-16` |
| `1536` | `16` | `0.000000` | `8.01e-16` | `1.72e-16` | `6.66e-16` |
| `1536` | `32` | `0.000000` | `5.55e-16` | `7.83e-17` | `5.55e-16` |
| `1536` | `64` | `0.000000` | `4.85e-16` | `4.91e-17` | `5.55e-16` |
| `3072` | `16` | `0.000000` | `4.58e-16` | `6.15e-17` | `4.44e-16` |
| `3072` | `32` | `0.000000` | `4.44e-16` | `4.93e-17` | `4.44e-16` |
| `3072` | `64` | `0.000000` | `3.33e-16` | `1.96e-17` | `3.33e-16` |
| `6144` | `16` | `0.000000` | `3.33e-16` | `4.24e-17` | `3.33e-16` |
| `6144` | `32` | `0.000000` | `4.44e-16` | `3.12e-17` | `4.44e-16` |
| `6144` | `64` | `0.000000` | `2.22e-16` | `1.19e-17` | `2.22e-16` |
| `12288` | `16` | `0.000000` | `1.11e-16` | `9.38e-18` | `1.11e-16` |
| `12288` | `32` | `0.000000` | `1.11e-16` | `7.07e-18` | `1.11e-16` |
| `12288` | `64` | `0.000000` | `1.11e-16` | `4.97e-18` | `1.11e-16` |

所見:

- matched beam peak level は全条件 `0 dB` を維持した
- streaming / offline 差分は全帯域で machine precision レベルである
- `n_sample = 8192` では root 出力が flush 側へ寄るため、boundary `jump_abs` 指標は有効に取りにくい

短尺 sweep 実測時間:

- `81.94 s`

---

## 5. 長尺 continuity sweep

boundary `jump_abs` を見るため、root 出力が複数 chunk に分かれて現れる条件を追加した。

条件:

- `n_sample = 65536`
- `chunk_size = 128`
- `freq = 64, 1536, 12288 Hz`
- 実際の非零 root 出力 chunk 数: すべて `3`

結果:

| freq [Hz] | chunk_size | peak_response_db | max_abs_error | rms_error | max_jump_abs_error | max_boundary_jump_abs_error | nonzero output chunks |
|---|---:|---:|---:|---:|---:|---:|---:|
| `64` | `128` | `0.000000` | `1.34e-15` | `2.18e-16` | `8.90e-16` | `1.39e-17` | `3` |
| `1536` | `128` | `0.000000` | `5.55e-16` | `2.70e-17` | `4.44e-16` | `0.00e+00` | `3` |
| `12288` | `128` | `0.000000` | `1.11e-16` | `4.79e-18` | `2.22e-16` | `0.00e+00` | `3` |

長尺 continuity sweep 実測時間:

- `125.20 s`

所見:

- boundary `jump_abs` は `1.39e-17` 以下であり、可視的・実用的な段差は見られない
- matched beam peak level は長尺条件でも `0 dB` を維持した
- root 出力の chunk 境界でも streaming / offline 一致は保たれている

---

## 5.1 Phase B reconstructed-output sanity check (2026-07-02)

`leaf_independent_one_sided` を追加したため、
streaming / offline 一致だけでなく、再合成後 root-rate 出力そのものが
致命的に崩れていないかを代表条件で追加確認した。

条件:

- output path: `leaf_independent_one_sided`
- steering angle: `10 deg`
- source: constant-envelope analytic tone
- frequency: `1536 Hz`
- sample length: `16384`
- scan angles: `-30, -25, ..., 30 deg`

確認項目 1: 再合成後の root-rate 出力を再度 FFT したときの peak angle

- 各 angle の single-source scene を beamform して root-rate 時系列を得る
- その出力 FFT の `1536 Hz` bin 振幅を angle ごとに比較する
- 最大は `10 deg` であり、steering 方向と一致した

代表値:

| source angle [deg] | reconstructed FFT magnitude at `1536 Hz` |
|---|---:|
| `-30` | `0.5604` |
| `-20` | `0.6756` |
| `-10` | `0.7736` |
| `0` | `0.8384` |
| `5` | `0.8539` |
| `10` | `0.8590` |
| `15` | `0.8531` |
| `20` | `0.8360` |
| `30` | `0.7799` |

確認項目 2: 再合成後時間波形の boundary `jump_abs`

- irregular chunk streaming で root-rate 出力を生成
- `jump_abs = |y[n] - y[n-1]|` を直接評価
- chunk boundary の最大 jump は、信号全体の median jump の `0.437x` であり、
  境界だけ突出した段差は観測されなかった

代表値:

- `median jump_abs = 2.916e-01`
- `max boundary jump_abs = 1.274e-01`
- `boundary / median = 0.437`

この代表条件では、
Phase B 出力 path は少なくとも

- steering 方向の peak を root-rate 再合成後にも保っている
- root 出力 chunk 境界で不連続を作っていない

と判断してよい。

---


## 5.2 MVDR multi-frequency / multi-angle sanity check (2026-07-02)

CBF 側では broadside / matched 条件を先に見たが、
MVDR についても representative な多周波・多方位条件で
再合成後 root-rate 出力が致命的に崩れていないかを追加確認した。

条件:

- output path: `leaf_independent_one_sided`
- beamformer: `mvdr`
- steering angle: 基本条件 `10 deg`
- peak-angle sweep frequency: `192, 1536, 12288 Hz`
- scan angles: `-30, -20, -10, 0, 10, 20, 30 deg`
- continuity check cases: `(192 Hz, 0 deg)`, `(1536 Hz, 10 deg)`, `(12288 Hz, 20 deg)`
- interferer check cases:
  - `(192 Hz, target 0 deg, interferer -20 deg)`
  - `(1536 Hz, target 10 deg, interferer -25 deg)`
  - `(12288 Hz, target 20 deg, interferer -10 deg)`

確認項目 1: 再合成後 root-rate FFT の peak angle

| freq [Hz] | peak angle [deg] | max magnitude | second magnitude | margin |
|---|---:|---:|---:|---:|
| `192` | `10` | `0.8520` | `0.8250` | `0.0271` |
| `1536` | `10` | `0.8542` | `0.2079` | `0.6464` |
| `12288` | `10` | `0.9893` | `0.0293` | `0.9600` |

この sweep では、全代表周波数で peak は steering 方向 `10 deg` に来た。
低域 `192 Hz` は主 lobe が広いため margin は小さいが、peak 自体は正しい。

確認項目 2: 再合成後時間波形の boundary `jump_abs`

| freq [Hz] | source angle [deg] | max boundary jump_abs | jump_abs p95 | boundary / p95 |
|---|---:|---:|---:|---:|
| `192` | `0` | `6.10e-02` | `6.47e-02` | `0.943` |
| `1536` | `10` | `1.26e-01` | `7.97e-01` | `0.159` |
| `12288` | `20` | `1.63e-03` | `1.41e-02` | `0.116` |

この代表条件では、chunk boundary の jump は全体 jump 分布の `95 percentile` を超えず、
境界だけが突出する不連続は見られなかった。

確認項目 3: interferer 条件での MVDR 改善

| freq [Hz] | target angle [deg] | interferer angle [deg] | CBF rms error | MVDR rms error | MVDR / CBF |
|---|---:|---:|---:|---:|---:|
| `192` | `0` | `-20` | `1.8144` | `1.3560` | `0.747` |
| `1536` | `10` | `-25` | `1.5484` | `1.0774` | `0.696` |
| `12288` | `20` | `-10` | `1.1536` | `0.9757` | `0.846` |

少なくともこの 3 条件では、再合成後 root-rate 出力で見ても
MVDR は CBF より target 参照との差を小さくできた。

したがって、現時点では MVDR についても representative な多周波・多方位条件で

- steering 方向の peak は保たれる
- root-rate 時間波形に boundary jump は見られない
- interferer 条件では CBF より改善が出る

と整理してよい。

ただし、これはまだ representative sweep であり、
全 leaf / 全角度 / 全 update 条件の exhaustive evaluation ではない。

---

## 6. 現在の位置付け

今回の結果から、少なくとも以下は確認済みである。

- CBF / broadside 条件で matched peak level は崩れない
- root-rate 合成後時間波形に chunk 境界 jump は見られない
- `leaf_independent_one_sided` を含む representative な MVDR 多周波・多方位条件でも peak angle は保たれる
- representative な MVDR interferer 条件では、再合成後 root-rate 出力で見ても CBF より改善が出る
- streaming / offline 一致は回帰試験で確認済みである

一方で、今も残っている課題は以下である。

1. 本書の処理時間値は旧 prefix 再構成 prototype の履歴であり、現行実装の正式処理時間評価ではない
2. multiband interferer 条件での MVDR 実用評価はまだ未整理である
3. real-input beamforming streaming の正式評価結果はまだ別文書として整理しきれていない
4. 現行の各内部 node は `ComplexFIRHalfbandStageStreamingSynthesizer` の exact-by-construction 実装を使っており、最終的な高速化評価はなお別途必要である

したがって本書の成果は、

- 正式 streaming 構造の feasibility 確認
- representative 条件での continuity / peak-angle / MVDR sanity 確認
- 旧 benchmark 履歴の保存

として位置付けるのが適切である。
