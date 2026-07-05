# 運用スパースアレイへの SLC 導入検討

## 1. 目的

本書は、運用スパースアレイ、保存済み小数遅延 FIR バンク、151 本待受ビームを前提に、SLC をどの形で導入するかを整理する。

今回の対象は、固定整相後の beam output から共分散を作る beam-domain GSC / SLC である。周波数ビン別 SLC は性能面では有利だが、CPU 実時間性と統計安定性を評価条件に含め、時間領域 L=1 と narrowband 診断を用途別に分けて扱う。

## 2. 前提条件

評価対象は次のファイル入力で固定する。

```text
array:
  artifacts/beamforming/operational_sparse_array/operational_sparse_array_fs32768.json

fractional delay FIR bank:
  artifacts/beamforming/fractional_delay_filter_bank_65x63.npz

fixed 151-beam shading:
  artifacts/beamforming/operational_shading_fixed_beam/operational_kaiser_bessel_shading_151beam.json
```

151 本ビーム固定のシェーディングは、現状 `selected_kaiser_beta = 0.0` である。したがって active channel 内は矩形窓と等価であり、既存の小数遅延整相器へそのまま接続できる。非ゼロ beta を評価する場合は、channel 加重付き delay-and-sum と `sum(weights)` 正規化を整相器へ追加してから SLC 評価へ進む。

## 3. SLC 方式

### 3.1 時間領域 L=1 方式

時間領域 L=1 方式は、固定整相後の beam output を入力する beam-domain SLC とする。

```text
beam_output: [n_beam, n_sample]
target:      [n_target]
reference:   guard 外 beam
R_uu:        [n_ref, n_ref]
r_ud:        [n_target, n_ref]
w:           [n_target, n_ref]
```

固定整相は小数遅延 FIR を用いる。SLC の参照ビームは target beam の guard 外から選ぶ。target 近傍を参照へ混ぜると desired 成分を干渉として学習し、self-nulling を起こすためである。

### 3.2 scan 全ビーム SLC と target-centric SLC の違い

BL/FRAZ/BTR を全待受ビームで表示する診断では、各 beam を順番に look 方向として扱い、それぞれの mainlobe を保護して SLC を適用する。この方式では、実在する干渉源の方位に対応する beam も、その beam にとっては desired mainlobe である。

したがって、scan 全ビーム SLC の BL/FRAZ/BTR では、干渉源のピーク自体が消えない。これは異常ではなく、全方位監視の表示としては正しい。一方、target beam に混入する干渉源 sidelobe を評価するには、次の別指標が必要になる。

```text
target-centric leakage:
  指定 target beam 出力に対して、
  interferer-only または既知成分分離で干渉漏れ込み量を見る。

all-source-excluded sidelobe:
  BL 評価時に target mainlobe だけでなく、
  既知 source 方位すべての mainlobe 範囲を除外し、
  真の sidelobe floor / peak を比較する。
```

## 4. 10000 Hz 初期評価

追加した診断スクリプト:

```text
examples/beamforming/operational_array_fractional_delay_slc_diagnostics.py
```

評価条件:

```text
fs                  = 32768 Hz
frequency           = 10000 Hz
active channel      = 19 ch
active aperture     = 0.9 m
beam count          = 151
target              = 90 deg, 10000 Hz, 0 dB20
interferer          = 60 deg, 10000 Hz, -6 dB20
guard               = 10 beams
max_reference_beams = 48
slc_analysis_block  = 64 samples
n_snapshot          = 512
loading             = 3.0e-2
eta_limited         = 0.15
```

生成物:

```text
artifacts/beamforming/operational_fractional_delay_slc_diagnostics/10000Hz_151beam/summary.json
artifacts/beamforming/operational_fractional_delay_slc_diagnostics/10000Hz_151beam/slc_summary.json
artifacts/beamforming/operational_fractional_delay_slc_diagnostics/10000Hz_151beam/operational_slc_case_summary.json
artifacts/beamforming/operational_fractional_delay_slc_diagnostics/10000Hz_151beam/fraz.png
artifacts/beamforming/operational_fractional_delay_slc_diagnostics/10000Hz_151beam/btr.png
artifacts/beamforming/operational_fractional_delay_slc_diagnostics/10000Hz_151beam/slc_fraz.png
artifacts/beamforming/operational_fractional_delay_slc_diagnostics/10000Hz_151beam/slc_btr.png
artifacts/beamforming/operational_fractional_delay_slc_diagnostics/10000Hz_151beam/slc_bl_compare.png
```

結果:

```text
mainlobe_preserved             = true
mainlobe_level_delta_db        = +0.0003 dB
peak_azimuth_shift_deg         = 0.0 deg
mean_selected_reference_beams  = 48
limited_beam_count             = 151
disabled_beam_count            = 0
```

SLC による target mainlobe の崩れは見られなかった。一方、既存の `sidelobe_reduction_db` はほぼ 0 dB であった。これは scan 全ビーム SLC では interferer 方位の beam も保護対象になるため、干渉源ピークが残ることと整合する。

## 5. 問題点

### 5.1 sidelobe 指標が複数音源条件に未対応

現行の target 指標は target mainlobe 近傍だけを除外して guard 外最大値を計算する。そのため、同一周波数の interferer が存在すると、interferer の実ピークを sidelobe と誤って数える。

この条件では、SLC が target mainlobe を維持していても、`sidelobe_reduction_db` は SLC 性能を正しく表さない。

### 5.2 scan 表示では干渉源ピーク抑圧を期待しない

全方位 scan の BL/FRAZ/BTR は、各 beam を look 方向として出力する表示である。したがって、干渉源の到来方位に対応する beam 出力は残る。

干渉源を消す評価をしたい場合は、指定 target beam 出力に対する target-centric SLC の BTR、または interferer-only シーンを使った漏れ込み抑圧量で評価する。

### 5.3 全 beam が LIMITED_REFERENCE 扱い

151 本ビーム、guard=10 では raw reference は平均約 130.7 本ある。一方、初期評価では CPU と snapshot 数を考慮して `max_reference_beams=48` へ制限しているため、全 beam が `LIMITED_REFERENCE` になる。

これは異常ではないが、係数自由度を意図的に制限している状態である。今後、1 秒以内 CPU 処理の実測と合わせて、`max_reference_beams` と `slc_analysis_block_size` を再調整する。

## 6. 次に実装すること

次の順で進める。

1. 複数音源 BL 評価で、既知 source 方位すべての mainlobe を除外した sidelobe 指標を追加する。
2. target beam 出力だけを見る target-centric SLC 診断を追加し、interferer-only / target+interferer の漏れ込み低下量を測る。
3. 非ゼロ beta のシェーディングを SLC 評価へ入れるため、channel 加重付き小数遅延整相を実装する。
4. 1 秒入力に対する固定整相 + SLC の CPU 処理時間を測定し、`n_ref` 上限と block 長を決める。


## 7. 追加診断: 10000 Hz 同一周波数トーン条件ではキャンセル失敗

### 7.1 確認内容

10000 Hz、target 90 deg、interferer 60 deg の同一周波数トーン条件で、target beam だけを取り出して成分分解した。

結果は次である。

```text
target beam                 = 90.0 deg
reference raw count          = 130
reference selected count     = 48
interferer leakage before    = -34.06 dB
interferer leakage after     = -21.98 dB
cancel estimate from target  = -70.72 dB
cancel estimate from interferer = -7.98 dB
R condition number           = 20.65
```

この結果から、SLC は干渉漏れ込みを下げていない。むしろ target beam 上では interferer 成分を約 12 dB 悪化させている。

### 7.2 原因の解釈

この条件では target と interferer が同一周波数の決定論的トーンであり、複素 snapshot 上で強く相関する。beam-domain GSC の blocking projector により reference 側の target 空間応答は十分に下がっているが、target beam 出力 `d` には desired target が残る。

したがって、相互相関

```text
r_ud = E{u d*}
```

には、interferer reference と desired target の相関が混入する。これは SLC が「target beam に漏れた干渉」ではなく「target と相関した reference 成分」を説明しようとする状態であり、過大なキャンセル推定を作る。

今回の `cancel estimate from interferer = -7.98 dB` は、実際の target beam 上の interferer leakage `-34.06 dB` より 26 dB 以上大きい。これは係数が物理的な漏れ込み量を推定できていないことを示す。

### 7.3 判定

現行の narrowband scan SLC は、この同一周波数トーン条件では不採用とする。

少なくとも次を満たすまで、SLC 効果ありとは判定しない。

```text
interferer leakage after < interferer leakage before
かつ
protected target mainlobe delta が許容範囲内
かつ
SLC 出力パワーが固定整相より増加しない
```

### 7.4 次の修正方針

次に実装する内容は次とする。

1. target beam だけを評価する target-centric leakage 診断を正式に追加する。
2. SLC 後に target beam 出力または推定干渉成分が増加する場合、eta=0 として固定整相へ戻す safety gate を入れる。
3. 同一周波数の決定論的トーンではなく、非相関な広帯域干渉または target absent 区間を使った共分散推定条件を追加する。
4. 同一周波数・高相関干渉を扱う場合は、時間領域 L=1 SLC ではなく、追加制約付き GSC、帯域別 SLC、または学習区間分離を検討する。


## 8. L=1 時間領域方式: SLC 漏れ込み診断

### 8.1 実装

L=1 時間領域方式として、固定整相後の時間領域 beam output から共分散を 1 つ作る SLC 診断を追加した。

```text
src/spflow/beamforming/operational_time_domain_slc_diagnostics.py
examples/beamforming/operational_array_time_domain_slc_diagnostics.py
tests/beamforming/test_operational_time_domain_slc_diagnostics.py
```

この診断では、mixed / target-only / interferer-only の 3 ケースを同じ小数遅延固定整相へ通す。mixed から SLC 係数を学習し、その同じ係数を target-only / interferer-only にも適用することで、target beam 上でどの成分が削られたかを直接評価する。

### 8.2 評価条件

```text
fs                  = 32768 Hz
frequency           = 10000 Hz
active channel      = 19 ch
active aperture     = 0.9 m
beam count          = 151
target              = 90 deg, 10000 Hz, 0 dB20
interferer          = 60 deg, 10000 Hz, -6 dB20
guard               = 10 beams
reference beams     = 130
block size          = 32768 samples
loading             = 3.0e-2
eta                 = 1.0
```

生成物:

```text
artifacts/beamforming/operational_time_domain_slc_diagnostics/10000Hz_151beam/time_domain_slc_leakage_summary.json
artifacts/beamforming/operational_time_domain_slc_diagnostics/10000Hz_151beam/target_leakage_levels.png
```

### 8.3 評価結果

```text
mixed before                 = -0.47 dB20
mixed after raw SLC          = -18.59 dB20
target before                = -0.31 dB20
target after raw SLC         = -16.35 dB20
interferer leakage before    = -31.05 dB20
interferer leakage after     = -20.70 dB20
interferer reduction         = -10.35 dB
recommended output           = fixed_beamformer
```

時間領域 SLC は mixed 出力全体を下げているが、その主因は target 成分の自己消去である。interferer leakage は 10.35 dB 悪化しており、キャンセル性能は出ていない。

### 8.4 判定

この条件では、時間領域 SLC L=1、共分散 1 個の方式は不採用とする。

理由は次である。

- target と interferer が同一周波数の決定論的トーンであり、時間領域共分散でも target / interferer を分離できない。
- guard 外 reference に target と相関した成分が残り、`r_ud` が desired 成分を説明する方向へ寄る。
- 結果として target を大きく削り、interferer leakage は増加する。

### 8.5 次の方式候補

次は、以下の順で検討する。

1. target absent 区間または training 区間を使い、desired target を含まない共分散で SLC 係数を推定する。
2. 帯域別 SLC または STFT-bin SLC に進み、同一周波数トーンではなく広帯域干渉で評価する。
3. target response に対する多制約 GSC を導入し、desired 成分の blocking をより厳密にする。
4. safety gate は必須とし、SLC 後に target power または interferer leakage が悪化する場合は固定整相へ戻す。


### 追記: L=1 時間領域方式の運用出力定義

ここでいう L=1 時間領域方式は、単なる診断コードではなく、`BeamDomainSLC.process()` が実際に返す運用出力経路を指す。今回、以下を正式実装として追加した。

```text
raw SLC candidate:
  y_raw = d - eta * c

safety gate:
  出力パワー増加、出力パワー過大低下、推定キャンセル成分過大を検出

fallback:
  safety gate が発火した場合、mode=SAFETY_FALLBACK, eta=0 とし、Y は固定整相 target 出力を返す
```

10000 Hz 同一周波数トーン条件では raw SLC candidate は失敗する。

```text
raw target power delta       = -16.04 dB
raw interferer reduction     = -10.35 dB
safety reason                = output_power_drop
effective target power delta = 0.00 dB
effective interferer reduction = 0.00 dB
recommended output           = fixed_beamformer
```

したがって、この条件では SLC は採用されず、正式経路は固定整相へ戻る。これは性能が出たという意味ではなく、悪化する SLC 出力を後段へ出さないための最低限の運用安全策である。


### 追記: safety gate 無効での SLC 利用条件スイープ

固定整相後 beam output から時間領域共分散を作る SLC について、safety gate を無効化し、raw SLC の使える条件を確認した。今回は desired response blocking を入れた beam-domain GSC として評価した。

生成物:

```text
artifacts/beamforming/operational_time_domain_slc_condition_sweep/condition_sweep_summary.json
artifacts/beamforming/operational_time_domain_slc_eta_probe/eta_probe_summary.json
```

代表結果:

```text
target-only, eta=0.25:
  target power delta       = -0.99 dB
  判定                     = target 維持は概ね可能

target-only, eta=1.0:
  target power delta       = -2.98 dB
  判定                     = eta が大きすぎる

same frequency interferer, eta=0.25:
  target power delta       = -0.94 dB
  interferer reduction     = -13.01 dB
  判定                     = 不可。干渉を増やす

same frequency interferer, eta=1.0:
  target power delta       = -2.94 dB
  interferer reduction     = -23.76 dB
  判定                     = 不可。干渉を大きく増やす

different frequency interferer 8192 Hz, eta=0.5:
  target power delta       = -1.90 dB
  interferer reduction     = +5.85 dB
  判定                     = target 低下を許容できるなら候補

different frequency interferer 6144 Hz, eta=0.5:
  target power delta       = -1.90 dB
  interferer reduction     = +5.60 dB
  判定                     = target 低下を許容できるなら候補
```

結論:

- target-only 高 SNR では、desired response blocking を入れれば自己消去は大幅に改善する。ただし eta=1.0 は強すぎる。
- 同一周波数の決定論的 target / interferer は、時間領域 L=1 SLC では使えない。eta を下げても干渉成分が増える。
- 異周波数 interferer では SLC が効く条件がある。eta=0.5 付近で target 約 1.9 dB 低下、干渉約 5.6～5.8 dB 低減となった。
- 実運用では eta を固定 1.0 にせず、target 保護条件と干渉低減量のトレードオフで決める必要がある。

---

## 9. 2026-07-05 レビュー反映

### 9.1 実装へ反映した評価漏れ

`slc_same_frequency_interference` と `slc_runtime` では、SLC 共分散・係数の健全性が必須評価である。これに対して、時間領域 SLC 診断 summary は `weight_norm` だけを出し、condition number を出していなかった。

対応として、`BeamDomainSLC.process()` の結果へ `covariance_condition_number` を追加し、診断 summary の `slc_process.condition_number` へ `R_uu + loading I` の condition number を記録する。raw `R_uu` ではなく loaded covariance を見るのは、実際に `np.linalg.solve` が解く行列の数値安定性を評価するためである。

### 9.2 L=1 と時間タップ付き SLC の分離

現在の `BeamDomainSLC` は、

```text
y[n] = d[n] - w^H u[n]
```

を解く L=1 方式である。時間タップ付き SLC では、

```text
u_L[n] = [u[n], u[n-1], ..., u[n-L+1]]
```

として reference 行列を `[n_ref * L, n_sample - L + 1]` へ変換する必要がある。したがって、`tap_len > 1` は時間タップ付き SLC として reference 行列を `[n_ref * L, n_sample - L + 1]` へ展開し、先頭 `L-1` サンプルは履歴不足として固定整相出力を通す。

### 9.3 採否条件

時間領域 L=1 は、異周波数 interferer のように target / interferer の相関が十分に低い条件では候補になる。一方、同一周波数 deterministic tone のような高相関条件では、干渉を増やす結果が出ているため採用しない。

採用判定では以下を同時に確認する。

```text
slc_target_only:
  target_power_delta_db, waveform integrity, input/output level consistency

slc_same_frequency_interference:
  raw_interferer_reduction_db, mainlobe preservation, condition_number, weight_norm

slc_different_frequency_interference:
  raw_interferer_reduction_db, target_power_delta_db, waveform integrity

slc_runtime:
  realtime_factor, reference_beam_count, condition_number
```

### 9.4 次の設計作業

1. target absent / training 区間を使う共分散推定モードを設計する。
2. target が複数または近接する場合に備え、desired response blocking を multi-constraint 化する。
3. 時間タップ付き SLC は、reference 行列を `[n_ref * L, n_sample - L + 1]` へ展開し、先頭 `L-1` sample を固定整相として扱う条件で L=3 / 5 / 8 を評価する。
4. 時間タップ付き SLC でも採用条件を満たさない条件では、時間領域 MVDR / LCMV または STFT-bin beam-domain GSC の詳細設計へ進む。

---

## 10. 同一方位・複数周波数を扱う方針

同一方位に複数周波数がある条件では、空間方向の steering は同じである。したがって、beam-domain SLC により reference beam から片方だけを推定して消す設計にはしない。

運用上は、固定整相後の target beam または beamspace 出力を STFT / subband 化し、周波数成分ごとに別 track として扱う。

```text
fixed beam output:
  beam_output[beam, sample]

frequency analysis:
  X[beam, freq, frame]

同一方位・複数周波数の出力:
  target_azimuth_deg は共通
  target_frequency_hz ごとに level / leakage / waveform を評価
```

この条件では、FRAZ 上に同じ方位で複数 frequency ridge が立つことを期待する。BTR は方位方向では 1 本の ridge に見えてよい。周波数を分ける必要がある後段では、BTR ではなく周波数別 BTR または target beam の STFT level track を見る。

SLC は、同一方位の別周波数成分を消すためではなく、別方位から target beam へ入る sidelobe 成分を抑えるために使う。

---

## 11. 時間領域優先の運用方針

運用では、まず時間領域固定整相と時間領域 SLC を評価する。L=1 で不足する場合は、STFT へ進む前に `tap_len=3 / 5 / 8` の FIR 型 SLC を評価する。

STFT / subband 方式を使う場合も、リアルタイム経路と学習経路を分ける。リアルタイム側は固定整相と既に決まった係数の適用に限定し、MUSIC、EVD、MVDR / LCMV 重み更新は学習側で低頻度に行う。

この分離により、STFT 方式へ切り替えても、毎サンプルまたは毎短時間フレームで重い固有値分解や全 bin solve を行う設計を避ける。
