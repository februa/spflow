# beamforming責務分割設計

## 1. 目的

`src/spflow/beamforming`には、実運用の信号処理、シミュレーション入力生成、評価metric、
BL/FRAZ/BTR描画、方式比較scenario、係数設計と成果物保存が同居している。
本設計は、通常のPython関数・クラスとして組み合わせられる性質を保ったまま、
責務境界と依存方向を整理する。

部品数を機械的に減らすことは目的にしない。異なる責務を一つの巨大moduleへまとめず、
利用者が必要な層だけをimportできる状態を完成条件とする。

## 2. 現状

調査時点の`spflow.beamforming`は38 module、約20,700行である。

| 責務群 | module数 | 概算行数 | 現在の判断 |
| --- | ---: | ---: | --- |
| 信号処理本体 | 12 | 5,698 | `spflow.beamforming`に残す |
| 共分散・方位選択支援 | 7 | 2,362 | 実運用時の必要性をmoduleごとに判断する |
| 評価・描画・診断 | 13 | 8,096 | 小部品とscenario orchestrationを分ける |
| 設計・成果物生成 | 5 | 4,093 | 定義読込と係数生成CLIを分ける |
| 公開ファサード | 1 | 433 | 評価workflowの一括再公開を縮小する |

`spflow.beamforming.__all__`は179名、トップレベル`spflow.__all__`は161名であり、
信号処理部品だけでなく描画・report生成・運用条件付きscenarioも公開APIへ含まれている。

## 3. 依存方向

依存方向は次に固定する。

```text
evaluations / tools
    ├──> simulation
    └──> beamforming_evaluation ───> beamforming
```

実際のPython importでは、`beamforming_evaluation`が方式出力を評価するため
`beamforming`を参照してよい。逆に、実運用の`beamforming`数式・streaming処理から
Matplotlib、report pack、個別scenarioをimportしてはいけない。

`simulation`は方式入力を生成するが、BL/FRAZ/BTR判定や採否判定を持たない。
`beamforming_evaluation`は固定shape結果と小さな純粋関数を提供するが、
出力directory、JSON/CSV/PNG一式を生成する巨大なRuntimeや基底クラスを持たない。

## 4. private依存の分類

### 4.1 再利用部品として抽出するもの

複数moduleから既に利用され、入力と出力の契約を方式scenarioから独立して記述できる処理を抽出する。

| 旧private処理 | 新しい責務 | 配置 |
| --- | --- | --- |
| `_direction_from_az_el` | deg角度から方向余弦`[3]`への変換 | `simulation.tone_scene` |
| `_generate_target_scene` | 平面波toneとchannel非相関noiseの生成 | `simulation.tone_scene` |
| `_build_beam_grid` | equal-cos走査gridと表示軸の固定shape結果 | `beamforming_evaluation.scan_grid` |
| `_rms_level_db20` | 任意波形のRMS level | `beamforming_evaluation.level_metrics` |
| `_real_tone_rms_level_db20` | 正負周波数応答から実tone RMS levelを合成 | `beamforming_evaluation.level_metrics` |
| `_tone_level_db20_rms` | 時間波形の単一tone射影level | `beamforming_evaluation.signal_levels` |
| `_rfft_levels_db20` | one-sided per-bin RMS spectrum | `beamforming_evaluation.signal_levels` |
| `_compress_time_rms_levels` | BTR用非overlap block RMS | `beamforming_evaluation.signal_levels` |
| `_normalize_channel_weights` | 評価時channel shadingの境界検証 | `beamforming_evaluation.fractional_response` |
| `_build_fractional_beam_response_matrix` | 小数遅延固定整相のbeam-to-beam理論応答 | `beamforming_evaluation.fractional_response` |

level関数は、戻り値を単なる「dB」としない。`reference_rms`をAPI境界で受け、
`dB re reference_rms`として解釈する。one-sided spectrumはper-bin RMSであり、
band積分RMSとは区別する。BTR用block RMSからframe最大値を引いた後だけ
`dB re frame max`と呼ぶ。

### 4.2 scenario orchestrationとして残すもの

次の処理は、設定読込、方式実行、metric集計、描画、成果物保存を順番に結ぶworkflowである。
関数全体を共通部品にすると利用者へ新しい実行モデルを強制するため、汎用部品へ昇格させない。

- `_run_fractional_delay_diagnostics`
- `run_integer_delay_diagnostics`
- `run_integer_delay_slc_diagnostics`
- `run_operational_time_domain_slc_leakage_diagnostics`
- `run_operational_time_domain_adaptive_comparison`
- `run_operational_same_azimuth_frequency_separation_diagnostics`

これらは後続段階で`evaluations/beamforming`へ移す。移動前には、内部で使う数式・metric・
固定shape結果だけを上記の小部品へ置き換える。互換期間中に旧moduleから呼び出す場合も、
他moduleがそのprivate helperをimportする構造は増やさない。

### 4.3 後続段階で判断するもの

次は複数箇所から参照されるが、現状では診断設定や成果物命名と結合している。
先に公開関数へ名前だけ変更せず、scenario移動時に入力結果型を決めてから分離する。

- `_build_array_positions`: array定義と評価用自動生成が混在している
- `_resolve_source_specs`: legacy単一source設定との互換変換である
- `_evaluate_stage_source_metrics`: BL配列計算とscenario固有判定が混在している
- `_build_source_comparisons`: target保護とsidelobe判定の閾値が埋め込まれている
- 描画caption、ファイル名、summary辞書を組み立てるhelper群

## 5. 公開境界

### 5.1 `spflow.simulation`

`ToneSceneSource`、`ToneScene`、`direction_from_azimuth_elevation`、
`synthesize_tone_scene`を公開する。`ToneScene.signal`はshape`[n_ch, n_sample]`、
`time_axis_s`はshape`[n_sample]`である。`SimulationPrecision`により
`float32`または`float64`をscene単位で選択する。

既存の`TimeDelayDiagnosticSource`は`ToneSceneSource`を継承する互換名として残す。
新しいシミュレーションでは方式名を含まない`ToneSceneSource`を使用する。

### 5.2 `spflow.beamforming_evaluation`

次の責務moduleだけを置く。

- `scan_grid`: beam方向余弦と表示軸
- `signal_levels`: tone、spectrum、block RMS
- `level_metrics`: 任意波形と正負周波数応答のRMS level
- `fractional_response`: 小数遅延固定整相の理論応答

Matplotlib描画、ファイル保存、parameter sweep、採否判定はこの段階では含めない。
評価metricを自動採否へ使う場合は、人間の視覚順位との一致率を別途検証してから追加する。

## 6. 互換性

- `TimeDelayDiagnosticSource`のconstructorとfieldを維持する。
- 既存summary key、JSON/CSV/PNG名、BL/FRAZ/BTR軸、乱数seedを変更しない。
- tone sceneの既定精度は従来と同じ`float32`とする。
- `noise_level_db20`は従来どおり時間領域sample RMSの`dB re input RMS`であり、
  one-sided ASDではない。
- equal-cos方位gridを維持し、線形角度gridへ変更しない。

## 7. 検証

抽出後は次を確認する。

1. tone RMS 0 dB入力が0 dB re input RMSになる。
2. 実toneの片側周波数応答だけが1の場合、正負合成levelは約-3.0103 dBになる。
3. one-sided spectrumの整数bin toneが入力RMSと一致する。
4. BTR block RMSのshapeが`[n_time, n_beam]`になる。
5. equal-cos gridの方向余弦shapeが`[n_beam, 3]`になる。
6. 小数遅延応答行列のaxisが`[n_observation_beam, n_look_beam]`になる。
7. tone sceneの`SimulationPrecision.SINGLE/DOUBLE`が出力dtypeへ伝播する。
8. 既存の整数遅延、小数遅延、SLC、時間領域適応比較testが同じ判定を維持する。

## 8. 後続の整理順序

1. 本設計で抽出した部品へ既存diagnosticsを切り替える。
2. private helperを跨いでいる残りのscenario workflowを`evaluations/beamforming`へ移す。
3. `diagnostic_plotting`と成果物定義を評価支援packageへ移し、Matplotlib依存を実運用層から外す。
4. array/shadingの定義読込を`beamforming`へ残し、係数探索・plot・保存CLIを`tools/beamforming`へ移す。
5. `spflow.beamforming.__all__`と`spflow.__all__`からscenario workflowを段階的に外す。
6. `fixed_delay_diff_mvdr.py`を共分散積分、差分補正FIR、streaming合流の責務で分割する。

各段階で既存import互換を必要に応じて薄いファサードとして残す。ファサードは処理を複製せず、
新配置へ委譲するだけとする。
