# beamforming責務分割設計

## 1. 目的

`src/spflow/beamforming`には、実運用の信号処理、アレイ事前設計、評価metric、
BL/FRAZ/BTR描画、SLC、方式比較scenario、係数設計と成果物保存が同居していた。
本設計は、通常のPython関数・クラスとして組み合わせられる性質を保ったまま、
責務境界と依存方向を整理する。

部品数を機械的に減らすことは目的にしない。異なる責務を一つの巨大moduleへまとめず、
利用者が必要な層だけをimportできる状態を完成条件とする。

## 2. 現状

整理前の`spflow.beamforming`は38実装module、約20,700行であった。今回の移動後は
互換facadeを含む39 module、約13,400行となり、約7,300行のarray設計・評価・SLC実装を
責務別packageへ移した。module数が減らないのは旧import用facadeを残すためである。

| 責務群 | module数 | 概算行数 | 現在の判断 |
| --- | ---: | ---: | --- |
| 信号処理本体 | 12 | 5,698 | `spflow.beamforming`に残す |
| 共分散・方位選択支援 | 7 | 2,362 | 実運用時の必要性をmoduleごとに判断する |
| 評価・描画・診断 | 13 | 8,096 | 小部品とscenario orchestrationを分ける |
| 設計・成果物生成 | 5 | 4,093 | `spflow.array_design`へ実装を移動 |
| 公開ファサード | 1 | 433 | 旧import互換facadeを含め段階的に縮小する |

`spflow.beamforming.__all__`は179名、トップレベル`spflow.__all__`は161名であり、
信号処理部品だけでなく描画・report生成・運用条件付きscenarioも公開APIへ含まれている。

## 3. 改訂した責務境界

beamforming周辺は、処理順序ではなく再利用部品の責務で次の4入口へ分ける。

| package | 責務 | 責務に含めないもの |
| --- | --- | --- |
| `spflow.beamforming` | steering/実適用係数/遅延filterの設計と、設計済み係数の信号への適用 | array候補探索、BL描画、SLC |
| `spflow.array_design` | array幾何、active channel、shadingの事前設計と設計時評価 | 実時間信号処理、SLC |
| `spflow.beamforming_evaluation` | BL/FRAZ/BTR、level、metric、可視化支援 | 重み適用、運用状態保持 |
| `spflow.sidelobe_cancellation` | beamforming後のbeam出力に対するSLC | sensor信号からのbeamforming |

位相遅延filterはbeamforming出力を実現するための係数/フィルタ設計なので
`spflow.beamforming`へ残す。array設計は性能を決める重要要素だが、信号処理実行前の
事前設計であるため`spflow.array_design`へ分ける。重要度は配置理由にしない。

## 4. 依存方向

依存方向は次に固定する。

```text
evaluations / tools
    ├──> simulation
    ├──> array_design ─────────────> beamforming
    ├──> beamforming_evaluation ───> beamforming
    └──> sidelobe_cancellation
              ^
              └── beamforming output
```

実際のPython importでは、`beamforming_evaluation`が方式出力を評価するため
`beamforming`を参照してよい。逆に、実運用の`beamforming`数式・streaming処理から
Matplotlib、report pack、個別scenarioをimportしてはいけない。

`simulation`は方式入力を生成するが、BL/FRAZ/BTR判定や採否判定を持たない。
`beamforming_evaluation`は固定shape結果と小さな純粋関数を提供するが、
出力directory、JSON/CSV/PNG一式を生成する巨大なRuntimeや基底クラスを持たない。

`beamforming`の実装moduleから`array_design`または`sidelobe_cancellation`を参照してはいけない。
旧scenario workflowと`beamforming.__init__`は互換期間だけ例外とし、処理本体を持たない
facadeとして段階的に縮小する。

## 5. 信号へのビームフォーマ係数適用

CBF、MVDR、LCMV、GSCは理論上の重み設計方法が異なるが、信号へ実際に掛ける
係数`h`を設計側で確定すれば、適用式は共通化できる。

```text
y[out, n] = sum_dof h[dof, out] * x[dof, n]
```

理論式が`y=w^H x`で記述される方式では、設計器が`h=conj(w)`へ変換して返す。
共役規約は方式を知る設計側の責務であり、適用側へbooleanや列挙値で選択させない。
時間領域FIRは`h[ch,tap]`を通常の畳み込み係数として使い、周波数領域でも
`H[ch,beam,freq] X[ch,freq]`を積和する。どちらも適用時に追加の共役を取らない。

`spflow.beamforming.application`は方式名を持たず、次を提供する。

- `apply_beamformer`: 単一帯域snapshotの`h^T x`
- `apply_beamformer_bands`: band軸を保持した`h^T x`
- `apply_beamformer_filter_fft`: 設計済みfilter FFTの周波数領域積和
- `build_time_tapped_snapshot_matrix`: channel×tap自由度への展開
- `apply_time_domain_fir_beamformer`: 時間領域FIR係数の通常の畳み込み

時間領域と周波数領域を巨大なProcessorへ統合しない。shape、axis、境界処理が異なるため
関数は分けるが、「設計済み係数を共役せず適用するだけ」という契約を共有する。

公開の標準設計関数は`design_cbf_coefficients`、`design_mvdr_coefficients`、
`design_time_domain_lcmv_coefficients`のように、戻り値が実適用係数であることを名前で示す。
旧`design_*_weights`はimport互換のため残すが、同じ実適用係数を返す。

係数はNumPy配列のまま返す。現時点ではshapeと領域を関数名・docstringで十分に
区別でき、専用の`BeamformerCoefficients`型を導入するとNumPy処理、保存、既存コードへの
局所導入に余計な依存が増えるためである。FFT長、tap原点、sample rateなど、配列だけでは
防げない不整合が複数経路で実際に確認された場合に限り、固定shape結果型を再検討する。

## 6. private依存の分類

### 6.1 再利用部品として抽出するもの

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

### 6.2 scenario orchestrationとしてevaluationsへ移すもの

次の処理は、設定読込、方式実行、metric集計、描画、成果物保存を順番に結ぶworkflowである。
関数全体を共通部品にすると利用者へ新しい実行モデルを強制するため、汎用部品へ昇格させない。

- `_run_fractional_delay_diagnostics`
- `run_integer_delay_diagnostics`
- `run_integer_delay_slc_diagnostics`
- `run_operational_time_domain_slc_leakage_diagnostics`
- `run_operational_time_domain_adaptive_comparison`
- `run_operational_same_azimuth_frequency_separation_diagnostics`

2026-07-14に、上記workflowと性能比較・guard設計を
`evaluations/beamforming/scenarios/`へ移した。移動対象は次の9 moduleである。

- `time_delay_diagnostics`
- `time_delay_slc_diagnostics`
- `fractional_delay_slc_diagnostics`
- `operational_time_domain_slc_diagnostics`
- `operational_time_domain_adaptive_comparison`
- `operational_time_domain_frequency_separation_diagnostics`
- `fractional_delay_performance`
- `operational_fractional_delay_performance`
- `time_delay_guard_design`

個別条件を固定する実行entrypointは`evaluations/beamforming/`直下に残し、scenario実装を
`scenarios/`から呼ぶ。`evaluations/`は配布packageではないため、`spflow.beamforming`から
repo内scenarioへ委譲する互換facadeは作らない。これらのConfigと`run_*`もcore公開APIから外す。

### 6.3 後続段階で判断するもの

次は複数箇所から参照されるが、現状では診断設定や成果物命名と結合している。
先に公開関数へ名前だけ変更せず、scenario移動時に入力結果型を決めてから分離する。

- `_build_array_positions`: array定義と評価用自動生成が混在している
- `_resolve_source_specs`: legacy単一source設定との互換変換である
- `_evaluate_stage_source_metrics`: BL配列計算とscenario固有判定が混在している
- `_build_source_comparisons`: target保護とsidelobe判定の閾値が埋め込まれている
- 描画caption、ファイル名、summary辞書を組み立てるhelper群

## 7. 公開境界

### 7.1 `spflow.simulation`

`ToneSceneSource`、`ToneScene`、`direction_from_azimuth_elevation`、
`synthesize_tone_scene`を公開する。`ToneScene.signal`はshape`[n_ch, n_sample]`、
`time_axis_s`はshape`[n_sample]`である。`SimulationPrecision`により
`float32`または`float64`をscene単位で選択する。

scenario内の`TimeDelayDiagnosticSource`は`ToneSceneSource`を継承し、既存評価条件のfieldを維持する。
ライブラリ利用者は方式名を含まない`ToneSceneSource`を使用する。

### 7.2 `spflow.beamforming_evaluation`

次の責務moduleだけを置く。

- `scan_grid`: beam方向余弦と表示軸
- `signal_levels`: tone、spectrum、block RMS
- `level_metrics`: 任意波形と正負周波数応答のRMS level
- `fractional_response`: 小数遅延固定整相の理論応答
- `evaluation_arrays`: BL/FRAZ/BTR表示元の固定shape値
- `bl_component_metrics`、`abf_like_metrics`: target/noise/mixedとnon-source評価
- `evaluation_criteria`: 評価patternと必須・推奨metric
- `diagnostic_plotting`: 軸・referenceを明示するBL/FRAZ/BTR描画

Matplotlibを使う小さい描画関数とfigure保存は評価責務に含めるが、parameter sweep、
scenario実行、artifact pack一式の生成は含めない。
評価metricを自動採否へ使う場合は、人間の視覚順位との一致率を別途検証してから追加する。

### 7.3 `spflow.array_design`

`BandwiseArrayDesign`、運用array定義、shading定義、片舷array設計を公開する。
旧`spflow.beamforming.array_design`、`operational_sparse_array`、
`operational_shading`、`sparse_single_side_array_design`は互換facadeとし、実装を持たない。

### 7.4 `spflow.sidelobe_cancellation`

`BeamDomainSLC`とsource-mask SLCを公開する。入力はbeamforming済みの
`[n_beam, n_sample]`であり、sensor信号からbeamを生成する処理は持たない。
旧`spflow.beamforming.slc`と`source_mask_slc`は互換facadeとして残す。

## 8. 互換性

- 移動したscenario内では`TimeDelayDiagnosticSource`のconstructorとfieldを維持する。
- 既存summary key、JSON/CSV/PNG名、BL/FRAZ/BTR軸、乱数seedを変更しない。
- tone sceneの既定精度は従来と同じ`float32`とする。
- `noise_level_db20`は従来どおり時間領域sample RMSの`dB re input RMS`であり、
  one-sided ASDではない。
- equal-cos方位gridを維持し、線形角度gridへ変更しない。

## 9. 検証

抽出後は次を確認する。

1. tone RMS 0 dB入力が0 dB re input RMSになる。
2. 実toneの片側周波数応答だけが1の場合、正負合成levelは約-3.0103 dBになる。
3. one-sided spectrumの整数bin toneが入力RMSと一致する。
4. BTR block RMSのshapeが`[n_time, n_beam]`になる。
5. equal-cos gridの方向余弦shapeが`[n_beam, 3]`になる。
6. 小数遅延応答行列のaxisが`[n_observation_beam, n_look_beam]`になる。
7. tone sceneの`SimulationPrecision.SINGLE/DOUBLE`が出力dtypeへ伝播する。
8. 既存の整数遅延、小数遅延、SLC、時間領域適応比較testが同じ判定を維持する。

## 10. 後続の整理順序

1. ~~private helperを跨ぐscenario workflowを`evaluations/beamforming/scenarios`へ移す。~~ 完了。
2. ~~SLC diagnostics、performance、guard設計をcore公開APIから外す。~~ 完了。
3. `array_design`内に残るplot/report保存を小さい設計結果と`tools/array_design`へ分ける。
4. `spflow.beamforming_evaluation`の再公開範囲を見直し、core flat APIから評価名を外す。
5. `fixed_delay_diff_mvdr.py`を共分散積分、差分補正FIR、streaming合流の責務で分割する。

配布package内で完結する移動だけ、既存import互換を薄いファサードで残してよい。
repo内の`evaluations/`への委譲はインストール環境で成立しないため、互換facadeを作らない。

## 11. 逐次処理部品との接続とimport境界

beamformerは独自PipelineやProcessor基底クラスを要求せず、通常の`process`メソッドとして
NumPy配列を受け取る。`FrameBuffer.process`が返す0個・1個・複数個の完成frameは、
`Flow.map`のlist 1段展開規約により、そのまま`CBFBeamformer.process`へ接続できる。
overlap-save beamformerの`(band_index, valid_block)`はtupleを一つの意味値として保持し、
帯域metadataを失わず後段へ渡す。

`Flow`は値の運搬だけを担い、入力終端やflush順序を決めない。終端時は通常のPython制御で
`FrameBuffer.flush(pad=True)`を呼び、返された完成frameを`Flow.many`から同じbeamformerへ
合流させる。これにより、末尾端数を不完全状態として公開せず、巨大な実行モデルも導入しない。
決定論的な接続例は`examples/beamforming/streaming_cbf.py`に置く。

トップレベル`spflow`と互換`spflow.beamforming`のflat APIは遅延解決する。これにより、
`Flow`と`FrameBuffer`だけを使うprocessはbeamforming、filterbank、Matplotlibを読み込まず、
`CBFBeamformer`を使うprocessもCBFとその数式依存だけを読み込む。配布package内で移動した
互換公開名は維持するが、評価scenarioの公開名は除去し、新規コードでは責務を示すpackageまたは
repo内entrypointからのimportを使用する。
