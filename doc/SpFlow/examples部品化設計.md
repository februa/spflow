# examples 部品化設計

## 1. 目的

`examples/` に蓄積した方式検討コードから共通責務を抽出し、再利用可能な部品と、個別シナリオに残すべき処理の境界を定める。

本設計の目的は、example 群を一つの実行フレームワークへ置き換えることではない。通常の Python 関数と小さなデータクラスを組み合わせる方針を保ちつつ、信号処理式、評価定義、成果物仕様がファイルごとに再実装される状態を解消する。

## 2. 現状認識

2026-07-11 時点の `examples/` には、streaming、beamforming、filterbank、nonuniform を合わせて約 22,700 行の Python コードがある。逐次処理の基本例は、その責務を名前から判断できるように `examples/streaming/` に置く。特に `examples/beamforming/` には、次の異なる種類の処理が同じ階層に置かれている。

- ライブラリの最小使用例
- アレイ、音源、雑音のシナリオ定義
- scene renderer を使った入力信号生成
- steering、共分散、重み、ビーム出力の計算
- 方式比較、parameter sweep、採否判定
- BL、FRAZ、BTR、spectrum の描画
- CSV、NPZ、Markdown、ZIP の成果物生成
- 運用係数ファイルの生成 CLI

同名または同じ責務を持つ処理の重複例は次の通りである。

| 共通責務 | 確認できた実装数 | 主な重複内容 |
|---|---:|---|
| tone level から peak amplitude への変換 | 9 | RMS と peak の変換式 |
| source 位置から到来方向への変換 | 5 | 座標差の正規化と方位規約 |
| 方向から steering を生成 | 5 | 相対遅延と周波数位相回転 |
| target scene の生成 | 5 | source、receiver、environment の組み立て |
| CSV 出力 | 9 | field 集約、非有限値の処理、列順 |
| NPZ 出力 | 6 | 描画元配列と metadata の保存 |
| review index 出力 | 6 | 成果物リンク、条件、判定結果の Markdown 化 |
| BL/FRAZ/BTR review pack 描画 | 3 系統以上 | mask span、軸、dB reference、caption |
| SLC sweep の設定値検証 | 3 | mapping、number、optional number の検証 |

問題は行数そのものではない。信号処理上の同じ定義が複数ファイルへ複製され、修正時に一部だけが更新される危険があること、および example が「spflow の使い方」ではなく「評価基盤の実装場所」になっていることが問題である。

## 3. 部品化の原則

### 3.1 変わる理由が同じ処理だけをまとめる

再利用できそうに見える処理をすべて一つへ集約してはいけない。次の変更理由は分離する。

- 音響・信号処理式が変わる
- 外部 renderer やファイル形式が変わる
- 評価指標や合否条件が変わる
- 描画・成果物仕様が変わる
- 個別実験の条件が変わる

たとえば、tone の RMS level 変換は信号処理式として共通化できるが、source の配置はシナリオ固有である。BL 配列の定義は共通化できるが、どの source を表示するかはシナリオ固有である。

### 3.2 継承階層ではなく、値と関数の合成を使う

`BaseEvaluation`、`BaseScenario`、`BaseReportBuilder` のような基底クラスを中心にしない。設定と結果は固定 shape の dataclass、計算は入力と出力が明確な関数で表す。

```text
ScenarioSpec
  -> render_scene(...)
  -> BeamformingInput
  -> run_method(...)
  -> MethodOutput
  -> evaluate_pattern(...)
  -> EvaluationResult
  -> write_report_pack(...)
```

この流れを実行する巨大な Runtime は作らない。個別スクリプトの `main()` が通常の Python として必要な関数を明示的に呼ぶ。

### 3.3 実運用の信号処理と評価支援を区別する

部品化した処理をすべて `spflow.beamforming` の公開 API に入れてはいけない。

- 実運用の信号処理でも使う数式・処理は `src/spflow/` に置く。
- 複数方式の評価で共通するが、実運用には不要な処理は評価支援 package に置く。
- 単一の検討だけに意味がある条件と判定は個別 evaluation に残す。

## 4. 推奨カテゴリ

### 4.1 信号処理プリミティブ

配置先は `src/spflow/signal/` または既存の責務に合う `src/spflow/beamforming/` とする。

対象は、外部 renderer、plot、CSV に依存しない純粋計算である。

```text
level.py
    tone_rms_level_db_to_peak_amplitude
    noise_asd_level_db_to_sample_rms
    one_sided_rfft_bin_rms_power
    integrate_one_sided_band_rms_power
    linear_amplitude_to_db

geometry.py
    direction_from_positions
    direction_from_azimuth_elevation
    relative_arrival_delay

steering.py
    steering_from_delay
    steering_from_positions_and_direction
```

条件は、数式、shape、axis、単位、DC/Nyquist の境界処理を一意に定義できることである。これらは example 内の private helper として残すべきではない。

### 4.2 ストリーミング実行部品

配置先は既存の `src/spflow/frequency/`、`src/spflow/filterbank/`、`src/spflow/beamforming/` とする。

対象は、chunk 分割、overlap-save、出力時刻整列、ブロック連結など、方式に依存しない実行処理である。

```text
streaming.py
    process_chunks
    concatenate_timestamped_blocks
    crop_valid_interval
    calculate_streaming_boundary_error
```

単なる `for` loop を隠すだけの wrapper は作らない。サンプル時刻、遅延補償、有効区間、flush 規約を一つに定める場合だけ部品化する。

### 4.3 評価モデルと metric

配置先は `src/spflow/evaluation/beamforming/` を第一候補とする。この package は実運用の信号処理 package とは依存方向を分け、matplotlib や scene renderer を必須依存にしない。

```text
model.py
    SourceSpec
    ScenarioSpec
    MethodOutput
    EvaluationResult
    EvaluationPattern

spectral.py
    spectrum_level
    source_frequency_bl
    band_integrated_level

spatial.py
    source_mask
    mainlobe_mask
    peak_azimuth_error
    sidelobe_peak_margin
    false_peak_count

temporal.py
    btr_relative_level
    waveform_integrity

slc.py
    target_leakage_components
    covariance_health
    fallback_status
```

`EvaluationPattern` は Beamforming Evaluation の pattern ID を保持する。pattern ごとの required/recommended criteria は既存の `get_evaluation_criteria_for_pattern()` を唯一の選択元とし、個別スクリプトで再定義しない。

### 4.4 外部入力 adapter

配置先は `src/spflow/evaluation/adapters/` または独立した optional package とする。

```text
scene_renderer.py
    SceneRendererAdapter
    render_scenario
    verify_rendered_input_levels

matlab_raw.py
    load_float32_le
    load_array_positions
    load_complex_shading
    load_fractional_delay_bank
```

外部 package の型や座標規約を spflow 側の `ScenarioSpec` と `BeamformingInput` へ変換する責務だけを持つ。`sys.path` の変更、source level 変換、座標変換を各 example へ再実装しない。

### 4.5 成果物生成

配置先は `src/spflow/evaluation/reporting/` とする。ただし matplotlib は optional dependency とする。

```text
table.py
    write_csv_rows
    sanitize_scalar
    build_worst_case_rows

data.py
    write_plot_data_npz
    write_metadata_json

plot.py
    plot_bl_overlay
    plot_source_frequency_bl_overlay
    plot_bl_delta
    plot_fraz_delta
    plot_btr_panel

review_pack.py
    ReportArtifactDefinition
    write_review_index
    write_review_pack
```

描画関数は数値 metric を再計算しない。`EvaluationResult` と描画用配列を受け取り、軸、単位、dB reference、equal-cos cell edge を一貫して適用する。

### 4.6 個別 evaluation

配置先は `evaluations/beamforming/`、`evaluations/filterbank/`、`evaluations/nonuniform/` とする。

個別ファイルに残す責務は次に限定する。

- その検討で必要な scenario 一覧
- 比較する method と parameter grid
- 採否判定の閾値
- 共通部品を呼ぶ処理順序
- CLI argument と出力先

個別 evaluation は、原則として信号処理式、汎用 metric、BL/FRAZ/BTR の描画実装、CSV serializer を持たない。

### 4.7 examples

`examples/` には、公開 API の使い方を説明する短い実行例だけを置く。

```text
examples/streaming/frame_buffer.py
examples/streaming/flow_zero_one_many.py
examples/streaming/scheduled_update.py
examples/beamforming/delay_and_sum.py
examples/beamforming/fractional_delay_streaming.py
examples/beamforming/slc_safe_fallback.py
examples/filterbank/prdft_analysis_synthesis.py
```

各 example は次を満たす。

- 一つの設計概念だけを扱う。
- 目安として 200 行以内とする。超える場合は、ライブラリ側に不足する部品がないか確認する。
- artifact pack、parameter sweep、採否判定を生成しない。
- 外部 renderer がなくても実行できる小さな決定論的入力を使う。
- 通常の Python で書いた場合との対応が読み取れる。
- example 独自の汎用 helper を増やさない。

### 4.8 tools

運用係数や設定ファイルを生成する CLI は `tools/` に置く。

```text
tools/design_fractional_delay_filter_bank.py
tools/design_operational_shading.py
tools/design_operational_sparse_array.py
tools/generate_nonuniform_array_inputs.py
```

CLI は引数解釈とファイル入出力だけを担当し、設計計算は `src/spflow/` の公開関数を呼ぶ。

## 5. 最初に統合する候補

### 5.1 優先度 A: 定義が一意で重複が多いもの

1. level と one-sided spectrum の変換
2. direction、delay、steering の生成
3. BL、source-frequency BL、FRAZ、BTR の配列定義
4. scene renderer adapter
5. review pack の標準成果物と描画

これらは現在の評価結果の意味へ直接影響する。ファイルごとの差異を残すと、同じ `dB` 表示でも基準や FFT 正規化が異なる危険があるため、最優先で一元化する。

### 5.2 優先度 B: 同じ構造を持つ sweep と report

1. sweep parameter と result row の表現
2. worst-case 抽出
3. metadata と再現条件の保存
4. CSV、NPZ、Markdown の出力
5. runtime measurement

ここでは汎用 sweep engine を作らない。個別ファイルの明示的な loop は残し、結果の表現と保存だけを共通化する。

### 5.3 優先度 C: filterbank と nonuniform の評価共通化

1. 決定論的 test signal
2. chunk 分割と streaming/offline 同値比較
3. PR error、境界 jump、peak response の metric
4. candidate table と採点結果の保存

beamforming の評価モデルを無理に流用せず、共通する成果物層だけを共有する。

### 5.4 BL 評価方法は視覚評価との対応を定量化してから確定する

BL の配列生成、軸、単位、dB reference、描画条件は共通化できる。BL の parameter sweep と候補選別には再現可能な数値指標が必要であるため、最終的には人間の視覚評価を十分な精度で代替できる数値指標を確立する。一方、視覚評価との対応を検証していない score や固定閾値は、現時点では共通部品の確定仕様にしない。

peak、guard 外最大値、percentile、integrated level などの数値は、同じ BL 図を人間が見たときに感じる mainlobe の明瞭さ、sidelobe の目立ち方、source 間の分離性、局所的な悪化と順位が一致しない可能性がある。特に、狭い高ピークと広い低隆起、source 近傍と遠方 sidelobe、表示 dynamic range の違いは、単一指標では同じ感覚を表せない。

指標設計段階では、次の二系統を並行して記録する。

```text
数値観測:
    peak position、peak width、guard 外 peak、局所 peak、percentile、
    integrated level、source 間 valley、最大局所悪化

視覚観測:
    mainlobe の識別しやすさ、source 分離、不要ピークの目立ち方、
    裾の広がり、左右非対称、局所的な不自然さ
```

視覚比較では、方式間で azimuth 軸、y 軸範囲、dB reference、dynamic range、線幅、source marker、mask 表示を固定する。表示条件が異なる図の印象を比較してはいけない。

数値順位と視覚順位が不一致の場合は、次を残す。

- 比較した scenario と method
- 同一条件で描画した BL 配列と PNG
- 各数値指標の値と順位
- 視覚評価の観点別順位と判断理由
- 不一致を生んだと推定する形状特徴

対応事例を教師データとして、人間の視覚判断を説明できる指標の組み合わせ、重み、非線形な集約、または Pareto 判定を設計する。最低限、次を検証する。

- 視覚順位に対する Spearman 順位相関
- 二方式のどちらが良いかという対比較一致率
- 視覚的な合否ラベルに対する precision、recall、見逃し率
- 指標設計に使っていない周波数、source 配置、SNR、方式での性能
- reviewer が複数いる場合の reviewer 間一致度と、指標対 reviewer の一致度

要求精度を満たした数値指標は、parameter sweep の目的関数、候補の絞り込み、自動合否判定に使用してよい。満たさない間は数値を観測値として保存し、視覚評価を併用する。人間と一致しない事例は、指標が捉えていない形状特徴を特定する反例として残す。

## 6. 統合してはいけないもの

- 方式ごとに異なる covariance の推定規約
- fixed beam、MVDR、SLC の method 固有状態
- source-preserving scan と local leakage canceller の合否条件
- sparse array と shading の設計制約
- filterbank の PR 条件と beamforming の空間 metric
- 個別検討でのみ意味を持つ threshold
- 実験の意図を表す scenario 名、source 配置、parameter grid

これらを設定値だけで切り替える巨大関数へまとめると、型、成立条件、評価の意味が不明瞭になる。共通の入力・出力モデルを使いながら、方式固有関数として分ける。

## 7. 依存方向

依存方向は次に固定する。

```text
examples / evaluations / tools
            |
            v
spflow.evaluation  ->  spflow.beamforming / filterbank / frequency
            |                         |
            v                         v
      optional adapters          NumPy と信号処理部品
```

- `spflow.beamforming` は `spflow.evaluation` に依存しない。
- metric は reporting に依存しない。
- reporting は scenario の生成や method の実行を行わない。
- external adapter は optional dependency の未導入時に spflow の基本 import を失敗させない。
- examples 同士、evaluation script 同士を import しない。

## 8. 移行手順

### Phase 1: 数式の一元化

level、spectrum power、direction、steering の共通関数を追加し、既存 helper との同値テストを作る。呼び出し側を一つずつ置換し、全評価の数値が変化しないことを確認する。

### Phase 2: 評価データモデル

`ScenarioSpec`、`MethodOutput`、`EvaluationResult` を導入する。既存関数の戻り値を巨大な dict や shape 不明の tuple から固定 result 型へ移す。

### Phase 3: 観測 metric と成果物

BL/FRAZ/BTR の配列定義、mask、CSV/NPZ、標準 figure を統合する。BL の候補 metric と同一表示条件の視覚評価を対応付け、順位相関、対比較一致率、合否精度、未使用 scenario への汎化性能を評価する。妥当性を確認できた数値指標は sweep と自動判定へ使用し、未確認の指標は report の観測値として扱う。

### Phase 4: adapter

scene renderer と MATLAB raw 入力を共通 adapter へ移す。入力 level の spectrum 検証を adapter の境界テストとして固定する。

### Phase 5: ファイル再配置

方式評価を `evaluations/`、係数生成を `tools/` へ移し、`examples/` を短い公開 API 使用例だけにする。設計書、README、test の参照パスを同じ commit 群で更新する。

## 9. 検証方針

部品化は方式変更ではないため、移行前後で次を一致させる。

- time-domain output と spectrum の数値
- BL、source-frequency BL、FRAZ、BTR の元配列
- mainlobe、sidelobe、leakage、condition number の metric
- CSV の主要数値列
- fallback の発生条件と理由
- 入力 tone RMS level と noise ASD level の spectrum 検証値

plot の PNG binary 一致は要求しない。代わりに、描画元 NPZ、軸、shape、単位、dB reference、source marker の一致を確認する。

## 10. 結論

共通項の部品化は可能であり、特に level/spectrum、geometry/steering、評価 metric、scene renderer adapter、review pack は統合効果が大きい。

ただし、統合単位は「evaluation framework」ではなく、次の五つの独立層とする。

1. 信号処理プリミティブ
2. 外部入力 adapter
3. 評価モデルと metric
4. 成果物生成
5. 個別 scenario と method orchestration

この境界であれば、重複を除きながら、spflow の「普通の Python を主役とし、小さく直交する部品だけを提供する」という思想を維持できる。

## 11. 実装進捗

### 11.1 2026-07-11: 数式と表示配列の第一段階

方式比較、parameter sweep、採否判定を個別 evaluation に残し、それ以外のうち定義を一意にできる処理から部品化した。

```text
src/spflow/spectral_level.py
    tone RMS level -> peak amplitude
    noise ASD level -> sample RMS
    non-normalized rFFT -> one-sided bin RMS power
    band-integrated RMS power
    RMS amplitude -> referenced dB level

src/spflow/beamforming/geometry.py
    receiver/source position -> unit direction
    sensor position/direction -> relative arrival delay
    relative delay/frequency -> steering phase

src/spflow/beamforming/evaluation_arrays.py
    beam spectrum -> FRAZ
    FRAZ -> target-frequency BL
    FRAZ -> source-frequency BL
    beam-time RMS -> frame-max normalized BTR
    BL -> calibration candidate shape features
```

BL shape feature は観測値だけを返し、scalar score、閾値、status を持たない。方式比較、sweep、採否判定を共通部品へ混入させないためである。

BL と校正特徴量を生成する前に、入力信号の正しさを独立に確認する。信号生成確認と BL 評価を一つの example に混在させない。

`examples/beamforming/delay_and_sum.py` は、外部 renderer を使わず、単一平面波 tone の生成と整相加算だけを扱う。次を順に確認する。

1. 各 channel の tone RMS level が指定値と一致する。
2. source に近い sensor の到達遅延が負になり、FFT 位相が `-2πfτ` と一致する。
3. steering `exp(-j2πfτ)` を使った `w^H X` により channel が同相加算される。
4. distortionless な delay-and-sum 出力 level が入力 tone level と一致する。

BL、方式比較、parameter sweep、採否判定は、この信号生成確認に含めない。信号生成と整相規約の検証が通った後、別の校正入力生成処理から BL 群を作る。

SL と NL は次の振幅を区別する。

```text
SL の RMS amplitude:
    A_rms = 10^(SL/20)

実 cos 波へ渡す peak amplitude:
    A_peak = sqrt(2) * 10^(SL/20)

one-sided ASD として定義した NL の帯域 B [Hz] 内 RMS:
    A_noise_band_rms = 10^(NL/20) * sqrt(B)

FFT 長 N、sampling frequency fs の 1 bin 内 RMS:
    delta_f = fs/N
    A_noise_bin_rms = 10^(NL/20) * sqrt(delta_f)

DC〜Nyquist 全帯域の sample RMS:
    A_noise_sample_rms = 10^(NL/20) * sqrt(fs/2)
```

`sqrt(fs/2)/sqrt(M)` は、one-sided 全帯域を `M` 個の等帯域へ分けた 1 帯域分を表す。`M Hz resolution` を意味しない。分解能が 256 Hz なら `B=256 Hz` として `sqrt(256)` を掛ける。

既存の `scene_renderer_cbf_eval.py` についても、tone level 変換、位置から方向への変換、相対遅延、steering 位相を共通部品へ置き換えた。個別 example には scene の条件、実行順序、表示する診断だけを残す方向で段階的に移行する。

### 11.2 次の部品化対象

次の順序で移行を継続する。

1. 残る scene renderer example の level、geometry、steering 重複を置換する。
2. scene renderer 型と spflow の入力型の変換を adapter へ分離する。
3. CSV、NPZ、metadata、review index の serializer を成果物別に分離する。
4. BL/FRAZ/BTR plot が共通の表示条件オブジェクトを受け取るようにする。
5. `delay_and_sum.py` で信号生成規約を固定した後、別の校正入力生成処理から BL 群と人間評価入力を生成する。

## 12. BL指標校正の基準出力

### 12.1 単一source基準条件

`evaluations/beamforming/bl_baseline.py` により、信号生成を検証済みの平面波toneからBLを生成する。

```text
array: 8 channel uniform linear array
sensor spacing: 0.25 m
sound speed: 1500 m/s
source azimuth: 65 deg
source frequency: 1500 Hz
source level: 0 dB re input RMS
waiting beam axis: 0..180 deg, 1 deg step
source guard: ±10 deg
BL display range: -80..3 dB re input RMS
```

BLはsource条件を固定し、waiting beamごとのdelay-and-sum出力tone RMSを並べた`[n_beam]`配列である。beam patternのように入力source方位をsweepしたものではない。

### 12.2 現行特徴量

2026-07-11時点の出力は次である。

| 特徴量 | 値 | 定義上の注意 |
|---|---:|---|
| peak azimuth | 65.0 deg | source truthと一致 |
| peak azimuth error | 0.0 deg | waiting beam量子化を含む |
| peak level | 約0.0 dB re input RMS | distortionless応答 |
| -3 dB peak width | 28.0 deg | global peakを含む連結区間 |
| guard-outside peak | -1.595 dB re input RMS | ±10 deg guard外。mainlobe裾を含む |
| guard-outside p95 | -3.258 dB re input RMS | 方位sample percentile |
| guard-outside p99 | -1.911 dB re input RMS | 方位sample percentile |
| integrated guard-outside level | +11.025 dB re input RMS | guard外beam sample powerの無重み和 |
| source-to-guard peak margin | 1.595 dB | source内peakとguard外peakの差 |

### 12.3 現時点で判明した問題

- -3 dB幅が28 degであるのにguardが±10 degであるため、guard-outside peakはsidelobeではなくmainlobeの裾を測っている。
- 線状アレイ応答は方向余弦で決まるため、degree軸上ではmainlobeとnull間隔が左右非対称に見える。固定degree guardだけではmainlobe境界を正しく表せない。
- integrated guard-outside levelはbeam sampleのpowerを無重みで加算しており、beam本数と方位grid密度に依存する。物理的な方位積分でも人間の視覚量でもない。
- p95/p99もguard定義と方位samplingに依存するため、現時点では校正前の観測値である。

したがって、これらの値を方式の採否へ使わない。次段では、mainlobe境界をfirst-null、局所極小、方向余弦幅のどれで定義するかを整理し、同じBL図に対する人間評価との対応を測る。

### 12.4 視覚評価で確認する物理量

BL評価は、`target-only`、`noise-only`、`target+noise`を分ける。一つのmixed BLだけから、source由来の副極とnoise floorを分類しない。

#### Target-only BL

1. mainlobe peak方位とsource truthの誤差を確認する。
2. mainlobe peak levelと入力SLの誤差を確認する。distortionlessに正規化したCBF/ABFなら、指向性補正がない限り入力SLに近似する。
3. peakの左右にある最初の局所極小をfirst null候補とし、mainlobe境界を定義する。
4. mainlobe境界の外側で左右それぞれ最初の局所極大を第一副極候補とする。
5. 一様重み有限ULAのarray factorは厳密にはDirichlet型であり、多素子ではsincへ近似する。第一副極の約`-13.26 dB re mainlobe peak`をCBFのsanity referenceとする。
6. ABFでは第一副極低下だけでなく、他の副極、局所悪化、別方位へのenergy押し出しを確認する。
7. mainlobe外でmainlobeに近い大peakがある場合はgrating-lobe候補とする。

`-13 dB`はすべてのアレイへの固定合否閾値ではない。sensor数、shading、sparse/nonuniform配置、endfire付近、方位gridにより有限ULAの第一副極は変化するため、同じアレイ条件の理論array factorと比較する。

#### Noise-only BL

CBFにもABFにもnoise floorは存在する。ABFだけに現れる量として扱わない。

beam重み`w`とchannel noise covariance`R_n`に対し、出力noise powerは次で予測する。

```text
P_noise_out = w^H R_n w
```

channel間で無相関、等分散`σ_n^2`の白色雑音なら次になる。

```text
P_noise_out = σ_n^2 * sum_ch |w_ch|^2
```

矩形・distortionless正規化CBFでは`w_ch=1/N`であるため、target levelを保存しながらnoise powerは`1/N`になり、SNRは`10log10(N)` dB改善する。これは空間アレイゲインであり、FFT bin幅、帯域制限、時間平均による改善とは分けて扱う。

ABFではwaiting beamごとに`w^H R_n w`とnoise-only BLを比較し、CBF比の改善量とnoise増幅方位を記録する。

#### Target+noise BL

mixed BLは運用上のsource visibility確認に使う。targetとnoiseが無相関ならmixed powerがtarget-only powerとnoise-only powerの和で説明できるか確認する。target sidelobeをnoise floorと呼ばず、noiseの局所変動を決定論的な副極と呼ばない。

#### Grating lobe

ULAのgrating lobe発生は、主にsensor間隔と波長の比`d/λ`、steering方向、可視方向余弦範囲で決まる。開口長`D`は主にmainlobe幅とnull間隔を決める。候補peakが見つかった場合は、信号周波数から`λ=c/f`を求め、空間位相alias条件と候補方位が一致するか確認する。
