# examples 部品化設計

## 1. 目的

`examples/` に蓄積した方式検討コードから共通責務を抽出し、再利用可能な部品と、個別シナリオに残すべき処理の境界を定める。

本設計の目的は、example 群を一つの実行フレームワークへ置き換えることではない。通常の Python 関数と小さなデータクラスを組み合わせる方針を保ちつつ、信号処理式、評価定義、成果物仕様がファイルごとに再実装される状態を解消する。

## 2. 現状認識

2026-07-11 時点の `examples/` には、core、beamforming、filterbank、nonuniform を合わせて約 22,700 行の Python コードがある。特に `examples/beamforming/` には、次の異なる種類の処理が同じ階層に置かれている。

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

### 3.3 コアライブラリと評価支援を区別する

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

配置先は `src/spflow/evaluation/beamforming/` を第一候補とする。この package は信号処理コアとは依存方向を分け、matplotlib や scene renderer を必須依存にしない。

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

外部 package の型や座標規約をコアの `ScenarioSpec` と `BeamformingInput` へ変換する責務だけを持つ。`sys.path` の変更、source level 変換、座標変換を各 example へ再実装しない。

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
examples/core/frame_buffer.py
examples/core/flow_zero_one_many.py
examples/core/scheduled_update.py
examples/beamforming/fixed_delay_and_sum.py
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
      optional adapters          NumPy とコア部品
```

- `spflow.beamforming` は `spflow.evaluation` に依存しない。
- metric は reporting に依存しない。
- reporting は scenario の生成や method の実行を行わない。
- external adapter は optional dependency の未導入時にコア import を失敗させない。
- examples 同士、evaluation script 同士を import しない。

## 8. 移行手順

### Phase 1: 数式の一元化

level、spectrum power、direction、steering の共通関数を追加し、既存 helper との同値テストを作る。呼び出し側を一つずつ置換し、全評価の数値が変化しないことを確認する。

### Phase 2: 評価データモデル

`ScenarioSpec`、`MethodOutput`、`EvaluationResult` を導入する。既存関数の戻り値を巨大な dict や shape 不明の tuple から固定 result 型へ移す。

### Phase 3: metric と成果物

BL/FRAZ/BTR 定義、mask、worst-case、CSV/NPZ、標準 figure を統合する。Beamforming Evaluation の pattern ごとの required criteria が report に含まれることを検証する。

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
