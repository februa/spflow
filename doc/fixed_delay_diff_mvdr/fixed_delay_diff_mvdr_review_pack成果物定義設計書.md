# fixed_delay_diff_mvdr review_pack 成果物定義設計書

## 1. 目的

この設計書は、`artifacts/beamforming/fixed_delay_diff_mvdr/review_pack/` に出力する
report 成果物の意味を固定するためのものである。

fixed delay + difference MVDR の評価では、同じ BL / FRAZ / BTR という名前でも、
周波数断面、正規化基準、評価目的が異なると読み方が変わる。
そのため、各成果物について以下を明示する。

1. どの配列から作るか。
2. axis の意味と単位。
3. dB 表示の基準。
4. どの評価に使うか。
5. どの評価に使ってはいけないか。

Beamforming Evaluation の基準に従い、source-preserving scan では target と interferer を
観測 source として扱い、interferer を落とすことを合格条件にしない。
local leakage canceller として見る場合だけ、protected target beam への interferer leakage を評価する。

## 2. 共通定義

### 2.1 scenario

scenario は target と 0 個以上の interferer で構成する。
各 source は次を持つ。

```text
label:        source 名
azimuth:     方位 [deg]
frequency:   周波数 [Hz]
level:       入力 RMS 基準の相対レベル [dB re input RMS]
phase:       初期位相 [deg]
```

### 2.2 method

review_pack では次の method を同じ図・同じ CSV で比較する。

| method | 表示名 | 定義 |
|---|---|---|
| `fixed_baseline` | fixed | 事前計算小数遅延 FIR による固定整相。fallback として常に残す。 |
| `mvdr_oracle` | MVDR oracle | 周波数領域で直接計算した MVDR 重み。FIR 近似誤差を含めない参照。 |
| `diff_mvdr_fir512` | diff MVDR FIR512 | fixed に差分補正 FIR512 を加えた実装対象方式。 |

### 2.3 mask

`source_mask` は source 方位の周辺を表す boolean 配列である。
現在の review_pack では source 方位 ±3 deg を source mask とする。
`non_source_mask` はその補集合である。

```text
source_mask shape:     [n_beam]
non_source_mask shape: [n_beam]
axis=0:                beam 方位
unit:                  boolean
```

source-preserving scan では source mask 内の既知 source peak を false peak として扱わない。

### 2.4 dB 基準

BL と FRAZ の絶対レベルのように表示する値は `dB re input RMS` とする。
これは simulation の入力 RMS を基準にした相対 RMS レベルであり、物理音圧の絶対 dB ではない。

BL delta と FRAZ delta は fixed に対する差分である。
BTR は frame ごとに最大値を 0 dB に正規化するため、`dB re frame max` とする。

## 3. review_index.md

`review_index.md` は report の索引である。
各 scenario について、目的、source 条件、mask 種別、method ごとの判定要約、参照すべき図と npz を列挙する。

このファイルは数値判定の一次データではない。
数値判定には `scenario_summary.csv` と `worst_cases.csv` を使う。
図の読み方に迷う場合は、本設計書の図定義を優先する。

## 4. scenario_summary.csv

`scenario_summary.csv` は scenario × method の定量評価表である。

主な列の意味は以下である。

| 列 | 意味 |
|---|---|
| `source_peak_delta_db` | target 周波数 BL における source mask 内 peak の fixed との差分。 |
| `source_azimuth_error_deg` | target 周波数 BL の peak 方位と target 方位の差。 |
| `non_source_*_delta_db` | non-source mask 内の level 変化。known source 主ローブは含めない。 |
| `source_to_non_source_margin_delta_db` | source peak と non-source peak の margin 変化。 |
| `false_peak_count_delta` | source peak から指定 margin 以内に入る non-source peak 数の変化。 |
| `interferer_leakage_reduction_db` | protected target beam に入る interferer-only 成分の低減量。 |
| `target_mainlobe_delta_db` | target-only 成分の protected target beam level 変化。 |
| `mixed_target_beam_delta_db` | mixed 成分を target beam で見た level 変化。 |
| `q_reconstruction_rms_error` | 差分 FIR が周波数領域差分重みを再現する RMS 誤差。 |
| `loaded_condition_number_max` | MVDR 共分散行列の loading 後 condition number 最大値。 |

`source_peak_delta_db` は target 周波数断面の BL から計算する。
near-frequency interferer の visibility 判定には、これだけを使わない。

## 5. worst_cases.csv

`worst_cases.csv` は review 用の抽出表である。
各 metric の worst top 10、fallback 行、negative/watch 行、MVDR oracle と diff MVDR FIR512 の差が大きい行を集める。

この CSV は採否を自動決定するものではなく、レビュー優先順位を決めるための補助成果物である。

## 6. 図定義

### 6.1 bl_overlay.png

`bl_overlay.png` は target 周波数の BL 断面である。

```text
input:  fraz_levels_db[:, target_frequency_index]
shape:  [n_beam]
x-axis: azimuth [deg]
y-axis: RMS level [dB re input RMS]
```

用途:

- protected target 周波数での主ローブ保存を確認する。
- target beam 近傍の局所的な level 変化を確認する。
- same-frequency source の mixed 方位応答を見る。

使ってはいけない用途:

- target と interferer の周波数が異なる scenario で、interferer visibility を単独判定すること。
- BTR の track continuity を判定すること。

near-frequency interferer を target から 0.1 Hz ずらした場合、target 周波数 1536.0 Hz の BL には
1536.1 Hz の interferer peak が表示されない。これは抑圧ではなく、図の周波数断面定義による。

### 6.2 source_frequency_bl_overlay.png

`source_frequency_bl_overlay.png` は全 scenario で必ず生成する。
source-preserving scan の visibility 確認では、この図を優先して参照する。

```text
input:  fraz_levels_db[:, source_frequency_indices]
shape before reduction: [n_beam, n_source_frequency]
reduction: max over source_frequency axis
shape after reduction:  [n_beam]
x-axis: azimuth [deg]
y-axis: RMS level [dB re input RMS]
```

数式としては、method `m` の source-frequency BL を次で定義する。

```text
L_m(theta) = max_{f in F_source} FRAZ_m(theta, f)
F_source = {f_target, f_interferer_1, ...}
```

用途:

- target と interferer がそれぞれ自分の周波数で peak として残るかを確認する。
- near-frequency / different-frequency scenario の source visibility を 1 枚で確認する。
- `bl_overlay.png` の target 周波数断面だけでは見えない interferer peak を確認する。

使ってはいけない用途:

- 周波数方向の抑圧量を厳密に積分評価すること。
- どの周波数で peak が出たかを特定すること。周波数位置は FRAZ を見る。
- BTR の時間連続性を判定すること。

この図は source 真値周波数だけを使うため、広帯域ノイズ床や未知周波数の false peak 評価には使わない。

### 6.3 bl_delta.png

`bl_delta.png` は target 周波数 BL の fixed 差分である。

```text
MVDR oracle delta(theta)      = BL_mvdr_oracle(theta) - BL_fixed(theta)
diff MVDR FIR512 delta(theta) = BL_diff_mvdr_fir512(theta) - BL_fixed(theta)
unit: dB re fixed BL level
```

用途:

- target 周波数断面で、method が fixed に対して局所的に悪化していないかを見る。
- protected target 周波数の mainlobe preservation と local worsening を目視確認する。

使ってはいけない用途:

- interferer が別周波数の場合の source visibility 判定。
- BTR の時間方向連続性判定。

### 6.4 fraz_delta.png

`fraz_delta.png` は frequency-azimuth plane の fixed 差分である。

```text
input:  FRAZ_method(theta, f) - FRAZ_fixed(theta, f)
shape:  [n_beam, n_freq]
x-axis: azimuth [deg]
y-axis: frequency [Hz]
unit:   dB re fixed FRAZ level
```

用途:

- target と interferer の frequency ridge が保たれているかを見る。
- 周波数ごとの局所悪化、null、FIR 近似誤差を確認する。
- source-frequency BL overlay で見えた peak がどの周波数に属するか確認する。

使ってはいけない用途:

- BTR の時間方向 track continuity 判定。
- dB re frame max と混同した抑圧量判定。

### 6.5 btr_panel.png

`btr_panel.png` は method ごとの BTR を同じ color scale で並べた図である。

```text
input:  beam-time response after frame-wise normalization
shape:  [n_time, n_beam]
x-axis: azimuth [deg]
y-axis: time [s]
unit:   dB re frame max
```

用途:

- source track が時間方向に途切れないかを見る。
- fixed / MVDR oracle / diff MVDR FIR512 の track continuity を比較する。

使ってはいけない用途:

- 抑圧量の定量比較。
- source level の絶対比較。

BTR は frame ごとに最大値を 0 dB に正規化するため、method 間の絶対レベル差は保持しない。

## 7. 元データ npz 定義

`data/<scenario>.npz` は各図の描画前配列を保存する。

共通 axis:

```text
azimuth_deg:  [n_beam], deg
frequency_hz: [n_freq], Hz
time_sec:     [n_time], s
source_mask:  [n_beam], bool
non_source_mask: [n_beam], bool
```

BL 系配列:

```text
fixed_level_db:                       [n_beam]
mvdr_oracle_level_db:                 [n_beam]
diff_mvdr_fir512_level_db:            [n_beam]
fixed_source_frequency_level_db:      [n_beam]
mvdr_oracle_source_frequency_level_db:[n_beam]
diff_mvdr_fir512_source_frequency_level_db: [n_beam]
```

FRAZ 系配列:

```text
fixed_fraz_level_db:            [n_beam, n_freq]
mvdr_oracle_fraz_level_db:      [n_beam, n_freq]
diff_mvdr_fir512_fraz_level_db: [n_beam, n_freq]
```

BTR 系配列:

```text
fixed_btr_level_db:            [n_time, n_beam]
mvdr_oracle_btr_level_db:      [n_time, n_beam]
diff_mvdr_fir512_btr_level_db: [n_time, n_beam]
```

`*_level_db` は `dB re input RMS`、`*_btr_level_db` は `dB re frame max` である。

## 8. frequency_offset_sweep 成果物

`artifacts/beamforming/fixed_delay_diff_mvdr/frequency_offset_sweep/` は、
target と interferer の周波数差を増やしたとき、どこから分解可能とみなすかを調べる成果物である。

```text
frequency_offset_sweep.csv: 掃引結果の一次データ
frequency_offset_sweep.md:  掃引条件と結論の要約
frequency_offset_sweep.png: offset に対する visibility / leakage reduction の図
```

現在の判定では、解析周波数軸へ source 周波数を明示した場合は 0.1 Hz でも別周波数として扱える。
ただし 1 秒観測の STFT bin 幅を 1 Hz とするため、1 秒観測基準での分解可能条件は offset 1 Hz 以上とする。

## 9. 採否時の読み方

採否は 1 枚の図だけで決めない。

- source visibility は `source_frequency_bl_overlay.png` と FRAZ で確認する。
- protected target beam の保存は `bl_overlay.png`、`bl_delta.png`、`scenario_summary.csv` で確認する。
- interferer leakage reduction は `scenario_summary.csv` の interferer-only metric で確認する。
- track continuity は `btr_panel.png` で確認する。
- BTR は `dB re frame max` なので、抑圧量の定量比較には使わない。
- fixed_baseline は常に fallback として残す。
