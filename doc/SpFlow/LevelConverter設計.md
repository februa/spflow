# LevelConverter設計

## 1. 目的

入力信号を生成する地点と信号処理結果を評価する地点が離れていても、dB reference、
10log10/20log10、RMS/power、one-sided/two-sided、FFT正規化の不整合を早期検出する。

公開の中心は`LevelConverter`とする。利用者は数式一覧からfactoryを選び、
immutableな入力definitionと出力definitionをConverterへ渡す。`Contract`という名称や
任意lambda、数式DSLを利用者へ要求しない。

## 2. 責務境界

`LevelConverter`が担うものは次である。

- 入力dB definitionから入力側線形量への変換
- 測定済み出力線形量から出力dB definitionへの変換
- 物理量、reference、積分量/密度、sidednessの接続検証
- 図、CSV、JSONへ記録するreference labelと採用数式の公開

次は責務に含めない。

- 波形からのRMS測定
- FFTとFFT長による正規化
- one-sided bin powerへの変換
- ASD/PSDの帯域積分
- tone、noise、狭帯域、広帯域という信号分類

これらは`spflow.spectral_level`などの小関数で処理し、測定済み線形量だけを
Converterへ渡す。狭帯域と広帯域は型を分けず、one-sided bin RMS powerを
明示された占有/評価帯域で積分する同じ規約を使う。

## 3. 共通比較量

入出力definitionは、原則として次の無次元量へ接続する。

```text
normalized mean-square ratio
    = 評価対象の平均二乗値 / referenceの平均二乗値
```

RMSとpower、ASDとPSDは表面式が異なっても、それぞれ同じ積分済みmean-square、
またはmean-square densityへ接続できる。密度と積分量の間には帯域積分が必要なので、
Converter生成時に暗黙接続せず`ValueError`とする。

## 4. 登録済みdefinition

factory名は外側から内側へ`level_<logarithmic_mapping>_<linear_quantity_definition>`と読む。
suffix全体をドキュメントへ登録した一つのlinear quantity definition名として扱う。

| factory | 線形入力/観測値 | 数式 | 境界条件 |
| --- | --- | --- | --- |
| `level_20log10_rms` | RMS amplitude | `L=20log10(A_rms/A_ref)` | peakや瞬時値ではない |
| `level_10log10_power` | 積分済みpower | `L=10log10(P/P_ref)` | PSDを直接渡さない |
| `level_10log10_conjpair_power` | 正規化済み内部正周波数係数`z` | `L=10log10(2abs(z)^2/A_ref^2)` | DC、Nyquist、解析信号、complex basebandには使わない |
| `level_20log10_onesided_asd` | one-sided ASD | `L=20log10(ASD_1/ASD_ref)` | 単位は振幅/√Hz、積分幅を別途指定 |
| `level_20log10_twosided_asd` | two-sided ASD | `L=20log10(ASD_2/ASD_ref)` | sidednessを省略しない |
| `level_10log10_onesided_psd` | one-sided PSD | `L=10log10(PSD_1/PSD_ref)` | 単位は振幅²/Hz、積分幅を別途指定 |
| `level_10log10_twosided_psd` | two-sided PSD | `L=10log10(PSD_2/PSD_ref)` | sidednessを省略しない |

`rfftbin`と`band`はlevel definitionではない。`one_sided_rfft_bin_rms_power`が
非正規化rFFTをFFT長`N`で正規化し、`integrate_one_sided_band_rms_power`が
選択bandを積分するため、重複factoryを作らない。

## 5. スペクトル規約

実信号の内部周波数では、正負周波数が共役対を構成する。

```text
conjpair power = abs(z)^2 + abs(conj(z))^2 = 2abs(z)^2
onesided PSD = 2 * twosided PSD
onesided ASD = sqrt(2) * twosided ASD
```

DCと偶数長FFTのNyquistは共役相手を別binに持たないため係数2を適用しない。
複素解析信号とcomplex basebandにもこの係数を自動適用しない。

非正規化rFFT`X[k]`のone-sided per-bin RMS powerは次とする。

```text
DC, Nyquist: abs(X[k])^2 / N^2
内部bin:     2abs(X[k])^2 / N^2
```

全bin和は時間領域mean-squareと一致する。

## 6. ASD/PSDの帯域積分

ASDは広帯域信号の別名ではなく、振幅スペクトル密度である。一定ASDなら、明示した
積分帯域幅`B`に対して次となる。

```text
A_band = ASD * sqrt(B)
```

一般には次を使う。

```text
A_band = sqrt(integral_B ASD(f)^2 df)
P_band = integral_B PSD(f) df
```

帯域全体では占有/評価帯域を、FFT 1 binでは`delta_f=fs/N`を`B`として指定する。
bin数とHzを混同しない。

## 7. 利用例

```python
from spflow import (
    LevelConverter,
    level_10log10_conjpair_power,
    level_20log10_rms,
)

input_definition = level_20log10_rms(
    reference_rms=1.0,
    reference_label="input RMS",
)
output_definition = level_10log10_conjpair_power(
    reference_rms=1.0,
    reference_label="input RMS",
)
converter = LevelConverter(
    input_definition=input_definition,
    output_definition=output_definition,
)

tone_peak = converter.input_to_real_cosine_peak(-6.0)
result_level_db = converter.output_to_level(normalized_positive_frequency_coefficient)
```

`ToneScene`へ同じConverterを渡し、出力評価でも保持したConverterを使えば、
入力地点と出力地点のreferenceと数式を一つの値として引き回せる。

## 8. 互換性

`level_db_to_rms_amplitude`、`tone_rms_level_db_to_peak_amplitude`、
`noise_asd_level_db_to_band_rms`、`rms_amplitude_to_level_db`は互換ファサードとして残す。
実装式は`LevelConverter`へ委譲し、既存利用者へ新しい実行モデルを強制しない。
