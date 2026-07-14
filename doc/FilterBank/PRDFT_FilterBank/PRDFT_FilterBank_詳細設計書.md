# PRDFT FilterBank 詳細設計書（Codex実装用）

## 1. 目的

SPFlowへ、PRDFTフィルタバンクを中心とした汎用フィルタバンク基盤を追加する。

本設計は **PRDFT専用実装ではなく、将来的な
Wavelet・非均一FilterBank・Polyphase FilterBank
の基盤**となることを目的とする。

設計思想は SPFlow に合わせ、

-   小さい部品
-   Flow による明示的な処理記述
-   過度なカプセル化を避ける

を採用する。

------------------------------------------------------------------------

# 2. 基本方針

追加する部品

    filterbank/
        polyphase.py
        modulation.py
        prdft.py
        delay.py

    filterbank/design/
        prototype.py
        response_resampler.py

    frequency/
        overlap_save.py
        fft.py
        ifft.py
        valid_region.py
        bin_selector.py

    beamforming/
        covariance.py
        mvdr_filter.py
        mvdr_weight_designer.py

追加しない部品

-   SubbandFlow
-   Bandwise
-   ProcessorPipeline
-   汎用 Convolver 階層

理由

SPFlow は Flow を中心としたライブラリであり、
処理を隠すラッパーは可読性を下げるため。

------------------------------------------------------------------------

# 3. PRDFT

Analysis

    real signal
        ↓
    Polyphase
        ↓
    DFT Modulation
        ↓
    Subband signal

出力

    shape

    [n_ch, n_band, n_subband_sample]

帯域順

    FFT順

    0 ... 15 | 16 ... 31

------------------------------------------------------------------------

# 4. 実信号対応

主処理対象

    正側16帯域

負側

    Pending

    共役写像＋PRDFT位相補正

初期実装では

-   正側のみ設計
-   負側生成方式は検証対象

とする。

------------------------------------------------------------------------

# 5. サブバンド処理

構成

    Subband time signal
            ↓
    FrameBuffer.push()
            ↓
    FFT
            ↓
    CovarianceTap
            ↓
    StepScheduler
            ↓
    MVDR Filter
            ↓
    IFFT
            ↓
    ValidRegionExtractor

Flow の中に FrameBuffer を入れない。

FrameBuffer は Flow の前で使用する。

------------------------------------------------------------------------

# 6. Overlap-save

標準仕様

    valid_size = 1024
    fft_size = 2048

ただし

    fft_size
    valid_size
    filter_length

は全てパラメータ化する。

------------------------------------------------------------------------

# 7. Rxx推定

標準方式

    2048 FFT
        ↓
    BinSelector
        ↓
    CovarianceEstimator

追加の短FFTは使用しない。

比較実験のため

    mode="reuse_filter_fft"

    mode="short_fft"

へ切替可能な構造とする。

------------------------------------------------------------------------

# 8. FrequencyResponseResampler

役割

    H_short
        ↓
    IFFT
        ↓
    時間領域で不足分を0埋め
        ↓
    FFT
        ↓
    H_long

FFT次数変更専用部品とする。

デフォルト

    axis = -1

------------------------------------------------------------------------

# 9. MVDR

Rxx

設定した積分時間に相当するブロック列から推定する。

MVDR重み

数秒に1回程度の周期で更新する。

Filter適用

重み更新の間は、同じ重みを保持したまま毎block出力し、
入力信号との内積で beamforming を適用する。

------------------------------------------------------------------------

# 10. Delay

PRDFT遅延補償

    DelayCompensator

手動指定

将来はフィルタファイルから取得可能とする。

------------------------------------------------------------------------

# 11. 利用コード例

``` python
X = fb.analysis(x)

Y = np.zeros(
    (n_beam, n_band, n_subband_sample),
    dtype=np.complex64,
)

for band in positive_bands:

    x_band = X[:, band, :]

    frames = frame_buffers[band].push(x_band)

    y_frames = (
        Flow.from_iterable(frames)
        .map(filter_ffts[band].process)
        .map(covariance_taps[band].process)
        .map(step_schedulers[band].process)
        .map(mvdr_filters[band].process)
        .map(filter_iffts[band].process)
        .map(valid_extractors[band].process)
        .flatten()
        .to_list()
    )

    Y[:, band, :] = np.asarray(y_frames)
```

処理の流れを隠さないことを最優先とする。

------------------------------------------------------------------------

# 12. Pending

## P1

正側16帯域から負側16帯域生成

-   共役写像
-   PRDFT位相補正

------------------------------------------------------------------------

## P2

Rxx推定方式比較

-   reuse_filter_fft
-   short_fft

処理量

SINR

Null性能

を比較する。


現時点の `PolyphaseDFTFilterBank`（critically sampled block DFT bank）で確認した事項:

-   長FFTサイズ `L = M * S` に対し、`L` 点FFT結果を `M` 点おきに間引いた値は、`S` 点短FFTを `M` ブロック分計算した結果の和に一致する。
-   したがって、間引いた長FFTは各短FFTブロックを個別には保持していない。
-   このため、`reuse_filter_fft` を単純なビン間引きとして実装すると、`short_fft` と同等の時間分解能・共分散更新レートは得られない。
-   短FFTを省略したい場合は、長FFTの全ビンを使って追加の周波数方向混合または時間方向再分割が必要であり、単純な間引きでは代替できない。

------------------------------------------------------------------------

# 13. 実装順序

1.  Polyphase
2.  PRDFT
3.  ResponseResampler
4.  Overlap-save
5.  CovarianceTap
6.  MVDR Filter
7.  DelayCompensator
8.  正負帯域写像検証
9.  Rxx推定方式比較

------------------------------------------------------------------------

# 14. PR確認結果

## 14.1 確認対象

初期実装では `DFT_FilterBank` 内の Beamforming は実施しない。

確認対象は以下のみとする。

-   実数信号のサブバンド分割
-   正側帯域のみを用いた合成
-   解析/合成のみでの perfect reconstruction

処理方式は以下である。

    real signal
        ↓
    frame slicing
        ↓
    prototype window
        ↓
    rFFT
        ↓
    positive-frequency subbands
        ↓
    irFFT
        ↓
    overlap-add
        ↓
    reconstructed real signal

## 14.2 試験条件

実装確認は `.venv` 上で実施した。

主な条件は以下である。

-   sampling rate: `16000 Hz`
-   signal length: `4096 sample`
-   `fft_size = 32`
-   `hop_size = 16`
-   prototype: sine window
-   入力信号: 雑音なし実数余弦波

帯域境界間隔は以下である。

    fs / fft_size = 500 Hz

このため、低周波・高周波に加え、帯域境界近傍として
`500 Hz`, `1000 Hz`, `4000 Hz` の前後も確認した。

評価周波数は以下である。

    10, 100,
    490, 500, 510,
    990, 1000, 1010,
    3990, 4000, 4010,
    7500, 7900 [Hz]

評価量は以下とした。

-   `max_abs_error = max(|y[n] - x[n]|)`
-   `rms_error`
-   `jump_abs = |e[n] - e[n-1]|`, ただし `e[n] = y[n] - x[n]`
-   `max_jump_abs_error`
-   `max_signal_jump_abs = max(|x[n] - x[n-1]|)`
-   `ratio_jump_error_to_signal = max_jump_abs_error / max_signal_jump_abs`

## 14.3 結果

代表結果は以下である。

| freq [Hz] | max_abs_error | rms_error | max_jump_abs_error | max_signal_jump_abs | ratio_jump_error_to_signal |
|---|---:|---:|---:|---:|---:|
| 10 | `1.78e-15` | `1.15e-16` | `2.33e-15` | `3.93e-03` | `5.94e-13` |
| 100 | `2.11e-15` | `1.16e-16` | `1.55e-15` | `3.93e-02` | `3.96e-14` |
| 490 | `2.22e-15` | `1.25e-16` | `2.55e-15` | `1.92e-01` | `1.33e-14` |
| 500 | `1.78e-15` | `1.29e-16` | `1.44e-15` | `1.95e-01` | `7.40e-15` |
| 510 | `4.00e-15` | `1.36e-16` | `3.89e-15` | `2.00e-01` | `1.94e-14` |
| 1000 | `5.00e-15` | `1.46e-16` | `3.44e-15` | `3.83e-01` | `8.99e-15` |
| 4000 | `1.78e-15` | `1.18e-16` | `1.77e-15` | `1.00e+00` | `1.77e-15` |
| 7500 | `3.00e-15` | `1.37e-16` | `3.77e-15` | `1.98e+00` | `1.91e-15` |
| 7900 | `4.44e-16` | `1.08e-16` | `5.55e-16` | `2.00e+00` | `2.78e-16` |

## 14.4 判断

確認した全周波数で、再構成誤差はほぼ機械精度レベルであった。

また `jump_abs` は、入力信号自身の 1 sample 間変化量に対して
`1e-13` から `1e-15` 程度であり、
低周波・高周波・帯域境界近傍のいずれでも
不自然な段差は観測されなかった。

したがって、初期実装の範囲である

-   サブバンド分割
-   正側帯域のみの合成
-   Beamforming なし

という条件において、PR は確保できたと判断する。

Beamforming の実装は、この PR 確認後に進める。

------------------------------------------------------------------------

# 15. Overlap-save CBF 実装結果

## 15.1 実装内容

全帯域全ビンを対象に、サブバンド時系列へ overlap-save を適用する CBF 経路を追加した。

処理方式は以下である。

    multichannel time signal
        ↓
    FullDFTFilterBank.analysis
        ↓
    complex full-band subband signals
        ↓
    per-band overlap-save framing
        frame_size = 2048
        valid_size = 1024
        50% overlap
        ↓
    FFT
        ↓
    overlap_save(conj(steering_vector))
        ↓
    transpose-only projection
        ↓
    IFFT
        ↓
    valid region extraction
        後半 1024 sample を採用
        ↓
    FullDFTFilterBank.synthesis
        ↓
    beamformed time signal

今回の実装では、各帯域の CBF 重みを time tap 1 の steering filter として扱い、
複素共役はフィルタ作成時に織り込んでいる。
そのため適用時は共役転置ではなく、転置相当の積和で整相する。

## 15.2 試験条件

`.venv` 上で `scene_renderer` を用いて確認した。

主な条件は以下である。

- sampling rate: `16000 Hz`
- signal length: `20000 sample`
- array: `4 ch`, spacing `0.04 m`
- target bearing: `20 deg`
- sound speed: `343 m/s`
- source frequency: `1000 Hz`
- signal level: `0 dB`
- additive noise level: `-60 dB`
- filterbank: `fft_size = 32`, `hop_size = 16`
- overlap-save: `frame_size = 2048`, `valid_size = 1024`
- subband streaming chunk size: `257 sample`

比較対象は以下の 2 系統とした。

- 従来の全帯域 CBF: `apply_beamformer_bands(X, W)`
- 今回追加した overlap-save CBF

## 15.3 結果

得られた主要値は以下である。

| metric | value |
|---|---:|
| `target_response_db` | `0.000000000000` |
| `max_subband_diff_to_direct` | `5.51e-15` |
| `rms_subband_diff_to_direct` | `5.88e-16` |
| `rms_time_error_to_reference` | `2.66e-02` |
| `max_time_error_to_reference` | `2.49e+00` |
| `max_subband_reanalysis_error` | `3.55e-01` |
| `rms_subband_reanalysis_error` | `1.79e-01` |

## 15.4 判断

overlap-save CBF のサブバンド出力は、従来の全帯域 CBF と機械精度レベルで一致した。
したがって、

- steering vector を複素共役後に overlap-save フィルタ化すること
- 整相時は共役転置ではなく転置相当で適用すること
- 後半 50% を valid region として採用すること

という今回の実装方針自体は、数値的に整合している。

一方で、`FullDFTFilterBank.synthesis()` 後の時間波形を再度 `analysis()` した結果は、
`Y` と一致しなかった。
これは overlap-save CBF 実装そのものではなく、
現行の `FullDFTFilterBank` が「任意の複素サブバンド列をそのまま合成して再解析すると同じ複素サブバンド列へ戻る」
構造にはまだなっていないことを示している。

したがって、今回の段階で確認できたことは以下である。

- overlap-save を導入した CBF 経路は、既存の全帯域 CBF と一致する
- 目標帯域応答は `0 dB` を維持する
- 時間波形への復元までは実行できる

今後は `FullDFTFilterBank` を、任意の複素サブバンド処理後でも再解析整合が取れる
本来のサブバンド分割系へ作り直す必要がある。

------------------------------------------------------------------------

# 16. PolyphaseDFTFilterBank

## 16.1 位置付け

現行の `PRDFTFilterBank` / `FullDFTFilterBank` は、
WOLA / STFT 型の解析合成系として残す。

これを壊さず、別クラスとして
`PolyphaseDFTFilterBank`
を追加した。

目的は、任意の複素サブバンド列 `Y` に対して

    analysis(synthesis(Y)) = Y

を満たす複素サブバンド系の基準実装を持つことである。

## 16.2 今回の実装範囲

初期版 `PolyphaseDFTFilterBank` は、
複素 full-band の critically sampled block DFT bank として実装した。

処理方式は以下である。

    complex time signal
        ↓
    block framing
        hop_size = fft_size
        ↓
    full complex DFT
        ↓
    complex subband coefficients

    complex subband coefficients
        ↓
    full complex IDFT
        ↓
    block concatenation
        ↓
    complex time signal

この構造により、任意の複素サブバンド列を一度時間波形へ戻し、
再解析しても元の複素サブバンド列へ戻る。

## 16.3 確認項目

`.venv` 上の試験で以下を確認した。

- 複素時間波形に対して `synthesis(analysis(x)) = x`
- 任意の複素サブバンド列に対して `analysis(synthesis(Y)) = Y`
- 多チャネル複素サブバンド列でも同じ性質を維持する

このクラスは、今後 overlap-save ベースのサブバンド処理系を
複素サブバンド空間として整理していく際の基準クラスとする。

## 16.4 overlap-save CBF 検証での位置付け

`PolyphaseDFTFilterBank` 初期版は、
prototype FIR 付きの本命 PRDFT filter bank ではなく、
あくまで

    critically sampled block DFT bank

であることに注意する。

したがって、現時点で確認できる主眼は

- 任意の複素サブバンド列を整合したまま扱えること
- overlap-save によるサブバンド内処理後でも
  `analysis(synthesis(Y)) = Y`
  を保てること

である。

`.venv` 上で `scene_renderer` を用いて、
`PolyphaseDFTFilterBank` 上の overlap-save CBF を確認した。

主な条件は以下である。

- sampling rate: `16000 Hz`
- signal length: `40000 sample`
- array: `4 ch`, spacing `0.04 m`
- target bearing: `20 deg`
- sound speed: `343 m/s`
- source frequency: `1000 Hz`
- signal level: `0 dB`
- additive noise level: `-60 dB`
- filterbank: `fft_size = 32`
- overlap-save: `frame_size = 2048`, `valid_size = 1024`
- subband streaming chunk size: `257 sample`

確認結果は以下である。

- overlap-save CBF 出力 `Y` は、直接計算した全帯域 CBF 出力と機械精度で一致した
- `PolyphaseDFTFilterBank.synthesis(Y)` で時間波形へ戻した後、
  `analysis()` すると元の `Y` に機械精度で一致した
- target 応答は `0 dB` を維持した

したがって、`PolyphaseDFTFilterBank` 初期版は
複素サブバンド処理系の基準実装として利用可能である。

## 16.5 PolyphaseDFTFilterBank + overlap-save CBF sweep

低域・高域・帯域境界近傍を含む複数周波数で、
`PolyphaseDFTFilterBank` 上の overlap-save CBF を評価した。

評価観点は以下とした。

- `target_response_db`
- 直接計算した全帯域 CBF と overlap-save CBF の一致度
- `analysis(synthesis(Y)) = Y` の再解析整合度
- 参照実数余弦波に対する時間波形誤差

主な条件は以下である。

- sampling rate: `16000 Hz`
- signal length: `40000 sample`
- array: `4 ch`, spacing `0.04 m`
- target bearing: `20 deg`
- sound speed: `343 m/s`
- signal level: `0 dB`
- additive noise level: `-60 dB`
- filterbank: `fft_size = 32`
- overlap-save: `frame_size = 2048`, `valid_size = 1024`
- subband streaming chunk size: `257 sample`

結果は以下である。

| freq [Hz] | nearest_bin | target_response_db | max_subband_diff_to_direct | max_subband_reanalysis_error | rms_time_error_to_reference | max_time_error_to_reference |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0 | `0.000000000000` | `1.455e-14` | `3.553e-15` | `1.206e-02` | `5.658e-02` |
| 100 | 0 | `0.000000000000` | `1.438e-14` | `3.553e-15` | `1.129e-01` | `4.980e-01` |
| 490 | 1 | `0.000000000000` | `9.566e-15` | `5.255e-15` | `1.212e-02` | `5.676e-02` |
| 500 | 1 | `0.000000000000` | `1.078e-14` | `5.217e-15` | `4.980e-04` | `2.055e-03` |
| 510 | 1 | `0.000000000000` | `1.074e-14` | `4.830e-15` | `1.213e-02` | `5.680e-02` |
| 990 | 2 | `0.000000000000` | `1.005e-14` | `4.252e-15` | `1.229e-02` | `5.815e-02` |
| 1000 | 2 | `0.000000000000` | `1.014e-14` | `5.330e-15` | `4.980e-04` | `2.055e-03` |
| 1010 | 2 | `0.000000000000` | `1.137e-14` | `3.972e-15` | `1.231e-02` | `5.823e-02` |
| 3990 | 8 | `0.000000000000` | `9.566e-15` | `2.512e-15` | `1.310e-02` | `6.513e-02` |
| 4000 | 8 | `0.000000000000` | `8.032e-15` | `3.553e-15` | `4.980e-04` | `2.055e-03` |
| 4010 | 8 | `0.000000000000` | `1.081e-14` | `3.553e-15` | `1.309e-02` | `6.523e-02` |
| 7500 | 15 | `-0.000000000000` | `1.025e-14` | `5.152e-15` | `4.980e-04` | `2.055e-03` |
| 7900 | 16 | `0.000000000000` | `1.085e-14` | `1.971e-15` | `3.959e-01` | `9.956e-01` |

判断は以下である。

- overlap-save CBF 出力 `Y` は、全評価周波数で直接計算した全帯域 CBF 出力と機械精度で一致した
- `analysis(synthesis(Y)) = Y` も、全評価周波数で機械精度で成立した
- 一方で時間波形誤差は、DFT bin center に一致する `500`, `1000`, `4000`, `7500 Hz` では小さいが、off-bin や Nyquist 近傍では大きくなった

これは `PolyphaseDFTFilterBank` 初期版が prototype なしの block DFT bank であり、
複素サブバンド整合の基準実装としては有効だが、
prototype FIR を伴う本来の帯域分離性能まではまだ持っていないことと整合している。

したがって、MVDR に進む前段としては、

- overlap-save 処理後の複素サブバンド列 `Y` を整合したまま扱えること
- CBF 実装そのものは direct 計算と一致すること

は確認できたと判断する。

一方で、時間波形側の周波数依存誤差をさらに抑えるには、
次段階として prototype 付きの本命 polyphase DFT filter bank が必要である。

------------------------------------------------------------------------

# 17. Prototype付き Polyphase DFT Bank 初期実装

## 17.1 追加部品

以下を追加した。

- `PrototypeFilter`
- `PolyphaseDecomposition`
- `PrototypeAnalysisDFTFilterBank`
- `PrototypeSynthesisDFTFilterBank`
- `PRChecker`

初期条件は以下で固定した。

- `n_band = 32`
- `decimation = 32`
- `prototype_length = 256`
- `band_order = FFT順`
- `delay compensation = 手動指定`

## 17.2 位置付け

今回の prototype 付き系は、現行の `PolyphaseDFTFilterBank` baseline を壊さずに、
その上に prototype FIR と polyphase decomposition を追加するための初期実装である。

ただし default の `PrototypeFilter.block_dft_baseline()` は、
`prototype_length = 256` を持つものの、
最初の 32 tap のみが有効な sparse prototype であり、
動作としては block DFT baseline と同値である。

この default は以下の目的で残す。

- polyphase 構造そのものの実装確認
- `PrototypeAnalysisDFTFilterBank` / `PrototypeSynthesisDFTFilterBank` の PR 基準確認
- 後から非自明な prototype を入れた際の比較基準

## 17.3 評価用 prototype

非自明な prototype 評価用として、
`PrototypeFilter.windowed_sinc()` を追加した。

これは本命の PR prototype 設計ではなく、
まずは

- 低域通過型の prototype を投入できること
- passband / stopband の簡易評価ができること

を確認するためのものである。

## 17.4 テスト

初期実装では以下を確認した。

- sparse prototype baseline に対して `synthesis(analysis(x)) = x`
- sparse prototype baseline に対して `analysis(synthesis(Y)) = Y`
- `PolyphaseDecomposition` の shape と係数配置
- `windowed_sinc` prototype の stopband attenuation が passband より十分低いこと

これにより、現行の block DFT baseline を保持したまま、
prototype 付き本命実装へ拡張するための最小骨格が整った。

## 17.5 別スクリプトでの設計・最適化・評価

prototype フィルタの設計、簡易最適化、評価をライブラリ本体から切り離して実施するため、

- `tools/design_prdft_prototype.py`

を追加した。

このスクリプトでは以下を行う。

- `PrototypeFilter.block_dft_baseline()` を基準値として評価
- `PrototypeFilter.windowed_sinc()` を複数 cutoff で生成
- delay compensation を格子探索
- `PRChecker` による PR 誤差評価
- 周波数応答から stopband attenuation を評価
- 候補を score 順に一覧表示

現時点の結果では、windowed-sinc 候補は stopband attenuation を改善できる一方、
現行の初期 synthesis 構造では PR 誤差がまだ大きいことが確認できる。

したがってこのスクリプトは、

- prototype の帯域分離性能の比較
- delay compensation の探索
- 今後の本命 PR prototype 設計の改善確認

に用いる。 

## 17.6 PR確保可能な prototype 設計の原因分析

現時点で PR を確保可能な prototype を求めるため、
`tools/design_prdft_prototype.py` を更新し、
analysis prototype に対して dual synthesis prototype を別設計する方式で評価した。

原因分析の要点は以下である。

- 同一 prototype を analysis / synthesis にそのまま流用しても、PR に必要な dual 条件は一般には満たされない
- 現行の prototype-bank 初期構造では、PR 条件は各 phase ごとの相関条件へ還元される
- そのため、非自明な prototype では stopband attenuation は改善しても、PR が自然には成立しない
- dual synthesis prototype を最小二乗で別設計しても、現行構造では nontrivial prototype に対して PR 誤差が残る

この条件で cutoff / delay を sweep した結果、
PR を厳密に満たした最良候補は

    PrototypeFilter.block_dft_baseline()

であった。

代表結果は以下である。

| rank | name | delay_blocks | delay_samples | exact_pr | stopband_attenuation_db | max_abs_error | rms_error |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `block_dft_baseline` | `0` | `0` | `True` | `13.433` | `9.899e-16` | `2.664e-16` |
| 2 | `windowed_sinc@0.03125000` | `0` | `0` | `False` | `61.142` | `2.310e+00` | `3.089e-01` |
| 3 | `windowed_sinc@0.02812500` | `0` | `0` | `False` | `58.669` | `2.668e+00` | `2.465e-01` |

したがって、現時点で

- PR を厳密に確保する
- 現行の prototype-bank 初期構造を維持する

という条件を両立する最適 prototype は、
block DFT baseline 相当の sparse prototype である。

生成した最良 prototype は以下へ保存した。

- `artifacts/prototype_design/optimal_pr_prototype.npz`
- `artifacts/prototype_design/optimal_pr_prototype.json`

次に本当に nontrivial な prototype で PR を確保したい場合は、
prototype 係数の最適化だけではなく、
polyphase / modulation 構造そのものを本命の PRDFT 形へ拡張する必要がある。

------------------------------------------------------------------------

# 18. 明示的 Modulation 方式の本命系

## 18.1 追加クラス

prototype-bank 初期実装とは別に、
DFT 変調を明示的に表現する本命系クラスを追加した。

- `DFTModulatedFilterDesigner`
- `PRDFTAnalysisBank`
- `PRDFTSynthesisBank`

この系では、prototype から直接

- 解析フィルタ群 `h_k[n]`
- 合成フィルタ群 `g_k[n]`

を DFT 変調により生成し、
analysis / synthesis を

- decimation
- interpolation
- overlap-add

として明示的に実行する。

## 18.2 現行 prototype-bank との違い

`PrototypeAnalysisDFTFilterBank` / `PrototypeSynthesisDFTFilterBank` は、
polyphase の shape と prototype 展開を確認するための初期実装である。

一方で `PRDFTAnalysisBank` / `PRDFTSynthesisBank` は、
より本命に近い形として

- まず modulated FIR filter bank を作る
- その filter bank を decimation / interpolation で適用する

という構成を取る。

したがって、

- polyphase 表現の確認
- DFT modulation の確認
- PR の実検証

を分離して進められる。

## 18.3 現時点の確認結果

`PrototypeFilter.block_dft_baseline()` に対しては、
`PRDFTAnalysisBank` / `PRDFTSynthesisBank` でも厳密 PR を確認した。

一方で `windowed_sinc` prototype をそのまま投入すると、
stopband attenuation は改善するが PR は成立しない。

これは、現時点での本命系クラス追加によって
構造の誤りを切り分けられるようになったことを意味する。

すなわち、

- baseline では structure は正しく動く
- nontrivial prototype で PR が崩れるのは prototype / dual 設計側の問題である

と判断できる。

## 18.4 PrototypePairDesigner

analysis / synthesis prototype の対設計器として、
`PrototypePairDesigner` を追加した。

この設計器は、明示的 modulation 方式の `PRDFTAnalysisBank` / `PRDFTSynthesisBank` に対して、
全 decimation 位相のインパルス応答を同時に扱う cascade matrix を構成し、
それが遅延付き delta 応答へ近づくように synthesis prototype を最小二乗で設計する。

要点は以下である。

- 1位相だけでなく `0 .. decimation-1` の全位相インパルスを同時に設計条件へ入れる
- `delay_samples` を変えて paired synthesis prototype を生成できる
- PR 条件に対する residual を直接評価できる

## 18.5 最適化スクリプトの実用化

`tools/design_prdft_prototype.py` を CLI 化し、
以下の条件を外から変更できるようにした。

- `n_band`
- `decimation`
- `prototype_length`
- `n_samples`
- `seed`
- `cutoff_scale_list`
- `delay_start`
- `delay_stop`
- `delay_step`
- `artifact_name`

このスクリプトは

- analysis prototype 候補生成
- paired synthesis prototype 設計
- pair residual 評価
- 有限長 PR 評価
- 周波数応答評価
- 最良 pair の保存

を 1 回で実行する。

## 18.6 現時点の解釈

multi-phase の pair residual 自体は、
`windowed_sinc` に対しても非常に小さくできることを確認した。

一方で、有限長信号の `synthesis(analysis(x))` では依然として大きな誤差が残る。

これは、

- paired prototype 設計そのもの
- 明示的 modulation による内部 filter bank 条件
- 有限長 signal の framing / padding / edge handling

のうち、最後の有限長処理まで含めた本命 PR 条件がまだ未完成であることを示す。

したがって現段階では、

- pair designer により対設計の骨格は実装済み
- しかし nontrivial prototype の実用 PR には、有限長境界条件まで含めた構造調整がなお必要

と整理する。

------------------------------------------------------------------------

# 19. 本命 Polyphase PRDFT 系の追加

## 19.1 追加クラス

explicit modulation 系を残したまま、
本命の polyphase + DFT/IDFT 構造を別クラスとして追加した。

- `PolyphasePRDFTAnalysisBank`
- `PolyphasePRDFTSynthesisBank`
- `PolyphasePRPairDesigner`

この系では、prototype を polyphase branch に分解し、

- analysis: block 列に対する polyphase branch 和
- DFT による帯域化
- synthesis: IDFT 後の各 branch 列の畳み込み
- block domain overlap-add

として構成した。

## 19.2 explicit modulation 系との違い

従来の `PRDFTAnalysisBank` / `PRDFTSynthesisBank` は、
prototype をそのまま full-length modulated FIR 群として適用する評価系である。

一方、今回の `PolyphasePRDFTAnalysisBank` / `PolyphasePRDFTSynthesisBank` は、
critically sampled DFT bank の本来の polyphase branch 構造を明示し、
有限長入力に対しても

- front padding
- back padding
- branch 畳み込み
- delay compensation

を明示的に扱う。

## 19.3 Pair 設計器の変更点

`PolyphasePRPairDesigner` は、
analysis prototype を polyphase decomposition した各 branch に対して、
branch-wise convolution が遅延付き delta に近づくように
synthesis branch を最小二乗で設計する。

すなわち設計条件を

    branch_e[m] (*) branch_g[m] ≈ delta[m - d]

へ直接落としている。

これにより、以前の cascade matrix ベースよりも
本命構造に沿った pair 設計が可能になった。

## 19.4 テスト結果

追加テストで以下を確認した。

- `block_dft_baseline` では `PolyphasePRDFTAnalysisBank` / `PolyphasePRDFTSynthesisBank` でも厳密 PR
- `PolyphasePRPairDesigner` は `block_dft_baseline` の dual を厳密に回復
- `windowed_sinc` に対しては、branch-wise dual 設計により finite-length valid-region 誤差が改善

回帰試験は `.venv` 上で

    pytest -q

を実行し、`66 passed` を確認した。

## 19.5 評価スクリプトの更新

`tools/design_prdft_prototype.py` を更新し、
以下を切り替え可能にした。

- `--structure explicit_modulation`
- `--structure polyphase_pr`

`polyphase_pr` では追加で

- `--delay-block-start/stop/step`
- `--synthesis-prototype-length-list`

を指定できる。

これにより、analysis prototype 長とは別に
synthesis prototype 長も sweep しながら、

- pair residual
- full-length PR
- valid-region PR
- stopband attenuation

を一括評価できる。

## 19.6 現時点の評価

以下の条件で `polyphase_pr` を評価した。

- `n_band = 32`
- `decimation = 32`
- `analysis prototype_length = 256`
- `synthesis prototype_length = 256, 512`
- `delay_blocks = 0 .. 12`
- `regularization = 0, 1e-6, 1e-4`

最良候補は引き続き

    block_dft_baseline

であり、厳密 PR を維持した。

一方、nontrivial prototype の代表として
`windowed_sinc@0.02812500` では、
例えば

- `synthesis_prototype_length = 512`
- `delay_blocks = 12`

の条件で

- `pair_rms_error = 1.280e-03`
- `pr_rms_error = 3.520e-01`
- `valid_pr_rms_error = 7.719e-03`
- `stopband_attenuation_db = 58.669`

を確認した。

## 19.7 判断

今回の追加により、
nontrivial prototype を評価する際の本命構造は
explicit modulation の近似系ではなく、
polyphase branch 条件に基づく形へ整理できた。

ただし現時点では、
`windowed_sinc` のような非自明 prototype に対して

- stopband attenuation の改善
- valid-region 誤差の低減

までは達成できているが、
full-length exact PR はまだ成立していない。

したがって次段階は、

- analysis/synthesis prototype 対の最適化をさらに進めること
- 必要なら synthesis prototype 長を延長すること
- finite-length の cropping rule を、用途に応じて valid-region ベースで整理すること

である。

------------------------------------------------------------------------

# 20. finite-length cropping rule の整理

## 20.1 方針

`FiniteLengthPRChecker` を、単なる 1 つの PR 誤差評価器ではなく、

- padding 規約
- cropping 規約
- valid-region 規約

を分離して扱う部品へ更新した。

これにより、用途に応じて

- 入力長に整列した再構成を見たいのか
- padded 全長で連続性を見たいのか
- 境界過渡を除いた valid region だけ見たいのか

を明示できるようにした。

## 20.2 実装した cropping rule

`FiniteLengthPRChecker.check()` / `reconstruct()` に以下を追加した。

- `crop_mode='input_aligned'`
  - 従来互換
  - front padding 分だけ先頭を落とし、入力長と同じ長さを比較する
- `crop_mode='full'`
  - front/back padding を含む padded 全長で比較する
- `crop_mode='valid'`
  - analysis/synthesis の最大 transient を両端から落とした central region を比較する
- `crop_mode='custom'`
  - `crop_front`, `crop_length` を手動指定する

重要なのは、参照信号側も同じ cropping 規約で切り出すようにした点である。
すなわち、`full` / `valid` / `custom` では、
再構成側だけでなく padded reference 側にも同じ slice を適用する。

## 20.3 valid-region rule

valid-region の誤差集計も別規約として分離した。

- `valid_region_mode='transient'`
  - `max(analysis_transient, synthesis_transient)` を両端から除去
- `valid_region_mode='analysis'`
  - analysis 側 transient のみ除去
- `valid_region_mode='synthesis'`
  - synthesis 側 transient のみ除去
- `valid_region_mode='none'`
  - 出力全体を valid region とみなす
- `valid_region_mode='custom'`
  - `valid_margin` を手動指定する

これにより、

- full-length 波形品質
- 入力整列後の誤差
- 実際に利用する中央領域だけの誤差

を別々に論じられるようになった。

## 20.4 prototype 最適化スクリプトの拡張

`tools/design_prdft_prototype.py` を更新し、
評価条件を外から変えても最適化可能なようにした。

追加・整理した主な引数は以下である。

- `--eval-length-list`
- `--eval-seed-list`
- `--crop-mode`
- `--crop-front`
- `--crop-length`
- `--valid-region-mode`
- `--valid-margin`
- `--pad-front`
- `--pad-back`
- `--score-mode`
- `--synthesis-prototype-length-list`

評価は複数の `(signal_length, seed)` をまとめて実行し、
`max_abs_error`, `rms_error`, `valid_rms_error` などを worst-case 集約する。

`score_mode` は以下を選べる。

- `worst_valid`
  - valid region の worst-case 誤差を重視
- `worst_full`
  - full-length 誤差を重視
- `balanced`
  - full / valid の両方を半々で評価

## 20.5 確認結果

`.venv` 上で以下の例を実行した。

    python tools/design_prdft_prototype.py \
        --structure polyphase_pr \
        --n-band 32 \
        --decimation 32 \
        --prototype-length 256 \
        --n-samples 2048 \
        --seed 1 \
        --eval-length-list 1024 2048 \
        --eval-seed-list 1 2 \
        --delay-block-start 0 \
        --delay-block-stop 12 \
        --delay-block-step 1 \
        --regularization-list 0 1e-6 1e-4 \
        --synthesis-prototype-length-list 256 512 \
        --crop-mode valid \
        --valid-region-mode none \
        --score-mode balanced

この条件では、`block_dft_baseline` が引き続き最良であり、
4 条件の worst-case 集約でも厳密 PR を維持した。

生成物は以下へ保存した。

- `artifacts/prototype_design/polyphase_pr_valid_demo/prototype_pair.npz`
- `artifacts/prototype_design/polyphase_pr_valid_demo/prototype_pair.json`

## 20.6 回帰試験

追加した有限長評価規約について、以下を確認した。

- `crop_mode='full'` で baseline が厳密 PR
- `crop_mode='valid'` + `valid_region_mode='custom'` でも baseline が厳密 PR

回帰試験は `.venv` 上で

    pytest -q

を実行し、`68 passed` を確認した。

------------------------------------------------------------------------

# 21. MVDR 動作検証への移行

## 21.1 前提

CBF については、`PolyphaseDFTFilterBank` baseline 上で

- target 応答が `0 dB`
- direct 計算と overlap-save 実装が一致
- `analysis(synthesis(Y)) = Y` が成立

することを確認済みである。

したがって、MVDR の動作検証は

- baseline の複素サブバンド整合性を利用する
- まず beamformer 数式そのものの妥当性を確認する

という方針で進める。

## 21.2 現行 MVDR 評価スクリプトの見直し

従来の `scene_renderer_mvdr_eval.py` は、
単一 active band のみで時間波形へ戻していたため、
時間波形誤差や scan peak をもって MVDR の良否を論じるには不適切であった。

そこで、評価方式を以下へ変更した。

- filter bank: `PolyphaseDFTFilterBank(fft_size=32)`
- target: `1000 Hz`, `20 deg`
- interferer: `1000 Hz`, `-30 deg`
- target level: `0 dB`
- interferer level: `0 dB`
- array: `4 ch`, spacing `0.04 m`

評価量は以下とした。

- `cbf_target_response_db`
- `mvdr_target_response_db`
- `cbf_interferer_response_db`
- `mvdr_interferer_response_db`
- target-only 参照波形に対する `rms_time_error`
- `analysis(synthesis(Y_mvdr)) = Y_mvdr` の再解析誤差

## 21.3 結果

`.venv` 上で `evaluations/beamforming/scene_renderer_mvdr_eval.py` を実行し、
以下を得た。

| metric | value |
|---|---:|
| `target_bin` | `2` |
| `cbf_target_response_db` | `0.000000000000` |
| `mvdr_target_response_db` | `0.000000000000` |
| `cbf_interferer_response_db` | `-0.015823905846` |
| `mvdr_interferer_response_db` | `-4.891907994333` |
| `cbf_rms_time_error_to_target_reference` | `7.058197363947e-01` |
| `mvdr_rms_time_error_to_target_reference` | `4.026145549266e-01` |
| `cbf_max_time_error_to_target_reference` | `9.981798217090e-01` |
| `mvdr_max_time_error_to_target_reference` | `5.693829640052e-01` |
| `max_subband_reanalysis_error` | `8.980271370237e-16` |
| `rms_subband_reanalysis_error` | `2.661721740614e-16` |

## 21.4 判断

この条件では、MVDR は

- target 応答 `0 dB` を維持した
- interferer 応答を CBF より約 `4.88 dB` 低減した
- target-only 参照に対する時間波形誤差も CBF より改善した
- baseline 複素サブバンド列として `analysis(synthesis(Y_mvdr)) = Y_mvdr` を機械精度で満たした

したがって、baseline 上の beamformer 数式検証としては、
CBF の次段として MVDR 検証へ進める条件を満たしたと判断する。

## 21.5 追加した回帰試験

以下を追加した。

- zero-trace covariance でも `diag_load` により MVDR weight 設計が破綻しないこと
- `PolyphaseDFTFilterBank` baseline 上で、MVDR が CBF より interferer 抑圧と target 参照誤差で改善すること

回帰試験は `.venv` 上で

    pytest -q

を実行し、`77 passed` を確認した。

------------------------------------------------------------------------

# 22. MVDR 共分散積分と忘却係数

## 22.1 方針

MVDR の共分散行列 `Rxx` は、時間方向に積分する方式とした。
積分係数は固定値ではなく、

- 積分時間 `T`
- 実際に積分が更新されるレート `rate`

から決める。

ここで `rate` は、システム全体の処理周期ではなく、
実際に共分散積分が 1 回更新されるレートを意味する。
本実装では baseline `PolyphaseDFTFilterBank(fft_size=32)` を使うため、
今回の例では

    rate = fs / fft_size

を採用した。

## 22.2 忘却係数

実装した忘却係数 `alpha` は以下である。

    alpha = min(2 / (1 + T * rate), 1)

共分散更新は

    Rxx[n] = (1 - alpha) * Rxx[n-1] + alpha * Rxx_current[n]

とした。

`src/spflow/beamforming/covariance.py` に以下を追加した。

- `forgetting_factor_from_integration_time(integration_time, rate)`
- `CovarianceEstimator.from_integration_time(integration_time, rate)`

既存の `smoothing` 指定は後方互換のため残しつつ、
新規実装では `forgetting_factor` を主に使う。

## 22.3 MVDR 評価での使用条件

`evaluations/beamforming/scene_renderer_mvdr_eval.py` と回帰試験では、
以下で確認した。

- `fs = 16000 Hz`
- `fft_size = 32`
- `integration_time = 0.25 s`
- `rate = fs / fft_size = 500 Hz`

このとき

    alpha = 2 / (1 + 0.25 * 500)
          = 0.015873015873...

である。

## 22.4 結果

この忘却係数を使った場合でも、
MVDR の代表結果は以下であった。

- `cbf_target_response_db = 0 dB`
- `mvdr_target_response_db = 0 dB`
- `cbf_interferer_response_db = -0.0158 dB`
- `mvdr_interferer_response_db = -4.8919 dB`
- `cbf_rms_time_error_to_target_reference = 7.058e-01`
- `mvdr_rms_time_error_to_target_reference = 4.026e-01`
- `analysis(synthesis(Y_mvdr)) = Y_mvdr` は機械精度で成立

したがって、忘却係数つき時間積分共分散でも、
baseline 上の MVDR 動作検証は成立したと判断する。

## 22.5 回帰試験

以下を追加した。

- 忘却係数が `min(2 / (1 + T*rate), 1)` に一致すること
- `CovarianceEstimator.from_integration_time()` が期待どおりに指数積分すること
- この積分方式を使った MVDR が CBF より interferer 抑圧と target 参照誤差で改善すること

回帰試験は `.venv` 上で

    pytest -q

を実行し、`72 passed` を確認した。

------------------------------------------------------------------------

# 23. ビーム探索方向生成 `make_directions`

## 23.1 追加部品

ビーム応答の各方位設計用として、以下を追加した。

- `spflow.beamforming.make_directions`

返り値は

- `Dir3d`: `shape = (3, n_beam_all)` の direction cosine
- `AxisAz`: 水平方位軸
- `AxisEl`: 俯仰方位軸

である。

この `Dir3d` は、そのまま

    delay = array_pos @ Dir3d / sonic_speed

として遅延時間計算へ使える。

## 23.2 パラメータ

実装した主な引数は以下である。

- `az_min_deg`, `az_max_deg`
- `el_min_deg`, `el_max_deg`
- `n_beam_az_real`
- `n_beam_az_virtual`
- `n_beam_el`
- `array_side`
  - `'right side'`
  - `'left side'`
  - `'forward'`
- `el_preset_deg`

`el_preset_deg` の default は

    sort([18.1, 10.6, 6.0, -30], 'ascend')

に相当する

    [-30.0, 6.0, 10.6, 18.1]

とした。

## 23.3 水平方位の作成規約

- `forward`
  - 等角度空間
  - `linspace(az_min, az_max, n_beam_az)`
- `right side` / `left side`
  - 等 cos 空間
  - 実ビーム数と虚ビーム数から total beam 数を作る

Y 成分の符号は

- `right side`: 正
- `left side`: 負
- `forward`: `sin(theta)` の符号をそのまま使用

とした。

## 23.4 テスト

以下を確認した。

- `right side` では `Dir3d(2, :) >= 0`
- `left side` では `right side` に対して Y 成分のみ符号反転
- `forward` では `AxisAz` が等角度列になり、Y 成分が正負の両方を持つ
- 全方向ベクトルが unit norm

回帰試験は `.venv` 上で

    pytest -q

を実行し、`75 passed` を確認した。

------------------------------------------------------------------------

# 24. scene_renderer試験時の信号レベル整合

scene_renderer を使う CBF / MVDR 評価では、source 振幅を

    sqrt(2) * 10^(SL / 20)

で与えるように統一した。

ここで `SL` は dB 表記の目標実効値である。
この定義により、例えば `SL = 0 dB` のとき、生成する実数余弦波の
RMS 値は `1.0` になる。

したがって、beam 応答の dB 評価は

    20 * log10(|response|)

で行い、distortionless 条件の期待値は `0 dB` とする。

以前の `10 * log10(2 * |response|^2)` は peak 振幅基準では等価だが、
本設計書では RMS 基準で整理するため上記へ統一する。

------------------------------------------------------------------------

# 25. RMS基準整合後の CBF / MVDR sweep 結果

## 25.1 試験条件

RMS 基準を揃えた後、`scene_renderer` を用いた sweep を再実施した。

共通条件は以下である。

- sampling rate: `16000 Hz`
- array: `4 ch`, spacing `0.04 m`
- sound speed: `343 m/s`
- target bearing: `20 deg`
- target elevation: `0 deg`
- signal level: `0 dB RMS`
- source peak amplitude: `sqrt(2) * 10^(SL / 20)`
- beam response dB: `20 * log10(|response|)`

CBF sweep 条件は以下である。

- filter bank: `PolyphaseDFTFilterBank(fft_size=32)`
- signal length: `40000 sample`
- additive noise level: `-60 dB`
- overlap-save: `frame_size = 2048`, `valid_size = 1024`
- subband streaming chunk size: `257 sample`

MVDR sweep 条件は以下である。

- filter bank: `PolyphaseDFTFilterBank(fft_size=32)`
- signal length: `40000 sample`
- interferer bearing: `-30 deg`
- interferer elevation: `0 deg`
- interferer level: `0 dB RMS`
- covariance integration time: `0.25 s`
- covariance update rate: `fs / fft_size = 500 Hz`
- forgetting factor:

      alpha = min(2 / (1 + T * rate), 1)
            = 0.015873015873...

評価周波数は以下である。

    10, 100,
    490, 500, 510,
    990, 1000, 1010,
    3990, 4000, 4010,
    7500, 7900 [Hz]

## 25.2 CBF sweep 結果

`evaluations/beamforming/scene_renderer_cbf_polyphase_sweep.py` の結果は以下である。

| freq [Hz] | nearest_bin | target_response_db | max_subband_diff_to_direct | max_subband_reanalysis_error | rms_time_error_to_reference | max_time_error_to_reference |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0 | `0.000000000000` | `2.157e-14` | `3.553e-15` | `1.705e-02` | `7.939e-02` |
| 100 | 0 | `0.000000000000` | `2.222e-14` | `7.105e-15` | `1.596e-01` | `7.036e-01` |
| 490 | 1 | `0.000000000000` | `1.589e-14` | `7.324e-15` | `1.714e-02` | `7.978e-02` |
| 500 | 1 | `0.000000000000` | `1.449e-14` | `7.227e-15` | `4.980e-04` | `2.055e-03` |
| 510 | 1 | `0.000000000000` | `1.507e-14` | `7.944e-15` | `1.715e-02` | `7.984e-02` |
| 990 | 2 | `0.000000000000` | `1.589e-14` | `7.109e-15` | `1.738e-02` | `8.175e-02` |
| 1000 | 2 | `0.000000000000` | `1.495e-14` | `7.133e-15` | `4.980e-04` | `2.055e-03` |
| 1010 | 2 | `0.000000000000` | `1.465e-14` | `5.660e-15` | `1.739e-02` | `8.185e-02` |
| 3990 | 8 | `0.000000000000` | `1.432e-14` | `3.972e-15` | `1.851e-02` | `9.151e-02` |
| 4000 | 8 | `0.000000000000` | `1.241e-14` | `3.553e-15` | `4.980e-04` | `2.055e-03` |
| 4010 | 8 | `0.000000000000` | `1.518e-14` | `3.972e-15` | `1.851e-02` | `9.169e-02` |
| 7500 | 15 | `-0.000000000000` | `1.168e-14` | `7.317e-15` | `4.980e-04` | `2.055e-03` |
| 7900 | 16 | `0.000000000000` | `1.507e-14` | `3.553e-15` | `5.599e-01` | `1.407e+00` |

CBF について確認できることは以下である。

- target 応答は全周波数で `0 dB` を維持した
- overlap-save CBF 出力と direct 計算は全周波数で機械精度一致した
- `analysis(synthesis(Y)) = Y` も全周波数で機械精度成立した
- 一方で時間波形誤差は bin center では小さく、off-bin と Nyquist 近傍では増大した

したがって、baseline `PolyphaseDFTFilterBank` 上の CBF 実装自体は整合している。
高域での時間波形誤差増大は、beamformer 数式よりも
prototype なし block DFT bank の帯域表現限界を反映している。

## 25.3 MVDR sweep 結果

`evaluations/beamforming/scene_renderer_mvdr_polyphase_sweep.py` の結果は以下である。

| freq [Hz] | target_bin | cbf_target_db | mvdr_target_db | cbf_interferer_db | mvdr_interferer_db | cbf_rms_err | mvdr_rms_err | max_reanalysis_err |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0 | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `9.997e-01` | `1.000e+00` | `1.593e-03` |
| 100 | 0 | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `9.775e-01` | `9.612e-01` | `4.053e-02` |
| 490 | 1 | `0.000000` | `0.000000` | `-0.003955` | `-16.660461` | `9.991e-01` | `4.513e-01` | `1.599e-03` |
| 500 | 1 | `0.000000` | `0.000000` | `-0.003955` | `-26.621371` | `9.995e-01` | `4.666e-02` | `3.558e-15` |
| 510 | 1 | `0.000000` | `0.000000` | `-0.003955` | `-9.412008` | `9.995e-01` | `7.176e-01` | `1.657e-03` |
| 990 | 2 | `0.000000` | `0.000000` | `-0.015824` | `-8.979921` | `9.975e-01` | `7.092e-01` | `5.698e-03` |
| 1000 | 2 | `0.000000` | `0.000000` | `-0.015824` | `-4.891915` | `9.982e-01` | `5.694e-01` | `1.807e-15` |
| 1010 | 2 | `0.000000` | `0.000000` | `-0.015824` | `-2.407835` | `9.984e-01` | `3.948e-01` | `7.214e-03` |
| 3990 | 8 | `0.000000` | `0.000000` | `-0.254770` | `-1.326252` | `9.693e-01` | `9.717e-01` | `1.253e-05` |
| 4000 | 8 | `0.000000` | `-0.000000` | `-0.254770` | `-0.296104` | `9.711e-01` | `9.665e-01` | `0.000e+00` |
| 4010 | 8 | `0.000000` | `0.000000` | `-0.254770` | `0.751710` | `9.726e-01` | `9.638e-01` | `1.374e-05` |
| 7500 | 15 | `-0.000000` | `0.000000` | `-0.911293` | `-0.082845` | `9.004e-01` | `9.905e-01` | `2.782e-17` |
| 7900 | 16 | `0.000000` | `0.000000` | `-1.040401` | `-5.468351` | `4.452e-01` | `9.881e-01` | `1.954e-01` |

## 25.4 MVDR の良否整理

結果を帯域ごとに整理すると以下である。

| 帯域 | 周波数例 | target 応答 | interferer 抑圧 | target 参照誤差 | 判断 |
|---|---|---|---|---|---|
| DC 近傍 | `10`, `100 Hz` | 維持 | 改善なし | 改善ほぼなし | 良否保留 |
| 低域の off-bin / on-bin | `490`, `500`, `510 Hz` | 維持 | 大きく改善 | 明確に改善 | 良好 |
| 中低域 | `990`, `1000`, `1010 Hz` | 維持 | 改善 | 改善 | 良好 |
| 中高域 | `3990`, `4000 Hz` | 維持 | わずかに改善 | 改善は小さい | 限定的 |
| 4000 Hz 超の off-bin | `4010 Hz` | 維持 | 悪化 | わずかに改善 | 不安定 |
| 高域 | `7500 Hz` | 維持 | 悪化 | 悪化 | 不良 |
| Nyquist 近傍 | `7900 Hz` | 維持 | 一見改善 | target 誤差悪化、再解析誤差増大 | 不良 |

ここで重要なのは、現時点の不良を直ちに
「アレイ設計上の限界」とは言えないことである。

理由は以下である。

- 同じ sweep で CBF は全帯域 `0 dB` target 応答を保っている
- baseline `PolyphaseDFTFilterBank` は prototype なし block DFT bank であり、off-bin と高域で時間波形表現が粗い
- MVDR の共分散は `fft_size = 32` の粗い周波数分解で推定しており、高域では 1 bin 内の target / interferer / leakage の影響を強く受ける
- `7900 Hz` では `max_reanalysis_err` 自体が大きく、beamformer 以前に複素サブバンド列の数値条件が悪化している

したがって、今回の sweep から言えることは以下である。

- 低域から中域では、現在の MVDR 実装は概ね妥当である
- 高域と Nyquist 近傍では、現状の baseline 上で安定に全帯域運用できる状態には達していない
- その主因候補は「アレイ設計そのもの」よりも、まず
  - prototype なし block DFT bank
  - 粗い周波数分解
  - bin 内 leakage を含んだ共分散推定
  の側にある

## 25.5 現段階での判断

理想的には MVDR は全帯域で使えるべきであるが、
今回の baseline 実装はその前提条件をまだ十分には満たしていない。

特に、MVDR が全帯域で安定に効くかを論じるには、
少なくとも以下の切り分けが必要である。

1. `fft_size = 32` 固定ではなく、周波数分解を上げた場合に高域の不安定性が減るか
2. prototype 付き本命 filter bank 上でも同じ高域悪化が出るか
3. interferer なし / target のみで MVDR weight を通したときに高域で自己歪みが出るか
4. array manifold と steering vector の不整合があるか

したがって現段階の整理は、

- CBF 基準系は整合済み
- MVDR は低域から中域で有効
- 全帯域運用の妥当性は未確定
- ただし現状結果だけでアレイ設計起因と断定するのは早い

とするのが妥当である。

------------------------------------------------------------------------

# 26. 0-10000 Hz 安定化に向けた切り分け結果

## 26.1 切り分けの狙い

第 25 章の sweep では、高域で MVDR が不安定であった。
その原因候補を切り分けるため、以下を順に確認した。

1. `fft_size` を増やせば高域不安定が解消するか
2. interferer を消した target-only 条件でも高域で自己歪みが出るか
3. array spacing を空間エイリアシング条件に合わせて詰めると改善するか

この確認用として、

- `evaluations/beamforming/scene_renderer_mvdr_stability_sweep.py`

を追加した。

このスクリプトでは以下を可変にした。

- `fs`
- `fft_size`
- `frequency list`
- `spacing_m`
- interferer の有無
- `diag_load`
- `integration_time`

## 26.2 0-10000 Hz を論じる前提条件

まず重要なのは、従来条件

- `fs = 16000 Hz`

では Nyquist が `8000 Hz` であるため、`10000 Hz` は評価帯域外であることである。

したがって `0-10000 Hz` を論じるには、少なくとも

- `fs >= 20000 Hz`

が必要であり、今回の切り分けでは

- `fs = 32000 Hz`

を用いた。

また、ULA の空間エイリアシング限界は

    f_alias = c / (2 * d)

であり、`c = 343 m/s`, `d = 0.04 m` のとき

    f_alias = 343 / (2 * 0.04)
            = 4287.5 Hz

である。

したがって、現行の `spacing = 0.04 m` では、
理想的に見ても `4.29 kHz` を超える帯域で空間エイリアシングが避けられない。

`10000 Hz` まで空間エイリアシングを避けるには

    d <= c / (2 * 10000)
      <= 0.01715 m

が必要である。

## 26.3 `fft_size` を増やしただけでは改善しない

`fs = 32000 Hz`, `spacing = 0.04 m` のまま、
`fft_size = 64, 128, 256` で sweep した。

代表結果は以下である。

| spacing [m] | fft_size | freq [Hz] | alias_limit [Hz] | cbf_rms_err | mvdr_rms_err | 改善有無 |
|---:|---:|---:|---:|---:|---:|:---|
| 0.040 | 64 | 4000 | 4287.5 | `9.711e-01` | `9.665e-01` | yes |
| 0.040 | 64 | 6000 | 4287.5 | `9.356e-01` | `9.851e-01` | no |
| 0.040 | 64 | 8000 | 4287.5 | `8.871e-01` | `9.917e-01` | no |
| 0.040 | 64 | 10000 | 4287.5 | `8.268e-01` | `9.947e-01` | no |
| 0.040 | 256 | 4000 | 4287.5 | `9.711e-01` | `9.665e-01` | yes |
| 0.040 | 256 | 6000 | 4287.5 | `9.356e-01` | `9.851e-01` | no |
| 0.040 | 256 | 8000 | 4287.5 | `8.871e-01` | `9.917e-01` | no |
| 0.040 | 256 | 10000 | 4287.5 | `8.268e-01` | `9.947e-01` | no |

この結果から、`fft_size` を 64 から 256 へ増やしても、
`4.29 kHz` を超える帯域での不安定性は本質的には改善しない。

したがって、主因は `fft_size` の小ささだけではない。

## 26.4 interferer を消すと自己歪みは出ない

同じ `spacing = 0.04 m`, `fs = 32000 Hz` で、
interferer なしの target-only 条件を確認した。

代表結果は以下である。

| spacing [m] | fft_size | freq [Hz] | cbf_rms_err | mvdr_rms_err | max_reanalysis_err |
|---:|---:|---:|---:|---:|---:|
| 0.040 | 64 | 6000 | `3.884e-08` | `3.793e-08` | `7.134e-15` |
| 0.040 | 64 | 8000 | `1.042e-08` | `1.042e-08` | `0.000e+00` |
| 0.040 | 64 | 10000 | `3.160e-08` | `2.702e-08` | `7.448e-15` |
| 0.040 | 256 | 6000 | `3.884e-08` | `3.793e-08` | `1.502e-14` |
| 0.040 | 256 | 8000 | `1.042e-08` | `1.042e-08` | `0.000e+00` |
| 0.040 | 256 | 10000 | `3.160e-08` | `2.702e-08` | `2.856e-14` |

したがって、現状の MVDR 実装は

- distortionless constraint 自体は高域でも保てている
- 高域不安定は self-distortion 単独ではない

と判断できる。

主問題は、interferer を含む共分散条件にある。

## 26.5 spacing を詰めると 10 kHz まで改善する

次に、空間エイリアシング条件を満たすよう spacing を詰めた。

### spacing = 0.017 m

このとき

    f_alias = 343 / (2 * 0.017)
            = 10088.2 Hz

であり、`10 kHz` までほぼ条件を満たす。

代表結果は以下である。

| spacing [m] | fft_size | freq [Hz] | cbf_rms_err | mvdr_rms_err | 改善有無 |
|---:|---:|---:|---:|---:|:---|
| 0.017 | 64 | 6000 | `9.882e-01` | `9.192e-01` | yes |
| 0.017 | 64 | 8000 | `9.791e-01` | `9.538e-01` | yes |
| 0.017 | 64 | 10000 | `9.674e-01` | `9.703e-01` | no |
| 0.017 | 256 | 6000 | `9.882e-01` | `9.192e-01` | yes |
| 0.017 | 256 | 8000 | `9.791e-01` | `9.538e-01` | yes |
| 0.017 | 256 | 10000 | `9.674e-01` | `9.703e-01` | no |

`10 kHz` では境界ぎりぎりのため、改善は不十分である。

### spacing = 0.015 m

このとき

    f_alias = 343 / (2 * 0.015)
            = 11433.3 Hz

であり、`10 kHz` に十分余裕がある。

代表結果は以下である。

| spacing [m] | fft_size | freq [Hz] | cbf_rms_err | mvdr_rms_err | 改善有無 |
|---:|---:|---:|---:|---:|:---|
| 0.015 | 64 | 6000 | `9.908e-01` | `8.973e-01` | yes |
| 0.015 | 64 | 8000 | `9.837e-01` | `9.410e-01` | yes |
| 0.015 | 64 | 10000 | `9.746e-01` | `9.619e-01` | yes |
| 0.015 | 256 | 6000 | `9.908e-01` | `8.973e-01` | yes |
| 0.015 | 256 | 8000 | `9.837e-01` | `9.410e-01` | yes |
| 0.015 | 256 | 10000 | `9.746e-01` | `9.619e-01` | yes |

したがって、`0-10000 Hz` の安定化に対して支配的なのは、
今回の切り分け範囲では `fft_size` よりも `spacing` である。

## 26.6 現段階の判断

今回の切り分けから、以下を結論とする。

- `fs = 16000 Hz` のままでは `10000 Hz` を論じられない
- `spacing = 0.04 m` では、空間エイリアシング限界が `4287.5 Hz` のため、`0-10000 Hz` で安定な MVDR を期待するのは無理がある
- `fft_size` を増やすだけでは、この問題は解消しない
- interferer なしでは高域自己歪みはほぼ出ないため、主因は beamformer の distortionless constraint 崩れではない
- interferer を含む条件で高域が崩れる主因は、まず array spacing に起因する空間エイリアシングである

したがって、少なくとも今回の `4 ch` ULA 条件では、
`0-10000 Hz` を安定に扱いたいなら

- `fs >= 32000 Hz`
- `spacing <= 0.015 m` 程度

が必要である。

## 26.7 次に進めるべき内容

現段階で、`0-10000 Hz` 安定化に向けた優先順位は以下である。

1. 評価条件を `fs = 32000 Hz`, `spacing = 0.015 m` へ変更して基準化する
2. その条件で prototype 付き本命 filter bank 上の MVDR を再評価する
3. 必要なら channel 数を増やし、高域での null 深さを改善する
4. 必要なら diagonal loading や更新則を高域側で帯域依存に最適化する

少なくとも、現行 `spacing = 0.04 m` のまま
`0-10000 Hz` 全帯域安定を目標にするのは、方式改善だけでは難しい。
まず array 条件の見直しが必要である。

------------------------------------------------------------------------

# 27. fs=32768 Hz, nCh=32, d=c/fs 条件での再評価

## 27.1 評価条件の変更

`0-10000 Hz` を妥当に論じるため、評価条件を以下へ変更した。

- sampling rate: `fs = 32768 Hz`
- channel count: `nCh = 32`
- sound speed: `c = 343 m/s`
- design frequency: `fd = fs / 2 = 16384 Hz`
- receiver spacing:

      d = c / fd * 0.5
        = c / fs
        = 343 / 32768
        = 0.010467529296875 m

このとき空間エイリアシング限界は

    c / (2 * d) = fs / 2 = 16384 Hz

となるため、`0-10000 Hz` は空間サンプリング条件の内側に入る。

この条件に合わせて、
`evaluations/beamforming/scene_renderer_mvdr_stability_sweep.py`
の default を以下へ変更した。

- `fs = 32768 Hz`
- `n_ch = 32`
- `spacing_m = c / fs` を自動計算
- `fft_size = 64, 128, 256`

## 27.2 interferer あり sweep 結果

`diag_load = 1e-3` のまま sweep すると、代表結果は以下であった。

| fft_size | freq [Hz] | cbf_rms_err | mvdr_rms_err | 改善有無 |
|---:|---:|---:|---:|:---|
| 64 | 100 | `9.484e-01` | `6.729e-01` | yes |
| 64 | 500 | `9.957e-01` | `9.705e-01` | yes |
| 64 | 1000 | `9.831e-01` | `9.939e-01` | no |
| 64 | 2000 | `9.346e-01` | `9.986e-01` | no |
| 64 | 4000 | `7.734e-01` | `9.997e-01` | no |
| 64 | 6000 | `6.086e-01` | `9.998e-01` | no |
| 64 | 8000 | `5.269e-01` | `9.992e-01` | no |
| 64 | 10000 | `5.015e-01` | `9.997e-01` | no |

`fft_size = 128, 256` でも本質は変わらず、
全帯域安定には至らなかった。

したがって、空間エイリアシングを外しても、
現状の MVDR 実装には別の支配要因が残っている。

## 27.3 interferer なし sweep 結果

同条件で interferer を除いた target-only 条件を確認したところ、
むしろ MVDR の target 参照誤差は大きかった。

代表結果は以下である。

| fft_size | freq [Hz] | cbf_rms_err | mvdr_rms_err |
|---:|---:|---:|---:|
| 64 | 500 | `2.910e-02` | `9.020e-01` |
| 64 | 1000 | `5.804e-02` | `9.794e-01` |
| 64 | 4000 | `2.197e-01` | `9.989e-01` |
| 64 | 8000 | `3.654e-01` | `9.998e-01` |
| 64 | 10000 | `3.937e-01` | `9.999e-01` |

これは、現行実装では target を含む共分散 `Rxx` からそのまま MVDR 重みを作っているため、
strong desired signal 条件で self-nulling が起きていることを示す。

したがって、現状の不安定性は

- 空間エイリアシングの問題だけではない
- `target-in-covariance` による self-nulling が強く支配している

と判断する。

## 27.4 diagonal loading の影響

`fft_size = 64` で `diag_load` を増やすと、周波数帯によっては改善した。
例えば以下の傾向を確認した。

- `diag_load = 0.1`
  - `500`, `1000`, `2000 Hz` では改善
  - `4000 Hz` 以上では不十分
- `diag_load = 1.0`
  - `4000 Hz` まで改善
  - `8000`, `10000 Hz` はなお不十分
- `diag_load = 10.0`
  - `4000 Hz` 付近ではさらに改善
  - `8000`, `10000 Hz` はなお CBF を下回る

したがって、diagonal loading だけでは `0-10000 Hz` 全帯域安定化は達成できなかった。

## 27.5 現段階の判断

今回の評価条件変更により、
少なくとも以下は確認できた。

- `fs = 32768 Hz`, `nCh = 32`, `d = c / fs` により、空間エイリアシング条件は `0-10000 Hz` で満たせる
- にもかかわらず現行 MVDR は全帯域で安定しない
- 主因は `target` を含む共分散推定に起因する self-nulling である可能性が高い
- diagonal loading は部分的に有効だが、単独では不十分

したがって次に必要なのは、array 条件ではなく MVDR の共分散設計見直しである。
具体的には以下が候補である。

1. target を含まない学習区間から `Rxx` を推定する
2. oracle 評価として interferer+noise のみで `Rxx` を作り、上限性能を確認する
3. blocking matrix / GSC 形へ切り替えて desired signal leakage を抑える
4. diagonal loading を帯域依存または SNR 依存にする

少なくとも、評価条件の妥当化だけでは `0-10000 Hz` 全帯域安定化は完了しない。
次段階は MVDR の `Rxx` 設計を変える必要がある。

------------------------------------------------------------------------

# 28. アレイ設計から見た期待しやすい帯域の整理

## 28.1 基本方針

アレイで期待しやすい帯域は、主に以下の 2 つで決まる。

- 受波器間隔 `d`
- アレイ開口長 `L ≒ (nCh - 1) d`

役割は分かれている。

- `d` は高周波側の上限を決める
- `L` は低周波側の分解能を決める

したがって、アレイ設計だけで見ても

- 高周波で破綻しやすい帯域
- 低周波で分離しにくい帯域

が存在する。

## 28.2 高周波側の設計指標

空間エイリアシングを避ける条件は

    f <= c / (2 d)

である。

この上限を

    f_alias = c / (2 d)

と置く。

`f > f_alias` では、グレーティングローブが発生し、
CBF / MVDR ともに設計どおりの性能を期待しにくい。

したがって、少なくとも

- `f_alias` は運用上の上限周波数以上

に設計する必要がある。

今回用いた設計式

    d = c / fs

では

    f_alias = c / (2 d)
            = fs / 2

となるため、アレイ側の空間サンプリング上限を
Nyquist と一致させる設計になっている。

## 28.3 低周波側の設計指標

低周波では空間エイリアシングは起きにくいが、
波長

    lambda = c / f

が長くなるため、有限の開口長では方位差が位相差として現れにくい。

このため、低周波側では

- 主ローブが太くなる
- 近接方位の分離が難しくなる
- null が浅くなる

という問題が出る。

開口長 `L` に対して何波長含むかを

    A_lambda = L / lambda = L f / c

と置くと、
`A_lambda` が十分大きいほど低周波でも分解しやすい。

設計目安としては、以下のように整理できる。

- `A_lambda << 1`
  - 開口が 1 波長よりかなり短い
  - 低周波では方位分解がほぼ期待できない
- `A_lambda ≈ 1`
  - やっと方位差が出始める帯域
  - CBF は効き始めるが、鋭い null は期待しにくい
- `A_lambda > 2` から `3`
  - 分解能と null 形成が期待しやすい帯域

この閾値は厳密な理論境界ではなく、
実用上の経験則として扱う。

## 28.4 期待しやすい帯域の設計式

以上をまとめると、アレイ設計だけから見た期待しやすい帯域は、
概ね以下で挟まれる。

高周波側上限:

    f_high <= c / (2 d)

低周波側目安:

    f_low ≈ k c / L

ここで `k` は必要な分解能に応じた係数であり、
経験的には以下を目安にできる。

- `k ≈ 1`
  - 最低限、効き始める帯域
- `k ≈ 2` から `3`
  - 安定に性能を期待しやすい帯域

したがって、設計指標としては

    k c / L <= f <= c / (2 d)

を、期待しやすい帯域の近似的な目安とみなせる。

## 28.5 チャネル数の効果

`nCh` を増やすと、同じ `d` でも開口長 `L` が伸びるため、
低周波側の分解能は改善する。

一方で、`d` を変えない限り

    f_alias = c / (2 d)

は変わらないため、高周波側上限は改善しない。

つまり

- `nCh` は主に低周波側の改善に効く
- `d` は主に高周波側の改善に効く

と整理できる。

## 28.6 狭間隔アレイと広間隔アレイ

固定 `nCh` の下では、以下のトレードオフがある。

### 狭間隔アレイ

- `d` が小さい
- `f_alias` は高くなり、高周波に有利
- 同じ `nCh` なら `L` は小さくなり、低周波分解能は悪化する

### 広間隔アレイ

- `d` が大きい
- `L` は大きくなり、低周波分解能は改善する
- `f_alias` は低くなり、高周波上限は下がる

したがって、

- 高周波重視なら狭間隔
- 低周波重視なら大開口

が必要であり、広帯域で両立したいなら

- `d` を小さくした上で
- `nCh` を増やして `L` も確保する

設計が必要になる。

## 28.7 今回の条件への当てはめ

### 条件 A: `nCh = 4`, `d = 0.04 m`

このとき

    L = (4 - 1) * 0.04 = 0.12 m
    f_alias = 343 / (2 * 0.04) = 4287.5 Hz

である。

低周波側の効き始め目安を `k = 1`、
安定に期待しやすい帯域を `k = 2` から `3` とすると

    f_low_start ≈ 343 / 0.12 = 2858 Hz
    f_low_good  ≈ 2*343 / 0.12 = 5717 Hz
    f_low_good  ≈ 3*343 / 0.12 = 8575 Hz

となる。

ただし高周波上限は `4287.5 Hz` なので、
この条件では

- 高周波側上限が低い
- 低周波で十分な分解能も得にくい

という厳しい条件になる。

### 条件 B: `nCh = 32`, `d = c / fs`, `fs = 32768 Hz`

このとき

    d = 343 / 32768 = 0.010467529 m
    L = 31 * d ≈ 0.324493 m
    f_alias = 16384 Hz

である。

同様に見ると

    f_low_start ≈ 343 / 0.324493 ≈ 1057 Hz
    f_low_good  ≈ 2*343 / 0.324493 ≈ 2114 Hz
    f_low_good  ≈ 3*343 / 0.324493 ≈ 3171 Hz

となる。

したがってこの条件では

- 高周波側は `10 kHz` まで空間エイリアシング上は扱える
- 低周波側は `1 kHz` 近傍から効き始め、`2-3 kHz` 以上で分解能を期待しやすい

と整理できる。

## 28.8 実務上の設計指標

実務上は、まず目標帯域 `[f_min, f_max]` を決め、
以下を満たすか確認するとよい。

1. 高周波条件

       d <= c / (2 f_max)

2. 低周波条件

       L >= k c / f_min

ここで `k` は要求性能に応じて選ぶ。

- おおまかな検出でよければ `k ≈ 1`
- 安定した分離や null 性能まで欲しければ `k ≈ 2` から `3`

また `L = (nCh - 1) d` なので、
結局は

- `d` を小さくして高周波を守る
- `nCh` を増やして低周波分解能を補う

ことが広帯域アレイ設計の基本になる。

## 28.9 今回の評価結果との関係

今回観測した挙動は、この設計指標と整合する。

- `d = 0.04 m` 条件で高域が崩れたのは、`f_alias = 4287.5 Hz` を超えていたため
- `d = c / fs`, `nCh = 32` にすると高周波上限の問題は大きく改善した
- その上で残った劣化は、アレイ設計だけではなく `Rxx` 推定や self-nulling の問題である

したがって、今後の評価では

- まずアレイ設計指標で対象帯域に入っているか確認する
- その後に MVDR 実装や `Rxx` 推定の妥当性を議論する

という順序が妥当である。

------------------------------------------------------------------------

# 29. 性能が出にくかった帯域に対する d 変更効果の整理

## 29.1 目的

第 28 章で整理したアレイ設計指標の観点から、
以前に性能が出にくかった帯域について

- `d` を変えると改善するのか
- 改善しないなら、アレイ以外の要因か

を確認した。

評価は以下の 2 条件で比較した。

- `covariance_source = mixture`
  - target を含む共分散
- `covariance_source = interferer-only`
  - self-nulling を避ける oracle 学習

共通条件は以下である。

- `fs = 32768 Hz`
- `nCh = 32`
- `fft_size = 64`
- `freq = 1000, 4000, 8000, 10000 Hz`
- `n_samples = 65536`
- target: `20 deg`
- interferer: `-30 deg`
- target/interferer level: `0 dB RMS`

比較した `d` は以下である。

- `0.04 m`
- `0.017 m`
- `0.015 m`
- `c / fs = 0.010467529 m`

## 29.2 `covariance_source = mixture` の結果

`Rxx` に target を含めた場合、代表結果は以下であった。

| d [m] | alias_limit [Hz] | freq [Hz] | cbf_rms_err | mvdr_rms_err | 改善有無 |
|---:|---:|---:|---:|---:|:---|
| 0.040000 | 4287.5 | 1000 | `7.929e-01` | `9.997e-01` | no |
| 0.017000 | 10088.2 | 1000 | `9.590e-01` | `9.985e-01` | no |
| 0.015000 | 11433.3 | 1000 | `9.675e-01` | `9.975e-01` | no |
| 0.010468 | 16384.0 | 1000 | `9.831e-01` | `9.923e-01` | no |
| 0.040000 | 4287.5 | 4000 | `4.594e-01` | `1.000e+00` | no |
| 0.017000 | 10088.2 | 4000 | `5.171e-01` | `9.999e-01` | no |
| 0.015000 | 11433.3 | 4000 | `5.984e-01` | `9.999e-01` | no |
| 0.010468 | 16384.0 | 4000 | `7.734e-01` | `9.997e-01` | no |
| 0.040000 | 4287.5 | 8000 | `7.841e-01` | `1.000e+00` | no |
| 0.017000 | 10088.2 | 8000 | `4.746e-01` | `9.999e-01` | no |
| 0.015000 | 11433.3 | 8000 | `4.596e-01` | `9.998e-01` | no |
| 0.010468 | 16384.0 | 8000 | `5.269e-01` | `9.992e-01` | no |
| 0.040000 | 4287.5 | 10000 | `9.565e-01` | `1.000e+00` | no |
| 0.017000 | 10088.2 | 10000 | `4.690e-01` | `9.999e-01` | no |
| 0.015000 | 11433.3 | 10000 | `4.690e-01` | `9.999e-01` | no |
| 0.010468 | 16384.0 | 10000 | `5.015e-01` | `9.997e-01` | no |

この結果から、`mixture` 学習では

- `d` を変えても MVDR の target 誤差はほぼ改善しない
- alias limit を十分上げても、自己抑圧が支配的である

と分かる。

したがって、この条件で性能が出ない主因はアレイ設計ではなく、
`target-in-covariance` による self-nulling である。

## 29.3 `covariance_source = interferer-only` の結果

次に、self-nulling を避けるため `Rxx` を interferer-only で作った。
代表結果は以下である。

| d [m] | alias_limit [Hz] | freq [Hz] | cbf_rms_err | mvdr_rms_err | 改善有無 |
|---:|---:|---:|---:|---:|:---|
| 0.040000 | 4287.5 | 1000 | `7.929e-01` | `2.272e-01` | yes |
| 0.017000 | 10088.2 | 1000 | `9.590e-01` | `2.451e-01` | yes |
| 0.015000 | 11433.3 | 1000 | `9.675e-01` | `2.459e-01` | yes |
| 0.010468 | 16384.0 | 1000 | `9.831e-01` | `2.457e-01` | yes |
| 0.040000 | 4287.5 | 4000 | `4.594e-01` | `4.247e-01` | yes |
| 0.017000 | 10088.2 | 4000 | `5.171e-01` | `7.848e-01` | no |
| 0.015000 | 11433.3 | 4000 | `5.984e-01` | `8.159e-01` | no |
| 0.010468 | 16384.0 | 4000 | `7.734e-01` | `8.775e-01` | no |
| 0.040000 | 4287.5 | 8000 | `7.841e-01` | `7.992e-01` | no |
| 0.017000 | 10088.2 | 8000 | `4.746e-01` | `5.151e-01` | no |
| 0.015000 | 11433.3 | 8000 | `4.596e-01` | `5.674e-01` | no |
| 0.010468 | 16384.0 | 8000 | `5.269e-01` | `7.254e-01` | no |
| 0.040000 | 4287.5 | 10000 | `9.565e-01` | `8.475e-01` | yes |
| 0.017000 | 10088.2 | 10000 | `4.690e-01` | `5.544e-01` | no |
| 0.015000 | 11433.3 | 10000 | `4.690e-01` | `5.661e-01` | no |
| 0.010468 | 16384.0 | 10000 | `5.015e-01` | `5.281e-01` | no |

## 29.4 原因整理

この比較から、性能が出ない要因は帯域によって分けて考えるべきである。

### 1. `1000 Hz` 付近

- `mixture` 学習では性能が出ない
- `interferer-only` 学習では全ての `d` で大きく改善する

したがって主因はアレイ設計ではなく、
`Rxx` に target が混入することによる self-nulling である。

### 2. `4000 Hz` 付近

- `d = 0.04 m` では oracle 条件で改善する
- `d` を狭くすると逆に改善しなくなる

これは、`d` を狭くしたことで高周波上限は伸びる一方、
同じ `nCh` では開口長 `L` が縮み、
`4000 Hz` 付近では低周波側分解能が不足し始めるためと解釈できる。

つまりこの帯域では

- 高周波上限だけでなく
- 開口長による分解能低下

も効いている。

### 3. `8000 Hz` 付近

- `d = 0.04 m` では明らかに alias limit 超過である
- `d` を狭めても oracle 条件で十分改善しない

したがって、ここでは

- `d = 0.04 m` 条件の不良には空間エイリアシングが含まれる
- しかし `d` を改善した後もなお、
  現行 32ch / 64点 block DFT / 単一 target-interferer 条件では
  CBF を大きく上回る安定性能は得られていない

と整理できる。

つまり、高域側ではアレイ設計改善は必要条件だが十分条件ではない。

### 4. `10000 Hz` 付近

- `d = 0.04 m` では alias limit 超過であり、アレイ設計不適
- `d = 0.017, 0.015, c/fs` にしても oracle 条件で安定改善しない

この帯域では

- 元の `d = 0.04 m` ではアレイ設計起因が明確
- ただし `d` を直した後も、
  開口、bandwidth、block DFT 近似、有限サンプル `Rxx` などの複合要因で
  期待した MVDR 利得がまだ出ていない

と考えるのが妥当である。

## 29.5 結論

`d` を変えても性能が出ないか、という問いへの答えは以下である。

- `1000 Hz` では、`d` を変えても本質は変わらず、主因は self-nulling
- `4000 Hz` では、`d` と `L` のトレードオフが効いており、単純に狭くすれば良いわけではない
- `8000 Hz`, `10000 Hz` では、元の `d = 0.04 m` は不適切だが、`d` 改善だけでは十分でない

したがって、性能が出ない原因は単一ではなく、少なくとも以下に分離される。

1. `Rxx` に target が入ることによる self-nulling
2. `d` による高周波上限不足
3. `L` による分解能不足
4. baseline block DFT bank と有限長 `Rxx` 推定の近似誤差

今後は帯域ごとに、どの要因が支配的かを切り分けて扱う必要がある。
特に

- `1 kHz` 近傍は `Rxx` 設計優先
- `4 kHz` 近傍は開口長と分解能の整理優先
- `8-10 kHz` はアレイ条件を満たした上で、なお残る高域側の MVDR 実装要因を評価

という進め方が妥当である。

------------------------------------------------------------------------

# 30. 指定した Rxx 学習方式での評価結果整理

## 30.1 学習方式

本章では、共分散行列 `Rxx` の学習方式を、指定された MATLAB 断片と同じ意味に揃えた。

方式は以下である。

    for block = 1 : ceil(rate * T)
        for fIdx = 1 : nBand
            x_f = X(:, fIdx, block) / nFFT
            cov = x_f * x_f^H
            Rxx(:, :, fIdx) = (1 - coef) * Rxx(:, :, fIdx) + coef * cov
        end
    end

ここで

- `coef` は忘却係数
- `rate` は共分散更新レート = 周波数分析幅
- `T` は積分時間
- 各 `fIdx` ごとに `nCh x nCh` の空間相関行列を得る

現在の実装では、これを

- `CovarianceEstimator.process_snapshot()`
- `integrate_band_covariances()`

として部品化した。

また積分時間は、経験則

    T * rate = 2 * nCh

を default とし、

    T = 2 * nCh / rate

で設定するようにした。

## 30.2 mixture 学習での結果

まず `Rxx` を target を含む mixture から学習した。

`fs = 32768 Hz`, `nCh = 32`, `fft_size = 64` の条件で、
代表結果は以下であった。

| freq [Hz] | cbf_rms_err | mvdr_rms_err | 改善有無 |
|---:|---:|---:|:---|
| 500 | `9.957e-01` | `9.815e-01` | yes |
| 1000 | `9.831e-01` | `9.923e-01` | no |
| 2000 | `9.346e-01` | `9.985e-01` | no |
| 4000 | `7.734e-01` | `9.997e-01` | no |
| 8000 | `5.269e-01` | `9.992e-01` | no |
| 10000 | `5.015e-01` | `9.997e-01` | no |

この結果から、指定方式のまま `Rxx = mixture` とすると、
低中域から高域まで self-nulling が強く出ることを確認した。

## 30.3 interferer-only 学習での結果

次に、同じ学習式のまま、`Rxx` の学習信号だけを
interferer-only に切り替えた oracle 評価を実施した。

代表結果は以下である。

| freq [Hz] | cbf_rms_err | mvdr_rms_err | 改善有無 |
|---:|---:|---:|:---|
| 500 | `9.957e-01` | `2.453e-01` | yes |
| 1000 | `9.831e-01` | `2.457e-01` | yes |
| 2000 | `9.346e-01` | `2.989e-01` | yes |
| 4000 | `7.734e-01` | `8.775e-01` | no |
| 8000 | `5.269e-01` | `7.254e-01` | no |
| 10000 | `5.015e-01` | `5.281e-01` | no |

したがって、指定した `Rxx` 学習方式そのものは成立するが、

- low/mid band では `Rxx` から target を外すだけで大きく改善する
- high band では `Rxx` を改善してもなお残る要因がある

と整理できる。

## 30.4 指定方式で分かったこと

この結果から、指定した `Rxx` 学習方式での失敗要因は
少なくとも以下に分けられる。

1. target を含む共分散学習による self-nulling
2. アレイ設計に起因する帯域依存の分解能不足
3. 高域側での有限 block / finite sample / baseline block DFT 近似の影響

特に `1000 Hz` 近傍では、主因は明らかに 1 である。
一方 `4000 Hz` 以上では、1 を避けても 2, 3 の影響が残る。

------------------------------------------------------------------------

# 31. 帯域ごとにアレイ開口を変える設計の考え方

## 31.1 問題意識

これまでの評価から、固定アレイでは

- 高周波では受波器間隔 `d` が効く
- 低周波では開口長 `L` が効く

というトレードオフが明確になった。

したがって、広帯域で性能を出したい場合、
全帯域で同じ有効アレイを使うのではなく、
周波数に応じて使用するアレイ開口を変える考え方が自然である。

## 31.2 望ましい方向性

理想的には、周波数ごとに使用チャネル範囲を変える。

- 低周波ほどアレイ端まで使い、開口長を最大化する
- 高周波ほどアレイ中央側だけを使い、局所的には狭間隔のサブアレイとして使う

これにより、

- 低周波では大開口による分解能を確保
- 高周波では狭間隔による alias 抑制を確保

できる。

## 31.3 望ましいアレイ形状

この考え方に立つと、配列は単純な等間隔 ULA よりも、
周波数依存のサブアレイ運用を前提にした形が望ましい。

要求は以下である。

- アレイ中央は受波器間隔が狭い
  - 高周波帯の運用に使う
- アレイ端に向かうほど有効開口が広がる
  - 低周波帯の運用に使う
- 低周波向けには端側まで含めた広開口サブアレイ
- 高周波向けには中央の高密度サブアレイ

つまり、実務的には

- dense inner aperture
- sparse outer aperture

を持つ多重開口アレイの考え方になる。

## 31.4 現在の評価結果との関係

今回の固定 `d` 評価で見えたことは、まさにこの必要性である。

- `d` を狭くすると高周波上限は改善する
- しかし固定 `nCh` では開口 `L` が縮み、低中域の分解能が悪化する
- `d` を広くすると low band は有利になりうるが、高域で alias が出る

したがって、固定アレイ 1 本で全帯域最適を狙うより、
帯域ごとに有効サブアレイを切り替える方が合理的である。

## 31.5 設計指針

今後のアレイ設計の方向性としては、以下を推奨する。

1. 高域上限 `f_max` を満たすため、中央サブアレイの間隔は

       d_inner <= c / (2 f_max)

   とする。

2. 低域下限 `f_min` で十分な分解能を得るため、外側まで含めた開口長は

       L_outer >= k c / f_min

   を満たすようにする。

3. beamforming 時には、周波数ごとに使用チャネル集合を変える。

   例えば

   - low band: outer + inner 全チャネル
   - mid band: 中央寄りの大部分
   - high band: inner dense array のみ

   とする。

## 31.6 実装上の意味

この方向へ進む場合、beamformer 実装は

- 全チャネル固定の steering vector

ではなく、band ごとに

- 使用チャネル集合
- 使用アレイ開口
- 対応する steering vector
- 対応する `Rxx`

を切り替える必要がある。

したがって、将来の実装としては

- band-wise channel selector
- band-wise steering builder
- band-wise covariance learner

を持つ構造へ拡張するのが自然である。

## 31.7 現段階での結論

固定 ULA のまま性能を評価するだけでは、
「方式の問題」と「アレイ設計の問題」が混ざりやすい。

今後は以下の 2 段階で整理するのが妥当である。

1. 現行固定アレイ上で `Rxx` 学習方式や self-nulling を切り分ける
2. 並行して、帯域ごとに有効開口を変える multi-aperture / nested な設計指針を導入する

つまり、あなたが述べた

- 低周波では端まで使う
- 高周波では中央だけ使う
- 中央は狭間隔、端側は広間隔

という方向性は、広帯域 beamforming の設計思想として妥当である。


------------------------------------------------------------------------

# 32. 大開口・多チャネル nested array による切り分け結果

## 32.1 目的

小アレイ条件で MVDR / CBF が不安定になるとき、
それが方式要因なのかアレイ設計要因なのかが混ざりやすい。

そこで、低・中・高の 3 狭帯域信号を同時に入れた 1 パターンだけを用い、
以下の 2 条件を比較した。

- 小アレイ: `32 ch`, 等間隔 `0.04 m`, full 使用
- 大アレイ: `256 ch`, outer spacing `5.0 m`, dense center `64 ch @ 0.01 m`, nested-progressive 使用

MVDR の共分散は `interferer-only` 学習とし、
self-nulling の影響を避けた上で純粋にアレイ設計の影響を見た。

## 32.2 試験信号

同時に入れた target 信号周波数は以下である。

- low: `100 Hz`
- mid: `2000 Hz`
- high: `8000 Hz`

interferer も同じ 3 周波数を含む multitone とした。

## 32.3 大アレイ条件で実際に使われたサブアレイ

大アレイ nested-progressive 条件では、ビンごとの `used_channels` は実際に以下となった。

- `100 Hz` 付近: `116 ch`, 開口 `260.63 m`, 最小間隔 `0.01 m`
- `2000 Hz` 付近: `66 ch`, 開口 `10.63 m`, 最小間隔 `0.01 m`
- `8000 Hz` 付近: `64 ch`, 開口 `0.63 m`, 最小間隔 `0.01 m`

つまり、

- 低域では sparse outer を含めた大開口
- 高域では dense inner を主体とした小開口

という intended な multi-aperture 動作になっている。

## 32.4 結果の要約

代表結果は以下であった。

- 小アレイ full
  - `total_cbf_rms_err = 1.146716e+00`
  - `total_mvdr_rms_err = 1.216789e+00`
- 大アレイ nested-progressive
  - `total_cbf_rms_err = 8.655137e-01`
  - `total_mvdr_rms_err = 9.150783e-01`

この 1 パターンでは、
大アレイ化により CBF / MVDR ともに全体誤差が明確に改善した。

特に `100 Hz` で `260.63 m` 級の有効開口を使えたことから、
低域側では小アレイ条件の劣化要因として
「開口不足」が支配的であることを確認できた。

## 32.5 今後の整理方針

以上より、今後小アレイ条件で同様の性能劣化が出た場合は、
少なくとも以下のように整理してよい。

1. 低域から中域の劣化
   - まずアレイ設計要因を疑う
   - 特に開口不足、使用チャネル数不足、帯域に対する有効サブアレイ不適合を確認する

2. 大アレイ化しても残る劣化
   - `Rxx` 学習方式
   - target 混入による self-nulling
   - finite sample / finite block の実装要因

したがって、
小アレイ条件だけで MVDR の方式良否を断定するのは不適切であり、
今後同様のエラーが出ても、まずは

- アレイ開口
- チャネル数
- ビンごとの `used_channels`

の観点からアレイ設計要因として整理する。

## 32.6 実装上の取り扱い

実運用では、アレイ受波器配置とシェーディング係数は
別途設計済みの値を使う前提とする。

そのため、実装側には外部 `np.ndarray` を直接受け取る入口を用意した。

- `BandwiseArrayDesign.from_ndarrays(channel_positions_m, shading_table)`
- `BandwiseArrayDesign.from_channel_positions_and_shading_table(channel_positions_m, shading_table)`

ここで

- `channel_positions_m`: shape `(n_ch,)` または `(n_ch, 3)`
- `shading_table`: shape `(n_ch, n_band)`

であり、`shading_table != 0` がビンごとの `used_channels` を表す。

これにより、実際の運用では外部設計済みの

- 受波器配列
- ビンごとのシェーディング係数

をそのまま beamforming 実装へ流し込める。


------------------------------------------------------------------------

# 33. 従来 0.5 Hz beamforming と現行案の処理量比較

## 33.1 目的

短FFTを嫌う理由は効率性であり、
`reuse_filter_fft` / `short_fft` の議論は
「短FFT相当量を作れるか」だけでなく
「本当に処理負荷が下がるか」で評価する必要がある。

そこで、従来設計である

- フィルタバンクなし
- 分析幅 `0.5 Hz`
- 周波数領域 beamforming

と、現行案である

- `PolyphaseDFTFilterBank(fft_size=32)`
- サブバンドごとの overlap-save beamforming
- `frame_size = 2048`, `valid_size = 1024`

を同じ近似式で比較した。

## 33.2 比較前提

共通前提は以下とした。

- sampling rate: `fs = 32768 Hz`
- channel count: `n_ch = 32`
- beam count: `n_beam = 1`
- MVDR重みは数秒に1回更新し、その間は固定重みで内積適用する
- FFT処理量の近似: `Cfft(N) = N log2 N`

従来設計の分析幅 `0.5 Hz` に対応する FFT 長は

    N = fs / 0.5 = 65536

である。

現行案では `fft_size = 32` のため、
サブバンドレートは

    32768 / 32 = 1024 Hz

となる。

また、`valid_size = 1024` なので、
各サブバンドの overlap-save は
概ね `1 frame / second / band` で更新される。

## 33.3 処理量の概算

### 従来設計: 0.5 Hz long FFT beamforming

1 block あたり

- `32 ch x 65536 FFT`
- 各周波数 bin での整相和
- `1 beam x 65536 IFFT`

とみなす。

概算は以下である。

- `32 ch FFT`: `32 * 65536 * 16 = 33.55M`
- beamforming和: `32 * 65536 = 2.10M`
- `1 beam IFFT`: `65536 * 16 = 1.05M`
- 合計: 約 `36.7M / block`

`65536 sample` は `2 second` に相当するが、
従来方式は `50% overlap` で運用するため、処理レートは `1 second` 周期である。

したがって、従来方式の常時処理量は

- 約 `36.7M / second`

となる。

### 現行候補A: short_fft を省いた案

各サブバンド・各秒あたり

- `32 ch x 2048 FFT`
- 各周波数 bin での beamforming 和
- `1 beam x 2048 IFFT`

とみなす。

1 band あたりの概算は以下である。

- `32 ch FFT`: `32 * 2048 * 11 = 0.721M`
- beamforming和: `32 * 2048 = 0.066M`
- `1 beam IFFT`: `2048 * 11 = 0.023M`
- 小計: 約 `0.809M / band / second`

`32 band` 合計では

- overlap-save本体: 約 `25.9M / second`

となる。

さらにフィルタバンク入出力を加える。

- 入力側 `32 ch x 1024 block/s x 32pt FFT`: 約 `5.24M / second`
- 出力側 `1 beam x 1024 block/s x 32pt IFFT`: 約 `0.16M / second`

よって、short_fft を省いた案の全体では

- 約 `31.3M / second`

となる。

### 現行案: short_fft を追加した案

ここでは `Rxx` 推定用の short FFT を

- `32 pt short_fft`
- `hop = 32`（非重複）
- 各 subband の `valid_size = 1024` 区間に対して `32` 回実行

と仮定する。

このとき 1 band あたりの追加処理量は

- `32 * Cfft(32) = 32 * (32 * 5) = 5120`
- すなわち約 `0.00512M / band / second`

である。

`32 band` 合計では

- short_fft追加分: 約 `0.164M / second`

となる。

したがって、short_fft を追加した現行案全体では

- 約 `31.3M + 0.164M = 31.46M / second`

となる。

## 33.4 比較表

| 方式 | 主な処理 | 概算処理量 | 1秒あたり換算 |
|---|---|---:|---:|
| 従来案 50% overlap | `32 x 65536 FFT` + binごとの整相和 + `65536 IFFT` を 1秒ごと | 約 `36.7M / 秒` | 約 `36.7M / 秒` |
| 別案 short_fft省略 | `32 band` それぞれで `32ch x 2048 FFT` + beamforming + `2048 IFFT` + `32pt DFT bank` + `32pt IDFT bank` | 約 `31.3M / 秒` | 約 `31.3M / 秒` |
| 現行案 short_fft追加 | 上記 + `32pt short_fft x 32回 / band / second` | 約 `31.46M / 秒` | 約 `31.46M / 秒` |
| 片側16帯域案 short_fft省略 | `16 band` のみ `32ch x 2048 FFT` + beamforming + `2048 IFFT` を実行し、残り16帯域は位相遅延を考慮した複素共役で再構成。filter bank は 32 band のまま | 約 `18.35M / 秒` | 約 `18.35M / 秒` |
| 片側16帯域案 short_fft追加 | 上記 + `32pt short_fft x 32回 / band / second` を 16 band 分だけ追加 | 約 `18.43M / 秒` | 約 `18.43M / 秒` |

## 33.5 解釈

この比較では、従来案・現行案ともに
「MVDR重み計算そのもの」は数秒に1回の低頻度処理とみなし、
主に常時計算される FFT / IFFT / 内積処理を比較対象とした。

さらに、32帯域のうち正側16帯域だけを実処理し、負側16帯域を
位相遅延を考慮した複素共役で再構成する案も、実入力・実出力に限れば比較対象に含めた。

この比較から、少なくとも現行条件では以下のように整理できる。

1. 従来案は `65536 sample`, `50% overlap` のため、比較すべき基準処理量は約 `36.7M / second` である。
2. この基準に対して、short_fft省略案は約 `31.3M / second`、short_fft追加案は約 `31.46M / second` であり、どちらも従来案より軽い。
3. 片側16帯域案が成立するなら、short_fft省略で約 `18.35M / second`、short_fft追加でも約 `18.43M / second` まで下げられる。
4. したがって、処理量だけを見れば 16帯域処理 + 共役再構成はかなり有力である。
5. ただしこの案は、任意の複素サブバンド列を扱う基準系としては使えず、実入力から実出力へ戻すときにのみ成立する。
6. さらに DC / Nyquist の扱い、PRDFT の帯域順、負側再構成時の位相補正、overlap-save に含まれる遅延との整合を満たす必要がある。
7. したがって、処理量の観点では有望だが、成立条件を満たす専用実装として整理すべきである。

## 33.6 片側16帯域案の成立条件

32帯域のうち 16 帯域だけを実処理して負側を再構成する案は、少なくとも以下の条件では現実的である。

- 入力が実信号であり、最終出力も実信号へ戻すこと
- DC と Nyquist を独立に扱い、共役ペアに含めないこと
- 正側帯域で得た処理後サブバンド `Y[k]` から、負側帯域を `Y[-k] = conj(Y[k])` に相当する形で再構成できること
- ただし実際の PRDFT / overlap-save 実装では、帯域インデックスの並びと有効遅延に応じた位相補正を併用すること
- 負側帯域の steering / weight / overlap-save フィルタが、正側帯域の複素共役対として整合すること

一方で、以下の用途にはそのまま使えない。

- 任意の複素サブバンド列 `Y` を扱う一般基準系
- 正負帯域で非対称な重みや非線形処理を与える場合
- 負側帯域まで独立に `Rxx` や beam weight を設計したい場合

したがって、本案の位置付けは

- 任意複素サブバンド処理の一般形ではない
- 実信号用の beamforming 最適化としては有力

である。

## 33.7 サブバンド内 beamforming 後に単純な複素共役で負側を作れない理由

実信号の周波数スペクトルは、時間領域で見れば

    X[-k] = conj(X[k])

という Hermitian 対称を持つ。

しかし、今回扱っている量は

- 生の time-domain FFT スペクトル

ではなく、

- 解析 filter bank を通した複素サブバンド時系列
- その上で overlap-save framing
- bandwise beamforming
- valid region 抽出

まで入った「処理後の複素サブバンド列」である。

このため、正側サブバンドの処理後出力から負側を単純に `conj()` で作っても、
一般には元の全帯域処理と一致しない。

主な理由は以下である。

1. 生FFTの共役対称と、複素サブバンド列の共役関係は同じではない。

   filter bank 出力 `Y[k, n]` は、単なる bin 値ではなく
   「解析変調された複素包絡」である。
   したがって負側帯域は、正側帯域の単純共役ではなく、
   帯域インデックスの写像と解析位相を含めた対応として扱う必要がある。

2. bandwise beamforming 後の信号には、重みやフィルタの位相が入る。

   正側で使った steering / weight / overlap-save filter が
   負側で厳密に共役対になっていなければ、
   出力 `Y[-k]` は `conj(Y[k])` にならない。
   特に重みを time-domain filter 化した場合は、
   conjugation をどの段階で織り込んだかまで一致させる必要がある。

3. overlap-save は時間位置を持つ処理であり、valid region 抽出が入る。

   今回の実装では

   - frame_size = 2048
   - valid_size = 1024
   - 後半 1024 sample を採用

   であり、各帯域出力には「どの時間位置の有効区間か」という情報が含まれる。
   負側を単純共役で生成すると、この有効遅延と位相進み/遅れの整合が崩れ、
   合成後の時間波形に不連続や位相ずれが出る。

4. DC / Nyquist は共役ペアではない。

   32帯域中の全てが 16 組の完全な共役ペアになるわけではない。
   DC と Nyquist は独立成分として扱う必要があり、
   ここを通常帯域と同じ規則で複素共役生成すると整合しない。

5. PRDFT / FullDFT の帯域順とインデックス対応を明示しないと誤る。

   実装上の帯域順は FFT 順であり、
   数式上の `-k` と配列上の index は自動的には一致しない。
   そのため、単純に「前半を後半へ複素共役コピー」しても、
   対応する負側帯域へ正しく配置されるとは限らない。

6. 任意の複素サブバンド処理を入れた時点で、Hermitian 対称は自動では保存されない。

   線形 beamforming であっても、正側だけ処理して負側を後付け生成するなら、
   その処理全体が「実信号を保つ対称性」を満たすように設計されている必要がある。
   これは単に出力を `conj()` するだけでは保証されない。

以上より、

- 正側16帯域だけを実処理して負側を再構成する案はあり得る
- ただし必要なのは単純な複素共役ではなく、
  帯域写像、位相補正、DC/Nyquist の別扱い、overlap-save 遅延整合まで含んだ専用再構成則である

と整理できる。

逆に言えば、これらの整合条件を明示せずに

- beamforming 後の正側サブバンドをそのまま `conj()` して負側へ置く

だけでは、全帯域処理と一致しないのは自然である。

## 33.8 再構成させるための整合の方法

片側16帯域だけを実処理して負側を再構成したい場合、
必要なのは「正側出力を単に複素共役すること」ではなく、
以下の整合則を明示して適用することである。

### 33.8.1 基本方針

正側の処理後サブバンド列を `Y_pos[k, n]` とすると、
負側は概念的には

    Y_neg[-k, n] = exp(j * phi[k]) * conj(Y_pos[k, n - d[k]])

のような形で再構成する必要がある。

ここで

- `k`: 正側帯域 index
- `n`: サブバンド時間 index
- `d[k]`: 有効遅延に対応する時間シフトまたは位相傾き
- `phi[k]`: 帯域写像、解析変調、overlap-save valid region 位置に起因する位相補正

である。

実装上は、少なくとも以下を固定しなければならない。

- 正側帯域 index と負側帯域 index の対応表
- DC / Nyquist の別扱い規則
- overlap-save valid region の時間原点
- positive/negative で対になる steering / weight / filter の生成規則
- 再構成後に満たすべき Hermitian 対称の定義点

### 33.8.2 実装時の整合項目

1. 帯域写像の整合

   `k -> k_neg` の対応を FFT 順 index で明示する。
   数式の `-k` を配列 index に暗黙対応させない。

2. 位相補正の整合

   正側と負側で解析基底の位相が異なるため、
   再構成時に per-band の複素回転を入れる。

3. 遅延の整合

   overlap-save では後半 valid region だけを採用するため、
   負側生成でも同じ有効時間原点に揃える。

4. filter 化した重みの整合

   steering / CBF weight / MVDR weight を time-domain filter 化したなら、
   負側用フィルタは正側の単純共役ではなく
   「解析系・適用系の convention に整合した共役対」として生成する。

5. 実出力条件の整合

   合成直前の全帯域列が Hermitian 対称になることを確認し、
   時間領域出力の虚部が機械精度レベルまで落ちることを条件とする。

## 33.9 再構成のための設計手法

再構成則を安定に実装するには、以下の順で設計するのがよい。

### 33.9.1 まず全帯域基準系を固定する

最初に、32帯域すべてを明示計算する基準実装を作る。

- 正側も負側も独立に処理する
- overlap-save 後の全帯域出力 `Y_full` を得る
- `synthesis(Y_full)` を参照出力とする

片側16帯域案は、常にこの `Y_full` を参照して設計する。

### 33.9.2 負側再構成則を band-by-band で同定する

各正側帯域 `k` に対し、
対応する負側帯域 `k_neg` を取り、
全帯域基準系から

    Y_full[k_neg, n] ≈ a[k] * conj(Y_full[k, n - d[k]])

となる `a[k]`, `d[k]` を同定する。

ここで `a[k]` は複素位相補正、`d[k]` は遅延補正である。

実装上はまず

- `d[k] = 0` を仮定して複素比 `a[k]` を推定
- それで誤差が残る帯域だけ `d[k]` を探索

と進めると切り分けしやすい。

### 33.9.3 filter 設計規約を positive/negative で対にする

負側を後付け再構成するなら、正側処理に使う filter / weight 生成器自体を
「共役対を作れる規約」で統一する必要がある。

具体的には

- steering vector の定義式
- conjugation を織り込む段階
- overlap-save filter FFT の作り方
- runtime で転置を使うか共役転置を使うか

を positive/negative で一貫させる。

### 33.9.4 DC / Nyquist を独立 branch として設計する

DC と Nyquist は共役ペア再構成に含めず、
個別 branch として別処理する。

これを曖昧にすると、全帯域合成後の虚部残差や境界不連続の原因になる。

## 33.10 再構成則の検証手段

この方式は、単に時間波形がそれらしく見えるだけでは不十分である。
以下の 3 層で検証する必要がある。

### 33.10.1 サブバンド領域での検証

全帯域基準系 `Y_full` と、片側16帯域案から再構成した `Y_recon` を直接比較する。

確認項目:

- `max_abs(Y_recon - Y_full)`
- `rms_abs(Y_recon - Y_full)`
- bandごとの複素比残差
- DC / Nyquist の個別誤差

まずこの比較で誤差が小さくならない限り、時間波形側の議論に進まない。

### 33.10.2 時間波形での検証

`y_full = synthesis(Y_full)` と `y_recon = synthesis(Y_recon)` を比較する。

確認項目:

- `max_abs(y_recon - y_full)`
- `rms_abs(y_recon - y_full)`
- 出力虚部 RMS
- block 境界での `jump_abs`
- 既存参照波形に対する誤差

特に overlap-save を使う以上、
block 境界の不連続が悪化していないことを必ず確認する。

### 33.10.3 条件 sweep による検証

単一周波数 1 点だけでは不十分であり、少なくとも以下を sweep する。

- bin center 周波数
- off-bin 周波数
- DC 近傍
- Nyquist 近傍
- 低域 / 中域 / 高域
- CBF と MVDR の両方

さらに、

- 狭帯域単音
- 複数同時 tone
- noise 混在

でも比較し、再構成則が beamforming 条件に依存して崩れないかを確認する。

### 33.10.4 合格基準

最低限の合格基準は以下である。

- `Y_recon` が `Y_full` に機械精度または十分小さい誤差で一致すること
- `synthesis(Y_recon)` が `synthesis(Y_full)` に一致すること
- 実出力の虚部が十分小さいこと
- boundary jump が full-band 基準系より悪化しないこと

この 4 条件を満たして初めて、
「片側16帯域 + 負側再構成」が方式として成立したと言える。

## 33.11 `reuse_filter_fft` への含意

前節の確認より、`reuse_filter_fft` を単純な long FFT のビン間引きとして実装しても
`short_fft` と同等の時間分解能は得られない。

したがって、短FFTを本当に省略したいなら、評価観点は

- 短FFT相当量を厳密再構成できるか

ではなく、

- `Rxx` 推定に必要な統計量だけを long FFT から低コストに得られるか

である。

厳密再構成は追加変換を要するため、
多くの場合 `short_fft` より軽い保証はない。

よって、処理量の観点で第一候補となるのは依然として

- `reuse_filter_fft` を long FFT ベースの近似 `Rxx` 推定として使う方向

である。
