# EBAE 詳細設計書

## 1. 目的と責務

EBAE（Dominant Mode Rejection with Eigenvector-Beam Association and Excision）は、外部で完成した周波数別センサ間共分散を固有値分解し、Nadakuditi/Edelman AIC（N/E AIC）で信号数を推定する。雑音部分空間から得たMUSIC疑似スペクトルにより信号固有ベクトルと方位を対応付け、待受方位に対応する信号を保護しながら、それ以外の信号固有modeをCBF重みから除外する。

本方式はMVDRの制約付き最小分散問題を解かない。MVDRの置換候補ではあるが、数式、実装、評価はMVDRと独立に扱う。

本設計の責務は次である。

1. 完成済み入力共分散のHermitian固有値分解と降順整列
2. N/E AICによる周波数bin別信号数推定
3. MUSIC疑似スペクトルによる信号固有ベクトルと方位の対応付け
4. sigmoid型固有ベクトル除外係数を使ったEBAE重み設計
5. 完成重みと診断量の固定shapeでの公開
6. 正規化不能時のCBF fallback

FFT、共分散の構成・積分、整数遅延、FIR実現、重み適用、周波数間追跡、BL/FRAZ/BTR評価は責務に含めない。

## 2. 入出力と軸

| 量 | shape | 意味・単位 |
|---|---:|---|
| `R` | `[n_bin,M,M]` | 完成済み空間共分散。axis 0はFFT bin、axis 1,2はactive channel。単位は入力power |
| `A` | `[M,n_beam,n_bin]` | 未正規化steering。axis 0はchannel、axis 1は待受方位、axis 2はFFT bin |
| `W_opt` | `[M,n_beam,n_bin]` | EBAE重み。同じaxis規約 |
| `music` | `[n_beam,n_bin]` | MUSIC線形疑似スペクトル |
| `Ns` | `[n_bin]` | binごとの推定信号数 |

全FFT binは完全に独立して処理する。隣接bin間で信号数、固有ベクトル、対応方位を平滑化または追跡しない。

## 3. 入力共分散の契約とsnapshot数

入力共分散はHermitianな完成値として外部から受け取る。EBAEは共分散の作成方法、時間切り出し、位相整合、座標変換を規定しない。

\[
R[k]=E\{X[k]X[k]^H\}
\]

N/E AICに用いる独立snapshot数を`L=rate*T`とし、運用設定は次を必須とする。

\[
rate\,T=M^2
\]

したがって`L=M^2`である。`rate`は独立snapshot/s、`T`は秒、`M`は当該binのactive channel数である。相関したoverlap snapshotを独立snapshotとして数えてはならない。実装は`rate*T`と`M^2`が一致しない設定を拒否する。

## 4. 固有値分解

各binのHermitian共分散を次のように分解する。

\[
R=U\Lambda U^H
\]

固有値と対応固有ベクトルは固有値降順へ並べる。

\[
\lambda_1\geq\lambda_2\geq\cdots\geq\lambda_M\geq0
\]

`U[:,i]`が`lambda[i]`に対応する。Hermitian行列専用の`eigh`を使い、丸め誤差による微小な負固有値は0へ丸める。非Hermitianまたは非有限共分散は方式入力の契約違反として拒否する。

## 5. N/E AIC信号数推定

候補信号数を`n`、`0 <= n < min(M,L)`とする。降順固有値のうち`lambda[n:M]`を雑音候補とし、次を計算する。

\[
t_n=(M-n)\frac{\sum_{i=n+1}^{M}\lambda_i^2}{\left(\sum_{i=n+1}^{M}\lambda_i\right)^2}
\]

\[
c=\frac{M}{L},\qquad t_d=M\left[t_n-(1+c)\right]
\]

\[
J(n)=\frac{t_d^2}{2c^2}+2(n+1)
\]

\[
N_s=\arg\min_n J(n)
\]

複素観測であるためNadakuditi/Edelman式の`beta=2`を用いた形に一致する。雑音候補固有値の和が0で統計量を定義できない候補は`+inf`として選択対象から外す。

## 6. MUSICと信号対応付け

信号部分固有ベクトルを`U_s=[u_1,...,u_Ns]`、雑音部分を`U_n=[u_(Ns+1),...,u_M]`とする。未正規化steering `a(theta)`に対して次を計算する。

\[
P_{MUSIC}(\theta)=\frac{1}{\sum_{i=N_s+1}^{M}|u_i^Ha(\theta)|^2}
\]

MUSIC値が大きいbeam indexを`Ns`個選び、値の降順で、固有値降順の信号固有ベクトルへ一対一に対応付ける。最小方位間隔は設けない。同値の場合だけ結果を決定論的にするためbeam index昇順を使う。

## 7. EBAE重み

未正規化steeringからCBF重みを作る。

\[
w_0(\theta_b)=\frac{a(\theta_b)}{a(\theta_b)^Ha(\theta_b)}
\]

雑音固有値平均を次とする。

\[
\alpha=\frac{1}{M-N_s}\sum_{i=N_s+1}^{M}\lambda_i
\]

信号固有modeごとのロバスト化係数は次である。

\[
\beta_i(\theta_b)=\frac{\lambda_i-\alpha}{\lambda_i+DL(\theta_b)\alpha}
\]

既定値は全beamで`DL=1`とする。値を大きくすると除外量が小さくなりCBFへ近づく。

固有ベクトルとCBF重みの正規化重なりpowerを次の`rho_i`と定義する。

\[
\rho_i(\theta_b)=
\frac{|u_i^Hw_0(\theta_b)|^2}
{|w_0(\theta_b)^Hw_0(\theta_b)|}
\]

分母`w_0^H w_0`は実数かつ非負であるが、元の方式定義との対応を明示するため絶対値を残す。`rho_i`は無次元であり、信号固有ベクトル`u_i`と待受方位のCBF重みがどの程度重なるかを表す。

固有ベクトル除外係数は、対応方位以外では1とする。信号固有ベクトル`u_i`の対応方位を`theta_i`とすると、`rho_i`を用いた式は次である。

\[
\delta_i(\theta_b)=
\begin{cases}
1,&\theta_b\neq\theta_i\\
1-\dfrac{1}{1+\exp\left[-sigm_a\left(\rho_i(\theta_b)-sigm_b\right)\right]},
&\theta_b=\theta_i
\end{cases}
\]

`rho_i`を展開した完全な式は次であり、実装はこの式と同一である。

\[
\delta_i(\theta_b)=
\begin{cases}
1,&\theta_b\neq\theta_i\\
1-\dfrac{1}{1+\exp\left[-sigm_a\left(
\dfrac{|u_i^Hw_0(\theta_b)|^2}{|w_0(\theta_b)^Hw_0(\theta_b)|}-sigm_b
\right)\right]},&\theta_b=\theta_i
\end{cases}
\]

既定値は`sigm_a=10`、`sigm_b=0.5`とする。方位一致は浮動小数点角度ではなく、同じbeam indexかで判定する。

一時重みは次である。

\[
w_{tmp}(\theta_b)=w_0(\theta_b)-\sum_{i=1}^{N_s}
\delta_i(\theta_b)\beta_i(\theta_b)
\left(u_i^Hw_0(\theta_b)\right)u_i
\]

最終正規化の分母だけは正規化済みCBF重みではなく、未正規化steeringを使う。

\[
w_{opt}(\theta_b)=\frac{w_{tmp}(\theta_b)}{a(\theta_b)^Hw_{tmp}(\theta_b)}
\]

これにより`a(theta_b)^H w_opt(theta_b)=1`を満たす。分母が非有限または絶対値`1e-12`以下なら、途中重みを公開せず、そのbinの全beamをCBFへ戻す。`1e-12`は数値的に正規化不能な零近傍だけを検出する絶対下限であり、性能調整用閾値ではない。

## 8. 公開API

- `EbaeConfig`: `rate`, `T`, sigmoid、DL、安定化下限
- `estimate_signal_count_ne_aic`: 降順固有値から`Ns`を推定
- `calculate_music_spectrum`: 雑音部分空間からMUSICを計算
- `design_ebae_weights_band`: 単一FFT binを設計
- `design_ebae_weights`: `[n_bin,M,M]`をbin独立に一括設計
- `EbaeBandResult`, `EbaeResult`: shape固定の完成結果

## 9. 正式実装確認と評価接続条件

本セッションでは数式単体、shape、方位対応、無歪正規化、bin独立性、`rate*T=M^2`契約を確認する。MVDR置換の採否は行わない。

別セッションの評価方式へ接続するときは、少なくともtarget-only、noise-only、interferer-only、mixedを分離し、固定CBFとEBAEを同じ完成共分散、FFT、steering、方位軸で比較する。BLは`dB re mainlobe peak`または明記した入力基準、出力levelは`dB re input RMS`など基準を明記する。N/E AICの推定数、MUSIC対応方位、fallback binも成果物へ保存する。

### 11.7 整相方式との部品境界

EBAEは、周波数binごとの完成共分散とsteeringを入力として重みを返す部品である。共分散構成とFIR実現座標は責務に含めない。これらの設計、方式ID、FIR長評価は`doc/SpFlow/整相方式設計結果.md`へ集約する。

EBAEとMVDRは同じ共分散・steering契約へ交換可能に接続する。方式比較で差が生じた場合は、共分散構成とFIR実現座標を先に固定し、その後にEBAEとMVDRの重み設計差を評価する。
## 12. EBAE/MVDR 1パターン基本動作確認

### 12.1 目的

整相方式やFIR長との組合せ評価へ進む前に、EBAEとMVDRを同じ理想共分散へ接続し、bin中心かつ待受beam直上の単一信号に対する基本応答が大きく矛盾しないことを確認する。

本確認は`fixed_beam_single_source`の静的1-bin sanity checkであり、整相方式の採否、FIR長、FRAZ、BTR、streaming成立性は判定しない。

### 12.2 条件

| 項目 | 条件 |
|---|---:|
| channel数 | 8 |
| array | ULA、1000 Hzにおける半波長間隔 |
| sample rate | 8000 Hz |
| FFT長 | 256 sample |
| source bin | index 32、1000 Hz |
| source方位 | 60 deg、1 deg刻みbeam grid上 |
| source level | 0 dB re input RMS |
| channel雑音power | -10 dB re input RMS^2/channel |
| 共分散 | `R=a a^H+0.1 I` |
| EBAE | `DL=1`、`sigm_a=10`、`sigm_b=0.5` |
| N/E AIC snapshot数 | `L=M^2=64` |
| MVDR | 同じ完成共分散、追加loadingなし |

sourceはFFT bin中心とbeam中心へ厳密に一致させた。共分散に`0.1 I`を含めて正定値としたため、MVDRには追加の数値安定化loadingを加えていない。EBAEの`DL=1`は方式既定値であり、MVDRのdiagonal loadingと同じ意味ではない。

### 12.3 確認結果

| 指標 | EBAE DL=1 | MVDR |
|---|---:|---:|
| N/E AIC信号数 | 1 | 該当なし |
| MUSIC対応方位 | 60 deg | 該当なし |
| target peak方位 | 60 deg | 60 deg |
| target peak誤差 | 0 deg | 0 deg |
| target level | 約0 dB re input RMS | 0 dB re input RMS |
| distortionless error | `2.22e-16` | `2.29e-23` |
| target beam weight norm | `0.353553` | `0.353553` |
| guard外peak | `-36.524 dB re input RMS` | `-42.408 dB re input RMS` |
| target-only BL RMS差 | `5.829 dB re MVDR` | 0 dB |
| target-only BL最大絶対差 | `5.916 dB re MVDR` | 0 dB |

EBAEは信号数1を推定し、信号固有ベクトルを正しい60 degへ対応付けた。EBAEとMVDRはいずれもtarget peak方位、target level、target beamのwhite-noise gainに対応するweight normを一致させた。したがって、共役、steering符号、無歪正規化、AIC、MUSICの基本接続に破綻は見られない。

一方、非target方位ではEBAEの応答がMVDRより約5.9 dB高い。両BLはほぼ同じ形状であり、EBAEだけ全体に抑圧が浅い。これは`DL=1`により

\[
\beta_i=\frac{\lambda_i-\alpha}{\lambda_i+\alpha}
\]

として固有mode除外量を弱め、厳密な最小分散解よりCBF側へ寄せるロバスト化の設計意図と整合する。したがって、本条件で確認された「ほぼ同じ」は、target方位、target利得、主ローブ・sidelobe形状が一致するという意味であり、全方位のlevelが数値的に一致するという意味ではない。

### 12.4 成果物

| 成果物 | 配置 |
|---|---|
| 評価実装 | `evaluations/beamforming/ebae_mvdr_bin_center_sanity.py` |
| 回帰試験 | `tests/beamforming/test_ebae_mvdr_bin_center_sanity.py` |
| 指標CSV | `artifacts/beamforming/ebae_mvdr_bin_center_sanity/review_pack/scenario_summary.csv` |
| 描画配列 | `artifacts/beamforming/ebae_mvdr_bin_center_sanity/review_pack/data/single_source_bin_center_beam_center.npz` |
| BL図 | `artifacts/beamforming/ebae_mvdr_bin_center_sanity/review_pack/figures/single_source_bin_center_beam_center/bl_overlay.png` |
| 成果物定義 | `artifacts/beamforming/ebae_mvdr_bin_center_sanity/review_pack/review_index.md` |
