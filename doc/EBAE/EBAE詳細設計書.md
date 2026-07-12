# EBAE 詳細設計書

## 1. 目的と責務

EBAE（Dominant Mode Rejection with Eigenvector-Beam Association and Excision）は、S0方式で推定した周波数別センサ間共分散を固有値分解し、Nadakuditi/Edelman AIC（N/E AIC）で信号数を推定する。雑音部分空間から得たMUSIC疑似スペクトルにより信号固有ベクトルと方位を対応付け、待受方位に対応する信号を保護しながら、それ以外の信号固有modeをCBF重みから除外する。

本方式はMVDRの制約付き最小分散問題を解かない。MVDRの置換候補ではあるが、数式、実装、評価はMVDRと独立に扱う。

本設計の責務は次である。

1. 完成済みS0共分散のHermitian固有値分解と降順整列
2. N/E AICによる周波数bin別信号数推定
3. MUSIC疑似スペクトルによる信号固有ベクトルと方位の対応付け
4. sigmoid型固有ベクトル除外係数を使ったEBAE重み設計
5. 完成重みと診断量の固定shapeでの公開
6. 正規化不能時のCBF fallback

FFT、S0共分散の積分、重み適用、周波数間追跡、BL/FRAZ/BTR評価は責務に含めない。

## 2. 入出力と軸

| 量 | shape | 意味・単位 |
|---|---:|---|
| `R` | `[n_bin,M,M]` | S0空間共分散。axis 0はFFT bin、axis 1,2はactive channel。単位は入力power |
| `A` | `[M,n_beam,n_bin]` | 未正規化steering。axis 0はchannel、axis 1は待受方位、axis 2はFFT bin |
| `W_opt` | `[M,n_beam,n_bin]` | EBAE重み。同じaxis規約 |
| `music` | `[n_beam,n_bin]` | MUSIC線形疑似スペクトル |
| `Ns` | `[n_bin]` | binごとの推定信号数 |

全FFT binは完全に独立して処理する。隣接bin間で信号数、固有ベクトル、対応方位を平滑化または追跡しない。

## 3. S0共分散とsnapshot数

入力共分散は、整数遅延を与えない同一時間blockのFFTセンサ出力から作るS0方式とする。

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

別セッションの評価方式へ接続するときは、少なくともtarget-only、noise-only、interferer-only、mixedを分離し、固定CBFとEBAEを同じS0共分散、FFT、steering、方位軸で比較する。BLは`dB re mainlobe peak`または明記した入力基準、出力levelは`dB re input RMS`など基準を明記する。N/E AICの推定数、MUSIC対応方位、fallback binも成果物へ保存する。

## 10. 固定遅延＋小数遅延主枝と差分補正枝への接続

### 10.1 信号数0の境界条件

N/E AICの推定結果が`Ns=0`の場合、信号固有modeの除外和は空和で0となる。

\[
w_{tmp}(\theta_b)=w_0(\theta_b)
\]

CBF重みは`a(theta_b)^H w_0(theta_b)=1`を満たすため、最終正規化を適用しても次となる。

\[
w_{opt}(\theta_b)
=\frac{w_0(\theta_b)}{a(\theta_b)^Hw_0(\theta_b)}
=w_0(\theta_b)
\]

したがって、信号数0ではEBAE適応重みが数値誤差を除いてCBF重みと一致する。これは推定対象の信号modeがないときに不要な適応処理を行わない安全側の境界条件であり、実装もこの場合は`fixed_weights`をそのまま返す。

### 10.2 正規化前のEBAE差分量

EBAEの固有mode除外和を次の差分量として定義する。

\[
q_{raw}(\theta_b)
=\sum_{i=1}^{N_s}\delta_i(\theta_b)\beta_i(\theta_b)
\left(u_i^Hw_0(\theta_b)\right)u_i
\]

このとき、正規化前重みは固定整相主枝から差分補正枝を引く既存構造と同じ形になる。

\[
w_{tmp}(\theta_b)=w_0(\theta_b)-q_{raw}(\theta_b)
\]

したがって、以前検討した「整数遅延＋小数遅延FIR」の固定整相主枝と、差分補正FIR枝の構造をEBAEへ再利用できる。ただし、既存の名称`差分MVDR`は補正量が`w_0-w_MVDR`であることを表すため、EBAE接続後の枝は`EBAE差分補正枝`と呼び、MVDR重みを使用していると誤解させない。

### 10.3 最終正規化を含む差分重み

`q_raw`をそのまま時間領域差分補正枝へ与えると、合成後の実効重みは`w_tmp`であり、設計済みの`w_opt`とは正規化係数だけ異なる。正規化分母を次とする。

\[
g(\theta_b)=a(\theta_b)^H\left(w_0(\theta_b)-q_{raw}(\theta_b)\right)
\]

\[
w_{opt}(\theta_b)=\frac{w_0(\theta_b)-q_{raw}(\theta_b)}{g(\theta_b)}
\]

固定整相主枝を変更せず、既存の`主枝－差分枝`で正規化後のEBAE重みを厳密に再現する差分重みは次である。

\[
q_{EBAE}(\theta_b)
=w_0(\theta_b)-w_{opt}(\theta_b)
=w_0(\theta_b)
-\frac{w_0(\theta_b)-q_{raw}(\theta_b)}{g(\theta_b)}
\]

これを用いると次が厳密に成立する。

\[
w_0(\theta_b)-q_{EBAE}(\theta_b)=w_{opt}(\theta_b)
\]

この方法は既存の差分補正FIR設計器へ完成済み適応重み`w_opt`を渡し、`q=w_0-w_opt`を作る現在の責務分離と一致する。また`Ns=0`では`q_raw=0`かつ`g=1`であるため、`q_EBAE=0`となり差分補正枝は無出力になる。

別案として、`q_raw`をそのまま差分枝へ適用した合成出力を後段で`1/conj(g)`倍する方法も数式上は同値である。ビーム出力規約が`y=w^H X`であるため、重みを`1/g`倍したとき出力側の係数は`1/conj(g)`になる。しかし、この方法はbin・beam別の複素gainを合成後段へ追加し、固定主枝と補正枝の完成値公開境界を複雑にする。このため正式接続では、正規化済み`w_opt`から`q_EBAE=w_0-w_opt`を作る方式を第一候補とする。

### 10.4 接続前に確認する項目

EBAE差分補正枝を実装する前に、次を評価で確認する。

1. 周波数領域で`w_0-q_EBAE`と`w_opt`が一致すること。
2. FIR化後も待受応答`(w_0-q_FIR)^H a`が1に近いこと。
3. `Ns=0`で差分補正出力が数値床に留まり、CBF出力と一致すること。
4. `Ns`または対応方位が更新された境界で、未完成の`q_EBAE`を公開しないこと。
5. target-only、noise-only、interferer-only、mixedの各出力で、正規化によるtarget level変化と干渉低減を分離して記録すること。
