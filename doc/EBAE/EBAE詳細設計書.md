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

## 11. EBAEを適用する3方式の整理

### 11.1 比較の基準

EBAEの適用方法として、次の3方式を区別する。

| method ID | 方式名 | 共分散座標 | 実時間出力構造 | EBAEの数式上の意味 |
|---|---|---|---|---|
| `direct_s0_ebae` | 直接EBAE重み方式 | 元センサ座標のS0共分散 | 完成EBAE重みを直接適用 | 本設計で定義した基準EBAE |
| `fixed_integer_fractional_ebae_difference` | EBAE差分補正枝方式 | 元センサ座標のS0共分散 | 整数遅延＋小数遅延の固定主枝からEBAE差分補正枝を減算 | 完成重みが一致する限り、直接EBAEと同じ空間フィルタを別構造で実現 |
| `integer_delay_then_ebae` | 整数遅延＋EBAE補正方式 | S0共分散を待受beam別の整数遅延後座標へ位相回転 | 整数遅延後の残留整相と固有mode除外をEBAEで実施 | 同じS0共分散のunitary座標変換であり、理想条件では直接S0-EBAEと同じ |

ここで「補正」は2つの意味を持ち得るため混同しない。

- `EBAE差分補正枝`の補正は、完成EBAE重みと固定主枝重みの**差分を実装する並列枝**を意味する。
- `整数遅延＋EBAE補正`の補正は、整数遅延後に残る**小数遅延、残留位相、信号固有mode**をEBAE重みで処理することを意味する。

### 11.2 方式A: 直接EBAE重み方式

元センサ入力を`X[k]`、S0共分散を`R_S0[k]`とする。

\[
R_{S0}[k]=E\{X[k]X[k]^H\}
\]

この共分散、元センサ座標の未正規化steering`a(theta_b,k)`、CBF重み`w_0(theta_b,k)`から、7章の式で`w_opt(theta_b,k)`を設計する。出力は完成重みを直接適用する。

\[
Y_{direct}(\theta_b,k)=w_{opt}(\theta_b,k)^H X[k]
\]

この方式は現在の`design_ebae_weights`が表す基準方式であり、EBAEアルゴリズムそのものの成立性を評価する。固定遅延主枝、差分補正FIR、整数遅延後共分散の影響を含まないため、他2方式の誤差を切り分ける参照になる。

直接適用をSTFT、filter bank、または周波数応答から作った多channel FIRで実装する場合、有限長化、窓、overlap、再合成による誤差はEBAEの数式ではなく実装誤差として別に記録する。

### 11.3 方式B: EBAE差分補正枝方式

整数遅延＋小数遅延FIRの固定整相主枝が実際に持つ周波数重みを`w_fixed(theta_b,k)`とする。理想steeringから作ったCBF重みで代用せず、整数遅延表、小数遅延FIR、channel shading、FIR群遅延を含む実周波数応答を使う。

直接EBAEで設計した完成重みを`w_opt(theta_b,k)`とし、差分補正重みを次で定義する。

\[
q_{EBAE}(\theta_b,k)
=w_{fixed}(\theta_b,k)-w_{opt}(\theta_b,k)
\]

固定主枝出力と差分補正枝出力を同じ時間基準で減算する。

\[
Y_{difference}(\theta_b,k)
=w_{fixed}(\theta_b,k)^H X[k]
-q_{EBAE}(\theta_b,k)^H X[k]
\]

差分重みが周波数領域で厳密に再現される場合は次が成立する。

\[
Y_{difference}(\theta_b,k)
=w_{opt}(\theta_b,k)^H X[k]
=Y_{direct}(\theta_b,k)
\]

したがって、この方式はEBAEの信号数推定、MUSIC対応付け、固有mode除外の意味を変えない。変わるのは完成重みの**実装分解**だけである。

ただし、次の場合は直接EBAEと同じ意味にならない。

1. `w_fixed`を実際の主枝応答ではなく理想CBFで近似した場合。
2. 差分FIRの有限tap化により`q_EBAE`を再現できない場合。
3. 固定主枝と差分枝の群遅延またはblock境界が一致しない場合。
4. EBAE重みの最終正規化前の`q_raw`だけを差分枝へ与え、正規化差分を含めない場合。
5. 重み更新時に主枝と差分枝が異なる完成世代の係数を参照した場合。

`Ns=0`では基準EBAE重みは理想CBF`w_0`へ戻る。一方、実固定主枝`w_fixed`が理想CBFと有限精度で異なる場合、差分は次となる。

\[
q_{EBAE}=w_{fixed}-w_0
\]

この場合、差分枝は完全な無出力ではなく、固定遅延＋小数遅延FIRの実応答を理想CBFへ合わせる実装誤差補正を含む。`Ns=0`で差分枝を必ず0にしたい場合は、EBAE内部の`w_0`を実固定主枝応答と同じ規約で定義し直す必要があり、その方式は現在の理想steering基準EBAEとは別契約になる。

### 11.4 方式C: 整数遅延＋EBAE補正方式

待受beam`theta_b`に対応するchannel別整数遅延演算を、周波数bin上の行列`D_b[k]`で表す。整数遅延後のセンサ入力、共分散、steeringは次となる。

\[
X_D(\theta_b,k)=D_b[k]X[k]
\]

\[
R_D(\theta_b,k)
=E\{X_D(\theta_b,k)X_D(\theta_b,k)^H\}
=D_b[k]R[k]D_b[k]^H
\]

\[
a_D(\theta,\theta_b,k)=D_b[k]a(\theta,k)
\]

`D_b`は待受beamごとに異なるため、共分散とsteeringも`theta_b`ごとの座標を持つ。整数遅延後のEBAE重みを`v_opt(theta_b,k)`とすると、出力は次である。

\[
Y_{integer+EBAE}(\theta_b,k)
=v_{opt}(\theta_b,k)^H X_D(\theta_b,k)
\]

元センサ座標で等価な重みは次となる。

\[
w_{equivalent}(\theta_b,k)=D_b[k]^H v_{opt}(\theta_b,k)
\]

理想的な周波数領域位相回転として`D_b`を適用し、`R_D=D_b R_S0 D_b^H`と`a_D=D_b a`を厳密に使い、EBAEの全量を同じ座標へ変換する場合、これはunitaryな座標変換である。固有値、N/E AICの信号数、MUSIC値、元座標へ戻した完成重みは理論上、直接S0-EBAEと等価になり得る。

実時間の整数遅延bufferをFFT前に適用し、そのblockから共分散を新たに再推定する方式は、本書でいうS1ではない。これは`integer_delay_reestimated_covariance_ebae`など別IDを与えるべき追加候補である。channelごとの切り出し時刻、block境界、過渡波形、有限窓がS0と異なるため、この追加候補では一般に次が成立しない。

\[
R_{reestimated}(\theta_b,k)
\neq D_b[k]R_{S0}[k]D_b[k]^H
\]

この場合、固有値、N/E AICの推定信号数、雑音部分空間、MUSIC対応方位、`delta_i`、最終EBAE重みのすべてが変わり得る。したがって、実bufferから共分散を再推定する追加候補は、直接S0-EBAEおよびS1-EBAEの実装違いではなく、**待受beam別に再推定した共分散を使う別方式**として扱う。

また、現在のEBAE公開APIは共通のS0共分散`[n_bin,M,M]`を全待受beamへ使用する。整数遅延後に共分散を再推定する方式では、概念上`[n_beam,n_bin,M,M]`の共分散、またはbeamごとに独立した逐次状態が必要であり、現在の実装契約には含まれない。

### 11.5 小数遅延整相の意味

3方式では小数遅延整相の担当が異なる。

| 方式 | 整数遅延 | 小数遅延・残留位相 | 固有mode除外 |
|---|---|---|---|
| 直接EBAE | 前段では使用しない | `w_opt`の周波数依存位相が担当 | `w_opt`が担当 |
| EBAE差分補正枝 | 固定主枝が担当 | 固定主枝の小数遅延FIRを基準に、差分枝を含む合成重みが最終応答を担当 | 差分枝を含む合成重みが担当 |
| 整数遅延＋EBAE補正 | 前段bufferが担当 | 整数遅延後の`v_opt`が残留小数遅延と位相を担当 | `v_opt`が担当 |

したがって「整数遅延＋EBAE補正」は、整数遅延後にEBAEを単なる妨害除去器として足す方式ではない。残留小数遅延整相と固有mode除外を同じ重みが同時に担う。steering、共分散、正規化を整数遅延前の座標のまま混在させてはならない。

### 11.6 採否ではなく切り分けとしての3方式評価

正式評価では同じ入力波形、active channel、方位軸、周波数軸、更新時刻を用い、次の順で切り分ける。

1. `direct_s0_ebae`でEBAE数式自体の信号数推定、方位対応、target保持、非target抑圧を確認する。
2. `fixed_integer_fractional_ebae_difference`と`direct_s0_ebae`を比較し、差をFIR近似、主枝応答、群遅延、係数切替へ限定できるか確認する。
3. `integer_delay_then_ebae`と`direct_s0_ebae`を比較し、同じS0共分散の位相回転で固有値、信号数、MUSIC値、元座標へ戻した完成重みが一致することを確認する。

各方式について、少なくとも次を保存する。

- `Ns[k]`とN/E AIC値
- MUSIC疑似スペクトルと対応beam index
- `w^H a`または整数遅延座標の`v^H a_D`
- target-only、noise-only、interferer-only、mixedの帯域積分RMS
- BLのtarget peak、guard外peak、source分離valley
- 差分枝方式では`w_fixed-q_FIR-w_opt`の周波数応答誤差
- 整数遅延前段方式では`R_S1-D_b R_S0 D_b^H`の相対誤差。この式でS1を定義する
- latency、block境界、重み更新境界、fallback状態

3方式のBL/FRAZ/BTRは同じ表示条件とdB基準で比較する。絶対levelは`dB re input RMS`などの基準、相対差は`dB re direct S0 EBAE`または`dB re fixed integer+fractional output`を明記する。直接方式との差が観測された場合、EBAE方式の優劣と即断せず、まず共分散座標、steering座標、FIR再現誤差、時間整合のどこで差が生じたかを判定する。

### 11.7 過去のMVDRにおけるS0・S1・T1・T2結果との関係

過去のMVDR比較では、方式を次のように区別していた。

| 旧ID | 共分散・適用方式 | 過去結果の群 |
|---|---|---|
| S0 | 整数遅延なし、同一時間blockの粗い共分散から直接設計 | 低周波・粗い分析幅で裾が広く、他3方式と異なる |
| S1 | S0共分散とS0重みを整数遅延後のchannel位相基準へ回転し、遅延後入力へ適用 | 理論上S0と同じ完成出力になるべき |
| T1 | 方位別時間切り出し共分散から設計し、元入力へ直接適用 | 正しいT2と数値誤差内で一致するべき |
| T2 | T1と同じ完成共分散を整数遅延後座標へ変換して適用 | T1のunitary座標変換であり、元座標へ戻せばT1と同じ |

代表的な低周波endfire条件として保存された結果では、`integer_delay_then_mvdr`と方位別時間切り出し共分散の2方式がほぼ同じBL/FRAZとなり、S0だけ低周波側に広い応答が残っていた。しかし、この結果を「正しいS1がT1・T2へ近づいた」と解釈してはならない。

評価実装`direction_cut_mvdr_spatial_spectral_review.py`を確認すると、`integer_delay_then_mvdr`の共分散を作る箇所で、S0の`coarse_covariance`を位相回転せず、次のように方位別時間切り出し共分散を再計算していた。

```python
integer_aligned_coarse_covariance = _direction_cut_covariance(...)
rotated_coarse_covariance = D * integer_aligned_coarse_covariance * D.conj()
```

したがって、保存結果の`integer_delay_then_mvdr`は、名称上はS1でも、共分散内容はT系を整数遅延後座標へ回転したものだった。T1・T2へ近づいたのは当然であり、この結果はS1とS0の差を示す証拠として無効である。

意図されたS1は、同じS0共分散とsteeringを整数遅延位相行列`D_b[k]`で次のように変換する。

\[
R_{S1}(\theta_b,k)=D_b[k]R_{S0}[k]D_b[k]^H
\]

\[
a_{S1}(\theta,\theta_b,k)=D_b[k]a(\theta,k)
\]

S1座標で設計した重みを`v_S1`とすると、元入力座標へ戻した等価重みは次である。

\[
w_{S1,equivalent}(\theta_b,k)=D_b[k]^H v_{S1}(\theta_b,k)
\]

`D_b`がunitaryなchannel別位相回転で、loading規約もunitary変換に対して不変なら、S0とS1は次を満たす。

\[
w_{S1,equivalent}(\theta_b,k)=w_{S0}(\theta_b,k)
\]

したがって、正しいS1はS0のcoherenceや固有値を改善しない。S0重みを整数遅延後入力の位相基準へ移しただけであり、元入力座標へ戻せばS0と同じ方式である。

S0の同一時間block共分散は、長大開口、粗い分析幅、広帯域信号の組合せで、bin内の周波数成分を単一steering vectorとして表せず、channel pairごとに概ね次のcoherence低下を持つ。

\[
\operatorname{sinc}\left(\Delta f\,\tau_{ij}\right)
\]

このcoherence低下により、単一sourceであっても信号powerが複数固有modeへ分散し、共分散rankと固有空間が理想的な狭帯域モデルから外れる。T1・T2は方位別時間切り出しにより同一波面区間を揃えるため、S0より信号部分空間を回復する。一方、正しいS1はS0共分散のunitary位相回転なので固有値とcoherenceを変えず、この劣化を回復しない。

#### EBAEで予想される影響

EBAEはMVDRより共分散固有空間へ直接依存するため、S0/S1群とT1/T2群の差を無視できない。S0と、そのunitary座標変換であるS1が持つcoherence低下は、EBAEの各段へ次のように伝わる。

1. 単一sourceの固有値が複数modeへ分散し、N/E AICが`Ns`を過大推定する可能性がある。
2. 雑音部分へ本来の信号成分が漏れ、MUSIC peakが広がる、移動する、または複数peakになる可能性がある。
3. 信号固有ベクトルと方位の対応が崩れ、保護すべきmodeへ`delta_i=1`を適用する可能性がある。
4. 雑音固有値平均`alpha`と`beta_i`が変わり、除外量がT1/T2系と異なる。
5. 結果として、target保持、非target抑圧、`Ns=0`へ戻る条件がT1/T2系と異なる。

したがって、EBAEでも次の群分けを第一仮説とする。

```text
S0共分散と同じ固有空間で設計する方式:
    direct_s0_ebae
    integer_delay_then_ebae       （正しいS1。S0の位相座標変換）
    fixed_integer_fractional_ebae_difference
        ※差分枝の完成重みをS0共分散から設計する場合

coherenceを回復した共分散で設計する方式:
    direction_cut_direct_ebae     （T1相当、今後の比較候補）
    direction_cut_integer_ebae    （T2相当、今後の比較候補）
```

EBAE差分補正枝方式は、固定主枝に整数遅延＋小数遅延FIRを持っていても、統計ルートでS0共分散から`w_opt`を設計する限りS0群に属する。実時間主枝を高精度にしても、S0共分散の固有空間劣化は修復されない。これは実行経路の整相方式と、重み設計に使う共分散方式を分離して考える必要があることを示す。

一方、正しい整数遅延＋EBAE補正方式はS0共分散をunitary位相回転したS1を使うため、EBAEでも固有値、N/E AIC、MUSIC値、`rho_i`、`delta_i`、元座標へ戻した完成重みがS0-EBAEと一致することを期待する。T1・T2へ近づく根拠はない。過去の`integer_delay_then_mvdr`結果はT系共分散が混入しているため、この仮説の検証には使わない。

#### 3方式比較への反映

先に定義した3方式だけを比較すると、実装構造差と共分散方式差が同時に変わる。比較は次の2段階に分ける。

1. **重み実装構造の同値性確認**
   - `direct_s0_ebae`
   - `fixed_integer_fractional_ebae_difference`
   - 両方とも同じS0共分散、同じ`w_opt`を使い、差をFIR化と時間整合だけへ限定する。
2. **S0とS1の座標変換同値性確認**
   - `direct_s0_ebae`
   - `integer_delay_then_ebae`（正しいS1）
   - 同じS0共分散から`R_S1=D R_S0 D^H`を作り、元座標へ戻した完成重みとcomplex出力が一致することを確認する。
3. **共分散coherence回復の効果確認**
   - `direct_s0_ebae`
   - 将来接続する`direction_cut_direct_ebae`（T1）
   - 将来接続する`direction_cut_integer_ebae`（T2）
   - `Ns`、固有値比、信号部分空間角、MUSIC peak、`delta_i`、完成重みを比較する。

EBAEでは、S0とS1が同じ中間量を持ち、T1とT2が別の同じ群を作るかを確認する。少なくとも次を比較する。

- binごとの`Ns`
- 降順固有値と雑音平均`alpha`
- 信号部分空間間のprincipal angle
- MUSIC peak方位とpeak順位
- 固有ベクトルごとの対応beam index
- `rho_i`、`delta_i`、`beta_i`
- 正規化前`q_raw`と完成`w_opt`

この確認により、S0とS1、およびT1とT2がそれぞれ同値となる理由を、EBAE内部の信号数推定、方位対応、固有mode除外まで含めて説明できる。

### 11.8 並列差分補正枝と補正済み小数遅延FIRの数学的等価性

ここでは共分散方式のS0/S1比較とは分離し、**同じ完成MVDR重みを2つの実装構造で実現する場合**を整理する。比較対象は次の2方式である。

1. `(整数遅延＋小数遅延の固定主枝)－(MVDR差分補正枝)`として並列に計算する方式。
2. MVDR差分補正を小数遅延FIR係数へあらかじめ合成し、`整数遅延＋補正済み小数遅延FIR`として1枝で計算する方式。

この2方式は、整数遅延後に別のS1共分散からMVDRを再設計するかどうかを表す区別ではない。**同じ差分重みを並列枝として保持するか、係数へ焼き込むかという実装分解の違い**である。

#### 周波数領域での等価性

整数遅延＋小数遅延の固定主枝が元センサ入力`X[k]`に対して持つ完成重みを`w_fixed[k]`、同じ入力座標で表した完成MVDR重みを`w_mvdr[k]`とする。MVDR差分補正重みは次である。

\[
q_{mvdr}[k]=w_{fixed}[k]-w_{mvdr}[k]
\]

並列差分補正枝方式の出力は次となる。

\[
Y_A[k]
=w_{fixed}[k]^H X[k]-q_{mvdr}[k]^H X[k]
\]

差分重みの定義を代入すると、次が厳密に成立する。

\[
Y_A[k]
=\left(w_{fixed}[k]-q_{mvdr}[k]\right)^H X[k]
=w_{mvdr}[k]^H X[k]
\]

一方、補正済み小数遅延FIR方式では、整数遅延と補正済みFIRを合わせた完成周波数重みを次とする。

\[
w_{corrected}[k]
=w_{fixed}[k]-q_{mvdr}[k]
\]

1枝での出力は次である。

\[
Y_B[k]=w_{corrected}[k]^H X[k]
\]

したがって、

\[
w_{corrected}[k]=w_{mvdr}[k]
\]

を満たすように係数を合成できれば、

\[
Y_A[k]=Y_B[k]=w_{mvdr}[k]^H X[k]
\]

となる。つまり、理想的な線形時不変処理としては、並列の`固定主枝－差分補正枝`と、差分を焼き込んだ`整数遅延＋補正済み小数遅延FIR`は数学的に同じ意味である。

#### 時間領域FIRでの表現

実適用係数は`y=w^H X`の共役規約を含むため、固定主枝FIRを`h_fixed[ch,tap]`、差分補正FIRを`h_q[ch,tap]`とする。両者が同じ整数遅延基準、同じtap原点、同じshape`[n_ch,n_tap]`を持つ場合、補正済みFIRは次となる。

\[
h_{corrected}[ch,tap]
=h_{fixed}[ch,tap]-h_q[ch,tap]
\]

畳み込みは線形であるため、

\[
(h_{fixed}*x)[n]-(h_q*x)[n]
=((h_{fixed}-h_q)*x)[n]
= (h_{corrected}*x)[n]
\]

が成立する。したがって、差分補正結果を出力で引くか、FIR係数で先に引くかは数学的意味を変えない。

#### 実装上同じにならない条件

実装では、次の条件があると2方式は一致しない。

1. 固定主枝と差分補正枝でinteger delay、FIR群遅延、tap原点、block latencyが異なる。
2. それぞれを独立に有限tap化した後で出力減算し、`h_fixed-h_q`を同じtap長で直接構成した場合と打切り誤差が異なる。
3. 係数量子化、飽和、channel shading、active channel maskの適用順が異なる。
4. 並列枝の係数更新時刻がずれ、異なる完成世代の`w_fixed`と`q_mvdr`を一時的に組み合わせる。
5. 片方だけにcrossfade、窓、overlap-save、fallbackが適用される。
6. `q_mvdr`を作った`w_fixed`と、実時間主枝が実際に使う`w_fixed`が一致しない。

このため、方式同値性は最終BLだけでなく、次の周波数応答誤差で直接確認する。

\[
\epsilon_w[k]
=w_{fixed}[k]-q_{mvdr}[k]-w_{corrected}[k]
\]

加えて、同一入力に対するcomplex出力最大絶対誤差、RMS誤差、block境界誤差を記録する。

#### S1方式との区別

`整数遅延＋補正済み小数遅延FIR`という実装名だけでは、重みをどの共分散から設計したかは決まらない。同じS0またはT1/T2由来の完成`w_mvdr`を係数へ焼き込むだけなら、並列差分補正枝方式と数学的に同じである。

一方、整数遅延後の実信号から共分散を新しく再推定し、その再推定共分散から残留MVDR重みを設計する場合は、完成重み自体が変わり得る。この再推定方式をS1とは呼ばない。この場合の比較は、

\[
w_{mvdr,S0/T}[k]
\stackrel{?}{=}D[k]^H v_{mvdr,reestimated}[k]
\]

であり、単なるFIR合成の同値性ではない。正しいS1は再推定を行わず、S0共分散、steering、重みへ整数遅延分の位相を戻すだけなので、理想条件ではS0と同値である。

#### EBAEへの置換

MVDRをEBAEへ置き換える場合も同じ関係を使う。完成EBAE重み`w_opt`に対して、

\[
q_{EBAE}=w_{fixed}-w_{opt}
\]

を作れば、並列EBAE差分補正枝と、`q_EBAE`を小数遅延FIRへ焼き込んだ補正済み1枝は、同じ時間・周波数基準で係数を再現する限り数学的に同じである。AIC、MUSIC、`delta_i`、`beta_i`の意味も変わらない。S0から正しいS1への変更もunitary座標変換なので意味を変えない。これらの意味が変わるのは、EBAEへ入力する共分散をT系または整数遅延後の再推定共分散へ変更したときである。
