# beam-domain GSC 検討

## 1. 目的

本書は、固定整相後のビーム出力から共分散を作り、beam-domain GSC として SLC を構成する方式を検討する。

既存の SLC 実装は、guard 外ビームを reference として target beam から推定干渉を差し引く構成である。ここではそれを一段明確化し、固定整相後のビーム空間で desired 成分を blocking してから共分散を作る方式として整理する。

ただし、リアルタイム CPU 処理を設計条件とする今回の方式では、最初から周波数ビンごとの beam-domain GSC を本番経路には入れない。現行方式は、固定整相後の時間領域 `beam_output[beam, sample]` から共分散を 1 つ作る時間領域 SLC とする。本書の beam-domain GSC は、時間領域 SLC の限界を確認した後に導入する拡張方式、および narrowband 診断の理論整理として扱う。

---

## 2. 前提

固定整相後のビーム出力を次で表す。

```text
x[beam, frame]
```

時間領域の実信号をそのまま使う場合は `frame = sample` と見なす。STFT / narrowband 処理では、周波数ごとに次の複素 snapshot を使う。

```text
X[f][beam, frame]
```

本検討では、SLC の共分散はチャネル信号からではなく、固定整相後のビーム出力から作る。

---

## 3. beam-domain GSC の基本構造

target 方位に対応する固定整相ビーム出力を主経路とし、guard 外ビーム出力から desired 成分を blocking したものを参照経路とする。

```text
fixed beam output x
  ├─ main path
  │    d = c^H x
  │
  └─ reference path
       u = B^H x
       ↓
       adaptive canceller
       c_hat = w^H u

y = d - eta * c_hat
```

ここで、`c` は主経路選択ベクトル、`B` は blocking matrix、`u` は desired 成分を抑えた参照信号である。

---

## 4. 固定整相ビーム応答ベクトル

beam-domain GSC では、target 方位の信号が各固定ビームへどう漏れるかを表す応答ベクトルが必要である。

周波数 `f`、target 方位 `theta_t` に対して、固定整相後ビームの応答を次で表す。

```text
a_t[f]: [n_beam]
```

`a_t[f][b]` は、target 方位の平面波が observation beam `b` に出る複素応答である。

既存診断コードでは、整数遅延版は `_build_beam_response_matrix()`、小数遅延版は `_build_fractional_beam_response_matrix()` で次の行列を作っている。

```text
A[f][beam_obs, beam_look]
```

target beam index を `t` とすれば、

```text
a_t[f] = A[f][:, t]
```

である。

---

## 5. blocking の作り方

## 5.1 guard 外 reference beam を使う方式

target beam と guard 領域を除外し、reference beam index を作る。

```text
R = {guard 外 beam index}
```

reference beam 出力は、

```text
x_R[f, frame] = X[f][R, frame]
```

である。

このまま共分散を作ると、reference beam に target mainlobe / sidelobe 成分が混入して self-nulling を起こす可能性がある。そこで、target 応答の reference 部分を使って desired 成分を射影除去する。

```text
a_R[f] = a_t[f][R]
```

射影行列を、

\[
P_R[f]
=
I
-
\frac{a_R[f]a_R^H[f]}{a_R^H[f]a_R[f]}
\]

とする。

blocking 後の reference snapshot は、

\[
u[f,k]
=
P_R[f]x_R[f,k]
\]

である。

この方式は、現行の `_apply_frequency_selective_scan_slc()` に近い。既存実装は `response_matrix[selected_reference_beams, beam_index]` を `a_R` とし、`projector @ snapshots[selected_reference_beams, :]` で blocking している。

---

## 5.2 全ビーム nullspace を使う方式

guard 外だけでなく全ビーム空間の target 応答 `a_t` に対して、nullspace basis `B_t` を作る方式もある。

\[
B_t^H a_t = 0
\]

を満たす `B_t` を使い、

\[
u[f,k] = B_t^H x[f,k]
\]

とする。

この方式は理論的には GSC に近いが、次の欠点がある。

- `n_beam - 1` 次元の参照になるため共分散が大きい。
- 片舷ビーム数が多い場合、snapshot 数に対して自由度が過大になりやすい。
- guard による物理的な mainlobe 保護と独立に動くため、評価時の説明が難しい。

現時点では、全ビーム nullspace 方式は比較対象に留める。

---

## 6. 共分散の作り方

blocking 後 reference を `U`、main path を `d` とする。

```text
U[f]: [n_ref, n_frame]
d[f]: [n_frame]
```

reference 共分散は、

\[
R_{uu}[f]
=
\frac{1}{K}U[f]U^H[f]
\]

target-reference 相互相関は、

\[
r_{ud}[f]
=
\frac{1}{K}U[f]d^*[f]
\]

である。

SLC 係数は、

\[
w[f]
=
\left(R_{uu}[f]+\lambda[f]I\right)^{-1}r_{ud}[f]
\]

とする。

出力は、

\[
y[f,k]
=
d[f,k]
-
\eta[f]w^H[f]u[f,k]
\]

である。

---

## 7. main path の選び方

beam-domain GSC では main path の定義も選択肢になる。

## 7.1 target beam をそのまま使う

最も単純な方式である。

```text
d[f,k] = X[f][target_beam, k]
```

利点:

- 現在の BL / FRAZ / BTR 評価と整合する。
- target beam の before / after 比較が直感的である。

欠点:

- target 方位がビーム格子の間にある場合、main path に量子化誤差が残る。

## 7.2 target 応答に整合する beam-domain fixed combiner を使う

target 応答 `a_t` を使って、周辺複数ビームから main path を作る方式である。

```text
d[f,k] = c_t^H[f] x[f,k]
```

ただし、この方式は出力ビーム軸の意味が変わるため、現時点では採用しない。

現段階では、main path は target beam そのものとする。

---

## 8. 推奨方式

現時点で推奨する beam-domain GSC は次である。

```text
方式:
  guard 外 reference beam + response-vector blocking + covariance SLC

main path:
  target beam 出力

reference path:
  guard 外ビーム出力

blocking:
  reference beam 上の target 応答 a_R を projector で除去

covariance:
  blocking 後 reference U から R_uu を作る

weight:
  (R_uu + loading I)^-1 r_ud

output:
  y = d - eta * w^H u
```

この方式は、現行実装を自然に拡張でき、固定整相後のビーム出力から共分散を作るという要求にも合う。

---

## 9. STFT / サブバンド化した場合の処理単位

拡張実装では、周波数ごとに beam-domain GSC を行う。初期本番経路は時間領域 SLC L=1 とし、本節はその後の周波数依存拡張を定義する。

```text
X: [n_beam, n_freq, n_frame]
```

周波数 `f` ごとに次を行う。

1. target beam index を決める。
2. 周波数依存 guard `guard[f]` から reference beam を決める。
3. 固定整相の beam response matrix から `a_R[f]` を得る。
4. `P_R[f]` で reference を blocking する。
5. `R_uu[f]` と `r_ud[f]` を更新する。
6. `w[f]` を解く。
7. `eta[f]` を掛けて target beam から差し引く。

---

## 10. 既存実装との対応

現行の `_apply_frequency_selective_scan_slc()` は、既に以下を行っている。

- `BeamGuardSelector` による reference beam 選定
- `response_matrix[selected_reference_beams, beam_index]` による `a_R` の取得
- `P = I - aa^H / (a^H a)` による desired blocking
- blocking 後 reference からの共分散作成
- `BlockLeastSquaresSlcSolver` による重み計算
- `eta_limited / eta_normal` によるキャンセル強度制御

したがって、beam-domain GSC 検討としては、現行診断コードを捨てる必要はない。むしろ、現行コードを以下のように整理するのが妥当である。

- 診断用 narrowband GSC として位置づける。
- STFT / サブバンド版では、同じ計算を周波数ビンごとに一般化する。
- guard / eta / loading を周波数依存テーブルから読む。

---

## 11. 評価で見るべき指標

beam-domain GSC では、以下を分けて評価する。

```text
mainlobe_peak_shift_deg
mainlobe_level_delta_db
sidelobe_reduction_db
mirror_reduction_db
local_margin_improvement_db
condition_number_Ruu
weight_norm
selected_reference_beam_count
disabled_or_limited_count
```

特に、`mainlobe_level_delta_db` と `sidelobe_reduction_db` は同時に見る必要がある。sidelobe が下がっても mainlobe が同程度以上に下がる場合は、方式としては不十分である。

---

## 12. 懸念点

## 12.1 response matrix の精度

blocking は `a_R` の精度に依存する。小数遅延固定整相では FIR 応答込みの `steering_response()` を使う必要がある。

整数遅延近似の response matrix を使うと、blocking がずれて desired leakage が残り、self-nulling が増える可能性がある。

## 12.2 ビーム間相関の高さ

隣接ビームは強く相関する。reference beam が多すぎると、`R_uu` が悪条件になりやすい。

対策:

- reference beam 間引き
- 対角 loading
- `sample_per_dof` による自由度制限
- condition number による fallback

## 12.3 target が複数ある場合

複数 target の場合、blocking で消すべき desired 応答は 1 本ではない。

target 応答行列を、

```text
A_T[f]: [n_ref, n_target]
```

とし、

\[
P_R[f]
=
I
-
A_T[f]\left(A_T^H[f]A_T[f]\right)^{-1}A_T^H[f]
\]

を使う必要がある。

target 応答同士が近接して悪条件になる場合は、擬似逆または対角 loading が必要である。

---

## 13. 段階的な実装方針

本章は、時間領域 SLC L=1 の評価後に周波数依存拡張が必要になった場合の段階である。初期本番経路の実装順ではない。

## 13.1 第1段階

現行 narrowband 診断を beam-domain GSC 診断として整理する。

確認項目:

- response-vector blocking の有無で比較する。
- `R_uu` の条件数を出す。
- `w` のノルムを出す。
- `eta` sweep を行う。

## 13.2 第2段階

周波数依存 guard / eta / loading を入力テーブル化する。

確認項目:

- 低域では保守的にする。
- 高域では sidelobe reduction を重視する。
- `mainlobe_level_delta_db` の上限を設ける。

## 13.3 第3段階

STFT / サブバンド版 beam-domain GSC を実装する。

処理単位:

```text
X[beam, freq, frame]
```

出力はまず再合成せず、BL / FRAZ / BTR 評価と summary JSON を優先する。

---

## 14. 結論

固定整相後のビーム出力から共分散を作る beam-domain GSC としては、次を拡張候補にする。

```text
guard 外 reference beam
  + target response vector による blocking projector
  + blocking 後 reference covariance
  + loaded least-squares canceller
  + frequency-dependent guard / eta / loading
```

これは、現行の narrowband SLC 診断を理論的に整理した方式であり、STFT / サブバンド化にも自然に拡張できる。

ただし、STFT / サブバンド化は周波数依存の guard、eta、loading、blocking を持てるようにする手段であり、それだけで固定整相 BL 全体の sidelobe envelope が下がる保証はない。後段 GSC の目的関数が target beam 出力 power または局所 marker 低減に寄ったままなら、制約されていない guard 外方位へ sidelobe が押し出される可能性が残る。

したがって、STFT / サブバンド版 beam-domain GSC へ進む場合は、周波数ビンごとの SLC を単純に並べるのではなく、次を同時に評価する。

```text
- 周波数別 guard / reference / loading / eta
- dense angle grid 上の guard 外 peak
- 第一副極改善量
- percentile / integrated sidelobe level
- white noise gain または weight norm
- train / test 分離した off-grid 干渉条件
```

次に必要なのは、時間領域 SLC L=1 と時間領域 MVDR / LCMV / GSC の評価結果を踏まえ、beam-domain GSC を「BL 全体の sidelobe reducer」と「局所 leakage canceller」に分けて再定義することである。BL 全体の低減を狙う場合は、固定整相側の shading / active aperture 設計、または周波数別・dense sector envelope 制約付きの重み設計を先に検討する。局所 leakage 低減を狙う場合に限り、本書の beam-domain GSC を帯域別または周波数ビン別の拡張として実装する。
