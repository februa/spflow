# Nonuniform FilterBank 処理量見積り

## 注意

本書の数値は、2026-07-02 時点の現行正式実装
`leaf_independent_one_sided + OLS output path`
に合わせて更新したものである。

重要なのは、`1672 bin` という値は

- one-side 統計経路で同時に保持する状態数
- 重み更新経路で扱う正側 bin 数

を表す一方、実時間の output path は
one-side 重みから作った full complex `filter FFT` を用いるため、
`A. 実時間パス` の beamforming 内積は `1672 bin` 基準ではなく
full `N_l` bin 基準で数える必要がある点である。

---

## 1. 目的

本書は、不均一フィルタバンク方式の処理量比較を

- `A. 実時間パス`
- `B. 重み更新パス`
- `C. 固定コスト`

に分離して整理するための文書である。

ここで

- `A. 実時間パス` は毎 frame 必ず走る `FFT -> beamforming 内積 -> IFFT`
- `B. 重み更新パス` は更新周期を落とせる `Rxx` 更新と MVDR 重み計算
- `C. 固定コスト` は beamforming 条件に依らず走る `front-end` と nonuniform tree

を指す。

以後、方式そのもののリアルタイム性比較は `A` を主比較対象とし、
`B` は `1 update あたり` の比較、`C` は別加算とする。

---

## 2. 比較条件

### 2.1 基準 block

実時間パス比較では、analytic 複素入力 `32768 samples` を
1 block 処理するときの計算量を比べる。

このとき比較対象は

- full-band 一括方式: `32768 FFT -> bin-wise beamforming -> 32768 IFFT`
- nonuniform leaf 方式: leaf ごとの shared frame `FFT -> full complex filter FFT 適用 -> IFFT` の総和

である。

ここでは

- `Cfft(N) = N log2(N)`

を FFT / IFFT の近似コストとして用いる。

### 2.2 現行 nonuniform leaf 条件

`Nsig = 32768` の block を現行 formal tree へ通したとき、
各 leaf の代表条件は以下である。

| band | leaf packet 長 `L` | used channels `m_l` | FFT size `N_l` | hop `H_l` | frame 数 `F_l` | one-sided bins `B_l` |
|---|---:|---:|---:|---:|---:|---:|
| `0-128 Hz` | `263` | `32` | `256` | `128` | `3` | `129` |
| `128-256 Hz` | `263` | `32` | `256` | `128` | `3` | `129` |
| `256-512 Hz` | `520` | `24` | `256` | `128` | `5` | `129` |
| `512-1024 Hz` | `1034` | `20` | `512` | `256` | `5` | `257` |
| `1024-2048 Hz` | `2062` | `16` | `512` | `256` | `9` | `257` |
| `2048-4096 Hz` | `4118` | `12` | `512` | `256` | `17` | `257` |
| `4096-8192 Hz` | `8229` | `8` | `512` | `256` | `33` | `257` |
| `8192-16384 Hz` | `16451` | `4` | `512` | `256` | `65` | `257` |

補足:

- `F_l` は `ceil(L / H_l)` であり、overlap-save buffer の初期ゼロ履歴と末尾 flush を含んだ runtime frame 数である
- one-side 統計経路として同時に保持する正側 bin 数は `1672` である
- output path で実際に内積する full complex bin 数は、leaf 同時保持で `3328` である
- `32768 sample` 1 block の統計経路側 bin 出現回数は `34572`、output path 側 full bin 出現回数は `68864` である
- したがって `1672 / 16385` は統計状態数の比較には使えるが、1 block の実時間内積コスト比には使えない

---

## 3. A. 実時間パス比較

### 3.1 full-band 一括方式

32 ch, analytic 正側 `16385 bins` を持つ `32768 FFT` 一括方式では、
1 block あたりの実時間パスは

```text
Cost_rt_full
  = 32 * Cfft(32768)
  + 32 * 16385
  + Cfft(32768)
```

である。

数値化すると

- `32 ch FFT`: `15.72864M`
- beamforming 内積: `0.52432M`
- `1 beam IFFT`: `0.49152M`
- 合計: `16.74448M / block`

となる。

### 3.2 nonuniform leaf 方式

現行正式実装では、statistics path は one-side だが、
output path は one-side 重みから作った full complex `filter FFT`
を overlap-save で適用する。

したがって 1 block あたりの実時間パスは

```text
Cost_rt_nonuni_formal
  = sum_l F_l * (m_l * Cfft(N_l) + m_l * N_l + Cfft(N_l))
```

で与えられる。

ここで `m_l * N_l` になっている理由は、
runtime の beamforming 内積が正側 `B_l` ではなく full `N_l`
bin 上で走るためである。

band ごとの内訳は以下である。

| band | FFT cost | beamforming 内積 | IFFT cost | subtotal |
|---|---:|---:|---:|---:|
| `0-128 Hz` | `196608` | `24576` | `6144` | `227328` |
| `128-256 Hz` | `196608` | `24576` | `6144` | `227328` |
| `256-512 Hz` | `245760` | `30720` | `10240` | `286720` |
| `512-1024 Hz` | `460800` | `51200` | `23040` | `535040` |
| `1024-2048 Hz` | `663552` | `73728` | `41472` | `778752` |
| `2048-4096 Hz` | `940032` | `104448` | `78336` | `1122816` |
| `4096-8192 Hz` | `1216512` | `135168` | `152064` | `1503744` |
| `8192-16384 Hz` | `1198080` | `133120` | `299520` | `1630720` |

合計は

- leaf FFT 総和: `5.117952M`
- beamforming 内積総和: `0.577536M`
- leaf IFFT 総和: `0.616960M`
- 合計: `6.312448M / block`

である。

### 3.3 実時間パスの解釈

この節で言いたいことは、

- 全体の実時間処理は nonuniform leaf 方式の方が軽い
- ただし、軽くなっている主因は beamforming 内積ではなく `FFT/IFFT` である

という 2 点である。

まず総実時間パスでは

- full-band 一括方式: `16.74448M / block`
- nonuniform leaf 方式: `6.312448M / block`
- 比: `6.312448 / 16.74448 = 0.377`

なので、nonuniform leaf 方式は full-band 一括方式の約 `37.7%`、
言い換えると約 `2.65x` 軽い。

次に `FFT/IFFT` だけをまとめて見ると

- full-band 一括方式: `15.72864M + 0.49152M = 16.22016M / block`
- nonuniform leaf 方式: `5.117952M + 0.616960M = 5.734912M / block`
- 比: `5.734912 / 16.22016 = 0.354`

であり、`FFT/IFFT` 部は約 `35.4%`、
言い換えると約 `2.83x` 軽い。

一方、beamforming 内積だけを見ると

- full-band 一括方式: `0.52432M / block`
- nonuniform leaf 方式: `0.577536M / block`
- 比: `1.101`

であり、内積だけでは nonuniform leaf 方式の方が約 `10.1%` 重い。

したがって、現行正式実装の利得は

- one-side 化した統計経路で重み更新側の状態数を減らしていること
- high band で `used_channels` を減らしていること
- 小さい FFT に分割して `FFT/IFFT` コストを大きく下げていること

の合成効果である。

逆に言えば、`1672 bin` という値は
重み更新経路と状態数の削減には効いているが、
現行正式 OLS output path の beamforming 内積そのものが
`1/10` になることを意味する値ではない。

---

## 4. B. 重み更新パス比較

### 4.1 比較単位

重み更新パスは実時間パスと更新周期が異なるため、
`1 update あたり` の計算量として別比較する。

ここでの比較対象は

- covariance 更新
- diagonal loading
- MVDR 線形方程式解法
- 必要なら重みの次数変換

である。

更新レート `U_w` は運用条件依存なので、方式自体の比較では固定しない。
必要なら最後に

```text
total_weight_cost_per_second = U_w * Cost_weight_per_update
```

として秒換算する。

### 4.2 full-band 一括方式の式

full-band 一括方式では、正側 `16385 bins`, `32 ch` を持つので
1 update あたりの代表式は

```text
Cost_w_full
  ~= 16385 * (alpha_cov * 32^2 + alpha_solve * 32^3 + alpha_misc * 32^2)
```

と書ける。

ここで

- `alpha_cov`: covariance 更新の定数因子
- `alpha_solve`: MVDR solve の定数因子
- `alpha_misc`: loading や正規化などの定数因子

である。

### 4.3 nonuniform leaf 方式の式

nonuniform leaf 方式では、leaf ごとに bin 数と使用チャネル数が異なるため

```text
Cost_w_nonuni
  ~= sum_l B_l * (alpha_cov * m_l^2 + alpha_solve * m_l^3 + alpha_misc * m_l^2)
```

となる。

現行 leaf 条件に対して、更新レートを除いた構造比較を行うと

- 総 one-sided bins: `1672`
- `sum_l B_l * m_l^2 = 564656`
- `sum_l B_l * m_l^3 = 13938240`

である。

一方、full-band 一括方式では

- `16385 * 32^2 = 16778240`
- `16385 * 32^3 = 536903680`

なので、構造的な比較比は

- covariance 規模 proxy: `564656 / 16778240 = 0.0337`
- solve 規模 proxy: `13938240 / 536903680 = 0.0260`

となる。

したがって、重み更新パスは更新レートを無視した構造比較だけ見ても、
full-band 一括方式よりかなり小さい条件で計算できる可能性が高い。

### 4.4 現時点での注意

ただし、重み更新パスについては

- 実際の update scheduler
- covariance 積分時間
- update の間引き率
- 次数変換の実装方式

で最終値が変わる。

よって本書では、`B` については

- まず `1 update あたり` の構造比較を固定する
- `1 秒あたり` の比較は運用 update rate が決まってから載せる

方針とする。

---

## 5. C. 固定コスト

`front-end` と nonuniform tree は、beamforming 重み更新とは別に
常時走る固定コストとして扱う。

### 5.1 nonuniform tree

正式実装方式としては

- `ComplexPRHalfbandStage` の stage-level streaming を stateful polyphase FIR とする

方針を採る。

このとき internal stage `v` のコストは近似的に

```text
Cost_stage(v) ~= f_v * (L_a(v) + L_s(v))
```

となるため、tree 全体は

```text
Cost_tree ~= sum_v f_v * (L_a(v) + L_s(v))
```

で線形に見積もれる。

今回の 8-leaf 非対称木では internal node rate の総和は `65024 Hz` なので、
全 stage が同一 tap 長 `L_a = L_s = L` なら

```text
Cost_tree ~= 65024 * 2L
```

となる。

### 5.2 causal analytic front-end

`causal analytic front-end` は tree 外側の固定コストとして別計上する。
現時点では構造は固定したが、正式な処理量比較値はまだ未算入である。

---

## 6. Python 実装上の補足

Python 正式実装の定数因子削減として、2026-07-02 時点で以下は実装済みである。

- `used_channels` が contiguous な leaf では slice selection を使う
- covariance 更新を bandwise vectorize する
- MVDR 重み更新を stacked linear solve へまとめる
- output path と statistics path で frame FFT を共有する

実測では、`short_fft_size = 128`, `n_used_channels = 16` の比較で
約 `2.55x` の高速化を確認した。

したがって、今後の最適化対象は

- leaf 全体 end-to-end の残る定数因子
- tree / front-end / beamformer の合算最適化
- C++ 実装前提の更なる削減

へ移っている。

---

## 7. Pending

### P-COST-1

各 leaf の正式 `used_channels` が最終固定前である。

### P-COST-2

重み更新パスの正式 update rate が未固定である。

### P-COST-3

`causal analytic front-end` の正式処理量が未算入である。

### P-COST-4

nonuniform tree の正式 tap 長固定後の総コスト算入が未完了である。

### P-COST-5

重み更新パスにおける次数変換コストの正式算入が未完了である。

---

## 8. 現時点での結論

現時点では、処理量について以下のように整理できる。

1. 方式比較は `A. 実時間パス`, `B. 重み更新パス`, `C. 固定コスト` に分けるべきである
2. `32768 sample` 1 block の実時間パスでは、現行 nonuniform leaf 方式は full-band 一括方式の約 `37.7%` である
3. beamforming 内積だけを見ると約 `110.1%` であり、現行正式 OLS output path の利得本体は `used_channels` 削減と小 FFT 化にある
4. 重み更新パスは、更新レートを除いた構造比較でも full-band 一括方式よりかなり小さい条件に落とせる
5. 最終総処理量は `front-end` と tree 本体を別加算して確定する

したがって、本書の今後の正式比較手順は

- まず `A. 実時間パス` を方式比較の主指標として固定する
- 次に `B. 重み更新パス` を `1 update あたり` で整理する
- 最後に `C. 固定コスト` を加算して end-to-end の総処理量を確定する

という順序で進める。
