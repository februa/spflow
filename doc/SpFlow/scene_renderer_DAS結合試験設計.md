# scene_renderer・delay-and-sum結合試験設計

## 1. 目的

scene_rendererが生成する多CH波面をspflowの通常delay-and-sumへ接続し、両プロジェクト間の
ArrayFrame、到来方向、到達遅延、位相符号、channel順序、音速単位が一致することを確認する。

scene_renderer単体試験はCH間位相・遅延を解析式と比較する。本結合試験は、その波面をspflowの
steeringとDAS重みで整相したとき、所望の水平方位・俯仰方位にビーム出力ピークが現れることを確認する。

## 2. シーン分離方針

複数方位の音源を同時に入力しない。各azimuth/elevation条件を独立したpytest parameterとして、
1 sceneにつき1 target toneだけを生成する。これにより、他音源のメインローブ、サイドローブ、
位相干渉を方位推定誤差と混同しない。

## 3. アレイと走査条件

ULAでは水平全方位と俯仰を一意に識別できないため、3x3x3、27素子の非共面アレイを使用する。
素子座標はArrayFrameのBow/Starboard/Up各軸で`[-0.12, 0, 0.12] m`とする。

tone周波数は2048 Hz、音速は1500 m/sで、波長は約0.732 mである。素子間隔0.12 mは半波長未満のため、
本試験条件では空間aliasを避けられる。

waiting directionは次とする。

```text
azimuth:  -180 degから160 degまで20 deg間隔
elevation: -60, -30, 0, 30, 60 deg
```

入力方位は水平四象限、背面、正負俯仰へ分散し、全点を別々に描画する。
実運用で俯仰をpresetまたは直接値として指定することを反映し、真上`+90 deg`と真下`-90 deg`は
水平beam gridへ重複配置せず、それぞれ1つの直接指定候補として追加する。極ではazimuthが定義されないため、
azimuth誤差ではなく真値・推定値の単位方向ベクトル間角距離を評価する。

receiver headingは`0, 30, 45, 123, 210, 270 deg`を含め、WorldFrame上の向きが変わっても
ArrayFrame相対方向へ同じDAS steeringで集束することを確認する。

## 4. DASと評価指標

spflowの`relative_arrival_delay()`、`steering_from_relative_delay()`、`design_cbf_weights()`、
`apply_beamformer()`を用いる。重みは`w=a/(a^H a)`、出力は`y=w^H x`である。

評価指標は次とする。

- peak azimuth circular error: 走査grid上で0 deg
- peak elevation error: 走査grid上で0 deg
- 所望方向の出力RMS: 1.0
- 所望方向の出力level: 0 dB re input RMS

広帯域入力では500～1500 Hzの帯域制限雑音音源を各方向へ個別に配置し、周波数binごとのDAS出力を
one-sided RMS powerとして帯域積分する。band-integrated power最大方向の方向ベクトル角距離を評価し、
所望方向の出力band RMSが入力RMS 1を保存することを確認する。

noise-onlyでは空間白色背景雑音を入力し、矩形DASの理論値

```text
P_out = sigma_in^2 * sum_ch |w_ch|^2 = sigma_in^2 / N
array gain = 10log10(N)
```

と観測出力を比較する。target+noiseでは同一DAS重みをtarget-only、noise-only、mixedへ適用し、
無相関成分について`P_mixed ≈ P_target + P_noise`となることを確認する。

このnoise-only結合試験により、scene_rendererが`seed+channel`をsample indexへ直接XORしていた実装で
CH間相関が残ることを検出した。seed自体をSplitMix64でhashしてからsample indexと混合するよう修正し、
scene_renderer単体試験でもcovariance未指定時の非対角相関を監視する。

本試験は方式比較、parameter sweep、採否判定を扱わない。DASを基準器として、scene_rendererとspflowの
方位・位相契約が閉じていることを検証する結合試験である。
