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

## 4. DASと評価指標

spflowの`relative_arrival_delay()`、`steering_from_relative_delay()`、`design_cbf_weights()`、
`apply_beamformer()`を用いる。重みは`w=a/(a^H a)`、出力は`y=w^H x`である。

評価指標は次とする。

- peak azimuth circular error: 走査grid上で0 deg
- peak elevation error: 走査grid上で0 deg
- 所望方向の出力RMS: 1.0
- 所望方向の出力level: 0 dB re input RMS

本試験は方式比較、parameter sweep、採否判定を扱わない。DASを基準器として、scene_rendererとspflowの
方位・位相契約が閉じていることを検証する結合試験である。
