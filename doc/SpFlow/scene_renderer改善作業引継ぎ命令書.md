# scene_renderer 改善作業引継ぎ命令書

## 1. この文書の使い方

この文書は、`scene_renderer` の別プロジェクトを開いた Codex に渡し、そのプロジェクト内で改善作業を継続するための命令書である。

作業者は、まず対象プロジェクトの `AGENTS.md`、README、公開 API、テスト、依存関係を確認し、本書と競合する規約がある場合は対象プロジェクトの規約を優先すること。既存 API の実態を確認せず、本書のクラス名やモジュール構成をそのまま実装してはならない。本書が固定するのは責務、物理量、結果契約、受入条件であり、識別子は既存設計に合わせてよい。

## 2. 作業目的

`scene_renderer` を、音源波形とアレイ受信信号の生成責務を一貫して持つ部品へ改善する。

現在の利用側では、評価スクリプトごとに次の処理を実装している。

- receiver、array、source、scene、時間軸の構築
- tone の RMS level から peak amplitude への変換
- source 方位と receiver pose の関連付け
- renderer 出力の実数化と dtype 変換
- source の基準波形の再生成
- renderer 外部での白色雑音生成と加算
- target-only、interference-only、noise-only、mixed を得るための再レンダリング

これらは方式比較や parameter sweep 固有の処理ではない。`scene_renderer` 側で定義を一意にし、利用側が信号生成規約を再実装しなくてよい状態にする。

## 3. 責務境界

### 3.1 scene_renderer が持つ責務

- tone、帯域雑音などの音源波形生成
- SL、NLと生成振幅の対応
- target、interferer、ambient noise の生成
- 音源から各受波器までの伝搬
- receiver pose と array geometry の適用
- source、interference、noise、mixed の成分別出力
- 使用した時間軸、受波器位置、乱数seed、レベル定義などの再現情報
- 入力条件と実際の生成信号が一致することの検証可能性

### 3.2 scene_renderer が持たない責務

- steering vector の設計
- 共分散行列の推定
- CBF、MVDR、SLCなどの重み設計
- ビームフォーマ出力の計算
- BL、FRAZ、BTRの評価
- 方式比較、parameter sweep、採否判定

これらは利用側のアレイ信号処理ライブラリまたは評価コードが持つ。

### 3.3 禁止する曖昧な名称

新しいパッケージ、モジュール、クラス、関数に `core` という名称を使わないこと。名称から、scene定義、level変換、waveform生成、propagation、render結果などの責務が分かるようにする。

## 4. 現行APIの扱い

既存の低水準APIが次の形で存在する場合、直ちに削除しない。

```python
rendered = SceneRenderer().render(scene, receiver, axis_t)
```

このAPIは任意の既存`Scene`を描画する低水準APIとして維持する。改善では、この上に評価・検証で安全に使える高水準APIを追加する。

既存 API の破壊的変更が必要な場合は、呼び出し箇所、移行方法、非推奨期間を調査し、設計書に記録してから行うこと。

## 5. 必須のレベル定義

### 5.1 tone

`SL`を実正弦波のRMS levelとして指定する場合、RMS amplitudeは次である。

\[
A_{\mathrm{tone,rms}} = 10^{SL/20}
\]

実正弦波を生成するときのpeak amplitudeは次である。

\[
A_{\mathrm{tone,peak}} = \sqrt{2}\,10^{SL/20}
\]

API上で`level`や`amplitude`だけを受け取り、RMSかpeakかを暗黙にしてはならない。少なくとも型、フィールド名、列挙値のいずれかで基準を明示する。

### 5.2 白色雑音・帯域雑音

`NL`をone-sided amplitude spectral density levelとし、単位基準を `dB re reference/sqrt(Hz)` とする。one-sided帯域幅を `B` Hzとしたとき、帯域積分後のRMS amplitudeは次である。

\[
A_{\mathrm{noise,rms}} = 10^{NL/20}\sqrt{B}
\]

DCからNyquistまでの理想的なone-sided白色雑音では `B = f_s/2` なので、sample RMSは次になる。

\[
\sigma_{\mathrm{sample}} = 10^{NL/20}\sqrt{f_s/2}
\]

特定の解析帯域幅が256 Hzなら、その帯域内RMSは次である。

\[
A_{256\,\mathrm{Hz,rms}} = 10^{NL/20}\sqrt{256}
\]

`sqrt(256)`で割る式は、既に積分した帯域を256個の等帯域へ分割する場合など、元の帯域との関係を定義したときだけ成立する。「256 Hz分解能」という表現だけで割ってはならない。

### 5.3 dB表記

`dB`は単独の物理単位ではない。入力フィールド、結果metadata、テスト名、文書では、少なくとも次を区別する。

- tone RMS level: `dB re <reference> RMS`
- tone peak amplitude: linear peak amplitude
- noise ASD level: `dB re <reference>/sqrt(Hz)`
- band-integrated noise RMS level: `dB re <reference> RMS over <band>`
- sample RMS amplitude: linear RMS amplitude

## 6. 必要な高水準入力契約

具体的な識別子は既存設計に合わせてよいが、少なくとも次を表現できる不変な設定型を設ける。

### 6.1 tone source

- 一意な成分IDまたは名前
- target、interfererなどのrole
- azimuth、elevation、distance
- frequency [Hz]
- RMS SLとその基準、または明示的なpeak amplitude
- envelope
- 必要なら初期位相

### 6.2 noise source / ambient field

- 一意な成分IDまたは名前
- ASD NLとその基準
- one-sided有効帯域 `[low_hz, high_hz]`
- 空間モデル
- 乱数seed

最低限、次の空間モデルを区別する。

- channelごとに独立な空間白色雑音
- 単一方位から到来する帯域雑音

この2つは共分散行列が異なるため、単なる`white_noise=True`で混同してはならない。

### 6.3 receiver and sampling

- 受波器位置 `shape [n_ch, 3]`、単位m
- receiver pose
- sound speed [m/s]
- sampling frequency [Hz]
- sample countまたはduration [s]
- 出力実数/複素数の規約
- 出力dtype

## 7. 必要な出力契約

高水準APIはtupleではなく、固定された結果型を返す。少なくとも次を保持する。

```text
mixed:                  [n_ch, n_sample]
source_components:      component_id -> [n_ch, n_sample]
noise_components:       component_id -> [n_ch, n_sample]
time_s:                 [n_sample]
receiver_positions_m:   [n_ch, 3]
metadata:               level、帯域、seed、座標系、sound speedなど
```

必要であれば`interference_components`を独立させてもよい。ただし、roleによる分類とcomponent IDによる個別取得が両立しなければならない。

次の関係を数値誤差内で満たすこと。

\[
x_{\mathrm{mixed}}
=
\sum_i x_{\mathrm{source},i}
+
\sum_j x_{\mathrm{noise},j}
\]

同じsceneを成分別に再レンダリングして差を取る設計は避ける。同一の時間軸、位相、envelope、乱数実現値から、1回の描画で成分を得られること。

## 8. 座標系と位相規約

次を公開docstringと設計書に明記する。

- azimuthの0度方向と正方向
- elevationの定義
- world座標からarray座標への変換
- receiver headingの適用順序
- far-field / near-fieldの条件
- 相対到達遅延の符号
- 周波数領域で観測波面を表した場合の位相符号

既知方位の単一toneをULAへ入力し、各チャンネルの位相差が

\[
\Delta\phi = -2\pi f\Delta\tau
\]

またはプロジェクトが採用する明記済みの符号規約と一致することを試験する。符号だけを合わせるための利用側補正を前提にしてはならない。

## 9. 受入試験

### 9.1 tone level

- `SL = 0 dB re unit RMS`で十分長い実正弦波を生成する。
- peak amplitudeが`√2`に近いこと。
- 時間領域RMSが1に近いこと。
- FFTのone-sided RMS powerをtone bin周辺で積分しても1に近いこと。
- FFT binに一致しない周波数でも、十分な積分帯域を取れば同じ結果になること。

### 9.2 noise ASD and band RMS

- ASD `NL`、sampling frequency `fs`、有効帯域`B`を指定する。
- 時間領域または帯域積分スペクトルのRMSが`10^(NL/20)√B`へ統計誤差内で一致すること。
- FFT長を変更しても、同じ物理帯域を積分したRMSが変化しないこと。
- seedが同じなら同じ波形、seedが異なれば異なる波形になること。

### 9.3 spatial model

- channel-independent noiseは、十分なsample数で非対角共分散が0へ近づくこと。
- directional band noiseは、期待する到達遅延とチャンネル間相関を持つこと。
- 両者の設定型またはroleが混同されないこと。

### 9.4 component decomposition

- 2 source + 1 noiseのsceneを1回描画する。
- 個別成分の総和とmixedが許容誤差内で一致すること。
- component IDごとのshape、dtype、sample数が一致すること。
- sourceを共分散推定から除外したい利用者が、interference+noiseだけを成分和で構築できること。

### 9.5 geometry and phase

- 既知間隔のULA、既知方位、複数周波数で、解析的な到達遅延・位相差と一致すること。
- broadside、endfire、0 Hz、Nyquist近傍、負の方位を含めること。
- grating lobeを評価するのは利用側の責務だが、生成波面の位相差はaliasを含め解析式と一致すること。

### 9.6 backward compatibility

- 既存の低水準`render(scene, receiver, axis_t)`の代表試験が通ること。
- 高水準APIと同じsceneを低水準APIで描画した結果が、定義した条件下で一致すること。

## 10. 実装品質条件

- 公開クラスと公開関数に、責務、非責務、shape、axis、単位、例外、境界条件を含むdocstringを書く。
- 到達遅延、位相回転、FFT、雑音正規化、成分加算には、数式または物理的意味との対応をコメントする。
- NumPy配列には可能な範囲で`numpy.typing.NDArray`を使う。
- 戻り値の型をフラグで変えない。
- `Optional`を未確認で使用しない。
- `np.bool_`、`np.integer`、`np.floating`をPython組み込み型として暗黙に扱わない。
- 数値安定化項、許容誤差、dB閾値には理由を書く。
- formatter、linter、型チェック、全テストを実行し、結果を報告する。

## 11. 推奨作業順序

1. 現行API、内部データモデル、既存テスト、利用例を調査する。
2. tone amplitudeとnoise amplitudeの現在の意味を実装から特定する。
3. 責務境界と高水準APIの設計書をscene_rendererプロジェクト内に作る。
4. level変換を純粋関数または明示的な値型として実装し、単体試験を通す。
5. 成分IDとroleを持つ入力設定型を実装する。
6. 固定結果型と1回描画での成分分離を実装する。
7. spatially independent noiseとdirectional band noiseを実装・検証する。
8. 座標・位相規約の解析試験を追加する。
9. 低水準APIとの互換試験を追加する。
10. READMEに最小使用例とレベル規約を記載する。
11. 実装、設計書、試験を同一の変更としてcommitする。

## 12. 完了時に報告する内容

- 追加・変更した公開API
- tone SL、noise NLの厳密な意味
- 出力shape、dtype、座標・位相規約
- 成分分離の加法整合誤差
- tone RMSとnoise ASD/band RMSの測定誤差
- 空間白色雑音の共分散確認結果
- 既存APIとの互換性
- formatter、linter、型チェック、テスト結果
- spflow側で置換すべき既存`render_scene`ヘルパーの移行例

## 13. spflowへ戻すための最小使用例

scene_renderer改善後、spflow側では次の情報だけでCBF/MVDR評価へ接続できる状態を目標とする。

```python
rendered = renderer.render_components(scene_spec)

x_target = rendered.components_by_role("target")
x_interference = rendered.components_by_role("interference")
x_noise = rendered.components_by_role("noise")

# MVDRの初回校正では、目標信号による自己抑圧と実装誤りを分離するため、
# targetを含めないinterference+noiseから共分散を推定する。
x_covariance = x_interference + x_noise
x_mixed = rendered.mixed
```

`components_by_role`という名称や、複数成分を返す際の加算方法は固定しない。ただし、利用側がprivate属性やscene内部構造を参照せず、target-only、noise-only、interference+noise、mixedを明示的に取り出せることを完了条件とする。
