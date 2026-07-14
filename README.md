# spflow

`spflow`は、逐次信号処理を通常のPythonコードとして記述するための軽量な部品集である。

責務を限定した次の5部品を提供する。

- `Option`: ネストした辞書の値をdot accessで参照する
- `Flow`: 0個・1個・複数個の値を同じinterfaceで次段へ運ぶ
- `FrameBuffer`: overlapを含む固定長frameの切り出しを行う
- `StepScheduler`: 反復item処理を複数stepへ分割する
- `DoubleBufferCallback`: 完成した値だけを外部へ公開する

## GitHubからのインストール

リポジトリをcloneする。

```bash
git clone https://github.com/februa/spflow.git
cd spflow
```

現在のPython環境へインストールする。

```bash
pip install .
```

別のprojectを開発しながら`spflow`も変更する場合は、editable installを使用する。

```bash
pip install -e .
```

## APIドキュメント

実装済み機能は、リポジトリ内の
[`doc/SpFlow/実装済み機能一覧.md`](doc/SpFlow/実装済み機能一覧.md) から責務と import パスを検索できる。

リポジトリ内の実行コードは責務で配置を分けている。`examples/`は短い公開API使用例、
`evaluations/`は方式評価・sweep・diagnostics、`tools/`は運用係数や入力ファイルを生成するCLIである。
この一覧は Python の module docstring、公開クラス、公開関数、`__all__` から自動生成する。

型、引数、戻り値、docstring を含む HTML API リファレンスを生成する場合は、
ドキュメント用の追加依存を導入して生成ツールを実行する。

```bash
pip install -e ".[docs]"
python tools/build_api_docs.py
```

HTML は `build/api-docs/` に生成される。実装変更後に、コミット済みの機能一覧が
最新であることだけを検査する場合は次を実行する。

```bash
python tools/build_api_docs.py --check
```

dB入力と出力評価のreference、RMS/power、one-sided/two-sided規約を同じ変換定義で
接続する場合は`LevelConverter`を使用する。登録済み数式と責務境界は
[`doc/SpFlow/LevelConverter設計.md`](doc/SpFlow/LevelConverter設計.md)を参照する。

Beamforming関連の責務境界は次のとおりである。

- `spflow.beamforming`: beamformer、重み設計、共分散、SLCなどの信号処理本体
- `spflow.simulation`: 決定論的な入力scene、数値精度、逐次シミュレーション支援
- `spflow.beamforming_evaluation`: scan grid、level、理論応答などの小さな評価支援部品
- `evaluations/beamforming`: 個別scenario、parameter sweep、方式比較、成果物生成

詳細は[`doc/SpFlow/beamforming責務分割設計.md`](doc/SpFlow/beamforming責務分割設計.md)を参照する。

## Beamforming評価環境

図を生成するBeamforming評価や、同梱したscene rendererを使う評価には、追加の開発toolとvendor packageが必要になる。

`scene_renderer` submoduleを含む一式をlocalへ取得した場合は、次のようにインストールする。

```bash
git submodule update --init --recursive
pip install -e ".[dev,beamforming-eval]"
pip install -e vendor/scene_renderer
```

同梱したsubmoduleを使わず、pipでGitHubから`scene_renderer`をインストールする場合は、次を実行する。

```bash
pip install -e ".[dev,beamforming-eval,vendor]"
```

逐次diff-MVDRの共分散比較では、JSON parameter fileを指定できる。

```bash
python evaluations/beamforming/evaluate_streaming_diff_mvdr_covariance_compare.py \
  --config evaluations/beamforming/streaming_diff_mvdr_covariance_compare_config.json
```

既定の3秒評価用configを再生成する場合は、次を実行する。

```bash
python evaluations/beamforming/evaluate_streaming_diff_mvdr_covariance_compare.py \
  --write-default-config evaluations/beamforming/streaming_diff_mvdr_covariance_compare_config.json
```

configでは、sampling rate、channel数、FFT長、積分時間、beam軸、出力directory、source scenarioを指定する。

## 他のprojectから使う

インストール後は、同じPython環境の他のprojectから`spflow`をimportできる。

```python
from spflow import Flow, FrameBuffer, Option
```

project構成例を次に示す。

```text
my_project/
├── main.py
└── venv/  # optional
```

`main.py`の例を次に示す。

```python
import numpy as np

from spflow import Flow, FrameBuffer, Option


opt = Option(
    {
        "stft": {
            "nfft": 4,
            "hop": 2,
        }
    }
)

buffer = FrameBuffer(
    frame_size=opt.stft.nfft,
    hop_size=opt.stft.hop,
)

x = np.arange(8, dtype=float)
frames = Flow.from_value(x).map(buffer.process).to_list()

print(len(frames))
print(frames[0])
```

## 実行例

実行可能な最小例は[examples/streaming/basic_pipeline.py](examples/streaming/basic_pipeline.py)にある。

インストール後に次を実行する。

```bash
python -m examples.streaming.basic_pipeline
```

この例では、次の部品の組み合わせを確認できる。

- `Option`によるネストした設定値の参照
- `FrameBuffer`によるoverlap付きframeの切り出し
- `Flow`による0個・1個・複数個の出力の伝播

`None`を入力のない周期として現在段へ通知し、完成出力がない周期では後段を呼ばない例もある。

```bash
python -m examples.streaming.none_cycle
```

この例では、状態を持つ処理を4周期すべてで更新しつつ、2周期ごとの完成値だけを
後段へ渡す。`Flow`は周期を決めず、各段の0個・1個・複数個の出力だけを接続する。

`StepScheduler`の最新完成値を毎周期使う経路と、新しい完成値だけをFlowで通知する経路の
違いは、次の最小例で確認できる。

```bash
python -m examples.streaming.step_scheduler_completion
```

同じ周期の処理がSchedulerの完成状態に依存する場合は`StepResult`を通常のPythonで確認し、
独立した完成通知だけを`updated_value()`からFlowへ接続する。

`Flow.map()`は遅延DAGを構築せず、呼び出すたびに保持中の全項目へ即時適用する。
同じFlowへ複数回`map()`しても項目単位の分岐にはならない。分岐間に状態や実行順序の
依存関係がある場合は、`to_list()`で通常のPythonへ戻し、項目ごとのループで明示する。
正しい使用例と期待どおりにならない例は
[`SpFlow 基本設計書`](doc/SpFlow/SpFlow_基本設計書.md)の「7. Flow の設計」に示す。

FFTまでの0個・1個・複数個のframeを`Flow`で運び、完成frameごとの共分散計算、
MVDR係数設計、完成係数の置換、`h^T x`を通常のPythonで順序付ける例もある。

```bash
python -m examples.beamforming.streaming_mvdr_weights
```

各完成FFT frameで係数設計を先に進め、設計が完了すれば同じframeへ新係数を適用する。
設計が複数frameにまたがる場合は、全帯域が完成するまで固定CBFまたは前回完成した適応係数を
使用する。1 chunkから複数frameが完成しても、この順序をframeごとに維持する。

外部のscene rendererを使わない、決定論的なdelay-and-sumの例も用意している。

```bash
python -m examples.beamforming.delay_and_sum
```

この例では、合成平面波toneの指定RMS level、channelごとの到来位相、
delay-and-sum後のlevelを確認する。BL指標の生成や方式の採否判定は扱わない。

中心となる処理は次のとおりである。

```python
from types import SimpleNamespace

import numpy as np

from spflow import Flow, FrameBuffer, Option


def make_env(opt):
    env = SimpleNamespace()
    env.opt = opt
    env.input_buffer = FrameBuffer(
        frame_size=opt.stft.nfft,
        hop_size=opt.stft.hop,
        axis=-1,
    )
    return env


def calc_fft(frame, env):
    return np.fft.fft(frame, n=env.opt.stft.nfft, axis=-1)


def calc_power(x):
    return np.abs(x) ** 2


def process_frame(x, env):
    return (
        Flow.from_value(x)
        .map(env.input_buffer.process)
        .map(calc_fft, env)
        .map(calc_power)
        .to_list()
    )
```
