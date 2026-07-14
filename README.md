# spflow

`spflow` is a lightweight utility package for writing sequential signal processing code in plain Python.

It provides five focused building blocks:

- `Option`: dot-access wrapper around nested dictionaries
- `Flow`: lightweight container for zero, one, or many values
- `FrameBuffer`: frame slicing utility with overlap support
- `StepScheduler`: scheduler for iterative item processing
- `DoubleBufferCallback`: callback base that publishes only completed values

## Install From GitHub

Clone the repository:

```bash
git clone https://github.com/februa/spflow.git
cd spflow
```

Install it into your current Python environment:

```bash
pip install .
```

If you want to modify `spflow` while developing another project, use editable install instead:

```bash
pip install -e .
```

## API Documentation

実装済み機能は、リポジトリ内の
[`doc/SpFlow/実装済み機能一覧.md`](doc/SpFlow/実装済み機能一覧.md) から責務と import パスを検索できる。
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

## Beamforming Evaluation Environment

Beamforming evaluations that render figures or use the vendored scene renderer need the optional development tools and vendor package.

For a full local checkout with the vendored `scene_renderer` submodule:

```bash
git submodule update --init --recursive
pip install -e ".[dev,beamforming-eval]"
pip install -e vendor/scene_renderer
```

If you do not use the vendored submodule and want pip to install `scene_renderer` from GitHub instead:

```bash
pip install -e ".[dev,beamforming-eval,vendor]"
```

The streaming diff-MVDR covariance comparison accepts a JSON parameter file:

```bash
python evaluations/beamforming/evaluate_streaming_diff_mvdr_covariance_compare.py \
  --config evaluations/beamforming/streaming_diff_mvdr_covariance_compare_config.json
```

To regenerate the default 3 second evaluation config:

```bash
python evaluations/beamforming/evaluate_streaming_diff_mvdr_covariance_compare.py \
  --write-default-config evaluations/beamforming/streaming_diff_mvdr_covariance_compare_config.json
```

The config controls sampling rate, channel count, FFT length, integration duration, beam axis, output directory, and source scenarios.
## Use From Another Project

After installation, you can import `spflow` from any other project in the same environment.

```python
from spflow import Flow, FrameBuffer, Option
```

Example project layout:

```text
my_project/
├── main.py
└── venv/  # optional
```

Example `main.py`:

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

## Example

A runnable example is included in [examples/streaming/basic_pipeline.py](examples/streaming/basic_pipeline.py).

Run it after installation:

```bash
python -m examples.streaming.basic_pipeline
```

It demonstrates:

- `Option` for nested configuration
- `FrameBuffer` for overlapped framing
- `Flow` for propagating zero, one, or many outputs

`None`を入力のない周期として現在段へ通知し、完成出力がない周期では後段を呼ばない例もある。

```bash
python -m examples.streaming.none_cycle
```

この例では、状態を持つ処理を4周期すべてで更新しつつ、2周期ごとの完成値だけを
後段へ渡す。`Flow`は周期を決めず、各段の0個・1個・複数個の出力だけを接続する。

A deterministic delay-and-sum example without an external scene renderer is also available:

```bash
python -m examples.beamforming.delay_and_sum
```

This example first verifies synthetic plane-wave tone generation: requested RMS level,
per-channel arrival phase, and the level after delay-and-sum. It intentionally does not
generate BL metrics or make an adoption decision.

Core example:

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
