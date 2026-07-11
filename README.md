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

## Beamforming Evaluation Environment

Beamforming examples that render figures or use the vendored scene renderer need the optional development tools and vendor package.

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
python examples/beamforming/evaluate_streaming_diff_mvdr_covariance_compare.py \
  --config examples/beamforming/streaming_diff_mvdr_covariance_compare_config.json
```

To regenerate the default 3 second evaluation config:

```bash
python examples/beamforming/evaluate_streaming_diff_mvdr_covariance_compare.py \
  --write-default-config examples/beamforming/streaming_diff_mvdr_covariance_compare_config.json
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
