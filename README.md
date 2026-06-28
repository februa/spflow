# spflow

`spflow` is a lightweight utility package for writing sequential signal processing code in plain Python.

It provides five focused building blocks:

- `Option`: dot-access wrapper around nested dictionaries
- `Flow`: lightweight container for zero, one, or many values
- `FrameBuffer`: frame slicing utility with overlap support
- `StepScheduler`: scheduler for iterative item processing
- `DoubleBufferCallback`: callback base that publishes only completed values

## Installation

```bash
pip install .
```

## Example

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
