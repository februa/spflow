"""FrameBuffer と Flow を使った最小パイプライン例。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from spflow import Flow, FrameBuffer, Option


def make_env(opt: Option) -> SimpleNamespace:
    """basic pipeline 用の共有状態とフレームバッファを初期化する。"""
    env = SimpleNamespace()
    env.opt = opt
    env.input_buffer = FrameBuffer(
        frame_size=opt.stft.nfft,
        hop_size=opt.stft.hop,
        axis=-1,
    )
    return env


def calc_fft(frame: np.ndarray, env: SimpleNamespace) -> np.ndarray:
    """入力フレームの FFT を計算する。"""
    return np.fft.fft(frame, n=env.opt.stft.nfft, axis=-1)


def calc_power(x: np.ndarray) -> np.ndarray:
    """複素スペクトルからパワーを計算する。"""
    return np.abs(x) ** 2


def process_frame(x: np.ndarray, env: SimpleNamespace) -> list[np.ndarray]:
    """入力チャンクをフレーム化し、FFT とパワー計算を順に適用する。"""
    return (
        Flow.from_value(x)
        .map(env.input_buffer.process)
        .map(calc_fft, env)
        .map(calc_power)
        .to_list()
    )


def main() -> None:
    """最小パイプラインを実行し、各チャンクのパワースペクトルを表示する。"""
    opt = Option(
        {
            "stft": {
                "nfft": 4,
                "hop": 2,
            }
        }
    )
    env = make_env(opt)

    chunks = [
        np.array([0.0, 1.0]),
        np.array([2.0, 3.0]),
        np.array([4.0, 5.0]),
        np.array([6.0, 7.0]),
    ]

    for idx, chunk in enumerate(chunks, start=1):
        outputs = process_frame(chunk, env)
        print(f"chunk={idx} outputs={len(outputs)}")
        for frame_idx, spectrum in enumerate(outputs, start=1):
            print(f"  frame={frame_idx} power={np.round(spectrum, 3)}")


if __name__ == "__main__":
    main()
