"""spflow.filterbank.polyphase を実装するモジュール。"""

from __future__ import annotations

import numpy as np


def frame_signal(x: np.ndarray, frame_size: int, hop_size: int) -> tuple[np.ndarray, int]:
    """Slice the last axis into overlapped frames, padding the tail as needed."""

    if frame_size <= 0:
        raise ValueError("frame_size must be positive.")
    if hop_size <= 0:
        raise ValueError("hop_size must be positive.")

    n_samples = x.shape[-1]
    if n_samples == 0:
        return np.zeros(x.shape[:-1] + (frame_size, 0), dtype=x.dtype), 0

    n_frames = max(1, int(np.ceil((n_samples - frame_size) / hop_size)) + 1)
    padded_length = frame_size + max(0, n_frames - 1) * hop_size
    pad_width = padded_length - n_samples

    if pad_width > 0:
        padded = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad_width)])
    else:
        padded = x

    frames = []
    for frame_idx in range(n_frames):
        start = frame_idx * hop_size
        stop = start + frame_size
        frames.append(padded[..., start:stop])

    return np.stack(frames, axis=-1), pad_width


def overlap_add(
    frames: np.ndarray,
    window: np.ndarray,
    hop_size: int,
    length: int | None = None,
) -> np.ndarray:
    """Overlap-add frames along the last axis and normalize by window power."""

    if frames.ndim < 2:
        raise ValueError("frames must have frame and frame-index axes.")
    if hop_size <= 0:
        raise ValueError("hop_size must be positive.")

    frame_size = frames.shape[-2]
    n_frames = frames.shape[-1]
    out_length = frame_size + max(0, n_frames - 1) * hop_size

    output = np.zeros(frames.shape[:-2] + (out_length,), dtype=frames.dtype)
    weight = np.zeros((out_length,), dtype=np.float32)
    window_sq = np.square(window.astype(np.float32, copy=False))

    for frame_idx in range(n_frames):
        start = frame_idx * hop_size
        stop = start + frame_size
        output[..., start:stop] += frames[..., :, frame_idx] * window
        weight[start:stop] += window_sq

    nonzero = weight > np.finfo(np.float32).eps
    output[..., nonzero] /= weight[nonzero]

    if length is not None:
        return output[..., :length]
    return output
