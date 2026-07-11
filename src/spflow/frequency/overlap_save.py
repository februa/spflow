"""spflow.frequency.overlap_save を実装するモジュール。"""

from __future__ import annotations

import numpy as np


class OverlapSaveBuffer:
    """逐次ブロック列から overlap-save 用フレームを構成する。

    入力は時間軸上で連続する有効区間ブロックであり、各ブロック長は `valid_size`
    サンプル以下を想定する。出力は `frame_size` サンプルのフレーム列で、
    各フレームは直前の履歴 `frame_size - valid_size` サンプルと新規入力ブロックを
    連結したものになる。

    FFT 自体や FIR 畳み込みは責務に含めず、あくまで overlap-save に必要な
    履歴管理と境界条件のゼロ初期化だけを担当する。
    """

    def __init__(self, frame_size: int, valid_size: int, axis: int = -1) -> None:
        if frame_size <= 0:
            raise ValueError("frame_size must be positive.")
        if valid_size <= 0:
            raise ValueError("valid_size must be positive.")
        if valid_size > frame_size:
            raise ValueError("valid_size must not exceed frame_size.")

        self.frame_size = frame_size
        self.valid_size = valid_size
        self.overlap_size = frame_size - valid_size
        self.axis = axis
        self._pending: np.ndarray | None = None
        self._history: np.ndarray | None = None

    def push(self, x: np.ndarray) -> list[np.ndarray]:
        """入力ブロックを蓄積し、構成可能な overlap-save フレームを返す。

        Args:
            x: 入力サンプル列。shape は `[..., n_sample]`。
                `axis` が時間軸を表し、それ以外の軸はチャネルやビームなどの
                付随次元として保持する。

        Returns:
            overlap-save フレームのリスト。各要素の shape は `[..., frame_size]`。
            返却時の時間軸位置は入力と同じ `axis` に戻す。

        Raises:
            ValueError: `axis` が入力次元範囲外の場合。
            ValueError: 既存バッファと比較して、時間軸以外の shape が一致しない場合。

        Notes:
            初回フレームでは過去履歴が存在しないため、履歴領域はゼロで初期化する。
            これは causal FIR の立ち上がり区間を安全側に扱うためである。
        """
        arr = np.asarray(x)
        work_axis = self._normalize_axis(arr.ndim)
        # moveaxis 後の shape は [..., n_sample] とし、末尾軸だけを時間軸として扱う。
        # これによりチャネル数やビーム数に依存せず同じ履歴更新ロジックを再利用できる。
        moved = np.moveaxis(arr, work_axis, -1)

        if self._pending is None:
            self._pending = moved.copy()
        else:
            previous_pending = self._pending
            if not isinstance(previous_pending, np.ndarray):
                # _pending は ndarray または None だけを保持する。ここは else 分岐の型契約を実行時にも固定する。
                raise RuntimeError("overlap-save pending state must be an ndarray.")
            if previous_pending.ndim != moved.ndim or previous_pending.shape[:-1] != moved.shape[:-1]:
                raise ValueError("Input shape mismatch except along processing axis.")
            self._pending = np.concatenate([previous_pending, moved], axis=-1)

        if self._history is None:
            history_shape = moved.shape[:-1] + (self.overlap_size,)
            # 初回フレームには過去サンプルが存在しないため、履歴領域をゼロで埋める。
            # overlap-save ではこのゼロ区間が先頭過渡応答に対応する。
            self._history = np.zeros(history_shape, dtype=moved.dtype)

        frames: list[np.ndarray] = []
        pending = self._pending
        history = self._history
        if not isinstance(pending, np.ndarray) or not isinstance(history, np.ndarray):
            # 上の初期化後は両状態が必ず ndarray でなければ overlap-save frame を構成できない。
            raise RuntimeError("overlap-save state initialization failed.")
        while pending.shape[-1] >= self.valid_size:
            block = pending[..., : self.valid_size]
            # frame shape: [..., frame_size]
            # 末尾 `valid_size` サンプルが今回の新規有効区間、先頭 `overlap_size`
            # サンプルが直前フレーム末尾の履歴に対応する。
            frame = np.concatenate([history, block], axis=-1)
            frames.append(np.moveaxis(frame.copy(), -1, work_axis))
            if self.overlap_size > 0:
                # 次フレームに必要なのは現在フレーム末尾 `overlap_size` サンプルだけである。
                # overlap-save の重複区間をそのまま履歴として保持する。
                history = frame[..., -self.overlap_size :].copy()
            pending = pending[..., self.valid_size :]
        self._history = history
        self._pending = pending
        return frames

    def process(self, x: np.ndarray) -> list[np.ndarray]:
        """`push` の別名として入力ブロックを処理する。

        Args:
            x: 入力サンプル列。shape は `[..., n_sample]`。

        Returns:
            構成できた overlap-save フレームのリスト。
        """
        return self.push(x)

    def flush(self, pad: bool = True, fill_value: float = 0.0) -> list[np.ndarray]:
        """未出力の端数ブロックを必要に応じてゼロ詰めして排出する。

        Args:
            pad: `True` の場合、末尾端数を `valid_size` まで埋めて最終フレームを返す。
                `False` の場合、端数は安全側として破棄する。
            fill_value: `pad=True` 時に末尾へ補う値。通常は 0.0 を用いる。

        Returns:
            最終 overlap-save フレームのリスト。

        Notes:
            最終ブロックが `valid_size` 未満のままでは FFT フレームを組めないため、
            `pad=True` ではゼロ詰めして長さを揃える。これは末尾不足時に
            仮想的な無音サンプルが続くとみなす境界条件である。
        """
        if self._pending is None or self._pending.shape[-1] == 0:
            self.reset()
            return []
        if not pad:
            self.reset()
            return []

        pad_width = self.valid_size - self._pending.shape[-1]
        padded = np.pad(
            self._pending,
            [(0, 0)] * (self._pending.ndim - 1) + [(0, pad_width)],
            constant_values=fill_value,
        )
        self._pending = padded
        frames = self.push(np.zeros(padded.shape[:-1] + (0,), dtype=padded.dtype))
        self.reset()
        return frames

    def reset(self) -> None:
        """内部バッファを初期状態へ戻す。

        過去履歴と未確定入力を破棄し、新しい信号系列の処理を開始できる状態にする。
        """
        self._pending = None
        self._history = None

    def _normalize_axis(self, ndim: int) -> int:
        axis = self.axis
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError("axis is out of bounds for input.")
        return axis


class ValidRegionExtractor:
    """処理済み overlap-save フレームから有効区間だけを抜き出す。

    overlap-save の出力フレームには、先頭に循環畳み込み由来の無効区間が含まれる。
    このクラスは末尾 `valid_size` サンプルのみを返し、線形畳み込みと一致する
    有効区間を呼び出し側へ渡す。
    """

    def __init__(self, frame_size: int, valid_size: int, axis: int = -1) -> None:
        if frame_size <= 0:
            raise ValueError("frame_size must be positive.")
        if valid_size <= 0:
            raise ValueError("valid_size must be positive.")
        if valid_size > frame_size:
            raise ValueError("valid_size must not exceed frame_size.")
        self.frame_size = frame_size
        self.valid_size = valid_size
        self.axis = axis

    def process(self, frame: np.ndarray) -> np.ndarray:
        """1 フレームから有効区間を抽出する。

        Args:
            frame: 処理済みフレーム。shape は `[..., frame_size]`。
                `axis` が時間軸を表す。

        Returns:
            有効区間。shape は `[..., valid_size]`。
        """
        arr = np.asarray(frame)
        work_axis = self._normalize_axis(arr.ndim)
        # moveaxis 後の shape は [..., frame_size]。
        # overlap-save では線形畳み込みに一致するのは末尾 `valid_size` サンプルである。
        moved = np.moveaxis(arr, work_axis, -1)
        valid = moved[..., -self.valid_size :]
        return np.moveaxis(valid, -1, work_axis)

    def _normalize_axis(self, ndim: int) -> int:
        axis = self.axis
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError("axis is out of bounds for input.")
        return axis


def make_filter_fft(filters: np.ndarray, frame_size: int, axis: int = -1) -> np.ndarray:
    """時間領域 FIR を overlap-save 用の周波数応答へ変換する。

    Args:
        filters: 時間領域 FIR 係数。shape は `[..., n_tap]`。
            `axis` がタップ軸を表す。
        frame_size: FFT 長。単位はサンプル数。
        axis: FIR タップ軸。

    Returns:
        フィルタ FFT。shape は `[..., frame_size]`。

    Raises:
        ValueError: `frame_size` が正でない場合。
        ValueError: `axis` が入力次元範囲外の場合。
        ValueError: フィルタ長が `frame_size` を超える場合。
    """
    if frame_size <= 0:
        raise ValueError("frame_size must be positive.")

    arr = np.asarray(filters, dtype=np.complex64)
    work_axis = axis
    if work_axis < 0:
        work_axis += arr.ndim
    if work_axis < 0 or work_axis >= arr.ndim:
        raise ValueError("axis is out of bounds for filters.")

    # moveaxis 後の shape は [..., n_tap]。
    # FFT を末尾軸へ固定することで、先頭軸を ch / beam / band として一括処理できる。
    moved = np.moveaxis(arr, work_axis, -1)
    if moved.shape[-1] > frame_size:
        raise ValueError("filter length must not exceed frame_size.")

    # overlap-save では FIR を frame_size 長へゼロ拡張して FFT し、
    # 周波数領域で点ごとの積へ持ち込む。
    padded = np.zeros(moved.shape[:-1] + (frame_size,), dtype=np.complex64)
    padded[..., : moved.shape[-1]] = moved
    # axis=-1 はゼロ拡張後の時間軸であり、各 FIR の離散時間周波数応答 H[k] を与える。
    spectrum = np.fft.fft(padded, axis=-1)
    return np.moveaxis(spectrum, -1, work_axis)
