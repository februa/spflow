"""spflow.beamforming.mvdr_weight_designer を実装するモジュール。"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..callback import DoubleBufferCallback


def _as_steering_matrix(steering: np.ndarray) -> np.ndarray:
    steering_matrix = np.asarray(steering, dtype=np.complex64)
    if steering_matrix.ndim == 1:
        steering_matrix = steering_matrix[:, np.newaxis]
    if steering_matrix.ndim != 2:
        raise ValueError("steering must have shape (n_ch, n_beam).")
    return steering_matrix


def _validate_bandwise_inputs(covariance: np.ndarray, steering_array: np.ndarray) -> tuple[int, int, int]:
    if covariance.ndim != 3:
        raise ValueError("Rxx must have shape (n_band, n_ch, n_ch).")
    if steering_array.ndim != 3:
        raise ValueError("steering must have shape (n_ch, n_beam, n_band) for bandwise design.")
    if covariance.shape[1] != covariance.shape[2]:
        raise ValueError("Rxx must contain square covariance matrices.")
    if covariance.shape[0] != steering_array.shape[2]:
        raise ValueError("Rxx and steering must agree on n_band.")
    if covariance.shape[1] != steering_array.shape[0]:
        raise ValueError("Rxx and steering must agree on n_ch.")
    return covariance.shape[0], covariance.shape[1], steering_array.shape[1]


def design_mvdr_weights(Rxx: np.ndarray, steering: np.ndarray, diag_load: float = 1e-3) -> np.ndarray:
    """単一帯域の MVDR 重みを設計する。

    Args:
        Rxx: 空間共分散。shape は `[n_ch, n_ch]`。
        steering: 目標方向ステアリング。shape は `[n_ch]` または `[n_ch, n_beam]`。
        diag_load: 対角ローディング係数。無次元。

    Returns:
        MVDR 重み。shape は `[n_ch, n_beam]`。

    Notes:
        MVDR は `w = R^{-1} a / (a^H R^{-1} a)` により、
        目標方向に対する無歪条件を満たしつつ出力電力を最小化する。
    """
    covariance = np.asarray(Rxx, dtype=np.complex64)
    steering_matrix = _as_steering_matrix(steering)

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Rxx must have shape (n_ch, n_ch).")
    if covariance.shape[0] != steering_matrix.shape[0]:
        raise ValueError("Rxx and steering must agree on n_ch.")
    if diag_load < 0.0:
        raise ValueError("diag_load must be non-negative.")

    loaded = covariance.copy()
    if diag_load > 0.0:
        # trace(R)/n_ch を各チャネル平均電力の代表値とみなし、
        # その diag_load 倍を対角へ加える。1e-3 は完全ゼロでは不安定になりやすい
        # 少数スナップショット条件で、重み発散を避けるための弱い正則化である。
        base = np.real(np.trace(loaded)) / loaded.shape[0]
        load = diag_load * (base if base > 0.0 else 1.0)
        loaded = loaded + load * np.eye(loaded.shape[0], dtype=np.complex64)

    # solve(R, A) は各ビーム steering a に対する R^{-1} a をまとめて解く。
    response = np.linalg.solve(loaded, steering_matrix)
    # denom[beam] = a^H R^{-1} a。これで正規化すると w^H a = 1 になる。
    denom = np.sum(steering_matrix.conj() * response, axis=0)
    return response / denom[np.newaxis, :]


def design_mvdr_weights_bands(Rxx: np.ndarray, steering: np.ndarray, diag_load: float = 1e-3) -> np.ndarray:
    """帯域ごとの MVDR 重みを一括設計する。"""
    covariance = np.asarray(Rxx, dtype=np.complex64)
    steering_array = np.asarray(steering, dtype=np.complex64)
    if diag_load < 0.0:
        raise ValueError("diag_load must be non-negative.")

    n_band, n_ch, _ = _validate_bandwise_inputs(covariance, steering_array)
    loaded = covariance.copy()
    if diag_load > 0.0:
        # 各帯域ごとに平均電力スケールを求め、その帯域の条件数悪化だけを補償する。
        base = np.real(np.trace(loaded, axis1=1, axis2=2)) / n_ch
        load = diag_load * np.where(base > 0.0, base, 1.0)
        loaded = loaded + load[:, np.newaxis, np.newaxis] * np.eye(n_ch, dtype=np.complex64)[np.newaxis, :, :]

    # steering_batch shape: [n_band, n_ch, n_beam]
    # moveaxis により solve が要求する先頭バッチ軸へ帯域軸を移す。
    steering_batch = np.moveaxis(steering_array, -1, 0)
    response = np.linalg.solve(loaded, steering_batch)
    denom = np.sum(steering_batch.conj() * response, axis=1)
    weights = response / denom[:, np.newaxis, :]
    return np.moveaxis(weights, 0, -1)


def design_mvdr_weights_with_channel_window(
    Rxx: np.ndarray,
    steering: np.ndarray,
    channel_window: np.ndarray,
    diag_load: float = 1e-3,
) -> np.ndarray:
    """チャネル選択テーブルを考慮した MVDR 重みを設計する。

    実運用では shading を連続重みではなくチャネル有効/無効の選択器として解釈し、
    `channel_window != 0` のチャネルだけで縮退共分散を作って MVDR を解く。
    """
    covariance = np.asarray(Rxx, dtype=np.complex64)
    steering_array = np.asarray(steering, dtype=np.complex64)
    window = np.asarray(channel_window, dtype=np.float32)

    if covariance.ndim == 2:
        steering_matrix = _as_steering_matrix(steering_array)
        if window.ndim == 2:
            if window.shape[1] != 1:
                raise ValueError('channel_window must have one band for 2D covariance input.')
            window = window[:, 0]
        if window.ndim != 1 or window.shape[0] != covariance.shape[0]:
            raise ValueError('channel_window must have shape (n_ch,) for 2D covariance input.')
        used = window != 0.0
        if not np.any(used):
            raise ValueError('channel_window must select at least one channel.')
        # 非選択チャネルまで含めて MVDR を解くと、ゼロ開口なのに数値的に寄与してしまう。
        # そのため選択チャネルだけで縮退問題を解き、最後に元 shape へ戻す。
        reduced = design_mvdr_weights(covariance[np.ix_(used, used)], steering_matrix[used], diag_load=diag_load)
        full = np.zeros_like(steering_matrix)
        full[used] = reduced
        return full

    if covariance.ndim != 3:
        raise ValueError('Rxx must have shape (n_ch, n_ch) or (n_band, n_ch, n_ch).')
    if steering_array.ndim != 3:
        raise ValueError('steering must have shape (n_ch, n_beam, n_band) for bandwise design.')
    if window.ndim == 1:
        if window.shape[0] != covariance.shape[1]:
            raise ValueError('channel_window must agree on n_ch.')
        window = np.repeat(window[:, np.newaxis], covariance.shape[0], axis=1)
    if window.ndim != 2:
        raise ValueError('channel_window must have shape (n_ch,) or (n_ch, n_band).')
    if covariance.shape[0] != steering_array.shape[2]:
        raise ValueError('Rxx and steering must agree on n_band.')
    if covariance.shape[1] != covariance.shape[2]:
        raise ValueError('Rxx must contain square covariance matrices.')
    if covariance.shape[1] != steering_array.shape[0]:
        raise ValueError('Rxx and steering must agree on n_ch.')
    if window.shape != (steering_array.shape[0], steering_array.shape[2]):
        raise ValueError('channel_window must have shape (n_ch, n_band).')

    n_band, n_ch = covariance.shape[0], covariance.shape[1]
    n_beam = steering_array.shape[1]
    weights = np.zeros((n_ch, n_beam, n_band), dtype=np.complex64)
    for band_idx in range(n_band):
        used = window[:, band_idx] != 0.0
        if not np.any(used):
            raise ValueError('Each band must select at least one channel.')
        # 帯域ごとに有効開口が異なるため、MVDR は各 band の部分行列で独立に解く。
        reduced = design_mvdr_weights(
            covariance[band_idx][np.ix_(used, used)],
            steering_array[used, :, band_idx],
            diag_load=diag_load,
        )
        weights[used, :, band_idx] = reduced
    return weights


class MVDRWeightDesigner:
    """単一帯域または多帯域の MVDR 重みを設計する薄いラッパー。

    このクラスは共分散 shape に応じて単一帯域版と多帯域版を切り替える。
    共分散推定やスケジューリングは責務に含めない。
    """

    def __init__(self, diag_load: float = 1e-3) -> None:
        if diag_load < 0.0:
            raise ValueError("diag_load must be non-negative.")
        self.diag_load = diag_load

    def process(self, Rxx: np.ndarray, steering: np.ndarray) -> np.ndarray:
        """共分散とステアリングから MVDR 重みを計算する。"""
        covariance = np.asarray(Rxx, dtype=np.complex64)
        steering_array = np.asarray(steering, dtype=np.complex64)

        if covariance.ndim == 2:
            return design_mvdr_weights(covariance, steering_array, diag_load=self.diag_load)

        return design_mvdr_weights_bands(covariance, steering_array, diag_load=self.diag_load)


class MVDRWeightCallback(DoubleBufferCallback):
    """StepScheduler 用の帯域別 MVDR 重み更新コールバック。

    1 ステップで 1 帯域ずつ重みを更新し、全帯域完了後に publish する。
    重い行列演算を時間分散するための補助クラスである。
    """

    def __init__(self, diag_load: float = 1e-3) -> None:
        super().__init__()
        self.designer = MVDRWeightDesigner(diag_load=diag_load)

    def signature(self, inputs: Any) -> Any:
        """入力更新サイクルを識別するシグネチャを返す。"""
        if "signature" in inputs:
            return inputs["signature"]
        return (id(inputs["Rxx"]), id(inputs["steering"]))

    def make_initial_output(self, inputs: Any) -> np.ndarray:
        """出力バッファの初期値を作る。

        `steering` と同じ `[n_ch, n_beam, n_band]` shape を確保し、
        未計算帯域はゼロ重みで初期化する。
        """
        steering = np.asarray(inputs["steering"], dtype=np.complex64)
        if steering.ndim != 3:
            raise ValueError("steering must have shape (n_ch, n_beam, n_band).")
        return np.zeros_like(steering)

    def make_work_buffer(self, inputs: Any) -> np.ndarray:
        """現在出力と同じ shape の作業バッファを確保する。"""
        return np.zeros_like(self.prev)

    def make_items(self, inputs: Any):
        """更新対象帯域インデックス列を返す。"""
        covariance = np.asarray(inputs["Rxx"], dtype=np.complex64)
        if covariance.ndim != 3:
            raise ValueError("Rxx must have shape (n_band, n_ch, n_ch).")
        return range(covariance.shape[0])

    def update_item(self, item: Any, inputs: Any) -> None:
        """指定帯域の MVDR 重みだけを更新する。

        Args:
            item: 帯域インデックス。
            inputs: `Rxx` と `steering` を含む辞書。
        """
        band_idx = int(item)
        covariance = np.asarray(inputs["Rxx"], dtype=np.complex64)
        steering = np.asarray(inputs["steering"], dtype=np.complex64)
        self.work[:, :, band_idx] = self.designer.process(
            covariance[band_idx],
            steering[:, :, band_idx],
        )
