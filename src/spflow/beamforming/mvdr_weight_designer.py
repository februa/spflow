"""spflow.beamforming.mvdr_weight_designer を実装するモジュール。"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..callback import DoubleBufferCallback
from .cbf import design_cbf_coefficients


def _as_steering_matrix(steering: np.ndarray) -> np.ndarray:
    steering_matrix = np.asarray(steering, dtype=np.complex64)
    if steering_matrix.ndim == 1:
        steering_matrix = steering_matrix[:, np.newaxis]
    if steering_matrix.ndim != 2:
        raise ValueError("steering must have shape (n_ch, n_beam).")
    return steering_matrix


def _validate_bandwise_inputs(
    covariance: np.ndarray, steering_array: np.ndarray
) -> tuple[int, int, int]:
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


def design_mvdr_coefficients(
    Rxx: np.ndarray, steering: np.ndarray, diag_load: float = 1e-3
) -> np.ndarray:
    """単一帯域のMVDR実適用係数を設計する。

    Args:
        Rxx: 空間共分散。shape は `[n_ch, n_ch]`。
        steering: 目標方向ステアリング。shape は `[n_ch]` または `[n_ch, n_beam]`。
        diag_load: 対角ローディング係数。無次元。

    Returns:
        MVDR係数`h`。shapeは`[n_ch, n_beam]`。
        適用時は共役を追加せず`y=h^T x`として使う。

    Notes:
        MVDRの理論重み`w=R^{-1}a/(a^H R^{-1}a)`から、実適用係数
        `h=conj(w)`へ設計境界で変換する。これにより`h^T a=w^H a=1`となる。
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
    theoretical_weights = response / denom[np.newaxis, :]
    # 適用側はh^T xだけを計算するため、理論重みwの共役を実適用係数として返す。
    return np.conj(theoretical_weights)


def design_mvdr_coefficients_bands(
    Rxx: np.ndarray, steering: np.ndarray, diag_load: float = 1e-3
) -> np.ndarray:
    """帯域ごとのMVDR実適用係数を一括設計する。

    Args:
        Rxx: 帯域別空間共分散。shapeは`[n_band, n_ch, n_ch]`。
        steering: 帯域別ステアリング。shapeは`[n_ch, n_beam, n_band]`。
        diag_load: 各帯域の平均対角powerに対する対角ロード比。無次元。

    Returns:
        `y=h^T x`へ直接使う係数。shapeは`[n_ch, n_beam, n_band]`。

    Raises:
        ValueError: shapeが整合しない、またはdiag_loadが負の場合。
        numpy.linalg.LinAlgError: 対角ロード後の共分散を解けない場合。
    """
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
        loaded = (
            loaded
            + load[:, np.newaxis, np.newaxis] * np.eye(n_ch, dtype=np.complex64)[np.newaxis, :, :]
        )

    # steering_batch shape: [n_band, n_ch, n_beam]
    # moveaxis により solve が要求する先頭バッチ軸へ帯域軸を移す。
    steering_batch = np.moveaxis(steering_array, -1, 0)
    response = np.linalg.solve(loaded, steering_batch)
    denom = np.sum(steering_batch.conj() * response, axis=1)
    theoretical_weights = response / denom[:, np.newaxis, :]
    return np.conj(np.moveaxis(theoretical_weights, 0, -1))


def design_mvdr_coefficients_with_channel_window(
    Rxx: np.ndarray,
    steering: np.ndarray,
    channel_window: np.ndarray,
    diag_load: float = 1e-3,
) -> np.ndarray:
    """チャネル選択テーブルを考慮したMVDR実適用係数を設計する。

    Args:
        Rxx: 単一帯域`[n_ch, n_ch]`または帯域別
            `[n_band, n_ch, n_ch]`の空間共分散。
        steering: 単一帯域`[n_ch, n_beam]`または帯域別
            `[n_ch, n_beam, n_band]`のステアリング。
        channel_window: shape`[n_ch]`または`[n_ch, n_band]`の無次元窓。
            非零channelだけをMVDR設計へ使用する。
        diag_load: 平均対角powerに対する対角ロード比。無次元。

    Returns:
        steeringと同じshapeの実適用係数。無効channelは0とする。

    Raises:
        ValueError: shapeが整合しない、全channelが無効、またはdiag_loadが負の場合。
        numpy.linalg.LinAlgError: 有効channelの共分散を解けない場合。

    境界条件:
        shadingを連続係数ではなくchannel有効・無効の選択器として解釈し、
        `channel_window != 0`の部分共分散だけでMVDRを解く。
    """
    covariance = np.asarray(Rxx, dtype=np.complex64)
    steering_array = np.asarray(steering, dtype=np.complex64)
    window = np.asarray(channel_window, dtype=np.float32)

    if covariance.ndim == 2:
        steering_matrix = _as_steering_matrix(steering_array)
        if window.ndim == 2:
            if window.shape[1] != 1:
                raise ValueError("channel_window must have one band for 2D covariance input.")
            window = window[:, 0]
        if window.ndim != 1 or window.shape[0] != covariance.shape[0]:
            raise ValueError("channel_window must have shape (n_ch,) for 2D covariance input.")
        used = window != 0.0
        if not np.any(used):
            raise ValueError("channel_window must select at least one channel.")
        # 非選択チャネルまで含めて MVDR を解くと、ゼロ開口なのに数値的に寄与してしまう。
        # そのため選択チャネルだけで縮退問題を解き、最後に元 shape へ戻す。
        reduced = design_mvdr_coefficients(
            covariance[np.ix_(used, used)],
            steering_matrix[used],
            diag_load=diag_load,
        )
        full = np.zeros_like(steering_matrix)
        full[used] = reduced
        return full

    if covariance.ndim != 3:
        raise ValueError("Rxx must have shape (n_ch, n_ch) or (n_band, n_ch, n_ch).")
    if steering_array.ndim != 3:
        raise ValueError("steering must have shape (n_ch, n_beam, n_band) for bandwise design.")
    if window.ndim == 1:
        if window.shape[0] != covariance.shape[1]:
            raise ValueError("channel_window must agree on n_ch.")
        window = np.repeat(window[:, np.newaxis], covariance.shape[0], axis=1)
    if window.ndim != 2:
        raise ValueError("channel_window must have shape (n_ch,) or (n_ch, n_band).")
    if covariance.shape[0] != steering_array.shape[2]:
        raise ValueError("Rxx and steering must agree on n_band.")
    if covariance.shape[1] != covariance.shape[2]:
        raise ValueError("Rxx must contain square covariance matrices.")
    if covariance.shape[1] != steering_array.shape[0]:
        raise ValueError("Rxx and steering must agree on n_ch.")
    if window.shape != (steering_array.shape[0], steering_array.shape[2]):
        raise ValueError("channel_window must have shape (n_ch, n_band).")

    n_band, n_ch = covariance.shape[0], covariance.shape[1]
    n_beam = steering_array.shape[1]
    coefficients = np.zeros((n_ch, n_beam, n_band), dtype=np.complex64)
    for band_idx in range(n_band):
        used = window[:, band_idx] != 0.0
        if not np.any(used):
            raise ValueError("Each band must select at least one channel.")
        # 帯域ごとに有効開口が異なるため、MVDR は各 band の部分行列で独立に解く。
        reduced = design_mvdr_coefficients(
            covariance[band_idx][np.ix_(used, used)],
            steering_array[used, :, band_idx],
            diag_load=diag_load,
        )
        coefficients[used, :, band_idx] = reduced
    return coefficients


def design_mvdr_weights(
    Rxx: np.ndarray, steering: np.ndarray, diag_load: float = 1e-3
) -> np.ndarray:
    """単一帯域MVDR実適用係数を返す互換名。

    Args:
        Rxx: 空間共分散。shapeは`[n_ch, n_ch]`。
        steering: ステアリング。shapeは`[n_ch]`または`[n_ch, n_beam]`。
        diag_load: 平均対角powerに対する対角ロード比。無次元。

    Returns:
        `y=h^T x`へ直接使う係数。shapeは`[n_ch, n_beam]`。

    Raises:
        ValueError: shapeが不正、またはdiag_loadが負の場合。
        numpy.linalg.LinAlgError: 共分散を解けない場合。
    """
    return design_mvdr_coefficients(Rxx, steering, diag_load=diag_load)


def design_mvdr_weights_bands(
    Rxx: np.ndarray, steering: np.ndarray, diag_load: float = 1e-3
) -> np.ndarray:
    """帯域別MVDR実適用係数を返す互換名。

    Args:
        Rxx: shape`[n_band, n_ch, n_ch]`の帯域別共分散。
        steering: shape`[n_ch, n_beam, n_band]`のステアリング。
        diag_load: 平均対角powerに対する対角ロード比。無次元。

    Returns:
        shape`[n_ch, n_beam, n_band]`の実適用係数。

    Raises:
        ValueError: shapeが整合しない、またはdiag_loadが負の場合。
        numpy.linalg.LinAlgError: 共分散を解けない場合。
    """
    return design_mvdr_coefficients_bands(Rxx, steering, diag_load=diag_load)


def design_mvdr_weights_with_channel_window(
    Rxx: np.ndarray,
    steering: np.ndarray,
    channel_window: np.ndarray,
    diag_load: float = 1e-3,
) -> np.ndarray:
    """channel window付きMVDR実適用係数を返す互換名。

    Args:
        Rxx: 単一帯域または帯域別共分散。
        steering: 対応するステアリングベクトル。
        channel_window: 有効channelを非零で示すshape`[n_ch]`または
            `[n_ch, n_band]`の無次元窓。
        diag_load: 平均対角powerに対する対角ロード比。無次元。

    Returns:
        steeringと同じshapeの実適用係数。

    Raises:
        ValueError: shapeが整合しない、全channelが無効、またはdiag_loadが負の場合。
        numpy.linalg.LinAlgError: 有効channelの共分散を解けない場合。
    """
    return design_mvdr_coefficients_with_channel_window(
        Rxx,
        steering,
        channel_window,
        diag_load=diag_load,
    )


class MVDRWeightDesigner:
    """単一帯域または多帯域のMVDR実適用係数を設計する互換ラッパー。

    このクラスは共分散 shape に応じて単一帯域版と多帯域版を切り替える。
    共分散推定やスケジューリングは責務に含めない。
    """

    def __init__(self, diag_load: float = 1e-3) -> None:
        if diag_load < 0.0:
            raise ValueError("diag_load must be non-negative.")
        self.diag_load = diag_load

    def process(self, Rxx: np.ndarray, steering: np.ndarray) -> np.ndarray:
        """共分散とステアリングからMVDR実適用係数を計算する。"""
        covariance = np.asarray(Rxx, dtype=np.complex64)
        steering_array = np.asarray(steering, dtype=np.complex64)

        if covariance.ndim == 2:
            return design_mvdr_coefficients(covariance, steering_array, diag_load=self.diag_load)

        return design_mvdr_coefficients_bands(covariance, steering_array, diag_load=self.diag_load)


@dataclass(frozen=True)
class MVDRWeightSnapshot:
    """時間分割して設計する帯域別MVDR係数の入力snapshot。

    共分散、ステアリング、generationを一つの固定型へまとめ、異なる時刻の部分計算が
    混ざらない境界を作る。生成時に配列を`complex64`でcopyして読み取り専用にするため、
    呼び出し元が元配列をin-place更新しても進行中計算には影響しない。

    Attributes:
        covariance: 帯域別空間共分散。shapeは`[n_band, n_ch, n_ch]`、単位は入力振幅の二乗。
        steering: 目標方向ステアリング。shapeは`[n_ch, n_beam, n_band]`、無次元複素応答。
        generation: 共分散snapshotを一意に識別するhash可能な値。

    信号への係数適用、処理周期、共分散推定は責務に含めない。
    """

    covariance: NDArray[np.complex64]
    steering: NDArray[np.complex64]
    generation: Hashable

    def __post_init__(self) -> None:
        """shapeとgenerationを検証し、所有権を分離した読み取り専用配列を保持する。

        Raises:
            ValueError: covarianceまたはsteeringのshapeが整合しない場合。
            TypeError: generationがhash可能でない場合。
        """
        covariance = np.array(self.covariance, dtype=np.complex64, copy=True)
        steering = np.array(self.steering, dtype=np.complex64, copy=True)
        _validate_bandwise_inputs(covariance, steering)
        try:
            hash(self.generation)
        except TypeError as exc:
            raise TypeError("generation must be hashable.") from exc

        # snapshot作成後のin-place変更で帯域ごとの計算世代が混ざらないよう、
        # snapshotが所有する配列を書込禁止にする。
        covariance.flags.writeable = False
        steering.flags.writeable = False
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "steering", steering)


class MVDRWeightCallback(
    DoubleBufferCallback[
        MVDRWeightSnapshot,
        int,
        NDArray[np.complex64],
    ]
):
    """StepScheduler用の帯域別MVDR実適用係数更新コールバック。

    1ステップで1帯域ずつ係数を更新し、全帯域完了後にpublishする。
    初回未完成周期には固定整相CBF係数を返し、適応係数が未完成でも信号を失わない。
    重い行列演算を時間分散するための補助クラスであり、共分散推定や信号適用は担わない。
    """

    def __init__(
        self,
        diag_load: float = 1e-3,
        initial_coefficients: NDArray[Any] | None = None,
    ) -> None:
        """MVDR設計器と初回fallbackを構成する。

        Args:
            diag_load: 共分散へ加える対角loading比。無次元で、0以上とする。
            initial_coefficients: 任意の固定fallback係数。shapeはsnapshotのsteeringと同じ
                `[n_ch, n_beam, n_band]`。`None`ではsteeringから固定CBF係数を生成する。

        Raises:
            ValueError: diag_loadが負の場合。係数shapeは最初のsnapshot受領時に検証する。
        """
        super().__init__()
        self.designer = MVDRWeightDesigner(diag_load=diag_load)
        self._initial_coefficients = (
            None
            if initial_coefficients is None
            else np.array(initial_coefficients, dtype=np.complex64, copy=True)
        )

    def signature(self, inputs: MVDRWeightSnapshot) -> Hashable:
        """共分散snapshotの明示的なgenerationを返す。

        Args:
            inputs: shape検証済みの帯域別MVDR入力snapshot。

        Returns:
            `inputs.generation`。配列identityからの暗黙推定は行わない。
        """
        return inputs.generation

    def make_initial_output(self, inputs: MVDRWeightSnapshot) -> NDArray[np.complex64]:
        """適応係数完成前に公開する固定ビームフォーマ係数を作る。

        Args:
            inputs: steering shapeが`[n_ch, n_beam, n_band]`の入力snapshot。

        Returns:
            steeringと同じshapeの実適用係数。既定では`h=conj(a/(a^H a))`により
            `h^T a=1`を満たす固定CBF係数を返す。

        Raises:
            ValueError: 指定fallbackのshapeがsteeringと一致しない場合。
        """
        if self._initial_coefficients is None:
            # MVDRが未完成でも目標方向を無歪で通すため、同じsteeringの固定CBFを安全側に使う。
            return np.asarray(design_cbf_coefficients(inputs.steering), dtype=np.complex64)
        if self._initial_coefficients.shape != inputs.steering.shape:
            raise ValueError("initial_coefficients and steering must have the same shape.")
        return self._initial_coefficients.copy()

    def make_work_buffer(self, inputs: MVDRWeightSnapshot) -> NDArray[np.complex64]:
        """全帯域のMVDR係数を格納する作業bufferを作る。

        Args:
            inputs: steering shapeが`[n_ch, n_beam, n_band]`の入力snapshot。

        Returns:
            steeringと同じshapeのゼロ初期化した`complex64`配列。
        """
        return np.zeros_like(inputs.steering)

    def make_items(self, inputs: MVDRWeightSnapshot) -> range:
        """更新対象となる全帯域indexを周波数軸順に返す。

        Args:
            inputs: covariance shapeが`[n_band, n_ch, n_ch]`の入力snapshot。

        Returns:
            `0`から`n_band - 1`までの帯域index。
        """
        return range(inputs.covariance.shape[0])

    def update_item(self, item: int, inputs: MVDRWeightSnapshot) -> None:
        """指定帯域のMVDR実適用係数だけを更新する。

        Args:
            item: 周波数帯域index。範囲は`0 <= item < n_band`。
            inputs: covarianceとsteeringを持つ同一generationのsnapshot。

        Raises:
            RuntimeError: schedulerの開始処理を経ず作業bufferが存在しない場合。
            numpy.linalg.LinAlgError: 対角loading後の共分散を解けない場合。
        """
        work = self.work
        if work is None:
            raise RuntimeError("work buffer was not created.")
        # work shape: [n_ch, n_beam, n_band]。itemで指定した周波数帯域だけを更新する。
        work[:, :, item] = self.designer.process(
            inputs.covariance[item],
            inputs.steering[:, :, item],
        )
