"""spflow.beamforming.cbf を実装するモジュール。"""

from __future__ import annotations

import numpy as np

from ..frequency import OverlapSaveBuffer, ValidRegionExtractor, make_filter_fft
from .application import apply_beamformer, apply_beamformer_bands, apply_beamformer_filter_fft


def _as_steering_matrix(steering: np.ndarray) -> np.ndarray:
    steering_matrix = np.asarray(steering, dtype=np.complex64)
    if steering_matrix.ndim == 1:
        steering_matrix = steering_matrix[:, np.newaxis]
    if steering_matrix.ndim != 2:
        raise ValueError("steering must have shape (n_ch, n_beam).")
    return steering_matrix


def apply_channel_window_to_steering(steering: np.ndarray, channel_window: np.ndarray) -> np.ndarray:
    """ステアリングへチャネル別または帯域別の shading を掛ける。

    Args:
        steering: ステアリングベクトル。shape は `[n_ch, n_beam]` または
            `[n_ch, n_beam, n_band]`。
        channel_window: チャネル窓。shape は `[n_ch]` または `[n_ch, n_band]`。

    Returns:
        shading 適用後のステアリング。入力 `steering` と同じ shape。

    Raises:
        ValueError: 入力 shape が想定と異なる場合。
    """
    steering_array = np.asarray(steering, dtype=np.complex64)
    window = np.asarray(channel_window, dtype=np.float32)

    if steering_array.ndim == 2:
        if window.ndim == 2:
            if window.shape[1] != 1:
                raise ValueError('channel_window must have one band for 2D steering input.')
            window = window[:, 0]
        if window.ndim != 1 or window.shape[0] != steering_array.shape[0]:
            raise ValueError('channel_window must have shape (n_ch,) for 2D steering input.')
        # steering shape: [n_ch, n_beam]
        # window[:, None] を掛けることで、各ビームに共通なチャネル shading を与える。
        return steering_array * window[:, np.newaxis]

    if steering_array.ndim != 3:
        raise ValueError('steering must have shape (n_ch, n_beam) or (n_ch, n_beam, n_band).')
    if window.ndim == 1:
        if window.shape[0] != steering_array.shape[0]:
            raise ValueError('channel_window must agree on n_ch.')
        window = np.repeat(window[:, np.newaxis], steering_array.shape[2], axis=1)
    if window.ndim != 2:
        raise ValueError('channel_window must have shape (n_ch,) or (n_ch, n_band).')
    if window.shape[0] != steering_array.shape[0] or window.shape[1] != steering_array.shape[2]:
        raise ValueError('channel_window and steering must agree on n_ch and n_band.')
    # steering shape: [n_ch, n_beam, n_band]
    # window[:, None, :] により、チャネルごと・帯域ごとの shading を全ビームへ一括適用する。
    return steering_array * window[:, np.newaxis, :]


def design_cbf_coefficients(steering: np.ndarray) -> np.ndarray:
    """固定整相CBFの実適用係数をステアリングから設計する。

    Args:
        steering: ステアリングベクトル。shape は `[n_ch, n_beam]` または
            `[n_ch, n_beam, n_band]`。

    Returns:
        CBF係数`h`。入力と同じshapeで、適用時は`y=h^T x`とする。
        理論重み`w=a/(a^H a)`に対して`h=conj(w)`であり、
        無歪条件`h^T a=w^H a=1`を満たす。

    Raises:
        ValueError: ステアリング shape が想定と異なる場合。
        ValueError: 零ベクトルのステアリングが含まれる場合。
    """
    steering_array = np.asarray(steering, dtype=np.complex64)
    if steering_array.ndim == 3:
        coefficients = np.zeros_like(steering_array)
        for band_idx in range(steering_array.shape[-1]):
            coefficients[:, :, band_idx] = design_cbf_coefficients(steering_array[:, :, band_idx])
        return coefficients

    steering_matrix = _as_steering_matrix(steering_array)
    # norm[beam] = a[beam]^H a[beam]。
    # CBF ではステアリングそのものを重み方向とし、エネルギーで正規化して
    # 目標方向利得を 1 に揃える。
    norm = np.sum(np.abs(steering_matrix) ** 2, axis=0, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("steering vectors must be non-zero.")
    # 適用側は共役なしのh^T xだけを計算するため、設計境界でh=conj(w)へ変換する。
    return np.conj(steering_matrix / norm)


def design_cbf_coefficients_with_channel_window(steering: np.ndarray, channel_window: np.ndarray) -> np.ndarray:
    """チャネルshading適用後のCBF実適用係数を設計する。

    Args:
        steering: ステアリングベクトル。shapeは`[n_ch, n_beam]`または
            `[n_ch, n_beam, n_band]`。
        channel_window: チャネル窓。shapeは`[n_ch]`または`[n_ch, n_band]`。

    Returns:
        `y=h^T x`へ直接渡せるCBF係数。shapeはsteeringと同じ。

    Raises:
        ValueError: steeringまたはchannel_windowのshapeが整合しない場合。
    """
    return design_cbf_coefficients(apply_channel_window_to_steering(steering, channel_window))


def design_cbf_weights(steering: np.ndarray) -> np.ndarray:
    """`design_cbf_coefficients()`の互換名として実適用係数を返す。

    Args:
        steering: ステアリングベクトル。shapeは`[n_ch, n_beam]`または
            `[n_ch, n_beam, n_band]`。値は無次元複素応答である。

    Returns:
        `y=h^T x`へ直接使うCBF実適用係数。shapeはsteeringと同じ。

    Raises:
        ValueError: steeringのshapeが不正、または零ベクトルを含む場合。

    境界条件:
        新規コードでは、戻り値が理論重みではなく実適用係数であることを
        名前で示す`design_cbf_coefficients()`を使用する。
    """
    return design_cbf_coefficients(steering)


def design_cbf_weights_with_channel_window(steering: np.ndarray, channel_window: np.ndarray) -> np.ndarray:
    """channel window付きCBF実適用係数を返す互換名。

    Args:
        steering: shape`[n_ch, n_beam]`または`[n_ch, n_beam, n_band]`の
            ステアリングベクトル。
        channel_window: shape`[n_ch]`または`[n_ch, n_band]`の無次元窓。

    Returns:
        `y=h^T x`へ直接使う係数。shapeはsteeringと同じ。

    Raises:
        ValueError: steeringとchannel_windowのshapeが整合しない場合。
    """
    return design_cbf_coefficients_with_channel_window(steering, channel_window)


def design_cbf_overlap_save_filters(steering: np.ndarray, frame_size: int) -> np.ndarray:
    """CBF実適用係数をoverlap-save用フィルタFFTへ変換する。

    Args:
        steering: ステアリングベクトル。shape は `[n_ch, n_beam]` または
            `[n_ch, n_beam, n_band]`。
        frame_size: FFT 長。単位はサンプル数。

    Returns:
        フィルタ FFT。shape は `[n_ch, n_beam, n_band, frame_size]` 相当。

    Notes:
        設計器が返した実適用係数をそのままFIR tapとする。実行時は
        周波数ビンごとの転置内積だけを行い、追加の複素共役を取らない。
    """
    coefficients = design_cbf_coefficients(steering)
    taps = coefficients[..., np.newaxis]
    return make_filter_fft(taps, frame_size=frame_size, axis=-1)


class CBFBeamformer:
    """固定CBF実適用係数でサブバンドスナップショットを投影する。

    このクラスは既知ステアリングから固定係数を構成し、各フレームを
    ビーム出力へ写像する。共分散推定や適応更新は責務に含めない。
    """

    def __init__(self, steering: np.ndarray, channel_window: np.ndarray | None = None) -> None:
        self.coefficients = (
            design_cbf_coefficients(steering)
            if channel_window is None
            else design_cbf_coefficients_with_channel_window(steering, channel_window)
        )

    @property
    def weights(self) -> np.ndarray:
        """互換名として保持中の実適用係数を返す。

        Returns:
            shape`[n_ch, n_beam]`または`[n_ch, n_beam, n_band]`の係数。
            戻り値は`coefficients`と同じ配列であり、`y=h^T x`へ直接使用する。

        境界条件:
            新規コードでは意味を明示する`coefficients`属性を使用する。
        """
        return self.coefficients

    def process(self, X: np.ndarray) -> np.ndarray:
        """サブバンド観測を固定 CBF でビーム出力へ変換する。

        Args:
            X: 観測スナップショット。単一帯域では shape は `[n_ch, n_frame]`。
                帯域込みでは `[n_ch, n_band]` または `[n_ch, n_band, n_frame]`。

        Returns:
            ビーム出力。単一帯域では shape `[n_beam, n_frame]`、
            帯域込みでは `[n_beam, n_band]` または `[n_beam, n_band, n_frame]`。
        """
        snapshots = np.asarray(X)
        if snapshots.ndim == 2 and self.coefficients.ndim == 3:
            return apply_beamformer_bands(snapshots, self.coefficients)
        return apply_beamformer(snapshots, self.coefficients)


class CBFOverlapSaveBeamformer:
    """帯域別CBF実適用係数をoverlap-save FIRとして逐次適用する。

    入力は各帯域の複素時間列であり、各帯域を独立に FFT ブロック処理して
    ビーム出力へ変換する。フィルタ設計は固定で、適応係数更新は行わない。
    """

    def __init__(
        self,
        steering: np.ndarray,
        frame_size: int = 2048,
        valid_size: int = 1024,
        channel_window: np.ndarray | None = None,
    ) -> None:
        if valid_size <= 0:
            raise ValueError("valid_size must be positive.")
        if frame_size <= 0:
            raise ValueError("frame_size must be positive.")
        if valid_size > frame_size:
            raise ValueError("valid_size must not exceed frame_size.")

        steering_array = np.asarray(steering, dtype=np.complex64)
        if steering_array.ndim == 2:
            steering_array = steering_array[:, :, np.newaxis]
        if steering_array.ndim != 3:
            raise ValueError("steering must have shape (n_ch, n_beam, n_band).")
        if channel_window is not None:
            steering_array = apply_channel_window_to_steering(steering_array, channel_window)

        self.frame_size = frame_size
        self.valid_size = valid_size
        self.filter_ffts = design_cbf_overlap_save_filters(steering_array, frame_size=frame_size)
        self.n_band = steering_array.shape[2]
        self.buffers = [
            OverlapSaveBuffer(frame_size=frame_size, valid_size=valid_size, axis=-1)
            for _ in range(self.n_band)
        ]
        self.valid_extractors = [
            ValidRegionExtractor(frame_size=frame_size, valid_size=valid_size, axis=-1)
            for _ in range(self.n_band)
        ]

    def process(self, X: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """複素サブバンド時系列ブロックを overlap-save CBF で処理する。

        Args:
            X: 入力サブバンド。shape は `[n_ch, n_band, n_sample]`。
                axis=0 はチャネル、axis=1 は帯域、axis=2 は時間サンプルである。

        Returns:
            `(band_idx, valid_block)` のリスト。`valid_block` の shape は
            `[n_beam, valid_size]`。
        """
        subbands = np.asarray(X, dtype=np.complex64)
        if subbands.ndim != 3:
            raise ValueError("X must have shape (n_ch, n_band, n_sample).")
        if subbands.shape[1] != self.n_band:
            raise ValueError("X and steering must agree on n_band.")

        outputs: list[tuple[int, np.ndarray]] = []
        for band_idx in range(self.n_band):
            frames = self.buffers[band_idx].process(subbands[:, band_idx, :])
            for frame in frames:
                # frame shape: [n_ch, frame_size]
                # axis=-1 の FFT により、各チャネル時間列を周波数ビンへ写す。
                frame_fft = np.fft.fft(frame, n=self.frame_size, axis=-1)
                filtered_frame = apply_beamformer_filter_fft(
                    frame_fft,
                    self.filter_ffts[:, :, band_idx, :],
                )
                # IFFT 後の shape は [n_beam, frame_size]。
                # overlap-save の無効先頭区間は後段 Extractor が除去する。
                time_frame = np.fft.ifft(filtered_frame, n=self.frame_size, axis=-1)
                valid = self.valid_extractors[band_idx].process(time_frame)
                outputs.append((band_idx, valid))
        return outputs

    def flush(self) -> list[tuple[int, np.ndarray]]:
        """末尾端数をゼロ詰めして最終有効区間を回収する。"""
        outputs: list[tuple[int, np.ndarray]] = []
        for band_idx in range(self.n_band):
            frames = self.buffers[band_idx].flush(pad=True, fill_value=0.0)
            for frame in frames:
                frame_fft = np.fft.fft(frame, n=self.frame_size, axis=-1)
                filtered_frame = apply_beamformer_filter_fft(
                    frame_fft,
                    self.filter_ffts[:, :, band_idx, :],
                )
                time_frame = np.fft.ifft(filtered_frame, n=self.frame_size, axis=-1)
                valid = self.valid_extractors[band_idx].process(time_frame)
                outputs.append((band_idx, valid))
        return outputs
