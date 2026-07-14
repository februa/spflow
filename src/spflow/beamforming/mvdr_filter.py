"""spflow.beamforming.mvdr_filter を実装するモジュール。"""

from __future__ import annotations

import numpy as np

from ..frequency import OverlapSaveBuffer, ValidRegionExtractor, make_filter_fft
from ..level_conversion import LevelConverter, level_20log10_rms
from .application import apply_beamformer, apply_beamformer_bands, apply_beamformer_filter_fft

_UNITY_RESPONSE_LEVEL_CONVERTER = LevelConverter.for_definition(
    level_20log10_rms(reference_rms=1.0, reference_label="unit response")
)


def beam_response_rms_db(response: np.ndarray | complex) -> float:
    """複素ビーム応答の絶対値をunit response基準のRMS levelへ変換する。

    Args:
        response: scalar-likeな複素ビーム応答。無次元。

    Returns:
        `20log10(abs(response))`。単位はdB re unit response。

    Raises:
        ValueError: responseがscalar-likeでない、または非有限の場合。

    境界条件:
        ゼロ応答は`-inf`を返す。この関数は波形RMS測定やFFT正規化を行わない。
    """
    response_scalar = np.asarray(response, dtype=np.complex64)
    if response_scalar.size != 1:
        raise ValueError("response must be scalar-like.")
    return _UNITY_RESPONSE_LEVEL_CONVERTER.output_rms_to_level(
        float(np.abs(response_scalar.reshape(-1)[0]))
    )


def design_mvdr_overlap_save_filters(coefficients: np.ndarray, frame_size: int) -> np.ndarray:
    """MVDR実適用係数をoverlap-save用フィルタFFTへ変換する。

    Args:
        coefficients: `y=h^T x`へ直接使うMVDR係数。shapeは
            `[n_ch, n_beam]`または`[n_ch, n_beam, n_band]`。
        frame_size: FFT長。単位はsample。

    Returns:
        filter FFT。shapeは`[n_ch, n_beam, n_band, frame_size]`。

    Raises:
        ValueError: coefficientsのshapeが想定と異なる場合。
    """
    beam_coefficients = np.asarray(coefficients, dtype=np.complex64)
    if beam_coefficients.ndim == 2:
        beam_coefficients = beam_coefficients[:, :, np.newaxis]
    if beam_coefficients.ndim != 3:
        raise ValueError("coefficients must have shape (n_ch, n_beam) or (n_ch, n_beam, n_band).")

    # 設計側でh=conj(w)へ変換済みなので、FIR tapへ追加の共役を掛けない。
    taps = beam_coefficients[..., np.newaxis]
    return make_filter_fft(taps, frame_size=frame_size, axis=-1)


class MVDRFilter:
    """保持済みまたは外部指定のMVDR実適用係数を観測へ適用する。

    このクラスは`y=h^T x`の係数適用だけを担当し、共分散推定や係数再設計は
    責務に含めない。既存APIとの互換性のため引数名`weights`は維持するが、
    値は設計側で共役規約を反映済みの実適用係数である。
    """

    def __init__(self, weights: np.ndarray | None = None) -> None:
        self.weights = None if weights is None else np.asarray(weights, dtype=np.complex64)

    def update_weights(self, weights: np.ndarray) -> None:
        """保持するMVDR実適用係数を更新する。

        Args:
            weights: 新しい実適用係数。shape は `[n_ch, n_beam]` または
                `[n_ch, n_beam, n_band]`。
        """
        self.weights = np.asarray(weights, dtype=np.complex64)

    def process(self, X: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        """観測をMVDR実適用係数でビーム出力へ変換する。

        Args:
            X: 観測スナップショット。単一帯域では `[n_ch, n_frame]`、
                帯域込みでは `[n_ch, n_band]` または `[n_ch, n_band, n_frame]`。
            weights: 今回だけ使う実適用係数。省略時は内部保持係数を用いる。

        Returns:
            ビーム出力。
        """
        active_weights = self.weights if weights is None else np.asarray(weights, dtype=np.complex64)
        if active_weights is None:
            raise ValueError("weights are not set.")
        if np.asarray(active_weights).ndim == 3:
            return apply_beamformer_bands(X, active_weights)
        return apply_beamformer(X, active_weights)


class MVDROverlapSaveBeamformer:
    """帯域別MVDR実適用係数をoverlap-save FIRとして逐次適用する。

    各帯域の複素時間列を独立にFFTブロック処理し、MVDR係数を周波数領域FIR
    として畳み込む。係数更新と実行は分離し、処理中の適応推定はここでは行わない。
    """

    def __init__(
        self,
        weights: np.ndarray,
        frame_size: int = 2048,
        valid_size: int = 1024,
    ) -> None:
        if valid_size <= 0:
            raise ValueError("valid_size must be positive.")
        if frame_size <= 0:
            raise ValueError("frame_size must be positive.")
        if valid_size > frame_size:
            raise ValueError("valid_size must not exceed frame_size.")

        beam_weights = np.asarray(weights, dtype=np.complex64)
        if beam_weights.ndim == 2:
            beam_weights = beam_weights[:, :, np.newaxis]
        if beam_weights.ndim != 3:
            raise ValueError("weights must have shape (n_ch, n_beam, n_band).")

        self.frame_size = frame_size
        self.valid_size = valid_size
        self.n_band = beam_weights.shape[2]
        self.n_beam = beam_weights.shape[1]
        self.weights = beam_weights.copy()
        self.filter_ffts = design_mvdr_overlap_save_filters(self.weights, frame_size=frame_size)
        self.buffers = [
            OverlapSaveBuffer(frame_size=frame_size, valid_size=valid_size, axis=-1)
            for _ in range(self.n_band)
        ]
        self.valid_extractors = [
            ValidRegionExtractor(frame_size=frame_size, valid_size=valid_size, axis=-1)
            for _ in range(self.n_band)
        ]

    def update_weights(self, weights: np.ndarray) -> None:
        """帯域別MVDR実適用係数と対応するフィルタFFTを更新する。"""
        beam_weights = np.asarray(weights, dtype=np.complex64)
        if beam_weights.ndim == 2:
            beam_weights = beam_weights[:, :, np.newaxis]
        if beam_weights.shape != self.weights.shape:
            raise ValueError("updated weights must match the original shape.")
        self.weights = beam_weights.copy()
        self.filter_ffts = design_mvdr_overlap_save_filters(self.weights, frame_size=self.frame_size)

    def process(self, X: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """入力サブバンドを overlap-save MVDR で処理する。

        Args:
            X: 入力サブバンド。shape は `[n_ch, n_band, n_sample]`。

        Returns:
            `(band_idx, valid_block)` のリスト。`valid_block` の shape は
            `[n_beam, valid_size]`。
        """
        subbands = np.asarray(X, dtype=np.complex64)
        if subbands.ndim != 3:
            raise ValueError("X must have shape (n_ch, n_band, n_sample).")
        if subbands.shape[1] != self.n_band:
            raise ValueError("X and weights must agree on n_band.")

        outputs: list[tuple[int, np.ndarray]] = []
        for band_idx in range(self.n_band):
            frames = self.buffers[band_idx].process(subbands[:, band_idx, :])
            for frame in frames:
                # 各帯域 frame shape は [n_ch, frame_size]。
                # FFT で線形畳み込みを周波数領域の積へ変換する。
                frame_fft = np.fft.fft(frame, n=self.frame_size, axis=-1)
                filtered_frame = apply_beamformer_filter_fft(
                    frame_fft,
                    self.filter_ffts[:, :, band_idx, :],
                )
                # IFFT 後の先頭無効区間は overlap-save の重複領域に対応する。
                time_frame = np.fft.ifft(filtered_frame, n=self.frame_size, axis=-1)
                valid = self.valid_extractors[band_idx].process(time_frame)
                outputs.append((band_idx, valid))
        return outputs

    def flush(self) -> list[tuple[int, np.ndarray]]:
        """末尾不足ブロックをゼロ詰めして最終有効区間を回収する。"""
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
