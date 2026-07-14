"""spflow.beamforming.mvdr_filter を実装するモジュール。"""

from __future__ import annotations

import numpy as np

from ..frequency import OverlapSaveBuffer, ValidRegionExtractor, make_filter_fft
from ..level_conversion import LevelConverter, level_20log10_rms

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


def apply_beamformer(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """単一帯域のビームフォーマ重みを観測へ適用する。

    Args:
        X: 観測スナップショット。shape は `[n_ch, n_frame]`。
        weights: ビーム重み。shape は `[n_ch]` または `[n_ch, n_beam]`。

    Returns:
        ビーム出力。shape は `[n_beam, n_frame]`。
    """
    snapshots = np.asarray(X, dtype=np.complex64)
    beam_weights = np.asarray(weights, dtype=np.complex64)

    if snapshots.ndim != 2:
        raise ValueError("X must have shape (n_ch, n_frame).")
    if beam_weights.ndim == 1:
        beam_weights = beam_weights[:, np.newaxis]
    if beam_weights.ndim != 2:
        raise ValueError("weights must have shape (n_ch, n_beam).")
    if snapshots.shape[0] != beam_weights.shape[0]:
        raise ValueError("X and weights must agree on n_ch.")

    # X shape: [n_ch, n_frame]
    # W shape: [n_ch, n_beam]
    # einsum("cf,cb->bf") は各ビームについて y[b, f] = Σ_ch conj(w[ch, b]) x[ch, f] を計算する。
    return np.einsum("cf,cb->bf", snapshots, beam_weights.conj(), optimize=True)


def apply_beamformer_bands(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """帯域ごとのビーム重みを観測へ一括適用する。"""
    snapshots = np.asarray(X, dtype=np.complex64)
    beam_weights = np.asarray(weights, dtype=np.complex64)

    if beam_weights.ndim != 3:
        raise ValueError("weights must have shape (n_ch, n_beam, n_band).")

    if snapshots.ndim == 2:
        if snapshots.shape[0] != beam_weights.shape[0] or snapshots.shape[1] != beam_weights.shape[2]:
            raise ValueError("X and weights must agree on n_ch and n_band.")
        # X shape: [n_ch, n_band]
        # W shape: [n_ch, n_beam, n_band]
        # 帯域軸 b を保ったまま ch 軸だけを内積する。
        return np.einsum("cb,cdb->db", snapshots, beam_weights.conj(), optimize=True)

    if snapshots.ndim == 3:
        if snapshots.shape[0] != beam_weights.shape[0] or snapshots.shape[1] != beam_weights.shape[2]:
            raise ValueError("X and weights must agree on n_ch and n_band.")
        # X shape: [n_ch, n_band, n_frame]
        # 出力 shape: [n_beam, n_band, n_frame]。
        return np.einsum("cbf,cdb->dbf", snapshots, beam_weights.conj(), optimize=True)

    raise ValueError("X must have shape (n_ch, n_band) or (n_ch, n_band, n_frame).")


def apply_beamformer_filter_fft(X_fft: np.ndarray, filter_fft: np.ndarray) -> np.ndarray:
    """overlap-save 用フィルタ FFT をマルチチャネルフレームへ適用する。

    `filter_fft` 側へあらかじめ重みの複素共役を焼き込んでいるため、
    実行時は単純な積和だけで `w^H x` と等価な投影を行える。
    """
    spectra = np.asarray(X_fft, dtype=np.complex64)
    filters = np.asarray(filter_fft, dtype=np.complex64)

    if spectra.ndim != 2:
        raise ValueError("X_fft must have shape (n_ch, n_freq).")
    if filters.ndim != 3:
        raise ValueError("filter_fft must have shape (n_ch, n_beam, n_freq).")
    if spectra.shape[0] != filters.shape[0] or spectra.shape[1] != filters.shape[2]:
        raise ValueError("X_fft and filter_fft must agree on n_ch and n_freq.")

    # X_fft shape: [n_ch, n_freq]
    # H shape: [n_ch, n_beam, n_freq]
    # 周波数ビンごとに Σ_ch X[ch, k] H[ch, beam, k] を計算する。
    return np.einsum("cf,cbf->bf", spectra, filters, optimize=True)


def design_mvdr_overlap_save_filters(weights: np.ndarray, frame_size: int) -> np.ndarray:
    """MVDR 重みを overlap-save 用フィルタ FFT へ変換する。"""
    beam_weights = np.asarray(weights, dtype=np.complex64)
    if beam_weights.ndim == 2:
        beam_weights = beam_weights[:, :, np.newaxis]
    if beam_weights.ndim != 3:
        raise ValueError("weights must have shape (n_ch, n_beam) or (n_ch, n_beam, n_band).")

    taps = np.conjugate(beam_weights)[..., np.newaxis]
    return make_filter_fft(taps, frame_size=frame_size, axis=-1)


class MVDRFilter:
    """保持済みまたは外部指定の MVDR 重みを観測へ適用する。

    このクラスは重み適用だけを担当し、共分散推定や重み再設計は責務に含めない。
    """

    def __init__(self, weights: np.ndarray | None = None) -> None:
        self.weights = None if weights is None else np.asarray(weights, dtype=np.complex64)

    def update_weights(self, weights: np.ndarray) -> None:
        """保持する MVDR 重みを更新する。

        Args:
            weights: 新しい重み。shape は `[n_ch, n_beam]` または
                `[n_ch, n_beam, n_band]`。
        """
        self.weights = np.asarray(weights, dtype=np.complex64)

    def process(self, X: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        """観測を MVDR 重みでビーム出力へ変換する。

        Args:
            X: 観測スナップショット。単一帯域では `[n_ch, n_frame]`、
                帯域込みでは `[n_ch, n_band]` または `[n_ch, n_band, n_frame]`。
            weights: 今回だけ使う重み。省略時は内部保持重みを用いる。

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
    """帯域別 MVDR 重みを overlap-save FIR として逐次適用する。

    各帯域の複素時間列を独立に FFT ブロック処理し、MVDR 重みを周波数領域 FIR
    として畳み込む。重み更新と実行は分離し、処理中の適応推定はここでは行わない。
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
        """帯域別 MVDR 重みと対応するフィルタ FFT を更新する。"""
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
