"""固有ベクトル・ビーム対応付けと除外に基づく EBAE 重みを設計する。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]
ComplexArray = NDArray[np.complexfloating[Any, Any]]
IntArray = NDArray[np.integer[Any]]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class EbaeConfig:
    """EBAE の信号数推定と固有ベクトル除外条件を保持する。

    このクラスは入力共分散の独立 snapshot レート、積分時間、sigmoid パラメータ、
    diagonal loading 係数を入力として保持する。重みや信号を計算する責務は持たない。
    信号処理上は、各 FFT bin で独立に実行する EBAE 重み更新の固定条件に位置づく。

    Attributes:
        snapshot_rate_hz: 入力共分散へ入る独立 snapshot のレート。単位は snapshot/s。
        integration_time_sec: 入力共分散の積分時間。単位は s。
        sigmoid_slope: 固有ベクトル除外係数 sigmoid の傾き ``sigm_a``。無次元。
        sigmoid_midpoint: sigmoid の変曲点 ``sigm_b``。無次元。
        diagonal_loading: ロバスト化係数 ``DL``。無次元で、1 を既定値とする。
        normalization_floor: 分母を零と判定する絶対下限。無次元。
    """

    snapshot_rate_hz: float
    integration_time_sec: float
    sigmoid_slope: float = 10.0
    sigmoid_midpoint: float = 0.5
    diagonal_loading: float = 1.0
    normalization_floor: float = 1.0e-12

    def __post_init__(self) -> None:
        """設定値の範囲を検証する。

        Raises:
            ValueError: レート、積分時間、sigmoid、loading、または安定化下限が不正な場合。
        """
        if self.snapshot_rate_hz <= 0.0:
            raise ValueError("snapshot_rate_hz must be positive.")
        if self.integration_time_sec <= 0.0:
            raise ValueError("integration_time_sec must be positive.")
        if self.sigmoid_slope <= 0.0:
            raise ValueError("sigmoid_slope must be positive.")
        if not 0.0 <= self.sigmoid_midpoint <= 1.0:
            raise ValueError("sigmoid_midpoint must be in [0, 1].")
        if self.diagonal_loading < 0.0:
            raise ValueError("diagonal_loading must be non-negative.")
        if self.normalization_floor <= 0.0:
            raise ValueError("normalization_floor must be positive.")


@dataclass(frozen=True)
class EbaeBandResult:
    """単一 FFT bin の EBAE 設計結果を表す。

    入力共分散から得た固有値・固有ベクトル、N/E AIC、MUSIC 対応方位、適応重みを
    固定 shape で返す。共分散推定、FFT、重み適用は責務に含めない。

    Attributes:
        weights: EBAE 重み。shape は ``[n_ch,n_beam]``。
        eigenvalues: 降順固有値。shape は ``[n_ch]``。
        eigenvectors: 対応する固有ベクトル。shape は ``[n_ch,n_ch]``、axis=1 が固有vector。
        aic_values: 候補信号数ごとの N/E AIC。shape は ``[min(n_ch,L)]``。
        signal_count: 推定信号数 ``Ns``。
        music_spectrum: MUSIC の線形疑似スペクトル。shape は ``[n_beam]``。
        associated_beam_indices: 信号固有vectorに対応する方位index。shape は ``[Ns]``。
        used_fallback: 数値異常により CBF 重みを返した場合は True。
    """

    weights: ComplexArray
    eigenvalues: FloatArray
    eigenvectors: ComplexArray
    aic_values: FloatArray
    signal_count: int
    music_spectrum: FloatArray
    associated_beam_indices: IntArray
    used_fallback: bool


@dataclass(frozen=True)
class EbaeResult:
    """全 FFT bin の EBAE 設計結果を表す。

    各 bin を完全に独立して設計した重みと診断量を返す。FFT、S/T 共分散更新、
    ビーム出力計算は責務に含めない。

    Attributes:
        weights: EBAE 重み。shape は ``[n_ch,n_beam,n_bin]``。
        signal_counts: bin ごとの推定信号数。shape は ``[n_bin]``。
        music_spectra: MUSIC 疑似スペクトル。shape は ``[n_beam,n_bin]``。
        associated_beam_indices: 対応方位。shape は ``[n_bin,n_ch-1]``。未使用要素は -1。
        fallback_bins: CBF fallback の有無。shape は ``[n_bin]``。
    """

    weights: ComplexArray
    signal_counts: IntArray
    music_spectra: FloatArray
    associated_beam_indices: IntArray
    fallback_bins: BoolArray


def estimate_signal_count_ne_aic(
    eigenvalues: ComplexArray | FloatArray, snapshot_count: int
) -> tuple[int, FloatArray]:
    """Nadakuditi/Edelman AIC により信号数を推定する。

    Args:
        eigenvalues: 降順の共分散固有値。shape は ``[M]``、単位は入力power。
        snapshot_count: 共分散積分に使った独立 snapshot 数 ``L``。

    Returns:
        ``(Ns, aic_values)``。``Ns`` は最小 AIC の候補index、``aic_values`` の
        shape は ``[min(M,L)]``。

    Raises:
        ValueError: 固有値が1次元でない、非有限、負、または ``L <= 0`` の場合。
    """
    input_values = np.asarray(eigenvalues)
    real_dtype = (
        np.dtype(np.float32)
        if input_values.dtype in (np.dtype(np.float32), np.dtype(np.complex64))
        else np.dtype(np.float64)
    )
    values = np.asarray(np.real(input_values), dtype=real_dtype)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("eigenvalues must have shape (M,) with M > 0.")
    if not bool(np.all(np.isfinite(values))) or bool(np.any(values < 0.0)):
        raise ValueError("eigenvalues must be finite and non-negative.")
    if snapshot_count <= 0:
        raise ValueError("snapshot_count must be positive.")

    sensor_count = values.size
    candidate_count = min(sensor_count, snapshot_count)
    c = sensor_count / float(snapshot_count)
    aic_values = np.empty(candidate_count, dtype=real_dtype)
    for signal_count in range(candidate_count):
        # lambda[n:M] は候補 n を信号部分として除いた雑音固有値である。
        # N/E AIC の t_n は雑音固有値の二次momentと一次moment二乗の比を用いる。
        noise_values = values[signal_count:]
        noise_sum = float(np.sum(noise_values))
        if noise_sum <= 0.0:
            # 雑音powerが厳密に零なら統計量を定義できないため、その候補を選択不能にする。
            aic_values[signal_count] = np.inf
            continue
        t_n = (
            noise_values.size * float(np.sum(noise_values * noise_values)) / (noise_sum * noise_sum)
        )
        t_d = sensor_count * (t_n - (1.0 + c))
        aic_values[signal_count] = (t_d * t_d) / (2.0 * c * c) + 2.0 * (signal_count + 1)

    if not bool(np.any(np.isfinite(aic_values))):
        raise ValueError("N/E AIC is undefined because all noise eigenvalue sums are zero.")
    return int(np.argmin(aic_values)), aic_values


def calculate_music_spectrum(
    noise_eigenvectors: ComplexArray, steering: ComplexArray
) -> FloatArray:
    """雑音部分空間から MUSIC 疑似スペクトルを計算する。

    Args:
        noise_eigenvectors: 雑音固有vector。shape は ``[n_ch,n_noise]``。
        steering: 未正規化ステアリング。shape は ``[n_ch,n_beam]``。

    Returns:
        ``1 / sum_i |u_i^H a(theta)|^2``。shape は ``[n_beam]``、線形値。

    Raises:
        ValueError: shape、有限性、または雑音固有vector数が不正な場合。
    """
    complex_dtype = (
        np.dtype(np.complex64)
        if np.result_type(noise_eigenvectors, steering) == np.dtype(np.complex64)
        else np.dtype(np.complex128)
    )
    real_dtype = (
        np.dtype(np.float32) if complex_dtype == np.dtype(np.complex64) else np.dtype(np.float64)
    )
    noise_space = np.asarray(noise_eigenvectors, dtype=complex_dtype)
    steering_matrix = np.asarray(steering, dtype=complex_dtype)
    if noise_space.ndim != 2 or noise_space.shape[1] == 0:
        raise ValueError("noise_eigenvectors must have shape (n_ch, n_noise) with n_noise > 0.")
    if steering_matrix.ndim != 2 or steering_matrix.shape[0] != noise_space.shape[0]:
        raise ValueError("steering must have shape (n_ch, n_beam).")
    if not bool(np.all(np.isfinite(noise_space))) or not bool(np.all(np.isfinite(steering_matrix))):
        raise ValueError("noise eigenvectors and steering must be finite.")

    # projection shape: [n_noise,n_beam]。各方位steeringの雑音部分空間への射影を表す。
    projection = noise_space.conj().T @ steering_matrix
    denominator = np.sum(np.abs(projection) ** 2, axis=0)
    # 雑音部分空間と厳密に直交する理想方位は MUSIC=+inf とし、最大値選択を保つ。
    return np.divide(
        1.0,
        denominator,
        out=np.full(denominator.shape, np.inf, dtype=real_dtype),
        where=denominator > 0.0,
    )


def _select_music_peaks(music_spectrum: FloatArray, signal_count: int) -> IntArray:
    if signal_count == 0:
        return np.empty(0, dtype=np.int64)
    if signal_count > music_spectrum.size:
        raise ValueError("signal_count must not exceed n_beam.")

    # 最小方位間隔は設けず、理想条件で異なる最大値が得られるという方式前提に従う。
    # 安定sortにより同値の場合も小さいbeam indexを先にし、結果を決定論的にする。
    ordered = np.argsort(-music_spectrum, kind="stable")
    return np.asarray(ordered[:signal_count], dtype=np.int64)


def design_ebae_weights_band(
    covariance: ComplexArray,
    steering: ComplexArray,
    *,
    snapshot_count: int,
    config: EbaeConfig,
) -> EbaeBandResult:
    """単一 FFT bin の EBAE 重みを設計する。

    Args:
        covariance: 入力空間共分散。shape は ``[n_ch,n_ch]``、単位は入力power。
        steering: 未正規化ステアリング。shape は ``[n_ch,n_beam]``。
        snapshot_count: N/E AIC に使う独立 snapshot 数 ``L=rate*T``。
        config: EBAE の固定設計条件。

    Returns:
        固有分解、信号数、MUSIC 対応方位、完成 EBAE 重みを含む結果。

    Raises:
        ValueError: shape、有限性、snapshot数、または設定が不正な場合。

    Notes:
        異常な正規化分母が生じた beam は、未完成の適応重みを公開せず CBF へ戻す。
    """
    complex_dtype = (
        np.dtype(np.complex64)
        if np.result_type(covariance, steering) == np.dtype(np.complex64)
        else np.dtype(np.complex128)
    )
    real_dtype = (
        np.dtype(np.float32) if complex_dtype == np.dtype(np.complex64) else np.dtype(np.float64)
    )
    covariance_matrix = np.asarray(covariance, dtype=complex_dtype)
    steering_matrix = np.asarray(steering, dtype=complex_dtype)
    if covariance_matrix.ndim != 2 or covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        raise ValueError("covariance must have shape (n_ch, n_ch).")
    if steering_matrix.ndim != 2 or steering_matrix.shape[0] != covariance_matrix.shape[0]:
        raise ValueError("steering must have shape (n_ch, n_beam).")
    if not bool(np.all(np.isfinite(covariance_matrix))) or not bool(
        np.all(np.isfinite(steering_matrix))
    ):
        raise ValueError("covariance and steering must be finite.")
    if snapshot_count <= 0:
        raise ValueError("snapshot_count must be positive.")
    configured_snapshot_count = config.snapshot_rate_hz * config.integration_time_sec
    if not np.isclose(configured_snapshot_count, float(snapshot_count), rtol=1.0e-9, atol=1.0e-9):
        raise ValueError("snapshot_rate_hz * integration_time_sec must equal snapshot_count.")

    # 有限snapshotから構成した共分散には丸め誤差や上流の片側更新による
    # anti-Hermitian成分が残り得る。評価を例外で止めず、最も近いHermitian行列
    # (R + R^H) / 2へ射影して、積分不足の影響は固有値・AIC・重みへ残す。
    covariance_hermitian = np.asarray(
        0.5 * (covariance_matrix + covariance_matrix.conj().T), dtype=complex_dtype
    )

    # eigh は Hermitian 共分散に対して実固有値と直交固有vectorを返す。
    # 戻り順は昇順なので、信号部分を先頭に置くため両方を降順へ反転する。
    eigenvalues_ascending, eigenvectors_ascending = np.linalg.eigh(covariance_hermitian)
    eigenvalues = np.maximum(np.real(eigenvalues_ascending[::-1]), 0.0)
    eigenvectors = eigenvectors_ascending[:, ::-1]
    signal_count, aic_values = estimate_signal_count_ne_aic(eigenvalues, snapshot_count)

    # steering_norm[beam] = a^H a。w0=a/(a^H a) は CBF の無歪重みである。
    steering_norm = np.sum(np.abs(steering_matrix) ** 2, axis=0)
    if bool(np.any(steering_norm <= config.normalization_floor)):
        raise ValueError("steering vectors must be non-zero.")
    fixed_weights = steering_matrix / steering_norm[np.newaxis, :]

    noise_eigenvectors = eigenvectors[:, signal_count:]
    music_spectrum = calculate_music_spectrum(noise_eigenvectors, steering_matrix)
    associated_beam_indices = _select_music_peaks(music_spectrum, signal_count)
    if signal_count == 0:
        return EbaeBandResult(
            weights=fixed_weights,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            aic_values=aic_values,
            signal_count=0,
            music_spectrum=music_spectrum,
            associated_beam_indices=associated_beam_indices,
            used_fallback=False,
        )

    # alpha は雑音部分固有値の算術平均であり、各binの雑音power推定値を表す。
    noise_power = float(np.mean(eigenvalues[signal_count:]))
    temporary_weights = fixed_weights.copy()
    for signal_index in range(signal_count):
        signal_vector = eigenvectors[:, signal_index]
        signal_eigenvalue = float(eigenvalues[signal_index])
        beta_denominator = signal_eigenvalue + config.diagonal_loading * noise_power
        beta = (
            0.0
            if beta_denominator <= config.normalization_floor
            else ((signal_eigenvalue - noise_power) / beta_denominator)
        )

        # overlap[beam] = u_i^H w0(theta_b)。固有modeをCBF重みから除く射影係数である。
        overlap = signal_vector.conj() @ fixed_weights
        fixed_power = np.sum(np.abs(fixed_weights) ** 2, axis=0)
        # rho_i(theta_b)=|u_i^H w0(theta_b)|^2/|w0(theta_b)^H w0(theta_b)|。
        # rho_i は無次元で、信号固有vectorと待受CBF重みの正規化重なりpowerを表す。
        rho_i = np.abs(overlap) ** 2 / np.abs(fixed_power)

        # δ_i(theta_b) は、対応方位以外では1、対応方位ではご提示の反転sigmoid式を使う。
        # 1 - 1/(1+exp(-sigm_a*(|u_i^H w0|^2/|w0^H w0|-sigm_b)))
        # により、対応信号とCBF重みの整合が高いほど、その固有modeを除外しない。
        delta = np.ones(steering_matrix.shape[1], dtype=real_dtype)
        matched_beam_index = int(associated_beam_indices[signal_index])
        sigmoid_argument = config.sigmoid_slope * (
            rho_i[matched_beam_index] - config.sigmoid_midpoint
        )
        delta[matched_beam_index] = 1.0 - 1.0 / (1.0 + np.exp(-sigmoid_argument))

        # signal_vector[:,None] * overlap[None,:] shape: [n_ch,n_beam]。
        # 各待受beamから信号固有mode方向の成分だけを δ_i β_i の強さで除く。
        temporary_weights -= (
            beta * signal_vector[:, np.newaxis] * overlap[np.newaxis, :] * delta[np.newaxis, :]
        )

    # 最終分母には正規化前steering a(theta_b) を用い、a^H w_opt=1を保証する。
    normalization = np.sum(steering_matrix.conj() * temporary_weights, axis=0)
    invalid = (~np.isfinite(normalization)) | (np.abs(normalization) <= config.normalization_floor)
    weights = temporary_weights.copy()
    weights[:, ~invalid] /= normalization[np.newaxis, ~invalid]
    if bool(np.any(invalid)):
        # 正規化不能な途中重みは完成値として公開せず、該当bin全体を安全なCBFへ戻す。
        weights = fixed_weights
    return EbaeBandResult(
        weights=weights,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        aic_values=aic_values,
        signal_count=signal_count,
        music_spectrum=music_spectrum,
        associated_beam_indices=associated_beam_indices,
        used_fallback=bool(np.any(invalid)),
    )


def design_ebae_weights(
    covariance: ComplexArray,
    steering: ComplexArray,
    *,
    config: EbaeConfig,
) -> EbaeResult:
    """全 FFT bin の EBAE 重みを完全に独立して設計する。

    Args:
        covariance: 入力共分散。shape は ``[n_bin,n_ch,n_ch]``。
        steering: 未正規化ステアリング。shape は ``[n_ch,n_beam,n_bin]``。
        config: ``rate*T=M^2`` を満たす EBAE 設定。

    Returns:
        bin 別の完成重み、信号数、MUSIC、対応方位、fallback 状態。

    Raises:
        ValueError: 入力 shape または ``rate*T=M^2`` の契約を満たさない場合。
    """
    complex_dtype = (
        np.dtype(np.complex64)
        if np.result_type(covariance, steering) == np.dtype(np.complex64)
        else np.dtype(np.complex128)
    )
    real_dtype = (
        np.dtype(np.float32) if complex_dtype == np.dtype(np.complex64) else np.dtype(np.float64)
    )
    covariance_array = np.asarray(covariance, dtype=complex_dtype)
    steering_array = np.asarray(steering, dtype=complex_dtype)
    if covariance_array.ndim != 3 or covariance_array.shape[1] != covariance_array.shape[2]:
        raise ValueError("covariance must have shape (n_bin, n_ch, n_ch).")
    if steering_array.ndim != 3:
        raise ValueError("steering must have shape (n_ch, n_beam, n_bin).")
    if (
        covariance_array.shape[0] != steering_array.shape[2]
        or covariance_array.shape[1] != steering_array.shape[0]
    ):
        raise ValueError("covariance and steering must agree on n_bin and n_ch.")

    bin_count, channel_count = covariance_array.shape[0], covariance_array.shape[1]
    beam_count = steering_array.shape[1]
    snapshot_count = channel_count * channel_count
    weights = np.empty((channel_count, beam_count, bin_count), dtype=complex_dtype)
    signal_counts = np.empty(bin_count, dtype=np.int64)
    music_spectra = np.empty((beam_count, bin_count), dtype=real_dtype)
    associated = np.full((bin_count, max(channel_count - 1, 0)), -1, dtype=np.int64)
    fallback_bins = np.empty(bin_count, dtype=np.bool_)
    for bin_index in range(bin_count):
        band_result = design_ebae_weights_band(
            covariance_array[bin_index],
            steering_array[:, :, bin_index],
            snapshot_count=snapshot_count,
            config=config,
        )
        weights[:, :, bin_index] = band_result.weights
        signal_counts[bin_index] = band_result.signal_count
        music_spectra[:, bin_index] = band_result.music_spectrum
        associated[bin_index, : band_result.signal_count] = band_result.associated_beam_indices
        fallback_bins[bin_index] = band_result.used_fallback
    return EbaeResult(weights, signal_counts, music_spectra, associated, fallback_bins)
