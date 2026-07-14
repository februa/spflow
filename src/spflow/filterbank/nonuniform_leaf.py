"""非均一フィルタバンク leaf の局所ビームフォーミング処理を実装する。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeVar

import numpy as np

from .._validation import require, require_non_negative_float, require_positive_int
from ..beamforming.cbf import design_cbf_coefficients
from ..beamforming.covariance import CovarianceEstimator, forgetting_factor_from_integration_time
from ..beamforming.mvdr_filter import apply_beamformer_filter_fft
from ..beamforming.mvdr_weight_designer import design_mvdr_coefficients_bands
from ..frequency import OverlapSaveBuffer, ValidRegionExtractor, make_filter_fft
from .formal_complex_pr_stage import FormalBandPacket
from .nonuniform_tree import NonuniformBandPacket, NonuniformBandSpec

BeamformerMode = Literal["cbf", "mvdr"]
LeafOutputPathMode = Literal["leaf_independent_one_sided"]
PacketT = TypeVar("PacketT", NonuniformBandPacket, FormalBandPacket)


@dataclass(frozen=True)
class NonuniformLeafProcessorConfig:
    """leaf 単位ビームフォーミングの設定を表す。

    旧設計との接続のため `long_fft_*` / `short_fft_*` という名前は残すが、
    正式実装では次の 1 本構成だけを許可する。

    - output path: overlap-save 契約で時間波形を連続出力する経路
    - statistics path: 正側ビンだけで共分散と重み更新を行う経路

    両 path は同じ frame FFT 条件を共有し、statistics path はその正側ビンだけを使う。
    """

    spec: NonuniformBandSpec
    used_channels: np.ndarray
    steering: np.ndarray
    long_fft_frame_size: int
    long_fft_valid_size: int
    short_fft_size: int
    short_fft_hop_size: int
    beamformer_mode: BeamformerMode = "mvdr"
    # 外部設定から不正文字列が来た場合も __post_init__ で明示的な ValueError にするため、入力型は str とする。
    output_path_mode: str = "leaf_independent_one_sided"
    integration_time: float = 0.0
    weight_update_period: float = 0.0
    diag_load: float = 1e-3

    def __post_init__(self) -> None:
        """設定値を正規化し、正式 one-side OLS 構造の前提を固定する。"""

        # 入力正規化
        used = np.asarray(self.used_channels, dtype=np.int32)
        steering = np.asarray(self.steering, dtype=np.complex64)

        # 入力検証
        require(used.ndim == 1 and used.size > 0, "used_channels must be a non-empty 1D array.")
        require(bool(np.all(used >= 0)), "used_channels must be non-negative.")
        require(np.unique(used).size == used.size, "used_channels must not contain duplicates.")
        require_positive_int("long_fft_frame_size", self.long_fft_frame_size)
        require_positive_int("long_fft_valid_size", self.long_fft_valid_size)
        require_positive_int("short_fft_size", self.short_fft_size)
        require_positive_int("short_fft_hop_size", self.short_fft_hop_size)
        require(
            self.long_fft_valid_size <= self.long_fft_frame_size,
            "long_fft_valid_size must not exceed long_fft_frame_size.",
        )
        require(
            self.short_fft_hop_size <= self.short_fft_size,
            "short_fft_hop_size must not exceed short_fft_size.",
        )
        require_non_negative_float("integration_time", self.integration_time)
        require_non_negative_float("weight_update_period", self.weight_update_period)
        require_non_negative_float("diag_load", self.diag_load)
        require(self.beamformer_mode in ("cbf", "mvdr"), "beamformer_mode must be 'cbf' or 'mvdr'.")
        require(
            self.output_path_mode == "leaf_independent_one_sided",
            "Only 'leaf_independent_one_sided' is supported in the formal nonuniform leaf implementation.",
        )
        require(
            steering.ndim in (2, 3),
            "steering must have shape (n_ch, n_beam) or (n_ch, n_beam, n_freq).",
        )
        require(
            self.long_fft_frame_size == self.short_fft_size,
            "Formal one-side OLS leaf requires long_fft_frame_size == short_fft_size.",
        )
        require(
            self.long_fft_valid_size == self.short_fft_hop_size,
            "Formal one-side OLS leaf requires long_fft_valid_size == short_fft_hop_size.",
        )

        object.__setattr__(self, "used_channels", used)
        object.__setattr__(self, "steering", steering)

    @property
    def uses_shared_one_sided_output(self) -> bool:
        """正式構造では常に shared frame FFT を使う。"""
        return True

    @property
    def output_fft_size(self) -> int:
        """output path が使う共有 frame FFT 次数を返す。"""
        return self.short_fft_size

    @property
    def output_hop_size(self) -> int:
        """output path が使う overlap-save hop を返す。"""
        return self.short_fft_hop_size

    @property
    def statistics_fft_size(self) -> int:
        """statistics path が参照する共有 frame FFT 次数を返す。"""
        return self.short_fft_size

    @property
    def statistics_hop_size(self) -> int:
        """statistics path の更新 hop を返す。"""
        return self.short_fft_hop_size


def resample_frequency_response(response: np.ndarray, target_fft_size: int, axis: int = -1) -> np.ndarray:
    """周波数応答の FFT 次数を変換する。"""

    require_positive_int("target_fft_size", target_fft_size)

    arr = np.asarray(response, dtype=np.complex64)
    work_axis = axis
    if work_axis < 0:
        work_axis += arr.ndim
    if work_axis < 0 or work_axis >= arr.ndim:
        raise ValueError("axis is out of bounds for response.")

    if arr.shape[work_axis] == target_fft_size:
        return arr.copy()

    time = np.fft.ifft(arr, axis=work_axis)
    moved = np.moveaxis(time, work_axis, -1)
    resized = np.zeros(moved.shape[:-1] + (target_fft_size,), dtype=np.complex64)
    keep = min(moved.shape[-1], target_fft_size)
    resized[..., :keep] = moved[..., :keep]
    spectrum = np.fft.fft(resized, axis=-1)
    return np.moveaxis(spectrum, -1, work_axis)


def one_sided_bin_count(fft_size: int) -> int:
    """one-side 複素スペクトルのビン数を返す。"""

    require_positive_int("fft_size", fft_size)
    return fft_size // 2 + 1


def expand_one_sided_response(response: np.ndarray, fft_size: int, axis: int = -1) -> np.ndarray:
    """one-side 応答から full FFT 応答を鏡像展開で再構成する。"""

    require_positive_int("fft_size", fft_size)

    arr = np.asarray(response, dtype=np.complex64)
    work_axis = axis
    if work_axis < 0:
        work_axis += arr.ndim
    if work_axis < 0 or work_axis >= arr.ndim:
        raise ValueError("axis is out of bounds for response.")

    expected = one_sided_bin_count(fft_size)
    if arr.shape[work_axis] != expected:
        raise ValueError("response does not match the one-sided bin count of fft_size.")

    moved = np.moveaxis(arr, work_axis, -1)
    full = np.zeros(moved.shape[:-1] + (fft_size,), dtype=np.complex64)
    full[..., :expected] = moved
    if fft_size > 2:
        full[..., expected:] = moved[..., 1:-1][..., ::-1]
    return np.moveaxis(full, -1, work_axis)


def expand_positive_spectrum_to_full_fft(response: np.ndarray, fft_size: int, axis: int = -1) -> np.ndarray:
    """正側だけ持つ複素スペクトルを負側 zero-fill で full FFT 配列へ展開する。"""

    require_positive_int("fft_size", fft_size)

    arr = np.asarray(response, dtype=np.complex64)
    work_axis = axis
    if work_axis < 0:
        work_axis += arr.ndim
    if work_axis < 0 or work_axis >= arr.ndim:
        raise ValueError("axis is out of bounds for response.")

    expected = one_sided_bin_count(fft_size)
    if arr.shape[work_axis] != expected:
        raise ValueError("response does not match the one-sided bin count of fft_size.")

    moved = np.moveaxis(arr, work_axis, -1)
    full = np.zeros(moved.shape[:-1] + (fft_size,), dtype=np.complex64)
    full[..., :expected] = moved
    return np.moveaxis(full, -1, work_axis)


_OLS_FIR_PINV_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _ols_filter_delay_samples(fft_size: int, valid_size: int) -> int:
    """正式 one-side OLS では追加 delay を入れず 0 を返す。"""

    return 0


def _ols_fir_pinv(fft_size: int, valid_size: int) -> np.ndarray:
    """OLS 制約付き FIR 射影の擬似逆行列を返す。"""

    key = (fft_size, valid_size)
    cached = _OLS_FIR_PINV_CACHE.get(key)
    if cached is not None:
        return cached

    max_filter_length = fft_size - valid_size + 1
    frequency_index = np.arange(fft_size, dtype=np.float32)[:, np.newaxis]
    tap_index = np.arange(max_filter_length, dtype=np.float32)[np.newaxis, :]
    design_matrix = np.exp(-2j * np.pi * frequency_index * tap_index / fft_size).astype(np.complex64)
    pinv = np.linalg.pinv(design_matrix).astype(np.complex64)
    _OLS_FIR_PINV_CACHE[key] = pinv
    return pinv


def design_one_sided_ols_filter_fft(
    coefficients_positive: np.ndarray,
    *,
    fft_size: int,
    valid_size: int,
    axis: int = -1,
) -> np.ndarray:
    """one-side実適用係数からoverlap-save契約を満たすfilter FFTを作る。

    Args:
        coefficients_positive: one-sided周波数格子上の実適用係数。
            frequency axisは`axis`で指定し、長さは`fft_size // 2 + 1`。
        fft_size: full FFT長。単位はsample。
        valid_size: overlap-saveで公開する有効sample数。
        axis: coefficients_positiveの周波数axis。

    Returns:
        causal FIRへ射影したfull filter FFT。shapeは周波数axisだけが
        `fft_size`へ変わり、dtypeはcomplex64。

    Raises:
        ValueError: fft_size、valid_size、axis、one-sided bin数が不正な場合。

    境界条件:
    one-side係数だけを保持しつつoutput pathを正式なoverlap-saveとして扱うため、
    まず暫定 full 応答を定め、その full 応答を

    - `M <= N - H + 1`

    tap の causal FIR へ最小二乗射影する。
    こうして得た FIR を tail zero-padding して `N` 点 FFT したものが
    runtime の `filter FFT` である。
    """

    require_positive_int("fft_size", fft_size)
    require_positive_int("valid_size", valid_size)
    require(valid_size <= fft_size, "valid_size must not exceed fft_size.")

    arr = np.asarray(coefficients_positive, dtype=np.complex64)
    work_axis = axis
    if work_axis < 0:
        work_axis += arr.ndim
    if work_axis < 0 or work_axis >= arr.ndim:
        raise ValueError("axis is out of bounds for coefficients_positive.")

    # 設計器が返したhを周波数応答としてそのままFIRへ射影し、追加の共役は取らない。
    desired_full_response = expand_one_sided_response(arr, fft_size, axis=work_axis)
    frequency_index = np.arange(fft_size, dtype=np.float32)
    delay_samples = _ols_filter_delay_samples(fft_size, valid_size)
    delay_phase = np.exp(-2j * np.pi * frequency_index * delay_samples / fft_size).astype(np.complex64)
    desired_shifted_response = desired_full_response * delay_phase
    moved = np.moveaxis(desired_shifted_response, work_axis, -1)
    pinv = _ols_fir_pinv(fft_size, valid_size)
    causal_taps = np.einsum("mn,...n->...m", pinv, moved, optimize=True).astype(np.complex64)
    return make_filter_fft(np.moveaxis(causal_taps, -1, work_axis), frame_size=fft_size, axis=work_axis)


class NonuniformLeafProcessor:
    """leaf 帯域内だけでビームフォーミングを完結させる処理器。"""

    def __init__(self, config: NonuniformLeafProcessorConfig) -> None:
        """leaf の共有 frame FFT 経路、統計更新、形式メタデータ状態を初期化する。"""

        # ------------------------------------------------------------------
        # 固定設定
        # ------------------------------------------------------------------
        self.config = config
        self.used_channels = config.used_channels.copy()
        self._output_fft_size = config.output_fft_size
        self._output_valid_size = config.output_hop_size

        # ------------------------------------------------------------------
        # overlap-save 出力経路
        # ------------------------------------------------------------------
        self.output_buffer = OverlapSaveBuffer(
            frame_size=self._output_fft_size,
            valid_size=self._output_valid_size,
            axis=-1,
        )
        self.valid_extractor = ValidRegionExtractor(
            frame_size=self._output_fft_size,
            valid_size=self._output_valid_size,
            axis=-1,
        )

        # ------------------------------------------------------------------
        # used_channels の切り出し高速化
        # ------------------------------------------------------------------
        self._used_channels_contiguous = bool(
            self.used_channels.size == 1 or np.all(np.diff(self.used_channels) == 1)
        )
        self._used_channels_start = int(self.used_channels[0])
        self._used_channels_stop = int(self.used_channels[-1]) + 1
        self._used_channels_identity_only = bool(
            self._used_channels_start == 0
            and self._used_channels_stop == self.used_channels.size
            and self._used_channels_contiguous
        )

        # ------------------------------------------------------------------
        # 正側グリッド、初期重み、OLS filter FFT
        # ------------------------------------------------------------------
        self._n_short_positive_bins = one_sided_bin_count(config.statistics_fft_size)
        self._n_output_positive_bins = one_sided_bin_count(self._output_fft_size)
        self._steering_short_positive = self._prepare_steering_positive(config.steering)
        self._current_weights_short_positive = design_cbf_coefficients(self._steering_short_positive)
        self._current_weights_output_positive = self._prepare_output_weights_positive(
            self._current_weights_short_positive
        )
        self._output_alignment_delay_samples = _ols_filter_delay_samples(
            self._output_fft_size,
            self._output_valid_size,
        )
        self._current_filter_fft = design_one_sided_ols_filter_fft(
            self._current_weights_output_positive,
            fft_size=self._output_fft_size,
            valid_size=self._output_valid_size,
            axis=-1,
        )
        self._output_delay_pending: np.ndarray | None = None

        # ------------------------------------------------------------------
        # MVDR 更新状態
        # ------------------------------------------------------------------
        self._statistics_update_rate_hz = float(config.spec.nominal_sample_rate_hz / config.statistics_hop_size)
        self._weight_update_period_frames = max(
            1,
            int(np.ceil(config.weight_update_period * self._statistics_update_rate_hz)),
        )
        self._statistics_frames_since_update = 0
        self._current_covariances_positive = np.zeros(
            (self._n_short_positive_bins, self.used_channels.size, self.used_channels.size),
            dtype=np.complex64,
        )

        # ------------------------------------------------------------------
        # formal tree 接続用の時間メタデータ
        # ------------------------------------------------------------------
        self._formal_input_template: FormalBandPacket | None = None
        self._formal_next_input_time_origin_at_root_rate: int | None = None
        self._formal_next_output_time_origin_at_root_rate: int | None = None

        if config.beamformer_mode == "mvdr":
            alpha = forgetting_factor_from_integration_time(
                config.integration_time,
                self._statistics_update_rate_hz,
            )
            self._cov_estimator = CovarianceEstimator(forgetting_factor=alpha)
        else:
            self._cov_estimator = None

    @property
    def n_used_channels(self) -> int:
        """この leaf が使う実チャネル数を返す。"""
        return int(self.used_channels.size)

    @property
    def n_beam(self) -> int:
        """この leaf が保持するビーム数を返す。"""
        return int(self._current_weights_short_positive.shape[1])

    @property
    def output_path_mode(self) -> LeafOutputPathMode:
        """正式実装の output path 名を返す。"""
        # config 初期化時にこの値以外を拒否しているため、公開値は固定Literalとして返す。
        return "leaf_independent_one_sided"

    @property
    def uses_shared_frame_fft(self) -> bool:
        """正式実装では常に shared frame FFT を使う。"""
        return True

    @property
    def output_fft_size(self) -> int:
        """output path の共有 frame FFT 次数を返す。"""
        return self._output_fft_size

    @property
    def output_valid_size(self) -> int:
        """output path の overlap-save valid 長を返す。"""
        return self._output_valid_size

    @property
    def output_inner_product_bin_count(self) -> int:
        """one-side 重みが保持する正側ビン数を返す。"""
        return self._n_output_positive_bins

    @property
    def output_uses_one_sided_bins(self) -> bool:
        """output path は one-side 重みから OLS filter FFT を構成する。"""
        return True

    @property
    def current_weights_short(self) -> np.ndarray:
        """現在の short-grid 重みを full FFT 形へ戻して返す。"""
        return expand_one_sided_response(
            self._current_weights_short_positive,
            self.config.statistics_fft_size,
            axis=-1,
        )

    @property
    def current_filter_fft(self) -> np.ndarray:
        """現在の output path 用 OLS filter FFT を返す。"""
        return self._current_filter_fft.copy()

    @property
    def current_covariances(self) -> np.ndarray:
        """現在の共分散を full FFT 形へ戻して返す。"""
        return expand_one_sided_response(
            self._current_covariances_positive,
            self.config.statistics_fft_size,
            axis=0,
        )

    def process(self, packet: NonuniformBandPacket) -> list[NonuniformBandPacket]:
        """通常 tree 用 band packet を処理して連続出力へ変換する。"""

        self._validate_packet(packet)
        selected = self._select_channels(packet.samples)
        return self._process_shared_fft_chunks(
            selected,
            lambda frame_fft: self._emit_output_frame_from_fft(frame_fft, packet.spec),
        )

    def flush(self, spec: NonuniformBandSpec | None = None) -> list[NonuniformBandPacket]:
        """残留内部状態を zero-pad flush して packet を返す。"""

        active_spec = self.config.spec if spec is None else spec
        outputs = self._flush_shared_fft_chunks(
            lambda frame_fft: self._emit_output_frame_from_fft(frame_fft, active_spec),
        )
        final_packet = self._finalize_output_delay_packet(active_spec)
        if final_packet is not None:
            outputs.append(final_packet)
        self._reset_after_flush(reset_formal_state=True)
        return outputs

    def process_formal_packet(self, packet: FormalBandPacket) -> list[FormalBandPacket]:
        """formal metadata 付き packet を処理する。"""

        self._validate_formal_packet(packet)
        self._update_formal_input_cursor(packet)
        selected = self._select_channels(packet.complex_samples)
        return self._process_shared_fft_chunks(
            selected,
            lambda frame_fft: self._emit_output_frame_formal_from_fft(frame_fft, packet),
        )

    def flush_formal(self) -> list[FormalBandPacket]:
        """formal metadata 付き残留状態を flush する。"""

        if self._formal_input_template is None:
            self._reset_after_flush(reset_formal_state=False)
            return []

        formal_input_template = self._formal_input_template
        outputs = self._flush_shared_fft_chunks(
            lambda frame_fft: self._emit_output_frame_formal_from_fft(frame_fft, formal_input_template),
        )
        final_packet = self._finalize_output_delay_formal_packet(formal_input_template)
        if final_packet is not None:
            outputs.append(final_packet)
        self._reset_after_flush(reset_formal_state=True)
        return outputs

    def _process_shared_fft_chunks(
        self,
        selected: np.ndarray,
        emit_frame: Callable[[np.ndarray], PacketT | None],
    ) -> list[PacketT]:
        """共有 frame FFT 1 本で output path と statistics path を同時に進める。"""

        outputs: list[PacketT] = []
        for selected_chunk in self._iter_internal_chunks(selected):
            for frame in self.output_buffer.process(selected_chunk):
                frame_fft = np.fft.fft(frame, n=self._output_fft_size, axis=-1)
                packet = emit_frame(frame_fft)
                if packet is not None:
                    outputs.append(packet)
                self._update_statistics_from_frame_fft(frame_fft)
        return outputs

    def _flush_shared_fft_chunks(self, emit_frame: Callable[[np.ndarray], PacketT | None]) -> list[PacketT]:
        """共有 frame FFT 経路の末尾状態を flush する。"""

        outputs: list[PacketT] = []
        for frame in self.output_buffer.flush(pad=True, fill_value=0.0):
            frame_fft = np.fft.fft(frame, n=self._output_fft_size, axis=-1)
            packet = emit_frame(frame_fft)
            if packet is not None:
                outputs.append(packet)
            self._update_statistics_from_frame_fft(frame_fft)
        return outputs

    def _emit_output_frame_from_fft(self, frame_fft: np.ndarray, spec: NonuniformBandSpec) -> NonuniformBandPacket | None:
        """OLS filter FFT を適用し、delay 補償後の valid 部だけを packet として返す。"""

        filtered = apply_beamformer_filter_fft(frame_fft, self._current_filter_fft)
        time_frame = np.fft.ifft(filtered, n=self._output_fft_size, axis=-1)
        valid = self.valid_extractor.process(time_frame)
        aligned = self._consume_output_delay(valid, final=False)
        if aligned.shape[-1] == 0:
            return None
        return NonuniformBandPacket(spec=spec, samples=np.asarray(aligned, dtype=np.complex64).copy())

    def _emit_output_frame_formal_from_fft(
        self,
        frame_fft: np.ndarray,
        packet: FormalBandPacket,
    ) -> FormalBandPacket | None:
        """OLS 出力 frame を formal metadata 付き packet に変換する。"""

        valid_packet = self._emit_output_frame_from_fft(frame_fft, self.config.spec)
        if valid_packet is None:
            return None
        return self._make_formal_output_packet(packet, valid_packet.samples)

    def _finalize_output_delay_packet(self, spec: NonuniformBandSpec) -> NonuniformBandPacket | None:
        """flush 時に残っている delay 補償待ちサンプルを通常 packet として回収する。"""

        aligned = self._consume_output_delay(None, final=True)
        if aligned.shape[-1] == 0:
            return None
        return NonuniformBandPacket(spec=spec, samples=np.asarray(aligned, dtype=np.complex64).copy())

    def _finalize_output_delay_formal_packet(self, packet: FormalBandPacket) -> FormalBandPacket | None:
        """flush 時に残っている delay 補償待ちサンプルを formal packet として回収する。"""

        aligned = self._consume_output_delay(None, final=True)
        if aligned.shape[-1] == 0:
            return None
        return self._make_formal_output_packet(packet, aligned)

    def _make_formal_output_packet(self, packet: FormalBandPacket, samples: np.ndarray) -> FormalBandPacket:
        """formal tree が必要とする時間原点を連続に保って packet 化する。"""

        output_origin = self._formal_output_time_origin()
        step = self._formal_sample_step_at_root_rate()
        valid_len = int(samples.shape[-1])
        self._formal_next_output_time_origin_at_root_rate = output_origin + valid_len * step
        return FormalBandPacket(
            band_id=packet.band_id,
            f_low_hz=packet.f_low_hz,
            f_high_hz=packet.f_high_hz,
            sample_rate_hz=packet.sample_rate_hz,
            time_origin_at_root_rate=output_origin,
            delay_samples_at_root_rate=packet.delay_samples_at_root_rate,
            complex_samples=np.asarray(samples, dtype=np.complex64).copy(),
        )

    def _update_statistics_from_frame_fft(self, frame_fft: np.ndarray) -> None:
        """共有 frame FFT の正側ビンから共分散と MVDR 重みを更新する。"""

        if self.config.beamformer_mode != "mvdr":
            return
        assert self._cov_estimator is not None

        frame_fft_positive = frame_fft[:, : self._n_short_positive_bins]
        self._current_covariances_positive = self._cov_estimator.process_snapshots(
            np.moveaxis(frame_fft_positive, -1, 0),
            normalization=float(self.config.statistics_fft_size),
        )
        self._statistics_frames_since_update += 1
        if self._statistics_frames_since_update >= self._weight_update_period_frames:
            self._refresh_mvdr_weights()
            self._statistics_frames_since_update = 0

    def _refresh_mvdr_weights(self) -> None:
        """最新共分散から MVDR 重みと OLS filter FFT を再設計する。"""

        self._current_weights_short_positive = design_mvdr_coefficients_bands(
            self._current_covariances_positive,
            self._steering_short_positive,
            diag_load=self.config.diag_load,
        )
        self._current_weights_output_positive = self._prepare_output_weights_positive(
            self._current_weights_short_positive
        )
        self._current_filter_fft = design_one_sided_ols_filter_fft(
            self._current_weights_output_positive,
            fft_size=self._output_fft_size,
            valid_size=self._output_valid_size,
            axis=-1,
        )

    def _prepare_output_weights_positive(self, weights_short_positive: np.ndarray) -> np.ndarray:
        """output path が参照する正側重み列を正規化して返す。"""

        arr = np.asarray(weights_short_positive, dtype=np.complex64)
        if arr.shape[-1] != self._n_output_positive_bins:
            raise ValueError("weights_short_positive does not match the configured output bin count.")
        return arr.copy()

    def _prepare_steering_positive(self, steering: np.ndarray) -> np.ndarray:
        """steering を used_channels + 正側ビン表現へ正規化する。"""

        steering_array = np.asarray(steering, dtype=np.complex64)
        if steering_array.ndim == 2:
            steering_array = steering_array[:, :, np.newaxis]

        if steering_array.shape[0] == self.used_channels.size:
            reduced = steering_array
        else:
            if np.max(self.used_channels) >= steering_array.shape[0]:
                raise ValueError("steering does not contain all used_channels.")
            if self._used_channels_contiguous:
                reduced = steering_array[self._used_channels_start : self._used_channels_stop, :, :]
            else:
                reduced = steering_array[self.used_channels, :, :]

        if reduced.shape[2] == 1:
            reduced = np.repeat(reduced, self._n_short_positive_bins, axis=2)
        elif reduced.shape[2] == self.config.statistics_fft_size:
            reduced = reduced[:, :, : self._n_short_positive_bins]
        elif reduced.shape[2] == self._n_short_positive_bins:
            reduced = reduced.copy()
        if reduced.shape[2] != self._n_short_positive_bins:
            raise ValueError("steering must provide either 1 bin, one-sided bin count, or short_fft_size bins.")
        return reduced.copy()

    def _validate_packet(self, packet: NonuniformBandPacket) -> None:
        """通常 tree packet の帯域整合を確認する。"""
        if packet.spec != self.config.spec:
            raise ValueError("packet.spec does not match the configured leaf spec.")

    def _validate_formal_packet(self, packet: FormalBandPacket) -> None:
        """formal packet の帯域整合を確認する。"""
        if not np.isclose(packet.f_low_hz, self.config.spec.f_low_hz, atol=1e-9):
            raise ValueError("packet.f_low_hz does not match the configured leaf spec.")
        if not np.isclose(packet.f_high_hz, self.config.spec.f_high_hz, atol=1e-9):
            raise ValueError("packet.f_high_hz does not match the configured leaf spec.")
        if not np.isclose(packet.sample_rate_hz, self.config.spec.nominal_sample_rate_hz, atol=1e-9):
            raise ValueError("packet.sample_rate_hz does not match the configured leaf spec.")

    def _update_formal_input_cursor(self, packet: FormalBandPacket) -> None:
        """formal packet の時間原点が leaf 内で連続であることを確認する。"""

        step = self._formal_sample_step_at_root_rate()
        packet_len = int(packet.complex_samples.shape[-1])
        if self._formal_input_template is None:
            self._formal_input_template = packet
            self._formal_next_input_time_origin_at_root_rate = packet.time_origin_at_root_rate + packet_len * step
            self._formal_next_output_time_origin_at_root_rate = packet.time_origin_at_root_rate
            return

        assert self._formal_next_input_time_origin_at_root_rate is not None
        template = self._formal_input_template
        if packet.band_id != template.band_id:
            raise ValueError("formal packet band_id must stay constant within one leaf processor instance.")
        if packet.delay_samples_at_root_rate != template.delay_samples_at_root_rate:
            raise ValueError(
                "formal packet delay_samples_at_root_rate must stay constant within one leaf processor instance."
            )
        if packet.time_origin_at_root_rate != self._formal_next_input_time_origin_at_root_rate:
            raise ValueError("formal packet time_origin_at_root_rate is not contiguous.")
        self._formal_next_input_time_origin_at_root_rate += packet_len * step

    def _formal_sample_step_at_root_rate(self) -> int:
        """この leaf 1 sample が root rate で何 sample 進むかを返す。"""
        return 1 << self.config.spec.tree_depth

    def _formal_output_time_origin(self) -> int:
        """次の formal 出力 packet が持つ root-rate 時間原点を返す。"""
        if self._formal_next_output_time_origin_at_root_rate is None:
            raise RuntimeError("formal output cursor is not initialized.")
        return self._formal_next_output_time_origin_at_root_rate

    def _reset_formal_state(self) -> None:
        """formal tree 接続用の内部カーソルを初期化する。"""
        self._formal_input_template = None
        self._formal_next_input_time_origin_at_root_rate = None
        self._formal_next_output_time_origin_at_root_rate = None

    def _reset_after_flush(self, *, reset_formal_state: bool) -> None:
        """flush 後に stateful 部品を初期化する。"""
        self.output_buffer.reset()
        self._output_delay_pending = None
        if self._cov_estimator is not None:
            self._cov_estimator.reset()
        self._statistics_frames_since_update = 0
        if reset_formal_state:
            self._reset_formal_state()

    def _consume_output_delay(self, valid: np.ndarray | None, *, final: bool) -> np.ndarray:
        """OLS 近似のために入れた固定遅延を streaming 出力側で打ち消す。"""

        delay = self._output_alignment_delay_samples
        prefix_shape = (self.n_beam,)
        if delay == 0:
            if valid is None:
                return np.zeros(prefix_shape + (0,), dtype=np.complex64)
            return np.asarray(valid, dtype=np.complex64).copy()

        if valid is None:
            combined = self._output_delay_pending
            if combined is None or combined.shape[-1] == 0:
                return np.zeros(prefix_shape + (0,), dtype=np.complex64)
            padded = np.pad(combined, [(0, 0)] * (combined.ndim - 1) + [(0, delay)])
            self._output_delay_pending = None
            return np.asarray(padded[..., :-delay], dtype=np.complex64).copy()

        moved = np.asarray(valid, dtype=np.complex64)
        if self._output_delay_pending is None:
            combined = moved
        else:
            combined = np.concatenate([self._output_delay_pending, moved], axis=-1)

        if final:
            padded = np.pad(combined, [(0, 0)] * (combined.ndim - 1) + [(0, delay)])
            self._output_delay_pending = None
            return np.asarray(padded[..., :-delay], dtype=np.complex64).copy()

        if combined.shape[-1] <= delay:
            self._output_delay_pending = combined.copy()
            return np.zeros(combined.shape[:-1] + (0,), dtype=np.complex64)

        self._output_delay_pending = combined[..., -delay:].copy()
        return np.asarray(combined[..., :-delay], dtype=np.complex64).copy()

    def _iter_internal_chunks(self, selected: np.ndarray):
        """共有 hop 単位に入力を分割して内部処理へ流す。"""

        arr = np.asarray(selected, dtype=np.complex64)
        if arr.shape[-1] == 0:
            return
        step = self._output_valid_size
        for start in range(0, arr.shape[-1], step):
            yield arr[..., start : start + step]

    def _select_channels(self, samples: np.ndarray) -> np.ndarray:
        """入力から used_channels だけを切り出す。"""

        arr = np.asarray(samples, dtype=np.complex64)
        if arr.ndim == 1:
            if self.used_channels.size != 1 or self.used_channels[0] != 0:
                raise ValueError("1D samples are only valid for a single used channel at index 0.")
            return arr[np.newaxis, :]
        if arr.ndim != 2:
            raise ValueError("packet.samples must have shape (n_ch, n_sample) or (n_sample,).")
        if np.max(self.used_channels) >= arr.shape[0]:
            raise ValueError("packet.samples does not contain all used_channels.")
        if self._used_channels_identity_only and arr.shape[0] == self.used_channels.size:
            return arr
        if self._used_channels_contiguous:
            return arr[self._used_channels_start : self._used_channels_stop, :]
        return arr[self.used_channels, :]
