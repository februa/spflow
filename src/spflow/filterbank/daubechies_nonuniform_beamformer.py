"""Daubechies/QMF 系の非均一ビームフォーマ全体構成を実装する。"""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np

from .._validation import require, require_non_negative_float, require_positive_float
from ..beamforming.array_design import BandwiseArrayDesign
from .causal_analytic_frontend import CausalAnalyticFrontend
from .formal_complex_pr_stage import FormalBandPacket, FormalComplexPRHalfbandStage
from .formal_nonuniform_tree import FormalNonuniformAnalysisResult, FormalNonuniformTreeFilterBank
from .nonuniform_leaf import BeamformerMode, LeafOutputPathMode, NonuniformLeafProcessor, NonuniformLeafProcessorConfig
from .design.complex_halfband_stage import make_daubechies_qmf_candidate
from .halfband_stage_candidates import get_known_qmf_candidate


def make_reference_dense_sparse_array_design() -> BandwiseArrayDesign:
    """中央密・端疎の基準直線アレイ設計を返す。"""

    positions = np.array(
        [
            -0.395,
            -0.355,
            -0.315,
            -0.275,
            -0.235,
            -0.195,
            -0.155,
            -0.115,
            -0.075,
            -0.065,
            -0.055,
            -0.045,
            -0.035,
            -0.025,
            -0.015,
            -0.005,
            0.005,
            0.015,
            0.025,
            0.035,
            0.045,
            0.055,
            0.065,
            0.075,
            0.115,
            0.155,
            0.195,
            0.235,
            0.275,
            0.315,
            0.355,
            0.395,
        ],
        dtype=np.float32,
    )
    active_channel_counts = [32, 32, 24, 20, 16, 12, 8, 4]
    active_channel_indices_per_band = [
        _centered_subset_indices(positions.size, count)
        for count in active_channel_counts
    ]
    return BandwiseArrayDesign.from_channel_positions_and_active_indices(
        channel_positions_m=positions,
        n_band=len(active_channel_indices_per_band),
        active_indices_per_band=active_channel_indices_per_band,
    )


class DaubechiesNonuniformBeamformer:
    """formal metadata 付き非均一 tree を用いるビームフォーマ。"""

    def __init__(
        self,
        *,
        fs_hz: float = 32768.0,
        candidate_name: str = "daubechies_qmf_order4_taps8",
        array_design: BandwiseArrayDesign | None = None,
        frontend: CausalAnalyticFrontend | None = None,
        beamformer_mode: BeamformerMode = "cbf",
        output_path_mode: LeafOutputPathMode = "leaf_independent_one_sided",
        integration_time: float = 0.0,
        weight_update_period: float = 0.0,
        diag_load: float = 1e-3,
        steering: np.ndarray | Mapping[str, np.ndarray] | None = None,
    ) -> None:
        """解析 tree、leaf 処理設定、重み設計条件をまとめて初期化する。"""

        # 基本パラメータ検証
        self.fs_hz = float(fs_hz)
        require_positive_float("fs_hz", self.fs_hz)
        require_non_negative_float("integration_time", integration_time)
        require_non_negative_float("weight_update_period", weight_update_period)
        require_non_negative_float("diag_load", diag_load)
        require(
            beamformer_mode in ("cbf", "mvdr"),
            "beamformer_mode must be 'cbf' or 'mvdr'.",
        )
        require(
            output_path_mode == "leaf_independent_one_sided",
            "Only 'leaf_independent_one_sided' is supported in the formal nonuniform beamformer.",
        )

        # ビームフォーマ全体設定
        self.array_design = make_reference_dense_sparse_array_design() if array_design is None else array_design
        self.beamformer_mode: BeamformerMode = beamformer_mode
        self.output_path_mode: LeafOutputPathMode = output_path_mode
        self.integration_time = float(integration_time)
        self.weight_update_period = float(weight_update_period)
        self.diag_load = float(diag_load)

        # フィルタバンク本体
        self.filterbank = FormalNonuniformTreeFilterBank.default_for_fs(
            self.fs_hz,
            candidate_name=candidate_name,
            frontend=frontend,
        )
        candidate = _resolve_daubechies_candidate(candidate_name)
        self.filterbank.stage = FormalComplexPRHalfbandStage(candidate.make_stage(), root_sample_rate_hz=self.fs_hz)
        self.stage = self.filterbank.stage
        self.band_specs = self.filterbank.band_specs
        self.root_band_hz = self.filterbank.root_band_hz
        self._fullband_steering = self._prepare_fullband_steering(steering)
        # leaf ごとの処理設定
        require(
            self.array_design.n_band == len(self.band_specs),
            "array_design.n_band must match the number of leaf bands.",
        )
        self.leaf_configs = self._build_leaf_processor_configs()

    def analyze_analytic(self, x: np.ndarray) -> FormalNonuniformAnalysisResult:
        """解析信号を非均一 tree へ分解する。"""

        arr = np.asarray(x, dtype=np.complex64)
        if arr.ndim != 2:
            raise ValueError("x must have shape (n_ch, n_sample).")
        if arr.shape[0] != self.array_design.n_ch:
            raise ValueError("x and array_design must agree on n_ch.")
        return self.filterbank.analyze_analytic(arr)

    def analyze_real(self, x: np.ndarray) -> FormalNonuniformAnalysisResult:
        """実信号入力を causal analytic front-end 込みで分解する。"""

        arr = self._as_multichannel_real(x)
        return self.filterbank.analyze_real(arr)

    def beamform_analysis_result(self, result: FormalNonuniformAnalysisResult) -> FormalNonuniformAnalysisResult:
        """各 leaf packet に対して局所ビームフォーミングを適用する。"""

        if not isinstance(result, FormalNonuniformAnalysisResult):
            raise TypeError("result must be a FormalNonuniformAnalysisResult.")

        packet_map = {packet.band_id: packet for packet in result.packets}
        beamformed_packets = []

        # 処理順を見える形で固定する
        for spec in self.band_specs:
            try:
                packet = packet_map[spec.band_id]
            except KeyError as exc:
                raise ValueError(f"Missing formal leaf packet for {spec.band_id}.") from exc
            processor = NonuniformLeafProcessor(self.leaf_configs[spec.band_id])
            emitted = processor.process_formal_packet(packet)
            emitted.extend(processor.flush_formal())
            beamformed_packets.append(self._merge_formal_leaf_outputs(packet, emitted))

        beamformed_packets.sort(key=lambda packet: packet.f_low_hz)
        return FormalNonuniformAnalysisResult(
            packets=tuple(beamformed_packets),
            node_sample_lengths=result.node_sample_lengths,
            original_length=result.original_length,
            analytic_length=result.analytic_length,
            padded_length=result.padded_length,
            analytic_input=result.analytic_input,
            frontend_delay_samples_at_root_rate=result.frontend_delay_samples_at_root_rate,
            frontend_time_origin_at_root_rate=result.frontend_time_origin_at_root_rate,
        )

    def beamform_packets(self, result: FormalNonuniformAnalysisResult) -> FormalNonuniformAnalysisResult:
        """既存 API 名との互換エイリアス。"""

        return self.beamform_analysis_result(result)

    def beamform_analytic(self, x: np.ndarray) -> np.ndarray:
        """解析信号入力を leaf ビームフォーミング後に再合成する。"""

        analyzed = self.analyze_analytic(x)
        beamformed = self.beamform_analysis_result(analyzed)
        return self.filterbank.synthesize(beamformed, analytic_output=True)

    def beamform_real(self, x: np.ndarray) -> np.ndarray:
        """実信号入力を解析・leaf ビーム形成・再合成まで一括実行する。"""

        analyzed = self.analyze_real(x)
        beamformed = self.beamform_analysis_result(analyzed)
        return self.filterbank.synthesize(beamformed)

    def _merge_formal_leaf_outputs(
        self,
        reference_packet: FormalBandPacket,
        emitted: list[FormalBandPacket],
    ) -> FormalBandPacket:
        """leaf から断続的に出た packet を元の長さへ束ね直す。"""

        if not emitted:
            return FormalBandPacket(
                band_id=reference_packet.band_id,
                f_low_hz=reference_packet.f_low_hz,
                f_high_hz=reference_packet.f_high_hz,
                sample_rate_hz=reference_packet.sample_rate_hz,
                time_origin_at_root_rate=reference_packet.time_origin_at_root_rate,
                delay_samples_at_root_rate=reference_packet.delay_samples_at_root_rate,
                complex_samples=np.zeros((1, 0), dtype=np.complex64),
            )

        samples = np.concatenate([packet.complex_samples for packet in emitted], axis=-1)
        samples = samples[..., : reference_packet.complex_samples.shape[-1]]
        first = emitted[0]
        return FormalBandPacket(
            band_id=reference_packet.band_id,
            f_low_hz=reference_packet.f_low_hz,
            f_high_hz=reference_packet.f_high_hz,
            sample_rate_hz=reference_packet.sample_rate_hz,
            time_origin_at_root_rate=first.time_origin_at_root_rate,
            delay_samples_at_root_rate=first.delay_samples_at_root_rate,
            complex_samples=samples,
        )

    def _build_leaf_processor_configs(self) -> dict[str, NonuniformLeafProcessorConfig]:
        """leaf ごとの FFT 条件と使用チャネル条件を構成する。"""

        configs: dict[str, NonuniformLeafProcessorConfig] = {}
        for band_idx, spec in enumerate(self.band_specs):
            valid_size = int(round(spec.nominal_sample_rate_hz / spec.target_resolution_hz))
            if valid_size <= 0:
                raise ValueError("derived valid_size must be positive.")
            if valid_size & (valid_size - 1):
                raise ValueError("derived valid_size must be a power of two.")
            used_channels = self.array_design.active_channel_indices(band_idx)
            steering = self._leaf_steering_for_band(band_idx)
            hop_size = max(1, valid_size // 2)
            configs[spec.band_id] = NonuniformLeafProcessorConfig(
                spec=spec,
                used_channels=used_channels,
                steering=steering,
                long_fft_frame_size=valid_size,
                long_fft_valid_size=hop_size,
                short_fft_size=valid_size,
                short_fft_hop_size=hop_size,
                beamformer_mode=self.beamformer_mode,
                output_path_mode=self.output_path_mode,
                integration_time=self.integration_time,
                weight_update_period=self.weight_update_period,
                diag_load=self.diag_load,
            )
        return configs

    def _prepare_fullband_steering(
        self,
        steering: np.ndarray | Mapping[str, np.ndarray] | None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """全帯域 steering を共通配列または band_id 辞書へ正規化する。"""

        if steering is None:
            return np.ones((self.array_design.n_ch, 1), dtype=np.complex64)

        if isinstance(steering, Mapping):
            normalized: dict[str, np.ndarray] = {}
            expected_band_ids = {spec.band_id for spec in self.band_specs}
            if set(steering.keys()) != expected_band_ids:
                raise ValueError('steering dict must contain exactly one entry for each band_id.')
            for band_id, steering_value in steering.items():
                steering_array = np.asarray(steering_value, dtype=np.complex64)
                if steering_array.ndim == 1:
                    steering_array = steering_array[:, np.newaxis]
                if steering_array.ndim not in (2, 3):
                    raise ValueError('Each steering dict value must have shape (n_ch, n_beam) or (n_ch, n_beam, n_freq).')
                if steering_array.shape[0] != self.array_design.n_ch:
                    raise ValueError('steering and array_design must agree on n_ch.')
                normalized[band_id] = steering_array.copy()
            return normalized

        steering_array = np.asarray(steering, dtype=np.complex64)
        if steering_array.ndim == 1:
            steering_array = steering_array[:, np.newaxis]
        if steering_array.ndim not in (2, 3):
            raise ValueError('steering must have shape (n_ch, n_beam) or (n_ch, n_beam, n_freq).')
        if steering_array.shape[0] != self.array_design.n_ch:
            raise ValueError('steering and array_design must agree on n_ch.')
        return steering_array.copy()

    def _leaf_steering_for_band(self, band_idx: int) -> np.ndarray:
        """leaf 単位に必要な steering を切り出す。"""

        steering = self._fullband_steering
        if isinstance(steering, dict):
            return steering[self.band_specs[band_idx].band_id].copy()
        if steering.ndim == 2:
            return steering
        return steering[:, :, band_idx]

    def _as_multichannel_real(self, x: np.ndarray) -> np.ndarray:
        """1ch または多チャネル実信号を `(n_ch, n_sample)` に揃える。"""

        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            return np.repeat(arr[np.newaxis, :], self.array_design.n_ch, axis=0)
        if arr.ndim != 2:
            raise ValueError("x must have shape (n_sample,) or (n_ch, n_sample).")
        if arr.shape[0] != self.array_design.n_ch:
            raise ValueError("x and array_design must agree on n_ch.")
        return arr


def _centered_subset_indices(n_ch: int, count: int) -> np.ndarray:
    """中央付近の `count` チャネルを選ぶ添字列を返す。"""

    require(0 < count <= n_ch, "count must be in [1, n_ch].")
    start = (n_ch - count) // 2
    stop = start + count
    return np.arange(start, stop, dtype=np.int64)


def _resolve_daubechies_candidate(candidate_name: str):
    """既知候補名または規則名から QMF 候補を解決する。"""

    try:
        return get_known_qmf_candidate(candidate_name)
    except ValueError:
        match = re.fullmatch(r"daubechies_qmf_order(\d+)_taps(\d+)", candidate_name)
        if match is None:
            raise
        order = int(match.group(1))
        taps = int(match.group(2))
        if taps != 2 * order:
            raise ValueError("daubechies_qmf candidate_name must satisfy taps == 2 * order.")
        return make_daubechies_qmf_candidate(order)
