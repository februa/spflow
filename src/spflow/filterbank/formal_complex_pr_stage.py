"""spflow.filterbank.formal_complex_pr_stage を実装するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .complex_halfband_stage import ComplexFIRHalfbandStage
from .halfband_stage_candidates import get_known_qmf_candidate


@dataclass(frozen=True)
class FormalBandPacket:
    """formal 非一様木で使う packet 契約を表す。"""

    band_id: str
    f_low_hz: float
    f_high_hz: float
    sample_rate_hz: float
    time_origin_at_root_rate: int
    delay_samples_at_root_rate: int
    complex_samples: np.ndarray

    def __post_init__(self) -> None:
        samples = np.asarray(self.complex_samples, dtype=np.complex64)
        if samples.ndim == 0:
            raise ValueError("complex_samples must have at least one dimension.")
        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive.")
        if self.f_high_hz <= self.f_low_hz:
            raise ValueError("f_high_hz must be greater than f_low_hz.")
        if not np.isclose(self.f_high_hz - self.f_low_hz, 0.5 * self.sample_rate_hz, atol=1e-9):
            raise ValueError("formal packet requires f_high_hz - f_low_hz == sample_rate_hz / 2.")
        object.__setattr__(self, "complex_samples", samples)

    @property
    def bandwidth_hz(self) -> float:
        """packet が表す帯域幅を返す。"""
        return float(self.f_high_hz - self.f_low_hz)

    @property
    def center_frequency_hz(self) -> float:
        """packet の帯域中心周波数を返す。"""
        return 0.5 * (self.f_low_hz + self.f_high_hz)


class FormalComplexPRHalfbandStage:
    """明示 FIR halfband stage を formal packet 契約へ包むラッパー。

    解析・合成そのものに加えて、帯域境界、sample rate、root-rate 遅延・時刻原点を
    packet として運ぶ。高レベルな木接続は責務に含めない。
    """

    def __init__(self, stage: ComplexFIRHalfbandStage, *, root_sample_rate_hz: float) -> None:
        if root_sample_rate_hz <= 0.0:
            raise ValueError("root_sample_rate_hz must be positive.")
        self.stage = stage
        self.root_sample_rate_hz = float(root_sample_rate_hz)
        self.analysis_delay_parent_samples = int(self.stage.filters.analysis_low.size - 1)
        self.synthesis_delay_parent_samples = int(
            max(0, self.stage.filters.synthesis_low.size - 1 - self.stage.filters.delay_compensation)
        )

    @classmethod
    def from_candidate(
        cls,
        candidate_name: str = "daubechies_qmf_order4_taps8",
        *,
        root_sample_rate_hz: float = 32768.0,
    ) -> "FormalComplexPRHalfbandStage":
        """既知 QMF 候補名から stage を構築する。"""
        candidate = get_known_qmf_candidate(candidate_name)
        return cls(candidate.make_stage(), root_sample_rate_hz=root_sample_rate_hz)

    def analyze_packet(self, parent: FormalBandPacket) -> tuple[FormalBandPacket, FormalBandPacket]:
        """親 packet を low/high 子 packet へ解析する。"""
        self._validate_packet(parent)
        low, high = self.stage.analysis(parent.complex_samples)
        f_mid_hz = 0.5 * (parent.f_low_hz + parent.f_high_hz)
        child_rate_hz = 0.5 * parent.sample_rate_hz
        parent_scale = self._root_scale(parent.sample_rate_hz)
        child_time_origin = parent.time_origin_at_root_rate + self.stage.filters.analysis_phase * parent_scale
        added_delay = self.analysis_delay_parent_samples * parent_scale
        child_delay = parent.delay_samples_at_root_rate + added_delay
        # 高域枝は parent 帯域上半分に対応するため、child 基底帯域 [0, Fs/4] へ戻すために
        # exp(-j 2π f_shift t) の周波数シフトを掛ける。
        high = self._frequency_shift_packet(
            high,
            shift_hz=-(f_mid_hz - parent.f_low_hz),
            sample_rate_hz=child_rate_hz,
            time_origin_at_root_rate=child_time_origin,
        )

        low_packet = FormalBandPacket(
            band_id=self._make_band_id(parent.f_low_hz, f_mid_hz),
            f_low_hz=parent.f_low_hz,
            f_high_hz=f_mid_hz,
            sample_rate_hz=child_rate_hz,
            time_origin_at_root_rate=child_time_origin,
            delay_samples_at_root_rate=child_delay,
            complex_samples=low,
        )
        high_packet = FormalBandPacket(
            band_id=self._make_band_id(f_mid_hz, parent.f_high_hz),
            f_low_hz=f_mid_hz,
            f_high_hz=parent.f_high_hz,
            sample_rate_hz=child_rate_hz,
            time_origin_at_root_rate=child_time_origin,
            delay_samples_at_root_rate=child_delay,
            complex_samples=high,
        )
        return low_packet, high_packet

    def synthesize_packets(
        self,
        low_packet: FormalBandPacket,
        high_packet: FormalBandPacket,
        *,
        length: int | None = None,
    ) -> FormalBandPacket:
        """隣接する low/high 兄弟 packet を親 packet へ再合成する。"""
        self._validate_sibling_packets(low_packet, high_packet)
        high_unshifted = self._frequency_shift_packet(
            high_packet.complex_samples,
            shift_hz=high_packet.f_low_hz - low_packet.f_low_hz,
            sample_rate_hz=high_packet.sample_rate_hz,
            time_origin_at_root_rate=high_packet.time_origin_at_root_rate,
        )
        recon = self.stage.synthesis(low_packet.complex_samples, high_unshifted, length=length)
        parent_rate_hz = 2.0 * low_packet.sample_rate_hz
        parent_scale = self._root_scale(parent_rate_hz)
        added_delay = self.synthesis_delay_parent_samples * parent_scale
        return FormalBandPacket(
            band_id=self._make_band_id(low_packet.f_low_hz, high_packet.f_high_hz),
            f_low_hz=low_packet.f_low_hz,
            f_high_hz=high_packet.f_high_hz,
            sample_rate_hz=parent_rate_hz,
            time_origin_at_root_rate=min(
                low_packet.time_origin_at_root_rate,
                high_packet.time_origin_at_root_rate,
            )
            - self.stage.filters.analysis_phase * parent_scale,
            delay_samples_at_root_rate=min(
                low_packet.delay_samples_at_root_rate,
                high_packet.delay_samples_at_root_rate,
            )
            + added_delay,
            complex_samples=recon,
        )

    def _validate_packet(self, packet: FormalBandPacket) -> None:
        if not isinstance(packet, FormalBandPacket):
            raise TypeError("packet must be a FormalBandPacket.")
        self._root_scale(packet.sample_rate_hz)

    def _validate_sibling_packets(self, low_packet: FormalBandPacket, high_packet: FormalBandPacket) -> None:
        self._validate_packet(low_packet)
        self._validate_packet(high_packet)
        if not np.isclose(low_packet.sample_rate_hz, high_packet.sample_rate_hz, atol=1e-9):
            raise ValueError("sibling packets must have identical sample_rate_hz.")
        if not np.isclose(low_packet.f_high_hz, high_packet.f_low_hz, atol=1e-9):
            raise ValueError("sibling packets must be contiguous in frequency.")
        if low_packet.time_origin_at_root_rate != high_packet.time_origin_at_root_rate:
            raise ValueError("sibling packets must have identical time_origin_at_root_rate.")
        if low_packet.complex_samples.shape != high_packet.complex_samples.shape:
            raise ValueError("sibling packets must have identical sample shapes.")

    def _root_scale(self, sample_rate_hz: float) -> int:
        ratio = self.root_sample_rate_hz / float(sample_rate_hz)
        rounded = int(round(ratio))
        if rounded <= 0 or not np.isclose(ratio, rounded, atol=1e-9):
            raise ValueError("sample_rate_hz must divide root_sample_rate_hz by an integer factor.")
        return rounded

    @staticmethod
    def _make_band_id(f_low_hz: float, f_high_hz: float) -> str:
        return f"{f_low_hz:g}-{f_high_hz:g}Hz"

    def _frequency_shift_packet(
        self,
        x: np.ndarray,
        *,
        shift_hz: float,
        sample_rate_hz: float,
        time_origin_at_root_rate: int,
    ) -> np.ndarray:
        arr = np.asarray(x, dtype=np.complex64)
        if arr.shape[-1] == 0 or np.isclose(shift_hz, 0.0, atol=1e-18):
            return arr.copy()

        sample_step_root_rate = self._root_scale(sample_rate_hz)
        # time_index_at_root_rate shape: [n_sample]
        # root-rate 時刻へ直した t に対し exp(-j 2π f_shift t / Fs_root) を掛け、
        # packet の基底帯域原点を周波数シフトする。
        time_index_at_root_rate = time_origin_at_root_rate + sample_step_root_rate * np.arange(arr.shape[-1], dtype=np.float32)
        phase = np.exp(-1j * 2.0 * np.pi * shift_hz * time_index_at_root_rate / self.root_sample_rate_hz)
        return arr * phase
