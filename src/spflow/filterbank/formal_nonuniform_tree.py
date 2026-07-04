"""spflow.filterbank.formal_nonuniform_tree を実装するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .causal_analytic_frontend import CausalAnalyticFrontend
from .formal_complex_pr_stage import FormalBandPacket, FormalComplexPRHalfbandStage
from .nonuniform_tree import NonuniformBandSpec, NonuniformTreeFilterBank


@dataclass(frozen=True)
class FormalNonuniformAnalysisResult:
    """formal 非一様木の解析結果と再合成メタデータを保持する。"""

    packets: tuple[FormalBandPacket, ...]
    node_sample_lengths: dict[str, int]
    original_length: int
    analytic_length: int
    padded_length: int
    analytic_input: bool
    frontend_delay_samples_at_root_rate: int = 0
    frontend_time_origin_at_root_rate: int = 0


@dataclass(frozen=True)
class _FormalTreeNode:
    f_low_hz: float
    f_high_hz: float
    spec: NonuniformBandSpec | None = None
    low_child: "_FormalTreeNode | None" = None
    high_child: "_FormalTreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.spec is not None


class FormalNonuniformTreeFilterBank:
    """formal packet 契約を使う非一様 FIR 木フィルタバンク。

    packet ごとに周波数帯域・root-rate 時刻原点・遅延を運ぶことで、
    木の各段を明示 FIR 実装へ落としても接続関係を失わないようにする。
    """

    def __init__(
        self,
        band_specs: list[NonuniformBandSpec],
        *,
        fs_hz: float,
        candidate_name: str = "daubechies_qmf_order4_taps8",
        frontend: CausalAnalyticFrontend | None = None,
    ) -> None:
        if fs_hz <= 0.0:
            raise ValueError("fs_hz must be positive.")
        if not band_specs:
            raise ValueError("band_specs must not be empty.")

        self.fs_hz = float(fs_hz)
        self.band_specs = tuple(sorted(band_specs, key=lambda spec: spec.f_low_hz))
        self.root_band_hz = 0.5 * self.fs_hz
        self.root = self._build_tree(0.0, self.root_band_hz, list(self.band_specs))
        self.max_depth = max(spec.tree_depth for spec in self.band_specs)
        self.root_block_size = 1 << self.max_depth
        self.stage = FormalComplexPRHalfbandStage.from_candidate(
            candidate_name,
            root_sample_rate_hz=self.fs_hz,
        )
        self.frontend = CausalAnalyticFrontend.default() if frontend is None else frontend

    @classmethod
    def default_for_fs(
        cls,
        fs_hz: float = 32768.0,
        *,
        candidate_name: str = "daubechies_qmf_order4_taps8",
        frontend: CausalAnalyticFrontend | None = None,
    ) -> "FormalNonuniformTreeFilterBank":
        """既定周波数向けの formal 非一様木を構築する。"""
        reference = NonuniformTreeFilterBank.default_for_fs(fs_hz)
        return cls(
            list(reference.band_specs),
            fs_hz=fs_hz,
            candidate_name=candidate_name,
            frontend=frontend,
        )

    def analyze_analytic(self, x: np.ndarray) -> FormalNonuniformAnalysisResult:
        """複素 analytic 入力を formal packet 木へ解析する。"""
        arr = np.asarray(x, dtype=np.complex64)
        root_packet = FormalBandPacket(
            band_id=self._node_band_id(self.root),
            f_low_hz=0.0,
            f_high_hz=self.root_band_hz,
            sample_rate_hz=self.fs_hz,
            time_origin_at_root_rate=0,
            delay_samples_at_root_rate=0,
            complex_samples=self._pad_to_root_block(arr),
        )
        return self._analyze_root_packet(
            root_packet,
            original_length=int(arr.shape[-1]),
            analytic_length=int(arr.shape[-1]),
            analytic_input=True,
            frontend_delay_samples_at_root_rate=0,
            frontend_time_origin_at_root_rate=0,
        )

    def analyze_real(self, x: np.ndarray) -> FormalNonuniformAnalysisResult:
        """実数入力を causal analytic front-end 後に formal 木へ解析する。"""
        arr = np.asarray(x, dtype=np.float32)
        frontend_result = self.frontend.analyze(arr, pad_tail=True)
        root_packet = FormalBandPacket(
            band_id=self._node_band_id(self.root),
            f_low_hz=0.0,
            f_high_hz=self.root_band_hz,
            sample_rate_hz=self.fs_hz,
            time_origin_at_root_rate=frontend_result.time_origin_at_root_rate,
            delay_samples_at_root_rate=frontend_result.delay_samples_at_root_rate,
            complex_samples=self._pad_to_root_block(frontend_result.samples),
        )
        return self._analyze_root_packet(
            root_packet,
            original_length=int(arr.shape[-1]),
            analytic_length=int(frontend_result.samples.shape[-1]),
            analytic_input=False,
            frontend_delay_samples_at_root_rate=frontend_result.delay_samples_at_root_rate,
            frontend_time_origin_at_root_rate=frontend_result.time_origin_at_root_rate,
        )

    def synthesize(
        self,
        result: FormalNonuniformAnalysisResult,
        *,
        analytic_output: bool = False,
    ) -> np.ndarray:
        """formal 解析結果から root-rate 波形を再合成する。"""
        if not isinstance(result, FormalNonuniformAnalysisResult):
            raise TypeError("result must be a FormalNonuniformAnalysisResult.")

        # 葉 packet を band_id で引けるようにし、木構造に従って bottom-up で再合成する。
        packet_map = {
            packet.band_id: packet
            for packet in result.packets
        }
        reconstructed = self._synthesize_node(self.root, packet_map, result.node_sample_lengths)
        reconstructed = reconstructed.complex_samples[..., : result.analytic_length]
        if analytic_output or result.analytic_input:
            return reconstructed
        return self.frontend.recover_real(reconstructed, length=result.original_length)

    def _analyze_root_packet(
        self,
        root_packet: FormalBandPacket,
        *,
        original_length: int,
        analytic_length: int,
        analytic_input: bool,
        frontend_delay_samples_at_root_rate: int,
        frontend_time_origin_at_root_rate: int,
    ) -> FormalNonuniformAnalysisResult:
        node_sample_lengths: dict[str, int] = {}
        packets: list[FormalBandPacket] = []
        self._analyze_node(self.root, root_packet, packets, node_sample_lengths)
        packets.sort(key=lambda packet: packet.f_low_hz)
        return FormalNonuniformAnalysisResult(
            packets=tuple(packets),
            node_sample_lengths=node_sample_lengths,
            original_length=original_length,
            analytic_length=analytic_length,
            padded_length=int(root_packet.complex_samples.shape[-1]),
            analytic_input=analytic_input,
            frontend_delay_samples_at_root_rate=frontend_delay_samples_at_root_rate,
            frontend_time_origin_at_root_rate=frontend_time_origin_at_root_rate,
        )

    def _analyze_node(
        self,
        node: _FormalTreeNode,
        packet: FormalBandPacket,
        packets: list[FormalBandPacket],
        node_sample_lengths: dict[str, int],
    ) -> None:
        node_sample_lengths[self._node_band_id(node)] = int(packet.complex_samples.shape[-1])
        if node.is_leaf:
            packets.append(packet)
            return

        low_packet, high_packet = self.stage.analyze_packet(packet)
        assert node.low_child is not None
        assert node.high_child is not None
        self._analyze_node(node.low_child, low_packet, packets, node_sample_lengths)
        self._analyze_node(node.high_child, high_packet, packets, node_sample_lengths)

    def _synthesize_node(
        self,
        node: _FormalTreeNode,
        packet_map: dict[str, FormalBandPacket],
        node_sample_lengths: dict[str, int],
    ) -> FormalBandPacket:
        if node.is_leaf:
            try:
                return packet_map[self._node_band_id(node)]
            except KeyError as exc:
                raise ValueError(f"Missing formal band packet for {self._node_band_id(node)}.") from exc

        assert node.low_child is not None
        assert node.high_child is not None
        low_packet = self._synthesize_node(node.low_child, packet_map, node_sample_lengths)
        high_packet = self._synthesize_node(node.high_child, packet_map, node_sample_lengths)
        return self.stage.synthesize_packets(
            low_packet,
            high_packet,
            length=node_sample_lengths[self._node_band_id(node)],
        )

    def _pad_to_root_block(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.complex64)
        if arr.ndim == 0:
            raise ValueError("input must have at least one dimension.")
        remainder = arr.shape[-1] % self.root_block_size
        if remainder == 0:
            return arr
        pad_width = self.root_block_size - remainder
        pad_spec = [(0, 0)] * arr.ndim
        pad_spec[-1] = (0, pad_width)
        return np.pad(arr, pad_spec)

    def _build_tree(
        self,
        f_low_hz: float,
        f_high_hz: float,
        specs: list[NonuniformBandSpec],
    ) -> _FormalTreeNode:
        exact = [
            spec
            for spec in specs
            if math.isclose(spec.f_low_hz, f_low_hz, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(spec.f_high_hz, f_high_hz, rel_tol=0.0, abs_tol=1e-9)
        ]
        if exact:
            if len(exact) != 1 or len(specs) != 1:
                raise ValueError("Band specs must define a unique non-overlapping tree.")
            return _FormalTreeNode(f_low_hz=f_low_hz, f_high_hz=f_high_hz, spec=exact[0])

        mid_hz = 0.5 * (f_low_hz + f_high_hz)
        low_specs = [spec for spec in specs if spec.f_high_hz <= mid_hz + 1e-9]
        high_specs = [spec for spec in specs if spec.f_low_hz >= mid_hz - 1e-9]
        if not low_specs or not high_specs or len(low_specs) + len(high_specs) != len(specs):
            raise ValueError("Band specs do not form a valid dyadic tree.")
        return _FormalTreeNode(
            f_low_hz=f_low_hz,
            f_high_hz=f_high_hz,
            low_child=self._build_tree(f_low_hz, mid_hz, low_specs),
            high_child=self._build_tree(mid_hz, f_high_hz, high_specs),
        )

    @staticmethod
    def _node_band_id(node: _FormalTreeNode) -> str:
        if node.spec is not None:
            return node.spec.band_id
        return f"{node.f_low_hz:g}-{node.f_high_hz:g}Hz"
