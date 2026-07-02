"""spflow.filterbank.formal_nonuniform_streaming を実装するモジュール。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .complex_halfband_stage import (
    ComplexFIRHalfbandStageStreamingAnalyzer,
    ComplexFIRHalfbandStageStreamingSynthesizer,
)
from .formal_complex_pr_stage import FormalBandPacket
from .formal_nonuniform_tree import (
    FormalNonuniformAnalysisResult,
    FormalNonuniformTreeFilterBank,
    _FormalTreeNode,
)


@dataclass(frozen=True)
class FormalPacketBlock:
    """One streaming update worth of formal leaf packets."""

    packets: tuple[FormalBandPacket, ...]
    final: bool = False


@dataclass
class _FormalStreamingAnalysisNodeState:
    node: _FormalTreeNode
    filterbank: FormalNonuniformTreeFilterBank
    sample_rate_hz: float
    time_origin_at_root_rate: int
    delay_samples_at_root_rate: int
    analyzer: ComplexFIRHalfbandStageStreamingAnalyzer | None = None
    low_child: "_FormalStreamingAnalysisNodeState | None" = None
    high_child: "_FormalStreamingAnalysisNodeState | None" = None
    emitted_samples: int = 0

    @classmethod
    def build(
        cls,
        node: _FormalTreeNode,
        *,
        filterbank: FormalNonuniformTreeFilterBank,
        sample_rate_hz: float,
        time_origin_at_root_rate: int,
        delay_samples_at_root_rate: int,
    ) -> "_FormalStreamingAnalysisNodeState":
        if node.is_leaf:
            return cls(
                node=node,
                filterbank=filterbank,
                sample_rate_hz=sample_rate_hz,
                time_origin_at_root_rate=time_origin_at_root_rate,
                delay_samples_at_root_rate=delay_samples_at_root_rate,
            )

        scale = filterbank.stage._root_scale(sample_rate_hz)
        child_rate_hz = 0.5 * sample_rate_hz
        child_time_origin = time_origin_at_root_rate + filterbank.stage.stage.filters.analysis_phase * scale
        child_delay = delay_samples_at_root_rate + filterbank.stage.analysis_delay_parent_samples * scale
        assert node.low_child is not None
        assert node.high_child is not None
        return cls(
            node=node,
            filterbank=filterbank,
            sample_rate_hz=sample_rate_hz,
            time_origin_at_root_rate=time_origin_at_root_rate,
            delay_samples_at_root_rate=delay_samples_at_root_rate,
            analyzer=ComplexFIRHalfbandStageStreamingAnalyzer(filterbank.stage.stage),
            low_child=cls.build(
                node.low_child,
                filterbank=filterbank,
                sample_rate_hz=child_rate_hz,
                time_origin_at_root_rate=child_time_origin,
                delay_samples_at_root_rate=child_delay,
            ),
            high_child=cls.build(
                node.high_child,
                filterbank=filterbank,
                sample_rate_hz=child_rate_hz,
                time_origin_at_root_rate=child_time_origin,
                delay_samples_at_root_rate=child_delay,
            ),
        )

    def process(self, x: np.ndarray) -> list[FormalBandPacket]:
        arr = np.asarray(x, dtype=np.complex64)
        if arr.ndim == 0:
            raise ValueError("input chunk must have at least one dimension.")
        if arr.shape[-1] == 0:
            return []

        if self.node.is_leaf:
            packet = FormalBandPacket(
                band_id=self._band_id(),
                f_low_hz=self.node.f_low_hz,
                f_high_hz=self.node.f_high_hz,
                sample_rate_hz=self.sample_rate_hz,
                time_origin_at_root_rate=self._current_time_origin(),
                delay_samples_at_root_rate=self.delay_samples_at_root_rate,
                complex_samples=arr.copy(),
            )
            self.emitted_samples += int(arr.shape[-1])
            return [packet]

        assert self.analyzer is not None
        assert self.low_child is not None
        assert self.high_child is not None
        low, high = self.analyzer.process(arr)
        return self._route_children(low, high)

    def flush(self) -> list[FormalBandPacket]:
        if self.node.is_leaf:
            return []

        assert self.analyzer is not None
        low, high = self.analyzer.flush()
        return self._route_children(low, high, final=True)

    def _route_children(
        self,
        low: np.ndarray,
        high: np.ndarray,
        *,
        final: bool = False,
    ) -> list[FormalBandPacket]:
        assert self.low_child is not None
        assert self.high_child is not None
        outputs: list[FormalBandPacket] = []
        outputs.extend(self.low_child.process(low))

        if high.shape[-1] > 0:
            high = self.filterbank.stage._frequency_shift_packet(
                high,
                shift_hz=-0.25 * self.sample_rate_hz,
                sample_rate_hz=self.high_child.sample_rate_hz,
                time_origin_at_root_rate=self.high_child._current_time_origin(),
            )
        outputs.extend(self.high_child.process(high))

        if final:
            outputs.extend(self.low_child.flush())
            outputs.extend(self.high_child.flush())
        return outputs

    def _band_id(self) -> str:
        if self.node.spec is not None:
            return self.node.spec.band_id
        return f"{self.node.f_low_hz:g}-{self.node.f_high_hz:g}Hz"

    def _current_time_origin(self) -> int:
        scale = self.filterbank.stage._root_scale(self.sample_rate_hz)
        return self.time_origin_at_root_rate + self.emitted_samples * scale


class FormalNonuniformTreeStreamingAnalyzer:
    """Exact-by-construction streaming analyzer for the formal FIR tree."""

    def __init__(self, filterbank: FormalNonuniformTreeFilterBank) -> None:
        self.filterbank = filterbank
        self._root_state = _FormalStreamingAnalysisNodeState.build(
            self.filterbank.root,
            filterbank=self.filterbank,
            sample_rate_hz=self.filterbank.fs_hz,
            time_origin_at_root_rate=0,
            delay_samples_at_root_rate=0,
        )
        self._analytic_input: np.ndarray | None = None
        self._packet_chunks: dict[str, list[FormalBandPacket]] = {
            spec.band_id: [] for spec in self.filterbank.band_specs
        }

    def process_analytic(self, x: np.ndarray) -> list[FormalPacketBlock]:
        arr = np.asarray(x, dtype=np.complex64)
        if arr.ndim == 0:
            raise ValueError("input chunk must have at least one dimension.")
        if arr.shape[-1] == 0:
            return []

        if self._analytic_input is None:
            self._analytic_input = arr.copy()
        else:
            if self._analytic_input.shape[:-1] != arr.shape[:-1]:
                raise ValueError("streaming input shape mismatch except along time axis.")
            self._analytic_input = np.concatenate([self._analytic_input, arr], axis=-1)

        packets = self._root_state.process(arr)
        return self._make_blocks(packets, final=False)

    def flush(self) -> list[FormalPacketBlock]:
        packets = self._root_state.flush()
        return self._make_blocks(packets, final=True)

    def result(self) -> FormalNonuniformAnalysisResult:
        if self._analytic_input is None:
            packets = []
            for spec in self.filterbank.band_specs:
                packets.append(
                    FormalBandPacket(
                        band_id=spec.band_id,
                        f_low_hz=spec.f_low_hz,
                        f_high_hz=spec.f_high_hz,
                        sample_rate_hz=spec.nominal_sample_rate_hz,
                        time_origin_at_root_rate=0,
                        delay_samples_at_root_rate=0,
                        complex_samples=np.zeros((0,), dtype=np.complex64),
                    )
                )
            return FormalNonuniformAnalysisResult(
                packets=tuple(packets),
                node_sample_lengths={self.filterbank._node_band_id(self.filterbank.root): 0},
                original_length=0,
                analytic_length=0,
                padded_length=0,
                analytic_input=True,
            )
        return self.filterbank.analyze_analytic(self._analytic_input)

    def _make_blocks(self, packets: list[FormalBandPacket], *, final: bool) -> list[FormalPacketBlock]:
        if not packets and not final:
            return []

        packets.sort(key=lambda packet: packet.f_low_hz)
        for packet in packets:
            self._packet_chunks[packet.band_id].append(packet)
        return [FormalPacketBlock(tuple(packets), final=final)]


class OracleFormalNonuniformTreeStreamingSynthesizer:
    """Reference synthesizer that rebuilds the full root prefix on every update."""

    def __init__(self, filterbank: FormalNonuniformTreeFilterBank) -> None:
        self.filterbank = filterbank
        self._packet_chunks: dict[str, list[FormalBandPacket]] = {
            spec.band_id: [] for spec in self.filterbank.band_specs
        }
        self._emitted = 0
        self._sample_prefix_shape: tuple[int, ...] | None = None

    def process_block(self, block: FormalPacketBlock) -> np.ndarray:
        if not isinstance(block, FormalPacketBlock):
            raise TypeError("block must be a FormalPacketBlock.")
        for packet in block.packets:
            self._append_packet(packet)

        reconstructed = self._reconstruct_node(self.filterbank.root, final=block.final)
        samples = reconstructed.complex_samples
        new = samples[..., self._emitted :]
        self._emitted = int(samples.shape[-1])
        return new

    def _append_packet(self, packet: FormalBandPacket) -> None:
        if packet.band_id not in self._packet_chunks:
            raise ValueError(f"Unknown band packet {packet.band_id}.")
        if self._sample_prefix_shape is None:
            self._sample_prefix_shape = packet.complex_samples.shape[:-1]
        chunks = self._packet_chunks[packet.band_id]
        if chunks:
            first = chunks[0]
            scale = self.filterbank.stage._root_scale(packet.sample_rate_hz)
            prior_len = sum(int(chunk.complex_samples.shape[-1]) for chunk in chunks)
            expected_origin = first.time_origin_at_root_rate + prior_len * scale
            if packet.time_origin_at_root_rate != expected_origin:
                raise ValueError("packet time_origin_at_root_rate is not contiguous.")
            if packet.delay_samples_at_root_rate != first.delay_samples_at_root_rate:
                raise ValueError("packet delay_samples_at_root_rate must stay constant per band.")
            if not np.isclose(packet.sample_rate_hz, first.sample_rate_hz, atol=1e-9):
                raise ValueError("packet sample_rate_hz must stay constant per band.")
        chunks.append(packet)

    def _reconstruct_node(self, node: _FormalTreeNode, *, final: bool) -> FormalBandPacket:
        if node.is_leaf:
            return self._leaf_packet(node)

        assert node.low_child is not None
        assert node.high_child is not None
        low_packet = self._reconstruct_node(node.low_child, final=final)
        high_packet = self._reconstruct_node(node.high_child, final=final)
        common = min(low_packet.complex_samples.shape[-1], high_packet.complex_samples.shape[-1])
        if common <= 0:
            return self._empty_packet(node)

        target_length = (
            self.filterbank.stage.stage.full_synthesis_length(common)
            if final
            else self.filterbank.stage.stage.stable_synthesis_length(common)
        )
        return self.filterbank.stage.synthesize_packets(
            self._truncate_packet(low_packet, common),
            self._truncate_packet(high_packet, common),
            length=target_length,
        )

    def _leaf_packet(self, node: _FormalTreeNode) -> FormalBandPacket:
        assert node.spec is not None
        chunks = self._packet_chunks[node.spec.band_id]
        if not chunks:
            return self._empty_packet(node)

        first = chunks[0]
        for chunk in chunks[1:]:
            if chunk.delay_samples_at_root_rate != first.delay_samples_at_root_rate:
                raise ValueError("leaf chunks must keep constant delay metadata.")
        samples = np.concatenate([chunk.complex_samples for chunk in chunks], axis=-1)
        return FormalBandPacket(
            band_id=first.band_id,
            f_low_hz=first.f_low_hz,
            f_high_hz=first.f_high_hz,
            sample_rate_hz=first.sample_rate_hz,
            time_origin_at_root_rate=first.time_origin_at_root_rate,
            delay_samples_at_root_rate=first.delay_samples_at_root_rate,
            complex_samples=samples,
        )

    @staticmethod
    def _truncate_packet(packet: FormalBandPacket, length: int) -> FormalBandPacket:
        return FormalBandPacket(
            band_id=packet.band_id,
            f_low_hz=packet.f_low_hz,
            f_high_hz=packet.f_high_hz,
            sample_rate_hz=packet.sample_rate_hz,
            time_origin_at_root_rate=packet.time_origin_at_root_rate,
            delay_samples_at_root_rate=packet.delay_samples_at_root_rate,
            complex_samples=packet.complex_samples[..., :length],
        )

    def _empty_packet(self, node: _FormalTreeNode) -> FormalBandPacket:
        prefix_shape = () if self._sample_prefix_shape is None else self._sample_prefix_shape
        return FormalBandPacket(
            band_id=self.filterbank._node_band_id(node),
            f_low_hz=node.f_low_hz,
            f_high_hz=node.f_high_hz,
            sample_rate_hz=2.0 * (node.f_high_hz - node.f_low_hz),
            time_origin_at_root_rate=0,
            delay_samples_at_root_rate=0,
            complex_samples=np.zeros(prefix_shape + (0,), dtype=np.complex64),
        )


@dataclass
class _FormalStreamingSynthesisNodeState:
    node: _FormalTreeNode
    filterbank: FormalNonuniformTreeFilterBank
    low_child: "_FormalStreamingSynthesisNodeState | None" = None
    high_child: "_FormalStreamingSynthesisNodeState | None" = None
    stage_synthesizer: ComplexFIRHalfbandStageStreamingSynthesizer | None = None

    def __post_init__(self) -> None:
        self.output_queue: deque[FormalBandPacket] = deque()
        self._next_output_time_origin_at_root_rate: int | None = None
        self._output_delay_samples_at_root_rate: int | None = None
        self._stage_flushed = False

    @classmethod
    def build(
        cls,
        node: _FormalTreeNode,
        *,
        filterbank: FormalNonuniformTreeFilterBank,
    ) -> "_FormalStreamingSynthesisNodeState":
        if node.is_leaf:
            return cls(node=node, filterbank=filterbank)

        assert node.low_child is not None
        assert node.high_child is not None
        return cls(
            node=node,
            filterbank=filterbank,
            low_child=cls.build(node.low_child, filterbank=filterbank),
            high_child=cls.build(node.high_child, filterbank=filterbank),
            stage_synthesizer=ComplexFIRHalfbandStageStreamingSynthesizer(filterbank.stage.stage),
        )

    @property
    def sample_rate_hz(self) -> float:
        return 2.0 * (self.node.f_high_hz - self.node.f_low_hz)

    def route_packet(self, packet: FormalBandPacket) -> None:
        if self.node.is_leaf:
            assert self.node.spec is not None
            if packet.band_id != self.node.spec.band_id:
                raise ValueError(f"Unexpected leaf packet {packet.band_id} for {self.node.spec.band_id}.")
            self.output_queue.append(packet)
            return

        assert self.low_child is not None
        assert self.high_child is not None
        if packet.f_high_hz <= self.low_child.node.f_high_hz + 1e-9:
            self.low_child.route_packet(packet)
            return
        if packet.f_low_hz >= self.high_child.node.f_low_hz - 1e-9:
            self.high_child.route_packet(packet)
            return
        raise ValueError("packet does not belong to a unique synthesis subtree.")

    def process_available(self, *, input_final: bool) -> None:
        if self.node.is_leaf:
            return

        assert self.low_child is not None
        assert self.high_child is not None
        assert self.stage_synthesizer is not None

        self.low_child.process_available(input_final=input_final)
        self.high_child.process_available(input_final=input_final)

        while True:
            low_packet = self.low_child.peek_output_packet()
            high_packet = self.high_child.peek_output_packet()
            if low_packet is None or high_packet is None:
                break

            self._validate_child_packets(low_packet, high_packet)
            common = min(
                int(low_packet.complex_samples.shape[-1]),
                int(high_packet.complex_samples.shape[-1]),
            )
            if common <= 0:
                break

            low_chunk = self.low_child.consume_output_prefix(common)
            high_chunk = self.high_child.consume_output_prefix(common)
            self._initialize_output_cursor(low_chunk, high_chunk)

            high_unshifted = self.filterbank.stage._frequency_shift_packet(
                high_chunk.complex_samples,
                shift_hz=high_chunk.f_low_hz - low_chunk.f_low_hz,
                sample_rate_hz=high_chunk.sample_rate_hz,
                time_origin_at_root_rate=high_chunk.time_origin_at_root_rate,
            )
            produced = self.stage_synthesizer.process(low_chunk.complex_samples, high_unshifted)
            self._emit_output_samples(produced)

        if (
            input_final
            and not self._stage_flushed
            and self.low_child.is_exhausted(input_final=input_final)
            and self.high_child.is_exhausted(input_final=input_final)
        ):
            tail = self.stage_synthesizer.flush()
            self._emit_output_samples(tail)
            self._stage_flushed = True

    def is_exhausted(self, *, input_final: bool) -> bool:
        if self.node.is_leaf:
            return input_final and not self.output_queue
        return self._stage_flushed and not self.output_queue

    def peek_output_packet(self) -> FormalBandPacket | None:
        if not self.output_queue:
            return None
        return self.output_queue[0]

    def pop_output_packet(self) -> FormalBandPacket | None:
        if not self.output_queue:
            return None
        return self.output_queue.popleft()

    def consume_output_prefix(self, length: int) -> FormalBandPacket:
        if length <= 0:
            raise ValueError("length must be positive.")
        packet = self.pop_output_packet()
        if packet is None:
            raise RuntimeError("attempted to consume from an empty output queue.")
        packet_len = int(packet.complex_samples.shape[-1])
        if length > packet_len:
            raise ValueError("requested prefix exceeds packet length.")
        if length == packet_len:
            return packet

        prefix, remainder = _split_formal_packet(packet, length, self.filterbank.stage._root_scale(packet.sample_rate_hz))
        assert remainder is not None
        self.output_queue.appendleft(remainder)
        return prefix

    def _validate_child_packets(self, low_packet: FormalBandPacket, high_packet: FormalBandPacket) -> None:
        if not np.isclose(low_packet.sample_rate_hz, high_packet.sample_rate_hz, atol=1e-9):
            raise ValueError("child packets must have identical sample_rate_hz.")
        if not np.isclose(low_packet.f_high_hz, high_packet.f_low_hz, atol=1e-9):
            raise ValueError("child packets must be contiguous in frequency.")
        if low_packet.time_origin_at_root_rate != high_packet.time_origin_at_root_rate:
            raise ValueError("child packets must have identical time_origin_at_root_rate.")
        if low_packet.complex_samples.shape[:-1] != high_packet.complex_samples.shape[:-1]:
            raise ValueError("child packets must have identical prefix sample shape.")

    def _initialize_output_cursor(self, low_packet: FormalBandPacket, high_packet: FormalBandPacket) -> None:
        if self._next_output_time_origin_at_root_rate is not None:
            return
        parent_scale = self.filterbank.stage._root_scale(self.sample_rate_hz)
        self._next_output_time_origin_at_root_rate = (
            low_packet.time_origin_at_root_rate - self.filterbank.stage.stage.filters.analysis_phase * parent_scale
        )
        self._output_delay_samples_at_root_rate = (
            min(
                low_packet.delay_samples_at_root_rate,
                high_packet.delay_samples_at_root_rate,
            )
            + self.filterbank.stage.synthesis_delay_parent_samples * parent_scale
        )

    def _emit_output_samples(self, samples: np.ndarray) -> None:
        arr = np.asarray(samples, dtype=np.complex64)
        if arr.ndim == 0:
            raise ValueError("synthesized samples must have at least one dimension.")
        if arr.shape[-1] == 0:
            return
        if self._next_output_time_origin_at_root_rate is None or self._output_delay_samples_at_root_rate is None:
            raise RuntimeError("output cursor must be initialized before emitting samples.")

        packet = FormalBandPacket(
            band_id=self.filterbank._node_band_id(self.node),
            f_low_hz=self.node.f_low_hz,
            f_high_hz=self.node.f_high_hz,
            sample_rate_hz=self.sample_rate_hz,
            time_origin_at_root_rate=self._next_output_time_origin_at_root_rate,
            delay_samples_at_root_rate=self._output_delay_samples_at_root_rate,
            complex_samples=arr.copy(),
        )
        self.output_queue.append(packet)
        self._next_output_time_origin_at_root_rate += int(arr.shape[-1]) * self.filterbank.stage._root_scale(self.sample_rate_hz)


class FormalNonuniformTreeStreamingSynthesizer:
    """Incremental streaming synthesizer for the formal FIR tree."""

    def __init__(self, filterbank: FormalNonuniformTreeFilterBank) -> None:
        self.filterbank = filterbank
        self._root_state = _FormalStreamingSynthesisNodeState.build(
            self.filterbank.root,
            filterbank=self.filterbank,
        )
        self._sample_prefix_shape: tuple[int, ...] | None = None
        self._input_final = False

    def process_block(self, block: FormalPacketBlock) -> np.ndarray:
        if not isinstance(block, FormalPacketBlock):
            raise TypeError("block must be a FormalPacketBlock.")
        if block.final:
            self._input_final = True

        for packet in block.packets:
            if self._sample_prefix_shape is None:
                self._sample_prefix_shape = packet.complex_samples.shape[:-1]
            self._root_state.route_packet(packet)

        self._root_state.process_available(input_final=self._input_final)
        return self._drain_root_output()

    def _drain_root_output(self) -> np.ndarray:
        pieces = []
        while True:
            packet = self._root_state.pop_output_packet()
            if packet is None:
                break
            pieces.append(packet.complex_samples)
        if not pieces:
            prefix_shape = () if self._sample_prefix_shape is None else self._sample_prefix_shape
            return np.zeros(prefix_shape + (0,), dtype=np.complex64)
        return np.concatenate(pieces, axis=-1)


def _split_formal_packet(
    packet: FormalBandPacket,
    prefix_length: int,
    root_scale: int,
) -> tuple[FormalBandPacket, FormalBandPacket | None]:
    packet_len = int(packet.complex_samples.shape[-1])
    if prefix_length <= 0 or prefix_length > packet_len:
        raise ValueError("prefix_length must be in [1, packet_len].")

    prefix = FormalBandPacket(
        band_id=packet.band_id,
        f_low_hz=packet.f_low_hz,
        f_high_hz=packet.f_high_hz,
        sample_rate_hz=packet.sample_rate_hz,
        time_origin_at_root_rate=packet.time_origin_at_root_rate,
        delay_samples_at_root_rate=packet.delay_samples_at_root_rate,
        complex_samples=packet.complex_samples[..., :prefix_length],
    )
    if prefix_length == packet_len:
        return prefix, None

    remainder = FormalBandPacket(
        band_id=packet.band_id,
        f_low_hz=packet.f_low_hz,
        f_high_hz=packet.f_high_hz,
        sample_rate_hz=packet.sample_rate_hz,
        time_origin_at_root_rate=packet.time_origin_at_root_rate + prefix_length * root_scale,
        delay_samples_at_root_rate=packet.delay_samples_at_root_rate,
        complex_samples=packet.complex_samples[..., prefix_length:],
    )
    return prefix, remainder
