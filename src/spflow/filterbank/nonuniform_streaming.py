"""spflow.filterbank.nonuniform_streaming を実装するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nonuniform_tree import (
    NonuniformAnalysisResult,
    NonuniformBandPacket,
    NonuniformTreeFilterBank,
)


@dataclass(frozen=True)
class NonuniformPacketBlock:
    """One root-block worth of leaf-band packets."""

    packets: tuple[NonuniformBandPacket, ...]


class NonuniformTreeStreamingAnalyzer:
    """Streaming analyzer for the exact complex PR tree.

    This first implementation validates streaming on the internal complex tree only.
    Real-input streaming is intentionally deferred because the current analytic front-end
    used by `NonuniformTreeFilterBank.analyze_real()` is an offline FFT-based helper.
    """

    def __init__(self, filterbank: NonuniformTreeFilterBank) -> None:
        self.filterbank = filterbank
        self._buffer: np.ndarray | None = None
        self._original_length = 0
        self._padded_length = 0
        self._packet_chunks: dict[str, list[np.ndarray]] = {
            spec.band_id: [] for spec in self.filterbank.band_specs
        }

    def process_analytic(self, x: np.ndarray) -> list[NonuniformPacketBlock]:
        arr = np.asarray(x, dtype=np.complex64)
        if arr.ndim == 0:
            raise ValueError("input chunk must have at least one dimension.")
        if arr.shape[-1] == 0:
            return []

        self._original_length += int(arr.shape[-1])
        if self._buffer is None:
            self._buffer = arr.copy()
        else:
            self._buffer = np.concatenate([self._buffer, arr], axis=-1)

        outputs: list[NonuniformPacketBlock] = []
        while self._buffer is not None and self._buffer.shape[-1] >= self.filterbank.root_block_size:
            block = self._buffer[..., : self.filterbank.root_block_size]
            tail = self._buffer[..., self.filterbank.root_block_size :]
            self._buffer = tail if tail.shape[-1] > 0 else None
            outputs.append(self._analyze_full_block(block))
            self._padded_length += self.filterbank.root_block_size
        return outputs

    def flush(self) -> list[NonuniformPacketBlock]:
        if self._buffer is None or self._buffer.shape[-1] == 0:
            self._buffer = None
            return []

        remainder = self._buffer.shape[-1]
        pad_width = self.filterbank.root_block_size - remainder
        pad_spec = [(0, 0)] * self._buffer.ndim
        pad_spec[-1] = (0, pad_width)
        block = np.pad(self._buffer, pad_spec)
        self._buffer = None
        self._padded_length += self.filterbank.root_block_size
        return [self._analyze_full_block(block)]

    def result(self) -> NonuniformAnalysisResult:
        packets = []
        for spec in self.filterbank.band_specs:
            chunks = self._packet_chunks[spec.band_id]
            if chunks:
                samples = np.concatenate(chunks, axis=-1)
            else:
                empty_shape = (0,)
                samples = np.zeros(empty_shape, dtype=np.complex64)
            packets.append(NonuniformBandPacket(spec, samples))
        return NonuniformAnalysisResult(
            packets=tuple(packets),
            original_length=self._original_length,
            padded_length=self._padded_length,
            analytic_input=True,
        )

    def _analyze_full_block(self, block: np.ndarray) -> NonuniformPacketBlock:
        packets: list[NonuniformBandPacket] = []
        self.filterbank._analyze_node(self.filterbank.root, block, packets)
        packets.sort(key=lambda packet: packet.spec.f_low_hz)
        for packet in packets:
            self._packet_chunks[packet.spec.band_id].append(np.asarray(packet.samples, dtype=np.complex64))
        return NonuniformPacketBlock(tuple(packets))


class NonuniformTreeStreamingSynthesizer:
    """Streaming synthesizer for root-block packet groups."""

    def __init__(self, filterbank: NonuniformTreeFilterBank) -> None:
        self.filterbank = filterbank

    def process_block(self, block: NonuniformPacketBlock) -> np.ndarray:
        packet_map = {
            packet.spec.band_id: np.asarray(packet.samples, dtype=np.complex64)
            for packet in block.packets
        }
        return self.filterbank._synthesize_node(self.filterbank.root, packet_map)
