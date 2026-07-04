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
    """root block 1 個ぶんの leaf-band packet 群を保持する。"""

    packets: tuple[NonuniformBandPacket, ...]


class NonuniformTreeStreamingAnalyzer:
    """厳密 PR な非一様複素木の逐次解析器。

    現段階では内部複素木の streaming 整合性確認が責務であり、
    実数入力 front-end の streaming 化までは責務に含めない。
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
        """複素 analytic 入力チャンクを root block 単位で解析する。"""
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
        """末尾端数を root block 長までゼロ詰めして最終 packet 群を返す。"""
        if self._buffer is None or self._buffer.shape[-1] == 0:
            self._buffer = None
            return []

        remainder = self._buffer.shape[-1]
        pad_width = self.filterbank.root_block_size - remainder
        pad_spec = [(0, 0)] * self._buffer.ndim
        pad_spec[-1] = (0, pad_width)
        # root block 未満の端数では木を 1 段も下れないため、最後はゼロ詰めして完全 block 化する。
        block = np.pad(self._buffer, pad_spec)
        self._buffer = None
        self._padded_length += self.filterbank.root_block_size
        return [self._analyze_full_block(block)]

    def result(self) -> NonuniformAnalysisResult:
        """これまでに蓄積した全 packet をオフライン形式へ再構成する。"""
        packets = []
        for spec in self.filterbank.band_specs:
            chunks = self._packet_chunks[spec.band_id]
            if chunks:
                samples = np.concatenate(chunks, axis=-1)
            else:
                samples = np.zeros((0,), dtype=np.complex64)
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
    """root block ごとの packet 群を逐次再合成するシンプル実装。"""

    def __init__(self, filterbank: NonuniformTreeFilterBank) -> None:
        self.filterbank = filterbank

    def process_block(self, block: NonuniformPacketBlock) -> np.ndarray:
        """1 個の packet block を root-rate 複素時間列へ戻す。"""
        packet_map = {
            packet.spec.band_id: np.asarray(packet.samples, dtype=np.complex64)
            for packet in block.packets
        }
        return self.filterbank._synthesize_node(self.filterbank.root, packet_map)
