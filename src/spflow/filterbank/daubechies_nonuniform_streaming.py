"""daubechies_nonuniform_streaming を実装するモジュール。"""

from __future__ import annotations

import numpy as np

from .._validation import require
from .daubechies_nonuniform_beamformer import DaubechiesNonuniformBeamformer
from .formal_nonuniform_streaming import (
    FormalNonuniformTreeStreamingAnalyzer,
    FormalNonuniformTreeStreamingSynthesizer,
    FormalPacketBlock,
)
from .nonuniform_leaf import NonuniformLeafProcessor


class DaubechiesNonuniformBeamformerStreaming:
    """Daubechies 系 nonuniform beamformer の streaming 実装。

    処理フローは次の 4 段で固定する。

    1. root-rate 複素入力を formal nonuniform analysis tree へ流す
    2. 各 leaf packet を対応する `NonuniformLeafProcessor` で beamform する
    3. beamformed leaf packet 群を formal synthesis tree へ戻す
    4. flush 時に末尾の未出力分だけを切り詰めて返す
    """

    def __init__(self, beamformer: DaubechiesNonuniformBeamformer | None = None, **beamformer_kwargs) -> None:
        """streaming 実行に必要な stateful 部品を束ねる。"""
        # 1. 固定パラメータを持つ beamformer 本体を用意する。
        self.beamformer = (
            DaubechiesNonuniformBeamformer(**beamformer_kwargs)
            if beamformer is None
            else beamformer
        )

        # 2. root-rate 側の streaming analyzer / synthesizer を初期化する。
        self._analysis_streamer = FormalNonuniformTreeStreamingAnalyzer(self.beamformer.filterbank)
        self._synthesis_streamer = FormalNonuniformTreeStreamingSynthesizer(self.beamformer.filterbank)

        # 3. 各 leaf は独立 state を持つので、band_id ごとに processor を保持する。
        self._leaf_processors = {
            band_spec.band_id: NonuniformLeafProcessor(self.beamformer.leaf_configs[band_spec.band_id])
            for band_spec in self.beamformer.band_specs
        }
        first_processor = next(iter(self._leaf_processors.values()))
        self._n_beam = first_processor.n_beam

        # 4. flush 時の切り詰めに必要な長さ bookkeeping を持つ。
        self._original_length = 0
        self._emitted_length = 0
        self._is_flushed = False

    def process_analytic(self, x: np.ndarray) -> np.ndarray:
        """複素 analytic 入力を 1 chunk 処理し、出力できる分だけ返す。"""
        require(not self._is_flushed, "process_analytic() cannot be called after flush().")
        normalized = self._normalize_analytic_input(x)
        if normalized.shape[-1] == 0:
            return self._empty_output()

        self._original_length += int(normalized.shape[-1])
        root_output_chunks = []

        # analysis block ごとに leaf beamforming -> root synthesis を行う。
        for packet_block in self._analysis_streamer.process_analytic(normalized):
            root_chunk = self._process_analysis_block_to_root_output(packet_block)
            if root_chunk.shape[-1] > 0:
                root_output_chunks.append(root_chunk)

        if not root_output_chunks:
            return self._empty_output()
        return np.concatenate(root_output_chunks, axis=-1)

    def flush(self) -> np.ndarray:
        """内部 state を flush し、残っている root 出力を返す。"""
        if self._is_flushed:
            return self._empty_output()

        root_output_chunks = []

        # 1. analysis tree 側に残っている block を最後まで回収する。
        for packet_block in self._analysis_streamer.flush():
            root_chunk = self._process_analysis_block_to_root_output(packet_block)
            if root_chunk.shape[-1] > 0:
                root_output_chunks.append(root_chunk)

        # 2. leaf processor ごとの末尾 packet を回収する。
        final_leaf_packets = self._collect_final_leaf_packets()
        final_root_chunk = self._synthesis_streamer.process_block(
            FormalPacketBlock(tuple(final_leaf_packets), final=True)
        )

        # 3. 入力長を超えて出てきた tail は最後に切り詰める。
        remaining_length = max(0, self._original_length - self._emitted_length)
        final_root_chunk = final_root_chunk[..., :remaining_length]
        if final_root_chunk.shape[-1] > 0:
            self._emitted_length += int(final_root_chunk.shape[-1])
            root_output_chunks.append(final_root_chunk)

        self._is_flushed = True
        if not root_output_chunks:
            return self._empty_output()
        return np.concatenate(root_output_chunks, axis=-1)

    def _normalize_analytic_input(self, x: np.ndarray) -> np.ndarray:
        """`(n_ch, n_sample)` の複素入力へ正規化し、shape を検証する。"""
        normalized = np.asarray(x, dtype=np.complex64)
        require(normalized.ndim == 2, "x must have shape (n_ch, n_sample).")
        require(
            normalized.shape[0] == self.beamformer.array_design.n_ch,
            "x and array_design must agree on n_ch.",
        )
        return normalized

    def _process_analysis_block_to_root_output(self, packet_block: FormalPacketBlock) -> np.ndarray:
        """analysis block 1 個を leaf beamforming し、root 出力へ変換する。"""
        beamformed_packets = []
        for formal_packet in packet_block.packets:
            leaf_processor = self._leaf_processors[formal_packet.band_id]
            beamformed_packets.extend(leaf_processor.process_formal_packet(formal_packet))

        beamformed_packets.sort(key=lambda packet: packet.f_low_hz)
        synthesized = self._synthesis_streamer.process_block(
            FormalPacketBlock(tuple(beamformed_packets), final=False)
        )
        self._emitted_length += int(synthesized.shape[-1])
        return synthesized

    def _collect_final_leaf_packets(self) -> list:
        """全 leaf processor の flush 出力 packet を帯域順に集める。"""
        final_leaf_packets = []
        for band_spec in self.beamformer.band_specs:
            final_leaf_packets.extend(self._leaf_processors[band_spec.band_id].flush_formal())
        final_leaf_packets.sort(key=lambda packet: packet.f_low_hz)
        return final_leaf_packets

    def _empty_output(self) -> np.ndarray:
        """出力がまだ存在しない場合の空配列を返す。"""
        return np.zeros((self._n_beam, 0), dtype=np.complex64)
