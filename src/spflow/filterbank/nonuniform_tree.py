"""spflow.filterbank.nonuniform_tree を実装するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class NonuniformBandSpec:
    """Leaf-band metadata for the nonuniform complex PR tree."""

    band_id: str
    f_low_hz: float
    f_high_hz: float
    target_resolution_hz: float
    tree_depth: int
    nominal_sample_rate_hz: float

    @property
    def bandwidth_hz(self) -> float:
        return float(self.f_high_hz - self.f_low_hz)

    @property
    def center_frequency_hz(self) -> float:
        return 0.5 * (self.f_low_hz + self.f_high_hz)


@dataclass(frozen=True)
class NonuniformBandPacket:
    """Complex subband samples emitted for one leaf band."""

    spec: NonuniformBandSpec
    samples: np.ndarray


@dataclass(frozen=True)
class NonuniformAnalysisResult:
    """Analysis output plus metadata required for exact synthesis."""

    packets: tuple[NonuniformBandPacket, ...]
    original_length: int
    padded_length: int
    analytic_input: bool


@dataclass(frozen=True)
class _TreeNode:
    f_low_hz: float
    f_high_hz: float
    spec: NonuniformBandSpec | None = None
    low_child: "_TreeNode | None" = None
    high_child: "_TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.spec is not None


class ComplexHalfbandPRBlockStage:
    """Exact 2-channel complex PR stage based on a 2-point DFT on sample pairs.

    This is the minimal exact stage used for the first nonuniform-tree PR validation.
    It is intentionally a baseline splitter, not the final high-selectivity prototype.
    """

    def analysis(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(x, dtype=np.complex64)
        if arr.shape[-1] % 2 != 0:
            raise ValueError("analysis input length must be even.")

        blocks = arr.reshape(arr.shape[:-1] + (-1, 2))
        spectra = np.fft.fft(blocks, axis=-1)
        return spectra[..., 0], spectra[..., 1]

    def synthesis(self, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        low_arr = np.asarray(low, dtype=np.complex64)
        high_arr = np.asarray(high, dtype=np.complex64)
        if low_arr.shape != high_arr.shape:
            raise ValueError("low and high branches must have identical shapes.")

        stacked = np.stack([low_arr, high_arr], axis=-1)
        blocks = np.fft.ifft(stacked, axis=-1)
        return blocks.reshape(blocks.shape[:-2] + (-1,))


class NonuniformTreeFilterBank:
    """Complex PR tree filter bank for the first nonuniform exact-PR validation step."""

    def __init__(self, band_specs: list[NonuniformBandSpec], *, fs_hz: float) -> None:
        if fs_hz <= 0.0:
            raise ValueError("fs_hz must be positive.")
        if not band_specs:
            raise ValueError("band_specs must not be empty.")

        self.fs_hz = float(fs_hz)
        self.band_specs = tuple(sorted(band_specs, key=lambda spec: spec.f_low_hz))
        self.root_band_hz = 0.5 * self.fs_hz
        self.stage = ComplexHalfbandPRBlockStage()
        self.root = self._build_tree(0.0, self.root_band_hz, list(self.band_specs))
        self.max_depth = max(spec.tree_depth for spec in self.band_specs)
        self.root_block_size = 1 << self.max_depth

    @classmethod
    def default_for_fs(cls, fs_hz: float = 32768.0) -> "NonuniformTreeFilterBank":
        if fs_hz <= 0.0:
            raise ValueError("fs_hz must be positive.")
        nyquist = 0.5 * fs_hz
        if not math.isclose(nyquist, 16384.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("default_for_fs currently expects fs_hz == 32768.")

        bands = [
            (0.0, 128.0, 1.0),
            (128.0, 256.0, 1.0),
            (256.0, 512.0, 2.0),
            (512.0, 1024.0, 2.0),
            (1024.0, 2048.0, 4.0),
            (2048.0, 4096.0, 8.0),
            (4096.0, 8192.0, 16.0),
            (8192.0, 16384.0, 32.0),
        ]

        specs = []
        for f_low_hz, f_high_hz, resolution_hz in bands:
            width = f_high_hz - f_low_hz
            depth = int(round(math.log2(nyquist / width)))
            rate = fs_hz / float(1 << depth)
            specs.append(
                NonuniformBandSpec(
                    band_id=f"{int(f_low_hz)}-{int(f_high_hz)}Hz",
                    f_low_hz=f_low_hz,
                    f_high_hz=f_high_hz,
                    target_resolution_hz=resolution_hz,
                    tree_depth=depth,
                    nominal_sample_rate_hz=rate,
                )
            )
        return cls(specs, fs_hz=fs_hz)

    def analyze_real(self, x: np.ndarray) -> NonuniformAnalysisResult:
        arr = np.asarray(x, dtype=np.float32)
        analytic = self._analytic_signal(arr)
        return self._analyze(analytic, analytic_input=False)

    def analyze_analytic(self, x: np.ndarray) -> NonuniformAnalysisResult:
        arr = np.asarray(x, dtype=np.complex64)
        return self._analyze(arr, analytic_input=True)

    def synthesize(self, result: NonuniformAnalysisResult, *, analytic_output: bool = False) -> np.ndarray:
        if not isinstance(result, NonuniformAnalysisResult):
            raise TypeError("result must be a NonuniformAnalysisResult.")

        packet_map = {packet.spec.band_id: np.asarray(packet.samples, dtype=np.complex64) for packet in result.packets}
        reconstructed = self._synthesize_node(self.root, packet_map)
        reconstructed = reconstructed[..., : result.original_length]
        if analytic_output or result.analytic_input:
            return reconstructed
        return np.real(reconstructed)

    def _analyze(self, analytic: np.ndarray, *, analytic_input: bool) -> NonuniformAnalysisResult:
        if analytic.ndim == 0:
            raise ValueError("input must have at least one dimension.")
        padded = self._pad_to_root_block(analytic)
        packets: list[NonuniformBandPacket] = []
        self._analyze_node(self.root, padded, packets)
        packets.sort(key=lambda packet: packet.spec.f_low_hz)
        return NonuniformAnalysisResult(
            packets=tuple(packets),
            original_length=int(analytic.shape[-1]),
            padded_length=int(padded.shape[-1]),
            analytic_input=analytic_input,
        )

    def _analyze_node(
        self,
        node: _TreeNode,
        x: np.ndarray,
        packets: list[NonuniformBandPacket],
    ) -> None:
        if node.is_leaf:
            packets.append(NonuniformBandPacket(node.spec, np.asarray(x, dtype=np.complex64).copy()))
            return

        low, high = self.stage.analysis(x)
        assert node.low_child is not None
        assert node.high_child is not None
        self._analyze_node(node.low_child, low, packets)
        self._analyze_node(node.high_child, high, packets)

    def _synthesize_node(self, node: _TreeNode, packet_map: dict[str, np.ndarray]) -> np.ndarray:
        if node.is_leaf:
            assert node.spec is not None
            try:
                return packet_map[node.spec.band_id]
            except KeyError as exc:
                raise ValueError(f"Missing band packet for {node.spec.band_id}.") from exc

        assert node.low_child is not None
        assert node.high_child is not None
        low = self._synthesize_node(node.low_child, packet_map)
        high = self._synthesize_node(node.high_child, packet_map)
        return self.stage.synthesis(low, high)

    def _pad_to_root_block(self, x: np.ndarray) -> np.ndarray:
        remainder = x.shape[-1] % self.root_block_size
        if remainder == 0:
            return np.asarray(x, dtype=np.complex64)

        pad_width = self.root_block_size - remainder
        pad_spec = [(0, 0)] * x.ndim
        pad_spec[-1] = (0, pad_width)
        return np.pad(np.asarray(x, dtype=np.complex64), pad_spec)

    def _build_tree(
        self,
        f_low_hz: float,
        f_high_hz: float,
        specs: list[NonuniformBandSpec],
    ) -> _TreeNode:
        exact = [
            spec
            for spec in specs
            if math.isclose(spec.f_low_hz, f_low_hz, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(spec.f_high_hz, f_high_hz, rel_tol=0.0, abs_tol=1e-9)
        ]
        if exact:
            if len(exact) != 1 or len(specs) != 1:
                raise ValueError("Band specs must define a unique non-overlapping tree.")
            return _TreeNode(f_low_hz=f_low_hz, f_high_hz=f_high_hz, spec=exact[0])

        mid_hz = 0.5 * (f_low_hz + f_high_hz)
        low_specs = [spec for spec in specs if spec.f_high_hz <= mid_hz + 1e-9]
        high_specs = [spec for spec in specs if spec.f_low_hz >= mid_hz - 1e-9]
        if not low_specs or not high_specs or len(low_specs) + len(high_specs) != len(specs):
            raise ValueError("Band specs do not form a valid dyadic tree.")

        return _TreeNode(
            f_low_hz=f_low_hz,
            f_high_hz=f_high_hz,
            low_child=self._build_tree(f_low_hz, mid_hz, low_specs),
            high_child=self._build_tree(mid_hz, f_high_hz, high_specs),
        )

    @staticmethod
    def _analytic_signal(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        n_sample = arr.shape[-1]
        spectrum = np.fft.fft(arr, axis=-1)
        multiplier = np.zeros(n_sample, dtype=np.float32)
        if n_sample == 0:
            return np.zeros_like(arr, dtype=np.complex64)
        if n_sample % 2 == 0:
            multiplier[0] = 1.0
            multiplier[n_sample // 2] = 1.0
            multiplier[1 : n_sample // 2] = 2.0
        else:
            multiplier[0] = 1.0
            multiplier[1 : (n_sample + 1) // 2] = 2.0
        return np.fft.ifft(spectrum * multiplier, axis=-1)
