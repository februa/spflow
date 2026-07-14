"""周波数帯域ごとのアレイ選択結果を表す値を提供する。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .._validation import (
    require,
    require_index_in_range,
    require_non_negative_int,
    require_positive_float,
    require_positive_int,
)


def _pairwise_distance_matrix(positions_3d: np.ndarray) -> np.ndarray:
    """3 次元センサ座標から距離行列を作る。"""
    diffs = positions_3d[:, np.newaxis, :] - positions_3d[np.newaxis, :, :]
    return np.linalg.norm(diffs, axis=-1)


@dataclass(frozen=True)
class BandwiseArrayDesign:
    """帯域ごとの使用チャネル情報を含むアレイ設計。

    `channel_positions_m` は次の 2 形式を受け付ける。

    - shape `(n_ch,)`: 直線アレイ上の 1 次元座標
    - shape `(n_ch, 3)`: 3 次元座標

    `shading_table` は shape `(n_ch, n_band)` を取り、各帯域で
    どのチャネルを使うか、また必要ならどの shading を掛けるかを表す。
    """

    channel_positions_m: np.ndarray
    shading_table: np.ndarray

    def __post_init__(self) -> None:
        """入力 ndarray を正規化し、幾何情報の整合性を確認する。"""
        positions = self._normalize_channel_positions(self.channel_positions_m)
        shading_table = self._normalize_shading_table(self.shading_table, n_ch=positions.shape[0])

        object.__setattr__(self, "channel_positions_m", positions)
        object.__setattr__(self, "shading_table", shading_table)

    # ------------------------------------------------------------------
    # 生成 helper
    # ------------------------------------------------------------------
    @staticmethod
    def _centered_positions(n_ch: int, spacing_m: float) -> np.ndarray:
        """中心対称な等間隔直線アレイ座標を返す。"""
        return (np.arange(n_ch, dtype=np.float32) - 0.5 * (n_ch - 1)) * spacing_m

    @classmethod
    def _normalize_channel_positions(cls, positions: np.ndarray) -> np.ndarray:
        """センサ座標を `float32` 配列へ正規化し、幾何条件を検証する。"""
        normalized = np.asarray(positions, dtype=np.float32)

        # 1 次元直線アレイの検証
        if normalized.ndim == 1:
            require(normalized.size > 0, "channel_positions_m must not be empty.")
            require(
                not np.any(np.diff(normalized) <= 0.0),
                "1D channel_positions_m must be strictly increasing.",
            )
            return normalized

        # 3 次元座標の検証
        require(
            normalized.ndim == 2 and normalized.shape[1] == 3,
            "channel_positions_m must have shape (n_ch,) or (n_ch, 3).",
        )
        require(normalized.shape[0] > 0, "channel_positions_m must not be empty.")

        distance_matrix = _pairwise_distance_matrix(normalized)
        distance_matrix[np.eye(normalized.shape[0], dtype=bool)] = np.inf
        require(
            not np.any(distance_matrix == 0.0),
            "3D channel_positions_m must not contain duplicated sensors.",
        )
        return normalized

    @staticmethod
    def _normalize_shading_table(shading_table: np.ndarray, *, n_ch: int) -> np.ndarray:
        """shading table を `float32` 配列へ正規化し、shape を検証する。"""
        normalized = np.asarray(shading_table, dtype=np.float32)
        require(normalized.ndim == 2, "shading_table must have shape (n_ch, n_band).")
        require(
            normalized.shape[0] == n_ch,
            "channel_positions_m and shading_table must agree on n_ch.",
        )
        return normalized

    @staticmethod
    def _validate_rectangular_inputs(n_ch: int, spacing_m: float, n_band: int) -> None:
        """帯域ごとの矩形選択器を作るときの共通入力を検証する。"""
        require_positive_int("n_ch", n_ch)
        require_positive_float("spacing_m", spacing_m)
        require_positive_int("n_band", n_band)

    def _validate_band_index(self, band_index: int) -> None:
        """帯域添字の範囲を確認する。"""
        require_index_in_range("band_index", band_index, self.n_band)

    # ------------------------------------------------------------------
    # 生成 API
    # ------------------------------------------------------------------
    @classmethod
    def from_ndarrays(
        cls,
        *,
        channel_positions_m: np.ndarray,
        shading_table: np.ndarray,
    ) -> "BandwiseArrayDesign":
        """外部で設計済みの ndarray から直接インスタンスを作る。"""
        return cls(
            channel_positions_m=np.asarray(channel_positions_m, dtype=np.float32),
            shading_table=np.asarray(shading_table, dtype=np.float32),
        )

    @classmethod
    def from_channel_positions_and_shading_table(
        cls,
        *,
        channel_positions_m: np.ndarray | list[float] | list[list[float]],
        shading_table: np.ndarray,
    ) -> "BandwiseArrayDesign":
        """位置と shading table をそのまま受け取る別名 constructor。"""
        return cls.from_ndarrays(
            channel_positions_m=np.asarray(channel_positions_m, dtype=np.float32),
            shading_table=np.asarray(shading_table, dtype=np.float32),
        )

    @classmethod
    def from_uniform_linear_centered_rectangular(
        cls,
        *,
        n_ch: int,
        spacing_m: float,
        n_band: int,
        active_counts: np.ndarray | list[int],
    ) -> "BandwiseArrayDesign":
        """中心対称直線アレイに対し、帯域ごとの矩形使用チャネルを作る。"""
        cls._validate_rectangular_inputs(n_ch, spacing_m, n_band)

        counts = np.asarray(active_counts, dtype=np.int64)
        require(counts.shape == (n_band,), "active_counts must have shape (n_band).")
        require(
            not np.any(counts <= 0) and not np.any(counts > n_ch),
            "active_counts must be in [1, n_ch].",
        )

        positions = cls._centered_positions(n_ch, spacing_m)
        shading_table = np.zeros((n_ch, n_band), dtype=np.float32)
        for band_index, active_channel_count in enumerate(counts.tolist()):
            start = (n_ch - active_channel_count) // 2
            stop = start + active_channel_count
            shading_table[start:stop, band_index] = 1.0

        return cls(channel_positions_m=positions, shading_table=shading_table)

    @classmethod
    def from_uniform_linear_frequency_progressive_rectangular(
        cls,
        *,
        n_ch: int,
        spacing_m: float,
        fs: float,
        n_band: int,
        sound_speed: float,
        aperture_wavelengths: float = 4.0,
        min_active_ch: int = 4,
        force_odd_counts: bool = False,
    ) -> "BandwiseArrayDesign":
        """周波数が上がるほど開口を狭める矩形選択器を作る。"""
        cls._validate_rectangular_inputs(n_ch, spacing_m, n_band)
        require_positive_float("fs", fs)
        require_positive_float("sound_speed", sound_speed)
        require_positive_float("aperture_wavelengths", aperture_wavelengths)
        require(0 < min_active_ch <= n_ch, "min_active_ch must be in [1, n_ch].")

        frequencies_hz = np.abs(np.fft.fftfreq(n_band, d=1.0 / fs))
        max_aperture_m = (n_ch - 1) * spacing_m
        active_counts = np.full(n_band, n_ch, dtype=np.int64)

        for band_index, frequency_hz in enumerate(frequencies_hz.tolist()):
            if frequency_hz <= 0.0:
                active_counts[band_index] = n_ch
                continue

            desired_aperture_m = min(max_aperture_m, aperture_wavelengths * (sound_speed / frequency_hz))
            active_channel_count = int(np.floor(desired_aperture_m / spacing_m)) + 1
            active_channel_count = max(min_active_ch, min(n_ch, active_channel_count))
            if force_odd_counts and (active_channel_count % 2 == 0) and active_channel_count < n_ch:
                active_channel_count += 1
            active_counts[band_index] = min(n_ch, active_channel_count)

        return cls.from_uniform_linear_centered_rectangular(
            n_ch=n_ch,
            spacing_m=spacing_m,
            n_band=n_band,
            active_counts=active_counts,
        )

    @classmethod
    def from_channel_positions_and_active_indices(
        cls,
        *,
        channel_positions_m: np.ndarray | list[float] | list[list[float]],
        n_band: int,
        active_indices_per_band: Sequence[np.ndarray | Sequence[int]],
    ) -> "BandwiseArrayDesign":
        """帯域ごとの使用チャネル index から shading table を組み立てる。"""
        positions = np.asarray(channel_positions_m, dtype=np.float32)
        require(
            positions.ndim in (1, 2),
            "channel_positions_m must have shape (n_ch,) or (n_ch, 3).",
        )

        n_ch = positions.shape[0]
        require_positive_int("n_band", n_band)
        require(
            len(active_indices_per_band) == n_band,
            "active_indices_per_band must have length n_band.",
        )

        shading_table = np.zeros((n_ch, n_band), dtype=np.float32)
        for band_index, active_indices in enumerate(active_indices_per_band):
            # 運用時のチャネル番号は int32 で保持し、NumPy の添字利用時だけ内部で整数として解釈させる。
            index_array = np.asarray(active_indices, dtype=np.int32)
            require(index_array.ndim == 1, "active channel indices must be 1D.")
            require(index_array.size > 0, "each band must select at least one channel.")
            require(
                not np.any(index_array < 0) and not np.any(index_array >= n_ch),
                "active channel index is out of range.",
            )
            shading_table[index_array, band_index] = 1.0

        return cls(channel_positions_m=positions, shading_table=shading_table)

    @classmethod
    def from_nested_sparse_linear_frequency_progressive(
        cls,
        *,
        n_dense_ch: int,
        dense_spacing_m: float,
        n_outer_pairs: int,
        outer_spacing_m: float,
        fs: float,
        n_band: int,
        sound_speed: float,
        aperture_wavelengths: float = 4.0,
        min_active_ch: int = 4,
    ) -> "BandwiseArrayDesign":
        """中央密・外側疎の直線アレイを帯域ごとの選択付きで作る。"""
        require_positive_int("n_dense_ch", n_dense_ch)
        require_positive_float("dense_spacing_m", dense_spacing_m)
        require_non_negative_int("n_outer_pairs", n_outer_pairs)
        require(
            outer_spacing_m >= dense_spacing_m,
            "outer_spacing_m must be at least dense_spacing_m.",
        )
        require_positive_float("fs", fs)
        require_positive_int("n_band", n_band)
        require_positive_float("sound_speed", sound_speed)
        require_positive_float("aperture_wavelengths", aperture_wavelengths)
        require_positive_int("min_active_ch", min_active_ch)

        inner_positions = cls._centered_positions(n_dense_ch, dense_spacing_m)
        if n_outer_pairs == 0:
            positions = inner_positions
            inner_start = 0
        else:
            inner_edge = 0.5 * (n_dense_ch - 1) * dense_spacing_m
            outer_positions = inner_edge + outer_spacing_m * np.arange(1, n_outer_pairs + 1, dtype=np.float32)
            positions = np.concatenate([-outer_positions[::-1], inner_positions, outer_positions])
            inner_start = n_outer_pairs
        inner_aperture_m = inner_positions[-1] - inner_positions[0] if n_dense_ch > 1 else 0.0

        frequencies_hz = np.abs(np.fft.fftfreq(n_band, d=1.0 / fs))
        max_aperture_m = positions[-1] - positions[0]
        active_indices_per_band: list[np.ndarray] = []

        for frequency_hz in frequencies_hz.tolist():
            if frequency_hz <= 0.0:
                active_indices_per_band.append(np.arange(positions.size, dtype=np.int64))
                continue

            desired_aperture_m = min(max_aperture_m, aperture_wavelengths * (sound_speed / frequency_hz))
            if desired_aperture_m <= inner_aperture_m:
                inner_count = min(
                    n_dense_ch,
                    max(min_active_ch, int(np.floor(desired_aperture_m / dense_spacing_m)) + 1),
                )
                start = inner_start + (n_dense_ch - inner_count) // 2
                stop = start + inner_count
                active_indices_per_band.append(np.arange(start, stop, dtype=np.int64))
                continue

            center_mask = np.abs(positions) <= 0.5 * desired_aperture_m + 1e-12
            active_indices = np.flatnonzero(center_mask)
            if active_indices.size < min_active_ch:
                center = positions.size // 2
                start = max(0, center - (min_active_ch // 2))
                stop = min(positions.size, start + min_active_ch)
                start = max(0, stop - min_active_ch)
                active_indices = np.arange(start, stop, dtype=np.int64)
            active_indices_per_band.append(active_indices.astype(np.int64))

        return cls.from_channel_positions_and_active_indices(
            channel_positions_m=positions,
            n_band=n_band,
            active_indices_per_band=active_indices_per_band,
        )

    # ------------------------------------------------------------------
    # 基本プロパティ
    # ------------------------------------------------------------------
    @property
    def n_ch(self) -> int:
        """総チャネル数。"""
        return int(self.channel_positions_m.shape[0])

    @property
    def n_band(self) -> int:
        """帯域数。"""
        return int(self.shading_table.shape[1])

    @property
    def used_mask(self) -> np.ndarray:
        """各帯域で使用するチャネルを示す真偽表。"""
        return self.shading_table != 0.0

    @property
    def is_explicit_3d(self) -> bool:
        """3 次元座標が明示的に与えられているか。"""
        return self.channel_positions_m.ndim == 2

    # ------------------------------------------------------------------
    # 帯域ごとの問い合わせ API
    # ------------------------------------------------------------------
    def active_channel_indices(self, band_index: int) -> np.ndarray:
        """指定帯域で使用するチャネル index を返す。"""
        self._validate_band_index(band_index)
        return np.flatnonzero(self.used_mask[:, band_index])

    def active_channel_count(self, band_index: int) -> int:
        """指定帯域の使用チャネル数を返す。"""
        return int(self.active_channel_indices(band_index).size)

    def active_channel_counts_per_band(self) -> np.ndarray:
        """全帯域の使用チャネル数をまとめて返す。"""
        return np.count_nonzero(self.used_mask, axis=0).astype(np.int64)

    def shading_for_band(self, band_index: int) -> np.ndarray:
        """指定帯域の shading 係数を返す。"""
        self._validate_band_index(band_index)
        return self.shading_table[:, band_index].copy()

    def active_positions(self, band_index: int) -> np.ndarray:
        """指定帯域で有効なセンサ座標を返す。"""
        return self.channel_positions_m[self.active_channel_indices(band_index)].copy()

    def active_aperture_m(self, band_index: int) -> float:
        """指定帯域で使うセンサ群の開口長を返す。"""
        positions = self.active_positions(band_index)
        if positions.shape[0] <= 1:
            return 0.0
        if positions.ndim == 1:
            return float(positions[-1] - positions[0])

        distance_matrix = _pairwise_distance_matrix(positions)
        return float(np.max(distance_matrix))

    def minimum_spacing_m(self, band_index: int) -> float:
        """指定帯域で使うセンサ群の最小間隔を返す。"""
        positions = self.active_positions(band_index)
        if positions.shape[0] <= 1:
            return float("inf")
        if positions.ndim == 1:
            return float(np.min(np.diff(positions)))

        distance_matrix = _pairwise_distance_matrix(positions)
        distance_matrix[np.eye(positions.shape[0], dtype=bool)] = np.inf
        return float(np.min(distance_matrix))

    def spatial_alias_limit_hz(self, band_index: int, sound_speed: float) -> float:
        """指定帯域における空間 alias 限界周波数を返す。"""
        require_positive_float("sound_speed", sound_speed)
        min_spacing_m = self.minimum_spacing_m(band_index)
        if not np.isfinite(min_spacing_m):
            return float("inf")
        return float(sound_speed / (2.0 * min_spacing_m))

    def positions_3d(self, axis: int = 0) -> np.ndarray:
        """1 次元配置を必要に応じて 3 次元座標へ拡張して返す。"""
        if self.channel_positions_m.ndim == 2:
            return self.channel_positions_m.copy()

        require(axis in (0, 1, 2), "axis must be 0, 1, or 2.")
        positions_3d = np.zeros((self.n_ch, 3), dtype=np.float32)
        positions_3d[:, axis] = self.channel_positions_m
        return positions_3d

    # ------------------------------------------------------------------
    # 互換 API
    # ------------------------------------------------------------------
    def active_indices(self, band_idx: int) -> np.ndarray:
        """`active_channel_indices()` の互換エイリアス。"""
        return self.active_channel_indices(band_idx)

    def active_count(self, band_idx: int) -> int:
        """`active_channel_count()` の互換エイリアス。"""
        return self.active_channel_count(band_idx)

    def active_counts(self) -> np.ndarray:
        """`active_channel_counts_per_band()` の互換エイリアス。"""
        return self.active_channel_counts_per_band()
